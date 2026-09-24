//! Isolated Qwen3.8 dense-FFN TP primitive (decode, one token).
//!
//! Gate/up take output-row shards, SwiGLU remains local, and down takes
//! input-column shards. The two partial outputs meet only in an NCCL sum.
//! Adapted from ErikBPF/paddock `contrib/tp-06-ffn-block` (6142c62),
//! `tensor_ffn.rs`, for rank-local processes: its same-process `Link` and
//! device-to-device reduction are replaced by the Phase 3 `Communicator`.
//! Prenorm and residual stay with the caller; this does not change TP=1 or
//! install a distributed scheduler/model forward path.

use cudarc::driver::CudaSlice;
use paddock_models::mapped::MappedGguf;
use paddock_models::tensor_slice::{ShardKind, TensorSliceRequest};

use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::ops::gemv_any;

#[derive(Debug, thiserror::Error)]
pub enum FfnTpError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] GpuModelError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error("FFN TP: {0}")]
    Shape(String),
}

/// One rank's resident dense FFN. Both ranks receive the same normalized
/// activation; only the final hidden-width projection is replicated by sum.
/// No full tensor is uploaded on this path.
pub struct FfnTpRank {
    gate: QuantW,
    up: QuantW,
    down: QuantW,
    hidden: usize,
    local_ff: usize,
    rank: usize,
    gate_buf: CudaSlice<f32>,
    up_buf: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    reduced: CudaSlice<f32>,
}

impl FfnTpRank {
    pub fn load<C: Communicator>(
        exec: &GpuExecutor,
        map: &MappedGguf,
        layer: usize,
        group: &C,
    ) -> Result<Self, FfnTpError> {
        if group.world_size() != 2 || group.rank() >= 2 {
            return Err(FfnTpError::Shape("expected TP=2 and rank 0 or 1".into()));
        }
        let name = |part: &str| format!("blk.{layer}.ffn_{part}.weight");
        let dims = |part: &str| -> Result<[usize; 2], FfnTpError> {
            let n = name(part);
            let (info, _) = map.tensor_bytes(&n).map_err(GpuError::from)?;
            if info.dims.len() != 2 {
                return Err(FfnTpError::Shape(format!("{n} must be 2-D")));
            }
            Ok([
                usize::try_from(info.dims[0])
                    .map_err(|_| FfnTpError::Shape(format!("{n}: input overflow")))?,
                usize::try_from(info.dims[1])
                    .map_err(|_| FfnTpError::Shape(format!("{n}: output overflow")))?,
            ])
        };
        let [hidden, ff] = dims("gate")?;
        if hidden == 0 || ff == 0 || dims("up")? != [hidden, ff] || dims("down")? != [ff, hidden] {
            return Err(FfnTpError::Shape("gate/up/down dimensions disagree".into()));
        }
        // Phase 4's equal partition and each tensor's own block alignment are
        // checked again by load_quantw_shard before any GPU upload.
        let request = |kind| TensorSliceRequest {
            kind,
            rank: group.rank(),
            world_size: group.world_size(),
        };
        let gate = exec.load_quantw_shard(map, &name("gate"), request(ShardKind::OutputRows))?;
        let up = exec.load_quantw_shard(map, &name("up"), request(ShardKind::OutputRows))?;
        let down = exec.load_quantw_shard(map, &name("down"), request(ShardKind::InputColumns))?;
        let local_ff = ff / 2;
        Ok(Self {
            gate,
            up,
            down,
            hidden,
            local_ff,
            rank: group.rank(),
            gate_buf: exec.alloc(local_ff)?,
            up_buf: exec.alloc(local_ff)?,
            partial: exec.alloc(hidden)?,
            reduced: exec.alloc(hidden)?,
        })
    }

    /// Decode-only: input is already post-attention-normalized on each rank.
    /// The result is `down(silu(gate(x)) * up(x))`, identical on both ranks
    /// after the sum; the caller applies residual exactly once afterward.
    pub fn forward<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
    ) -> Result<&'a CudaSlice<f32>, FfnTpError> {
        if group.world_size() != 2
            || group.rank() != self.rank
            || input.len() != self.hidden
            || input.context().cu_ctx() != exec.stream.context().cu_ctx()
        {
            return Err(FfnTpError::Shape(
                "group or normalized input width changed".into(),
            ));
        }
        self.run(exec, input)?;
        group.after_compute(&exec.stream)?;
        group.all_reduce(&self.partial, &mut self.reduced)?;
        group.before_compute(&exec.stream)?;
        Ok(&self.reduced)
    }

    /// The collective-free compute run: gate/up GEMVs, SwiGLU, down GEMV.
    /// `forward` runs this between its NCCL fences; Phase 11 graph capture
    /// records exactly this run so a replay enqueues the identical kernels
    /// over the identical buffers (only their contents vary per token).
    pub(crate) fn run(
        &mut self,
        exec: &GpuExecutor,
        input: &CudaSlice<f32>,
    ) -> Result<(), FfnTpError> {
        gemv_any(exec, &self.gate, input, &mut self.gate_buf)?;
        gemv_any(exec, &self.up, input, &mut self.up_buf)?;
        exec.swiglu(&mut self.gate_buf, &self.up_buf, self.local_ff)?;
        gemv_any(exec, &self.down, &self.gate_buf, &mut self.partial)?;
        Ok(())
    }

    /// The post-run NCCL fences + all-reduce; returns the reduced output.
    pub(crate) fn finish<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
    ) -> Result<&'a CudaSlice<f32>, FfnTpError> {
        group.after_compute(&exec.stream)?;
        group.all_reduce(&self.partial, &mut self.reduced)?;
        group.before_compute(&exec.stream)?;
        Ok(&self.reduced)
    }
}

#[cfg(test)]
mod tests {
    // Host-only geometry oracle for the split axis: gate/up output channels
    // match down input channels, and only down's partial projections sum.
    #[test]
    fn dense_ffn_channel_split_reconstructs_serial() {
        let hidden = 8;
        let ff = 12;
        let x: Vec<f32> = (0..hidden).map(|i| i as f32 / 7.0 - 0.4).collect();
        let gate: Vec<Vec<f32>> = (0..ff)
            .map(|j| {
                (0..hidden)
                    .map(|i| (i * 3 + j * 5) as f32 / 80.0 - 0.3)
                    .collect()
            })
            .collect();
        let up: Vec<Vec<f32>> = (0..ff)
            .map(|j| {
                (0..hidden)
                    .map(|i| (i + j * 7) as f32 / 100.0 - 0.2)
                    .collect()
            })
            .collect();
        let down: Vec<Vec<f32>> = (0..hidden)
            .map(|i| (0..ff).map(|j| (i * 11 + j) as f32 / 90.0 - 0.5).collect())
            .collect();
        let local = |start: usize, end: usize| -> Vec<f32> {
            let act: Vec<f32> = (start..end)
                .map(|j| {
                    let g: f32 = gate[j].iter().zip(&x).map(|(w, v)| w * v).sum();
                    let u: f32 = up[j].iter().zip(&x).map(|(w, v)| w * v).sum();
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            (0..hidden)
                .map(|i| {
                    down[i][start..end]
                        .iter()
                        .zip(&act)
                        .map(|(w, a)| w * a)
                        .sum()
                })
                .collect()
        };
        let serial = local(0, ff);
        let left = local(0, ff / 2);
        let right = local(ff / 2, ff);
        for i in 0..hidden {
            assert!(((left[i] + right[i]) - serial[i]).abs() < 1e-6);
        }
    }
}
