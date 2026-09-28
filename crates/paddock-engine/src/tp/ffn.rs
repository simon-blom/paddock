//! Generic dense SwiGLU tensor-parallel FFN.
//!
//! Default transformer layout:
//! - gate/up: column parallel (output rows sharded)
//! - SwiGLU: rank-local
//! - down: row parallel (input columns sharded)
//! - output: sum all-reduce
//!
//! Models provide only weight names. A model-specific adapter may replace this
//! component entirely when an architecture has a materially better fast path.

use cudarc::driver::CudaSlice;
use paddock_models::mapped::MappedGguf;

use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::projection::gemv_quant;

use super::prefill::{ProjectionPrefillBackend, ProjectionStaging};
use super::{TpLinearMode, TpTopology, TpTopologyError};

#[derive(Debug, thiserror::Error)]
pub enum TpFfnError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] GpuModelError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error(transparent)]
    Topology(#[from] TpTopologyError),
    #[error("tensor-parallel FFN: {0}")]
    Shape(String),
}

#[derive(Debug, Clone, Copy)]
pub struct SwiGluWeightNames<'a> {
    pub gate: &'a str,
    pub up: &'a str,
    pub down: &'a str,
}

/// Row-batched generic FFN scratch.
pub(crate) struct SwiGluPrefillScratch {
    pub(crate) cap: usize,
    pub(crate) gate: CudaSlice<f32>,
    pub(crate) up: CudaSlice<f32>,
    pub(crate) partial: CudaSlice<f32>,
    pub(crate) reduced: CudaSlice<f32>,
}

impl SwiGluPrefillScratch {
    pub(crate) fn new(
        exec: &GpuExecutor,
        cap: usize,
        local_ff: usize,
        hidden: usize,
    ) -> Result<Self, GpuError> {
        if cap == 0 || local_ff == 0 || hidden == 0 {
            return Err(GpuError::Unsupported(
                "SwiGLU prefill scratch requires nonzero geometry".into(),
            ));
        }
        Ok(Self {
            cap,
            gate: exec.alloc(cap * local_ff)?,
            up: exec.alloc(cap * local_ff)?,
            partial: exec.alloc(cap * hidden)?,
            reduced: exec.alloc(cap * hidden)?,
        })
    }
}

/// Generic rank-local implementation of a conventional dense SwiGLU FFN.
pub struct SwiGluTpRank {
    gate: QuantW,
    up: QuantW,
    down: QuantW,
    hidden: usize,
    local_ff: usize,
    topology: TpTopology,
    gate_buf: CudaSlice<f32>,
    up_buf: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    reduced: CudaSlice<f32>,
}

impl SwiGluTpRank {
    pub fn load<C: Communicator>(
        exec: &GpuExecutor,
        map: &MappedGguf,
        names: SwiGluWeightNames<'_>,
        group: &C,
    ) -> Result<Self, TpFfnError> {
        let topology = TpTopology::from_group(group)?;
        let dims = |name: &str| -> Result<[usize; 2], TpFfnError> {
            let (info, _) = map.tensor_bytes(name).map_err(GpuError::from)?;
            if info.dims.len() != 2 {
                return Err(TpFfnError::Shape(format!("{name} must be 2-D")));
            }
            Ok([
                usize::try_from(info.dims[0])
                    .map_err(|_| TpFfnError::Shape(format!("{name}: input overflow")))?,
                usize::try_from(info.dims[1])
                    .map_err(|_| TpFfnError::Shape(format!("{name}: output overflow")))?,
            ])
        };
        let [hidden, ff] = dims(names.gate)?;
        if hidden == 0
            || ff == 0
            || dims(names.up)? != [hidden, ff]
            || dims(names.down)? != [ff, hidden]
        {
            return Err(TpFfnError::Shape(
                "gate/up/down dimensions disagree".into(),
            ));
        }
        let local_ff = topology.local_len(ff)?;
        let gate = exec.load_quantw_shard(
            map,
            names.gate,
            topology.tensor_slice(TpLinearMode::ColumnParallel.shard_kind()),
        )?;
        let up = exec.load_quantw_shard(
            map,
            names.up,
            topology.tensor_slice(TpLinearMode::ColumnParallel.shard_kind()),
        )?;
        let down = exec.load_quantw_shard(
            map,
            names.down,
            topology.tensor_slice(TpLinearMode::RowParallel { reduce: true }.shard_kind()),
        )?;
        Ok(Self {
            gate,
            up,
            down,
            hidden,
            local_ff,
            topology,
            gate_buf: exec.alloc(local_ff)?,
            up_buf: exec.alloc(local_ff)?,
            partial: exec.alloc(hidden)?,
            reduced: exec.alloc(hidden)?,
        })
    }

    pub fn topology(&self) -> TpTopology {
        self.topology
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn local_ff(&self) -> usize {
        self.local_ff
    }

    pub fn gate(&self) -> &QuantW {
        &self.gate
    }

    pub fn up(&self) -> &QuantW {
        &self.up
    }

    pub fn down(&self) -> &QuantW {
        &self.down
    }

    fn validate_group<C: Communicator>(&self, group: &C) -> Result<(), TpFfnError> {
        if TpTopology::from_group(group)? != self.topology {
            return Err(TpFfnError::Shape(
                "communicator topology changed after FFN load".into(),
            ));
        }
        Ok(())
    }

    pub fn forward<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
    ) -> Result<&'a CudaSlice<f32>, TpFfnError> {
        self.validate_group(group)?;
        if input.len() != self.hidden
            || input.context().cu_ctx() != exec.stream.context().cu_ctx()
        {
            return Err(TpFfnError::Shape(
                "normalized input width/context changed".into(),
            ));
        }
        self.run(exec, input)?;
        self.finish(exec, group)
    }

    /// Collective-free local FFN body. Suitable for graph capture.
    pub fn run(
        &mut self,
        exec: &GpuExecutor,
        input: &CudaSlice<f32>,
    ) -> Result<(), TpFfnError> {
        gemv_quant(exec, &self.gate, input, &mut self.gate_buf)?;
        gemv_quant(exec, &self.up, input, &mut self.up_buf)?;
        exec.swiglu(&mut self.gate_buf, &self.up_buf, self.local_ff)?;
        gemv_quant(exec, &self.down, &self.gate_buf, &mut self.partial)?;
        Ok(())
    }

    /// Collective-free row-batched FFN body.
    ///
    /// The generic FFN owns gate/up/down orchestration; the backend owns only
    /// projection dispatch policy. This is the override seam for a model with
    /// a faster prefill projection implementation.
    pub(crate) fn prefill_local<B: ProjectionPrefillBackend>(
        &mut self,
        exec: &GpuExecutor,
        backend: &B,
        input: &CudaSlice<f32>,
        rows: usize,
        scratch: &mut SwiGluPrefillScratch,
        staging: &mut ProjectionStaging,
    ) -> Result<(), TpFfnError> {
        if rows == 0
            || rows > scratch.cap
            || input.len() < rows * self.hidden
            || input.context().cu_ctx() != exec.stream.context().cu_ctx()
        {
            return Err(TpFfnError::Shape(
                "FFN prefill rows/input exceed configured scratch".into(),
            ));
        }
        staging.validate(self.hidden, rows)?;
        backend.prepare(exec, staging, input, self.hidden, rows)?;
        backend.project_prepared(exec, staging, &self.gate, &mut scratch.gate, rows)?;
        backend.project_prepared(exec, staging, &self.up, &mut scratch.up, rows)?;
        backend.swiglu_down(
            exec,
            staging,
            &self.down,
            &mut scratch.gate,
            &scratch.up,
            &mut scratch.partial,
            self.local_ff,
            rows,
        )?;
        Ok(())
    }

    /// Finish the row-parallel down projection with a sum reduction.
    pub fn finish<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
    ) -> Result<&'a CudaSlice<f32>, TpFfnError> {
        self.validate_group(group)?;
        group.after_compute(&exec.stream)?;
        group.all_reduce(&self.partial, &mut self.reduced)?;
        group.before_compute(&exec.stream)?;
        Ok(&self.reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_ffn_partition_reconstructs_for_multiple_world_sizes() {
        let hidden = 8usize;
        let ff = 12usize;
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
            .map(|i| {
                (0..ff)
                    .map(|j| (i * 11 + j) as f32 / 90.0 - 0.5)
                    .collect()
            })
            .collect();
        let partial = |start: usize, end: usize| -> Vec<f32> {
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
        let serial = partial(0, ff);
        for world_size in [2usize, 3, 4] {
            let local = ff / world_size;
            let mut sum = vec![0.0f32; hidden];
            for rank in 0..world_size {
                let p = partial(rank * local, (rank + 1) * local);
                for (dst, value) in sum.iter_mut().zip(p) {
                    *dst += value;
                }
            }
            for i in 0..hidden {
                assert!((sum[i] - serial[i]).abs() < 1e-6);
            }
        }
    }
}
