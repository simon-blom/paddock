//! Qwen3.8 TP FFN adapter.
//!
//! The conventional dense SwiGLU implementation lives in the generic TP
//! subsystem. Qwen supplies weight naming plus its existing span/prefill
//! adapter until shared prefill staging is extracted in the next phase.

use std::ops::{Deref, DerefMut};

use cudarc::driver::CudaSlice;
use paddock_models::mapped::MappedGguf;

use crate::gpu::distributed::Communicator;
use crate::gpu::GpuExecutor;
use crate::gpu_model::tp::ffn::{SwiGluTpRank, SwiGluWeightNames, TpFfnError};

use super::ops::{prefill_ffn_down_any, prefill_mm_pre_any, prefill_quant};
use super::tp_span::{SpanFfn, SpanGemmStaging};

pub type FfnTpError = TpFfnError;

/// Thin Qwen policy wrapper around the generic dense SwiGLU TP component.
pub struct FfnTpRank {
    inner: SwiGluTpRank,
}

impl Deref for FfnTpRank {
    type Target = SwiGluTpRank;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for FfnTpRank {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl FfnTpRank {
    pub fn load<C: Communicator>(
        exec: &GpuExecutor,
        map: &MappedGguf,
        layer: usize,
        group: &C,
    ) -> Result<Self, FfnTpError> {
        let gate = format!("blk.{layer}.ffn_gate.weight");
        let up = format!("blk.{layer}.ffn_up.weight");
        let down = format!("blk.{layer}.ffn_down.weight");
        Ok(Self {
            inner: SwiGluTpRank::load(
                exec,
                map,
                SwiGluWeightNames {
                    gate: &gate,
                    up: &up,
                    down: &down,
                },
                group,
            )?,
        })
    }

    /// Qwen's current span adapter. The projection implementation and sharded
    /// weights are generic; only the Qwen-owned prefill scratch/profile types
    /// remain here until Phase 3 extracts shared prefill staging.
    pub(crate) fn forward_rows_capacity<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
        xn: &CudaSlice<f32>,
        rows: usize,
        span: &'a mut SpanFfn,
        q: &mut SpanGemmStaging,
        layer: usize,
        profile: Option<&mut super::tp_prefill_profile::SpanProfile>,
    ) -> Result<&'a CudaSlice<f32>, FfnTpError> {
        let topology = self.topology();
        if crate::gpu_model::tp::TpTopology::from_group(group)? != topology
            || rows == 0
            || rows > span.cap
            || xn.len() < rows * self.hidden()
            || xn.context().cu_ctx() != exec.stream.context().cu_ctx()
        {
            return Err(FfnTpError::Shape(format!(
                "span capacity/input mismatch: expected world={} rank={} logical_rows=1..{} input_len>=rows*{}; actual world={} rank={} rows={} input_len={} context_match={}",
                topology.world_size(),
                topology.rank(),
                span.cap,
                self.hidden(),
                group.world_size(),
                group.rank(),
                rows,
                xn.len(),
                xn.context().cu_ctx() == exec.stream.context().cu_ctx()
            )));
        }

        prefill_quant(
            exec,
            &mut q.xq,
            &mut q.xs,
            &mut q.yq,
            xn,
            self.hidden(),
            rows,
        )?;
        for (w, out) in [
            (self.gate(), &mut span.gate as &mut CudaSlice<f32>),
            (self.up(), &mut span.up as &mut CudaSlice<f32>),
        ] {
            prefill_mm_pre_any(
                exec,
                w,
                &q.xq,
                &q.xs,
                &q.yq,
                &mut q.xsums,
                &mut q.ssums,
                &mut q.skfix,
                out,
                rows,
            )?;
        }
        super::tp_trace::trace_row(
            exec,
            "b.ffn-gate",
            layer,
            &span.gate,
            0,
            self.local_ff(),
        )?;
        super::tp_trace::trace_row(
            exec,
            "b.ffn-up",
            layer,
            &span.up,
            0,
            self.local_ff(),
        )?;
        prefill_ffn_down_any(
            exec,
            self.down(),
            &mut q.xq,
            &mut q.xs,
            &mut q.yq,
            &mut q.xsums,
            &mut q.ssums,
            &mut q.skfix,
            &mut span.gate,
            &span.up,
            &mut span.partial,
            self.local_ff(),
            rows,
        )?;

        let live = rows * self.hidden();
        if let Some(p) = profile {
            p.reduce(
                &exec.stream,
                group,
                &span.partial.slice(0..live),
                &mut span.reduced.slice_mut(0..live),
                2,
                layer,
            )
            .map_err(|err| FfnTpError::Shape(err.to_string()))?;
        } else {
            group.after_compute(&exec.stream)?;
            group.all_reduce(
                &span.partial.slice(0..live),
                &mut span.reduced.slice_mut(0..live),
            )?;
            group.before_compute(&exec.stream)?;
        }
        Ok(&span.reduced)
    }
}
