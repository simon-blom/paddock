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
use crate::tp::ffn::{SwiGluTpRank, SwiGluWeightNames, TpFfnError};

use super::tp_prefill_backend::Qwen35PrefillBackend;
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

    /// Qwen span adapter: generic FFN owns projection orchestration and scratch;
    /// Qwen retains only tracing/profiling policy around the collective.
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
        if crate::tp::TpTopology::from_group(group)? != topology
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

        self.inner.prefill_local(
            exec,
            &Qwen35PrefillBackend,
            xn,
            rows,
            span,
            q,
        )?;
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
