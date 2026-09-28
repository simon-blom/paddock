//! Qwen3.8 TP serving adapter.
//!
//! The serving/runtime machinery is engine-root generic (`crate::tp::serve`):
//! the coordinator, worker loop, command plumbing, cache sequencing and host
//! tests are shared verbatim with any future conventional model. This module
//! only binds `Qwen35TpRank` to the [`ServeModel`] interface and re-exports
//! the serve entry points under their historical paths.
use std::path::Path;
use std::sync::Arc;

use paddock_dist::config::Resolved;
use paddock_models::mapped::MappedGguf;

use crate::tp::cache::MirroredKv;
use crate::tp::serve::ServeModel;
use crate::gpu::{
    GpuExecutor, KvDtype,
    distributed::NcclCommunicator,
};

use super::tp_model::Qwen35TpRank;

/// `Qwen35TpRank`'s serving identity: the pinned Qwen3.8 checkpoint hash and
/// the model's own resume/plan policy.
impl ServeModel for Qwen35TpRank {
    const CHECKPOINT_SHA256: &'static str =
        "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482";

    type Error = super::tp_model::Qwen35TpError;

    fn load(
        exec: &Arc<GpuExecutor>,
        map: &MappedGguf,
        group: &NcclCommunicator,
        max_ctx: usize,
        slots: usize,
        kv_dtype: KvDtype,
        ckpt_slots: u32,
    ) -> Result<Self, Self::Error> {
        let mut model = Qwen35TpRank::load_slots(
            Arc::clone(exec),
            map,
            group,
            max_ctx,
            kv_dtype,
            slots,
        )?;
        // The lane's weights re-upload from the same mapped file, so the map
        // must outlive load.
        model.enable_prefill_lane(group, map)?;
        // Prefix-cache arm: the coordinator-resolved capacity sizes BOTH the
        // rank-local DeltaNet checkpoint pool and the mirrored radix's
        // free-list. Both ranks arm from the SAME TpInit value.
        model.enable_prefix_checkpoints(ckpt_slots as usize)?;
        Ok(model)
    }

    fn resolve_graph_mode_for_serve() -> bool {
        Qwen35TpRank::resolve_graph_mode_for_serve()
    }

    fn enable_graphs(&mut self, group: &NcclCommunicator) -> Result<(), Self::Error> {
        self.enable_tp_graphs(group)
    }

    fn vocab(&self) -> usize {
        Qwen35TpRank::vocab(self)
    }
    fn max_ctx(&self) -> usize {
        Qwen35TpRank::max_ctx(self)
    }
    fn supports_device_sampling(&self) -> bool {
        Qwen35TpRank::supports_device_sampling(self)
    }
    fn context_mem_bytes(&self) -> u64 {
        Qwen35TpRank::context_mem_bytes(self)
    }
    fn process_mem_used_bytes(&self) -> Option<u64> {
        Qwen35TpRank::process_mem_used_bytes(self)
    }
    fn synchronize(&self) -> Result<(), Self::Error> {
        Qwen35TpRank::synchronize(self)
    }
    fn reset(&mut self) -> Result<(), Self::Error> {
        Qwen35TpRank::reset(self)
    }
    fn reset_slot(&mut self, slot: usize) -> Result<(), Self::Error> {
        Qwen35TpRank::reset_slot(self, slot)
    }
    fn reset_lane_slot(&mut self, slot: usize) -> Result<(), Self::Error> {
        Qwen35TpRank::reset_lane_slot(self, slot)
    }
    fn reset_lane(&mut self) -> Result<(), Self::Error> {
        Qwen35TpRank::reset_lane(self)
    }
    fn has_prefill_lane(&self) -> bool {
        Qwen35TpRank::has_prefill_lane(self)
    }
    fn prefill_lane_done(&self) -> bool {
        Qwen35TpRank::prefill_lane_done(self)
    }
    fn prefill_lane_join(&self) -> Result<(), Self::Error> {
        Qwen35TpRank::prefill_lane_join(self)
    }
    fn prefill_lane_mark(&mut self) -> Result<cudarc::driver::CudaEvent, Self::Error> {
        Qwen35TpRank::prefill_lane_mark(self)
    }
    fn snapshot_slot_ckpt(&mut self, slot: usize, index: u32) -> Result<(), Self::Error> {
        Qwen35TpRank::snapshot_slot_ckpt(self, slot, index)
    }
    fn restore_slot_ckpt(&mut self, slot: usize, index: u32) -> Result<(), Self::Error> {
        Qwen35TpRank::restore_slot_ckpt(self, slot, index)
    }
    fn forward_token_slot(
        &mut self,
        group: &NcclCommunicator,
        logical: &MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
    ) -> Result<Vec<f32>, Self::Error> {
        Qwen35TpRank::forward_token_slot(self, group, logical, token, position, slot)
    }
    fn forward_token_sampled_slot(
        &mut self,
        group: &NcclCommunicator,
        logical: &MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
        plan: crate::sampler::DevicePlan,
    ) -> Result<u32, Self::Error> {
        Qwen35TpRank::forward_token_sampled_slot(self, group, logical, token, position, slot, plan)
    }
    fn forward_token_worker_slot(
        &mut self,
        group: &NcclCommunicator,
        logical: &MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
    ) -> Result<(), Self::Error> {
        Qwen35TpRank::forward_token_worker_slot(self, group, logical, token, position, slot)
    }
    fn sample_logits_slot(
        &mut self,
        plan: crate::sampler::DevicePlan,
    ) -> Result<u32, Self::Error> {
        Qwen35TpRank::sample_logits_slot(self, plan)
    }
    fn forward_span_advance(
        &mut self,
        group: &NcclCommunicator,
        logical: &MirroredKv,
        slot: usize,
        tokens: &[u32],
        position: usize,
    ) -> Result<(), Self::Error> {
        Qwen35TpRank::forward_span_advance(self, group, logical, slot, tokens, position)
    }
    fn forward_span_head_enqueue(&mut self, rows: usize) -> Result<(), Self::Error> {
        Qwen35TpRank::forward_span_head_enqueue(self, rows)
    }
    fn forward_span_head(&mut self, rows: usize) -> Result<Vec<f32>, Self::Error> {
        Qwen35TpRank::forward_span_head(self, rows)
    }
    fn forward_host_to_feedback(
        &mut self,
        group: &NcclCommunicator,
        logical: &MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
        plane: usize,
        plan: crate::sampler::DevicePlan,
    ) -> Result<cudarc::driver::CudaEvent, Self::Error> {
        Qwen35TpRank::forward_host_to_feedback(self, group, logical, token, position, slot, plane, plan)
    }
    fn forward_feedback_to_feedback(
        &mut self,
        group: &NcclCommunicator,
        logical: &MirroredKv,
        slot: usize,
        position: usize,
        source_plane: usize,
        next_plane: usize,
        plan: crate::sampler::DevicePlan,
    ) -> Result<Option<cudarc::driver::CudaEvent>, Self::Error> {
        Qwen35TpRank::forward_feedback_to_feedback(
            self, group, logical, slot, position, source_plane, next_plane, plan,
        )
    }
    fn prefill_lane_span_advance(
        &mut self,
        group: &NcclCommunicator,
        logical: &MirroredKv,
        slot: usize,
        tokens: &[u32],
        position: usize,
    ) -> Result<(), Self::Error> {
        Qwen35TpRank::prefill_lane_span_advance(self, group, logical, slot, tokens, position)
    }
    fn prefill_lane_span_finish(
        &mut self,
        slot: usize,
        rows: usize,
        plan: Option<crate::sampler::DevicePlan>,
    ) -> Result<cudarc::driver::CudaEvent, Self::Error> {
        Qwen35TpRank::prefill_lane_span_finish(self, slot, rows, plan)
    }
    fn prefill_lane_finisher(
        &mut self,
        slot: usize,
        plan: Option<crate::sampler::DevicePlan>,
    ) -> Result<cudarc::driver::CudaEvent, Self::Error> {
        Qwen35TpRank::prefill_lane_finisher(self, slot, plan)
    }
    fn prefill_lane_track_latest(&mut self) -> Result<(), Self::Error> {
        Qwen35TpRank::prefill_lane_track_latest(self)
    }
    fn prefill_lane_sampled_id_after(
        &self,
        ev: &cudarc::driver::CudaEvent,
        slot: usize,
    ) -> Result<u32, Self::Error> {
        Qwen35TpRank::prefill_lane_sampled_id_after(self, ev, slot)
    }
    fn prefill_lane_logits_after(
        &self,
        ev: &cudarc::driver::CudaEvent,
    ) -> Result<Vec<f32>, Self::Error> {
        Qwen35TpRank::prefill_lane_logits_after(self, ev)
    }
    fn promote_lane_slot(
        &mut self,
        lane_done: &cudarc::driver::CudaEvent,
        slot: usize,
        live: &[u32],
    ) -> Result<(), Self::Error> {
        Qwen35TpRank::promote_lane_slot(self, lane_done, slot, live)
    }
    fn feedback_id_after(
        &self,
        event: &cudarc::driver::CudaEvent,
        slot: usize,
        plane: usize,
    ) -> Result<u32, Self::Error> {
        Qwen35TpRank::feedback_id_after(self, event, slot, plane)
    }

    // ---- Qwen-owned serving policy ----
    fn sample_params(
        plan: crate::sampler::DevicePlan,
    ) -> Result<[u32; 4], String> {
        super::tp_model::tp_sample_params(plan).map_err(|e| e.to_string())
    }
    fn resume_decision(ckpt: Option<(usize, u32)>, t_len: usize, slots: usize) -> usize {
        super::tp_kv::tp_resume_decision(ckpt, t_len, slots)
    }
}

/// Rank-0 coordinator for the Qwen TP serve (the generic machinery bound to
/// `Qwen35TpRank`). Historical name kept for the runner/examples' imports.
pub type TpCoordinator = crate::tp::serve::TpCoordinator<Qwen35TpRank>;

/// The runner-facing Qwen TP generator: the generic serve's `TpGenerator`
/// bound to `Qwen35TpRank`.
pub type TpGenerator = crate::tp::serve::TpGenerator<Qwen35TpRank>;

/// Qwen TP worker entry: forwards to the generic worker loop with the model
/// binding applied. Historical path preserved for the runner's startup.
pub fn run_worker(
    stream: std::net::TcpStream,
    resolved: &Resolved,
    model: &Path,
    pack: &Path,
    gpu: usize,
) -> Result<(), String> {
    crate::tp::serve::run_worker::<Qwen35TpRank>(stream, resolved, model, pack, gpu)
}