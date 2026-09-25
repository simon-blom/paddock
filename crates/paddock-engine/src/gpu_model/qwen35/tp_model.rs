//! Eager whole-backbone TP=2 path shared by parity and serving.
//!
//! The accepted FFN, GQA/paged-KV and DeltaNet rank-local primitives are
//! composed with replicated embeddings, norms and lm_head. KV and recurrent
//! state are rank-local per slot; CUDA graphs remain disabled.
use std::sync::Arc;

use cudarc::driver::{CudaEvent, CudaSlice};
use paddock_models::{gguf::Value, mapped::MappedGguf};

use super::{TokEmbd, embed_any, gemv_any};
use super::{
    delta_tp::{DeltaTpError, DeltaTpRank},
    ffn_tp::{FfnTpError, FfnTpRank},
    gqa_tp::{GqaTpError, GqaTpRank},
    tp_graph::{TpGraphs, TpRunKey},
};
use crate::{
    gpu::distributed::{CollectiveError, Communicator},
    gpu::{DeviceTensor, GpuError, GpuExecutor, KvDtype, QuantW},
    gpu_model::gpt_oss::GpuModelError,
};

#[derive(Debug, thiserror::Error)]
pub enum Qwen35TpError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] GpuModelError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error(transparent)]
    Ffn(#[from] FfnTpError),
    #[error(transparent)]
    Gqa(#[from] GqaTpError),
    #[error(transparent)]
    Delta(#[from] DeltaTpError),
    #[error("Qwen3.8 TP integration: {0}")]
    Shape(String),
}

enum TpMixer {
    Full(Box<GqaTpRank>),
    Linear(Box<DeltaTpRank>),
}
struct TpLayer {
    attn_norm: DeviceTensor,
    post_norm: DeviceTensor,
    mixer: TpMixer,
    ffn: FfnTpRank,
}

/// One process's rank-local model path for deterministic TP=2 parity.
/// Inputs are fed one token at a time; CUDA graphs, speculation, scheduler
/// concurrency, and offload are deliberately not part of this API. The
/// optional feedback primitive samples rank 0's device-resident logits.
pub struct Qwen35TpRank {
    exec: Arc<GpuExecutor>,
    rank: usize,
    hidden: usize,
    vocab: usize,
    max_ctx: usize,
    slots: usize,
    /// The KV dtype both this lane and the prefill lane must serve (Phase 12).
    /// The lane fork re-reads it, so lane and decode slab widths can never
    /// silently diverge the way the hardcoded Fp16 they replaced allowed.
    kv_dtype: KvDtype,
    eps: f32,
    tok_embd: TokEmbd,
    layers: Vec<TpLayer>,
    out_norm: DeviceTensor,
    output: QuantW,
    token: CudaSlice<u32>,
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    sample_params: CudaSlice<u32>,
    sample_id: CudaSlice<u32>,
    // Two one-ID planes per slot. The coordinator owns plane reuse/draining;
    // token and logits scratch are shared and compute-stream ordered.
    feedback_ids: Vec<[CudaSlice<u32>; 2]>,
    /// Cross-lane sampler fence: the last event recorded after a `sample_rows`
    /// call on ANY lane. Every later sampler first waits it (process-global
    /// `pd_sr_*` pack scratch - see PreFillLane). None until the first device
    /// sample.
    sample_chain: Option<CudaEvent>,
    /// Phase 11 Stage A: the decode lane's rank-local graph cache. Empty
    /// until `enable_tp_graphs`; the prefill lane never captures.
    graphs: TpGraphs,
    /// The resolved graph-EXECUTION mode for this rank, stored as ordinary
    /// model state (upstream-readiness I4 fix). Default false = eager.
    /// `enable_tp_graphs` flips it true only after capture succeeds; every
    /// token forward consults this field, never an environment read, so a
    /// serving rank's sequencing is fixed at setup (rank 0 resolves
    /// `PADDOCK_TP_GRAPH` via `dev_var!`, the worker receives the decision
    /// in `TpInit.use_graphs`).
    graphs_enabled: bool,
    /// Second execution lane for overlapped prefill spans: a forked executor
    /// (own stream) plus per-lane backbone scratch re-allocated beside the
    /// decode lane's. Immutable weights (`tok_embd`, `layers`, `out_norm`,
    /// `output`) are shared; GQA KV, DeltaNet recurrent/conv and all working
    /// buffers stay single-lane. `None` until `enable_prefill_lane`.
    prefill: Option<Box<PreFillLane>>,
}

/// Prefill-lane execution state. `model` is a structural re-home of the same
/// weight/geometry onto the forked executor: every `CudaSlice` it holds is
/// reallocated on the prefill stream (weights re-uploaded from the mapped
/// GGUF, scratch re-`alloc`'d), never copied or shared with the decode lane.
/// KV and recurrent payloads are fresh zeroed allocations - a prefill lane
/// slot is only ever written by prefill steps.
pub(super) struct PreFillLane {
    exec: Arc<GpuExecutor>,
    model: Box<Qwen35TpRank>,
    /// Slot currently owned by the prefill lane (decoded there), if any.
    owner: Option<usize>,
    /// Event recorded after the latest finisher enqueue (rank 0); the
    /// non-blocking `prefill_lane_done` probe polls it.
    event: CudaEvent,
    /// Decode-stream event recorded after an in-flight slot promotion's
    /// copies (see `promote_lane_slot`). The next lane enqueue waits it so
    /// relaunched spans cannot race the promotion's reads of this lane's
    /// slabs. `None` when no promotion is outstanding.
    drained: Option<CudaEvent>,
}

impl Qwen35TpRank {
    /// Load the pinned Qwen3.8 dense backbone's rank-local projections and
    /// replicated embedding/norm/head tensors. The caller must verify model
    /// identity and initialize NCCL before calling this on either rank.
    pub fn load<C: Communicator>(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        group: &C,
        max_ctx: usize,
        dtype: KvDtype,
    ) -> Result<Self, Qwen35TpError> {
        Self::load_slots(exec, map, group, max_ctx, dtype, 1)
    }

    pub fn load_slots<C: Communicator>(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        group: &C,
        max_ctx: usize,
        dtype: KvDtype,
        slots: usize,
    ) -> Result<Self, Qwen35TpError> {
        if slots == 0
            || slots > 2
            || max_ctx
                .div_ceil(crate::kv_pool::BLOCK_TOKENS)
                .checked_mul(slots)
                .is_none()
        {
            return Err(Qwen35TpError::Shape("unsupported TP slot geometry".into()));
        }
        if group.world_size() != 2
            || group.rank() >= 2
            || max_ctx == 0
            || max_ctx > u32::MAX as usize
        {
            return Err(Qwen35TpError::Shape(
                "requires TP=2, rank 0/1 and a nonempty u32 context".into(),
            ));
        }
        let u = |key: &str| -> Result<usize, Qwen35TpError> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| Qwen35TpError::Shape(format!("missing or invalid {key}")))
        };
        let n_all = u("block_count")?;
        let n_nextn = map
            .gguf()
            .arch_field("nextn_predict_layers")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let n_layers = n_all
            .checked_sub(n_nextn)
            .ok_or_else(|| Qwen35TpError::Shape("MTP block count exceeds total blocks".into()))?;
        let hidden = u("embedding_length")?;
        let interval = u("full_attention_interval")?;
        let vocab = map
            .tensor_info("token_embd.weight")
            .and_then(|t| usize::try_from(t.dims.get(1).copied()?).ok())
            .ok_or_else(|| Qwen35TpError::Shape("invalid token embedding dimensions".into()))?;
        let eps = map
            .gguf()
            .arch_field("attention.layer_norm_rms_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-6);
        if n_layers == 0
            || hidden != 5120
            || interval == 0
            || vocab == 0
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(Qwen35TpError::Shape(
                "unsupported Qwen3.8 backbone metadata".into(),
            ));
        }
        validate_dense_gemv_coverage(map, n_layers, interval)?;
        let emb_name = "token_embd.weight";
        let emb_ty = map
            .tensor_info(emb_name)
            .ok_or_else(|| Qwen35TpError::Shape("missing token embeddings".into()))?
            .ggml_type;
        let tok_embd = if crate::gpu::kq_params(emb_ty).is_some() {
            TokEmbd::Kq(exec.repack_kquant(map, emb_name)?)
        } else {
            let tensor = exec.upload_raw(map, emb_name)?;
            if tensor.ty != paddock_models::ggml_type::GgmlType::Q8_0 {
                return Err(Qwen35TpError::Shape(format!(
                    "unsupported embedding type {:?}",
                    tensor.ty
                )));
            }
            TokEmbd::Q8(tensor)
        };
        let out_norm = exec.upload(map, "output_norm.weight")?;
        let output = exec.load_quantw(map, "output.weight")?;
        let mut layers = Vec::with_capacity(n_layers);
        let blocks = u32::try_from(
            max_ctx
                .div_ceil(crate::gpu_model::prefix_cache::BLOCK_TOKENS)
                .checked_mul(slots)
                .ok_or_else(|| Qwen35TpError::Shape("paged KV capacity overflow".into()))?,
        )
        .map_err(|_| Qwen35TpError::Shape("paged KV block count overflow".into()))?;
        for i in 0..n_layers {
            let prefix = format!("blk.{i}.");
            let attn_norm = exec.upload(map, &format!("{prefix}attn_norm.weight"))?;
            let post_norm = exec.upload(map, &format!("{prefix}post_attention_norm.weight"))?;
            let mixer = if (i + 1) % interval == 0 {
                TpMixer::Full(Box::new(
                    GqaTpRank::load_paged(&exec, map, i, group, max_ctx, dtype, blocks, slots)
                        .map_err(|e| Qwen35TpError::Shape(e.to_string()))?,
                ))
            } else {
                TpMixer::Linear(Box::new({
                    let mut delta = DeltaTpRank::load(&exec, map, i, group)
                        .map_err(|e| Qwen35TpError::Shape(e.to_string()))?;
                    if slots > 1 {
                        delta.enable_slots(&exec, slots)?;
                    }
                    delta
                }))
            };
            let ffn = FfnTpRank::load(&exec, map, i, group)
                .map_err(|e| Qwen35TpError::Shape(e.to_string()))?;
            layers.push(TpLayer {
                attn_norm,
                post_norm,
                mixer,
                ffn,
            });
        }
        let feedback_ids = (0..slots)
            .map(|_| Ok([exec.alloc_u32(1)?, exec.alloc_u32(1)?]))
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Self {
            token: exec.alloc_u32(1)?,
            x: exec.alloc(hidden)?,
            xn: exec.alloc(hidden)?,
            logits: exec.alloc(vocab)?,
            sample_params: exec.alloc_u32(4)?,
            sample_id: exec.alloc_u32(1)?,
            feedback_ids,
            sample_chain: None,
            prefill: None,
            graphs: TpGraphs::default(),
            graphs_enabled: false,
            exec,
            rank: group.rank(),
            hidden,
            vocab,
            max_ctx,
            slots,
            eps,
            tok_embd,
            layers,
            out_norm,
            output,
            kv_dtype: dtype,
        })
    }

    /// Phase 11 Stage A opt-in (`PADDOCK_TP_GRAPH=1`): capture the rank-local
    /// collective-free per-layer runs of the decode-lane model. Paged GQA
    /// mode only - the non-paged KV append follows a host-side counter and
    /// is not captured (isolated parity paths without the mirrored KV pool
    /// never enable graphs anyway). The DeltaNet layers key their captures
    /// per slot; both slots' first DeltaNet run is captured lazily. The
    /// prefill lane is never captured.
    pub fn enable_tp_graphs<C: Communicator>(&mut self, group: &C) -> Result<(), Qwen35TpError> {
        if self.rank != group.rank() || group.world_size() != 2 {
            return Err(Qwen35TpError::Shape(
                "rank changed under graph enable".into(),
            ));
        }
        for i in 0..self.layers.len() {
            match &mut self.layers[i].mixer {
                TpMixer::Full(gqa) => {
                    if !gqa.is_paged() {
                        return Err(Qwen35TpError::Shape(
                            "TP graphs require paged GQA (the non-paged KV append \
                             advances a host-side counter a graph cannot replay)"
                                .into(),
                        ));
                    }
                }
                TpMixer::Linear(_) => {}
            }
        }
        // Capture pass: DeltaNet slot 0's state pair is live in place, so its
        // runs bake the home buffers; other slots' captures happen lazily on
        // their first decode row (the swap must be live then too).
        for i in 0..self.layers.len() {
            self.capture_mixer_run(group, i, 0)?;
        }
        // Capture succeeded: flip the execution-mode state. Every token
        // forward now replays from the cache (I4: sequencing is model state
        // fixed at setup, not an environment read per process).
        self.graphs_enabled = true;
        Ok(())
    }

    /// How many per-layer graphs are cached (both lanes' probe surface).
    pub fn tp_graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Capture one layer's collective-free mixer run (the pre-attention or
    /// pre-FFN run) at position 0/slot `slot`, following the serial model's
    /// capture discipline: quiesce the stream, begin thread-local capture,
    /// record the run, end and instantiate. The graph bakes the CURRENT
    /// DeltaNet state-pair addresses - DeltaNet callers must have swapped
    /// slot `slot`'s state in. The recorded run EXECUTES nothing, so the
    /// caller still runs the run eagerly (or replays the graph) to produce
    /// its outputs.
    fn capture_mixer_run<C: Communicator>(
        &mut self,
        group: &C,
        layer: usize,
        slot: usize,
    ) -> Result<(), Qwen35TpError> {
        let exec = self.exec.clone();
        // Quiesce the stream so no in-flight work (the previous layer's
        // collectives included) is folded into the capture.
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("tp pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("tp begin_capture: {e}")))?;
        let rec = self.record_mixer_run(group, layer, slot);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("tp end_capture: {e}")));
        // Surface a record failure only after capture is cleanly ended.
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("tp capture produced no graph".into()))?;
        let key = match &self.layers[layer].mixer {
            TpMixer::Full(_) => TpRunKey::Attn(layer),
            TpMixer::Linear(_) => TpRunKey::AttnDelta(layer, slot),
        };
        self.graphs.insert(key, super::SendGraph(graph));
        Ok(())
    }

    /// Record (or replay) one layer's collective-free mixer run: the pre-
    /// attention norm plus the mixer's run up to its `partial`. The graph
    /// path stages nothing here - the caller stages token/position/slot and
    /// re-stages per replay outside capture.
    fn record_mixer_run<C: Communicator>(
        &mut self,
        _group: &C,
        layer: usize,
        slot: usize,
    ) -> Result<(), Qwen35TpError> {
        self.exec.rmsnorm_batch(
            &self.x,
            &self.layers[layer].attn_norm.buf,
            &mut self.xn,
            self.hidden,
            self.eps,
            1,
        )?;
        match &mut self.layers[layer].mixer {
            TpMixer::Full(gqa) => {
                gqa.attention_run(&self.exec, &self.xn)?;
            }
            TpMixer::Linear(delta) => {
                delta.decode_run(&self.exec, &self.xn, 1)?;
                // Stage the out-projection partial now: eager `forward`
                // does it after the run, inside `finish`'s contract; the
                // graph bakes the memset + GEMV too so a replay leaves
                // `partial` ready for the host-side collective.
                delta.finish_partial(&self.exec, 1)?;
            }
        }
        let _ = slot;
        Ok(())
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    /// The KV dtype this rank (and its prefill lane) serves - Phase 12. The
    /// coordinator reports it in its ready handshake and the tier namespace
    /// includes it, so an fp8 serve can never adopt an f16 tier store.
    pub fn kv_dtype(&self) -> KvDtype {
        self.kv_dtype
    }
    /// This rank's context-state bytes, exactly (Phase 12 rank-local memory
    /// accounting): each full-attn layer's rank-local K/V slab pair plus each
    /// DeltaNet layer's rank-local recurrent/conv slot state. Identical on
    /// both ranks by construction (the shard geometry is symmetric), so the
    /// service's memory-breakdown API can report "per rank" honestly without
    /// any cross-rank query.
    pub fn context_mem_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| match &layer.mixer {
                TpMixer::Full(gqa) => gqa.local_kv_bytes() as u64,
                TpMixer::Linear(delta) => delta.local_state_bytes() as u64,
            })
            .sum()
    }

    pub fn supports_device_sampling(&self) -> bool {
        self.rank == 0 && self.exec.has_sample_rows()
    }

    pub fn synchronize(&self) -> Result<(), Qwen35TpError> {
        self.exec.synchronize()?;
        Ok(())
    }
    /// This process's device-pool bytes (weights + context planes + scratch)
    /// as measured at load - the rank-local memory-accounting line that
    /// bounds `weights_mem_bytes` + `context_mem_bytes`. None when the
    /// driver cannot say; the accounting then reports context only.
    pub fn process_mem_used_bytes(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    /// Advance one token through the entire backbone and return all logits.
    /// `position` is rank-0-authorized and must advance identically on both
    /// ranks. `logical_kv` is the corresponding mirrored paged block table.
    pub fn forward_token<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
    ) -> Result<Vec<f32>, Qwen35TpError> {
        self.forward_token_slot(group, logical_kv, token, position, 0)
    }

    pub fn forward_token_slot<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
    ) -> Result<Vec<f32>, Qwen35TpError> {
        self.forward_token_gpu(group, logical_kv, token, position, slot)?;
        Ok(self.exec.to_host(&self.logits)?)
    }

    /// Rank-0-only finish: sample the replicated head in device memory and
    /// transfer one ID, not the full vocabulary. The worker follows the same
    /// forward but never chooses a token or draws scheduler RNG.
    pub fn forward_token_sampled_slot<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
        plan: crate::sampler::DevicePlan,
    ) -> Result<u32, Qwen35TpError> {
        if self.rank != 0 || !self.exec.has_sample_rows() {
            return Err(Qwen35TpError::Shape("TP device sampler unavailable".into()));
        }
        let params = tp_sample_params(plan)?;
        self.exec
            .stream
            .memcpy_htod(&params, &mut self.sample_params)
            .map_err(GpuError::from)?;
        self.forward_token_gpu(group, logical_kv, token, position, slot)?;
        Self::sample_resident(
            &self.exec,
            &mut self.sample_chain,
            self.vocab,
            &self.logits,
            &self.sample_params,
            &mut self.sample_id,
        )?;
        Ok(self
            .exec
            .stream
            .clone_dtoh(&self.sample_id)
            .map_err(GpuError::from)?[0])
    }

    /// Rank 0 enqueues one token forward WITHOUT any logits readback or
    /// device sample: rank 1 must execute the matching `forward_token_worker_slot`
    /// for the collectives to pair. Logits stay in `self.logits`; a separate
    /// sampled pass reads them (see `sample_logits_slot`).
    pub fn forward_token_enqueue<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
    ) -> Result<(), Qwen35TpError> {
        if group.world_size() != 2
            || group.rank() != self.rank
            || position >= self.max_ctx
            || slot >= self.slots
        {
            return Err(Qwen35TpError::Shape("rank or position changed".into()));
        }
        self.exec
            .stream
            .memcpy_htod(&[token], &mut self.token)
            .map_err(GpuError::from)?;
        self.forward_token_body(group, logical_kv, position, slot, self.graphs_enabled)
    }

    /// Sample the resident `self.logits` row on device after a
    /// `forward_token_enqueue`. Rank 1 must run the matching step via
    /// `forward_token_worker_slot` (its state advance pairs the collectives;
    /// it never samples).
    pub fn sample_logits_slot(
        &mut self,
        plan: crate::sampler::DevicePlan,
    ) -> Result<u32, Qwen35TpError> {
        if self.rank != 0 || !self.exec.has_sample_rows() {
            return Err(Qwen35TpError::Shape("TP device sampler unavailable".into()));
        }
        let params = tp_sample_params(plan)?;
        self.exec
            .stream
            .memcpy_htod(&params, &mut self.sample_params)
            .map_err(GpuError::from)?;
        Self::sample_resident(
            &self.exec,
            &mut self.sample_chain,
            self.vocab,
            &self.logits,
            &self.sample_params,
            &mut self.sample_id,
        )?;
        Ok(self
            .exec
            .stream
            .clone_dtoh(&self.sample_id)
            .map_err(GpuError::from)?[0])
    }

    /// Worker-side execution: all collectives and state advance, but no
    /// replicated full-vocabulary device-to-host transfer on rank 1.
    pub fn forward_token_worker<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
    ) -> Result<(), Qwen35TpError> {
        self.forward_token_worker_slot(group, logical_kv, token, position, 0)
    }

    pub fn forward_token_worker_slot<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
    ) -> Result<(), Qwen35TpError> {
        if self.rank != 1 {
            return Err(Qwen35TpError::Shape(
                "worker forward requires rank 1".into(),
            ));
        }
        self.forward_token_gpu(group, logical_kv, token, position, slot)?;
        self.exec.stream.synchronize().map_err(GpuError::from)?;
        Ok(())
    }

    /// Rank 0 enqueues a host-token forward and samples into a chosen plane.
    /// Rank 1 executes the matching step via `forward_token_worker_slot`.
    /// The completion event does not synchronize the host.
    pub fn forward_host_to_feedback<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
        plane: usize,
        plan: crate::sampler::DevicePlan,
    ) -> Result<CudaEvent, Qwen35TpError> {
        self.check_feedback(slot, plane, true)?;
        let params = tp_sample_params(plan)?;
        self.exec
            .stream
            .memcpy_htod(&params, &mut self.sample_params)
            .map_err(GpuError::from)?;
        self.forward_token_gpu(group, logical_kv, token, position, slot)?;
        self.sample_feedback(slot, plane)
    }

    /// Both ranks call in identical order after rank 0's source plane was
    /// sampled. NCCL sends that one ID to rank 1; the embedding input is a
    /// stream-ordered device copy into persistent `token` scratch, not an
    /// upload/readback. Only rank 0 samples the alternate plane.
    /// The coordinator must drain/read a plane before reusing it, and must
    /// authorize matching slot/position/plane parameters on both ranks.
    pub fn forward_feedback_to_feedback<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        slot: usize,
        position: usize,
        source_plane: usize,
        next_plane: usize,
        plan: crate::sampler::DevicePlan,
    ) -> Result<Option<CudaEvent>, Qwen35TpError> {
        self.check_feedback(slot, source_plane, false)?;
        check_feedback_plane(self.slots, slot, next_plane)?;
        if source_plane == next_plane {
            return Err(Qwen35TpError::Shape(
                "feedback planes must alternate".into(),
            ));
        }
        if group.rank() != self.rank || group.world_size() != 2 || position >= self.max_ctx {
            return Err(Qwen35TpError::Shape("rank or position changed".into()));
        }
        if self.rank == 0 {
            let params = tp_sample_params(plan)?;
            self.exec
                .stream
                .memcpy_htod(&params, &mut self.sample_params)
                .map_err(GpuError::from)?;
        }
        group.after_compute(&self.exec.stream)?;
        group.broadcast(&mut self.feedback_ids[slot][source_plane], 0)?;
        group.before_compute(&self.exec.stream)?;
        self.forward_device_feedback(group, logical_kv, position, slot, source_plane)?;
        if self.rank == 0 {
            Ok(Some(self.sample_feedback(slot, next_plane)?))
        } else {
            Ok(None)
        }
    }

    /// Poll without blocking; then read via the copy stream once complete.
    /// Never overwrite the selected plane before the read/drain finishes.
    pub fn feedback_done(&self, event: &CudaEvent) -> bool {
        self.exec.event_done(event)
    }

    pub fn feedback_id_after(
        &self,
        event: &CudaEvent,
        slot: usize,
        plane: usize,
    ) -> Result<u32, Qwen35TpError> {
        self.check_feedback(slot, plane, true)?;
        Ok(self
            .exec
            .to_host_u32_after(event, &self.feedback_ids[slot][plane], 0, 1)?[0])
    }

    fn check_feedback(
        &self,
        slot: usize,
        plane: usize,
        sampling: bool,
    ) -> Result<(), Qwen35TpError> {
        check_feedback_plane(self.slots, slot, plane)?;
        if sampling && (self.rank != 0 || !self.exec.has_sample_rows()) {
            return Err(Qwen35TpError::Shape("TP device sampler unavailable".into()));
        }
        Ok(())
    }

    fn sample_feedback(&mut self, slot: usize, plane: usize) -> Result<CudaEvent, Qwen35TpError> {
        Self::sample_resident(
            &self.exec,
            &mut self.sample_chain,
            self.vocab,
            &self.logits,
            &self.sample_params,
            &mut self.feedback_ids[slot][plane],
        )
    }

    /// One `sample_rows` call joined into the cross-lane sampler chain: the
    /// pack's sample_rows A/B/C kernels share process-global pd_sr_* scratch,
    /// so every sampler waits the previous lane's in-flight sampler (device-
    /// side event chain - no host block) and fences itself for the next one.
    /// Returns an event marking this sampler's completion (for later
    /// event-ordered readback). An associated function so the caller can pass
    /// disjoint fields of one struct (receiver + args would conflict).
    fn sample_resident(
        exec: &GpuExecutor,
        chain: &mut Option<CudaEvent>,
        vocab: usize,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        out: &mut CudaSlice<u32>,
    ) -> Result<CudaEvent, Qwen35TpError> {
        if let Some(prev) = chain.take() {
            exec.stream.wait(&prev).map_err(GpuError::from)?;
        }
        exec.sample_rows(logits, params, out, 1, vocab)?;
        let event = exec.record_event()?;
        *chain = Some(exec.record_event()?);
        Ok(event)
    }

    fn forward_device_feedback<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        position: usize,
        slot: usize,
        plane: usize,
    ) -> Result<(), Qwen35TpError> {
        self.exec
            .stream
            .memcpy_dtod(&self.feedback_ids[slot][plane], &mut self.token)
            .map_err(GpuError::from)?;
        self.forward_token_body(group, logical_kv, position, slot, self.graphs_enabled)
    }

    fn forward_token_gpu<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        token: u32,
        position: usize,
        slot: usize,
    ) -> Result<(), Qwen35TpError> {
        if group.world_size() != 2
            || group.rank() != self.rank
            || position >= self.max_ctx
            || slot >= self.slots
        {
            return Err(Qwen35TpError::Shape("rank or position changed".into()));
        }
        self.exec
            .stream
            .memcpy_htod(&[token], &mut self.token)
            .map_err(GpuError::from)?;
        self.forward_token_body(group, logical_kv, position, slot, self.graphs_enabled)
    }

    /// Setup-time graph-mode decision (rank 0 / coordinator only). Reads the
    /// experimental dev switch (`PADDOCK_TP_GRAPH=1`; `dev_var!` keeps it out
    /// of hardened builds entirely) ONCE during serving setup. The resolved
    /// value is composed into `TpInit` for the worker and - on both ranks -
    /// becomes model state via `enable_tp_graphs`. Execution NEVER reads this
    /// function: token forwards consult the stored `graphs_enabled` field
    /// (upstream-readiness I4), so a serving rank's sequencing is fixed
    /// before the first token and a hand-started worker cannot pair eager
    /// with graphed collectives.
    pub fn resolve_graph_mode_for_serve() -> bool {
        paddock_models::dev_var!("PADDOCK_TP_GRAPH").as_deref() == Ok("1")
    }

    /// The resolved graph-execution mode (ordinary model state: true only
    /// after a successful `enable_tp_graphs`; default eager). Test surface
    /// for the I4 invariant; production code reads nothing else.
    pub fn graphs_enabled(&self) -> bool {
        self.graphs_enabled
    }

    fn forward_token_body<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        position: usize,
        slot: usize,
        graphed: bool,
    ) -> Result<(), Qwen35TpError> {
        if graphed {
            return self.forward_token_body_graphed(group, logical_kv, position, slot);
        }
        embed_any(
            &self.exec,
            &self.tok_embd,
            &self.token,
            &mut self.x,
            self.hidden,
            1,
            None,
        )?;
        for layer in &mut self.layers {
            self.exec.rmsnorm_batch(
                &self.x,
                &layer.attn_norm.buf,
                &mut self.xn,
                self.hidden,
                self.eps,
                1,
            )?;
            let mixed = match &mut layer.mixer {
                TpMixer::Full(gqa) => {
                    gqa.forward_paged(&self.exec, group, &self.xn, slot, position, logical_kv)?
                }
                TpMixer::Linear(delta) => delta.decode_slot(&self.exec, group, &self.xn, slot)?,
            };
            self.exec.add(&mut self.x, mixed, self.hidden)?;
            self.exec.rmsnorm_batch(
                &self.x,
                &layer.post_norm.buf,
                &mut self.xn,
                self.hidden,
                self.eps,
                1,
            )?;
            let ffn = layer.ffn.forward(&self.exec, group, &self.xn)?;
            self.exec.add(&mut self.x, ffn, self.hidden)?;
        }
        self.exec.rmsnorm_batch(
            &self.x,
            &self.out_norm.buf,
            &mut self.xn,
            self.hidden,
            self.eps,
            1,
        )?;
        gemv_any(&self.exec, &self.output, &self.xn, &mut self.logits)?;
        Ok(())
    }

    /// The graphed twin of `forward_token_body` (Phase 11 Stage A,
    /// `PADDOCK_TP_GRAPH=1`): identical math and identical NCCL sequencing,
    /// with each per-layer collective-free run replayed from the rank-local
    /// graph cache instead of enqueued kernel-by-kernel. The collectives,
    /// event fences and residual adds stay eager host-side (NCCL outside
    /// graph capture); the varying inputs (token embedding source, position,
    /// slot, block table) are buffer contents re-staged by plain memcpys
    /// between replays.
    ///
    /// DeltaNet layers key their captures per slot: the graph bakes the
    /// slot-swapped recurrent/conv buffer addresses, so `decode_slot` must
    /// have swapped the row's slot in BEFORE the replay and swapped it back
    /// after the collective - the same pointer swap the eager path does.
    /// A slot's DeltaNet capture is taken on its first graphed row.
    fn forward_token_body_graphed<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        position: usize,
        slot: usize,
    ) -> Result<(), Qwen35TpError> {
        // Embedding stays eager: one kernel, nothing to amortize.
        embed_any(
            &self.exec,
            &self.tok_embd,
            &self.token,
            &mut self.x,
            self.hidden,
            1,
            None,
        )?;
        for i in 0..self.layers.len() {
            // ── attention half: stage the varying inputs (position, slot,
            // block table - the graphed kernels read the buffers' CONTENT),
            // swap the DeltaNet state pair in for this slot, then replay the
            // captured run.
            match &mut self.layers[i].mixer {
                TpMixer::Full(gqa) => {
                    gqa.stage_decode_inputs(&self.exec, slot, position, logical_kv)?;
                }
                TpMixer::Linear(delta) => {
                    delta.swap_state_in(slot)?;
                }
            }
            let key = match &self.layers[i].mixer {
                TpMixer::Full(_) => TpRunKey::Attn(i),
                TpMixer::Linear(_) => TpRunKey::AttnDelta(i, slot),
            };
            if self.graphs.get(key).is_none() {
                // First graphed row for this (layer, slot): capture the run
                // live (DeltaNet's swapped state pair is in place right now).
                self.capture_mixer_run(group, i, slot)?;
            }
            self.graphs
                .get(key)
                .expect("captured above")
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("tp graph launch: {e}")))?;
            // ── the collective + residual, eager and outside capture.
            let mixed_ref = match &mut self.layers[i].mixer {
                TpMixer::Full(gqa) => gqa.finish(&self.exec, group)?,
                TpMixer::Linear(delta) => {
                    // The captured graph already staged `partial`. Swap the
                    // (advanced) state pair back first - the swap is host-side
                    // pointer surgery touching no stream work - then reduce.
                    delta.swap_state_out(slot)?;
                    delta.finish(&self.exec, group)?
                }
            };
            self.exec.add(&mut self.x, mixed_ref, self.hidden)?;
            // ── FFN half: the captured Ffn graph IS the norm + run (no
            // eager norm here - RMSNorm is not idempotent and the graph
            // bakes it).
            if self.graphs.get(TpRunKey::Ffn(i)).is_none() {
                self.capture_ffn_run(group, i)?;
            }
            self.graphs
                .get(TpRunKey::Ffn(i))
                .expect("captured above")
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("tp ffn graph launch: {e}")))?;
            let ffn_ref = self.layers[i].ffn.finish(&self.exec, group)?;
            self.exec.add(&mut self.x, ffn_ref, self.hidden)?;
        }
        self.exec.rmsnorm_batch(
            &self.x,
            &self.out_norm.buf,
            &mut self.xn,
            self.hidden,
            self.eps,
            1,
        )?;
        gemv_any(&self.exec, &self.output, &self.xn, &mut self.logits)?;
        Ok(())
    }

    /// Capture the FFN half's collective-free run for `layer` (rmsnorm +
    /// gate/up/swiglu/down), mirroring `capture_mixer_run`'s discipline.
    fn capture_ffn_run<C: Communicator>(&mut self, _group: &C, layer: usize) -> Result<(), Qwen35TpError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("tp ffn pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|e| GpuError::Driver(format!("tp ffn begin_capture: {e}")))?;
        let rec = (|| {
            self.exec.rmsnorm_batch(
                &self.x,
                &self.layers[layer].post_norm.buf,
                &mut self.xn,
                self.hidden,
                self.eps,
                1,
            )?;
            self.layers[layer].ffn.run(&self.exec, &self.xn)?;
            Ok::<(), Qwen35TpError>(())
        })();
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("tp ffn end_capture: {e}")));
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("tp ffn capture produced no graph".into()))?;
        self.graphs
            .insert(TpRunKey::Ffn(layer), super::SendGraph(graph));
        Ok(())
    }

    pub fn reset_slot(&mut self, slot: usize) -> Result<(), Qwen35TpError> {
        if slot >= self.slots {
            return Err(Qwen35TpError::Shape("slot out of range".into()));
        }
        for layer in &mut self.layers {
            if let TpMixer::Linear(delta) = &mut layer.mixer {
                delta.reset_slot(&self.exec, slot)?;
            }
        }
        Ok(())
    }

    /// Build the second execution lane for overlapped prefill spans.
    ///
    /// The decode lane's `Qwen35TpRank` keeps every weight it loaded; the
    /// prefill lane gets a structural re-home of the same weight/geometry
    /// onto a forked executor: weight planes are re-uploaded from the mapped
    /// GGUF onto the fork's stream (never shared with, or copied from, the
    /// decode lane's device buffers), and every scratch plane is freshly
    /// allocated there. GQA paged KV, DeltaNet recurrent/conv payloads and
    /// per-layer working buffers are lane-local, indexed by the SAME slot
    /// ids the coordinator authorizes; they start zeroed. Both ranks build
    /// a lane so the prefill collectives pair; only rank 0 ever samples.
    pub fn enable_prefill_lane<C: Communicator>(
        &mut self,
        group: &C,
        map: &MappedGguf,
    ) -> Result<(), Qwen35TpError> {
        if self.prefill.is_some() {
            return Ok(());
        }
        if group.rank() != self.rank {
            return Err(Qwen35TpError::Shape(
                "rank changed under the lane fork".into(),
            ));
        }
        // fork_stream synchronizes the decode stream internally before
        // creating the lane's streams; no queued decode work races them.
        let lane_exec = Arc::new(self.exec.fork_stream()?);
        let emb_name = "token_embd.weight";
        let lane_embd = {
            let ty = map
                .tensor_info(emb_name)
                .ok_or_else(|| Qwen35TpError::Shape("missing token embeddings".into()))?
                .ggml_type;
            if crate::gpu::kq_params(ty).is_some() {
                TokEmbd::Kq(lane_exec.repack_kquant(map, emb_name)?)
            } else {
                let tensor = lane_exec.upload_raw(map, emb_name)?;
                if tensor.ty != paddock_models::ggml_type::GgmlType::Q8_0 {
                    return Err(Qwen35TpError::Shape(format!(
                        "unsupported embedding type {:?}",
                        tensor.ty
                    )));
                }
                TokEmbd::Q8(tensor)
            }
        };
        let blocks = u32::try_from(
            self.max_ctx
                .div_ceil(crate::gpu_model::prefix_cache::BLOCK_TOKENS)
                .checked_mul(self.slots)
                .ok_or_else(|| Qwen35TpError::Shape("paged KV capacity overflow".into()))?,
        )
        .map_err(|_| Qwen35TpError::Shape("paged KV block count overflow".into()))?;
        let mut lane = Qwen35TpRank {
            exec: lane_exec.clone(),
            rank: self.rank,
            hidden: self.hidden,
            vocab: self.vocab,
            max_ctx: self.max_ctx,
            slots: self.slots,
            eps: self.eps,
            tok_embd: lane_embd,
            layers: Vec::with_capacity(self.layers.len()),
            out_norm: lane_exec.upload(map, "output_norm.weight")?,
            output: lane_exec.load_quantw(map, "output.weight")?,
            token: lane_exec.alloc_u32(1)?,
            x: lane_exec.alloc(self.hidden)?,
            xn: lane_exec.alloc(self.hidden)?,
            logits: lane_exec.alloc(self.vocab)?,
            sample_params: lane_exec.alloc_u32(4)?,
            sample_id: lane_exec.alloc_u32(1)?,
            feedback_ids: (0..self.slots)
                .map(|_| Ok([lane_exec.alloc_u32(1)?, lane_exec.alloc_u32(1)?]))
                .collect::<Result<Vec<_>, GpuError>>()?,
            sample_chain: None,
            prefill: None,
            graphs: TpGraphs::default(),
            // The prefill lane never captures or replays graphs (see
            // enable_tp_graphs), so its execution mode is always eager.
            graphs_enabled: false,
            // Phase 12: the lane serves the SAME KV dtype as the decode lane.
            // The hardcoded Fp16 here was a latent divergence: a future fp8
            // decode lane would have forked an f16 lane whose slabs and
            // attention kernels disagree with every promotion copy.
            kv_dtype: self.kv_dtype,
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("blk.{i}.");
            let attn_norm = lane_exec.upload(map, &format!("{prefix}attn_norm.weight"))?;
            let post_norm =
                lane_exec.upload(map, &format!("{prefix}post_attention_norm.weight"))?;
            let mixer = match &layer.mixer {
                TpMixer::Full(_) => TpMixer::Full(Box::new(
                    GqaTpRank::load_paged(
                        &lane_exec,
                        map,
                        i,
                        group,
                        self.max_ctx,
                        self.kv_dtype,
                        blocks,
                        self.slots,
                    )
                    .map_err(|e| Qwen35TpError::Shape(e.to_string()))?,
                )),
                TpMixer::Linear(_) => {
                    let mut delta = DeltaTpRank::load(&lane_exec, map, i, group)
                        .map_err(|e| Qwen35TpError::Shape(e.to_string()))?;
                    if self.slots > 1 {
                        delta
                            .enable_slots(&lane_exec, self.slots)
                            .map_err(Qwen35TpError::from)?;
                    }
                    TpMixer::Linear(Box::new(delta))
                }
            };
            let ffn = FfnTpRank::load(&lane_exec, map, i, group).map_err(Qwen35TpError::from)?;
            lane.layers.push(TpLayer {
                attn_norm,
                post_norm,
                mixer,
                ffn,
            });
        }
        self.prefill = Some(Box::new(PreFillLane {
            exec: lane_exec.clone(),
            model: Box::new(lane),
            owner: None,
            event: lane_exec.record_event()?,
            drained: None,
        }));
        Ok(())
    }

    /// True when the prefill lane exists and runs on a stream distinct from
    /// the decode lane's.
    pub fn has_prefill_lane(&self) -> bool {
        self.prefill
            .as_ref()
            .is_some_and(|lane| lane.exec.stream.cu_stream() != self.exec.stream.cu_stream())
    }

    /// Fence any outstanding promotion copies: the lane stream waits the
    /// decode-stream `drained` event device-side (no host block). Every
    /// lane enqueue path calls this first so new lane work cannot touch
    /// slabs a promotion is still reading.
    fn fence_lane_drained(lane: &mut PreFillLane) -> Result<(), Qwen35TpError> {
        if let Some(prev) = lane.drained.take() {
            lane.model.exec.stream.wait(&prev).map_err(GpuError::from)?;
        }
        Ok(())
    }

    /// Run one prefill-lane prompt row: `(slot, token, position)` on this
    /// rank's lane executor. No sampling and no host synchronization - the
    /// row's kernels and collectives are enqueued and the call returns. The
    /// caller must have validated the position and mirrored the KV event on
    /// BOTH ranks in the same order (the collectives pair across ranks).
    pub fn prefill_lane_step<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        slot: usize,
        token: u32,
        position: usize,
    ) -> Result<(), Qwen35TpError> {
        let lane = self
            .prefill
            .as_mut()
            .ok_or_else(|| Qwen35TpError::Shape("prefill lane not enabled".into()))?;
        if slot >= self.slots || position >= self.max_ctx {
            return Err(Qwen35TpError::Shape(
                "prefill slot or position invalid".into(),
            ));
        }
        Self::fence_lane_drained(lane)?;
        lane.owner = Some(slot);
        lane.model
            .exec
            .stream
            .memcpy_htod(&[token], &mut lane.model.token)
            .map_err(GpuError::from)?;
        // The prefill lane is never captured, so its execution mode is
        // always eager (its model's graphs_enabled stays false anyway).
        lane.model
            .forward_token_body(group, logical_kv, position, slot, false)
    }

    /// Enqueue the span finisher for `slot`'s final prefill row on the lane:
    /// `Some(plan)` samples the lane's resident logits on device (rank 0,
    /// fenced against every other lane's sampler via the process sample
    /// chain); `None` leaves the full logits for the host readback. Either
    /// way an event marking the lane's completion is returned.
    pub fn prefill_lane_finisher(
        &mut self,
        slot: usize,
        plan: Option<crate::sampler::DevicePlan>,
    ) -> Result<CudaEvent, Qwen35TpError> {
        let lane = self
            .prefill
            .as_mut()
            .ok_or_else(|| Qwen35TpError::Shape("prefill lane not enabled".into()))?;
        if slot >= self.slots {
            return Err(Qwen35TpError::Shape("finisher slot out of range".into()));
        }
        Self::fence_lane_drained(lane)?;
        let Some(plan) = plan else {
            // Host-finisher path (rank 1, or planless): park a probe event
            // so `prefill_lane_done` tracks this enqueue on every rank.
            let probe = lane.model.exec.record_event()?;
            let ev = lane.model.exec.record_event()?;
            lane.event = probe;
            return Ok(ev);
        };
        if self.rank != 0 || !self.exec.has_sample_rows() {
            return Err(Qwen35TpError::Shape("TP device sampler unavailable".into()));
        }
        let params = tp_sample_params(plan)?;
        // Cross-lane sampler serialization: the pack's sample_rows A/B/C
        // kernels use process-global pd_sr_* scratch, so this lane's sampler
        // must not overlap any other lane's in-flight sampler. Device-side
        // event chain - no host block.
        if let Some(prev) = self.sample_chain.take() {
            lane.model
                .exec
                .stream
                .wait(&prev)
                .map_err(GpuError::from)?;
        }
        lane.model
            .exec
            .stream
            .memcpy_htod(&params, &mut lane.model.sample_params)
            .map_err(GpuError::from)?;
        lane.model.exec.sample_rows(
            &lane.model.logits,
            &lane.model.sample_params,
            &mut lane.model.feedback_ids[slot][0],
            1,
            self.vocab,
        )?;
        // Three events at the same point: one parks in the sample chain
        // (waited by the next sampler anywhere in the process), one is
        // returned to the flight for readback, one stays in the lane for
        // the non-blocking `prefill_lane_done` probe.
        let chain = lane.model.exec.record_event()?;
        let ev = lane.model.exec.record_event()?;
        let probe = lane.model.exec.record_event()?;
        lane.event = probe;
        self.sample_chain = Some(chain);
        Ok(ev)
    }

    /// Non-blocking: has the lane's latest finisher (and everything enqueued
    /// before it) completed? True when the lane has no finisher event yet.
    pub fn prefill_lane_done(&self) -> bool {
        match self.prefill.as_ref() {
            Some(lane) => lane.exec.event_done(&lane.event),
            None => true,
        }
    }

    /// Read the sampled finisher ID after `ev` fires (lane copy stream).
    pub fn prefill_lane_sampled_id_after(
        &self,
        ev: &CudaEvent,
        slot: usize,
    ) -> Result<u32, Qwen35TpError> {
        let lane = self
            .prefill
            .as_ref()
            .ok_or_else(|| Qwen35TpError::Shape("prefill lane not enabled".into()))?;
        Ok(lane
            .model
            .exec
            .to_host_u32_after(ev, &lane.model.feedback_ids[slot][0], 0, 1)?[0])
    }

    /// Read the full logits row after `ev` fires (lane copy stream).
    pub fn prefill_lane_logits_after(&self, ev: &CudaEvent) -> Result<Vec<f32>, Qwen35TpError> {
        let lane = self
            .prefill
            .as_ref()
            .ok_or_else(|| Qwen35TpError::Shape("prefill lane not enabled".into()))?;
        Ok(lane
            .model
            .exec
            .to_host_len_after(ev, &lane.model.logits, self.vocab)?)
    }

    /// Join the prefill lane: block until all lane GPU work (and the
    /// collectives it entered) completes. The span finish fence.
    pub fn prefill_lane_join(&self) -> Result<(), Qwen35TpError> {
        let lane = self
            .prefill
            .as_ref()
            .ok_or_else(|| Qwen35TpError::Shape("prefill lane not enabled".into()))?;
        lane.model.exec.synchronize()?;
        Ok(())
    }

    /// Record an event on the lane stream marking everything enqueued so
    /// far. Both ranks call this after `prefill_lane_join` to hand
    /// `promote_lane_slot` a completion marker (the join already guarantees
    /// it fires immediately; the device-side wait keeps the promotion
    /// contract uniform across ranks).
    pub fn prefill_lane_mark(&mut self) -> Result<CudaEvent, Qwen35TpError> {
        let lane = self
            .prefill
            .as_mut()
            .ok_or_else(|| Qwen35TpError::Shape("prefill lane not enabled".into()))?;
        Ok(lane.model.exec.record_event()?)
    }

    /// Promote one finished prefill slot's lane-local state onto the decode
    /// executor (rank 0, after the span join; both ranks in identical order
    /// on their own executors).
    ///
    /// The lane's KV slab and DeltaNet slot state are private allocations
    /// (cudarc slices cannot share a handle), so a finished prompt's context
    /// would be stranded on the lane. This copies it: on the DECODE stream,
    /// waiting the lane's finisher event device-side (cross-stream ordered,
    /// no host sync). Per GQA layer, each of the slot's live physical blocks
    /// (K and V, block-strided slices at the same pool-wide block id - the
    /// paged pool layout is `[n_blocks, 16, kv_dim]` on every rank's slab);
    /// plus the DeltaNet recurrent/conv pair. The lane's dtype is Fp16,
    /// matching both executors' KV loads. Host cost is a few cudaMemcpy D2D
    /// launches (~1-3 ms per finished prompt) - the TTFT win survives.
    ///
    /// Leaves a decode-stream event in the lane's `drained` slot: the next
    /// lane enqueue waits it so a relaunched span (or a released slot's
    /// reallocated blocks) cannot write the lane slabs while promotion
    /// copies are still reading them.
    pub fn promote_lane_slot(
        &mut self,
        lane_done: &CudaEvent,
        slot: usize,
        live: &[u32],
    ) -> Result<(), Qwen35TpError> {
        let lane = self
            .prefill
            .as_mut()
            .ok_or_else(|| Qwen35TpError::Shape("prefill lane not enabled".into()))?;
        if slot >= self.slots {
            return Err(Qwen35TpError::Shape("promotion slot out of range".into()));
        }
        let dec = &self.exec;
        // Decode stream idles behind the lane's finisher: everything the
        // copies below read is complete before any copy launches.
        dec.stream.wait(lane_done).map_err(GpuError::from)?;
        // Lane and decode layers cannot borrow in one pass through the
        // mixer enum, so promote per layer by index.
        let lane_model = &lane.model;
        for i in 0..self.layers.len() {
            let (lane_mixer, dec_mixer) = {
                let lane_ref = &lane_model.layers[i].mixer;
                let dec_ref = &mut self.layers[i].mixer;
                (lane_ref, dec_ref)
            };
            match (lane_mixer, dec_mixer) {
                (TpMixer::Full(src), TpMixer::Full(dst)) => {
                    let (src_k, src_v) = src.kv_slabs();
                    let (dst_k, dst_v) = dst.kv_slabs_mut();
                    let stride = src.block_stride();
                    for &blk in live {
                        let base = blk
                            .checked_mul(stride as u32)
                            .and_then(|off| usize::try_from(off).ok())
                            .ok_or_else(|| {
                                Qwen35TpError::Shape("promotion block offset overflow".into())
                            })?;
                        if base + stride > src_k.len() {
                            return Err(Qwen35TpError::Shape(
                                "promotion block outside lane KV pool".into(),
                            ));
                        }
                        dec.copy_region(src_k, base, dst_k, base, stride)?;
                        dec.copy_region(src_v, base, dst_v, base, stride)?;
                    }
                }
                (TpMixer::Linear(src), TpMixer::Linear(dst)) => {
                    let (src_rec, src_conv) = src
                        .slot_state(slot)
                        .ok_or_else(|| Qwen35TpError::Shape("lane slot missing".into()))?;
                    let (dst_rec, dst_conv) = dst
                        .slot_state_mut(slot)
                        .ok_or_else(|| Qwen35TpError::Shape("decode slot missing".into()))?;
                    if src_rec.len() != dst_rec.len() || src_conv.len() != dst_conv.len() {
                        return Err(Qwen35TpError::Shape(
                            "lane and decode DeltaNet state shapes differ".into(),
                        ));
                    }
                    dec.stream
                        .memcpy_dtod(src_rec, dst_rec)
                        .map_err(GpuError::from)?;
                    dec.stream
                        .memcpy_dtod(src_conv, dst_conv)
                        .map_err(GpuError::from)?;
                }
                _ => {
                    return Err(Qwen35TpError::Shape(
                        "lane and decode mixer kinds diverge".into(),
                    ))
                }
            }
        }
        lane.drained = Some(dec.record_event()?);
        Ok(())
    }


    /// Reset one logical slot's lane-local state (DeltaNet recurrent/conv;
    /// paged KV is masked by position and rewritten from zero). Called on
    /// slot release/reuse in addition to the decode lane's reset.
    pub fn reset_lane_slot(&mut self, slot: usize) -> Result<(), Qwen35TpError> {
        if let Some(lane) = self.prefill.as_mut() {
            Self::fence_lane_drained(lane)?;
            lane.model.reset_slot(slot)?;
            lane.model.exec.synchronize()?;
        }
        Ok(())
    }

    /// Clear every lane slot's state (full flush).
    pub fn reset_lane(&mut self) -> Result<(), Qwen35TpError> {
        if let Some(lane) = self.prefill.as_mut() {
            Self::fence_lane_drained(lane)?;
            lane.model.reset()?;
            lane.model.exec.synchronize()?;
        }
        Ok(())
    }

    /// Clear recurrent state before an exact replay. Paged KV is logically
    /// reset by the harness and overwritten from position zero on the next run.
    pub fn reset(&mut self) -> Result<(), Qwen35TpError> {
        for slot in 0..self.slots {
            self.reset_slot(slot)?;
        }
        Ok(())
    }
}

fn check_feedback_plane(slots: usize, slot: usize, plane: usize) -> Result<(), Qwen35TpError> {
    if slot >= slots || plane >= 2 {
        return Err(Qwen35TpError::Shape(
            "feedback slot or plane out of range".into(),
        ));
    }
    Ok(())
}

pub(super) fn tp_sample_params(
    plan: crate::sampler::DevicePlan,
) -> Result<[u32; 4], Qwen35TpError> {
    use crate::sampler::DevicePlan;
    match plan {
        DevicePlan::Greedy => Ok([0, 0, 1, 0]),
        DevicePlan::Categorical { inv_t, u }
            if inv_t.is_finite() && inv_t > 0.0 && u.is_finite() && (0.0..1.0).contains(&u) =>
        {
            Ok([inv_t.to_bits(), u.to_bits(), 2, 0])
        }
        _ => Err(Qwen35TpError::Shape(
            "unsupported TP device sampling plan".into(),
        )),
    }
}

fn validate_dense_gemv_coverage(
    map: &MappedGguf,
    n_layers: usize,
    interval: usize,
) -> Result<(), Qwen35TpError> {
    use paddock_models::ggml_type::GgmlType;

    // A MoE qwen35 file (e.g. Qwen3.6-35B-A3B) has no dense per-layer
    // ffn_gate/up/down - without this check it fails below with
    // "missing tensor blk.0.ffn_gate.weight", which names nothing a person
    // can act on. Name the architecture instead. (qwen35moe TP is Phase 14
    // scope; the dense TP lane never accepts an expert quartet.)
    if map.gguf().arch_field("expert_count").is_some() {
        return Err(Qwen35TpError::Shape(
            "this is an MoE qwen35 checkpoint (expert_count present); the TP lane serves dense backbones only".into(),
        ));
    }
    for layer in 0..n_layers {
        let mut parts = vec!["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];
        if (layer + 1) % interval == 0 {
            parts.extend([
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            ]);
        } else {
            parts.extend(["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"]);
        }
        for part in parts {
            let name = format!("blk.{layer}.{part}");
            let ty = map
                .tensor_info(&name)
                .ok_or_else(|| Qwen35TpError::Shape(format!("missing tensor {name}")))?
                .ggml_type;
            if ty != GgmlType::Q8_0 && crate::gpu::kq_params(ty).is_none() {
                return Err(Qwen35TpError::Shape(format!(
                    "{name}: {ty:?} has no validated quantized repack/dequant dispatch"
                )));
            }
        }
    }
    let ty = map
        .tensor_info("output.weight")
        .ok_or_else(|| Qwen35TpError::Shape("missing tensor output.weight".into()))?
        .ggml_type;
    if ty != GgmlType::Q8_0 && crate::gpu::kq_params(ty).is_none() {
        return Err(Qwen35TpError::Shape(format!(
            "output.weight: {ty:?} has no validated quantized repack/dequant dispatch"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_feedback_plane, tp_sample_params};
    use crate::sampler::DevicePlan;
    use paddock_models::ggml_type::GgmlType;

    // I4 invariant, host-checkable halves. The graph cache is a private
    // field with no Default-constructible GPU state, so a full instance
    // needs CUDA; what IS provable here is the contract that makes the
    // runtime field the only execution input:
    // - the setup-time resolver is a pure function of the env read (dev
    //   builds only), never cached globally;
    // - the execution-mode field starts false on every construction path
    //   and only `enable_tp_graphs` (capture success) flips it - asserted
    //   structurally by the field initializer being the ONLY false assignment
    //   and enable_tp_graphs the ONLY true assignment (grep-guarded below).

    /// The resolver must not memoize: two calls re-evaluate (a OnceLock
    /// here would have re-created the exact global the I4 fix removes).
    #[test]
    fn graph_mode_resolver_is_a_pure_env_read_not_a_global() {
        // Dev-build behavior: the read is live. Hardened builds compile the
        // read out entirely (dev_var!), so the resolver is constant false
        // there - both satisfy "no global state".
        let first = super::Qwen35TpRank::resolve_graph_mode_for_serve();
        let second = super::Qwen35TpRank::resolve_graph_mode_for_serve();
        assert_eq!(first, second, "resolver must be deterministic per read");
    }

    /// Structural guard for the default-eager / enable-only-true contract:
    /// `graphs_enabled` must be initialized false at every construction
    /// site, and the ONLY `= true` assignment must live inside
    /// `enable_tp_graphs`. Only the production half of the file is scanned
    /// (everything before `#[cfg(test)]`), so the guard cannot match its
    /// own source text.
    #[test]
    fn graphs_enabled_is_default_false_and_flipped_only_by_enable_tp_graphs() {
        let src = production_source();
        // Exactly two construction sites (load_slots, enable_prefill_lane's
        // lane struct) carry the eager default.
        let constructions = src.matches("graphs_enabled: false").count();
        assert_eq!(
            constructions, 2,
            "expected exactly two construction sites defaulting graphs_enabled to \
             false; found {constructions}"
        );
        // Count `= true` assignments to the field (ignores the `: false`
        // initializers and the accessor).
        let true_assignments: Vec<&str> = src
            .lines()
            .filter(|l| l.contains("self.graphs_enabled = true"))
            .collect();
        assert_eq!(
            true_assignments.len(),
            1,
            "graphs_enabled must flip true in exactly one place \
             (enable_tp_graphs after successful capture); got {true_assignments:?}"
        );
        // And that place must be inside enable_tp_graphs: locate both.
        let enable_start = src
            .find("pub fn enable_tp_graphs")
            .expect("enable_tp_graphs exists");
        let assign_offset = src.find("self.graphs_enabled = true").expect("assignment exists");
        assert!(
            assign_offset > enable_start,
            "the only true-assignment must come after enable_tp_graphs begins"
        );
    }

    /// Execution must consult the stored field, not an environment read:
    /// no `tp_graph_enabled()` call may remain in any forward path, and the
    /// ONLY env read for graph mode is the setup-time resolver.
    #[test]
    fn token_forward_paths_consult_model_state_not_the_environment() {
        let src = production_source();
        assert!(
            !src.contains("tp_graph_enabled"),
            "stale tp_graph_enabled symbol in production code"
        );
        // Every `forward_token_body(` invocation must pass self.graphs_enabled.
        for (idx, line) in src.lines().enumerate() {
            if line.contains("forward_token_body(") && line.contains("self.graphs_enabled") {
                continue;
            }
            if line.contains("forward_token_body(")
                && !line.contains("fn forward_token_body")
                && !line.contains("forward_token_body(group, logical_kv, position, slot, false)")
                && !line.contains("//")
            {
                // Multi-line call form: check the next few lines carry the field.
                let window: String = src.lines().skip(idx).take(8).collect::<Vec<_>>().join("\n");
                assert!(
                    window.contains("self.graphs_enabled"),
                    "forward_token_body call at line {} must consult the stored \
                     graphs_enabled field, got: {window}",
                    idx + 1
                );
            }
        }
        // Exactly one executable env read for graph mode (the setup-time
        // resolver). Doc comments mentioning the name are fine.
        let env_reads: Vec<&str> = src
            .lines()
            .filter(|l| {
                l.contains("PADDOCK_TP_GRAPH") && !l.trim_start().starts_with("///")
            })
            .collect();
        assert_eq!(
            env_reads.len(),
            1,
            "exactly one executable PADDOCK_TP_GRAPH read may exist (the \
             setup-time resolver); got {env_reads:?}"
        );
    }

    /// The production half of tp_model.rs (everything before the test
    /// module), so source-structure guards cannot match their own text.
    fn production_source() -> &'static str {
        let src = include_str!("tp_model.rs");
        src.split("#[cfg(test)]").next().expect("nonempty")
    }

    #[test]
    fn feedback_slot_planes_are_bounded() {
        for slot in 0..2 {
            for plane in 0..2 {
                assert!(check_feedback_plane(2, slot, plane).is_ok());
            }
        }
        assert!(check_feedback_plane(2, 2, 0).is_err());
        assert!(check_feedback_plane(1, 0, 2).is_err());
    }

    #[test]
    fn tp_device_sampler_packs_only_supported_plans() {
        assert_eq!(tp_sample_params(DevicePlan::Greedy).unwrap(), [0, 0, 1, 0]);
        assert_eq!(
            tp_sample_params(DevicePlan::Categorical {
                inv_t: 2.0,
                u: 0.25
            })
            .unwrap(),
            [2.0f32.to_bits(), 0.25f32.to_bits(), 2, 0]
        );
        assert!(
            tp_sample_params(DevicePlan::Categorical {
                inv_t: f32::NAN,
                u: 0.5
            })
            .is_err()
        );
        assert!(tp_sample_params(DevicePlan::Categorical { inv_t: 1.0, u: 1.0 }).is_err());
        assert!(
            tp_sample_params(DevicePlan::TruncCat {
                inv_t: 1.0,
                u: 0.5,
                k: 10,
                top_p: 0.9,
                min_p: 0.0
            })
            .is_err()
        );
    }

    #[test]
    fn mixed_quant_types_have_repack_coverage() {
        for ty in [
            GgmlType::Iq4Xs,
            GgmlType::Iq4Nl,
            GgmlType::Iq3S,
            GgmlType::Q3K,
        ] {
            assert!(crate::gpu::kq_params(ty).is_some());
        }
    }
}
