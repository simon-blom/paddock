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
    Full(GqaTpRank),
    Linear(DeltaTpRank),
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
                TpMixer::Full(
                    GqaTpRank::load_paged(&exec, map, i, group, max_ctx, dtype, blocks, slots)
                        .map_err(|e| Qwen35TpError::Shape(e.to_string()))?,
                )
            } else {
                TpMixer::Linear({
                    let mut delta = DeltaTpRank::load(&exec, map, i, group)
                        .map_err(|e| Qwen35TpError::Shape(e.to_string()))?;
                    if slots > 1 {
                        delta.enable_slots(&exec, slots)?;
                    }
                    delta
                })
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
        })
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

    pub fn supports_device_sampling(&self) -> bool {
        self.rank == 0 && self.exec.has_sample_rows()
    }

    pub fn synchronize(&self) -> Result<(), Qwen35TpError> {
        self.exec.synchronize()?;
        Ok(())
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
        self.exec.sample_rows(
            &self.logits,
            &self.sample_params,
            &mut self.sample_id,
            1,
            self.vocab,
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
        self.exec.sample_rows(
            &self.logits,
            &self.sample_params,
            &mut self.feedback_ids[slot][plane],
            1,
            self.vocab,
        )?;
        Ok(self.exec.record_event()?)
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
        self.forward_token_body(group, logical_kv, position, slot)
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
        self.forward_token_body(group, logical_kv, position, slot)
    }

    fn forward_token_body<C: Communicator>(
        &mut self,
        group: &C,
        logical_kv: &crate::gpu_model::qwen35::tp_kv::MirroredKv,
        position: usize,
        slot: usize,
    ) -> Result<(), Qwen35TpError> {
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
