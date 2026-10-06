//! Shared GGUF naming and acceptance policy for the bias-free plain-llama
//! graph served by the Granite family.
//!
//! Granite is the same dense stack plus four scalar multipliers. Keeping the
//! plain-llama gate and tensor names here gives the normal and TP loaders one
//! source of truth instead of letting the TP adapter grow a second model
//! definition.

use paddock_models::mapped::MappedGguf;

use crate::gpu_model::gpt_oss::GpuModelError;

pub(crate) const TOKEN_EMBD_WEIGHT: &str = "token_embd.weight";
pub(crate) const OUTPUT_NORM_WEIGHT: &str = "output_norm.weight";
pub(crate) const OUTPUT_WEIGHT: &str = "output.weight";

pub(crate) fn layer_weight(layer: usize, part: &str) -> String {
    format!("blk.{layer}.{part}")
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Policy {
    pub embedding_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    pub attention_scale: f32,
}

/// Validate the subset of `general.architecture = llama` that the Granite
/// graph implements. This is the policy shared by single-GPU and TP:
/// bias-free dense GQA/SwiGLU, full NORM-convention rotary, no unhandled rope
/// scaling, and no Granite-only scalar metadata.
pub(crate) fn validate(
    map: &MappedGguf,
    max_ctx: usize,
    head_dim: usize,
    ctx_train: usize,
) -> Result<Policy, GpuModelError> {
    let arch = map.gguf().architecture().unwrap_or("");
    if arch != "llama" {
        return Err(GpuModelError::Unsupported(format!(
            "plain-llama policy expected general.architecture=llama, got {arch:?}"
        )));
    }

    for k in [
        "embedding_scale",
        "residual_scale",
        "logit_scale",
        "attention.scale",
    ] {
        if map.gguf().arch_field(k).is_some() {
            return Err(GpuModelError::Unsupported(format!(
                "llama: file stamps llama.{k}, which no llama graph applies - refusing rather than ignoring a scalar the author thought mattered"
            )));
        }
    }

    if map.tensor_info("blk.0.attn_q.bias").is_some()
        || map.tensor_info("blk.0.attn_k.bias").is_some()
        || map.tensor_info("blk.0.attn_v.bias").is_some()
    {
        return Err(GpuModelError::Unsupported(
            "llama: attention biases present - this graph serves the bias-free llama stack only"
                .into(),
        ));
    }

    let rope_factor = map
        .gguf()
        .arch_field("rope.scaling.factor")
        .and_then(paddock_models::gguf::Value::as_f32);
    if map.gguf().arch_field("rope.scaling.type").is_some()
        || rope_factor.is_some_and(|v| v != 1.0)
        || map.tensor_info("rope_freqs.weight").is_some()
    {
        return Err(GpuModelError::Unsupported(
            "llama: rope scaling is stamped but unhandled - refusing to ignore it".into(),
        ));
    }

    if max_ctx > ctx_train {
        return Err(GpuModelError::Unsupported(format!(
            "llama: requested context {max_ctx} exceeds the trained {ctx_train} and the file stamps no rope scaling - serving past it would extrapolate silently"
        )));
    }

    Ok(Policy {
        embedding_scale: 1.0,
        residual_scale: 1.0,
        logit_scale: 1.0,
        attention_scale: 1.0 / (head_dim as f32).sqrt(),
    })
}
