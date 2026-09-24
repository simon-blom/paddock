//! Plain Llama uses the same dense graph as Granite, with identity residual,
//! embedding and logit scales, and 1/sqrt(head_dim) attention. MiniCPM5's
//! official GGUF is this graph, not the older scaled MiniCPM architecture.
use super::*;
use paddock_models::gguf::GgufFile;

pub(super) fn validate(file: &GgufFile) -> Result<()> {
    for key in [
        "embedding_scale",
        "residual_scale",
        "logit_scale",
        "attention.scale",
        "attention.sliding_window",
        "attention.logit_softcapping",
        "final_logit_softcapping",
        "deepstack_mapping",
        "rope.scaling.type",
    ] {
        if file.arch_field(key).is_some() {
            return Err(MetalError::Model(format!("llama: unsupported {key}")));
        }
    }
    if file
        .arch_field("rope.scaling.factor")
        .is_some_and(|v| v.as_f32() != Some(1.0))
    {
        return Err(MetalError::Model(
            "llama: unsupported rotary scaling factor".into(),
        ));
    }
    let layers = file
        .arch_field("block_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let width = file.arch_field("embedding_length").and_then(Value::as_u64);
    let heads = file
        .arch_field("attention.head_count")
        .and_then(Value::as_u64);
    for key in ["attention.key_length", "attention.value_length"] {
        if let Some(value) = file.arch_field(key) {
            let head_dim = width.zip(heads).and_then(|(w, h)| w.checked_div(h));
            if head_dim.is_none() || value.as_u64() != head_dim {
                return Err(MetalError::Model(format!("llama: unsupported {key}")));
            }
        }
    }
    // Reject every unused tensor, including biases, Q/K norms, rotary factors
    // and draft planes. Never accept a variant and silently omit its math.
    for tensor in &file.tensors {
        let name = tensor.name.as_str();
        let valid = matches!(
            name,
            "token_embd.weight" | "output_norm.weight" | "output.weight"
        ) || name
            .strip_prefix("blk.")
            .and_then(|s| s.split_once('.'))
            .is_some_and(|(index, suffix)| {
                index
                    .parse::<u64>()
                    .is_ok_and(|i| i < layers && i.to_string() == index)
                    && matches!(
                        suffix,
                        "attn_norm.weight"
                            | "attn_q.weight"
                            | "attn_k.weight"
                            | "attn_v.weight"
                            | "attn_output.weight"
                            | "ffn_norm.weight"
                            | "ffn_gate.weight"
                            | "ffn_up.weight"
                            | "ffn_down.weight"
                    )
            });
        if !valid {
            return Err(MetalError::Model(format!(
                "llama: unconsumed tensor {name}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_models::{ggml_type::GgmlType, gguf::TensorInfo};

    fn header() -> GgufFile {
        GgufFile {
            version: 3,
            alignment: 32,
            data_offset: 0,
            metadata: [
                ("general.architecture".into(), Value::Str("llama".into())),
                ("llama.block_count".into(), Value::U32(42)),
                ("llama.embedding_length".into(), Value::U32(2048)),
                ("llama.attention.head_count".into(), Value::U32(16)),
            ]
            .into(),
            tensors: Vec::new(),
        }
    }
    #[test]
    fn refuses_silent_llama_architecture_changes() {
        assert!(validate(&header()).is_ok());
        for name in [
            "embedding_scale",
            "residual_scale",
            "logit_scale",
            "attention.scale",
            "attention.sliding_window",
            "attention.logit_softcapping",
            "final_logit_softcapping",
            "attention.key_length",
            "attention.value_length",
            "deepstack_mapping",
            "rope.scaling.type",
            "rope.scaling.factor",
        ] {
            let mut h = header();
            h.metadata.insert(format!("llama.{name}"), Value::F32(2.0));
            assert!(validate(&h).is_err(), "{name}");
        }
        let mut h = header();
        h.metadata
            .insert("llama.attention.key_length".into(), Value::U32(128));
        h.metadata
            .insert("llama.attention.value_length".into(), Value::U32(128));
        assert!(validate(&h).is_ok());
        for name in [
            "rope_freqs.weight",
            "blk.41.attn_q.bias",
            "blk.41.attn_q_norm.weight",
            "blk.42.attn_norm.weight",
            "blk.01.attn_norm.weight",
        ] {
            let mut h = header();
            h.tensors.push(TensorInfo {
                name: name.into(),
                dims: vec![128],
                ggml_type: GgmlType::F32,
                raw_type: 0,
                offset: 0,
            });
            assert!(validate(&h).is_err(), "{name}");
        }
    }
}
