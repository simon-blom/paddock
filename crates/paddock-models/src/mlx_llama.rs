//! The official MiniCPM5-2B MLX affine checkpoint, not arbitrary Llama exports.
use crate::safetensors::StError;
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct MiniCpmConfig {
    pub context: usize,
    pub eps: f32,
    pub rope: f32,
}
impl MiniCpmConfig {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let path = dir.join("config.json");
        if std::fs::metadata(&path)?.len() > 1 << 20 {
            return Err(StError::Header("MiniCPM MLX config exceeds 1 MiB".into()));
        }
        let v = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| StError::Header(format!("MiniCPM MLX: {e}")))?;
        Self::parse(&v)
    }
    pub fn parse(v: &Value) -> Result<Self, StError> {
        let bad = |s: &str| StError::Header(format!("MiniCPM5 MLX: unsupported {s}"));
        for (key, expected) in [
            ("hidden_size", 2048),
            ("intermediate_size", 6144),
            ("num_hidden_layers", 42),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 2),
            ("head_dim", 128),
            ("vocab_size", 130560),
            ("max_position_embeddings", 131072),
        ] {
            if v[key].as_u64() != Some(expected) {
                return Err(bad(key));
            }
        }
        for (key, expected) in [
            ("model_type", "llama"),
            ("hidden_act", "silu"),
            ("torch_dtype", "bfloat16"),
        ] {
            if v[key] != expected {
                return Err(bad(key));
            }
        }
        if v["tie_word_embeddings"] != false {
            return Err(bad("tied embedding"));
        }
        if v["architectures"] != serde_json::json!(["LlamaForCausalLM"]) {
            return Err(bad("architectures"));
        }
        if v.get("dtype").is_some_and(|x| x != "bfloat16") {
            return Err(bad("dtype"));
        }
        for key in ["attention_bias", "mlp_bias", "rope_traditional"] {
            if v.get(key).is_some_and(|x| x != false) {
                return Err(bad(key));
            }
        }
        for key in [
            "rope_scaling",
            "rope_parameters",
            "sliding_window",
            "layer_types",
            "attention_multiplier",
            "embedding_multiplier",
            "residual_multiplier",
            "logits_scaling",
            "partial_rotary_factor",
        ] {
            if v.get(key).is_some_and(|x| !x.is_null()) {
                return Err(bad(key));
            }
        }
        let q = v
            .get("quantization")
            .or_else(|| v.get("quantization_config"))
            .ok_or_else(|| bad("missing quantization"))?;
        for q in std::iter::once(q).chain(v.get("quantization_config")) {
            if q["bits"] != 4
                || q["group_size"] != 64
                || q["mode"] != "affine"
                || q.as_object().is_none_or(|q| q.len() != 3)
            {
                return Err(bad("quantization (requires uniform affine4/group64)"));
            }
        }
        let positive = |key| {
            v[key]
                .as_f64()
                .map(|f| f as f32)
                .filter(|f| f.is_finite() && *f > 0.)
                .ok_or_else(|| bad(key))
        };
        Ok(Self {
            context: 131072,
            eps: positive("rms_norm_eps")?,
            rope: positive("rope_theta")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_minicpm_mlx_contract() {
        let good = serde_json::json!({"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":42,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":128,"vocab_size":130560,"max_position_embeddings":131072,"hidden_act":"silu","torch_dtype":"bfloat16","tie_word_embeddings":false,"rope_scaling":null,"rope_theta":5000000,"rms_norm_eps":1e-6,"quantization":{"bits":4,"group_size":64,"mode":"affine"}});
        assert!(MiniCpmConfig::parse(&good).is_ok());
        for (key, value) in [
            ("model_type", serde_json::json!("minicpm")),
            ("head_dim", serde_json::json!(64)),
            ("architectures", serde_json::json!(["MiniCPMForCausalLM"])),
            ("dtype", serde_json::json!("float16")),
            ("partial_rotary_factor", serde_json::json!(0.5)),
            ("attention_bias", serde_json::json!(true)),
            ("rope_theta", serde_json::json!(0)),
            ("sliding_window", serde_json::json!(4096)),
            ("rope_traditional", serde_json::json!(true)),
            (
                "quantization_config",
                serde_json::json!({"bits":8,"group_size":64,"mode":"affine"}),
            ),
        ] {
            let mut v = good.clone();
            v[key] = value;
            assert!(MiniCpmConfig::parse(&v).is_err(), "{key}");
        }
    }
}
