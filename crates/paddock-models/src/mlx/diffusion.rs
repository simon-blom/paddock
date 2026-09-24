//! Exact text-graph contract for the mixed affine4/8 DiffusionGemma export.
use serde_json::Value;
use std::io::Read;
use std::path::Path;

pub struct DiffusionConfig {
    pub context: usize,
    pub quantization: Value,
}
impl DiffusionConfig {
    pub fn read(path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path.join("config.json"))
            .map_err(|e| e.to_string())?
            .take((1 << 20) + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 1 << 20 {
            return Err("DiffusionGemma config exceeds 1 MiB".into());
        }
        Self::parse(&serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)
    }
    pub fn parse(v: &Value) -> Result<Self, String> {
        let fail = || "unsupported DiffusionGemma MLX graph or quantization".to_owned();
        let t = &v["text_config"];
        if v["model_type"] != "diffusion_gemma"
            || v["canvas_length"] != 256
            || v["tie_word_embeddings"] != true
            || t["model_type"] != "diffusion_gemma_text"
            || t["hidden_activation"] != "gelu_pytorch_tanh"
            || t["attention_bias"] != false
            || t["tie_word_embeddings"] != true
            || t["use_bidirectional_attention"] != "vision"
        {
            return Err(fail());
        }
        for (key, expected) in [
            ("hidden_size", 2816),
            ("intermediate_size", 2112),
            ("moe_intermediate_size", 704),
            ("num_hidden_layers", 30),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 8),
            ("num_global_key_value_heads", 2),
            ("head_dim", 256),
            ("global_head_dim", 512),
            ("num_experts", 128),
            ("top_k_experts", 8),
            ("vocab_size", 262144),
            ("sliding_window", 1024),
        ] {
            if t[key] != expected {
                return Err(format!("DiffusionGemma requires {key}={expected}"));
            }
        }
        if t["rms_norm_eps"].as_f64() != Some(1e-6)
            || t["final_logit_softcapping"].as_f64() != Some(30.)
            || t["rope_parameters"]["full_attention"]["rope_theta"].as_f64() != Some(1_000_000.)
            || t["rope_parameters"]["full_attention"]["partial_rotary_factor"].as_f64()
                != Some(0.25)
            || t["rope_parameters"]["sliding_attention"]["rope_theta"].as_f64() != Some(10_000.)
            || t["rope_parameters"]["full_attention"]["rope_type"] != "proportional"
            || t["rope_parameters"]["sliding_attention"]["rope_type"] != "default"
        {
            return Err(fail());
        }
        // The backend implements this sampler contract. A newer checkpoint
        // must not silently run with different denoising/stopping settings.
        if let Some(g) = v.get("generation_config") {
            for (key, expected) in [
                ("max_denoising_steps", 48.),
                ("confidence_threshold", 0.005),
                ("stability_threshold", 1.),
                ("t_max", 0.8),
                ("t_min", 0.4),
            ] {
                if g[key].as_f64() != Some(expected) {
                    return Err(format!("DiffusionGemma requires {key}={expected}"));
                }
            }
            if g["sampler_config"]["_cls_name"] != "EntropyBoundSamplerConfig"
                || g["sampler_config"]["entropy_bound"].as_f64() != Some(0.1)
            {
                return Err("unsupported DiffusionGemma sampler".into());
            }
        }
        let layers = t["layer_types"].as_array().ok_or_else(fail)?;
        if layers.len() != 30
            || layers.iter().enumerate().any(|(i, x)| {
                x != if i % 6 == 5 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
        {
            return Err(fail());
        }
        let context = t["max_position_embeddings"]
            .as_u64()
            .filter(|&n| n > 0 && n <= 262144)
            .ok_or_else(fail)? as usize;
        let q = &v["quantization"];
        if q["group_size"] != 64
            || q["bits"] != 4
            || q.get("mode").is_some_and(|mode| mode != "affine")
        {
            return Err(fail());
        }
        Ok(Self {
            context,
            quantization: q.clone(),
        })
    }
    pub fn bits(&self, name: &str) -> Result<usize, String> {
        let q = self.quantization.get(name).unwrap_or(&self.quantization);
        if q.get("mode").is_some_and(|mode| mode != "affine") {
            return Err(format!("{name}: only affine quantization is supported"));
        }
        if q["group_size"] != 64 {
            return Err(format!("{name}: affine group must be 64"));
        }
        match q["bits"].as_u64() {
            Some(4) => Ok(4),
            Some(8) => Ok(8),
            _ => Err(format!("{name}: expected affine4 or affine8")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        json!({"model_type":"diffusion_gemma","canvas_length":256,"tie_word_embeddings":true,
            "quantization":{"group_size":64,"bits":4,"model.decoder.embed_tokens":{"group_size":64,"bits":8}},
            "text_config":{"model_type":"diffusion_gemma_text","hidden_activation":"gelu_pytorch_tanh","attention_bias":false,"use_bidirectional_attention":"vision",
                "tie_word_embeddings":true,"hidden_size":2816,"intermediate_size":2112,"moe_intermediate_size":704,
                "num_hidden_layers":30,"num_attention_heads":16,"num_key_value_heads":8,"num_global_key_value_heads":2,
                "head_dim":256,"global_head_dim":512,"num_experts":128,"top_k_experts":8,"vocab_size":262144,"sliding_window":1024,
                "max_position_embeddings":262144,"rms_norm_eps":1e-6,"final_logit_softcapping":30.0,
                "rope_parameters":{"full_attention":{"rope_type":"proportional","rope_theta":1000000.0,"partial_rotary_factor":0.25},"sliding_attention":{"rope_type":"default","rope_theta":10000.0}},
                "layer_types":(0..30).map(|i|if i%6==5 {"full_attention"} else {"sliding_attention"}).collect::<Vec<_>>()}})
    }
    #[test]
    fn mixed_precision_is_per_tensor_not_a_filename_assumption() {
        let c = DiffusionConfig::parse(&fixture()).unwrap();
        assert_eq!(c.context, 262144);
        assert_eq!(c.bits("model.decoder.embed_tokens").unwrap(), 8);
        assert_eq!(
            c.bits("model.decoder.layers.0.experts.gate_up_proj")
                .unwrap(),
            4
        );
    }
    #[test]
    fn rejects_graph_and_codec_changes() {
        for (key, value) in [
            ("/canvas_length", json!(128)),
            ("/text_config/head_dim", json!(128)),
            ("/text_config/top_k_experts", json!(4)),
            ("/text_config/use_bidirectional_attention", json!("all")),
            (
                "/text_config/rope_parameters/full_attention/rope_type",
                json!("linear"),
            ),
            ("/tie_word_embeddings", json!(false)),
            ("/quantization/group_size", json!(128)),
        ] {
            let mut v = fixture();
            *v.pointer_mut(key).unwrap() = value;
            assert!(DiffusionConfig::parse(&v).is_err(), "{key}");
        }
        let mut v = fixture();
        v["quantization"]["mode"] = json!("mxfp4");
        assert!(DiffusionConfig::parse(&v).is_err());
        v["quantization"]["mode"] = json!("affine");
        v["quantization"]["model.decoder.embed_tokens"]["bits"] = json!(2);
        assert!(
            DiffusionConfig::parse(&v)
                .unwrap()
                .bits("model.decoder.embed_tokens")
                .is_err()
        );
    }
    #[test]
    fn rejects_unknown_denoising_schedule() {
        let mut v = fixture();
        v["generation_config"] = json!({
            "max_denoising_steps":48,"confidence_threshold":0.005,"stability_threshold":1,
            "t_max":0.8,"t_min":0.4,
            "sampler_config":{"_cls_name":"EntropyBoundSamplerConfig","entropy_bound":0.1}});
        assert!(DiffusionConfig::parse(&v).is_ok());
        for (key, value) in [
            ("/generation_config/max_denoising_steps", json!(64)),
            ("/generation_config/t_min", json!(0.2)),
            ("/generation_config/confidence_threshold", json!(0.01)),
            (
                "/generation_config/sampler_config/entropy_bound",
                json!(0.2),
            ),
        ] {
            let mut changed = v.clone();
            *changed.pointer_mut(key).unwrap() = value;
            assert!(DiffusionConfig::parse(&changed).is_err(), "{key}");
        }
    }
}
