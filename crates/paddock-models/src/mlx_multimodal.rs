//! Artifact-specific metadata for the approved Gemma/Muse MLX conversions.
//! A family name alone is not sufficient: these graphs include different
//! norm semantics, rotary layouts, image processors and quantized projectors.
use crate::safetensors::StError;
use serde_json::{Value, json};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultimodalFamily {
    Gemma31,
    Muse30,
}

#[derive(Clone, Debug)]
pub struct MultimodalConfig {
    pub family: MultimodalFamily,
    pub context: usize,
    pub eps: f32,
    pub rope: [f32; 2],
    pub softcap: f32,
    pub logit_scale: f32,
}

impl MultimodalConfig {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let path = dir.join("config.json");
        if std::fs::metadata(&path)?.len() > 1 << 20 {
            return Err(StError::Header("MLX config exceeds 1 MiB".into()));
        }
        let v = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let config = Self::parse(&v)?;
        let processor = dir.join("processor_config.json");
        if std::fs::metadata(&processor)?.len() > 1 << 20 {
            return Err(StError::Header("MLX processor config exceeds 1 MiB".into()));
        }
        let processor = serde_json::from_slice(&std::fs::read(processor)?)
            .map_err(|e| StError::Header(e.to_string()))?;
        config.validate_processor(&processor)?;
        Ok(config)
    }

    fn validate_processor(&self, v: &Value) -> Result<(), StError> {
        let muse = self.family == MultimodalFamily::Muse30;
        let image = &v["image_processor"];
        let mut expected = if muse {
            json!({"image_processor_type":"MuseGlimmerImageProcessor", "patch_size":14,
                "temporal_patch_size":2,"merge_size":2,"max_image_tokens":4096,
                "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]})
        } else {
            json!({"image_processor_type":"Gemma4ImageProcessor", "patch_size":16,
                "pooling_kernel_size":3,"max_soft_tokens":280,"image_seq_length":280,
                "do_convert_rgb":true,"do_resize":true,"do_rescale":true,"do_normalize":false,
                "resample":3,"image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]})
        };
        expected["rescale_factor"] = json!(1.0 / 255.0);
        if v["processor_class"]
            != if muse {
                "MuseGlimmerProcessor"
            } else {
                "Gemma4Processor"
            }
            || expected
                .as_object()
                .expect("fixed image processor object")
                .iter()
                .any(|(k, value)| &image[k] != value)
        {
            return Err(StError::Header("MLX Gemma/Muse: unsupported image processor; do not silently reinterpret checkpoint pixels".into()));
        }
        Ok(())
    }

    pub fn parse(v: &Value) -> Result<Self, StError> {
        let bad = |s: &str| StError::Header(format!("MLX Gemma/Muse: {s}"));
        let family = match v["model_type"].as_str() {
            Some("gemma4") => MultimodalFamily::Gemma31,
            Some("muse_glimmer") => MultimodalFamily::Muse30,
            _ => return Err(bad("expected gemma4 or muse_glimmer")),
        };
        let muse = family == MultimodalFamily::Muse30;
        let quant = json!({"bits":4,"group_size":64,"mode":"affine"});
        if v.get("quantization")
            .or_else(|| v.get("quantization_config"))
            != Some(&quant)
            || v.get("quantization_config").is_some_and(|q| q != &quant)
            || v["dtype"] != "bfloat16"
            || v.get("audio_config").is_some_and(|a| !a.is_null())
        {
            return Err(bad("requires affine4/group64, BF16, no audio tower"));
        }
        let t = &v["text_config"];
        let (width, ff, count, vocab, kh, hd, window, period) = if muse {
            (6656, 19968, 52, 202048, 2, 128, 2048, 4)
        } else {
            (5376, 21504, 60, 262144, 16, 256, 1024, 6)
        };
        for (key, expected) in [
            ("hidden_size", width),
            ("intermediate_size", ff),
            ("num_hidden_layers", count),
            ("vocab_size", vocab),
            ("num_attention_heads", 32),
            ("num_key_value_heads", kh),
            ("head_dim", hd),
            ("sliding_window", window),
        ] {
            if t[key].as_u64() != Some(expected) {
                return Err(bad(&format!("unsupported text {key}")));
            }
        }
        if t["attention_bias"] != false
            || t["tie_word_embeddings"] != !muse
            || t["hidden_activation"] != if muse { "silu" } else { "gelu_pytorch_tanh" }
            || t.get("attention_dropout")
                .is_some_and(|x| x.as_f64() != Some(0.))
            || t.get("rope_traditional").is_some_and(|x| x != false)
        {
            return Err(bad("unsupported text arithmetic"));
        }
        let types = t["layer_types"]
            .as_array()
            .ok_or_else(|| bad("missing layer_types"))?;
        if types.len() != count as usize
            || types.iter().enumerate().any(|(i, x)| {
                x != if (i + 1) % period == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
        {
            return Err(bad("unsupported layer ordering"));
        }
        if muse {
            if t["qk_scale_factor"] != json!(3.87)
                || t["post_norm_eps"] != json!(1e-8)
                || t["rope_parameters"] != json!({"rope_theta":500000.0,"rope_type":"default"})
                || t["layer_rope_theta"].as_array().is_none_or(|r| {
                    r.len() != count as usize
                        || r.iter()
                            .enumerate()
                            .any(|(i, x)| x.as_f64() != Some(if i % 4 == 3 { 0. } else { 500000. }))
                })
                || v["out_hidden_size"] != 6144
                || v["projector_hidden_size"] != 4096
                || v["projector_hidden_act"] != "gelu"
                || v["image_token_id"] != 200092
            {
                return Err(bad("unsupported Muse rotary/projector/norm configuration"));
            }
        } else if t["global_head_dim"] != 512
            || t["num_global_key_value_heads"] != 4
            || t["attention_k_eq_v"] != true
            || t["enable_moe_block"] != false
            || t["hidden_size_per_layer_input"] != 0
            || t["num_kv_shared_layers"] != 0
            || t["use_double_wide_mlp"] != false
            || t["use_bidirectional_attention"] != "vision"
            || t["rope_parameters"]
                != json!({
                "full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},
                "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}})
            || v["image_token_id"] != 258880
            || v["boi_token_id"] != 255999
            || v["eoi_token_id"] != 258882
            || v["vision_soft_tokens_per_image"] != 280
        {
            return Err(bad("unsupported Gemma rotary/KV/vision configuration"));
        }
        let vision = &v["vision_config"];
        for (key, expected) in [
            ("hidden_size", if muse { 1536 } else { 1152 }),
            ("intermediate_size", if muse { 8960 } else { 4304 }),
            ("num_hidden_layers", if muse { 50 } else { 27 }),
            ("num_attention_heads", 16),
            ("patch_size", if muse { 14 } else { 16 }),
        ] {
            if vision[key] != expected {
                return Err(bad(&format!("unsupported vision {key}")));
            }
        }
        if muse {
            if vision["patch_temporal"] != 2
                || vision["merge_size"] != 2
                || vision["pos_emb_height"] != 32
                || vision["pos_emb_width"] != 32
                || vision["hidden_act"] != "gelu"
                || vision["layer_norm_eps"] != json!(1e-5)
                || vision["rope_parameters"] != json!({"rope_theta":10000.0,"rope_type":"default"})
                || vision["layer_types"].as_array().is_none_or(|r| {
                    r.len() != 50
                        || r.iter().enumerate().any(|(i, x)| {
                            x != if i % 4 == 3 || i == 49 {
                                "full_attention"
                            } else {
                                "window_attention"
                            }
                        })
                })
            {
                return Err(bad("unsupported Muse vision configuration"));
            }
        } else if vision["head_dim"] != 72
            || vision["num_key_value_heads"] != 16
            || vision["pooling_kernel_size"] != 3
            || vision["position_embedding_size"] != 10240
            || vision["standardize"] != true
            || vision["use_clipped_linears"] != false
            || vision["hidden_activation"] != "gelu_pytorch_tanh"
            || vision["rms_norm_eps"] != json!(1e-6)
            || vision["rope_parameters"] != json!({"rope_theta":100.0,"rope_type":"default"})
        {
            return Err(bad("unsupported Gemma vision configuration"));
        }
        let positive = |key: &str| {
            t[key]
                .as_f64()
                .map(|v| v as f32)
                .filter(|v| v.is_finite() && *v > 0.)
                .ok_or_else(|| bad(key))
        };
        let context = t["max_position_embeddings"]
            .as_u64()
            .filter(|&n| n > 0 && n <= if muse { 131072 } else { 262144 })
            .ok_or_else(|| bad("invalid trained context"))? as usize;
        Ok(Self {
            family,
            context,
            eps: positive("rms_norm_eps")?,
            rope: if muse {
                [500000., 500000.]
            } else {
                [10000., 1000000.]
            },
            softcap: positive("final_logit_softcapping")?,
            logit_scale: if muse {
                positive("output_multiplier")?
            } else {
                1.
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn configs() -> [Value; 2] {
        [
            include_str!("../tests/fixtures/mlx-gemma31-config.json"),
            include_str!("../tests/fixtures/mlx-muse30-config.json"),
        ]
        .map(|s| serde_json::from_str(s).unwrap())
    }
    #[test]
    fn pinned_multimodal_metadata_has_distinct_graphs() {
        let [gemma, muse] = configs().map(|v| MultimodalConfig::parse(&v).unwrap());
        assert_eq!(gemma.family, MultimodalFamily::Gemma31);
        assert_eq!(gemma.context, 262144);
        assert_eq!(muse.family, MultimodalFamily::Muse30);
        assert_eq!(muse.context, 131072);
        assert_eq!(muse.eps, 1e-5);
    }
    #[test]
    fn image_processor_metadata_is_not_advisory() {
        let [gemma, muse] = configs().map(|v| MultimodalConfig::parse(&v).unwrap());
        let pairs = [
            (
                gemma,
                json!({"processor_class":"Gemma4Processor","image_processor":{
            "image_processor_type":"Gemma4ImageProcessor","patch_size":16,"pooling_kernel_size":3,
            "max_soft_tokens":280,"image_seq_length":280,"do_convert_rgb":true,"do_resize":true,
            "do_rescale":true,"do_normalize":false,"resample":3,"image_mean":[0.0,0.0,0.0],
            "image_std":[1.0,1.0,1.0],"rescale_factor":1.0/255.0}}),
            ),
            (
                muse,
                json!({"processor_class":"MuseGlimmerProcessor","image_processor":{
                "image_processor_type":"MuseGlimmerImageProcessor","patch_size":14,"temporal_patch_size":2,
                "merge_size":2,"max_image_tokens":4096,"image_mean":[0.5,0.5,0.5],
                "image_std":[0.5,0.5,0.5],"rescale_factor":1.0/255.0}}),
            ),
        ];
        for (config, good) in pairs {
            config.validate_processor(&good).unwrap();
            for key in ["patch_size", "image_mean", "image_std", "rescale_factor"] {
                let mut bad = good.clone();
                bad["image_processor"][key] = json!(0);
                assert!(config.validate_processor(&bad).is_err(), "accepted {key}");
            }
            assert!(config.validate_processor(&json!({})).is_err());
        }
    }
    #[test]
    fn rejects_unknown_quant_geometry_norm_rotary_and_vision() {
        for good in configs() {
            for (path, value) in [
                ("/model_type", json!("gemma4_text")),
                ("/quantization/bits", json!(8)),
                ("/quantization_config/group_size", json!(32)),
                ("/text_config/hidden_size", json!(2816)),
                ("/text_config/rms_norm_eps", json!(0)),
                ("/text_config/max_position_embeddings", json!(0)),
                ("/text_config/attention_bias", json!(true)),
                ("/text_config/layer_types/0", json!("full_attention")),
                ("/vision_config/patch_size", json!(32)),
                ("/vision_config/num_hidden_layers", json!(2)),
                ("/image_token_id", json!(123)),
            ] {
                let mut v = good.clone();
                *v.pointer_mut(path).unwrap() = value;
                assert!(MultimodalConfig::parse(&v).is_err(), "accepted {path}");
            }
            let mut v = good.clone();
            v["quantization"]["vision_tower"] = json!(false);
            assert!(MultimodalConfig::parse(&v).is_err());
        }
        let [mut gemma, mut muse] = configs();
        gemma["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
            json!(1.0);
        assert!(MultimodalConfig::parse(&gemma).is_err());
        muse["text_config"]["qk_scale_factor"] = json!(1.0);
        assert!(MultimodalConfig::parse(&muse).is_err());
    }
}
