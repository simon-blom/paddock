//! Prism's schema-2 MLX checkpoint contract. Never treat rotated weights as
//! ordinary Qwen affine weights, and never execute checkpoint-supplied Python.
use crate::{mlx::QwenConfig, safetensors::StError};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Clone, Debug)]
pub struct BonsaiConfig {
    pub text: QwenConfig,
    pub signs: BTreeMap<usize, Vec<f32>>,
}

fn bad(s: impl Into<String>) -> StError {
    StError::Header(format!("Bonsai MLX: {}", s.into()))
}

fn read_json(path: &Path) -> Result<Value, StError> {
    if std::fs::metadata(path)?.len() > 1 << 20 {
        return Err(bad("metadata exceeds 1 MiB"));
    }
    serde_json::from_slice(&std::fs::read(path)?).map_err(|e| bad(e.to_string()))
}

/// Complete transform manifest for the elected dense-27B graph. Scalar gate
/// projections are F32, unrotated, and intentionally absent from this list.
pub fn packed_modules() -> BTreeMap<String, usize> {
    let mut modules = BTreeMap::from([
        ("model.embed_tokens".into(), 5120),
        ("lm_head".into(), 5120),
    ]);
    for layer in 0..64 {
        for (suffix, width) in [
            ("mlp.gate_proj", 5120),
            ("mlp.up_proj", 5120),
            ("mlp.down_proj", 17408),
        ]
        .into_iter()
        .chain(if (layer + 1) % 4 == 0 {
            vec![
                ("self_attn.q_proj", 5120),
                ("self_attn.k_proj", 5120),
                ("self_attn.v_proj", 5120),
                ("self_attn.o_proj", 6144),
            ]
        } else {
            vec![
                ("linear_attn.in_proj_qkv", 5120),
                ("linear_attn.in_proj_z", 5120),
                ("linear_attn.out_proj", 6144),
            ]
        }) {
            modules.insert(format!("model.layers.{layer}.{suffix}"), width);
        }
    }
    modules
}

impl BonsaiConfig {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        Self::parse(
            &read_json(&dir.join("config.json"))?,
            &read_json(&dir.join("hadamard.json"))?,
        )
    }

    pub fn parse(config: &Value, rotation: &Value) -> Result<Self, StError> {
        for (key, expected) in [
            ("schema_version", json!(2)),
            ("model_type", json!("prism_hadamard_qwen35")),
            ("base_model_type", json!("qwen3_5")),
            ("tie_word_embeddings", json!(false)),
            ("tensor_namespace", json!("mlx-vlm-qwen3_5")),
            ("gdn_activation_layout", json!("grouped")),
            ("hadamard_config", json!("hadamard.json")),
            ("components", json!({"text":true,"vision":true,"mtp":false})),
            (
                "quantization",
                json!({"bits":2,"group_size":128,"mode":"affine"}),
            ),
        ] {
            if config[key] != expected {
                return Err(bad(format!("unsupported {key}")));
            }
        }
        if config
            .get("quantization_config")
            .is_some_and(|q| q != &config["quantization"])
            || config["text_config"]["mtp_num_hidden_layers"] != 0
        {
            return Err(bad("conflicting quantization or MTP configuration"));
        }
        let text = QwenConfig::parse_text(&config["text_config"])?;
        for (key, value) in [
            ("depth", json!(27)),
            ("hidden_size", json!(1152)),
            ("intermediate_size", json!(4304)),
            ("num_heads", json!(16)),
            ("num_position_embeddings", json!(2304)),
            ("out_hidden_size", json!(5120)),
            ("patch_size", json!(16)),
            ("spatial_merge_size", json!(2)),
            ("temporal_patch_size", json!(2)),
            ("in_channels", json!(3)),
            ("deepstack_visual_indexes", json!([])),
            ("hidden_act", json!("gelu_pytorch_tanh")),
        ] {
            if config["vision_config"][key] != value {
                return Err(bad(format!("unsupported vision {key}")));
            }
        }
        let expected = packed_modules();
        let modules = config["modules"]
            .as_array()
            .ok_or_else(|| bad("missing modules"))?;
        let mut seen = BTreeSet::new();
        for record in modules {
            let name = record["path"]
                .as_str()
                .ok_or_else(|| bad("invalid module path"))?;
            if !expected.contains_key(name)
                || !seen.insert(name.to_owned())
                || record["block"] != 1024
                || record["dtype"] != "float16"
                || record["embedding"] != (name == "model.embed_tokens")
            {
                return Err(bad(format!("unsupported/duplicate packed module {name}")));
            }
        }
        if seen != expected.keys().cloned().collect() {
            return Err(bad("incomplete packed module manifest"));
        }
        let prefix = "prism.hadamard.";
        let fields = rotation
            .as_object()
            .ok_or_else(|| bad("invalid rotation metadata"))?;
        let known = [
            "version",
            "block_size",
            "transform",
            "axis",
            "sign_mode",
            "weight_names",
            "inverse_weight_names",
            "sign_widths",
            "sign_values",
            "gdn_v_grouped",
        ];
        if fields.len() != known.len()
            || fields
                .keys()
                .any(|k| !known.iter().any(|s| k == &format!("{prefix}{s}")))
        {
            return Err(bad("unknown or missing rotation key"));
        }
        for (key, value) in [
            ("version", json!(1)),
            ("block_size", json!(1024)),
            ("transform", json!("normalized-sylvester-walsh-hadamard")),
            ("axis", json!("input-last-dimension")),
            ("sign_mode", json!("explicit")),
            ("gdn_v_grouped", json!(true)),
            (
                "inverse_weight_names",
                json!(["language_model.model.embed_tokens.weight"]),
            ),
        ] {
            if rotation[format!("{prefix}{key}")] != value {
                return Err(bad(format!("unsupported {prefix}{key}")));
            }
        }
        let names = rotation[format!("{prefix}weight_names")]
            .as_array()
            .ok_or_else(|| bad("missing weight names"))?;
        let names: Vec<_> = names
            .iter()
            .map(|s| s.as_str().ok_or_else(|| bad("invalid weight name")))
            .collect::<Result<_, _>>()?;
        let want: BTreeSet<_> = expected
            .keys()
            .filter(|s| s.as_str() != "model.embed_tokens")
            .map(|s| format!("language_model.{s}.weight"))
            .collect();
        if names.len() != want.len()
            || names
                .iter()
                .map(|s| (*s).to_owned())
                .collect::<BTreeSet<_>>()
                != want
        {
            return Err(bad("rotation manifest does not match the graph"));
        }
        let widths = rotation[format!("{prefix}sign_widths")]
            .as_array()
            .ok_or_else(|| bad("missing sign widths"))?;
        let values = rotation[format!("{prefix}sign_values")]
            .as_array()
            .ok_or_else(|| bad("missing signs"))?;
        let mut signs = BTreeMap::new();
        let mut offset = 0;
        for width in widths {
            let width = width
                .as_u64()
                .filter(|w| matches!(w, 5120 | 6144 | 17408))
                .ok_or_else(|| bad("unsupported sign width"))? as usize;
            let slice = values
                .get(offset..offset + width)
                .ok_or_else(|| bad("truncated signs"))?;
            let values = slice
                .iter()
                .map(|v| {
                    v.as_f64()
                        .filter(|s| *s == -1. || *s == 1.)
                        .map(|s| s as f32)
                        .ok_or_else(|| bad("sign must be -1 or +1"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if signs.insert(width, values).is_some() {
                return Err(bad("duplicate sign width"));
            }
            offset += width;
        }
        if offset != values.len() || signs.len() != 3 {
            return Err(bad("incomplete/trailing signs"));
        }
        Ok(Self { text, signs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Value, Value) {
        let mut config = crate::mlx::tests::config();
        for (k,v) in [
            ("schema_version",json!(2)),("model_type",json!("prism_hadamard_qwen35")),
            ("base_model_type",json!("qwen3_5")),("tensor_namespace",json!("mlx-vlm-qwen3_5")),
            ("gdn_activation_layout",json!("grouped")),("hadamard_config",json!("hadamard.json")),
            ("components",json!({"text":true,"vision":true,"mtp":false})),
            ("quantization",json!({"bits":2,"group_size":128,"mode":"affine"})),
            ("modules",json!(packed_modules().keys().map(|k| json!({"path":k,"block":1024,"embedding":k=="model.embed_tokens","dtype":"float16"})).collect::<Vec<_>>())),
            ("vision_config",json!({"depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
                "num_position_embeddings":2304,"out_hidden_size":5120,"patch_size":16,"spatial_merge_size":2,
                "temporal_patch_size":2,"in_channels":3,"deepstack_visual_indexes":[],"hidden_act":"gelu_pytorch_tanh"})),
        ] {config[k]=v;}
        config["text_config"]["mtp_num_hidden_layers"] = json!(0);
        let rotation = json!({
            "prism.hadamard.version":1,"prism.hadamard.block_size":1024,
            "prism.hadamard.transform":"normalized-sylvester-walsh-hadamard","prism.hadamard.axis":"input-last-dimension",
            "prism.hadamard.sign_mode":"explicit","prism.hadamard.gdn_v_grouped":true,
            "prism.hadamard.inverse_weight_names":["language_model.model.embed_tokens.weight"],
            "prism.hadamard.weight_names":packed_modules().keys().filter(|k|k.as_str()!="model.embed_tokens").map(|k|format!("language_model.{k}.weight")).collect::<Vec<_>>(),
            "prism.hadamard.sign_widths":[5120,6144,17408],"prism.hadamard.sign_values":vec![1.0;5120+6144+17408],
        });
        (config, rotation)
    }
    #[test]
    fn accepts_complete_contract_without_reinterpreting_bf16_qwen() {
        let (c, r) = fixture();
        let parsed = BonsaiConfig::parse(&c, &r).unwrap();
        assert_eq!(parsed.text.context, 262144);
        assert_eq!(packed_modules().len(), 402);
        assert_eq!(parsed.signs[&17408].len(), 17408);
        assert!(QwenConfig::parse(&c).is_err());
        assert!(BonsaiConfig::parse(&crate::mlx::tests::config(), &r).is_err());
    }
    #[test]
    fn rejects_ambiguous_or_incomplete_contracts() {
        let (c, r) = fixture();
        for (pointer, value) in [
            ("/schema_version", json!(1)),
            ("/gdn_activation_layout", json!("tiled")),
            ("/quantization/bits", json!(4)),
            ("/quantization/group_size", json!(64)),
            ("/modules/0/block", json!(0)),
            (
                "/modules/0/path",
                json!("model.layers.0.linear_attn.in_proj_a"),
            ),
            ("/modules/0/dtype", json!("bfloat16")),
            ("/text_config/mtp_num_hidden_layers", json!(1)),
            ("/vision_config/deepstack_visual_indexes", json!([8])),
            ("/hadamard_config", json!("../outside.json")),
        ] {
            let mut invalid = c.clone();
            *invalid.pointer_mut(pointer).unwrap() = value;
            assert!(BonsaiConfig::parse(&invalid, &r).is_err(), "{pointer}");
        }
        let mut invalid = c.clone();
        invalid["modules"].as_array_mut().unwrap().pop();
        assert!(BonsaiConfig::parse(&invalid, &r).is_err());
        let mut invalid = c.clone();
        invalid["modules"][1] = invalid["modules"][0].clone();
        assert!(BonsaiConfig::parse(&invalid, &r).is_err());
        for (key, value) in [
            ("version", json!(2)),
            ("block_size", json!(512)),
            ("axis", json!("output")),
            ("gdn_v_grouped", json!(false)),
            ("sign_widths", json!([5120, 5120, 17408])),
            ("inverse_weight_names", json!([])),
            ("weight_names", json!([])),
            ("unknown", json!(true)),
        ] {
            let mut invalid = r.clone();
            invalid[format!("prism.hadamard.{key}")] = value;
            assert!(BonsaiConfig::parse(&c, &invalid).is_err(), "{key}");
        }
        for value in [json!(0), json!(1.5), json!("1")] {
            let mut invalid = r.clone();
            invalid["prism.hadamard.sign_values"][0] = value;
            assert!(BonsaiConfig::parse(&c, &invalid).is_err());
        }
    }
}
