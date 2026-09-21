//! Strict, read-only planning of the user-selected MLX checkpoint. Presence
//! in this plan is not registry qualification. Reject unknown text tensors,
//! quantization overrides and geometry before the first GPU allocation.
use super::affine::{self, A4G32, A8G64};
use crate::{
    device::{MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

pub(super) const ROOT: &str = "language_model.model";
pub(super) const PLE: &str = "language_model.model.layers.1.ple";
pub(super) const SHARD_ROWS: usize = 2_500_012;
pub(super) const FOLDED_NORM: u32 = 0x103;

#[derive(Debug, Clone)]
pub struct FlashNextMlxPlan {
    pub tensor_count: usize,
    pub checkpoint_text_bytes: u64,
    pub resident_weight_bytes: u64,
    pub ple_bytes: u64,
}

#[derive(Clone)]
struct Spec {
    shape: Vec<usize>,
    ty: u32,
}
pub(super) struct Source {
    pub(super) map: ShardedSafetensors,
    specs: BTreeMap<String, Spec>,
    pub(super) plan: FlashNextMlxPlan,
}

impl FlashNextMlxPlan {
    pub fn inspect(path: &Path) -> Result<Self> {
        Ok(Source::open(path)?.plan)
    }
}

fn config(v: &Value) -> Result<()> {
    for (key, expected) in [
        ("model_type", json!("qwen4_exp")),
        ("image_token_id", json!(248056)),
    ] {
        if v.get(key) != Some(&expected) {
            return Err(MetalError::Model(format!(
                "Flash Next MLX: unsupported {key}"
            )));
        }
    }
    let tc = &v["text_config"];
    let expected: Value = serde_json::from_str(
        r#"{
        "model_type":"qwen4_exp_text","dtype":"bfloat16","mamba_ssm_dtype":"float32",
        "hidden_size":2560,"num_hidden_layers":48,"vocab_size":248320,
        "max_position_embeddings":262144,"rms_norm_eps":1e-6,
        "num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        "full_attention_interval":4,"partial_rotary_factor":0.25,
        "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,
        "indexer_budget":2048,"indexer_compress_ratio":4,
        "linear_num_key_heads":16,"linear_num_value_heads":48,
        "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,
        "shared_expert_intermediate_size":640,"hc_count":4,"hc_lowrank":320,
        "ple_layer_ids":[2],"ple_embed_dim":2560,"ple_conv_kernel_size":4,
        "ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,
        "split_ngram_parts":128,"make_ngram_vocab_size_divisible_by":128,
        "tie_word_embeddings":false,"attention_bias":false,"hidden_act":"silu",
        "output_gate_type":"sigmoid","bos_token_id":248044,"eos_token_id":248044,
        "rope_parameters":{"rope_type":"default","rope_theta":10000000,
            "partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}
    }"#,
    )
    .expect("fixed Flash Next schema");
    for (key, value) in expected.as_object().expect("fixed schema is a JSON object") {
        if tc.get(key) != Some(value) {
            return Err(MetalError::Model(format!(
                "Flash Next MLX: unsupported text_config.{key}"
            )));
        }
    }
    let blocks: Vec<_> = (0..48)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    if tc["layer_types"] != json!(blocks) {
        return Err(MetalError::Model(
            "Flash Next MLX: unsupported layer_types".into(),
        ));
    }
    Ok(())
}

fn specs() -> BTreeMap<String, Spec> {
    let mut out = BTreeMap::new();
    let mut add = |name: String, shape: &[usize], ty| {
        out.insert(
            name,
            Spec {
                shape: shape.to_vec(),
                ty,
            },
        );
    };
    add(format!("{ROOT}.embed_tokens"), &[248320, 2560], A4G32);
    add("language_model.lm_head".into(), &[248320, 2560], A4G32);
    let mut hc = |name: String, inject: bool| {
        add(format!("{name}.hc_norm.weight"), &[10240], 0);
        add(
            format!("{name}.input_mix_weight_down"),
            &[320, 10240],
            A4G32,
        );
        add(format!("{name}.input_mix_weight_up"), &[10240, 320], A4G32);
        if inject {
            add(format!("{name}.block_inject_weight"), &[4, 10240], A4G32);
        }
    };
    hc(format!("{ROOT}.hyper_connection_mixer"), false);
    for li in 0..48 {
        for domain in ["attn", "mlp"] {
            hc(
                format!("{ROOT}.layers.{li}.{domain}_hyper_connection"),
                true,
            );
        }
    }
    for li in 0..48 {
        let root = format!("{ROOT}.layers.{li}");
        add(format!("{root}.mlp.gate"), &[512, 2560], A8G64);
        add(format!("{root}.mlp.shared_expert_gate"), &[1, 2560], A8G64);
        for (suffix, shape) in [
            ("gate_proj", [640, 2560]),
            ("up_proj", [640, 2560]),
            ("down_proj", [2560, 640]),
        ] {
            add(format!("{root}.mlp.shared_expert.{suffix}"), &shape, A4G32);
            add(
                format!("{root}.mlp.switch_mlp.{suffix}"),
                &[512, shape[0], shape[1]],
                A4G32,
            );
        }
        if li % 4 == 3 {
            for (suffix, shape) in [
                ("q_proj", [12288, 2560]),
                ("k_proj", [512, 2560]),
                ("v_proj", [512, 2560]),
                ("o_proj", [2560, 6144]),
                ("indexer.index_qk_proj", [640, 2560]),
            ] {
                add(format!("{root}.self_attn.{suffix}"), &shape, A4G32);
            }
            for (suffix, size) in [
                ("q_norm", 256),
                ("k_norm", 256),
                ("indexer.q_layernorm", 128),
                ("indexer.k_layernorm", 128),
            ] {
                add(format!("{root}.self_attn.{suffix}.weight"), &[size], 0);
            }
        } else {
            for (suffix, shape) in [
                ("in_proj_qkv", [10240, 2560]),
                ("in_proj_z", [6144, 2560]),
                ("in_proj_a", [48, 2560]),
                ("in_proj_b", [48, 2560]),
                ("out_proj", [2560, 6144]),
            ] {
                add(format!("{root}.linear_attn.{suffix}"), &shape, A4G32);
            }
            for (suffix, shape) in [
                ("conv1d.weight", vec![10240, 4, 1]),
                ("A_log", vec![48]),
                ("dt_bias", vec![48]),
                ("norm.weight", vec![128]),
            ] {
                add(format!("{root}.linear_attn.{suffix}"), &shape, 0);
            }
        }
    }
    for suffix in ["norm_key", "norm_query", "norm_conv"] {
        add(format!("{PLE}.{suffix}.weight"), &[10240], 0);
    }
    add(format!("{PLE}.key_proj"), &[10240, 2560], A4G32);
    add(format!("{PLE}.value_proj"), &[2560, 2560], A4G32);
    add(format!("{PLE}.conv1d.weight"), &[10240, 4, 1], 0);
    for shard in 0..128 {
        add(
            format!("{PLE}.ple_embedding.ngram_embedding.shards.{shard}"),
            &[SHARD_ROWS, 160],
            A4G32,
        );
    }
    out
}

impl Source {
    pub(super) fn open(path: &Path) -> Result<Self> {
        let cfg: Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json"))
                .map_err(|e| MetalError::Model(e.to_string()))?,
        )
        .map_err(|e| MetalError::Model(e.to_string()))?;
        config(&cfg)?;
        let map =
            ShardedSafetensors::open_dir(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let specs = specs();
        let quant = cfg["quantization"]
            .as_object()
            .ok_or_else(|| MetalError::Model("missing MLX quantization".into()))?;
        for (key, value) in quant {
            let expected = match key.as_str() {
                "bits" => json!(4),
                "group_size" => json!(32),
                "mode" => json!("affine"),
                _ => {
                    let spec = specs
                        .get(key)
                        .filter(|s| affine::is_affine(s.ty))
                        .ok_or_else(|| {
                            MetalError::Model(format!(
                                "unsupported MLX quantization override {key}"
                            ))
                        })?;
                    let (bits, group) = affine::format(spec.ty);
                    json!({"bits":bits,"group_size":group,"mode":"affine"})
                }
            };
            if value != &expected {
                return Err(MetalError::Model(format!(
                    "unsupported MLX quantization {key}"
                )));
            }
        }
        if quant.get("bits") != Some(&json!(4))
            || quant.get("group_size") != Some(&json!(32))
            || quant.get("mode") != Some(&json!("affine"))
        {
            return Err(MetalError::Model(
                "Flash Next MLX requires affine4/group32 default".into(),
            ));
        }
        let mut names = HashSet::new();
        let mut raw = 0u64;
        let mut resident = 0u64;
        let mut ple = 0u64;
        for (name, spec) in &specs {
            let size = if affine::is_affine(spec.ty) {
                if spec.ty == A8G64 && !quant.contains_key(name) {
                    return Err(MetalError::Model(format!(
                        "missing affine8 override {name}"
                    )));
                }
                let parts = affine::parts(&map, name, &spec.shape, spec.ty)?;
                for suffix in ["weight", "scales", "biases"] {
                    names.insert(format!("{name}.{suffix}"));
                }
                parts.iter().map(|b| b.len() as u64).sum::<u64>()
            } else {
                let (info, bytes) = map
                    .bytes(name)
                    .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
                if info.dtype != StDtype::Bf16
                    || info.shape != spec.shape
                    || bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .any(|v| u16::from_le_bytes(*v) & 0x7f80 == 0x7f80)
                {
                    return Err(MetalError::Model(format!("invalid BF16 parameter {name}")));
                }
                names.insert(name.clone());
                resident += bytes.len() as u64; // small parameters expand to F32 on the GPU
                bytes.len() as u64
            };
            raw += size;
            resident += size;
            if name.contains("ngram_embedding.shards.") {
                ple += size;
            }
        }
        // These are metadata, not CPU inference. GPU hashing uses the
        // validated constants already shared with the native GGUF path.
        let sizes = vec![
            20000003i64,
            20000023,
            20000033,
            20000047,
            20000059,
            20000063,
            20000069,
            20000077,
            20000081,
            20000093,
            20000107,
            20000147,
            20000153,
            20000159,
            20000161,
            20000171,
        ];
        let mut off = 0;
        let offsets = sizes
            .iter()
            .map(|&n| {
                let old = off;
                off += n;
                old
            })
            .collect::<Vec<_>>();
        for (suffix, expected) in [
            (
                "layer_multipliers",
                vec![23703573157769, 20109073645365, 8052911324071],
            ),
            ("ngram_heads_offsets", offsets),
            ("ngram_heads_vocab_sizes", sizes),
        ] {
            let name = format!("{PLE}.ple_embedding.{suffix}");
            let (info, bytes) = map
                .bytes(&name)
                .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
            let actual = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect::<Vec<_>>();
            if info.dtype != StDtype::I64 || info.shape != [expected.len()] || actual != expected {
                return Err(MetalError::Model(format!("unsupported PLE hash {name}")));
            }
            raw += bytes.len() as u64;
            names.insert(name);
        }
        for name in map.names() {
            if name.starts_with("language_model.") && !names.contains(name) {
                return Err(MetalError::Model(format!(
                    "unconsumed Flash Next text tensor {name}"
                )));
            }
        }
        let plan = FlashNextMlxPlan {
            tensor_count: names.len(),
            checkpoint_text_bytes: raw,
            resident_weight_bytes: resident,
            ple_bytes: ple,
        };
        Ok(Self { map, specs, plan })
    }

    pub(super) fn weight(&self, d: &MetalDevice, name: &str) -> Result<Weight> {
        let spec = self
            .specs
            .get(name)
            .ok_or_else(|| MetalError::Model(format!("unplanned MLX tensor {name}")))?;
        if affine::is_affine(spec.ty) {
            return affine::load(d, &self.map, name, &spec.shape, spec.ty);
        }
        let bytes = self
            .map
            .bytes(name)
            .expect("planned tensor was validated in the immutable checkpoint")
            .1;
        let input = d.upload_parts(&[bytes])?;
        let count = bytes.len() / 2;
        let buffer = d.alloc(count * 4)?;
        // Gated DeltaNet norm is conventional gamma. All other named norms
        // in this model are zero-centered; fold 1+w exactly once in F32.
        let norm = name.ends_with(".weight")
            && (name.contains(".hc_norm.")
                || name.contains(".norm_")
                || name.contains("_norm.")
                || name.contains("_layernorm."));
        let mode = if name.ends_with(".A_log") {
            1
        } else if norm {
            2
        } else {
            0
        };
        let cmd = d.begin()?;
        cmd.dispatch(
            "q4a_small",
            &[&input, &buffer],
            &[count as u32, mode],
            [count.div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        Ok(Weight {
            buffer,
            ty: if mode == 2 { FOLDED_NORM } else { 0 },
            k: *spec.shape.last().expect("planned tensor shape is nonempty"),
            n: 1,
        })
    }

    pub(super) fn table(&self, d: &MetalDevice) -> Result<Weight> {
        let mut parts = Vec::with_capacity(384);
        for shard in 0..128 {
            let base = format!("{PLE}.ple_embedding.ngram_embedding.shards.{shard}");
            let slices = affine::parts(&self.map, &base, &[SHARD_ROWS, 160], A4G32)?;
            for (slice, suffix) in slices.iter().zip(["weight", "scales", "biases"]) {
                parts.push((format!("{base}.{suffix}"), slice.len()));
            }
        }
        // One final allocation, retaining each compressed shard's exact
        // layout. Positional reads avoid touching a second 32 GB mmap view.
        let buffer = d.upload_with(parts.iter().map(|p| p.1).sum(), |out| {
            let mut offset = 0;
            for (name, size) in &parts {
                self.map
                    .read_into(name, &mut out[offset..offset + size])
                    .map_err(|e| MetalError::Model(e.to_string()))?;
                offset += size;
            }
            Ok(())
        })?;
        Ok(Weight {
            buffer,
            ty: A4G32,
            k: 160,
            n: SHARD_ROWS * 128,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "PADDOCK_FLASH_NEXT_MLX_MODEL: headers and small metadata only"]
    fn flash_next_mlx_checkpoint_inventory() {
        let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
        let p = FlashNextMlxPlan::inspect(Path::new(&path)).unwrap();
        eprintln!("{p:?}");
        assert_eq!(p.checkpoint_text_bytes, 110621031960);
        assert_eq!(p.ple_bytes, 32000153600);
        assert!(p.resident_weight_bytes > p.checkpoint_text_bytes);
    }
}
