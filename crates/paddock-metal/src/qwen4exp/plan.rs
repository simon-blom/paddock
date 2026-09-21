use crate::{MetalError, device::Result, iquant};
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::{collections::HashSet, path::Path};

/// Read-only schema/memory planning for the elected three-shard IQ3 export.
/// No GPU device, upload, runner capability or serving qualification is created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashNextPlan {
    pub tensor_count: usize,
    /// Raw compressed backbone planes, without the PLE table. CUDA's registry
    /// number describes its repacked seats, not this native Metal payload.
    pub backbone_bytes: u64,
    /// Raw IQ4_NL table. Residency/mapping policy is still a graph decision;
    /// do not describe this as free simply because Apple has unified memory.
    pub ple_bytes: u64,
}

impl FlashNextPlan {
    pub fn inspect(path: &Path) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        Self::validate_map(&map)
    }

    pub(super) fn validate_map(map: &MappedGguf) -> Result<Self> {
        let g = map.gguf();
        if g.architecture() != Some("qwen4exp") {
            return Err(MetalError::Model(
                "Flash Next needs qwen4exp, not qwen35/qwen3next".into(),
            ));
        }
        for (key, value) in [
            ("block_count", 48),
            ("context_length", 262144),
            ("embedding_length", 2560),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
            ("expert_count", 512),
            ("expert_used_count", 10),
            ("expert_feed_forward_length", 640),
            ("expert_shared_feed_forward_length", 640),
            ("ssm.conv_kernel", 4),
            ("ssm.state_size", 128),
            ("ssm.group_count", 16),
            ("ssm.time_step_rank", 48),
            ("ssm.inner_size", 6144),
            ("full_attention_interval", 4),
            ("rope.dimension_count", 64),
            ("hyper_connection.count", 4),
            ("hyper_connection.low_rank", 320),
            ("attention.indexer.head_count", 4),
            ("attention.indexer.key_length", 128),
            ("attention.indexer.top_k", 2048),
            ("ple.ngram_size", 3),
            ("ple.heads_per_ngram", 8),
            ("ple.conv_kernel", 4),
            ("ple.eos_token_id", 248044),
            ("ple.image_token_id", 248056),
            ("embedding_length_per_layer_input", 160),
        ] {
            if g.arch_field(key).and_then(Value::as_u64) != Some(value) {
                return Err(MetalError::Model(format!("Flash Next: unsupported {key}")));
            }
        }
        for (key, value) in [
            ("rope.freq_base", 1e7),
            ("attention.layer_norm_rms_epsilon", 1e-6),
        ] {
            if g.arch_field(key).and_then(Value::as_f32) != Some(value) {
                return Err(MetalError::Model(format!("Flash Next: unsupported {key}")));
            }
        }
        let heads = [
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
        ];
        let mut offset = 0u64;
        let offsets = heads
            .iter()
            .map(|&n| {
                let start = offset;
                offset += n;
                start
            })
            .collect::<Vec<_>>();
        for (key, expected) in [
            ("rope.dimension_sections", vec![11, 11, 10, 0]),
            (
                "attention.compress_ratios",
                (0..48).map(|i| if i % 4 == 3 { 4 } else { 0 }).collect(),
            ),
            ("ple.layers", vec![1]),
            (
                "ple.layer_multipliers",
                vec![23703573157769, 20109073645365, 8052911324071],
            ),
            ("ple.head_offsets", offsets),
            ("ple.head_vocab_sizes", heads.to_vec()),
        ] {
            let values = match g.arch_field(key) {
                Some(Value::Array(v)) => v.iter().map(Value::as_u64).collect::<Option<Vec<_>>>(),
                _ => None,
            };
            if values.as_ref() != Some(&expected) {
                return Err(MetalError::Model(format!("Flash Next: unsupported {key}")));
            }
        }
        let mut names = HashSet::new();
        let mut backbone_bytes = 0u64;
        let mut ple_bytes = 0u64;
        let mut check = |name: String, dims: &[usize], types: &[u32]| -> Result<()> {
            let (t, bytes) = map
                .tensor_bytes(&name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            let elements = dims.iter().try_fold(1u64, |a, &b| a.checked_mul(b as u64));
            if t.dims != dims.iter().map(|&d| d as u64).collect::<Vec<_>>()
                || !types.contains(&t.raw_type)
                || elements.and_then(|n| t.ggml_type.byte_size(n)) != Some(bytes.len() as u64)
            {
                return Err(MetalError::Model(format!(
                    "Flash Next {name}: unsupported shape/type/bytes"
                )));
            }
            if iquant::is_iq(t.raw_type) {
                iquant::validate(t.raw_type, dims, bytes.len())?;
            }
            if t.raw_type == 0
                && !bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
            {
                return Err(MetalError::Model(format!(
                    "Flash Next {name}: nonfinite F32 parameter"
                )));
            }
            if name.ends_with(".ssm_a")
                && !bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b) < 0.0)
            {
                return Err(MetalError::Model(format!(
                    "Flash Next {name}: expected stored -exp(A_log)"
                )));
            }
            if name == "per_layer_token_embd.weight" {
                ple_bytes = bytes.len() as u64;
            } else {
                backbone_bytes += bytes.len() as u64;
            }
            names.insert(name);
            Ok(())
        };
        for name in ["token_embd.weight", "output.weight"] {
            check(name.into(), &[2560, 248320], &[14])?;
        }
        check("output_hc_norm.weight".into(), &[10240], &[0])?;
        check("output_hc_down.weight".into(), &[10240, 320], &[8])?;
        check("output_hc_up.weight".into(), &[320, 10240], &[8])?;
        check(
            "per_layer_token_embd.weight".into(),
            &[160, 320001536],
            &[20],
        )?;
        for li in 0..48 {
            let mut c = |name: &str, dims: &[usize], types: &[u32]| {
                check(format!("blk.{li}.{name}"), dims, types)
            };
            for p in ["hc_attn", "hc_ffn"] {
                c(&format!("{p}_norm.weight"), &[10240], &[0])?;
                c(&format!("{p}_down.weight"), &[10240, 320], &[8])?;
                c(&format!("{p}_up.weight"), &[320, 10240], &[8])?;
                c(&format!("{p}_inject.weight"), &[10240, 4], &[0])?;
            }
            c("ffn_gate_inp.weight", &[2560, 512], &[0])?;
            c("ffn_gate_inp_shexp.weight", &[2560], &[0])?;
            // The IQ3-labelled artifact actually has IQ2_S gate/up except
            // layer 2 (IQ3_S), and IQ4_NL down on every layer. Never infer
            // a tensor layout from the artifact's marketing quant label.
            for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
                c(name, &[2560, 640, 512], if li == 2 { &[21] } else { &[22] })?;
            }
            c("ffn_down_exps.weight", &[640, 2560, 512], &[20])?;
            for name in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight"] {
                c(name, &[2560, 640], &[8, 14])?;
            }
            c("ffn_down_shexp.weight", &[640, 2560], &[8])?;
            if li % 4 == 3 {
                for (name, dims) in [
                    ("attn_q.weight", [2560, 12288]),
                    ("attn_k.weight", [2560, 512]),
                    ("attn_v.weight", [2560, 512]),
                    ("attn_output.weight", [6144, 2560]),
                ] {
                    c(name, &dims, &[14])?;
                }
                for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    c(name, &[256], &[0])?;
                }
                c("indexer.q_proj.weight", &[2560, 512], &[30])?;
                c("indexer.k_proj.weight", &[2560, 128], &[30])?;
                for name in ["indexer.q_norm.weight", "indexer.k_norm.weight"] {
                    c(name, &[128], &[0])?;
                }
            } else {
                c("attn_qkv.weight", &[2560, 10240], &[8, 14])?;
                c("attn_gate.weight", &[2560, 6144], &[8, 14])?;
                for name in ["ssm_alpha.weight", "ssm_beta.weight"] {
                    c(name, &[2560, 48], &[0])?;
                }
                c("ssm_conv1d.weight", &[4, 10240], &[0])?;
                c("ssm_a", &[48], &[0])?;
                c("ssm_dt.bias", &[48], &[0])?;
                c("ssm_norm.weight", &[128], &[0])?;
                c("ssm_out.weight", &[6144, 2560], &[14])?;
            }
            if li == 1 {
                c("ple_key.weight", &[2560, 10240], &[8])?;
                c("ple_value.weight", &[2560, 2560], &[8])?;
                c("ple_conv1d.weight", &[4, 10240], &[0])?;
                for name in [
                    "ple_norm_key.weight",
                    "ple_norm_query.weight",
                    "ple_norm_conv.weight",
                ] {
                    c(name, &[10240], &[0])?;
                }
            }
        }
        if names.len() != 1224
            || map.tensor_count() != names.len()
            || map.tensor_infos().any(|t| !names.contains(&t.name))
        {
            return Err(MetalError::Model(
                "Flash Next: unexpected/missing tensor, vision/MTP exports need a separate audit"
                    .into(),
            ));
        }
        Ok(Self {
            tensor_count: names.len(),
            backbone_bytes,
            ple_bytes,
        })
    }

    /// Lower bound only: all raw weights/table resident, F32 DeltaNet states
    /// and F16 ordinary KV. This intentionally excludes QSA/PLE caches, scratch,
    /// driver and OS overhead and therefore must not be used as a fit approval.
    pub fn resident_lower_bound(&self, context: usize, batch: usize) -> Result<u64> {
        if context == 0 || context > 262144 || batch == 0 || batch > 64 {
            return Err(MetalError::Model("Flash Next memory planning bounds: context 1..=262144, batch 1..=64; not execution support".into()));
        }
        let state = 36 * (48 * 128 * 128 + 3 * 10240) * 4;
        let kv = 12 * 512 * 2 * 2 * context as u64;
        self.backbone_bytes
            .checked_add(self.ple_bytes)
            .and_then(|n| n.checked_add((state + kv) * batch as u64))
            .ok_or_else(|| MetalError::Memory("Flash Next planning byte count overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planning_is_a_lower_bound_not_a_fit_approval() {
        let p = FlashNextPlan {
            tensor_count: 1224,
            backbone_bytes: 53150661120,
            ple_bytes: 28800138240,
        };
        assert_eq!(p.resident_lower_bound(4096, 4).unwrap(), 82824132096);
        for (ctx, batch) in [(0, 4), (262145, 4), (4096, 0), (4096, 65)] {
            assert!(p.resident_lower_bound(ctx, batch).is_err());
        }
        let bad = FlashNextPlan {
            backbone_bytes: u64::MAX,
            ..p
        };
        assert!(bad.resident_lower_bound(1, 1).is_err());
    }
    #[test]
    #[ignore = "requires full SHA-verified PADDOCK_FLASH_NEXT_MODEL three-shard family"]
    fn elected_artifact_schema() {
        let path = std::env::var("PADDOCK_FLASH_NEXT_MODEL").unwrap();
        let p = FlashNextPlan::inspect(Path::new(&path)).unwrap();
        assert_eq!(
            p,
            FlashNextPlan {
                tensor_count: 1224,
                backbone_bytes: 53150661120,
                ple_bytes: 28800138240
            }
        );
        eprintln!(
            "Flash Next schema: {p:?}; c4/4096 resident lower bound={} (not fit approval)",
            p.resident_lower_bound(4096, 4).unwrap()
        );
    }
}
