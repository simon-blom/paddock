//! MiniCPM5-2B TP serving shim — the proof of the generic TP stack.
//!
//! This file is deliberately SMALL. The architecture claim under test: a
//! second conventional model adds TP by declaring names/geometry/policy and
//! binding the generic machinery, without a second scheduler, cache protocol,
//! worker runtime, FFN, or GQA implementation. Everything below rides:
//!
//! - `tp::serve` (`ServeModel`, coordinator/worker/generator, host tests)
//! - `tp::conventional::ConventionalGqaRank` (Q/K/V + output sharding, paged
//!   mirrored KV, model-owned split/prefill policy)
//! - `tp::ffn::SwiGluTpRank` (gate/up/down sharding, live-prefix reduction)
//! - `tp::traversal` stage primitives, `tp::cache` mirrored KV lifecycle
//!
//! MiniCPM5-2B specifics (all policy, not machinery): llama tensor names,
//! no per-head Q/K norms, no sinks, NORM-convention YARN RoPE, the four
//! granite multipliers at identity (llama arch = granite at defaults).

use std::path::Path;
use std::sync::Arc;

use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuExecutor, KvDtype, QuantW};
use crate::gpu::distributed::NcclCommunicator;
use crate::tp::attention::AttentionWeightNames;
use crate::tp::conventional::{ConventionalAttentionSpec, ConventionalGqaRank, DecodeSplitPolicy};
use crate::tp::ffn::{SwiGluTpRank, SwiGluWeightNames};
use crate::tp::TpTopology;

/// The checkpoint identity this TP lane accepts (SHA-256 of the single-file
/// GGUF; verified against HF's LFS manifest at download time).
pub const MINICPM5_2B_Q8_0_SHA256: &str =
    "c5415f8989bf88a8288f1b55a3cc371af53c07b0faa220a63bd7a990cfaba078";

/// MiniCPM5-2B's serving geometry and policy, read from the checkpoint's
/// metadata (never hard-coded beyond identity): the plain-llama graph.
///
/// Host-testable: every field derives from GGUF metadata keys.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiniCpmTpSpec {
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub eps: f32,
    /// NORM-convention rope params (`rope_yarn_batch_norm`).
    pub rope: (f32, f32, f32, f32, f32, f32),
    /// The four llama-identity multipliers (documented policy; applied by the
    /// shim's spine if ever non-identity — MiniCPM ships all at identity).
    pub embedding_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
}

impl MiniCpmTpSpec {
    /// Read the spec from GGUF metadata. Refuses by name anything the llama
    /// graph does not serve (the granite loader's gate, mirrored).
    pub fn from_metadata(map: &MappedGguf) -> Result<Self, String> {
        use paddock_models::gguf::Value;
        let arch = map.gguf().architecture().unwrap_or("");
        if arch != "llama" {
            return Err(format!(
                "minicpm-tp: expected a llama-architecture checkpoint, got {arch:?}"
            ));
        }
        let u = |key: &str| -> Result<usize, String> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("minicpm-tp: missing or invalid {key}"))
        };
        let f = |key: &str| -> Result<f32, String> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .ok_or_else(|| format!("minicpm-tp: missing or invalid {key}"))
        };
        // The llama gate: granite scalars must NOT be stamped (they would be
        // silently dropped by a graph that does not apply them).
        for k in [
            "embedding_scale",
            "residual_scale",
            "logit_scale",
            "attention.scale",
        ] {
            if map.gguf().arch_field(k).is_some() {
                return Err(format!(
                    "minicpm-tp: llama file stamps llama.{k}, which no llama graph applies"
                ));
            }
        }
        let head_dim = u("attention.key_length")?;
        if head_dim != u("attention.value_length")? || head_dim != u("rope.dimension_count")? {
            return Err("minicpm-tp: key/value/rope lengths disagree".into());
        }
        let n_rot = head_dim;
        let base = f("rope.freq_base")?;
        let eps = f("attention.layer_norm_rms_epsilon")?;
        if !eps.is_finite() || eps <= 0.0 || !base.is_finite() || base <= 0.0 {
            return Err("minicpm-tp: invalid rope or norm metadata".into());
        }
        // No rope scaling: llama.scale_* presence is a refuse, not a default.
        for k in ["rope.freq_scale", "rope.scaling.factor"] {
            if map.gguf().arch_field(k).is_some() {
                return Err(format!("minicpm-tp: rope scaling ({k}) not served"));
            }
        }
        let ctx_train = u("context_length").unwrap_or(0);
        let rope = paddock_kernels::reference::ops::YarnRope::new(
            n_rot, base, 1.0, ctx_train, 0.0, 1.0, 32.0, 1.0,
        )
        .kernel_params();
        Ok(Self {
            width: u("embedding_length")?,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim,
            ff: u("feed_forward_length")?,
            eps,
            rope,
            // llama-identity multipliers, explicit in the spec.
            embedding_scale: 1.0,
            residual_scale: 1.0,
            logit_scale: 1.0,
        })
    }
}

/// One MiniCPM layer's TP rank state: the two generic ranks plus the two
/// layer norms (llama's pre/post residual norms). All weight loading goes
/// through the generic components' loaders — no weight-name duplication
/// beyond the llama name FORMULA this shim declares.
///
/// Pending GPU-spine seam: the fields read when the `ServeModel` impl wires
/// the decode/span/pipe forwards (the next slice on this branch; inference
/// validation is the user's GPU gate).
#[allow(dead_code)] // spine fields; consumed by the ServeModel impl slice
pub(crate) struct MiniCpmTpLayer {
    pub(crate) gqa: ConventionalGqaRank,
    pub(crate) ffn: SwiGluTpRank,
    pub(crate) attn_norm: cudarc::driver::CudaSlice<f32>,
    pub(crate) post_norm: cudarc::driver::CudaSlice<f32>,
}

/// The MiniCPM5-2B TP rank: conventional GQA + SwiGLU over the generic
/// machinery, composed per layer from `tp::conventional` and `tp::ffn`.
/// Embedding, final norm and LM head are replicated (identical on every
/// rank; only attention/FFN shards differ).
///
/// Pending GPU-spine seam: same as [`MiniCpmTpLayer`].
#[allow(dead_code)] // spine fields; consumed by the ServeModel impl slice
pub struct MiniCpmTpRank {
    pub spec: MiniCpmTpSpec,
    exec: Arc<GpuExecutor>,
    tok_embd: QuantW,
    out_norm: cudarc::driver::CudaSlice<f32>,
    lm_head: QuantW,
    layers: Vec<MiniCpmTpLayer>,
}

impl MiniCpmTpRank {
    /// Geometry gate against a rank's topology: complete groups, even KV
    /// split — the same refusal rule Qwen's load applies, now from the
    /// generic layer.
    pub fn validate_geometry(topology: crate::tp::TpTopology) -> Result<(), String> {
        let spec = Self::reference_spec();
        crate::tp::conventional::validate_load_geometry(
            topology,
            spec.width,
            spec.heads,
            spec.kv_heads,
            spec.head_dim,
        )
        .map_err(|e| e.to_string())
    }

    /// The checkpoint's serving geometry (static — it is a property of the
    /// pinned checkpoint, re-checked against metadata at load).
    pub fn reference_spec() -> MiniCpmTpSpec {
        MiniCpmTpSpec {
            width: 2048,
            heads: 16,
            kv_heads: 2,
            head_dim: 128,
            ff: 6144,
            eps: 1e-6,
            rope: (1.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            embedding_scale: 1.0,
            residual_scale: 1.0,
            logit_scale: 1.0,
        }
    }

    /// Load the whole rank from the mapped GGUF. The only weight knowledge
    /// in this file is the llama name formula:
    /// `blk.{i}.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down}.weight`,
    /// `blk.{i}.{attn_norm,ffn_norm}.weight`, `token_embd.weight`,
    /// `output.weight`, `model.norm.weight`.
    #[allow(dead_code)] // spine builder; consumed by the ServeModel impl slice
    fn load_layers(
        exec: &GpuExecutor,
        map: &MappedGguf,
        spec: &MiniCpmTpSpec,
        _topology: TpTopology,
        group: &NcclCommunicator,
        n_layers: usize,
        max_ctx: usize,
        slots: usize,
        kv_dtype: KvDtype,
    ) -> Result<Vec<MiniCpmTpLayer>, String> {
        let name = |i: usize, part: &str| format!("blk.{i}.{part}");
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let gqa = ConventionalGqaRank::load(
                exec,
                map,
                ConventionalAttentionSpec {
                    names: AttentionWeightNames {
                        q: &name(i, "attn_q.weight"),
                        k: &name(i, "attn_k.weight"),
                        v: &name(i, "attn_v.weight"),
                        output: &name(i, "attn_output.weight"),
                    },
                    // llama carries NO per-head Q/K norms — the generic
                    // hooks default to the exact identity.
                    q_norm: None,
                    k_norm: None,
                    // llama has no attention sinks.
                    sinks: None,
                    eps: spec.eps,
                    // RoPE is applied by the shim's spine (NORM convention,
                    // one rope call per plane per decode/span stage).
                    rope: None,
                    prefill_policy: crate::tp::attention::PagedPrefillPolicy {
                        // hd 128 rides the tiled prefill arm.
                        f16: false,
                        tiled: true,
                    },
                },
                spec.heads,
                spec.kv_heads,
                spec.head_dim,
                group,
                max_ctx,
                slots,
                // rank-local KV pool blocks: ctx/slots page granularity
                (max_ctx.div_ceil(crate::kv_pool::BLOCK_TOKENS) * slots) as u32,
                kv_dtype,
                // Fixed split policy: hd 128, group 8 — one pass suffices at
                // MiniCPM's small per-rank head count; the model owns the
                // policy choice (measured on-device later).
                DecodeSplitPolicy::SinglePass,
            )
            .map_err(|e| e.to_string())?;
            let ffn = SwiGluTpRank::load(
                exec,
                map,
                SwiGluWeightNames {
                    gate: &name(i, "ffn_gate.weight"),
                    up: &name(i, "ffn_up.weight"),
                    down: &name(i, "ffn_down.weight"),
                },
                group,
            )
            .map_err(|e| e.to_string())?;
            let upload = |part: &str| -> Result<cudarc::driver::CudaSlice<f32>, String> {
                Ok(exec.upload(map, &name(i, part)).map_err(|e| e.to_string())?.buf)
            };
            layers.push(MiniCpmTpLayer {
                gqa,
                ffn,
                attn_norm: upload("attn_norm.weight")?,
                post_norm: upload("ffn_norm.weight")?,
            });
        }
        Ok(layers)
    }
}


/// Host check entry: read the spec from a checkpoint without a device.
pub fn load_spec_for_host_check(map_path: &Path) -> Result<MiniCpmTpSpec, String> {
    let map = MappedGguf::open(map_path).map_err(|e| e.to_string())?;
    MiniCpmTpSpec::from_metadata(&map)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real checkpoint's metadata reads into the expected spec. Skipped
    /// when the model volume is absent (CI hosts do not mount it).
    #[test]
    fn minicpm_metadata_reads_expected_spec() {
        let path = match std::path::Path::new("/home/sime/models/minicpm5-2b/MiniCPM5-2B-Q8_0.gguf")
            .canonicalize()
        {
            Ok(p) if p.exists() => p,
            _ => return, // model volume not mounted
        };
        let map = MappedGguf::open(&path).unwrap();
        let spec = MiniCpmTpSpec::from_metadata(&map).unwrap();
        assert_eq!(spec.width, 2048);
        assert_eq!(spec.heads, 16);
        assert_eq!(spec.kv_heads, 2);
        assert_eq!(spec.head_dim, 128);
        assert_eq!(spec.ff, 6144);
        assert_eq!(spec.head_dim * spec.kv_heads, 256);
        // llama-identity multipliers
        assert_eq!(spec.embedding_scale, 1.0);
        assert_eq!(spec.residual_scale, 1.0);
        assert_eq!(spec.logit_scale, 1.0);
        // NORM rope params carried through (theta_scale = base^-1/rot)
        assert!(spec.rope.0.is_finite());
    }

    /// The complete-group partition: TP=2 = 8 Q heads / 1 KV head per rank;
    /// TP=3 refused (2 KV heads do not split over 3 ranks); TP=4 refused.
    #[test]
    fn minicpm_geometry_partitions_like_the_generic_rule() {
        use crate::tp::attention::GqaPartition;
        use crate::tp::TpTopology;
        for (rank, world) in [(0usize, 2usize), (1, 2)] {
            let t = TpTopology::new(rank, world).unwrap();
            let p = GqaPartition::new(t, 16, 2).unwrap();
            assert_eq!(p.local_heads, 8);
            assert_eq!(p.local_kv_heads, 1);
            assert_eq!(p.kv_start, rank);
        }
        let tp3 = TpTopology::new(0, 3).unwrap();
        assert!(GqaPartition::new(tp3, 16, 2).is_err());
        let tp4 = TpTopology::new(0, 4).unwrap();
        assert!(GqaPartition::new(tp4, 16, 2).is_err());
    }

    /// Q8_0 is the TP lane's proven weight class (Qwen3.8 TP serves Q8_0);
    /// its block layout must be shardable (row superblocks along in_dim).
    #[test]
    fn q8_0_block_layout_shardable() {
        assert!(paddock_models::ggml_type::GgmlType::Q8_0
            .block_layout()
            .is_some());
    }
}
