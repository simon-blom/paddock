use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::{collections::HashSet, path::Path};

impl Laguna {
    /// Strict checkpoint geometry, tensor and memory checks precede uploads.
    /// XS Q4_K_M and S UD-Q4_K_XL only. DFlash is a separate graph/gate.
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let source_versions = crate::offload::versions(path)?;
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        if map.gguf().architecture() != Some("laguna") {
            return Err(MetalError::Model(
                "Metal Laguna requires architecture=laguna".into(),
            ));
        }
        let u = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| MetalError::Model(format!("missing/invalid laguna.{key}")))
        };
        let f = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .filter(|n| n.is_finite())
                .ok_or_else(|| MetalError::Model(format!("missing/invalid laguna.{key}")))
        };
        let geometry = Geometry::for_width(u("embedding_length")?).ok_or_else(|| {
            MetalError::Model("Metal Laguna requires elected XS or S geometry".into())
        })?;
        let Geometry {
            width,
            layers: layer_count,
            ff,
            active,
            sliding_heads,
            dense_ff,
        } = geometry;
        for (key, expected) in [
            ("embedding_length", width),
            ("block_count", layer_count),
            ("attention.head_count_kv", 8),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("attention.sliding_window", 512),
            ("expert_count", EXPERTS),
            ("expert_used_count", active),
            ("expert_feed_forward_length", ff),
            ("expert_shared_feed_forward_length", ff),
            ("feed_forward_length", dense_ff),
            ("leading_dense_block_count", 1),
            ("expert_gating_func", 2),
            ("rope.dimension_count", 64),
            ("rope.dimension_count_swa", 128),
            ("rope.scaling.original_context_length", 8192),
            ("vocab_size", VOCAB),
        ] {
            if u(key)? != expected {
                return Err(MetalError::Model(format!(
                    "unsupported Laguna {key}; requires elected XS Q4_K_M or S UD-Q4_K_XL"
                )));
            }
        }
        let heads = match map.gguf().arch_field("attention.head_count") {
            Some(Value::Array(a)) => a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>(),
            _ => None,
        };
        if heads != Some((0..layer_count).map(|i| geometry.heads(i) as u64).collect())
            || !matches!(
                map.gguf().arch_field("expert_weights_norm"),
                Some(Value::Bool(true))
            )
            || map
                .gguf()
                .arch_field("rope.scaling.type")
                .and_then(Value::as_str)
                != Some("yarn")
        {
            return Err(MetalError::Model(
                "unsupported Laguna head/router/rotary metadata".into(),
            ));
        }
        for (key, expected) in [
            ("attention.layer_norm_rms_epsilon", 1e-6),
            ("expert_weights_scale", 2.5),
            ("rope.freq_base", 500000.),
            ("rope.freq_base_swa", 10000.),
            ("rope.scaling.factor", 32.),
            ("rope.scaling.yarn_attn_factor", 1.),
            ("rope.scaling.yarn_beta_fast", geometry.beta_fast()),
            ("rope.scaling.yarn_beta_slow", 1.),
        ] {
            if f(key)? != expected {
                return Err(MetalError::Model(format!("unsupported Laguna {key}")));
            }
        }
        let trained = u("context_length")?;
        if context == 0 || context > trained.min(32768) || max_batch == 0 || max_batch > 64 {
            return Err(MetalError::Model(
                "Metal Laguna context must be 1..=32768 within trained context, batch 1..=64"
                    .into(),
            ));
        }
        // Scalar metadata setup only. The GGUF factor=1 stamp is corrected
        // by YaRN's 1+0.1*ln(factor). The factor comes from this GGUF, not
        // a newer remote config with different context/rotary settings.
        let corr = |rotations: f32| {
            32. * (8192.0 / (rotations * 2. * std::f32::consts::PI)).ln() / 500000.0f32.ln()
        };
        let rope = [
            500000.,
            1. / 32.,
            corr(geometry.beta_fast()).floor().max(0.),
            corr(1.).ceil().min(63.),
            1. + 0.1 * 32.0f32.ln(),
        ]
        .map(f32::to_bits);
        // The elected S export deliberately retains factor=32 (256K context),
        // unlike the current HF config. Never silently substitute HF RoPE.
        let dense_types: &[u32] = if width == 2048 { &[12, 14] } else { &[8] };
        let expert_types: &[u32] = if width == 2048 {
            &[12, 14]
        } else {
            &[12, 13, 14]
        };
        let mut expected = HashSet::new();
        let mut payload = 0u64;
        let mut check = |name: String, dims: &[usize], types: &[u32]| -> Result<()> {
            let (t, bytes) = map
                .tensor_bytes(&name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            let elements = dims.iter().try_fold(1u64, |n, &d| n.checked_mul(d as u64));
            if t.dims.iter().map(|&d| d as usize).collect::<Vec<_>>() != dims
                || !types.contains(&t.raw_type)
                || elements.and_then(|n| t.ggml_type.byte_size(n)) != Some(bytes.len() as u64)
            {
                return Err(MetalError::Model(format!(
                    "{name}: unsupported shape/type/bytes {:?} {:?}",
                    t.dims, t.ggml_type
                )));
            }
            payload = payload
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| MetalError::Memory("Laguna payload overflow".into()))?;
            expected.insert(name);
            Ok(())
        };
        check(
            "token_embd.weight".into(),
            &[width, VOCAB],
            if width == 2048 { &[12] } else { &[8] },
        )?;
        check(
            "output.weight".into(),
            &[width, VOCAB],
            if width == 2048 { &[14] } else { &[8] },
        )?;
        check("output_norm.weight".into(), &[width], &[0])?;
        for i in 0..layer_count {
            let h = geometry.heads(i);
            for (name, dims, types) in [
                ("attn_norm.weight", vec![width], vec![0]),
                ("ffn_norm.weight", vec![width], vec![0]),
                ("attn_q_norm.weight", vec![128], vec![0]),
                ("attn_k_norm.weight", vec![128], vec![0]),
                ("attn_q.weight", vec![width, h * 128], dense_types.to_vec()),
                ("attn_k.weight", vec![width, KVWIDTH], dense_types.to_vec()),
                ("attn_v.weight", vec![width, KVWIDTH], dense_types.to_vec()),
                ("attn_gate.weight", vec![width, h], dense_types.to_vec()),
                (
                    "attn_output.weight",
                    vec![h * 128, width],
                    dense_types.to_vec(),
                ),
            ] {
                check(format!("blk.{i}.{name}"), &dims, &types)?;
            }
            let shared_ff = if i == 0 { dense_ff } else { ff };
            let suffix = if i == 0 { "" } else { "_shexp" };
            for (name, dims) in [
                ("gate", vec![width, shared_ff]),
                ("up", vec![width, shared_ff]),
                ("down", vec![shared_ff, width]),
            ] {
                check(
                    format!("blk.{i}.ffn_{name}{suffix}.weight"),
                    &dims,
                    dense_types,
                )?;
            }
            if i > 0 {
                check(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    &[width, EXPERTS],
                    &[0],
                )?;
                check(format!("blk.{i}.exp_probs_b.bias"), &[EXPERTS], &[0])?;
                for (name, dims) in [
                    ("gate", vec![width, ff, EXPERTS]),
                    ("up", vec![width, ff, EXPERTS]),
                    ("down", vec![ff, width, EXPERTS]),
                ] {
                    check(
                        format!("blk.{i}.ffn_{name}_exps.weight"),
                        &dims,
                        expert_types,
                    )?;
                }
            }
        }
        if let Some(t) = map.tensor_infos().find(|t| !expected.contains(&t.name)) {
            return Err(MetalError::Model(format!(
                "unexpected Laguna tensor {}",
                t.name
            )));
        }
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let blocks = page_stride * (max_batch + 1);
        // Full paged storage on all layers is an explicit interim storage
        // choice. It preserves radix reuse without a sliding-ring snapshot
        // copy, but it must not inherit CUDA's much smaller window estimate.
        let kv_layer = blocks * BLOCK_TOKENS * KVWIDTH * 2;
        let kv_bytes = (kv_layer * 2 * layer_count) as u64;
        let qwidth = sliding_heads * 128;
        let tile_cap = (CHUNK * active).div_ceil(16) + EXPERTS;
        let sizes = [
            CHUNK * 4,
            CHUNK * 8,
            max_batch * page_stride * 4,
            CHUNK * 4,
            CHUNK * 4,
            CHUNK * 8,
            CHUNK * width * 4,
            CHUNK * width * 4,
            CHUNK * qwidth * 4,
            CHUNK * KVWIDTH * 4,
            CHUNK * KVWIDTH * 4,
            CHUNK * sliding_heads * 4,
            CHUNK * qwidth * 4,
            CHUNK * sliding_heads * SPLITS * 130 * 4,
            CHUNK * width * 4,
            CHUNK * dense_ff * 4,
            CHUNK * dense_ff * 4,
            CHUNK * EXPERTS * 4,
            CHUNK * active * 4,
            CHUNK * active * 4,
            EXPERTS * CHUNK * active * 4,
            EXPERTS * 4,
            (1 + 2 * tile_cap) * 4,
            CHUNK * active * ff * 2 * 4,
            CHUNK * active * width * 4,
            max_batch * VOCAB * 4,
        ];
        let device = MetalDevice::new(budget)?;
        let required = payload
            .saturating_add(kv_bytes)
            .saturating_add(sizes.iter().sum::<usize>() as u64);
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Laguna weights/full paged KV/scratch require {required} bytes; grant {}",
                device.budget_bytes()
            )));
        }
        let w = |name: &str, dims: &[usize]| Weight::load(&device, &map, name, dims);
        let embedding = w("token_embd.weight", &[width, VOCAB])?;
        let head = w("output.weight", &[width, VOCAB])?;
        let output_norm = w("output_norm.weight", &[width])?;
        let mut layers = Vec::new();
        for i in 0..layer_count {
            let heads = geometry.heads(i);
            let w = |name: &str, dims: &[usize]| {
                Weight::load(&device, &map, &format!("blk.{i}.{name}"), dims)
            };
            let suffix = if i == 0 { "" } else { "_shexp" };
            let shared_ff = if i == 0 { dense_ff } else { ff };
            layers.push(Layer {
                heads,
                norm: w("attn_norm.weight", &[width])?,
                post: w("ffn_norm.weight", &[width])?,
                q: w("attn_q.weight", &[width, heads * 128])?,
                k: w("attn_k.weight", &[width, KVWIDTH])?,
                v: w("attn_v.weight", &[width, KVWIDTH])?,
                gate: w("attn_gate.weight", &[width, heads])?,
                qnorm: w("attn_q_norm.weight", &[128])?,
                knorm: w("attn_k_norm.weight", &[128])?,
                o: w("attn_output.weight", &[heads * 128, width])?,
                fg: w(&format!("ffn_gate{suffix}.weight"), &[width, shared_ff])?,
                fu: w(&format!("ffn_up{suffix}.weight"), &[width, shared_ff])?,
                fd: w(&format!("ffn_down{suffix}.weight"), &[shared_ff, width])?,
                experts: if i == 0 {
                    None
                } else {
                    Some(Experts {
                        router: w("ffn_gate_inp.weight", &[width, EXPERTS])?,
                        bias: w("exp_probs_b.bias", &[EXPERTS])?,
                        gate: w("ffn_gate_exps.weight", &[width, ff, EXPERTS])?,
                        up: w("ffn_up_exps.weight", &[width, ff, EXPERTS])?,
                        down: w("ffn_down_exps.weight", &[ff, width, EXPERTS])?,
                    })
                },
                keys: device.alloc(kv_layer)?,
                values: device.alloc(kv_layer)?,
            });
        }
        let weight_bytes = device.allocated_bytes() - kv_bytes;
        debug_assert_eq!(weight_bytes, payload);
        let mut sizes = sizes.into_iter();
        let mut a = || device.alloc(sizes.next().expect("scratch layout"));
        let scratch = Scratch {
            ids: a()?,
            meta: a()?,
            pages: a()?,
            output_rows: a()?,
            decode_rows: a()?,
            attention_tiles: a()?,
            x: a()?,
            norm: a()?,
            q: a()?,
            k: a()?,
            v: a()?,
            gate: a()?,
            attn: a()?,
            parts: a()?,
            delta: a()?,
            fg: a()?,
            fu: a()?,
            router: a()?,
            picks: a()?,
            probabilities: a()?,
            lists: a()?,
            counts: a()?,
            tiles: a()?,
            gu: a()?,
            expert_out: a()?,
            logits: a()?,
        };
        assert!(sizes.next().is_none());
        tracing::info!(
            weight_bytes,
            kv_bytes,
            allocated = device.allocated_bytes(),
            context,
            max_batch,
            ?geometry,
            "native Metal Laguna, full paged F16 KV; experimental text only"
        );
        Ok(Self {
            cold: None,
            source_versions,
            geometry,
            device,
            embedding,
            output_norm,
            head,
            layers,
            scratch,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            pending: VecDeque::new(),
            pool: KvPool::with_blocks(blocks as u32),
            radix: PagedRadix::new(),
            context,
            page_stride,
            rope,
            eps: 1e-6,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
