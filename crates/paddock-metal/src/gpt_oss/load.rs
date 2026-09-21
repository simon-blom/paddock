use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::{collections::HashSet, path::Path};

impl GptOss {
    /// Elected GGUF geometry only. All expert payloads stay native MXFP4;
    /// dense Q8/F16 and small F32 tensors are validated before GPU allocation.
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let source_versions = crate::offload::versions(path)?;
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        if map.gguf().architecture() != Some("gpt-oss") {
            return Err(MetalError::Model(
                "Metal GPT-OSS requires architecture=gpt-oss".into(),
            ));
        }
        let u = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| MetalError::Model(format!("missing/invalid gpt-oss.{key}")))
        };
        let f = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .filter(|v| v.is_finite())
                .ok_or_else(|| MetalError::Model(format!("missing/invalid gpt-oss.{key}")))
        };
        let count = u("block_count")?;
        let experts = u("expert_count")?;
        if !matches!((count, experts), (24, 32) | (36, 128))
            || u("embedding_length")? != WIDTH
            || u("attention.head_count")? != 64
            || u("attention.head_count_kv")? != 8
            || u("attention.key_length")? != 64
            || u("attention.value_length")? != 64
            || u("expert_feed_forward_length")? != WIDTH
            || u("expert_used_count")? != 4
            || u("attention.sliding_window")? != 128
            // The elected conversion omits this optional full-head field.
            // An explicitly different dimension is not silently accepted.
            || map.gguf().arch_field("rope.dimension_count")
                .is_some_and(|v| v.as_u64() != Some(64))
        {
            return Err(MetalError::Model(
                "unsupported GPT-OSS geometry; requires elected 20B/120B MXFP4".into(),
            ));
        }
        let trained = u("context_length")?;
        if context == 0 || context > trained.min(32768) || max_batch == 0 || max_batch > 64 {
            return Err(MetalError::Model(format!(
                "Metal GPT-OSS context must be 1..={}, batch 1..=64",
                trained.min(32768)
            )));
        }
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let base = f("rope.freq_base")?;
        let factor = f("rope.scaling.factor")?;
        let original = u("rope.scaling.original_context_length")?;
        if eps != 1e-5
            || base != 150000.0
            || factor != 32.0
            || original != 4096
            || map
                .gguf()
                .arch_field("rope.scaling.type")
                .and_then(Value::as_str)
                != Some("yarn")
        {
            return Err(MetalError::Model(
                "unsupported GPT-OSS normalization/YaRN metadata".into(),
            ));
        }
        // GGUF uses rounded YaRN correction dimensions, unlike the upstream
        // Torch float boundaries. This is metadata-derived scalar setup, not
        // host inference. Match the same-weight GGUF graph's convention.
        let corr =
            |n: f32| 32.0 * (original as f32 / (n * 2.0 * std::f32::consts::PI)).ln() / base.ln();
        let rope = [
            base,
            1.0 / factor,
            corr(32.).floor().max(0.),
            corr(1.).ceil().min(63.),
            1.0 + 0.1 * factor.ln(),
        ]
        .map(f32::to_bits);
        let mut expected = HashSet::new();
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
            expected.insert(name);
            Ok(())
        };
        check("token_embd.weight".into(), &[WIDTH, VOCAB], &[8])?;
        check("output.weight".into(), &[WIDTH, VOCAB], &[8])?;
        check("output_norm.weight".into(), &[WIDTH], &[0])?;
        for i in 0..count {
            for (name, dims, types) in [
                ("attn_norm.weight", vec![WIDTH], vec![0]),
                ("post_attention_norm.weight", vec![WIDTH], vec![0]),
                ("attn_q.weight", vec![WIDTH, QWIDTH], vec![8]),
                ("attn_k.weight", vec![WIDTH, KVWIDTH], vec![8]),
                ("attn_v.weight", vec![WIDTH, KVWIDTH], vec![8]),
                ("attn_output.weight", vec![QWIDTH, WIDTH], vec![8]),
                ("attn_q.bias", vec![QWIDTH], vec![0]),
                ("attn_k.bias", vec![KVWIDTH], vec![0]),
                ("attn_v.bias", vec![KVWIDTH], vec![0]),
                ("attn_output.bias", vec![WIDTH], vec![0]),
                ("attn_sinks.weight", vec![64], vec![0]),
                ("ffn_gate_inp.weight", vec![WIDTH, experts], vec![0]),
                ("ffn_gate_inp.bias", vec![experts], vec![0]),
                (
                    "ffn_gate_exps.weight",
                    vec![WIDTH, WIDTH, experts],
                    vec![39],
                ),
                ("ffn_up_exps.weight", vec![WIDTH, WIDTH, experts], vec![39]),
                (
                    "ffn_down_exps.weight",
                    vec![WIDTH, WIDTH, experts],
                    vec![39],
                ),
                ("ffn_gate_exps.bias", vec![WIDTH, experts], vec![0]),
                ("ffn_up_exps.bias", vec![WIDTH, experts], vec![0]),
                ("ffn_down_exps.bias", vec![WIDTH, experts], vec![0]),
            ] {
                check(format!("blk.{i}.{name}"), &dims, &types)?;
            }
        }
        if let Some(t) = map
            .gguf()
            .tensors
            .iter()
            .find(|t| !expected.contains(&t.name))
        {
            return Err(MetalError::Model(format!(
                "unexpected GPT-OSS tensor {}",
                t.name
            )));
        }
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let blocks = page_stride * (max_batch + 1);
        // Full paged KV on sliding layers intentionally preserves exact radix
        // resumes. Do not advertise the CUDA ring-cache memory footprint here.
        let kv_layer = blocks * BLOCK_TOKENS * KVWIDTH * 2;
        let kv_bytes = (kv_layer * 2 * count) as u64;
        let tile_cap = (CHUNK * 4).div_ceil(16) + experts;
        let sizes = [
            CHUNK * 4,
            CHUNK * 8,
            max_batch * page_stride * 4,
            CHUNK * 4,
            CHUNK * 4,
            CHUNK * 8,
            CHUNK * WIDTH * 4,
            CHUNK * WIDTH * 4,
            CHUNK * QWIDTH * 4,
            CHUNK * QWIDTH * 2,
            CHUNK * KVWIDTH * 4,
            CHUNK * KVWIDTH * 4,
            CHUNK * QWIDTH * 4,
            CHUNK * 64 * SPLITS * 66 * 4,
            CHUNK * WIDTH * 4,
            CHUNK * QWIDTH * 2,
            CHUNK * experts * 4,
            CHUNK * 4 * 4,
            CHUNK * 4 * 4,
            experts * CHUNK * 4 * 4,
            experts * 4,
            (1 + 2 * tile_cap) * 4,
            CHUNK * 4 * WIDTH * 2 * 4,
            CHUNK * 4 * WIDTH * 4,
            max_batch * VOCAB * 4,
        ];
        let device = MetalDevice::new(budget)?;
        let required = map
            .total_len()
            .saturating_add(kv_bytes)
            .saturating_add(sizes.iter().sum::<usize>() as u64);
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "GPT-OSS weights, full paged KV and scratch need {required} bytes; grant {}",
                device.budget_bytes()
            )));
        }
        let w = |name: &str, dims: &[usize]| Weight::load(&device, &map, name, dims);
        let embedding = w("token_embd.weight", &[WIDTH, VOCAB])?;
        let output_norm = w("output_norm.weight", &[WIDTH])?;
        let head = w("output.weight", &[WIDTH, VOCAB])?;
        let mut layers = Vec::new();
        for i in 0..count {
            let w = |name: &str, dims: &[usize]| {
                Weight::load(&device, &map, &format!("blk.{i}.{name}"), dims)
            };
            let expert = |name: &str| {
                device.upload(
                    map.tensor_bytes(&format!("blk.{i}.{name}.weight"))
                        .map_err(|e| MetalError::Model(e.to_string()))?
                        .1,
                )
            };
            layers.push(Layer {
                norm: w("attn_norm.weight", &[WIDTH])?,
                post: w("post_attention_norm.weight", &[WIDTH])?,
                q: w("attn_q.weight", &[WIDTH, QWIDTH])?,
                k: w("attn_k.weight", &[WIDTH, KVWIDTH])?,
                v: w("attn_v.weight", &[WIDTH, KVWIDTH])?,
                o: w("attn_output.weight", &[QWIDTH, WIDTH])?,
                qb: w("attn_q.bias", &[QWIDTH])?,
                kb: w("attn_k.bias", &[KVWIDTH])?,
                vb: w("attn_v.bias", &[KVWIDTH])?,
                ob: w("attn_output.bias", &[WIDTH])?,
                sinks: w("attn_sinks.weight", &[64])?,
                router: w("ffn_gate_inp.weight", &[WIDTH, experts])?,
                router_bias: w("ffn_gate_inp.bias", &[experts])?,
                gate: expert("ffn_gate_exps")?,
                up: expert("ffn_up_exps")?,
                down: expert("ffn_down_exps")?,
                gate_bias: w("ffn_gate_exps.bias", &[WIDTH, experts])?,
                up_bias: w("ffn_up_exps.bias", &[WIDTH, experts])?,
                down_bias: w("ffn_down_exps.bias", &[WIDTH, experts])?,
                keys: device.alloc(kv_layer)?,
                values: device.alloc(kv_layer)?,
            });
        }
        let weight_bytes = device.allocated_bytes() - kv_bytes;
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
            qhalf: a()?,
            k: a()?,
            v: a()?,
            attn: a()?,
            parts: a()?,
            delta: a()?,
            gemm: a()?,
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
            count,
            experts,
            weight_bytes,
            kv_bytes,
            "GPT-OSS native MXFP4 loaded on Metal"
        );
        Ok(Self {
            cold: None,
            source_versions,
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
            experts,
            context,
            page_stride,
            rope,
            eps,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
