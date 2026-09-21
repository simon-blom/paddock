use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
impl UnlimitedOcr {
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        if map.gguf().architecture() != Some("deepseek2-ocr") {
            return Err(error("requires architecture=deepseek2-ocr"));
        }
        for (key, value) in [
            ("embedding_length", WIDTH),
            ("block_count", LAYERS),
            ("attention.head_count", 10),
            ("attention.head_count_kv", 10),
            ("feed_forward_length", FF),
            ("context_length", 32768),
        ] {
            if map.gguf().arch_field(key).and_then(Value::as_u64) != Some(value as u64) {
                return Err(error(format!("unsupported {key}")));
            }
        }
        for (key, value) in [
            ("expert_count", 64),
            ("expert_used_count", 6),
            ("expert_feed_forward_length", 896),
            ("expert_shared_count", 2),
            ("leading_dense_block_count", 1),
            ("attention.sliding_window", 128),
            ("vocab_size", VOCAB),
            ("expert_group_count", 1),
            ("expert_group_used_count", 1),
        ] {
            if map.gguf().arch_field(key).and_then(Value::as_u64) != Some(value as u64) {
                return Err(error(format!("unsupported {key}")));
            }
        }
        if map
            .gguf()
            .arch_field("attention.layer_norm_rms_epsilon")
            .and_then(Value::as_f32)
            != Some(1e-6)
            || map.gguf().arch_field("rope.scaling.type").is_some()
            || map
                .gguf()
                .arch_field("rope.freq_base")
                .is_some_and(|v| v.as_f32() != Some(10000.))
        {
            return Err(error("unsupported normalization/rotary metadata"));
        }
        if context == 0 || context > 32768 || max_batch == 0 || max_batch > 16 {
            return Err(error(
                "context must be 1..32768, batch 1..16 (implementation ceilings)",
            ));
        }
        let mut schema = vec![
            ("token_embd.weight".to_owned(), vec![WIDTH, VOCAB], 8),
            ("output.weight".to_owned(), vec![WIDTH, VOCAB], 8),
            ("output_norm.weight".to_owned(), vec![WIDTH], 0),
        ];
        for i in 0..LAYERS {
            let ff = if i == 0 { FF } else { 1792 };
            let suffix = if i == 0 { "" } else { "_shexp" };
            for (name, dims, ty) in [
                ("attn_norm.weight", vec![WIDTH], 0),
                ("ffn_norm.weight", vec![WIDTH], 0),
                ("attn_q.weight", vec![WIDTH, QWIDTH], 8),
                ("attn_k.weight", vec![WIDTH, KVWIDTH], 8),
                ("attn_v.weight", vec![WIDTH, KVWIDTH], 8),
                ("attn_output.weight", vec![QWIDTH, WIDTH], 8),
                ("ffn_gate.weight", vec![WIDTH, ff], 8),
                ("ffn_up.weight", vec![WIDTH, ff], 8),
                ("ffn_down.weight", vec![ff, WIDTH], 8),
            ] {
                let name = if name.starts_with("ffn_") && name != "ffn_norm.weight" {
                    name.replace(".weight", &format!("{suffix}.weight"))
                } else {
                    name.into()
                };
                schema.push((format!("blk.{i}.{name}"), dims, ty));
            }
        }
        for i in 1..LAYERS {
            moe::schema(&mut schema, i);
        }
        let weight_bytes = validate(&map, &schema)?;
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let blocks = page_stride * (max_batch + 1);
        let kv_bytes = (blocks * BLOCK_TOKENS * KVWIDTH * 4 * LAYERS) as u64;
        let scratch_bytes = Scratch::bytes(page_stride, max_batch) + moe::Workspace::bytes();
        let device = MetalDevice::new(budget)?;
        if weight_bytes + kv_bytes + scratch_bytes as u64 > device.budget_bytes() {
            return Err(MetalError::Memory(
                "Unlimited-OCR decoder weights + KV + scratch exceed grant".into(),
            ));
        }
        let load = |name: &str, dims: &[usize]| Weight::load(&device, &map, name, dims);
        let embedding = load("token_embd.weight", &[WIDTH, VOCAB])?;
        let head = load("output.weight", &[WIDTH, VOCAB])?;
        let output_norm = load("output_norm.weight", &[WIDTH])?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let w = |n: &str, d: &[usize]| load(&format!("blk.{i}.{n}"), d);
            let ff = if i == 0 { FF } else { 1792 };
            let suffix = if i == 0 { "" } else { "_shexp" };
            layers.push(Layer {
                norm: w("attn_norm.weight", &[WIDTH])?,
                post: w("ffn_norm.weight", &[WIDTH])?,
                q: w("attn_q.weight", &[WIDTH, QWIDTH])?,
                k: w("attn_k.weight", &[WIDTH, KVWIDTH])?,
                v: w("attn_v.weight", &[WIDTH, KVWIDTH])?,
                o: w("attn_output.weight", &[QWIDTH, WIDTH])?,
                gate: w(&format!("ffn_gate{suffix}.weight"), &[WIDTH, ff])?,
                up: w(&format!("ffn_up{suffix}.weight"), &[WIDTH, ff])?,
                down: w(&format!("ffn_down{suffix}.weight"), &[ff, WIDTH])?,
                experts: if i == 0 {
                    None
                } else {
                    Some(moe::Experts::load(&device, &map, i)?)
                },
                keys: device.alloc(blocks * BLOCK_TOKENS * KVWIDTH * 2)?,
                values: device.alloc(blocks * BLOCK_TOKENS * KVWIDTH * 2)?,
            });
        }
        let scratch = Scratch::new(&device, page_stride, max_batch)?;
        let moe = moe::Workspace::new(&device)?;
        Ok(Self {
            moe,
            device,
            embedding,
            head,
            output_norm,
            layers,
            scratch,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            pending: VecDeque::new(),
            pool: KvPool::with_blocks(blocks as u32),
            radix: PagedRadix::new(),
            vision: None,
            encoding: VecDeque::new(),
            context,
            page_stride,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
// Complete schema validation before any tensor upload. Quant support is not
// inferred from the architecture name or a permissive shared weight loader.
pub(super) fn validate(map: &MappedGguf, schema: &[(String, Vec<usize>, u32)]) -> Result<u64> {
    if map.gguf().tensors.len() != schema.len() {
        return Err(error("unexpected tensor inventory"));
    }
    let mut bytes = 0;
    for (name, dims, ty) in schema {
        let (t, data) = map.tensor_bytes(name).map_err(|e| error(e.to_string()))?;
        if t.dims.iter().map(|&d| d as usize).collect::<Vec<_>>() != *dims
            || t.raw_type != *ty
            || t.ggml_type
                .byte_size(dims.iter().map(|&d| d as u64).product())
                != Some(data.len() as u64)
        {
            return Err(error(format!("{name}: unexpected shape/type/bytes")));
        }
        bytes += data.len() as u64;
    }
    Ok(bytes)
}
impl Scratch {
    fn sizes(stride: usize, batch: usize) -> [usize; 19] {
        [
            CHUNK,
            CHUNK * 2,
            CHUNK * 4,
            stride * batch,
            batch,
            CHUNK,
            CHUNK * 2,
            CHUNK * WIDTH,
            CHUNK * WIDTH,
            CHUNK * QWIDTH,
            CHUNK * KVWIDTH,
            CHUNK * KVWIDTH,
            CHUNK * QWIDTH,
            CHUNK * 10 * SPLITS * 130,
            CHUNK * WIDTH,
            CHUNK * FF,
            CHUNK * FF,
            batch * VOCAB,
            CHUNK * 2,
        ]
    }
    fn bytes(stride: usize, batch: usize) -> usize {
        Self::sizes(stride, batch).iter().sum::<usize>() * 4
    }
    fn new(d: &MetalDevice, stride: usize, batch: usize) -> Result<Self> {
        let sizes = Self::sizes(stride, batch);
        let a = |i: usize| d.alloc(sizes[i] * 4);
        Ok(Self {
            write_meta: a(18)?,
            ids: a(0)?,
            meta: a(1)?,
            rope: a(2)?,
            pages: a(3)?,
            output_rows: a(4)?,
            decode_rows: a(5)?,
            tiles: a(6)?,
            x: a(7)?,
            norm: a(8)?,
            q: a(9)?,
            k: a(10)?,
            v: a(11)?,
            attn: a(12)?,
            parts: a(13)?,
            delta: a(14)?,
            gate: a(15)?,
            up: a(16)?,
            logits: a(17)?,
        })
    }
}
