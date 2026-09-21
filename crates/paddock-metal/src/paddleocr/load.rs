use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
impl PaddleOcr {
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        if map.gguf().architecture() != Some("paddleocr") {
            return Err(error("requires architecture=paddleocr"));
        }
        for (key, value) in [
            ("embedding_length", WIDTH),
            ("block_count", LAYERS),
            ("attention.head_count", 16),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("feed_forward_length", FF),
            ("context_length", 131072),
        ] {
            if map.gguf().arch_field(key).and_then(Value::as_u64) != Some(value as u64) {
                return Err(error(format!("unsupported {key}")));
            }
        }
        for (key, value) in [
            ("attention.layer_norm_rms_epsilon", 1e-5),
            ("rope.freq_base", 500000.),
        ] {
            if map.gguf().arch_field(key).and_then(Value::as_f32) != Some(value) {
                return Err(error(format!("unsupported {key}")));
            }
        }
        if !matches!(map.gguf().arch_field("rope.dimension_sections"),Some(Value::Array(a)) if a.iter().map(Value::as_i64).collect::<Option<Vec<_>>>()==Some(vec![16,24,24,0]))
            || map.gguf().arch_field("rope.scaling.type").is_some()
        {
            return Err(error("unsupported rotary sections/scaling"));
        }
        if context == 0 || context > 32768 || max_batch == 0 || max_batch > 16 {
            return Err(error(
                "context must be 1..32768, batch 1..16 (implementation ceilings)",
            ));
        }
        let mut schema = vec![
            ("token_embd.weight".to_owned(), vec![WIDTH, VOCAB], 30),
            ("output.weight".to_owned(), vec![WIDTH, VOCAB], 30),
            ("output_norm.weight".to_owned(), vec![WIDTH], 0),
        ];
        for i in 0..LAYERS {
            for (name, dims, ty) in [
                ("attn_norm.weight", vec![WIDTH], 0),
                ("ffn_norm.weight", vec![WIDTH], 0),
                ("attn_q.weight", vec![WIDTH, QWIDTH], 30),
                ("attn_k.weight", vec![WIDTH, KVWIDTH], 30),
                ("attn_v.weight", vec![WIDTH, KVWIDTH], 30),
                ("attn_output.weight", vec![QWIDTH, WIDTH], 30),
                ("ffn_gate.weight", vec![WIDTH, FF], 30),
                ("ffn_up.weight", vec![WIDTH, FF], 30),
                ("ffn_down.weight", vec![FF, WIDTH], 30),
            ] {
                schema.push((format!("blk.{i}.{name}"), dims, ty));
            }
        }
        let weight_bytes = validate(&map, &schema)?;
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let blocks = page_stride * (max_batch + 1);
        let kv_bytes = (blocks * BLOCK_TOKENS * KVWIDTH * 4 * LAYERS) as u64;
        let scratch_bytes = Scratch::bytes(page_stride, max_batch);
        let device = MetalDevice::new(budget)?;
        if weight_bytes + kv_bytes + scratch_bytes as u64 > device.budget_bytes() {
            return Err(MetalError::Memory(
                "PaddleOCR decoder weights + KV + scratch exceed grant".into(),
            ));
        }
        let load = |name: &str, dims: &[usize]| Weight::load(&device, &map, name, dims);
        let embedding = load("token_embd.weight", &[WIDTH, VOCAB])?;
        let head = load("output.weight", &[WIDTH, VOCAB])?;
        let output_norm = load("output_norm.weight", &[WIDTH])?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let w = |n: &str, d: &[usize]| load(&format!("blk.{i}.{n}"), d);
            layers.push(Layer {
                norm: w("attn_norm.weight", &[WIDTH])?,
                post: w("ffn_norm.weight", &[WIDTH])?,
                q: w("attn_q.weight", &[WIDTH, QWIDTH])?,
                k: w("attn_k.weight", &[WIDTH, KVWIDTH])?,
                v: w("attn_v.weight", &[WIDTH, KVWIDTH])?,
                o: w("attn_output.weight", &[QWIDTH, WIDTH])?,
                gate: w("ffn_gate.weight", &[WIDTH, FF])?,
                up: w("ffn_up.weight", &[WIDTH, FF])?,
                down: w("ffn_down.weight", &[FF, WIDTH])?,
                keys: device.alloc(blocks * BLOCK_TOKENS * KVWIDTH * 2)?,
                values: device.alloc(blocks * BLOCK_TOKENS * KVWIDTH * 2)?,
            });
        }
        let scratch = Scratch::new(&device, page_stride, max_batch)?;
        Ok(Self {
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
    fn sizes(stride: usize, batch: usize) -> [usize; 18] {
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
            CHUNK * 16 * SPLITS * 130,
            CHUNK * WIDTH,
            CHUNK * FF,
            CHUNK * FF,
            batch * VOCAB,
        ]
    }
    fn bytes(stride: usize, batch: usize) -> usize {
        Self::sizes(stride, batch).iter().sum::<usize>() * 4
    }
    fn new(d: &MetalDevice, stride: usize, batch: usize) -> Result<Self> {
        let sizes = Self::sizes(stride, batch);
        let a = |i: usize| d.alloc(sizes[i] * 4);
        Ok(Self {
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
