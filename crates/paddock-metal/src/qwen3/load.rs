use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::path::Path;

impl Qwen3Encoder {
    pub fn load(path: &Path, context: usize, budget: Option<u64>) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let g = map.gguf();
        if g.architecture() != Some("qwen3") {
            return Err(MetalError::Model(
                "expected Qwen3 dense encoder GGUF".into(),
            ));
        }
        let u = |k: &str| -> Result<usize> {
            g.arch_field(k)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n > 0 && *n <= u32::MAX as usize)
                .ok_or_else(|| MetalError::Model(format!("invalid qwen3.{k}")))
        };
        let f = |k: &str| -> Result<f32> {
            g.arch_field(k)
                .and_then(Value::as_f32)
                .filter(|n| n.is_finite() && *n > 0.)
                .ok_or_else(|| MetalError::Model(format!("invalid qwen3.{k}")))
        };
        let width = u("embedding_length")?;
        let ff = u("feed_forward_length")?;
        let count = u("block_count")?;
        let heads = u("attention.head_count")?;
        let kv_heads = u("attention.head_count_kv")?;
        if !matches!(
            (width, ff, count, heads, kv_heads),
            (1024, 3072, 28, 16, 8) | (2560, 9728, 36, 32, 8) | (4096, 12288, 36, 32, 8)
        ) || u("attention.key_length")? != 128
            || u("attention.value_length")? != 128
            || (g.arch_field("rope.dimension_count").is_some() && u("rope.dimension_count")? != 128)
        {
            return Err(MetalError::Model(
                "unelected Qwen3 encoder geometry (0.6B/4B/8B hd128 only)".into(),
            ));
        }
        let trained = u("context_length")?.min(32768);
        if context == 0 || context > trained {
            return Err(MetalError::Model(format!(
                "encoder context must be 1..={trained}"
            )));
        }
        if g.arch_field("rope.scaling.type").is_some()
            || g.arch_field("attention.sliding_window").is_some()
        {
            return Err(MetalError::Model(
                "scaled/sliding Qwen3 encoders are not qualified".into(),
            ));
        }
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let rope = f("rope.freq_base")?;
        let vocab = map
            .tensor_info("token_embd.weight")
            .and_then(|t| t.dims.get(1))
            .copied()
            .filter(|&n| n > 0 && n <= u32::MAX as u64)
            .ok_or_else(|| MetalError::Model("invalid embedding vocabulary".into()))?
            as usize;
        // Keep the supported artifact class narrow: no accidental K-quant or
        // BF16 projections from merely recognizing the architecture string.
        for t in &g.tensors {
            if t.dims.len() == 2 && t.raw_type != 8 {
                return Err(MetalError::Model(format!(
                    "{}: Qwen3 Metal encoder requires Q8_0 matrices",
                    t.name
                )));
            }
            let expected = matches!(
                t.name.as_str(),
                "token_embd.weight" | "output_norm.weight" | "output.weight"
            ) || (0..count).any(|i| {
                [
                    "attn_norm",
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_q_norm",
                    "attn_k_norm",
                    "attn_output",
                    "ffn_norm",
                    "ffn_gate",
                    "ffn_up",
                    "ffn_down",
                ]
                .iter()
                .any(|n| t.name == format!("blk.{i}.{n}.weight"))
            });
            if !expected {
                return Err(MetalError::Model(format!(
                    "unexpected Qwen3 encoder tensor: {}",
                    t.name
                )));
            }
        }
        let device = MetalDevice::new(budget)?;
        let load = |name: &str, dims: &[usize]| Weight::load(&device, &map, name, dims);
        let embedding = load("token_embd.weight", &[width, vocab])?;
        let output_norm = load("output_norm.weight", &[width])?;
        let head = map
            .tensor_info("output.weight")
            .map(|_| load("output.weight", &[width, vocab]))
            .transpose()?;
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let l = |n: &str, d: &[usize]| load(&format!("blk.{i}.{n}.weight"), d);
            layers.push(Layer {
                norm: l("attn_norm", &[width])?,
                q: l("attn_q", &[width, heads * 128])?,
                k: l("attn_k", &[width, kv_heads * 128])?,
                v: l("attn_v", &[width, kv_heads * 128])?,
                qnorm: l("attn_q_norm", &[128])?,
                knorm: l("attn_k_norm", &[128])?,
                o: l("attn_output", &[heads * 128, width])?,
                ffn_norm: l("ffn_norm", &[width])?,
                gate: l("ffn_gate", &[width, ff])?,
                up: l("ffn_up", &[width, ff])?,
                down: l("ffn_down", &[ff, width])?,
            });
        }
        let weight_bytes = device.allocated_bytes();
        // Bounded, reusable O(rows) scratch. Full context must fit. Reserve
        // two worst-case result+metadata sets for enqueue-ahead, not only the
        // one-sequence happy path. No layer-proportional KV allocation.
        let qwidth = heads * 128;
        let kvwidth = kv_heads * 128;
        let scratch_per_row = 4 * (3 * width + 2 * qwidth + 2 * kvwidth + 2 * ff)
            + 2 * (qwidth + 2 * kvwidth + ff.max(qwidth));
        let live_per_row = scratch_per_row + 2 * (width * 4 + 28);
        let available = (device.budget_bytes() - weight_bytes) as usize;
        let row_budget = (available / live_per_row).min(65536) / 128 * 128;
        if row_budget < context {
            return Err(MetalError::Memory(
                "Qwen3 weights, full-context scratch and two result sets do not fit".into(),
            ));
        }
        // Do not reserve all free memory simply because it exists. The 64k
        // coalescing ceiling is also the largest scheduler admission unit.
        let row_budget = row_budget.min(context.max(8192).next_multiple_of(128));
        let a = |n: usize| device.alloc(row_budget * n);
        let scratch = Scratch {
            x: a(width * 4)?,
            norm: a(width * 4)?,
            delta: a(width * 4)?,
            q: a(qwidth * 4)?,
            k: a(kvwidth * 4)?,
            v: a(kvwidth * 4)?,
            qh: a(qwidth * 2)?,
            kh: a(kvwidth * 2)?,
            vh: a(kvwidth * 2)?,
            attn: a(qwidth * 4)?,
            gate: a(ff * 4)?,
            up: a(ff * 4)?,
            gemm: a(ff.max(qwidth) * 2)?,
        };
        tracing::info!(
            width,
            layers = count,
            context,
            row_budget,
            weight_bytes,
            memory = device.allocated_bytes(),
            "native Metal Qwen3 encoder loaded"
        );
        Ok(Self {
            identity: std::rc::Rc::new(()),
            device,
            embedding,
            output_norm,
            head,
            layers,
            scratch,
            width,
            heads,
            kv_heads,
            ff,
            vocab,
            context,
            row_budget,
            eps,
            rope,
            weight_bytes,
        })
    }
}
