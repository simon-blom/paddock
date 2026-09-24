//! Muse Glimmer's native Metal decoder geometry. Reuse the paged/ring cache
//! and continuous scheduler, not Gemma's graph constants. The GGUF already
//! folds norm offsets and query multipliers: never apply either twice.
use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};

#[cfg(test)]
#[path = "muse_projection_tests.rs"]
mod projection_tests;
#[cfg(test)]
#[path = "muse_tests.rs"]
mod tests;

// Image phases yield between layers, so their matrix width need not be the
// text tick's latency bound. Include this capacity in scratch and ring slack:
// all new keys are written before a chunk's earliest query reads its window.
pub(super) const IMAGE_CHUNK: usize = 2048;

// Image-sized GEMMs reuse one bounded F16 plane. At >=1024 rows the once-
// per-plane expansion plus direct BM128 MPP is cheaper than dequantizing
// inside every output tile. Narrow text/decode/verification keep their
// qualified compressed routes. The slab is not a persistent weight cache.
pub(super) fn image_projection(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
) {
    let k = planes[0].0.k;
    let prefix = k.div_ceil(128) * 128 * rows.div_ceil(128) * 128;
    assert!(rows >= 1024 && planes.iter().all(|(w, _)| w.ty == 8 && w.k == k));
    cmd.dispatch(
        "linear_input_padded",
        &[input, workspace],
        &[k as u32, 0, rows as u32],
        [prefix.div_ceil(256), 1, 1],
        256,
    );
    for &(w, out) in planes {
        if w.n < 1024 {
            w.linear_prepared(cmd, workspace, out, rows, 1.);
            continue;
        }
        assert!(workspace.len() >= (prefix + k * w.n) * 2);
        let p = [k as u32, w.n as u32, rows as u32, 8, 1f32.to_bits()];
        cmd.dispatch(
            "muse_q8_expand",
            &[&w.buffer, workspace],
            &p,
            [(k * w.n).div_ceil(1024), 1, 1],
            256,
        );
        cmd.dispatch(
            "linear_kexpanded128",
            &[workspace, out],
            &p,
            [w.n.div_ceil(64), rows.div_ceil(128), 1],
            128,
        );
    }
}

impl Gemma4 {
    pub(super) fn load_muse(
        map: &MappedGguf,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let bad = |s: &str| MetalError::Model(format!("Muse Glimmer Metal: {s}"));
        let u = |key: &str| map.gguf().arch_field(key).and_then(Value::as_u64);
        let optional_u = |key: &str, default| match map.gguf().arch_field(key) {
            None => Ok(default),
            Some(value) => value.as_u64().ok_or_else(|| bad(key)),
        };
        let f = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .filter(|v| v.is_finite() && *v > 0.)
                .ok_or_else(|| bad(key))
        };
        let (width, ff, count, vocab, hd, kh, window) = (
            6656usize,
            19968usize,
            52usize,
            202048usize,
            128usize,
            2usize,
            2048usize,
        );
        for (key, value) in [
            ("embedding_length", width),
            ("feed_forward_length", ff),
            ("block_count", count),
            ("attention.head_count", HEADS),
            ("attention.head_count_kv", kh),
            ("attention.key_length", hd),
            ("attention.value_length", hd),
            ("attention.sliding_window", window),
            ("attention.sliding_window_pattern", 4),
        ] {
            if u(key) != Some(value as u64) {
                return Err(bad(key));
            }
        }
        // The canonical exporter omits dimension_count when all head
        // channels rotate. An explicit partial-RoPE variant is not this graph.
        if optional_u("rope.dimension_count", hd as u64)? != hd as u64
            || optional_u("expert_count", 0)? != 0
            || optional_u("attention.shared_kv_layers", 0)? != 0
            || context == 0
            || Some(context as u64) > u("context_length")
            || max_batch == 0
            || max_batch >= CHUNK
        {
            return Err(bad(
                "unsupported graph, context, or batch (expected 1..512)",
            ));
        }
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let rope_base = f("rope.freq_base")?;
        let softcap = f("final_logit_softcapping")?;
        let logit_scale = f("logit_scale")?;
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let ring = context
            .min(window + IMAGE_CHUNK)
            .next_multiple_of(BLOCK_TOKENS);
        let blocks = page_stride
            .checked_mul(max_batch * 2 + 1)
            .filter(|&n| n <= u32::MAX as usize / BLOCK_TOKENS)
            .ok_or_else(|| bad("KV size overflow"))?;
        let global_bytes = blocks * BLOCK_TOKENS * kh * hd * 2;
        let sliding_bytes = ring * max_batch * 2 * kh * hd * 2;
        let kv_bytes = (global_bytes * 13 * 2 + sliding_bytes * 39 * 2) as u64;
        let gemm_bytes = (IMAGE_CHUNK * ff + width * ff).max((IMAGE_CHUNK + 32) * HEADS * hd) * 2;
        let scratch_bytes =
            (IMAGE_CHUNK * (width * 5 + HEADS * hd * 3 + kh * hd * 2 + ff * 2 + 9) * 4
                + max_batch * (page_stride + vocab + HEADS * SPLITS * (hd + 2)) * 4
                + gemm_bytes) as u64;
        let device = MetalDevice::new(budget)?;
        let required = map
            .total_len()
            .saturating_add(kv_bytes)
            .saturating_add(scratch_bytes);
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Muse weights + bounded KV + scratch need {:.2} GiB; grant {:.2} GiB",
                required as f64 / (1u64 << 30) as f64,
                device.budget_bytes() as f64 / (1u64 << 30) as f64
            )));
        }
        let embedding = Weight::load(&device, map, "token_embd.weight", &[width, vocab])?;
        let output = Some(Weight::load(
            &device,
            map,
            "output.weight",
            &[width, vocab],
        )?);
        let output_norm = Weight::load(&device, map, "output_norm.weight", &[width])?;
        // The common dispatch ABI has a rotary-factor input. Muse's kernels
        // never read it (global attention is NoPE); do not invent factors.
        let factors = Weight {
            buffer: device.alloc(4)?,
            ty: 0,
            k: 1,
            n: 1,
        };
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let sliding = i % 4 != 3;
            let w = |name: &str, dims: &[usize]| {
                Weight::load(&device, map, &format!("blk.{i}.{name}.weight"), dims)
            };
            let bytes = if sliding { sliding_bytes } else { global_bytes };
            layers.push(Layer {
                heads: HEADS,
                moe: None,
                norm: w("attn_norm", &[width])?,
                post_attn: w("post_attention_norm", &[width])?,
                ffn_norm: w("ffn_norm", &[width])?,
                post_ffn: w("post_ffw_norm", &[width])?,
                q: w("attn_q", &[width, HEADS * hd])?,
                k: w("attn_k", &[width, kh * hd])?,
                v: Some(w("attn_v", &[width, kh * hd])?),
                o: w("attn_output", &[HEADS * hd, width])?,
                attn_gate: Some(w("attn_gate", &[width, HEADS * hd])?),
                q_norm: w("attn_q_norm", &[hd])?,
                k_norm: w("attn_k_norm", &[hd])?,
                gate: w("ffn_gate", &[width, ff])?,
                up: w("ffn_up", &[width, ff])?,
                down: w("ffn_down", &[ff, width])?,
                scale: 1.,
                sliding,
                keys: device.alloc(bytes)?,
                values: device.alloc(bytes)?,
            });
        }
        let weight_bytes = device.allocated_bytes() - kv_bytes;
        let a = |n: usize| device.alloc(IMAGE_CHUNK * n * 4);
        let scratch = Scratch {
            rows: IMAGE_CHUNK,
            ids: a(1)?,
            meta: a(2)?,
            limits: a(1)?,
            pages: device.alloc(max_batch * page_stride * 4)?,
            outputs: a(1)?,
            tiles: a(2)?,
            decode_rows: a(1)?,
            x: a(width)?,
            norm: a(width)?,
            delta: a(width)?,
            q: a(HEADS * hd)?,
            k: a(kh * hd)?,
            v: a(kh * hd)?,
            attn: a(HEADS * hd)?,
            attn_gate: a(HEADS * hd)?,
            parts: device.alloc(max_batch * HEADS * SPLITS * (hd + 2) * 4)?,
            gate: a(ff)?,
            up: a(ff)?,
            gemm: device.alloc(gemm_bytes)?,
            logits: device.alloc(max_batch * vocab * 4)?,
        };
        let image_markers = match map.gguf().metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array(v)) => v
                .iter()
                .position(|v| v.as_str() == Some("<|image_start|>"))
                .zip(v.iter().position(|v| v.as_str() == Some("<|image_end|>")))
                .map(|(a, b)| (a as u32, b as u32)),
            _ => None,
        };
        Ok(Self {
            diffusion: None,
            muse: true,
            mlx: false,
            device,
            embedding,
            output,
            output_norm,
            factors,
            layers,
            scratch,
            moe_scratch: None,
            mtp: None,
            dflash: None,
            vision: None,
            image_markers,
            image_cache: Vec::new(),
            image_cache_reused: 0,
            encoding: VecDeque::new(),
            spec: None,
            verifying: false,
            greedy_verify: false,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            cache: (0..max_batch).map(|_| Checkpoint::default()).collect(),
            clock: 0,
            pending: VecDeque::new(),
            prefill_phase: None,
            pool: KvPool::with_blocks(blocks as u32),
            width,
            ff,
            vocab,
            context,
            window,
            ring,
            page_stride,
            eps,
            rope: [rope_base, rope_base],
            softcap,
            logit_scale,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
