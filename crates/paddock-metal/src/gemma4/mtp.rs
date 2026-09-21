//! Gemma's Q-only assistant reads the last SWA/global target caches. Unlike
//! Qwen nextn it has no KV of its own, no catch-up pass, and its own head.
//! Positions advance along the chain while attention remains bounded by the
//! committed target cursor. All steps share one GPU submission/readback.
use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};

const WIDTH: usize = 1024;
const FF: usize = 8192;

struct DraftLayer {
    norm: Weight,
    post_attn: Weight,
    ffn_norm: Weight,
    post_ffn: Weight,
    q: Weight,
    q_norm: Weight,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    scale: f32,
    target: usize,
}
pub(super) struct Mtp {
    pre: Weight,
    post: Weight,
    head: Weight,
    norm: Weight,
    factors: Weight,
    layers: Vec<DraftLayer>,
    eps: f32,
    rope: [f32; 2],
    pub pending: Buffer,
    pub cursor: Vec<Option<usize>>,
    h: Buffer,
    concat: Buffer,
    bounds: Buffer,
    drafted: Buffer,
}

impl Gemma4 {
    /// Attach before any prefill; spec-off retains the ordinary memory/graph.
    pub fn attach_mtp(&mut self, path: &Path) -> Result<()> {
        if self.mlx {
            return Err(MetalError::Model(
                "MLX Gemma assistant attachment requires separate BF16-KV qualification".into(),
            ));
        }
        if self.muse {
            return Err(MetalError::Model("Muse uses a DFlash drafter, not a Gemma MTP assistant; Muse Metal speculation is not yet qualified".into()));
        }
        self.require_committed()?;
        if self.mtp.is_some()
            || self.slots.iter().any(|s| !s.history.is_empty())
            || self.cache.iter().any(|s| !s.history.is_empty())
            || !self.pending.is_empty()
            || !self.encoding.is_empty()
        {
            return Err(MetalError::Model(
                "attach Gemma assistant once, before prefill".into(),
            ));
        }
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let bad = |s: &str| MetalError::Model(format!("Gemma assistant: {s}"));
        if map.gguf().architecture() != Some("gemma4-assistant") {
            return Err(bad("expected gemma4-assistant GGUF"));
        }
        let u = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| bad(key))
        };
        let heads = self.layers[0].heads;
        let global = self.layers.len() - 1;
        let sliding = global - 1;
        for (key, value) in [
            ("embedding_length", WIDTH),
            ("embedding_length_out", self.width),
            ("feed_forward_length", FF),
            ("block_count", 4),
            ("attention.head_count", heads),
            ("attention.shared_kv_layers", 4),
            ("attention.sliding_window", self.window),
            ("attention.key_length", 512),
            ("attention.key_length_swa", 256),
            ("attention.value_length", 512),
            ("attention.value_length_swa", 256),
            ("rope.dimension_count", 512),
            ("rope.dimension_count_swa", 256),
            ("embedding_length_per_layer_input", 0),
        ] {
            if u(key)? != value as u64 {
                return Err(bad(key));
            }
        }
        let f = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .filter(|x| x.is_finite() && *x > 0.)
                .ok_or_else(|| bad(key))
        };
        let pattern = match map.gguf().arch_field("attention.sliding_window_pattern") {
            Some(Value::Array(v)) if v.len() == 4 => v
                .iter()
                .enumerate()
                .all(|(i, x)| matches!(x, Value::Bool(b) if *b == (i < 3))),
            _ => false,
        };
        if !pattern || !self.layers[sliding].sliding || self.layers[global].sliding {
            return Err(bad("incompatible target shared-KV map"));
        }
        let kv_heads = match map.gguf().arch_field("attention.head_count_kv") {
            Some(Value::Array(v)) if v.len() == 4 => v.iter().enumerate().all(|(i, x)| {
                x.as_u64()
                    == Some(if i < 3 {
                        heads as u64 / 2
                    } else {
                        heads as u64 / 8
                    })
            }),
            _ => false,
        };
        if !kv_heads || self.context as u64 > u("context_length")? {
            return Err(bad("incompatible shared KV heads or context"));
        }
        let before = self.device.allocated_bytes();
        let w = |name: &str, dims: &[usize]| Weight::load(&self.device, &map, name, dims);
        let pre = w("nextn.pre_projection.weight", &[self.width * 2, WIDTH])?;
        let post = w("nextn.post_projection.weight", &[WIDTH, self.width])?;
        let head = w("token_embd.weight", &[WIDTH, self.vocab])?;
        let norm = w("output_norm.weight", &[WIDTH])?;
        let factors = w("rope_freqs.weight", &[256])?;
        if factors.ty != 0 {
            return Err(bad("rotary factors must be F32"));
        }
        let mut layers = Vec::new();
        for i in 0..4 {
            let hd = if i < 3 { 256 } else { 512 };
            let lw = |name: &str, dims: &[usize]| w(&format!("blk.{i}.{name}.weight"), dims);
            if map.tensor_info(&format!("blk.{i}.attn_k.weight")).is_some()
                || map.tensor_info(&format!("blk.{i}.attn_v.weight")).is_some()
            {
                return Err(bad("Q-only assistant must not own K/V projections"));
            }
            let scale = if map
                .tensor_info(&format!("blk.{i}.layer_output_scale.weight"))
                .is_some()
            {
                let s = lw("layer_output_scale", &[1])?;
                if s.ty != 0 {
                    return Err(bad("layer scale must be F32"));
                }
                unsafe { s.buffer.read_f32(0, 1)[0] }
            } else {
                1.
            };
            if !scale.is_finite() {
                return Err(bad("non-finite layer scale"));
            }
            layers.push(DraftLayer {
                norm: lw("attn_norm", &[WIDTH])?,
                post_attn: lw("post_attention_norm", &[WIDTH])?,
                ffn_norm: lw("ffn_norm", &[WIDTH])?,
                post_ffn: lw("post_ffw_norm", &[WIDTH])?,
                q: lw("attn_q", &[WIDTH, heads * hd])?,
                q_norm: lw("attn_q_norm", &[hd])?,
                o: lw("attn_output", &[heads * hd, WIDTH])?,
                gate: lw("ffn_gate", &[WIDTH, FF])?,
                up: lw("ffn_up", &[WIDTH, FF])?,
                down: lw("ffn_down", &[FF, WIDTH])?,
                scale,
                target: if i < 3 { sliding } else { global },
            });
        }
        let added = self.device.allocated_bytes() - before;
        let n = self.slots.len();
        let draft = Mtp {
            pre,
            post,
            head,
            norm,
            factors,
            layers,
            eps: f("attention.layer_norm_rms_epsilon")?,
            rope: [f("rope.freq_base_swa")?, f("rope.freq_base")?],
            pending: self.device.alloc(n * self.width * 4)?,
            cursor: vec![None; n],
            h: self.device.alloc(n * self.width * 4)?,
            concat: self.device.alloc(n * self.width * 8)?,
            bounds: self.device.alloc(n * 2 * 4)?,
            drafted: self.device.alloc(n * spec::BLOCK * 4)?,
        };
        self.ensure_verify()?;
        self.mtp = Some(draft);
        self.weight_bytes += added;
        tracing::info!(
            weight_bytes = added,
            "native Metal Gemma assistant attached; shared target KV, GPU-resident chain"
        );
        Ok(())
    }

    fn assistant_step(&self, cmd: &Commands<'_>, n: usize, length: usize) {
        let d = self.mtp.as_ref().expect("assistant attached");
        let s = &self.scratch;
        cmd.dispatch(
            "embed",
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[
                self.width as u32,
                n as u32,
                self.embedding.ty,
                (self.width as f32).sqrt().to_bits(),
            ],
            [(n * self.width).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "mtp_concat",
            &[&s.x, &d.h, &d.concat],
            &[self.width as u32, n as u32],
            [(n * self.width).div_ceil(256), 1, 1],
            256,
        );
        d.pre.linear(cmd, &d.concat, &s.x, n, 1., &s.gemm);
        cmd.dispatch(
            "rms",
            &[&s.x, &d.layers[0].norm.buffer, &s.norm],
            &[WIDTH as u32, d.layers[0].norm.ty, d.eps.to_bits()],
            [n, 1, 1],
            256,
        );
        let sandwich = |post: &Weight, next: &Weight, scale: f32| {
            cmd.dispatch(
                "gemma_sandwich",
                &[&s.x, &s.delta, &post.buffer, &next.buffer, &s.norm],
                &[
                    WIDTH as u32,
                    post.ty,
                    next.ty,
                    d.eps.to_bits(),
                    scale.to_bits(),
                ],
                [n, 1, 1],
                256,
            );
        };
        for (i, l) in d.layers.iter().enumerate() {
            let target = &self.layers[l.target];
            let hd = target.hd();
            let heads = target.heads;
            l.q.linear(cmd, &s.norm, &s.q, n, 1., &s.gemm);
            cmd.dispatch(
                "gemma_qnorm",
                &[&s.q, &l.q_norm.buffer, &s.meta, &d.factors.buffer],
                &[
                    heads as u32,
                    hd as u32,
                    l.q_norm.ty,
                    d.eps.to_bits(),
                    d.rope[usize::from(!target.sliding)].to_bits(),
                    u32::from(!target.sliding),
                ],
                [heads, n, 1],
                32,
            );
            let visible = if target.sliding {
                length.min(self.window)
            } else {
                length
            };
            let splits = visible
                .div_ceil(128)
                .max(16usize.div_ceil(n))
                .clamp(1, SPLITS);
            let p = [
                heads as u32,
                target.kh() as u32,
                self.page_stride as u32,
                if target.sliding {
                    self.window as u32
                } else {
                    0
                },
                self.ring as u32,
                splits as u32,
            ];
            cmd.dispatch(
                if target.sliding {
                    "gemma_decode256"
                } else {
                    "gemma_decode512"
                },
                &[
                    &s.q,
                    &target.keys,
                    &target.values,
                    &d.bounds,
                    &s.pages,
                    &s.decode_rows,
                    &s.parts,
                ],
                &p,
                [target.kh(), n, splits],
                128,
            );
            cmd.dispatch(
                "gemma_merge_shared",
                &[&s.parts, &s.attn, &s.decode_rows],
                &[heads as u32, splits as u32, hd as u32],
                [heads * n, 1, 1],
                32,
            );
            l.o.linear(cmd, &s.attn, &s.delta, n, 1., &s.gemm);
            sandwich(&l.post_attn, &l.ffn_norm, 1.);
            projections(
                cmd,
                &[(&l.gate, &s.gate), (&l.up, &s.up)],
                &s.norm,
                n,
                &s.gemm,
            );
            cmd.dispatch(
                "gemma_geglu",
                &[&s.gate, &s.up],
                &[(n * FF) as u32],
                [(n * FF).div_ceil(256), 1, 1],
                256,
            );
            l.down.linear(cmd, &s.gate, &s.delta, n, 1., &s.gemm);
            sandwich(
                &l.post_ffn,
                d.layers.get(i + 1).map_or(&d.norm, |l| &l.norm),
                l.scale,
            );
        }
        // Both projections consume the same final normalized hidden. The
        // assistant's vocabulary head has no target logit softcap.
        projections(
            cmd,
            &[(&d.head, &s.logits), (&d.post, &d.h)],
            &s.norm,
            n,
            &s.gemm,
        );
        cmd.dispatch(
            "spec_argmax",
            &[&s.logits, &s.ids],
            &[self.vocab as u32],
            [n, 1, 1],
            256,
        );
    }

    pub(super) fn mtp_draft(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.require_committed()?;
        let Some(d) = &self.mtp else { return Ok(None) };
        let n = pendings.len();
        if n == 0 || n > self.slots.len().min(CHUNK / spec::BLOCK) || k == 0 || k >= spec::BLOCK {
            return Ok(None);
        }
        let mut seen = vec![false; self.slots.len()];
        for &(slot, token) in pendings {
            if slot >= seen.len() || seen[slot] || token as usize >= self.vocab {
                return Err(MetalError::Model("invalid Gemma draft slot/token".into()));
            }
            seen[slot] = true;
            let len = self.slots[slot].history.len();
            if len == 0
                || len + k + 1 > self.context
                || d.cursor[slot] != Some(len)
                || self.pending.iter().any(|p| p.slot == slot)
            {
                return Ok(None);
            }
        }
        let mut pages = vec![0; self.slots.len() * self.page_stride];
        for (i, slot) in self.slots.iter().enumerate() {
            let b = slot.table.blocks();
            pages[i * self.page_stride..i * self.page_stride + b.len()].copy_from_slice(b);
        }
        let s = &self.scratch;
        unsafe {
            s.ids
                .write_u32(&pendings.iter().map(|p| p.1).collect::<Vec<_>>());
            s.meta.write_u32(
                &pendings
                    .iter()
                    .flat_map(|p| [p.0 as u32, self.slots[p.0].history.len() as u32])
                    .collect::<Vec<_>>(),
            );
            d.bounds.write_u32(
                &pendings
                    .iter()
                    .flat_map(|p| [p.0 as u32, self.slots[p.0].history.len() as u32 - 1])
                    .collect::<Vec<_>>(),
            );
            s.decode_rows.write_u32(&(0..n as u32).collect::<Vec<_>>());
            s.pages.write_u32(&pages);
        }
        let cmd = self.device.begin()?;
        for (i, &(slot, _)) in pendings.iter().enumerate() {
            copy_words(
                &cmd,
                &d.pending,
                &d.h,
                slot * self.width,
                i * self.width,
                self.width,
            );
        }
        let length = pendings
            .iter()
            .map(|p| self.slots[p.0].history.len())
            .max()
            .expect("nonempty");
        for step in 0..k {
            self.assistant_step(&cmd, n, length);
            cmd.dispatch(
                "gemma_draft_advance",
                &[&s.ids, &s.meta, &d.drafted],
                &[n as u32, step as u32, k as u32],
                [n.div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.finish()?;
        Ok(Some(
            unsafe { d.drafted.read_u32(n * k) }
                .chunks(k)
                .map(|r| r.to_vec())
                .collect(),
        ))
    }
}
