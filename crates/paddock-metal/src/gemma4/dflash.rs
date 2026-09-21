//! Muse Glimmer DFlash2, using Paddock's original Qwen-family draft primitives.
//! The target's raw residual taps enter layers [2,14,26,38,50] in GGUF
//! indexing. Sixteen noncausal noise rows share an anchor and target head;
//! dynamic convolutions and the rank-256 selector are part of the graph.
//! Conditioning rings share the target's snapshot/slack coverage. Speculative
//! suffixes remain masked above the committed cursor and never publish a prefix.
use super::*;
use objc2_metal::MTLBuffer;
use paddock_models::{gguf::Value, mapped::MappedGguf};

// A model-local wide projection election. Compact K-quant staging fits
// Apple10's 32 KiB threadgroup limit under validation; no operand or K-tile
// changes. Narrow drafts retain their measured shared ladder. Prepare the
// identical input once for independent K/V or gate/up consumers.
fn draft_project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
) {
    if rows <= 32
        || !planes
            .iter()
            .any(|(w, _)| matches!(w.ty, 12 | 13 | 14 | 23))
    {
        if planes.len() == 1 {
            planes[0]
                .0
                .linear(cmd, input, planes[0].1, rows, 1., workspace);
        } else {
            projections(cmd, planes, input, rows, workspace);
        }
        return;
    }
    let k = planes[0].0.k;
    assert!(planes.iter().all(|(w, _)| w.k == k));
    cmd.dispatch(
        "linear_input_padded",
        &[input, workspace],
        &[k as u32, 0, rows as u32],
        [
            (k.div_ceil(128) * 128 * rows.div_ceil(128) * 128).div_ceil(256),
            1,
            1,
        ],
        256,
    );
    for &(w, out) in planes {
        if matches!(w.ty, 12 | 13 | 14 | 23) {
            // Preserve the shared ladder's shape, including its single-
            // projection small-N exception and multi-projection BM96 rung.
            if !(65..=96).contains(&rows) && rows <= 128 && planes.len() == 1 && w.n <= 1024 {
                w.linear_prepared(cmd, workspace, out, rows, 1.);
                continue;
            }
            let tile = if (65..=96).contains(&rows) {
                96
            } else if rows <= 64 || (rows <= 128 && planes.len() == 1 && w.n <= 4096) {
                64
            } else {
                128
            };
            cmd.dispatch(
                match tile {
                    64 => "muse_df_ktile64",
                    96 => "muse_df_ktile96",
                    _ => "muse_df_ktile128",
                },
                &[&w.buffer, workspace, out],
                &[k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()],
                [w.n.div_ceil(32), rows.div_ceil(tile), 1],
                128,
            );
        } else {
            w.linear(cmd, input, out, rows, 1., workspace);
        }
    }
}
const DH: usize = 32;
const DK: usize = 8;
const DIM: usize = 128;
const TAPS: [usize; 5] = [2, 14, 26, 38, 50];
pub(super) const BLOCK: usize = 16;
#[cfg(test)]
#[path = "dflash_tests.rs"]
mod tests;

struct Conv {
    base: Weight,
    proj: Weight,
}
struct DraftLayer {
    norm: Weight,
    post: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    qn: Weight,
    kn: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    ac: Conv,
    fc: Conv,
    keys: Buffer,
    values: Buffer,
}
pub(super) struct Dflash {
    pub(super) budget: super::dflash_budget::Budget,
    fc: Weight,
    enc_norm: Weight,
    out_norm: Weight,
    selector: Weight,
    pred: Weight,
    succ: Weight,
    layers: Vec<DraftLayer>,
    pages: Buffer,
    positions: Buffer,
    bounds: Buffer,
    tiles: Buffer,
    ring_tokens: usize,
    pub(super) taps: Buffer,
    conditioning: Buffer,
    z: Buffer,
    x: Buffer,
    norm: Buffer,
    delta: Buffer,
    conv: Buffer,
    coeff: Buffer,
    q: Buffer,
    qn: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    gate: Buffer,
    up: Buffer,
    gemm: Buffer,
    logits: Buffer,
    top: Buffer,
    top_parts: Buffer,
    selector_h: Buffer,
    out: Buffer,
}

impl Gemma4 {
    /// The registered upstream GGUF retains both v2 convolutions and selector.
    /// Reject incompatible geometry/version; never run these weights as v1.
    pub fn attach_dflash(&mut self, path: &Path) -> Result<()> {
        if self.mlx {
            return Err(MetalError::Model(
                "MLX Muse DFlash attachment requires separate checkpoint qualification".into(),
            ));
        }
        self.require_committed()?;
        if !self.muse
            || self.mtp.is_some()
            || self.dflash.is_some()
            || !self.pending.is_empty()
            || !self.encoding.is_empty()
            || self.prefill_phase.is_some()
            || self.slots.iter().any(|s| !s.history.is_empty())
            || self.cache.iter().any(|s| !s.history.is_empty())
        {
            return Err(MetalError::Model(
                "attach DFlash2 once, before prefill".into(),
            ));
        }
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        if map.gguf().architecture() != Some("dflash") {
            return Err(MetalError::Model("expected DFlash2 GGUF".into()));
        }
        for (key, want) in [
            ("embedding_length", self.width),
            ("feed_forward_length", self.ff),
            ("block_count", 5),
            ("attention.head_count", DH),
            ("attention.head_count_kv", DK),
            ("attention.key_length", DIM),
            ("attention.value_length", DIM),
            ("attention.sliding_window", 2048),
            ("block_size", BLOCK),
            ("conv_kernel_size", 2),
            ("conv_group_size", 16),
            ("selector_rank", 256),
            ("selector_top_k", 16),
        ] {
            if map.gguf().arch_field(key).and_then(Value::as_u64) != Some(want as u64) {
                return Err(MetalError::Model(format!("incompatible DFlash2 {key}")));
            }
        }
        let taps = match map.gguf().arch_field("target_layers") {
            Some(Value::Array(v)) => v.iter().map(Value::as_u64).collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        if taps != TAPS.iter().map(|&n| Some(n as u64)).collect::<Vec<_>>()
            || !matches!(
                map.gguf().arch_field("attention.causal"),
                Some(Value::Bool(false))
            )
            || map
                .gguf()
                .arch_field("rope.freq_base")
                .and_then(Value::as_f32)
                != Some(500_000.)
            || map
                .gguf()
                .arch_field("attention.layer_norm_rms_epsilon")
                .and_then(Value::as_f32)
                != Some(self.eps)
            || map
                .gguf()
                .metadata
                .get("tokenizer.ggml.mask_token_id")
                .and_then(Value::as_u64)
                != Some(201818)
            || map.gguf().arch_field("logit_scale").and_then(Value::as_f32)
                != Some(self.logit_scale)
            || map
                .gguf()
                .arch_field("final_logit_softcapping")
                .and_then(Value::as_f32)
                != Some(self.softcap)
            || map
                .gguf()
                .arch_field("context_length")
                .and_then(Value::as_u64)
                .is_none_or(|n| n < self.context as u64)
            || !matches!(map.gguf().arch_field("attention.sliding_window_pattern"),
                Some(Value::Array(v)) if v.len()==5 && v.iter().all(|v|matches!(v,Value::Bool(true))))
        {
            return Err(MetalError::Model(
                "incompatible DFlash2 taps/causality/rotary/mask".into(),
            ));
        }
        let w = |n: &str, d: &[usize]| Weight::load(&self.device, &map, n, d);
        let norm = |n: &str, d: &[usize]| -> Result<Weight> {
            let v = w(n, d)?;
            if v.ty != 0 {
                return Err(MetalError::Model(format!("DFlash2 {n} must be F32")));
            }
            Ok(v)
        };
        let before = self.device.allocated_bytes();
        // Match the target ring's entire reusable interval. A shorter draft
        // ring would make a target-valid prefix restore silently stale.
        let ring_pages = self.ring / BLOCK_TOKENS;
        let ring_tokens = ring_pages * BLOCK_TOKENS;
        let kv = (self.slots.len() * 2) * ring_tokens * DK * DIM * 2;
        let mut layers = Vec::new();
        for i in 0..5 {
            let p = |n: &str| format!("blk.{i}.{n}");
            let cv = |n: &str| -> Result<Conv> {
                Ok(Conv {
                    base: norm(&p(&format!("{n}_base")), &[self.width, 2, 2])?,
                    proj: w(
                        &p(&format!("{n}_proj.weight")),
                        &[self.width, self.width / 4],
                    )?,
                })
            };
            layers.push(DraftLayer {
                norm: norm(&p("attn_norm.weight"), &[self.width])?,
                post: norm(&p("ffn_norm.weight"), &[self.width])?,
                q: w(&p("attn_q.weight"), &[self.width, DH * DIM])?,
                k: w(&p("attn_k.weight"), &[self.width, DK * DIM])?,
                v: w(&p("attn_v.weight"), &[self.width, DK * DIM])?,
                o: w(&p("attn_output.weight"), &[DH * DIM, self.width])?,
                qn: norm(&p("attn_q_norm.weight"), &[DIM])?,
                kn: norm(&p("attn_k_norm.weight"), &[DIM])?,
                gate: w(&p("ffn_gate.weight"), &[self.width, self.ff])?,
                up: w(&p("ffn_up.weight"), &[self.width, self.ff])?,
                down: w(&p("ffn_down.weight"), &[self.ff, self.width])?,
                ac: cv("attn_conv")?,
                fc: cv("ffn_conv")?,
                keys: self.device.alloc(kv)?,
                values: self.device.alloc(kv)?,
            });
        }
        let fc = w("fc.weight", &[self.width * 5, self.width])?;
        let enc_norm = norm("enc.output_norm.weight", &[self.width])?;
        let out_norm = norm("output_norm.weight", &[self.width])?;
        let selector = w("selector_hidden.weight", &[self.width, 256])?;
        let pred = w("selector_predecessor.weight", &[256, self.vocab])?;
        let succ = w("selector_successor.weight", &[256, self.vocab])?;
        if !matches!(pred.ty, 12 | 13 | 14 | 23) || !matches!(succ.ty, 12 | 13 | 14 | 23) {
            return Err(MetalError::Model(
                "DFlash2 selector requires exact K-quant codebooks".into(),
            ));
        }
        let added = self.device.allocated_bytes() - before - kv as u64 * 10;
        self.ensure_verify()?;
        let rows = self
            .spec
            .as_ref()
            .expect("verification buffers allocated")
            .rows;
        let a = |n| self.device.alloc(CHUNK * n * 4);
        // Conditioning follows the target's image-prefill capacity; noise
        // blocks and verification keep their separate, bounded row grant.
        let c = |n| self.device.alloc(self.scratch.rows * n * 4);
        let pages = self.device.upload(
            &(0..self.slots.len())
                .flat_map(|s| {
                    (0..self.page_stride)
                        .flat_map(move |p| ((s * ring_pages + p % ring_pages) as u32).to_le_bytes())
                })
                .collect::<Vec<_>>(),
        )?;
        self.dflash = Some(Dflash {
            budget: super::dflash_budget::Budget::default(),
            fc,
            enc_norm,
            out_norm,
            selector,
            pred,
            succ,
            layers,
            pages,
            positions: c(4)?,
            bounds: c(2)?,
            tiles: self.device.alloc(CHUNK * 2 * 4)?,
            ring_tokens,
            taps: c(self.width * 5)?,
            conditioning: c(self.width)?,
            z: c(self.width)?,
            x: a(self.width)?,
            norm: a(self.width)?,
            delta: a(self.width)?,
            conv: a(self.width)?,
            coeff: a(self.width / 4)?,
            q: a(DH * DIM)?,
            qn: a(DH * DIM)?,
            k: c(DK * DIM)?,
            v: c(DK * DIM)?,
            attn: a(DH * DIM)?,
            gate: a(self.ff)?,
            up: a(self.ff)?,
            gemm: self.device.alloc(self.scratch.rows * self.width * 5 * 2)?,
            logits: self.device.alloc(rows * self.vocab * 4)?,
            top: self.device.alloc(rows * 16 * 8)?,
            top_parts: self
                .device
                .alloc(rows * self.vocab.div_ceil(4096) * 16 * 8)?,
            selector_h: self.device.alloc(rows * 256 * 4)?,
            out: self.device.alloc(rows * 4)?,
        });
        self.weight_bytes += added;
        self.kv_bytes += kv as u64 * 10;
        tracing::info!(
            weight_bytes = added,
            kv_bytes = kv * 10,
            "native Metal Muse DFlash2 attached; block=16, conv=2, selector rank=256/top16"
        );
        Ok(())
    }

    pub(super) fn dflash_checkpoint(&self, cmd: &Commands<'_>, from: usize, to: usize) {
        let Some(d) = &self.dflash else { return };
        let words = d.ring_tokens * DK * DIM / 2;
        for layer in &d.layers {
            for b in [&layer.keys, &layer.values] {
                cmd.dispatch(
                    "spec_copy_words",
                    &[b, b],
                    &[(from * words) as u32, (to * words) as u32, words as u32],
                    [words.div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }

    pub(super) fn dflash_rows(&self, rows: &[(usize, u32, u32)]) {
        let Some(d) = &self.dflash else { return };
        let mut bounds = Vec::with_capacity(rows.len() * 2);
        let mut first = 0;
        while first < rows.len() {
            let mut end = first + 1;
            while end < rows.len() && rows[end].0 == rows[first].0 {
                end += 1;
            }
            for _ in first..end {
                bounds.extend([first as u32, end as u32]);
            }
            first = end;
        }
        // The preceding submission completed; only the engine thread owns
        // these uploads. Image positions are causal language-token positions.
        unsafe {
            d.positions
                .write_u32(&rows.iter().flat_map(|r| [r.2; 4]).collect::<Vec<_>>());
            d.bounds.write_u32(&bounds);
        }
    }

    pub(super) fn dflash_tap(&self, cmd: &Commands<'_>, layer: usize, m: usize) {
        if let Some(d) = &self.dflash
            && let Some(tap) = TAPS.iter().position(|&n| n == layer)
        {
            cmd.dispatch(
                "df_tap",
                &[&self.scratch.x, &d.taps],
                &[self.width as u32, m as u32, tap as u32],
                [(m * self.width).div_ceil(256), 1, 1],
                256,
            );
        }
    }
    pub(super) fn dflash_append(&self, cmd: &Commands<'_>, m: usize) {
        let Some(d) = &self.dflash else { return };
        let s = &self.scratch;
        draft_project(cmd, &[(&d.fc, &d.conditioning)], &d.taps, m, &d.gemm);
        cmd.dispatch(
            "rms",
            &[&d.conditioning, &d.enc_norm.buffer, &d.z],
            &[self.width as u32, 0, self.eps.to_bits()],
            [m, 1, 1],
            256,
        );
        for w in &d.layers {
            draft_project(cmd, &[(&w.k, &d.k), (&w.v, &d.v)], &d.z, m, &d.gemm);
            cmd.dispatch(
                "df_kstore",
                &[
                    &d.k,
                    &d.v,
                    &w.kn.buffer,
                    &s.meta,
                    &d.pages,
                    &w.keys,
                    &w.values,
                    &d.positions,
                    &d.bounds,
                ],
                &[
                    DK as u32,
                    self.page_stride as u32,
                    self.eps.to_bits(),
                    500_000f32.to_bits(),
                    d.ring_tokens as u32,
                    0,
                ],
                [DK, m, 1],
                32,
            );
        }
    }

    pub(super) fn dflash_draft(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.require_committed()?;
        if self.dflash.is_none() || pendings.is_empty() || k == 0 {
            return Ok(None);
        }
        // Even when the consumer takes a shorter prefix (adaptive policy or
        // a mixed budget), all predictions see the trained noncausal block.
        // Shortening noise input changes every position's prediction.
        let block = BLOCK;
        let take = k.min(BLOCK - 1);
        let rows = pendings.len() * block;
        if rows
            > self
                .spec
                .as_ref()
                .expect("verification buffers allocated")
                .rows
        {
            return Ok(None);
        }
        let mut seen = vec![false; self.slots.len()];
        for &(slot, t) in pendings {
            if slot >= seen.len()
                || self.pending.iter().any(|p| p.slot == slot)
                || self.image_slot_pending(slot)
                || seen[slot]
                || t as usize >= self.vocab
                || self.slots[slot].history.is_empty()
                || self.slots[slot].history.len() + block > self.context
            {
                return Ok(None);
            }
            seen[slot] = true;
        }
        let s = &self.scratch;
        let d = self.dflash.as_ref().expect("DFlash2 attached");
        let mut tokens = Vec::new();
        let mut meta = Vec::new();
        let mut tiles = Vec::new();
        for (i, &(slot, t)) in pendings.iter().enumerate() {
            tiles.extend([(i * block) as u32, block as u32]);
            for j in 0..block {
                tokens.push(if j == 0 { t } else { 201818 });
                meta.extend([slot as u32, (self.slots[slot].history.len() + j) as u32]);
            }
        }
        unsafe {
            s.ids.write_u32(&tokens);
            s.meta.write_u32(&meta);
            d.tiles.write_u32(&tiles);
            d.positions.write_u32(
                &meta
                    .chunks_exact(2)
                    .flat_map(|r| [r[1]; 4])
                    .collect::<Vec<_>>(),
            );
        }
        let cmd = self.device.begin()?;
        cmd.dispatch(
            "embed",
            &[&self.embedding.buffer, &s.ids, &d.x],
            &[
                self.width as u32,
                rows as u32,
                self.embedding.ty,
                1f32.to_bits(),
            ],
            [(rows * self.width).div_ceil(256), 1, 1],
            256,
        );
        let conv = |cv: &Conv, input: &Buffer, side: u32| {
            cmd.dispatch(
                "df_conv",
                &[input, &cv.base.buffer, &d.coeff, &d.conv],
                &[self.width as u32, rows as u32, block as u32, side],
                [(rows * self.width / 4).div_ceil(256), 1, 1],
                256,
            )
        };
        for w in &d.layers {
            cmd.dispatch(
                "rms",
                &[&d.x, &w.norm.buffer, &d.norm],
                &[self.width as u32, 0, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            draft_project(&cmd, &[(&w.ac.proj, &d.coeff)], &d.norm, rows, &d.gemm);
            conv(&w.ac, &d.norm, 0);
            draft_project(
                &cmd,
                &[(&w.q, &d.q), (&w.k, &d.k), (&w.v, &d.v)],
                &d.conv,
                rows,
                &d.gemm,
            );
            cmd.dispatch(
                "df_qnorm",
                &[&d.q, &w.qn.buffer, &d.positions, &d.qn],
                &[
                    DH as u32,
                    self.page_stride as u32,
                    self.eps.to_bits(),
                    500_000f32.to_bits(),
                ],
                [DH, rows, 1],
                32,
            );
            cmd.dispatch(
                "df_kstore",
                &[
                    &d.k,
                    &d.v,
                    &w.kn.buffer,
                    &s.meta,
                    &d.pages,
                    &w.keys,
                    &w.values,
                    &d.positions,
                    &d.bounds,
                ],
                &[
                    DK as u32,
                    self.page_stride as u32,
                    self.eps.to_bits(),
                    500_000f32.to_bits(),
                    d.ring_tokens as u32,
                    1,
                ],
                [DK, rows, 1],
                32,
            );
            cmd.dispatch(
                "attention_query",
                &[&d.qn, &d.gemm],
                &[(DH * DIM) as u32, 0, rows as u32],
                [((rows + 32) * DH * DIM).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "df_attention_muse",
                &[
                    &d.gemm, &w.keys, &w.values, &s.meta, &d.pages, &d.attn, &d.tiles,
                ],
                &[
                    DH as u32,
                    DK as u32,
                    self.page_stride as u32,
                    (1f32 / (DIM as f32).sqrt()).to_bits(),
                ],
                [DH, pendings.len(), 1],
                128,
            );
            draft_project(&cmd, &[(&w.o, &d.delta)], &d.attn, rows, &d.gemm);
            conv(&w.ac, &d.delta, 1);
            cmd.dispatch(
                "residual",
                &[&d.x, &d.conv],
                &[(rows * self.width) as u32, 1f32.to_bits()],
                [(rows * self.width).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "rms",
                &[&d.x, &w.post.buffer, &d.norm],
                &[self.width as u32, 0, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            draft_project(&cmd, &[(&w.fc.proj, &d.coeff)], &d.norm, rows, &d.gemm);
            conv(&w.fc, &d.norm, 0);
            draft_project(
                &cmd,
                &[(&w.gate, &d.gate), (&w.up, &d.up)],
                &d.conv,
                rows,
                &d.gemm,
            );
            cmd.dispatch(
                "swiglu",
                &[&d.gate, &d.up],
                &[(rows * self.ff) as u32],
                [(rows * self.ff).div_ceil(256), 1, 1],
                256,
            );
            draft_project(&cmd, &[(&w.down, &d.delta)], &d.gate, rows, &d.gemm);
            conv(&w.fc, &d.delta, 1);
            cmd.dispatch(
                "residual",
                &[&d.x, &d.conv],
                &[(rows * self.width) as u32, 1f32.to_bits()],
                [(rows * self.width).div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.dispatch(
            "rms",
            &[&d.x, &d.out_norm.buffer, &d.norm],
            &[self.width as u32, 0, self.eps.to_bits()],
            [rows, 1, 1],
            256,
        );
        self.output
            .as_ref()
            .expect("Muse has an untied target head")
            .linear(&cmd, &d.norm, &d.logits, rows, 1., &d.gemm);
        cmd.dispatch(
            "muse_softcap",
            &[&d.logits],
            &[
                (rows * self.vocab) as u32,
                self.softcap.to_bits(),
                self.logit_scale.to_bits(),
            ],
            [(rows * self.vocab).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "df_top16",
            &[&d.logits, &d.top_parts],
            &[self.vocab as u32],
            [self.vocab.div_ceil(4096), rows, 1],
            256,
        );
        cmd.dispatch(
            "df_top16_merge",
            &[&d.top_parts, &d.top],
            &[(self.vocab.div_ceil(4096) * 16) as u32],
            [rows, 1, 1],
            256,
        );
        draft_project(
            &cmd,
            &[(&d.selector, &d.selector_h)],
            &d.norm,
            rows,
            &d.gemm,
        );
        cmd.dispatch(
            "df_select",
            &[
                &d.top,
                &d.pred.buffer,
                &d.succ.buffer,
                &d.selector_h,
                &s.ids,
                &d.out,
            ],
            &[block as u32, d.pred.ty, d.succ.ty],
            [pendings.len(), 1, 1],
            256,
        );
        cmd.finish()?;
        // SAFETY: complete block chain, one compact token readback.
        let out = unsafe {
            std::slice::from_raw_parts(d.out.raw.contents().as_ptr().cast::<u32>(), rows).to_vec()
        };
        Ok(Some(
            out.chunks(block).map(|r| r[1..=take].to_vec()).collect(),
        ))
    }
}
