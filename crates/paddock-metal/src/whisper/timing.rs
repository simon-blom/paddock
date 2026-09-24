//! Native teacher-forced alignment. Only selected heads are materialized, only
//! over real audio frames, and only for a request that asks for word times.
//! Shared CUDA/Metal postprocessing defines the exact token-boundary contract.
use super::forward::{attention, linear, mlp, norm, plain};
use super::*;
use paddock_engine::whisper::align;

const CHUNK: usize = 32;

pub(super) fn reservation_bytes(ctx: usize) -> u64 {
    let heads = align::heads_for(L, 20, 128, V)
        .expect("supported large-v3 alignment")
        .heads
        .len();
    let capture = heads * (ctx + 1) * T * 4 + heads * 4;
    let scratch = CHUNK * (7 * D * 4 + FF * 4 + 20 * 6 * 66 * 4 + 3 * 4) + 7 * 4;
    (capture + scratch) as u64
}

struct LayerCapture {
    ids: Buffer,
    probs: Buffer,
    heads: usize,
}

impl Scratch {
    /// Decoder-only scratch. No encoder-sized workspace or vocabulary logits:
    /// teacher forcing already knows every token it will feed.
    fn for_alignment(d: &MetalDevice, rows: usize) -> Result<Self> {
        Ok(Self {
            x: d.alloc(rows * D * 4)?,
            norm: d.alloc(rows * D * 4)?,
            qkv: d.alloc(rows * 3 * D * 4)?,
            q: d.alloc(rows * D * 4)?,
            attn: d.alloc(rows * D * 4)?,
            up: d.alloc(rows * FF * 4)?,
            partial: d.alloc(rows * 20 * 6 * 66 * 4)?,
            slots: d.alloc(rows * 4)?,
            positions: d.alloc(rows * 4)?,
            tokens: d.alloc(rows * 4)?,
            k: d.alloc(4)?,
            v: d.alloc(4)?,
            conv: d.alloc(4)?,
            logits: d.alloc(4)?,
            rules: d.alloc(4)?,
            pick: d.alloc(4)?,
            stats: d.alloc(4)?,
        })
    }
}

impl Whisper {
    pub(super) fn align_tokens(
        &mut self,
        slot: usize,
        lang: u32,
        tokens: &[u32],
        samples: usize,
    ) -> Result<Vec<f32>> {
        self.alignment_input(slot, lang, tokens, samples)?;
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let start = std::time::Instant::now();
        let (mut weights, heads, rows, frames) =
            self.capture_alignment(slot, lang, tokens, samples, CHUNK)?;
        let capture_ms = start.elapsed().as_secs_f64() * 1000.;
        let post = std::time::Instant::now();
        align::normalize_over_tokens(&mut weights, heads, rows, frames);
        align::median_filter_rows(&mut weights, heads * rows, frames, 7);
        let cost = align::mean_heads_negated(&weights, heads, rows, frames);
        let path = align::dtw_path(&cost, rows, frames);
        let duration = samples as f32 / 16000.;
        let boundaries: Vec<_> = align::token_start_frames(&path, rows)
            .into_iter()
            .map(|f| (f as f32 * align::SECONDS_PER_FRAME).min(duration))
            .collect();
        if boundaries.len() != tokens.len() + 1
            || boundaries.iter().any(|v| !v.is_finite())
            || boundaries.windows(2).any(|p| p[0] > p[1])
        {
            return Err(error("invalid word alignment boundaries"));
        }
        tracing::debug!(
            capture_ms,
            post_ms = post.elapsed().as_secs_f64() * 1000.,
            heads,
            rows,
            frames,
            "Whisper Metal word alignment complete"
        );
        Ok(boundaries)
    }

    fn alignment_input(
        &self,
        slot: usize,
        lang: u32,
        tokens: &[u32],
        samples: usize,
    ) -> Result<()> {
        if slot >= self.capacity || self.lengths[slot].is_none() {
            return Err(error("word alignment requires an admitted encoder slot"));
        }
        if !self.langs.iter().any(|(_, id)| *id == lang) {
            return Err(error("word alignment language is not in the checkpoint"));
        }
        if tokens.len() > self.ctx - 4 || tokens.iter().any(|t| *t >= 50257) {
            return Err(error(
                "word alignment needs text-only tokens within decoder context",
            ));
        }
        if samples > 480000 || (samples == 0 && !tokens.is_empty()) {
            return Err(error("word alignment requires 1..480000 audio samples"));
        }
        Ok(())
    }

    /// The input position 3 (<|notimestamps|>) predicts the first text token.
    /// Capture it and every text input: N tokens yield N+1 boundaries. The
    /// method resets only this slot's decoder; encoder planes and other slots
    /// are unchanged. Every allocation is request-owned and released on return.
    pub(super) fn capture_alignment(
        &mut self,
        slot: usize,
        lang: u32,
        tokens: &[u32],
        samples: usize,
        chunk: usize,
    ) -> Result<(Vec<f32>, usize, usize, usize)> {
        self.alignment_input(slot, lang, tokens, samples)?;
        if !(1..=64).contains(&chunk) {
            return Err(error("invalid alignment chunk"));
        }
        let mut seq = vec![50258, lang, 50360, 50364];
        seq.extend_from_slice(tokens);
        let rows = tokens.len() + 1;
        let frames = samples.div_ceil(320).clamp(1, T);
        let mut selected = align::heads_for(L, 20, 128, V)
            .ok_or_else(|| error("no alignment head selection for loaded geometry"))?
            .heads;
        selected.sort_unstable();
        let last_layer = selected.last().expect("large-v3 heads").0;
        let mut captures = Vec::with_capacity(L);
        for li in 0..L {
            let ids: Vec<u32> = selected
                .iter()
                .filter(|(layer, _)| *layer == li)
                .map(|(_, h)| *h as u32)
                .collect();
            captures.push(if ids.is_empty() {
                None
            } else {
                Some(LayerCapture {
                    ids: upload(&self.device, &ids)?,
                    probs: self.device.alloc(ids.len() * rows * frames * 4)?,
                    heads: ids.len(),
                })
            });
        }
        let s = Scratch::for_alignment(&self.device, chunk.min(seq.len()))?;
        // Aligning is destructive to this slot's text cache, not its audio. A
        // subsequent normal decode must start again at position zero. Invalid
        // arguments/allocation failures above leave all live slots untouched.
        self.lengths[slot] = Some(0);
        self.last_rows = 0;
        for p0 in (0..seq.len()).step_by(chunk) {
            let p1 = (p0 + chunk).min(seq.len());
            let n = p1 - p0;
            // SAFETY: previous chunk completed, all arrays fit decoder scratch.
            unsafe {
                s.slots.write_u32(&vec![slot as u32; n]);
                s.positions
                    .write_u32(&(p0 as u32..p1 as u32).collect::<Vec<_>>());
                s.tokens.write_u32(&seq[p0..p1]);
            }
            let c = self.device.begin()?;
            c.dispatch(
                "wh_embed",
                &[
                    &self.embedding,
                    &self.dec_pos,
                    &s.tokens,
                    &s.positions,
                    &s.x,
                ],
                &[n as u32],
                [(n * D).div_ceil(256), 1, 1],
                256,
            );
            for (li, (layer, cache)) in self.dec.iter().zip(&self.cache).enumerate() {
                norm(&c, &layer.attn.norm, &s.x, &s.norm, n);
                plain(
                    &c,
                    &layer.attn.qkv,
                    &layer.attn.bias,
                    &s.norm,
                    &s.qkv,
                    D,
                    3 * D,
                    n,
                    0,
                );
                c.dispatch(
                    "wh_append",
                    &[
                        &s.qkv,
                        &layer.attn.bias,
                        &s.q,
                        &cache.k,
                        &cache.v,
                        &s.slots,
                        &s.positions,
                    ],
                    &[n as u32, self.ctx as u32],
                    [(n * D).div_ceil(256), 1, 1],
                    256,
                );
                attention(
                    &c,
                    &s,
                    &cache.k,
                    &cache.v,
                    self.ctx,
                    n,
                    p1.div_ceil(256),
                    false,
                );
                linear(&c, &layer.attn.out, &s.attn, &s.x, n, 2);
                norm(&c, &layer.cross_norm, &s.x, &s.norm, n);
                linear(&c, &layer.q, &s.norm, &s.q, n, 1);
                if let Some(capture) = &captures[li] {
                    c.dispatch(
                        "wh_align_probs",
                        &[&s.q, &cache.ck, &capture.ids, &capture.probs],
                        &[slot as u32, frames as u32, rows as u32, p0 as u32],
                        [capture.heads, n, 1],
                        256,
                    );
                }
                // No logits or layers after the last selected head are needed.
                if li == last_layer {
                    break;
                }
                attention(&c, &s, &cache.ck, &cache.cv, T, n, 6, true);
                linear(&c, &layer.out, &s.attn, &s.x, n, 2);
                mlp(&c, &layer.mlp, &s, n);
            }
            c.finish()?;
        }
        let mut weights = Vec::with_capacity(selected.len() * rows * frames);
        for capture in captures.iter().flatten() {
            // SAFETY: every chunk is complete and each captured row was written.
            weights.extend(unsafe { capture.probs.read_f32(0, capture.heads * rows * frames) });
        }
        if weights.iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(error("nonfinite alignment attention"));
        }
        Ok((weights, selected.len(), rows, frames))
    }
}
