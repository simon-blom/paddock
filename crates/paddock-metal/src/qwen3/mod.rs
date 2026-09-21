//! Qwen3 dense retrieval models, native GGUF Q8 on Apple10. Ragged causal
//! TensorOps attention, shared projection input conversions and GPU-only heads.
//! One scratch arena is queue-ordered across submissions; immutable metadata
//! and result buffers belong to each completion. No per-text forward loop.
mod load;
#[cfg(test)]
mod tests;

use crate::device::{Buffer, Completion, MetalDevice, MetalError, Result};
use crate::weights::{Weight, projections};
use paddock_engine::encoder::EncoderBackend;

struct Layer {
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    qnorm: Weight,
    knorm: Weight,
    o: Weight,
    ffn_norm: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}
struct Scratch {
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qh: Buffer,
    kh: Buffer,
    vh: Buffer,
    attn: Buffer,
    delta: Buffer,
    gate: Buffer,
    up: Buffer,
    gemm: Buffer,
}
pub struct Qwen3Encoder {
    identity: std::rc::Rc<()>,
    device: MetalDevice,
    embedding: Weight,
    output_norm: Weight,
    head: Option<Weight>,
    layers: Vec<Layer>,
    scratch: Scratch,
    width: usize,
    heads: usize,
    kv_heads: usize,
    ff: usize,
    vocab: usize,
    context: usize,
    row_budget: usize,
    eps: f32,
    rope: f32,
    weight_bytes: u64,
}

pub struct PendingEncoding {
    // Completion drops first and fences all GPU use, including the result.
    completion: Completion,
    identity: std::rc::Rc<()>,
    output: Buffer,
    n: usize,
    score: Option<(u32, u32)>,
    // Command-buffer retention alone is insufficient for honest accounting:
    // retain the Buffer wrappers as well until this result is collected.
    _inputs: Vec<Buffer>,
}

impl Qwen3Encoder {
    fn submit(
        &mut self,
        seqs: &[Vec<u32>],
        score: Option<(u32, u32)>,
        lane: usize,
    ) -> Result<PendingEncoding> {
        if lane != 0 || seqs.is_empty() {
            return Err(MetalError::Model(
                "empty encoder batch or invalid lane".into(),
            ));
        }
        if let Some((y, n)) = score
            && (y as usize >= self.vocab || n as usize >= self.vocab || y == n)
        {
            return Err(MetalError::Model("invalid yes/no head token ids".into()));
        }
        let rows = seqs
            .iter()
            .try_fold(0usize, |a, s| a.checked_add(s.len()))
            .ok_or_else(|| MetalError::Model("encoder batch size overflow".into()))?;
        if rows > self.row_budget
            || seqs.iter().any(|s| {
                s.is_empty()
                    || s.len() > self.context
                    || s.iter().any(|&t| t as usize >= self.vocab)
            })
        {
            return Err(MetalError::Model(format!(
                "encoder input exceeds context {}, batch rows {}, or vocabulary bounds (empty sequences are invalid)",
                self.context, self.row_budget
            )));
        }
        let mut ids = Vec::with_capacity(rows);
        let mut meta = Vec::with_capacity(2 * rows);
        let mut tiles = Vec::new();
        let mut last = Vec::with_capacity(seqs.len());
        for seq in seqs {
            let start = ids.len();
            ids.extend_from_slice(seq);
            for pos in 0..seq.len() {
                meta.extend_from_slice(&[start as u32, pos as u32]);
            }
            for pos in (0..seq.len()).step_by(32) {
                tiles.extend_from_slice(&[(start + pos) as u32, (seq.len() - pos).min(32) as u32]);
            }
            last.push((ids.len() - 1) as u32);
        }
        let upload = |words: &[u32]| -> Result<Buffer> {
            let b = self.device.alloc(std::mem::size_of_val(words))?;
            // Fresh buffer, never submitted or aliased by another pending job.
            unsafe {
                b.write_u32(words);
            }
            Ok(b)
        };
        let ids = upload(&ids)?;
        let meta = upload(&meta)?;
        let nt = tiles.len() / 2;
        let tiles = upload(&tiles)?;
        let last = upload(&last)?;
        let pooled = self.device.alloc(seqs.len() * self.width * 4)?;
        let scores = score
            .map(|_| self.device.alloc(seqs.len() * 4))
            .transpose()?;
        let sc = &self.scratch;
        let cmd = self.device.begin()?;
        cmd.dispatch(
            "embed",
            &[&self.embedding.buffer, &ids, &sc.x],
            &[
                self.width as u32,
                rows as u32,
                self.embedding.ty,
                1f32.to_bits(),
            ],
            [(rows * self.width).div_ceil(256), 1, 1],
            256,
        );
        for l in &self.layers {
            cmd.dispatch(
                "rms",
                &[&sc.x, &l.norm.buffer, &sc.norm],
                &[self.width as u32, l.norm.ty, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            projections(
                &cmd,
                &[(&l.q, &sc.q), (&l.k, &sc.k), (&l.v, &sc.v)],
                &sc.norm,
                rows,
                &sc.gemm,
            );
            for (input, norm, out, heads, store) in [
                (&sc.q, &l.qnorm, &sc.qh, self.heads, 0),
                (&sc.k, &l.knorm, &sc.kh, self.kv_heads, 1),
            ] {
                cmd.dispatch(
                    "qwen3_head_rope",
                    &[input, &norm.buffer, &meta, out, &sc.v, &sc.vh],
                    &[
                        heads as u32,
                        norm.ty,
                        self.rope.to_bits(),
                        self.eps.to_bits(),
                        store,
                    ],
                    [heads, rows, 1],
                    32,
                );
            }
            cmd.dispatch(
                "qwen3_attention",
                &[&sc.qh, &sc.kh, &sc.vh, &meta, &meta, &sc.attn, &tiles],
                &[
                    self.heads as u32,
                    self.kv_heads as u32,
                    0,
                    (1f32 / 128f32.sqrt()).to_bits(),
                ],
                [self.heads, nt, 1],
                128,
            );
            l.o.linear(&cmd, &sc.attn, &sc.delta, rows, 1., &sc.gemm);
            cmd.dispatch(
                "residual",
                &[&sc.x, &sc.delta],
                &[(rows * self.width) as u32, 1f32.to_bits()],
                [(rows * self.width).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "rms",
                &[&sc.x, &l.ffn_norm.buffer, &sc.norm],
                &[self.width as u32, l.ffn_norm.ty, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            projections(
                &cmd,
                &[(&l.gate, &sc.gate), (&l.up, &sc.up)],
                &sc.norm,
                rows,
                &sc.gemm,
            );
            cmd.dispatch(
                "swiglu",
                &[&sc.gate, &sc.up],
                &[(rows * self.ff) as u32],
                [(rows * self.ff).div_ceil(256), 1, 1],
                256,
            );
            l.down.linear(&cmd, &sc.gate, &sc.delta, rows, 1., &sc.gemm);
            cmd.dispatch(
                "residual",
                &[&sc.x, &sc.delta],
                &[(rows * self.width) as u32, 1f32.to_bits()],
                [(rows * self.width).div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.dispatch(
            "qwen3_pool",
            &[&sc.x, &self.output_norm.buffer, &last, &pooled],
            &[
                self.width as u32,
                self.output_norm.ty,
                self.eps.to_bits(),
                u32::from(score.is_none()),
            ],
            [seqs.len(), 1, 1],
            256,
        );
        if let (Some((y, n)), Some(out)) = (score, &scores) {
            let h = self.head.as_ref().unwrap_or(&self.embedding);
            cmd.dispatch(
                "qwen3_score",
                &[&pooled, &h.buffer, out],
                &[self.width as u32, h.ty, y, n],
                [seqs.len(), 1, 1],
                256,
            );
        }
        let completion = cmd.submit()?;
        let mut inputs = vec![ids, meta, tiles, last];
        let output = if let Some(s) = scores {
            inputs.push(pooled);
            s
        } else {
            pooled
        };
        Ok(PendingEncoding {
            completion,
            identity: self.identity.clone(),
            output,
            n: seqs.len(),
            score,
            _inputs: inputs,
        })
    }
}

impl EncoderBackend for Qwen3Encoder {
    type Pending = PendingEncoding;
    fn validate(&self, seqs: &[Vec<u32>]) -> std::result::Result<(), String> {
        let rows = seqs.iter().try_fold(0usize, |n, s| n.checked_add(s.len()));
        if seqs.is_empty()
            || rows.is_none_or(|n| n > self.row_budget)
            || seqs.iter().any(|s| {
                s.is_empty()
                    || s.len() > self.context
                    || s.iter().any(|&t| t as usize >= self.vocab)
            })
        {
            return Err(format!(
                "encoder input exceeds context {}, batch rows {}, or vocabulary bounds (empty sequences are invalid)",
                self.context, self.row_budget
            ));
        }
        Ok(())
    }
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weight_bytes)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn coalesce_row_budget(&self) -> usize {
        self.row_budget
    }
    // Apple queue ordering permits enqueue-ahead without duplicate arenas.
    // This is not concurrent compute; a second queue needs its own arena and
    // serving measurements before election. CUDA's dual lanes remain intact.
    fn lanes(&mut self) -> usize {
        1
    }
    fn pool_ready(&self, p: &Self::Pending) -> bool {
        p.completion.ready()
    }
    fn embed_submit(
        &mut self,
        s: &[Vec<u32>],
        lane: usize,
    ) -> std::result::Result<Self::Pending, String> {
        self.submit(s, None, lane).map_err(|e| e.to_string())
    }
    fn rerank_submit(
        &mut self,
        s: &[Vec<u32>],
        y: u32,
        n: u32,
        lane: usize,
    ) -> std::result::Result<Self::Pending, String> {
        self.submit(s, Some((y, n)), lane)
            .map_err(|e| e.to_string())
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> std::result::Result<Vec<Vec<f32>>, String> {
        if p.score.is_some() || !std::rc::Rc::ptr_eq(&p.identity, &self.identity) {
            return Err("wrong encoder completion kind".into());
        }
        p.completion.wait().map_err(|e| e.to_string())?;
        // The GPU already pooled and normalized; the host only copies bytes.
        let v = unsafe { p.output.read_f32(0, p.n * self.width) };
        if v.iter().any(|x| !x.is_finite()) {
            return Err("non-finite GPU embedding".into());
        }
        Ok(v.chunks_exact(self.width).map(<[f32]>::to_vec).collect())
    }
    fn rerank_collect(
        &mut self,
        p: &Self::Pending,
        y: u32,
        n: u32,
    ) -> std::result::Result<Vec<f32>, String> {
        if p.score != Some((y, n)) || !std::rc::Rc::ptr_eq(&p.identity, &self.identity) {
            return Err("wrong encoder completion head".into());
        }
        p.completion.wait().map_err(|e| e.to_string())?;
        let v = unsafe { p.output.read_f32(0, p.n) };
        if v.iter().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x)) {
            return Err("invalid GPU relevance score".into());
        }
        Ok(v)
    }
}
