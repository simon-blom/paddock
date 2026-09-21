//! Native PaddleOCR-VL 1.6: BF16 ERNIE decoder and NaViT document tower.
//! GPU-only tensor math, paged F16 KV and block-budgeted image admission.
//! Original kernels; graph semantics studied from Paddle's published model
//! and our CUDA family. Exact checkpoint scope, not family-wide qualification.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_engine::paged_radix::PagedRadix;
use std::{collections::VecDeque, path::Path};
mod forward;
#[cfg(test)]
mod kernel_tests;
mod load;
mod multimodal;
mod serving;
#[cfg(test)]
mod tests;
mod vision;

const CHUNK: usize = 512;
const SPLITS: usize = 16;
const WIDTH: usize = 1024;
const KVWIDTH: usize = 256;
const QWIDTH: usize = 2048;
const FF: usize = 3072;
const VOCAB: usize = 103424;
const IMAGE: u32 = 100295;
const LAYERS: usize = 18;
struct Layer {
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    post: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    keys: Buffer,
    values: Buffer,
}
struct Scratch {
    ids: Buffer,
    meta: Buffer,
    rope: Buffer,
    pages: Buffer,
    output_rows: Buffer,
    decode_rows: Buffer,
    tiles: Buffer,
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    parts: Buffer,
    delta: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,
}
#[derive(Default)]
struct Slot {
    table: BlockTable,
    history: Vec<u32>,
    reused: usize,
    mm: Option<multimodal::Layout>,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}
pub struct PaddleOcr {
    device: MetalDevice,
    embedding: Weight,
    output_norm: Weight,
    head: Weight,
    layers: Vec<Layer>,
    scratch: Scratch,
    slots: Vec<Slot>,
    pending: VecDeque<Pending>,
    pool: KvPool,
    radix: PagedRadix,
    vision: Option<vision::Vision>,
    encoding: VecDeque<multimodal::Encoding>,
    context: usize,
    page_stride: usize,
    weight_bytes: u64,
    kv_bytes: u64,
    pub last_gpu_seconds: f64,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("PaddleOCR Metal: {}", s.into()))
}
// BF16 weights are not narrowed or requantized. F32 activations cross both
// narrow SIMD and wide TensorOps routes; unrelated prompt rows introduce no
// additional F16 cut. Normalization/residual state remains F32.
fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    if rows < 8 {
        cmd.dispatch(
            "linear",
            &[&w.buffer, x, y],
            &[w.k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(4), rows, 1],
            128,
        );
    } else {
        cmd.dispatch(
            "vis_bmm32",
            &[&w.buffer, x, y, &w.buffer],
            &[w.k as u32, w.n as u32, rows as u32, 0],
            [w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
}
impl PaddleOcr {
    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= VOCAB)
        {
            return Err(error("invalid prefill slot/tokens/context"));
        }
        let s = &mut self.slots[slot];
        s.table.clear(&mut self.pool);
        s.history.clear();
        s.mm = None;
        let blocks = self.radix.match_prefix(tokens);
        let reused = blocks.len() * BLOCK_TOKENS;
        s.table.share_prefix(&blocks, &mut self.pool);
        s.history.extend_from_slice(&tokens[..reused]);
        s.reused = reused;
        Ok(reused)
    }
    fn publish(&mut self, slot: usize) {
        let s = &self.slots[slot];
        // Never put image-placeholder KV in the text radix. Full content-
        // keyed image reuse is separate work; no unsafe hash-only shortcut.
        if s.mm.is_none() {
            self.radix
                .insert(&s.history, s.table.blocks(), &mut self.pool);
        }
    }
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        let reused = self.prepare(slot, tokens)?;
        let mut last = Vec::new();
        for chunk in tokens[reused..].chunks(CHUNK) {
            let pos = self.slots[slot].history.len();
            let rows = chunk
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, (pos + i) as u32))
                .collect::<Vec<_>>();
            let output = if pos + chunk.len() == tokens.len() {
                vec![chunk.len() - 1]
            } else {
                Vec::new()
            };
            last = self.execute(&rows, &output)?;
        }
        self.publish(slot);
        Ok(last)
    }
}
