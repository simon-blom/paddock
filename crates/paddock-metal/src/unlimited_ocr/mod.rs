//! Native Unlimited-OCR Q8 decoder and SAM/CLIP DeepEncoder.
//! R-SWA keeps prompt KV globally visible and wraps only generated tokens.
//! Original GPU graph; no CPU tensor path, external backend or quant conversion.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_engine::paged_radix::PagedRadix;
use std::{collections::VecDeque, path::Path};
mod forward;
mod load;
mod moe;
mod multimodal;
mod serving;
#[cfg(test)]
mod tests;
mod vision;

const CHUNK: usize = 512;
const SPLITS: usize = 16;
const WIDTH: usize = 1280;
const KVWIDTH: usize = WIDTH;
const QWIDTH: usize = WIDTH;
const FF: usize = 6848;
const VOCAB: usize = 129280;
const IMAGE: u32 = 128815;
const LAYERS: usize = 12;
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
    experts: Option<moe::Experts>,
    keys: Buffer,
    values: Buffer,
}
struct Scratch {
    ids: Buffer,
    meta: Buffer,
    rope: Buffer,
    write_meta: Buffer,
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
    prompt: usize,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}
pub struct UnlimitedOcr {
    device: MetalDevice,
    embedding: Weight,
    output_norm: Weight,
    head: Weight,
    layers: Vec<Layer>,
    scratch: Scratch,
    moe: moe::Workspace,
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
    MetalError::Model(format!("Unlimited-OCR Metal: {}", s.into()))
}
// Q8 backbone retains F32 operands at every row count. The elected dense
// first FFN has K=6848, a multiple of 32 but not 64: use a bounds-aware tile.
fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    let (name, cols, tile) = if rows >= 16 {
        ("uocr_dense", 16, 32)
    } else if rows > 8 {
        ("linear_q8_r16", 4, 16)
    } else if rows > 4 {
        ("linear_q8_r8", 4, 8)
    } else if rows > 1 {
        ("linear_q8_r4", 4, 4)
    } else {
        ("linear_q8_r1", 4, 1)
    };
    cmd.dispatch(
        name,
        &[&w.buffer, x, y],
        &[w.k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()],
        [w.n.div_ceil(cols), rows.div_ceil(tile), 1],
        128,
    );
}
impl UnlimitedOcr {
    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        if self.pending.iter().any(|p| p.slot == slot) || self.encoding.iter().any(|p| p.owns(slot))
        {
            return Err(error("slot already prefilling or encoding"));
        }
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= VOCAB)
        {
            return Err(error("invalid prefill slot/tokens/context"));
        }
        let s = &mut self.slots[slot];
        s.prompt = tokens.len();
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
            self.radix.insert(
                &s.history[..s.prompt.min(s.history.len())],
                s.table.blocks(),
                &mut self.pool,
            );
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
