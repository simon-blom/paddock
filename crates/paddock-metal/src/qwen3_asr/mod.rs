//! Native Qwen3-ASR Q8 decoder with a bounded GPU audio encoder.
//! Paged F16 KV, F32 residuals, chunked prefill and decode-first scheduling.
//! Exact elected checkpoint only; audio content is never text-radix cached.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_engine::paged_radix::PagedRadix;
use std::{collections::VecDeque, path::Path};
pub(crate) mod aligner;
mod audio;
mod forward;
#[cfg(test)]
mod kernel_tests;
mod load;
mod multimodal;
mod safetensors;
mod serving;
#[cfg(test)]
mod tests;

const CHUNK: usize = 512;
const SPLITS: usize = 16;
const WIDTH: usize = 2048;
const KVWIDTH: usize = 1024;
const QWIDTH: usize = 2048;
const FF: usize = 6144;
const VOCAB: usize = 151936;
const AUDIO: u32 = 151676;
const LAYERS: usize = 28;
struct Layer {
    norm: Weight,
    q: Weight,
    qnorm: Weight,
    knorm: Weight,
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
pub struct Qwen3Asr {
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
    audio: Option<audio::Tower>,
    encoding: VecDeque<multimodal::Encoding>,
    context: usize,
    page_stride: usize,
    weight_bytes: u64,
    kv_bytes: u64,
    pub last_gpu_seconds: f64,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Qwen3-ASR Metal: {}", s.into()))
}
// F32 operands on both TensorOps and narrow SIMD routes preserve the
// normalization boundary when a decode joins other requests' prefill.
fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    let (name, cols, tile) = if rows >= 16 {
        ("laguna_dense_f32", 16, 32)
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
impl Qwen3Asr {
    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        if slot >= self.slots.len()
            || self.pending.iter().any(|p| p.slot == slot)
            || self.encoding.iter().any(|p| p.owns(slot))
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
        // Never put audio-placeholder KV in the text radix. Full content-
        // keyed audio reuse is separate work; no unsafe hash-only shortcut.
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
