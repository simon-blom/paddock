//! Native Whisper large-v3: KB, NB and Røst exact F16 GGUFs.
//! Encoder-decoder serving uses the common transcriber, not an LLM adapter.
//! No external runtime, CPU model graph, re-quantization or feature cache.
//!
//! Remaining serving work: admission currently completes an entire encoder
//! wave before decode resumes. Phase-budgeted admission with separate scratch
//! is the latency target; c=4 streaming gaps are not a qualified SOTA result.
//! Word alignment is an opt-in chunked teacher-forced pass over the retained
//! encoder planes; ordinary decoding stores no attention probability matrices.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_engine::{
    audio::MelFeatures,
    whisper::{LangProb, StepOut, TimeScale, WhisperBackend},
};
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::path::Path;
mod backend;
mod forward;
mod load;
#[cfg(test)]
mod tests;
mod timing;
const D: usize = 1280;
const FF: usize = 5120;
const T: usize = 1500;
const V: usize = 51866;
const L: usize = 32;
const MAX_BATCH: usize = 16;
const ENC_BATCH: usize = 4;
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Whisper: {}", s.into()))
}
fn upload(d: &MetalDevice, x: &[u32]) -> Result<Buffer> {
    let b = d.alloc(std::mem::size_of_val(x).max(4))?;
    // SAFETY: newly allocated, unsubmitted and exclusively owned.
    unsafe {
        b.write_u32(x);
    }
    Ok(b)
}
struct Linear {
    w: Buffer,
    b: Buffer,
    k: usize,
    n: usize,
}
struct Norm {
    w: Buffer,
    b: Buffer,
}
struct Attention {
    norm: Norm,
    qkv: Buffer,
    bias: Buffer,
    out: Linear,
}
struct Mlp {
    norm: Norm,
    up: Linear,
    down: Linear,
}
struct EncoderLayer {
    attn: Attention,
    mlp: Mlp,
}
struct DecoderLayer {
    attn: Attention,
    cross_norm: Norm,
    q: Linear,
    kv: Buffer,
    vb: Buffer,
    out: Linear,
    mlp: Mlp,
}
struct Cache {
    k: Buffer,
    v: Buffer,
    ck: Buffer,
    cv: Buffer,
}
struct Scratch {
    x: Buffer,
    norm: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    up: Buffer,
    conv: Buffer,
    partial: Buffer,
    logits: Buffer,
    slots: Buffer,
    positions: Buffer,
    tokens: Buffer,
    rules: Buffer,
    pick: Buffer,
    stats: Buffer,
}
pub struct Whisper {
    device: MetalDevice,
    conv1: Linear,
    conv2: Linear,
    enc_pos: Buffer,
    enc: Vec<EncoderLayer>,
    enc_ln: Norm,
    embedding: Buffer,
    head: Option<Buffer>,
    dec_pos: Buffer,
    dec: Vec<DecoderLayer>,
    dec_ln: Norm,
    langs: Vec<(String, u32)>,
    ctx: usize,
    capacity: usize,
    cache: Vec<Cache>,
    scratch: Option<Scratch>,
    /// A slot can only append at its current length. Re-admission resets it;
    /// cancellation/reuse never inherits a predecessor's decoder history.
    lengths: Vec<Option<usize>>,
    last_rows: usize,
    weights_bytes: u64,
}
