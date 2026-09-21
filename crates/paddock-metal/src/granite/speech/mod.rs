//! Native Granite Speech conformer + window Q-Former. Packed clips share
//! projections, never attention/convolution domains. Stage boundaries return
//! to the language scheduler; one bounded workspace is live at a time.
//!
//! Graph references: IBM's 4.1 base/Plus cards and llama.cpp b10909 mtmd
//! granite-speech.cpp (study only). Original Metal kernels, no host model
//! math. CTC drafting/audio prefix caching and broad serving qualification
//! remain open; this is an explicit-only experimental implementation.
use super::*;
use paddock_engine::audio::{MelFeatures, granite as mel};
pub(super) mod admission;
mod forward;
mod load;
#[cfg(test)]
mod tests;

const E: usize = 1024;
const F: usize = 4096;
pub(super) const MAX_FRAMES: usize = 6000;
pub(super) const MAX_CLIPS: usize = 16;
// Includes one packed wave, Q.R offsets, Plus capture/concatenation and all
// result spans. Host PCM/DSP and driver allocations are not included here.
pub(super) const WORKSPACE: u64 = 1 << 30;
struct Norm {
    w: Weight,
    b: Weight,
}
struct Linear {
    w: Weight,
    b: Weight,
}
struct Block {
    ff1: Norm,
    up1: Linear,
    down1: Linear,
    attn: Norm,
    qkv: Weight,
    rel: Weight,
    out: Linear,
    conv: Norm,
    pw1: Linear,
    dw: Weight,
    bn: Norm,
    pw2: Linear,
    ff2: Norm,
    up2: Linear,
    down2: Linear,
    post: Norm,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm: Norm,
}
struct Projector {
    sa: Attention,
    ca: Attention,
    up: Linear,
    down: Linear,
    norm: Norm,
}
pub(super) struct Tower {
    input: Linear,
    blocks: Vec<Block>,
    ctc: Linear,
    mid: Linear,
    queries: Buffer,
    projectors: Vec<Projector>,
    output: Linear,
    plus: bool,
}
pub(super) struct Job {
    sizes: Vec<usize>,
    rows: usize,
    padded: usize,
    queries: usize,
    phase: usize,
    tiles: Buffer,
    clips: Buffer,
    indices: Buffer,
    x: Buffer,
    norm: Buffer,
    tmp: Buffer,
    up: Buffer,
    glu: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qr: Buffer,
    attn: Buffer,
    ctc: Buffer,
    tap: Buffer,
    enc: Buffer,
    query: Buffer,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Granite Speech: {}", s.into()))
}
fn upload(d: &MetalDevice, xs: &[u32]) -> Result<Buffer> {
    d.upload(&xs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
pub(super) fn validate(m: &MelFeatures) -> Result<()> {
    if m.n_frames == 0
        || m.n_frames > MAX_FRAMES
        || m.n_samples > 120 * 16000
        || m.n_frames != mel::encoder_frames(m.n_samples)
        || m.n_frames.checked_mul(mel::INPUT_DIM) != Some(m.data.len())
        || m.data.iter().any(|v| !v.is_finite())
    {
        return Err(error(
            "invalid stacked-mel geometry or clip exceeds 120-second implementation limit",
        ));
    }
    Ok(())
}
