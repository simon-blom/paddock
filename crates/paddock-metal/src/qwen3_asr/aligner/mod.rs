//! Official BF16 Qwen3-ForcedAligner. One packed causal prefill, no decode
//! loop or per-layer KV pool. Timestamp rows alone reach the 5000-bin head;
//! only GPU argmax indices leave the device. Audio attention never crosses
//! clip boundaries. Broad quality/rival qualification remains open.
use super::{audio, error, safetensors};
use crate::device::{Buffer, Commands, MetalDevice, Result};
use crate::weights::Weight;
use paddock_engine::align::{AlignBackend, AlignLimits, AlignReq};
mod forward;
mod load;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(super) fn trace(name: &str, b: &Buffer, count: usize) {
    if let Some(dir) = std::env::var_os("PADDOCK_QALIGN_TRACE") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bytes: Vec<_> = unsafe { b.read_u32(count) }
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(dir.join(format!("{name}.f32")), bytes).unwrap();
    }
}
#[cfg(test)]
fn reference_input(name: &str, b: &Buffer, count: usize) {
    if std::env::var("PADDOCK_QALIGN_REFERENCE_STAGE").is_ok_and(|stage| stage != name) {
        return;
    }
    if let Some(dir) = std::env::var_os("PADDOCK_QALIGN_REFERENCE_INPUTS") {
        let path = std::path::PathBuf::from(dir).join(format!("{name}.f32"));
        if path.exists() {
            let raw = std::fs::read(path).unwrap();
            assert_eq!(raw.len(), count * 4);
            let values = raw
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            unsafe {
                b.write_u32(&values);
            }
        }
    }
}
#[cfg(test)]
fn trace_stage<'a>(
    d: &'a MetalDevice,
    c: Commands<'a>,
    layer: usize,
    name: &str,
    b: &Buffer,
    count: usize,
) -> Result<Commands<'a>> {
    if layer == 27 && std::env::var_os("PADDOCK_QALIGN_TRACE").is_some() {
        c.finish()?;
        if matches!(name, "op-norm" | "op-post" | "op-attn") {
            trace_bf16(d, name, b, count)?;
        } else {
            trace(name, b, count);
        }
        d.begin()
    } else {
        Ok(c)
    }
}
#[cfg(test)]
fn trace_bf16(d: &MetalDevice, name: &str, b: &Buffer, count: usize) -> Result<()> {
    let widened = d.alloc(count * 4)?;
    let c = d.begin()?;
    c.dispatch(
        "qalign_widen",
        &[b, &widened],
        &[count as u32],
        [count.div_ceil(256), 1, 1],
        256,
    );
    c.finish()?;
    trace(name, &widened, count);
    Ok(())
}
const WIDTH: usize = 1024;
const FF: usize = 3072;
const LABELS: usize = 5000;
const VOCAB: usize = 152064;
const TIMESTAMP: u32 = 151705;
const HEAD_ROWS: usize = 128;
struct Layer {
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    qnorm: Weight,
    knorm: Weight,
    o: Weight,
    post: Weight,
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
    selected: Buffer,
    logits: Buffer,
}
pub struct Qwen3Aligner {
    device: MetalDevice,
    embedding: Weight,
    final_norm: Weight,
    score: Weight,
    layers: Vec<Layer>,
    tower: audio::Tower,
    scratch: Scratch,
    context: usize,
    pub weight_bytes: u64,
}
impl Qwen3Aligner {
    pub const MAX_CLIP_SECONDS: f32 = 120.;
    fn validate_request(&self, r: &AlignReq) -> Result<()> {
        audio::validate(&r.mel)?;
        if r.mel.n_samples < 8000 || r.mel.n_frames != r.mel.n_samples / 160 {
            return Err(error("aligner requires clip-length/drop-last mel framing"));
        }
        let end = r
            .splice_at
            .checked_add(r.n_audio)
            .ok_or_else(|| error("audio span overflow"))?;
        if r.ids.is_empty()
            || r.ids.len() > self.context
            || r.ids.iter().any(|&t| t as usize >= VOCAB)
            || r.splice_at != 1
            || end >= r.ids.len()
            || r.ids[0] != 151669
            || r.ids[end] != 151670
            || r.ids[r.splice_at..end].iter().any(|&t| t != super::AUDIO)
            || r.n_audio != paddock_engine::audio::audio_token_count(r.mel.n_frames)
            || r.ts_rows.is_empty()
            || !r.ts_rows.len().is_multiple_of(2)
            || r.ts_rows.windows(2).any(|p| p[0] >= p[1])
            || r.ts_rows
                .iter()
                .any(|&i| i <= end || i >= r.ids.len() || r.ids[i] != TIMESTAMP)
            || r.ids.iter().filter(|&&t| t == TIMESTAMP).count() != r.ts_rows.len()
        {
            return Err(error(
                "invalid aligner token/audio/timestamp layout or context",
            ));
        }
        Ok(())
    }
}
impl AlignBackend for Qwen3Aligner {
    fn limits(&self) -> AlignLimits {
        AlignLimits {
            batch: 4,
            rows: self.context,
            frames: audio::MAX_FRAMES,
        }
    }
    fn validate(&self, r: &AlignReq) -> std::result::Result<(), String> {
        self.validate_request(r).map_err(|e| e.to_string())
    }
    fn run_batch(
        &mut self,
        reqs: &[&AlignReq],
        canceled: &dyn Fn(usize) -> bool,
    ) -> std::result::Result<Vec<std::result::Result<Vec<u32>, String>>, String> {
        objc2::rc::autoreleasepool(|_| self.execute(reqs, canceled)).map_err(|e| e.to_string())
    }
}
fn project(c: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    // F32 accumulation, checkpoint BF16 output boundary independent of batch.
    c.dispatch(
        "qalign_project",
        &[&w.buffer, x, y, x],
        &[w.k as u32, w.n as u32, rows as u32, 0],
        [w.n.div_ceil(64), rows.div_ceil(32), 1],
        128,
    );
}
