//! DeepEncoder: SAM ViT-B -> convolutional squeeze -> CLIP-L -> projector.
//! One live, bounded four-view wave; each transformer block yields to decode.
//! Position interpolation, pixel processing, relative bias and all math are GPU.
use super::*;
use paddock_engine::generator::VisionBudget;
mod load;
mod preprocess;
mod run;
pub(super) const BUDGET: VisionBudget = VisionBudget {
    min_pixels: 1,
    max_pixels: 32 * 640 * 640,
    max_edge: None,
    pixels_per_token: 640 * 640 / 110,
    min_tokens: 273,
    max_tokens: 3793,
};
// Scope bounds, not qualified capacities. Input and output are charged against
// the same Metal allocation ledger as decoder weights and KV.
pub(super) const MAX_IMAGES: usize = 16;
pub(super) const MAX_TOKENS: usize = 8192;
pub(super) struct Input {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub cols: usize,
    pub rows: usize,
}
impl Input {
    pub fn tokens(&self) -> usize {
        self.rows * 10 * (self.cols * 10 + 1) + 273
    }
}
struct Norm {
    w: Weight,
    b: Weight,
}
struct Linear {
    w: Weight,
    b: Weight,
}
struct Block {
    ln1: Norm,
    ln2: Norm,
    qkv: Linear,
    out: Linear,
    up: Linear,
    down: Linear,
    relative: Option<(Weight, Weight)>,
}
pub(super) struct Vision {
    patch: Linear,
    sam_pos: Weight,
    sam: Vec<Block>,
    clip: Vec<Block>,
    neck0: Weight,
    neck1: Norm,
    neck2: Weight,
    neck3: Norm,
    net2: Weight,
    net3: Weight,
    cls: Weight,
    clip_pos: Weight,
    pre: Norm,
    projector: Linear,
    nl: Weight,
    separator: Weight,
}
pub(super) struct Job {
    input: Input,
    source: Buffer,
    output: Option<Buffer>,
    next: usize,
    global_done: bool,
    wave: Option<Wave>,
}
struct Wave {
    count: usize,
    grid: usize,
    first: usize,
    layer: usize,
    x: Buffer,
    n: Buffer,
    part: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    ff: Buffer,
    rh: Buffer,
    rw: Buffer,
    gather: Buffer,
    neck: Buffer,
    neck_next: Buffer,
    sam: Buffer,
    cx: Buffer,
    concat: Buffer,
    projected: Buffer,
    sam_pos: Buffer,
    clip_pos: Buffer,
    global_tiles: Buffer,
    window_tiles: Buffer,
    clip_tiles: Buffer,
}
impl Vision {
    // Bound for one wave at the largest allowed geometry, both raster/packed
    // slab shapes, RGB/resampling temporaries and 16 slots of image output.
    // Deliberately conservative until manager's exact Metal admission is shared.
    pub(super) fn workspace_bound() -> u64 {
        2304 * 1024 * 1024
    }
    fn norm(c: &Commands<'_>, n: &Norm, x: &Buffer, y: &Buffer, rows: usize, eps: f32) {
        c.dispatch(
            "vis_ln",
            &[x, &n.w.buffer, &n.b.buffer, y],
            &[n.w.k as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    fn linear(c: &Commands<'_>, w: &Linear, x: &Buffer, y: &Buffer, rows: usize, mode: u32) {
        c.dispatch(
            if w.w.ty == 1 { "uov_mm" } else { "uov_mm_f32" },
            &[&w.w.buffer, x, y, &w.b.buffer],
            &[w.w.k as u32, w.w.n as u32, rows as u32, mode],
            [w.w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
    fn plain(c: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
        c.dispatch(
            if w.ty == 1 { "uov_mm" } else { "uov_mm_f32" },
            &[&w.buffer, x, y, &w.buffer],
            &[w.k as u32, w.n as u32, rows as u32, 0],
            [w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
}
