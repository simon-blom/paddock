//! DINOv3 dense prediction - `tic-forestry-v1`: a fine-tuned DINOv3 ViT-L/16
//! backbone under a four-stage multi-scale decoder. One square of aerial
//! photography in (`u8`, red / green / blue / near-infrared at 0.5 m), two
//! rasters out at 1 m: a land-cover class per pixel and a canopy height.
//!
//! This is the engine's first model with no tokens, no KV cache and no
//! sampling, and its serving profile is the opposite of chat: nobody waits on
//! one chip, a property is thousands of them and a county is millions. So the
//! unit the whole family is built around is a batch of chips, every op is
//! row-batched over `chips * rows`, and there is no per-chip path to keep warm.
//!
//! Reference is the training graph itself, not a model card: transformers'
//! `modeling_dinov3_vit.py` for the backbone and the training repo's `Fuse`
//! module for the decoder, checked against golden vectors the training stack
//! produced (twelve chips, reference argmax + height, full logits on three).
//! There is no llama.cpp to diff against for a model like this, so those
//! vectors are the parity oracle. Four things load cleanly and return noise
//! if a port gets them wrong, and each is pinned by a refusal or a gate:
//!
//!   1. Four input bands, not three; statistics in 0-1 units (divide by 255
//!      first). `gpu::dense_pred::dp_u8_patch_rows`.
//!   2. Position is ROPE only, at theta 100, on normalized patch-centre
//!      coordinates; there is no position tensor. `RopeTable`.
//!   3. LayerScale multiplies each sublayer before the residual add.
//!      `dp_res_ls_ln_h`.
//!   4. One class + four register tokens are attended to and then dropped
//!      before the decoder. Here they sit after the patch tokens (attention
//!      without a mask is permutation-equivariant), so "drop the prefix" is
//!      "read the first grid*grid rows of a chip" and nothing is gathered.
//!
//! A fifth the architecture notes leave out: the decoder's skip connection is
//! the 32x32 token grid at every stage, bilinearly resized (align_corners
//! False) to the stage's raster before the add. `dp_convt2_skip` samples it in
//! the same pass that lands the transposed conv.
//!
//! Precision class: f32 residual and accumulate over f16 weight planes and
//! f16 GEMM operands - the vision towers' class. bf16 -> f16 is exact for
//! every weight in f16's normal range and the loader refuses a plane that
//! would overflow; the reference ran bf16 autocast, whose operand rounding is
//! coarser than f16's, so the two agree to the reference's own noise.
//!
//! The backbone's activation interface is f16 too: every GEMM lands halves
//! and the seams and the attention between them read halves. Everything
//! between two GEMMs is a walk over a plane that measured at the DRAM roof, so
//! the bytes were the only lever left - and a GEMM output rounded to f16
//! keeps 11 significant bits where the reference's bf16 landing keeps 8. The
//! residual stream, every norm and every accumulate stay f32. Range is not a
//! question here: the largest GEMM output on the golden chips is 646.

mod forward;
mod load;

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::dinov3::Dinov3SegConfig;

use crate::gpu::{GpuExecutor, HalfTensor};

pub use crate::gpu_model::gpt_oss::GpuModelError;

/// The widest pass worth running, elected on the A6000 (sm_86): 72.7 chips/s
/// at 8, 75.8 at 16, 77.1 at 32. Past sixteen a doubling buys under 2% and
/// costs another 1.6 GB of resident workspace and another fifth of a second
/// for a lone chip - the serving seam runs every pass at the endpoint's width,
/// so a lone chip pays for the whole pass. Narrower is a legitimate endpoint
/// choice (an interactive endpoint wants 1-4); wider is not offered.
/// Re-measure with the ignored `throughput_at_pass_widths` test.
pub const MAX_PASS_WIDTH: usize = 16;

/// The cap for a given die. Consumer Blackwell (cc 12.0, the RTX PRO 6000
/// and the 5090) peaks narrower: 181.5 chips/s at 8 against 171 at 16 and
/// 166 at 32 (Max-Q, 300 W - the card is at its power cap from width 4 up,
/// so a wider pass buys no more work per joule and the wider GEMM grids lose
/// a little to their tails). Eight also halves the resident workspace and the
/// lone-chip latency. Every other die keeps [`MAX_PASS_WIDTH`].
pub fn max_pass_width(cc: (u32, u32)) -> usize {
    match cc {
        (12, 0) => 8,
        _ => MAX_PASS_WIDTH,
    }
}

/// LayerNorm weight + bias.
struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

/// One pre-LN LayerScale block.
struct Block {
    ln1: Norm,
    /// q|k|v stacked row-wise: one M = 3d GEMM instead of three M = d ones
    wqkv: HalfTensor,
    bq: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ls1: CudaSlice<f32>,
    ln2: Norm,
    up: HalfTensor,
    up_b: CudaSlice<f32>,
    down: HalfTensor,
    down_b: CudaSlice<f32>,
    ls2: CudaSlice<f32>,
}

/// conv3x3 (weight permuted tap-major for the im2row) -> GroupNorm -> GELU.
struct ConvGn {
    w: HalfTensor,
    b: CudaSlice<f32>,
    gn: Norm,
    groups: usize,
}

/// One decoder stage.
struct Stage {
    width: usize,
    /// 1x1 projection of the tapped hidden state onto this stage's width
    project: HalfTensor,
    /// stage 0 only: the projection's own bias. Later stages fold theirs into
    /// `up_b` - a bilinear resize of a constant is that constant.
    project_b: Option<CudaSlice<f32>>,
    /// stages 1..: the transposed conv into this stage, as a C_prev -> 4*C GEMM
    /// (rows tap-major), and convT.bias + project.bias
    up: Option<HalfTensor>,
    up_b: Option<CudaSlice<f32>>,
    blend: [ConvGn; 2],
}

/// Every buffer a forward pass touches, allocated once for `cap` chips.
/// Allocation inside a serving loop is a serve-killer (the PaddleOCR tower
/// measured 80 ms of host wall against 5 ms of kernels), and this geometry is
/// fixed, so there is nothing to grow.
struct Workspace {
    cap: usize,
    px: CudaSlice<u8>,
    /// the widest f16 GEMM input anywhere: the last stage's 3x3 im2row. In
    /// the backbone it is where attention lands and where a tap's f16 view of
    /// the residual goes.
    s16: CudaSlice<f16>,
    /// the pre-norm landing, kept apart from s16 so a tap cannot clobber it
    n16: CudaSlice<f16>,
    /// the residual stream - the one backbone plane that stays f32
    x: CudaSlice<f32>,
    /// the half activation interface: every GEMM of a block lands f16, the
    /// seams and the attention between them read (and write) f16. `ff` is
    /// the up projection's landing AND, after the in-place GELU, the down
    /// projection's input.
    qkv: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    proj: CudaSlice<f16>,
    ff: CudaSlice<f16>,
    /// only where the f16-landing GEMM is not the device's elected route
    /// (tcgen05 on cc 10.0): the f32 landing those GEMMs convert out of
    land32: Option<CudaSlice<f32>>,
    /// the projected taps, one per stage, [chips * tokens, width]
    taps: Vec<CudaSlice<f32>>,
    /// decoder planes: conv landing, stage input, convT landing, f16 activations
    ya: CudaSlice<f32>,
    xa: CudaSlice<f32>,
    g: CudaSlice<f32>,
    h16: CudaSlice<f16>,
    gn_part: CudaSlice<f32>,
    gn_stat: CudaSlice<f32>,
    /// stacked heads' landing, then the three outputs
    o: CudaSlice<f32>,
    cls: CudaSlice<u8>,
    height: CudaSlice<f32>,
    /// allocated on the first pass that asks for logits - the parity gate's
    /// view, not something a production sweep pays for
    logits: Option<CudaSlice<f16>>,
    bytes: u64,
}

/// What one pass returns, chip-major.
pub struct SegOutput {
    pub chips: usize,
    /// output raster side
    pub size: usize,
    /// `[chips][size][size]` argmax class codes
    pub classes: Vec<u8>,
    /// `[chips][size][size]` regression output, metres; never clamped
    pub height: Vec<f32>,
    /// `[chips][size][size][n_classes]` when asked for
    pub logits: Option<Vec<f16>>,
}

pub struct GpuDinov3Seg {
    exec: Arc<GpuExecutor>,
    cfg: Dinov3SegConfig,
    patch_w: HalfTensor,
    /// `[tokens, hidden]`: the patch-embedding bias on the patch rows, the
    /// class and register tokens on the tail rows - one broadcast add lands
    /// all three onto the zero-padded patch GEMM output
    embed_add: CudaSlice<f32>,
    rope_cos: CudaSlice<f32>,
    rope_sin: CudaSlice<f32>,
    blocks: Vec<Block>,
    stages: Vec<Stage>,
    /// class head rows then the height head's row: one GEMM, one landing
    out_w: HalfTensor,
    out_b: CudaSlice<f32>,
    /// the up projection lands bias + GELU in its epilogue (slot 624) instead
    /// of a pass over the plane; needs the f16 landing to be the elected route
    fuse_gelu: bool,
    ws: Workspace,
    weight_bytes: u64,
}

impl GpuDinov3Seg {
    pub fn config(&self) -> &Dinov3SegConfig {
        &self.cfg
    }
    /// Most chips one [`Self::segment`] call takes.
    pub fn max_batch(&self) -> usize {
        self.ws.cap
    }
    /// Bytes one input chip must be: side x side x bands, u8 HWC.
    pub fn chip_bytes(&self) -> usize {
        self.cfg.image_size * self.cfg.image_size * self.cfg.channels
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.ws.bytes
    }
}

/// DINOv3's rope, as cos/sin tables over the patch grid `[grid*grid, hd/2]`.
///
/// Built in the f32 order transformers uses, because the angles are the model:
/// patch centres `(i + 0.5) / n` mapped to (-1, 1); `inv_freq[j] =
/// 1 / theta^(j / (hd/4))` for `j < hd/4`; angle = `2*pi * coord * inv_freq`;
/// the first hd/4 angles read the ROW coordinate, the next hd/4 the column.
/// The table covers hd/2 because rotate_half pairs `(j, j + hd/2)` and the
/// reference tiles the same hd/2 angles across both halves.
pub(crate) struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl RopeTable {
    pub(crate) fn new(grid: usize, head_dim: usize, theta: f32) -> Self {
        let quarter = head_dim / 4;
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|j| 1.0f32 / theta.powf(j as f32 * (4.0f32 / head_dim as f32)))
            .collect();
        let coord = |i: usize| 2.0f32 * ((i as f32 + 0.5f32) / grid as f32) - 1.0f32;
        let tau = 2.0f32 * std::f32::consts::PI;
        let mut cos = vec![0f32; grid * grid * half];
        let mut sin = vec![0f32; grid * grid * half];
        for py in 0..grid {
            for px in 0..grid {
                let row = (py * grid + px) * half;
                for j in 0..quarter {
                    let ay = tau * coord(py) * inv_freq[j];
                    let ax = tau * coord(px) * inv_freq[j];
                    cos[row + j] = ay.cos();
                    sin[row + j] = ay.sin();
                    cos[row + quarter + j] = ax.cos();
                    sin[row + quarter + j] = ax.sin();
                }
            }
        }
        Self { cos, sin }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_table_is_antisymmetric_about_the_grid_centre() {
        // centres are symmetric about 0, so mirrored patches carry mirrored
        // angles: equal cos, opposite sin. A grid built from corners or from
        // integer positions fails this.
        let (g, hd) = (32usize, 64usize);
        let t = RopeTable::new(g, hd, 100.0);
        let half = hd / 2;
        for j in 0..half {
            let a = (3 * g + 5) * half + j;
            let b = ((g - 1 - 3) * g + (g - 1 - 5)) * half + j;
            assert!((t.cos[a] - t.cos[b]).abs() < 1e-6);
            assert!((t.sin[a] + t.sin[b]).abs() < 1e-6);
        }
        // frequency 0 is a full 2*pi*coord turn: the corner patch's row angle
        let c0 = 2.0f32 * (0.5f32 / 32.0) - 1.0;
        assert!((t.cos[0] - (2.0 * std::f32::consts::PI * c0).cos()).abs() < 1e-6);
    }
}
