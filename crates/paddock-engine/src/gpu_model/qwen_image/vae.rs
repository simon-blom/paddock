//! The VAE decoder: `AutoencoderKLQwenImage21`, the Wan-2.2 residual causal
//! VAE specialised to one frame - which makes every 3x3 a plain 2D conv, the
//! `time_conv` planes dead (never run on a first chunk) and the DupUp
//! shortcut a channel-mapped 2x nearest duplication. Official F32
//! safetensors; the blocks and the ops over them are `vae_ops.rs`, shared
//! with the encoder.
//!
//! Planes are NHWC f32; every conv is stripe-tiled im2row + the f16 GEMM,
//! so the 9x staging plane is bounded by a stripe, and the tail of the
//! decoder (the last three up blocks and the head) runs in horizontal bands
//! with a halo, so the activation planes are bounded by a band - exactly,
//! not blended: the halo covers every convolution's reach and the core rows
//! come out byte for byte as a whole-plane decode's. Decoder dims at
//! `decoder_base_dim` 144 and `dim_mult` [1, 2, 4, 8, 8]:
//!   conv_in 64 -> 1152 @ latent res; mid (res, attn, res) @ 1152;
//!   up 1152 -> 1152 x2 (2x each), 1152 -> 576 (2x), 576 -> 288 (2x),
//!   288 -> 144 (no up); norm_out; conv_out 144 -> 4 (RGBA).
//! The temporal factor of the first three up blocks is 2 (their DupUp reads
//! the odd channel of each pair when repeats is 4), the fourth's 1.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::safetensors::SafetensorsFile;

use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::LATENT_CHANNELS;
use super::vae_ops::{
    AttnBlock, Conv3, Conv3Mode, Loader, ResBlock, attention, conv3, f32_of, resblock,
};

/// `latents_mean` / `latents_std` from the VAE config - the packed latent
/// is de-normalised by them before decode and normalised by them after
/// encode. (One of the 64 measured means happens to round like pi/6; it is
/// a statistic, not a constant.)
#[allow(clippy::approx_constant)]
pub(super) const LATENTS_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
pub(super) const LATENTS_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
    2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075,
    3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149,
    3.7662, 3.6746, 3.4563, 3.8161,
];

/// The up block the decoder's tail is cut into bands from: up0 runs whole
/// (its planes are a sixty-fourth of the output's), up1 in bands of its own
/// into a full plane (`banded_up_block`), and up2 / up3 / up4 with the head
/// run band by band from that plane to the picture.
const BAND_FROM: usize = 2;
/// Bytes the widest band plane may take - the up3 output, 288 channels of
/// f32 at full resolution. Five such planes live through a residual block,
/// so this bounds the decode transient by the band rather than the picture:
/// measured whole-plane it was 6 GB at 1024^2 growing to 31 GB at 2752^2,
/// all of it the last two up blocks.
const BAND_PLANE_BYTES: usize = 512 << 20;
/// Rows of halo a band carries on each side, at the resolution the bands are
/// cut (4x latent, the up2 input): what the tail's convolutions reach back.
/// up2's three residual blocks (6 rows) and upsampler (1 row at 8x = 0.5),
/// up3's (6 at 8x = 3, upsampler 1 at 16x = 0.25), up4's (6 at 16x = 1.5)
/// and conv_out (1 at 16x = 0.25): 11.5, so 12. Rows inside the halo see
/// the band's zero-padded edge where the plane has data; rows of the core
/// never do, which is what makes the bands exact (the gate holds them to
/// the whole-plane bytes).
const BAND_HALO: usize = 12;

struct UpBlock {
    resnets: Vec<ResBlock>,
    /// the Resample conv (after the 2x nearest upsample), when the block ups
    upsampler: Option<Conv3>,
    c_in: usize,
    c_out: usize,
    /// DupUp geometry (temporal factor, channel repeats), when the block ups
    dupup: Option<(usize, usize)>,
}

pub struct VaeDecoder {
    exec: Arc<GpuExecutor>,
    post_quant_w: CudaSlice<f32>,
    post_quant_b: CudaSlice<f32>,
    lat_mean: CudaSlice<f32>,
    lat_std: CudaSlice<f32>,
    conv_in: Conv3,
    mid_res: [ResBlock; 2],
    mid_attn: AttnBlock,
    ups: Vec<UpBlock>,
    norm_out: CudaSlice<f32>,
    conv_out: Conv3,
    pub weights_bytes: u64,
}

impl VaeDecoder {
    pub fn load(exec: Arc<GpuExecutor>, path: &Path) -> Result<Self, GpuModelError> {
        let st = SafetensorsFile::open(path)
            .map_err(|e| GpuModelError::Unsupported(format!("vae: {e}")))?;
        let mut ld = Loader {
            exec: exec.clone(),
            st: &st,
            bytes: 0,
        };
        let (pq, _) = f32_of(&st, "post_quant_conv.weight")?;
        let post_quant_w = exec.to_device(&pq)?;
        let post_quant_b = ld.vec("post_quant_conv.bias")?;
        ld.bytes += pq.len() as u64 * 4;
        let conv_in = ld.conv3("decoder.conv_in")?;
        let base = conv_in.c_out; // 1152
        let mid_res = [
            ld.res("decoder.mid_block.resnets.0", base, base)?,
            ld.res("decoder.mid_block.resnets.1", base, base)?,
        ];
        let mid_attn = ld.attn("decoder.mid_block.attentions.0", base)?;
        // dims = decoder_base * [8, 8, 4, 2, 1]; temporal ups [T, T, T, F]
        let dims = [base, base, base / 2, base / 4, base / 8];
        let temporal = [true, true, true, false];
        let mut ups = Vec::with_capacity(5);
        for i in 0..5 {
            let (c_in, c_out) = (if i == 0 { base } else { dims[i - 1] }, dims[i]);
            let up = i != 4;
            let mut resnets = Vec::with_capacity(3);
            let mut cur = c_in;
            for r in 0..3 {
                resnets.push(ld.res(&format!("decoder.up_blocks.{i}.resnets.{r}"), cur, c_out)?);
                cur = c_out;
            }
            let (upsampler, dupup) = if up {
                let ft = if temporal[i] { 2 } else { 1 };
                let repeats = c_out * ft * 4 / c_in;
                (
                    Some(ld.conv3(&format!("decoder.up_blocks.{i}.upsampler.resample.1"))?),
                    Some((ft, repeats)),
                )
            } else {
                (None, None)
            };
            ups.push(UpBlock {
                resnets,
                upsampler,
                c_in,
                c_out,
                dupup,
            });
        }
        let norm_out = ld.vec("decoder.norm_out.gamma")?;
        let conv_out = ld.conv3("decoder.conv_out")?;
        let weights_bytes = ld.bytes;
        Ok(Self {
            lat_mean: exec.to_device(&LATENTS_MEAN)?,
            lat_std: exec.to_device(&LATENTS_STD)?,
            exec,
            post_quant_w,
            post_quant_b,
            conv_in,
            mid_res,
            mid_attn,
            ups,
            norm_out,
            conv_out,
            weights_bytes,
        })
    }

    /// Decode packed `[lh * lw][64]` latents (normalised, as the sampler
    /// leaves them) to `[16 lh][16 lw]` interleaved 8-bit RGBA. The tail of
    /// the decoder runs in horizontal bands sized to a fixed plane budget, so
    /// the render transient is bounded by a band, not by the picture.
    pub fn decode(
        &self,
        latents: &CudaSlice<f32>,
        lw: usize,
        lh: usize,
    ) -> Result<Vec<u8>, GpuModelError> {
        self.decode_with(latents, lw, lh, Some(self.band_rows(lw)))
    }

    /// The same decode with the tail cut into bands of `band_rows` (at the
    /// 4x-latent resolution the bands are cut at), or whole-plane when None.
    /// Both produce the same bytes - the gate holds the banded path to the
    /// whole one - so this exists for that gate and for reading the
    /// transient either way.
    pub fn decode_with(
        &self,
        latents: &CudaSlice<f32>,
        lw: usize,
        lh: usize,
        band_rows: Option<usize>,
    ) -> Result<Vec<u8>, GpuModelError> {
        let exec = &self.exec;
        let p = lw * lh;
        // z * std + mean, then the 1x1 post-quant conv (f32, it is 64 x 64)
        let mut z = exec.alloc(p * LATENT_CHANNELS)?;
        exec.copy_region(latents, 0, &mut z, 0, p * LATENT_CHANNELS)?;
        exec.dit_affine_cols(&mut z, &self.lat_std, &self.lat_mean, p, LATENT_CHANNELS)?;
        let mut x = exec.alloc(p * LATENT_CHANNELS)?;
        exec.matvec_f32_raw(
            &self.post_quant_w,
            LATENT_CHANNELS,
            LATENT_CHANNELS,
            &z,
            &mut x,
            p,
        )?;
        exec.bias_add(&mut x, &self.post_quant_b, p, LATENT_CHANNELS)?;
        // conv_in reads the raw latent plane (no norm before it)
        let mut x16 = exec.alloc_f16(p * LATENT_CHANNELS)?;
        exec.convert_f32_f16(&x, &mut x16, p * LATENT_CHANNELS)?;
        let (mut h, mut w) = (lh, lw);
        let mut x = conv3(exec, &x16, h, w, &self.conv_in, Conv3Mode::Same, None)?;
        super::dump(exec, "vae conv_in", &x, h * w * self.conv_in.c_out);
        // mid
        x = resblock(exec, &x, h, w, &self.mid_res[0])?;
        super::dump(exec, "vae mid res0", &x, h * w * self.mid_attn.c);
        x = attention(exec, &x, h, w, &self.mid_attn)?;
        super::dump(exec, "vae mid attn", &x, h * w * self.mid_attn.c);
        x = resblock(exec, &x, h, w, &self.mid_res[1])?;
        super::dump(exec, "vae mid res1", &x, h * w * self.mid_attn.c);
        // up0 whole (its planes are a sixty-fourth of the output's, and the
        // mid block needed the whole latent for its attention anyway); up1
        // in bands into a full plane - at 2752^2 its 1152-channel planes are
        // 2.2 GB each, and five of them lived through a residual block
        (x, h, w) = self.up_block(x, h, w, &self.ups[0])?;
        super::dump(exec, "vae up0", &x, h * w * self.ups[0].c_out);
        (x, h, w) = match band_rows {
            Some(_) => self.banded_up_block(&x, h, w, &self.ups[1])?,
            None => self.up_block(x, h, w, &self.ups[1])?,
        };
        super::dump(exec, "vae up1", &x, h * w * self.ups[1].c_out);
        let Some(band) = band_rows else {
            for (bi, blk) in self.ups[BAND_FROM..].iter().enumerate() {
                (x, h, w) = self.up_block(x, h, w, blk)?;
                super::dump(
                    exec,
                    &format!("vae up{}", bi + BAND_FROM),
                    &x,
                    h * w * blk.c_out,
                );
            }
            let rgba = self.head(&x, h, w)?;
            let n = h * w * self.conv_out.c_out;
            super::dump(exec, "vae rgba", &rgba, n);
            let mut u8s = exec.alloc_u8(n)?;
            exec.dit_to_u8(&rgba, &mut u8s, n)?;
            return Ok(exec.to_host_u8_len(&u8s, n)?);
        };
        // The tail in bands. A band is a contiguous row range of the up2
        // input (NHWC, so a row range is one slice) plus its halo, run
        // through the last three up blocks and the head as a small plane of
        // its own; only the core's rows are kept - `up` output rows per band
        // row after the tail's 2x stages. The halo rows see the band's
        // zero-padded edge where the real plane has data, the core rows
        // never do, so the kept bytes are the whole-plane decode's.
        let up = 1
            << self.ups[BAND_FROM..]
                .iter()
                .filter(|b| b.upsampler.is_some())
                .count();
        let (out_h, out_w) = (up * h, up * w);
        let c_rgba = self.conv_out.c_out;
        let c_in = self.ups[BAND_FROM].c_in;
        let mut rgba = exec.alloc_u8(out_h * out_w * c_rgba)?;
        let mut y0 = 0;
        while y0 < h {
            let core = band.max(1).min(h - y0);
            let ya = y0.saturating_sub(BAND_HALO);
            let yb = (y0 + core + BAND_HALO).min(h);
            let rows = yb - ya;
            let mut xb = exec.alloc(rows * w * c_in)?;
            exec.copy_region(&x, ya * w * c_in, &mut xb, 0, rows * w * c_in)?;
            let (mut hb, mut wb) = (rows, w);
            for blk in &self.ups[BAND_FROM..] {
                (xb, hb, wb) = self.up_block(xb, hb, wb, blk)?;
            }
            let band_rgba = self.head(&xb, hb, wb)?;
            drop(xb);
            let nb = hb * wb * c_rgba;
            let mut band_u8 = exec.alloc_u8(nb)?;
            exec.dit_to_u8(&band_rgba, &mut band_u8, nb)?;
            let (r0, rn) = (up * (y0 - ya), up * core);
            exec.copy_region(
                &band_u8,
                r0 * out_w * c_rgba,
                &mut rgba,
                up * y0 * out_w * c_rgba,
                rn * out_w * c_rgba,
            )?;
            y0 += core;
        }
        Ok(exec.to_host_u8_len(&rgba, out_h * out_w * c_rgba)?)
    }

    /// Band height at the 4x-latent resolution the bands are cut at, for a
    /// picture `lw` latents wide: the widest band plane (the up3 output -
    /// 288 channels of f32 at full resolution, four rows per band row) stays
    /// under `BAND_PLANE_BYTES`; never under 16 rows, or the halo would
    /// outweigh the core.
    fn band_rows(&self, lw: usize) -> usize {
        let out_w = super::VAE_SCALE * lw;
        let c = self.ups[BAND_FROM + 1].c_out;
        let rows_out = BAND_PLANE_BYTES / (out_w * c * 4).max(1);
        (rows_out / 4).max(16)
    }

    /// One up block computed in row bands into a full output plane: the same
    /// halo argument as the tail's, for one block alone - three residual
    /// blocks (6 rows) and the upsampler's conv (1 row at 2x) reach back 7
    /// rows of the input. The band height keeps the block's widest band
    /// plane (its output, `c_out` channels at 2x) under `BAND_PLANE_BYTES`.
    fn banded_up_block(
        &self,
        x: &CudaSlice<f32>,
        h: usize,
        w: usize,
        blk: &UpBlock,
    ) -> Result<(CudaSlice<f32>, usize, usize), GpuModelError> {
        const HALO: usize = 7;
        let exec = &self.exec;
        let up = if blk.upsampler.is_some() { 2 } else { 1 };
        let (out_h, out_w) = (up * h, up * w);
        let band = (BAND_PLANE_BYTES / (out_w * blk.c_out * 4).max(1) / up).max(16);
        let mut out = exec.alloc(out_h * out_w * blk.c_out)?;
        let mut y0 = 0;
        while y0 < h {
            let core = band.min(h - y0);
            let ya = y0.saturating_sub(HALO);
            let yb = (y0 + core + HALO).min(h);
            let rows = yb - ya;
            let mut xb = exec.alloc(rows * w * blk.c_in)?;
            exec.copy_region(x, ya * w * blk.c_in, &mut xb, 0, rows * w * blk.c_in)?;
            let (yb_plane, _, _) = self.up_block(xb, rows, w, blk)?;
            let (r0, rn) = (up * (y0 - ya), up * core);
            exec.copy_region(
                &yb_plane,
                r0 * out_w * blk.c_out,
                &mut out,
                up * y0 * out_w * blk.c_out,
                rn * out_w * blk.c_out,
            )?;
            y0 += core;
        }
        Ok((out, out_h, out_w))
    }

    /// One up block: three residual blocks, then (when it ups) the 2x nearest
    /// upsample + conv with the DupUp shortcut of the block's input added.
    /// Returns the plane and its geometry after the block.
    fn up_block(
        &self,
        x: CudaSlice<f32>,
        h: usize,
        w: usize,
        blk: &UpBlock,
    ) -> Result<(CudaSlice<f32>, usize, usize), GpuModelError> {
        let exec = &self.exec;
        let mut y = resblock(exec, &x, h, w, &blk.resnets[0])?;
        for r in &blk.resnets[1..] {
            y = resblock(exec, &y, h, w, r)?;
        }
        let (Some(up), Some((ft, repeats))) = (&blk.upsampler, blk.dupup) else {
            return Ok((y, h, w));
        };
        let mut y16 = exec.alloc_f16(h * w * blk.c_out)?;
        exec.convert_f32_f16(&y, &mut y16, h * w * blk.c_out)?;
        drop(y);
        let mut up_out = conv3(exec, &y16, h, w, up, Conv3Mode::Up2, None)?;
        drop(y16);
        exec.vae_dupup_add(&mut up_out, &x, h, w, blk.c_in, blk.c_out, ft, repeats)?;
        Ok((up_out, 2 * h, 2 * w))
    }

    /// norm_out + conv_out: the f32 RGBA plane, before the 8-bit quantise.
    fn head(
        &self,
        x: &CudaSlice<f32>,
        h: usize,
        w: usize,
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        let exec = &self.exec;
        let c = self.conv_out.c_in;
        let mut n16 = exec.alloc_f16(h * w * c)?;
        exec.vae_norm_f16(x, &self.norm_out, &mut n16, h * w, c, true)?;
        conv3(exec, &n16, h, w, &self.conv_out, Conv3Mode::Same, None)
    }
}
