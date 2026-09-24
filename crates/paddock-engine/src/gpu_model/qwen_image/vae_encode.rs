//! The VAE encoder - the editing lane's way in: a reference picture to the
//! packed, normalised latent the DiT conditions on. Same file and family as
//! the decoder (`AutoencoderKLQwenImage21`, one frame), mirrored:
//!   conv_in 4 -> 96; four residual down blocks 96 -> 96 (2x), 96 -> 192
//!   (2x), 192 -> 384 (2x), 384 -> 768 (2x) and one 768 -> 768 without a
//!   downsample; mid (res, attn, res) @ 768; norm_out + SiLU; conv_out 768
//!   -> 128 (mean | logvar); quant_conv 128 -> 128.
//! Each down block is two residual blocks then (when it downs) diffusers'
//! Resample - ZeroPad2d((0, 1, 0, 1)) and a stride-2 3x3 - with the
//! AvgDown shortcut of the block's input added: space-to-channel with a
//! zero frame padded in FRONT when the block is temporal (all but the
//! first and last), then a group mean - so half of a temporal block's
//! shortcut channels are zero, which is the checkpoint's own arithmetic.
//!
//! The pipeline takes the distribution's MEAN (`sample_mode="argmax"`), so
//! only the first 64 output channels of quant_conv are ever computed; the
//! logvar half is not loaded. The latent is then `(z - mean) / std` per
//! channel with the config's vectors - the inverse of what the decoder
//! undoes - and packed one token per latent pixel, raster order, exactly
//! the layout the sampler's latents already have.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::safetensors::SafetensorsFile;

use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::vae::{LATENTS_MEAN, LATENTS_STD};
use super::vae_ops::{
    AttnBlock, Conv1, Conv3, Conv3Mode, Loader, ResBlock, attention, conv1, conv3, resblock,
};
use super::{LATENT_CHANNELS, VAE_SCALE};

/// Channels of a reference picture as the VAE reads it: RGBA in [-1, 1].
pub const IMAGE_CHANNELS: usize = 4;

struct DownBlock {
    resnets: Vec<ResBlock>,
    /// the Resample conv (stride 2 after the zero pad), when the block downs
    downsampler: Option<Conv3>,
    c_in: usize,
    c_out: usize,
    /// AvgDown geometry: temporal factor (2 pads a zero frame in front) and
    /// spatial factor (2 when the block downs)
    ft: usize,
    fs: usize,
}

pub struct VaeEncoder {
    exec: Arc<GpuExecutor>,
    conv_in: Conv3,
    downs: Vec<DownBlock>,
    mid_res: [ResBlock; 2],
    mid_attn: AttnBlock,
    norm_out: CudaSlice<f32>,
    conv_out: Conv3,
    /// the mean half of quant_conv: `[64][128]`
    quant_mean: Conv1,
    /// per-channel `1 / std` and `-mean / std`: the normalisation as one
    /// affine
    inv_std: CudaSlice<f32>,
    neg_mean_over_std: CudaSlice<f32>,
    pub weights_bytes: u64,
}

impl VaeEncoder {
    pub fn load(exec: Arc<GpuExecutor>, path: &Path) -> Result<Self, GpuModelError> {
        let st = SafetensorsFile::open(path)
            .map_err(|e| GpuModelError::Unsupported(format!("vae: {e}")))?;
        let mut ld = Loader {
            exec: exec.clone(),
            st: &st,
            bytes: 0,
        };
        let conv_in = ld.conv3("encoder.conv_in")?;
        let base = conv_in.c_out; // 96
        // dims = base * [1, 1, 2, 4, 8, 8]; temporal downs [F, T, T, T], the
        // last block neither downs nor is temporal
        let dims = [base, 2 * base, 4 * base, 8 * base, 8 * base];
        let temporal = [false, true, true, true];
        let mut downs = Vec::with_capacity(5);
        for i in 0..5 {
            let (c_in, c_out) = (if i == 0 { base } else { dims[i - 1] }, dims[i]);
            let down = i != 4;
            let mut resnets = Vec::with_capacity(2);
            let mut cur = c_in;
            for r in 0..2 {
                resnets.push(ld.res(
                    &format!("encoder.down_blocks.{i}.resnets.{r}"),
                    cur,
                    c_out,
                )?);
                cur = c_out;
            }
            let downsampler = if down {
                Some(ld.conv3(&format!("encoder.down_blocks.{i}.downsampler.resample.1"))?)
            } else {
                None
            };
            downs.push(DownBlock {
                resnets,
                downsampler,
                c_in,
                c_out,
                ft: if down && temporal[i] { 2 } else { 1 },
                fs: if down { 2 } else { 1 },
            });
        }
        let top = dims[4];
        let mid_res = [
            ld.res("encoder.mid_block.resnets.0", top, top)?,
            ld.res("encoder.mid_block.resnets.1", top, top)?,
        ];
        let mid_attn = ld.attn("encoder.mid_block.attentions.0", top)?;
        let norm_out = ld.vec("encoder.norm_out.gamma")?;
        let conv_out = ld.conv3("encoder.conv_out")?;
        if conv_out.c_out != 2 * LATENT_CHANNELS {
            return Err(GpuModelError::Unsupported(format!(
                "vae encoder.conv_out has {} channels, expected {} (mean | logvar)",
                conv_out.c_out,
                2 * LATENT_CHANNELS
            )));
        }
        let quant_mean = ld.conv1("quant_conv", Some(LATENT_CHANNELS))?;
        let inv_std: Vec<f32> = LATENTS_STD.iter().map(|s| 1.0 / s).collect();
        let neg: Vec<f32> = LATENTS_MEAN
            .iter()
            .zip(&LATENTS_STD)
            .map(|(m, s)| -m / s)
            .collect();
        let weights_bytes = ld.bytes;
        Ok(Self {
            inv_std: exec.to_device(&inv_std)?,
            neg_mean_over_std: exec.to_device(&neg)?,
            exec,
            conv_in,
            downs,
            mid_res,
            mid_attn,
            norm_out,
            conv_out,
            quant_mean,
            weights_bytes,
        })
    }

    /// Encode an NHWC f32 `[height][width][4]` picture (RGBA in [-1, 1],
    /// sides multiples of the 16-pixel latent scale) to packed, normalised
    /// latents `[(height / 16) * (width / 16)][64]`.
    pub fn encode(
        &self,
        rgba: &CudaSlice<f32>,
        width: usize,
        height: usize,
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        if !width.is_multiple_of(VAE_SCALE)
            || !height.is_multiple_of(VAE_SCALE)
            || width == 0
            || height == 0
        {
            return Err(GpuModelError::Unsupported(format!(
                "vae encode: {width}x{height} is not a multiple of {VAE_SCALE}"
            )));
        }
        let exec = &self.exec;
        let (mut h, mut w) = (height, width);
        let mut x16 = exec.alloc_f16(h * w * IMAGE_CHANNELS)?;
        exec.convert_f32_f16(rgba, &mut x16, h * w * IMAGE_CHANNELS)?;
        let mut x = conv3(exec, &x16, h, w, &self.conv_in, Conv3Mode::Same, None)?;
        drop(x16);
        super::dump(exec, "vae enc conv_in", &x, h * w * self.conv_in.c_out);
        for (bi, blk) in self.downs.iter().enumerate() {
            let mut y = resblock(exec, &x, h, w, &blk.resnets[0])?;
            for r in &blk.resnets[1..] {
                y = resblock(exec, &y, h, w, r)?;
            }
            let mut out = match &blk.downsampler {
                Some(down) => {
                    let mut y16 = exec.alloc_f16(h * w * blk.c_out)?;
                    exec.convert_f32_f16(&y, &mut y16, h * w * blk.c_out)?;
                    drop(y);
                    conv3(exec, &y16, h, w, down, Conv3Mode::Down2, None)?
                }
                None => y,
            };
            exec.vae_avgdown_add(&mut out, &x, h, w, blk.c_in, blk.c_out, blk.ft, blk.fs)?;
            h /= blk.fs;
            w /= blk.fs;
            x = out;
            super::dump(exec, &format!("vae enc down{bi}"), &x, h * w * blk.c_out);
        }
        x = resblock(exec, &x, h, w, &self.mid_res[0])?;
        x = attention(exec, &x, h, w, &self.mid_attn)?;
        x = resblock(exec, &x, h, w, &self.mid_res[1])?;
        super::dump(exec, "vae enc mid", &x, h * w * self.mid_attn.c);
        let p = h * w;
        let c = self.conv_out.c_in;
        let mut n16 = exec.alloc_f16(p * c)?;
        exec.vae_norm_f16(&x, &self.norm_out, &mut n16, p, c, true)?;
        drop(x);
        let moments = conv3(exec, &n16, h, w, &self.conv_out, Conv3Mode::Same, None)?;
        drop(n16);
        let mut m16 = exec.alloc_f16(p * self.conv_out.c_out)?;
        exec.convert_f32_f16(&moments, &mut m16, p * self.conv_out.c_out)?;
        drop(moments);
        // the mean half of quant_conv, then (z - mean) / std
        let mut z = conv1(exec, &m16, p, &self.quant_mean)?;
        exec.dit_affine_cols(
            &mut z,
            &self.inv_std,
            &self.neg_mean_over_std,
            p,
            LATENT_CHANNELS,
        )?;
        super::dump(exec, "vae enc latents", &z, p * LATENT_CHANNELS);
        Ok(z)
    }
}
