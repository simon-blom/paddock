//! Reference RGBA -> deterministic posterior mean. Shares the decoder's
//! bounded convolution staging; no host tensor inference or logvar allocation.
use super::vae::{LATENTS_MEAN, LATENTS_STD};
use super::vae_ops::{
    AttnBlock, Conv1, Conv3, Conv3Mode, Loader, ResBlock, attention, conv1, conv3, resblock,
};
use super::*;
use paddock_models::safetensors::SafetensorsFile;

struct Down {
    res: [ResBlock; 2],
    sample: Option<Conv3>,
    ci: usize,
    co: usize,
    ft: usize,
    fs: usize,
}

pub(super) struct Encoder {
    exec: Rc<Ops>,
    input: Conv3,
    downs: Vec<Down>,
    mid: [ResBlock; 2],
    attn: AttnBlock,
    norm: Tensor<f32>,
    output: Conv3,
    mean: Conv1,
    inv: Tensor<f32>,
    bias: Tensor<f32>,
}

impl Encoder {
    pub fn load(exec: Rc<Ops>, path: &Path) -> Result<Self> {
        let st = SafetensorsFile::open(path).map_err(model_error)?;
        let mut ld = Loader {
            exec: exec.clone(),
            st: &st,
            bytes: 0,
        };
        let input = ld.conv3("encoder.conv_in")?;
        if (input.c_in, input.c_out) != (4, 96) {
            return Err(error("unsupported image VAE encoder"));
        }
        let mut downs = Vec::new();
        for (i, (ci, co)) in [(96, 96), (96, 192), (192, 384), (384, 768), (768, 768)]
            .into_iter()
            .enumerate()
        {
            downs.push(Down {
                res: [
                    ld.res(&format!("encoder.down_blocks.{i}.resnets.0"), ci, co)?,
                    ld.res(&format!("encoder.down_blocks.{i}.resnets.1"), co, co)?,
                ],
                sample: if i < 4 {
                    Some(ld.conv3(&format!("encoder.down_blocks.{i}.downsampler.resample.1"))?)
                } else {
                    None
                },
                ci,
                co,
                ft: if i > 0 && i < 4 { 2 } else { 1 },
                fs: if i < 4 { 2 } else { 1 },
            });
        }
        let mid = [
            ld.res("encoder.mid_block.resnets.0", 768, 768)?,
            ld.res("encoder.mid_block.resnets.1", 768, 768)?,
        ];
        let attn = ld.attn("encoder.mid_block.attentions.0", 768)?;
        let norm = ld.vec("encoder.norm_out.gamma")?;
        let output = ld.conv3("encoder.conv_out")?;
        let mean = ld.conv1("quant_conv", Some(64))?;
        if norm.len() != 768 * 4
            || (output.c_in, output.c_out) != (768, 128)
            || (mean.c_in, mean.c_out) != (128, 64)
        {
            return Err(error("invalid VAE posterior geometry"));
        }
        let inv = exec.to_device(&LATENTS_STD.map(|v| 1. / v))?;
        let bias = exec.to_device(
            &LATENTS_MEAN
                .iter()
                .zip(LATENTS_STD)
                .map(|(m, s)| -m / s)
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            exec,
            input,
            downs,
            mid,
            attn,
            norm,
            output,
            mean,
            inv,
            bias,
        })
    }

    pub fn encode(
        &self,
        rgba: &Tensor<f32>,
        width: usize,
        height: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Tensor<f32>> {
        if width == 0
            || height == 0
            || width > 2752
            || height > 2752
            || !width.is_multiple_of(32)
            || !height.is_multiple_of(32)
            || rgba.len() != width * height * 4 * 4
        {
            return Err(error("invalid reference-image VAE dimensions"));
        }
        check_cancelled(cancelled)?;
        let e = &self.exec;
        let (mut h, mut w) = (height, width);
        let mut input = e.alloc_f16(h * w * 4)?;
        e.convert_f32_f16(rgba, &mut input, h * w * 4)?;
        let mut x = conv3(e, &input, h, w, &self.input, Conv3Mode::Same, None)?;
        drop(input);
        for b in &self.downs {
            check_cancelled(cancelled)?;
            let y = resblock(e, &x, h, w, &b.res[0])?;
            let y = resblock(e, &y, h, w, &b.res[1])?;
            let mut y = if let Some(cv) = &b.sample {
                let mut half = e.alloc_f16(h * w * b.co)?;
                e.convert_f32_f16(&y, &mut half, h * w * b.co)?;
                drop(y);
                conv3(e, &half, h, w, cv, Conv3Mode::Down2, None)?
            } else {
                y
            };
            let count = h / b.fs * (w / b.fs) * b.co;
            e.run(
                "qi_avgdown",
                &[&y, &x],
                &[
                    h as u32,
                    w as u32,
                    b.ci as u32,
                    b.co as u32,
                    b.ft as u32,
                    b.fs as u32,
                ],
                [count.div_ceil(256), 1, 1],
                256,
            )?;
            h /= b.fs;
            w /= b.fs;
            std::mem::swap(&mut x, &mut y);
        }
        check_cancelled(cancelled)?;
        x = resblock(e, &x, h, w, &self.mid[0])?;
        x = attention(e, &x, h, w, &self.attn)?;
        x = resblock(e, &x, h, w, &self.mid[1])?;
        let mut half = e.alloc_f16(h * w * 768)?;
        e.vae_norm_f16(&x, &self.norm, &mut half, h * w, 768, true)?;
        drop(x);
        let moments = conv3(e, &half, h, w, &self.output, Conv3Mode::Same, None)?;
        drop(half);
        let mut half = e.alloc_f16(h * w * 128)?;
        e.convert_f32_f16(&moments, &mut half, h * w * 128)?;
        drop(moments);
        let mut z = conv1(e, &half, h * w, &self.mean)?;
        e.dit_affine_cols(&mut z, &self.inv, &self.bias, h * w, 64)?;
        Ok(z)
    }
}
