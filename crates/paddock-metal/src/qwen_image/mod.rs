//! Native Qwen-Image-2.1: quantized DiT + Qwen3-VL conditioning + bounded VAE.
//! No Python, subprocess inference, or CPU tensor fallback.
mod conditioning;
#[cfg(test)]
mod contract_tests;
#[cfg(test)]
mod diagnostic;
mod dit;
#[cfg(test)]
mod edit_tests;
mod mlx;
mod ops;
#[cfg(test)]
mod tests;
mod text;
mod vae;
mod vae_encode;
mod vae_ops;
mod vision;
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use ops::{Ops, Tensor};
use paddock_engine::image::{
    ImageBackend,
    model::*,
    schedule::{Schedule, mu_for_tokens},
};
use std::{path::Path, rc::Rc};

fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(s.into())
}
fn model_error(e: impl std::fmt::Display) -> MetalError {
    error(e.to_string())
}
fn check_cancelled(f: &dyn Fn() -> bool) -> Result<()> {
    if f() {
        Err(error("client went away"))
    } else {
        Ok(())
    }
}
fn words(d: &MetalDevice, x: &[u32]) -> Result<Buffer> {
    let b = d.alloc(size_of_val(x))?;
    // Fresh storage, before its first submission.
    unsafe { b.write_u32(x) };
    Ok(b)
}
fn tiles(d: &MetalDevice, rows: usize) -> Result<Buffer> {
    words(
        d,
        &(0..rows)
            .step_by(32)
            .flat_map(|i| [i as u32, (rows - i).min(32) as u32])
            .collect::<Vec<_>>(),
    )
}
fn project(w: &Weight, cmd: &Commands<'_>, x: &Buffer, y: &Buffer, rows: usize, gemm: &Buffer) {
    match w.ty {
        30 => cmd.dispatch(
            "vis_bmm64",
            &[&w.buffer, x, y, &w.buffer],
            &[w.k as u32, w.n as u32, rows as u32, 0],
            [w.n.div_ceil(64), rows.div_ceil(64), 1],
            128,
        ),
        0 => cmd.dispatch(
            "qi_f32_mm",
            &[&w.buffer, x, y],
            &[w.k as u32, w.n as u32, rows as u32, 0],
            [w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        ),
        1 => cmd.dispatch(
            "vis_hmm64",
            &[&w.buffer, x, y, &w.buffer],
            &[w.k as u32, w.n as u32, rows as u32, 0],
            [w.n.div_ceil(64), rows.div_ceil(64), 1],
            128,
        ),
        _ => w.linear(cmd, x, y, rows, 1., gemm),
    }
}
fn dump(exec: &Ops, tag: &str, x: &Tensor<f32>, n: usize) {
    if paddock_models::dev_var_os!("PADDOCK_QI_DUMP").is_none() {
        return;
    }
    // Every VAE primitive completes before returning.
    let v = unsafe { x.read_f32(0, n) };
    tracing::debug!(
        tag,
        allocated = exec.device.allocated_bytes(),
        sum = v.iter().map(|&v| v as f64).sum::<f64>(),
        nonfinite = v.iter().filter(|v| !v.is_finite()).count(),
        "Metal image stage"
    );
}

pub struct QwenImage {
    exec: Rc<Ops>,
    dit: dit::Dit,
    text: text::TextEncoder,
    vae: vae::VaeDecoder,
    edit: Option<EditLane>,
    weights: u64,
}
struct EditLane {
    encoder: vae_encode::Encoder,
    tower: vision::Vision,
}
impl QwenImage {
    pub fn load_mlx(root: &Path, context: usize, budget: Option<u64>) -> Result<Self> {
        mlx::pipeline(root)?;
        let exec = Rc::new(Ops {
            device: MetalDevice::new(budget)?,
        });
        let text = text::TextEncoder::load_mlx(&exec, &root.join("text_encoder"), context)?;
        let dit = dit::Dit::load_mlx(&exec.device, &root.join("transformer"))?;
        let vae_config = mlx::config(
            &root.join("vae"),
            serde_json::json!({
                "_class_name":"AutoencoderKLQwenImage21", "mlx_format":true,
                "z_dim":64, "decoder_base_dim":144, "dim_mult":[1,2,4,8,8],
                "scale_factor_spatial":16, "is_residual":true, "num_res_blocks":2,
                "in_channels":4, "out_channels":4
            }),
        )?;
        for (key, expected) in [
            ("latents_mean", vae::LATENTS_MEAN),
            ("latents_std", vae::LATENTS_STD),
        ] {
            let values = vae_config[key]
                .as_array()
                .ok_or_else(|| error(format!("VAE {key}")))?;
            if values.len() != expected.len()
                || values
                    .iter()
                    .zip(expected)
                    .any(|(v, e)| v.as_f64().map(|x| x as f32) != Some(e))
            {
                return Err(error(format!(
                    "VAE {key} does not match the checkpoint contract"
                )));
            }
        }
        let vae = vae::VaeDecoder::load(exec.clone(), &root.join("vae/model.safetensors"))?;
        let edit = Some(EditLane {
            tower: vision::Vision::load_mlx(&exec.device, &root.join("text_encoder"))?,
            encoder: vae_encode::Encoder::load(exec.clone(), &root.join("vae/model.safetensors"))?,
        });
        let weights = exec.device.allocated_bytes();
        tracing::info!(
            weights_bytes = weights,
            "native Metal Qwen-Image MLX affine-4/group-64 loaded"
        );
        Ok(Self {
            exec,
            text,
            dit,
            vae,
            edit,
            weights,
        })
    }
    pub fn load(
        dit: &Path,
        text: &Path,
        vae: &Path,
        context: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        Self::load_with_vision(dit, text, vae, None, context, budget)
    }
    pub fn load_with_vision(
        dit: &Path,
        text: &Path,
        vae: &Path,
        mmproj: Option<&Path>,
        context: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let exec = Rc::new(Ops {
            device: MetalDevice::new(budget)?,
        });
        let text = text::TextEncoder::load(&exec, text, context)?;
        let dit = dit::Dit::load(&exec.device, dit)?;
        let edit = mmproj
            .map(|path| -> Result<EditLane> {
                Ok(EditLane {
                    tower: vision::Vision::load(&exec.device, path)?,
                    encoder: vae_encode::Encoder::load(exec.clone(), vae)?,
                })
            })
            .transpose()?;
        let vae = vae::VaeDecoder::load(exec.clone(), vae)?;
        let weights = exec.device.allocated_bytes();
        tracing::info!(
            weights_bytes = weights,
            vae_weights_bytes = vae.weights_bytes,
            "native Metal Qwen-Image loaded"
        );
        Ok(Self {
            exec,
            dit,
            text,
            vae,
            edit,
            weights,
        })
    }
    fn render(
        &mut self,
        req: &GenerateRequest<'_>,
        partials: usize,
        on_partial: &mut dyn FnMut(usize, Rgba) -> std::result::Result<(), String>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Rgba> {
        if req.width < 32
            || req.height < 32
            || req.width > 2752
            || req.height > 2752
            || !req.width.is_multiple_of(32)
            || !req.height.is_multiple_of(32)
            || !(1..=100).contains(&req.steps)
            || !req.guidance.is_finite()
            || !(0.0..=20.0).contains(&req.guidance)
            || partials > 3
        {
            return Err(error(
                "invalid image dimensions, steps, guidance or previews",
            ));
        }
        let (lw, lh) = (req.width / VAE_SCALE, req.height / VAE_SCALE);
        let rows = lw * lh;
        let exec = &self.exec;
        let d = &exec.device;
        check_cancelled(cancelled)?;
        // Reject malformed slots/context before expensive VAE/tower work.
        self.text.validate(req.prompt_ids, req.drop)?;
        if let Some((ids, drop)) = req.negative {
            self.text.validate(ids, drop)?;
        }
        if req.references.len() > 10 {
            return Err(error("at most ten reference pictures are supported"));
        }
        let mut grids = Vec::with_capacity(req.references.len());
        for r in req.references {
            if r.width == 0
                || r.height == 0
                || r.width > 2752
                || r.height > 2752
                || !r.width.is_multiple_of(32)
                || !r.height.is_multiple_of(32)
                || r.rgba.len() != r.width * r.height * 4
                || r.rgba
                    .iter()
                    .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
            {
                return Err(error("invalid reference RGBA pixels or dimensions"));
            }
            grids.push((r.width / 32, r.height / 32));
        }
        let runs = conditioning::slots(req.prompt_ids, req.drop, req.image_pad_id, &grids)?;
        let negative_runs = req
            .negative
            .map(|(ids, drop)| conditioning::slots(ids, drop, req.image_pad_id, &grids))
            .transpose()?;
        let mut reference_latents = Vec::new();
        let mut reference_vision = Vec::new();
        if !req.references.is_empty() {
            let lane = self
                .edit
                .as_ref()
                .ok_or_else(|| error("reference editing requires the Qwen3-VL vision companion"))?;
            for r in req.references {
                check_cancelled(cancelled)?;
                let pixels = exec.to_device(r.rgba)?;
                reference_latents.push(lane.encoder.encode(&pixels, r.width, r.height, cancelled)?);
                reference_vision.push(
                    lane.tower
                        .encode(exec, &pixels, r.width, r.height, cancelled)?,
                );
            }
        }
        let images = reference_vision.iter().collect::<Vec<_>>();
        let latent_grids = grids
            .iter()
            .map(|(w, h)| (w * 2, h * 2))
            .collect::<Vec<_>>();
        #[cfg(test)]
        let render_started = std::time::Instant::now();
        let hidden =
            self.text
                .encode_vl(exec, req.prompt_ids, req.drop, &images, &runs, cancelled)?;
        let segments = conditioning::segments(
            req.prompt_ids.len() - req.drop,
            req.drop,
            &runs,
            &latent_grids,
        );
        let prefix = self.dit.prefix_vl(
            exec,
            &hidden,
            req.prompt_ids.len() - req.drop,
            &segments,
            &reference_latents,
            cancelled,
        )?;
        drop(hidden);
        let uncond = if req.guidance > 1. {
            if let Some((ids, drop)) = req.negative {
                let runs = negative_runs.as_deref().unwrap_or(&[]);
                let hidden = self
                    .text
                    .encode_vl(exec, ids, drop, &images, runs, cancelled)?;
                let segments = conditioning::segments(ids.len() - drop, drop, runs, &latent_grids);
                Some(self.dit.prefix_vl(
                    exec,
                    &hidden,
                    ids.len() - drop,
                    &segments,
                    &reference_latents,
                    cancelled,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        drop(images);
        drop(reference_vision);
        drop(reference_latents);
        let mut latents = exec.alloc::<f32>(rows * 64)?;
        exec.run(
            "qi_noise",
            &[&latents],
            &[
                rows as u32,
                req.seed as u32,
                (req.seed >> 32) as u32,
                req.noise_offset,
            ],
            [(rows * 64).div_ceil(256), 1, 1],
            256,
        )?;
        let mut velocity = exec.alloc::<f32>(rows * 64)?;
        let vc = exec.alloc::<f32>(rows * 64)?;
        let sc = dit::Scratch::new(
            d,
            rows,
            prefix.len.max(uncond.as_ref().map_or(0, |p| p.len)),
        )?;
        #[cfg(test)]
        if std::env::var_os("PADDOCK_QI_TIMING").is_some() {
            eprintln!(
                "image conditioning/setup {:.6}s",
                render_started.elapsed().as_secs_f64()
            );
        }
        let sched = Schedule::new(req.steps, mu_for_tokens(rows));
        let mut previews = (1..=partials)
            .map(|k| (k * req.steps / (partials + 1)).max(1) - 1)
            .collect::<Vec<_>>();
        previews.dedup();
        for i in 0..req.steps {
            check_cancelled(cancelled)?;
            let started = std::time::Instant::now();
            self.dit.step(
                exec,
                &prefix,
                &latents,
                &sc,
                sched.sigmas[i],
                lw,
                lh,
                &velocity,
                cancelled,
            )?;
            if let Some(u) = &uncond {
                self.dit.step(
                    exec,
                    u,
                    &latents,
                    &sc,
                    sched.sigmas[i],
                    lw,
                    lh,
                    &vc,
                    cancelled,
                )?;
                // vc holds unconditional; combine into it, then copy to the update plane.
                exec.run(
                    "qi_guidance",
                    &[&vc, &velocity],
                    &[(rows * 64) as u32, req.guidance.to_bits()],
                    [(rows * 64).div_ceil(256), 1, 1],
                    256,
                )?;
                exec.copy_region(&vc, 0, &mut velocity, 0, rows * 64)?;
            }
            tracing::debug!(
                step = i + 1,
                total = req.steps,
                elapsed_ms = started.elapsed().as_millis(),
                "Metal image denoise"
            );
            #[cfg(test)]
            if std::env::var_os("PADDOCK_QI_TIMING").is_some() {
                eprintln!("image step {i} {:.6}s", started.elapsed().as_secs_f64());
            }
            if let Some(index) = previews.iter().position(|&s| s == i) {
                check_cancelled(cancelled)?;
                let mut x0 = exec.alloc(rows * 64)?;
                exec.copy_region(&latents, 0, &mut x0, 0, rows * 64)?;
                exec.scale_add(&mut x0, &velocity, -sched.sigmas[i], rows * 64)?;
                let pixels = self.vae.decode(&x0, lw, lh)?;
                on_partial(
                    index,
                    Rgba {
                        width: req.width,
                        height: req.height,
                        pixels,
                    },
                )
                .map_err(error)?;
            }
            exec.scale_add(
                &mut latents,
                &velocity,
                sched.sigmas[i + 1] - sched.sigmas[i],
                rows * 64,
            )?;
        }
        drop(sc);
        drop(prefix);
        drop(uncond);
        drop(velocity);
        drop(vc);
        check_cancelled(cancelled)?;
        #[cfg(test)]
        let decode_started = std::time::Instant::now();
        let pixels = self.vae.decode(&latents, lw, lh)?;
        #[cfg(test)]
        if std::env::var_os("PADDOCK_QI_TIMING").is_some() {
            eprintln!("image VAE {:.6}s", decode_started.elapsed().as_secs_f64());
        }
        Ok(Rgba {
            width: req.width,
            height: req.height,
            pixels,
        })
    }
}
impl ImageBackend for QwenImage {
    fn weights_bytes(&self) -> u64 {
        self.weights
    }
    fn can_edit(&self) -> bool {
        self.edit.is_some()
    }
    fn generate(
        &mut self,
        req: &GenerateRequest<'_>,
        partials: usize,
        on_partial: &mut dyn FnMut(usize, Rgba) -> std::result::Result<(), String>,
        cancelled: &dyn Fn() -> bool,
    ) -> std::result::Result<Rgba, String> {
        self.render(req, partials, on_partial, cancelled)
            .map_err(|e| e.to_string())
    }
}
