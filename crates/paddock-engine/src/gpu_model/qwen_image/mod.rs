//! Qwen-Image-2.1 - the first image-generation family: text-to-image, and
//! editing with reference pictures.
//!
//! The parts: the single-stream DiT (a metadata-less GGUF, the unsloth
//! conversion, identified by tensor names - `load.rs`), the Qwen3-VL-8B text
//! encoder (the `qwen3vl` text stack the qwen3_asr family already runs,
//! tapped BEFORE its final norm - `text.rs`), the Wan-shaped residual VAE
//! decoder (`vae.rs`) and encoder (`vae_encode.rs`, the official F32
//! safetensors for both), and for editing the Qwen3-VL vision tower (the
//! qwen35 family's, with its DeepStack taps). `dit.rs` is the forward,
//! `sampler.rs` the flow-matching schedule, the seed's noise and the Euler
//! update.
//!
//! The serving shape (the model's own headline feature, made structural):
//! every prefix token - the text, and the reference pictures' latents at
//! their `<|image_pad|>` slots - modulates from t = 0 and sits under a
//! block-causal mask (text causal, each picture bidirectional within
//! itself), so the prefix never depends on the target or the step. It runs
//! ONCE per request through the 32 blocks and its post-rope K/V are kept per
//! layer; each of the 40 steps is then a target-only forward whose attention
//! reads `[prefix K/V | own K/V]` with nothing masked. Same math as
//! diffusers' extract-then-cached loop with one difference: step 1 is
//! uniform too. The text encoder reads the same pictures through the tower
//! (multi-axis rope, DeepStack into its first three layers), so the prompt
//! the DiT conditions on has seen them twice, as the model was trained.
//!
//! Numerics: f32 activations, the block GEMMs on the int8 (Q8_0) / W4A8
//! (k-quant) prefill lanes, attention on the f16-mma vision kernel with f32
//! accumulate, latents f32 throughout (diffusers stores them bf16 between
//! steps and rounds the timestep to bf16 before the sinusoid; sd.cpp keeps
//! f32 - so does this, being the same-weights reference we gate against).
//! Fixed-order everywhere: same seed, same file, same box -> same bytes.

mod dit;
mod load;
mod sampler;
mod text;
mod vae;
mod vae_encode;
mod vae_ops;

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::mapped::MappedGguf;

use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;

pub use dit::{DitModel, Prefix, PrefixSegment};
pub use load::is_dit_gguf;
pub use sampler::{Schedule, guidance_combine, mu_for_tokens};
pub use text::{TextEncoder, image_pad_runs, mrope_positions, t2i_prompt, ti2i_prompt};
pub use vae::VaeDecoder;
pub use vae_encode::{IMAGE_CHANNELS, VaeEncoder};

pub use crate::image::model::{
    AXES_DIMS_ROPE, GenerateRequest, LATENT_CHANNELS, ROPE_THETA, Reference, Rgba, SIZE_MULTIPLE,
    VAE_SCALE,
};

/// The editing lane's two extra parts, loaded together when the mmproj is
/// wired: the VAE encoder (a reference to its latents) and the Qwen3-VL
/// vision tower (a reference to the text encoder's image tokens).
pub struct EditLane {
    pub encoder: VaeEncoder,
    pub tower: crate::gpu_model::qwen35::vision::VisionModel,
}

/// Stage dump for bisecting a bad render: `PADDOCK_QI_DUMP=1` prints each
/// tagged plane's finite sum, largest magnitude and non-finite count, which
/// is how a NaN is walked back to the kernel that made it (the deepseek-ocr
/// and muse towers have the same instrument). A dev switch, sealed out of a
/// hardened build like every other.
pub(crate) fn dump(exec: &GpuExecutor, tag: &str, buf: &CudaSlice<f32>, n: usize) {
    if paddock_models::dev_var_os!("PADDOCK_QI_DUMP").is_none() {
        return;
    }
    match exec.to_host_len(buf, n) {
        Ok(h) => {
            let bad = h.iter().filter(|v| !v.is_finite()).count();
            let (mut s, mut mx) = (0f64, 0f32);
            for &v in h.iter().filter(|v| v.is_finite()) {
                s += v as f64;
                mx = mx.max(v.abs());
            }
            eprintln!("qi {tag:>18}: n={n:<10} sum={s:.4} max|x|={mx:.4} nonfinite={bad}");
        }
        Err(e) => eprintln!("qi {tag:>18}: readback failed: {e}"),
    }
}

/// The served family: DiT + text encoder + VAE on one executor, and the
/// editing lane when its tower is wired.
pub struct QwenImage {
    exec: Arc<GpuExecutor>,
    pub dit: DitModel,
    pub text: TextEncoder,
    pub vae: VaeDecoder,
    pub edit: Option<EditLane>,
}

impl QwenImage {
    /// Load every part. `dit` is the DiT GGUF, `text_encoder` the Qwen3-VL
    /// GGUF (text stack), `vae` the official safetensors file, `mmproj` the
    /// Qwen3-VL vision tower - with it the editing lane is served (the VAE
    /// encoder loads beside it), without it text-to-image only. `text_ctx`
    /// bounds the encoder's prompt, pictures included.
    pub fn load(
        exec: Arc<GpuExecutor>,
        dit: &Path,
        text_encoder: &Path,
        vae: &Path,
        mmproj: Option<&Path>,
        text_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        if !exec.has_dit() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack has no image-generation glue (slots 632-643) - rebuild it".into(),
            ));
        }
        let dit_map = MappedGguf::open(dit).map_err(crate::gpu::GpuError::from)?;
        let te_map = MappedGguf::open(text_encoder).map_err(crate::gpu::GpuError::from)?;
        let text = TextEncoder::load(exec.clone(), &te_map, text_ctx)?;
        let dit = DitModel::load(exec.clone(), &dit_map)?;
        let vae_dec = VaeDecoder::load(exec.clone(), vae)?;
        let edit = match mmproj {
            Some(p) => {
                let mm = MappedGguf::open(p).map_err(crate::gpu::GpuError::from)?;
                let proj = mm
                    .gguf()
                    .metadata
                    .get("clip.projector_type")
                    .and_then(paddock_models::gguf::Value::as_str)
                    .unwrap_or("");
                if proj != "qwen3vl_merger" {
                    return Err(GpuModelError::Unsupported(format!(
                        "mmproj is not a Qwen3-VL vision tower (projector_type {proj:?})"
                    )));
                }
                let mut tower =
                    crate::gpu_model::qwen35::vision::VisionModel::load(exec.clone(), &mm)?;
                if tower.deepstack_taps() == 0 {
                    return Err(GpuModelError::Unsupported(
                        "mmproj carries no DeepStack taps - Qwen3-VL's tower has three".into(),
                    ));
                }
                // Qwen3-VL resizes its learned pos-embd grid corner-aligned
                tower.set_align_corners(true);
                Some(EditLane {
                    encoder: VaeEncoder::load(exec.clone(), vae)?,
                    tower,
                })
            }
            None => None,
        };
        exec.release_staging();
        Ok(Self {
            exec,
            dit,
            text,
            vae: vae_dec,
            edit,
        })
    }

    /// Resident weight bytes, every part.
    pub fn weights_bytes(&self) -> u64 {
        self.dit.weights_bytes
            + self.text.weights_bytes()
            + self.vae.weights_bytes
            + self.edit_bytes()
    }

    /// The editing lane's own bytes (tower + VAE encoder), 0 without it.
    pub fn edit_bytes(&self) -> u64 {
        self.edit.as_ref().map_or(0, |e| {
            e.encoder.weights_bytes + e.tower.weight_bytes() as u64
        })
    }

    /// Whether reference pictures can be taken (the tower is wired).
    pub fn can_edit(&self) -> bool {
        self.edit.is_some()
    }

    /// The same bytes per part - `(dit, text encoder, vae)`. The catalog
    /// prices the three as separate artifacts (the DiT is the weights, the
    /// other two are companions), so the ledger has to be readable that way.
    pub fn weights_breakdown(&self) -> (u64, u64, u64) {
        (
            self.dit.weights_bytes,
            self.text.weights_bytes(),
            self.vae.weights_bytes,
        )
    }

    /// Text-to-image. Sizes are floored to the 32-multiple the pipeline uses.
    pub fn generate(&mut self, req: &GenerateRequest<'_>) -> Result<Rgba, GpuModelError> {
        self.generate_with(req, 0, &mut |_, _| Ok(()))
    }

    /// [`Self::generate`] with progressive previews: `partials` pictures of
    /// the x0 estimate along the way, handed to `on_partial` with their
    /// index. The estimate at a step is `x - sigma * v` - flow matching's
    /// prediction of the clean image from the current noisy latents and the
    /// velocity the model just produced - decoded with the same VAE as the
    /// final picture, so a preview is what the render is converging on, not
    /// a blurred stand-in. Previews land at evenly spaced steps (a third,
    /// two thirds ... of the way); each costs one decode, ~0.4 s at 1024^2
    /// against ~1 s a step. `on_partial` returning an error stops the render
    /// - that is how a caller who went away takes its request with it.
    pub fn generate_with(
        &mut self,
        req: &GenerateRequest<'_>,
        partials: usize,
        on_partial: &mut dyn FnMut(usize, Rgba) -> Result<(), GpuModelError>,
    ) -> Result<Rgba, GpuModelError> {
        self.generate_controlled(req, partials, on_partial, &|| false)
    }

    pub fn generate_controlled(
        &mut self,
        req: &GenerateRequest<'_>,
        partials: usize,
        on_partial: &mut dyn FnMut(usize, Rgba) -> Result<(), GpuModelError>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Rgba, GpuModelError> {
        let width = req.width / SIZE_MULTIPLE * SIZE_MULTIPLE;
        let height = req.height / SIZE_MULTIPLE * SIZE_MULTIPLE;
        if width == 0 || height == 0 {
            return Err(GpuModelError::Unsupported(format!(
                "size {}x{} is below the {SIZE_MULTIPLE}-pixel minimum",
                req.width, req.height
            )));
        }
        if req.steps == 0 {
            return Err(GpuModelError::Unsupported(
                "steps must be at least 1".into(),
            ));
        }
        let (lw, lh) = (width / VAE_SCALE, height / VAE_SCALE);
        let n_tokens = lw * lh;

        // 0. the references, each to its latents (the VAE encoder) and the
        //    text encoder's view of it (the tower) - the same pictures serve
        //    the negative prompt under guidance
        let refs = self.encode_references(req.references)?;

        // 1. the prefix (text, and the pictures' latents at their slots)
        //    K/V, once per prompt
        let cond = self.encode_prefix_vl(req.prompt_ids, req.drop, req.image_pad_id, &refs)?;
        let uncond = match (req.guidance > 1.0, req.negative) {
            (true, Some((ids, drop))) => {
                Some(self.encode_prefix_vl(ids, drop, req.image_pad_id, &refs)?)
            }
            _ => None,
        };
        drop(refs);

        // 2. the schedule and the seed's noise
        let sched = Schedule::new(req.steps, mu_for_tokens(n_tokens));
        let exec = self.exec.clone();
        let mut latents = exec.alloc(n_tokens * LATENT_CHANNELS)?;
        exec.dit_philox_randn(
            &mut latents,
            req.seed,
            req.noise_offset,
            n_tokens,
            LATENT_CHANNELS,
        )?;

        // 3. the denoising loop: one target-only forward per step (two under
        //    guidance), Euler in f32
        let len = n_tokens * LATENT_CHANNELS;
        let mut v_cond = exec.alloc(len)?;
        let mut v_uncond = exec.alloc(len)?;
        let longest = cond.len.max(uncond.as_ref().map_or(0, |u| u.len));
        self.dit.prepare_target(lw, lh, longest, cond.frame)?;
        // the steps a preview is decoded AFTER: k/(p+1) of the way for
        // k = 1..p, on the step whose velocity is in hand; a short render
        // that cannot space them out gets fewer (deduplicated), never two
        // of the same picture
        let mut preview_after: Vec<usize> = (1..=partials)
            .map(|k| (k * req.steps / (partials + 1)).max(1) - 1)
            .collect();
        preview_after.dedup();
        for i in 0..req.steps {
            if cancelled() {
                return Err(GpuModelError::Unsupported("client went away".into()));
            }
            let sigma = sched.sigmas[i];
            let dt = sched.sigmas[i + 1] - sigma;
            self.dit.step(&cond, &latents, sigma, &mut v_cond)?;
            let guided = if let Some(u) = &uncond {
                self.dit.step(u, &latents, sigma, &mut v_uncond)?;
                guidance_combine(&exec, &mut v_uncond, &v_cond, req.guidance, len)?;
                true
            } else {
                false
            };
            let v = if guided { &v_uncond } else { &v_cond };
            if let Some(k) = preview_after.iter().position(|&s| s == i) {
                // x0 = x - sigma * v, decoded as it stands
                let mut x0 = exec.alloc(len)?;
                exec.copy_region(&latents, 0, &mut x0, 0, len)?;
                exec.scale_add(&mut x0, v, -sigma, len)?;
                let pixels = self.vae.decode(&x0, lw, lh)?;
                on_partial(
                    k,
                    Rgba {
                        width,
                        height,
                        pixels,
                    },
                )?;
            }
            exec.scale_add(&mut latents, v, dt, len)?;
        }

        // 4. decode
        dump(&exec, "latents", &latents, n_tokens * LATENT_CHANNELS);
        let pixels = self.vae.decode(&latents, lw, lh)?;
        Ok(Rgba {
            width,
            height,
            pixels,
        })
    }

    /// Encode a prompt's text into the DiT's prefix K/V.
    fn encode_prefix(&mut self, ids: &[u32], drop: usize) -> Result<Prefix, GpuModelError> {
        let hidden = self.text.encode(ids, drop)?;
        let rows = ids.len() - drop;
        self.dit.encode_prefix(&[PrefixSegment::Text {
            hidden: &hidden,
            row0: 0,
            rows,
        }])
    }

    /// Every reference through the editing lane's two encoders.
    fn encode_references(
        &self,
        references: &[Reference<'_>],
    ) -> Result<Vec<EncodedReference>, GpuModelError> {
        if references.is_empty() {
            return Ok(Vec::new());
        }
        let lane = self.edit.as_ref().ok_or_else(|| {
            GpuModelError::Unsupported(
                "editing needs the vision tower - start this model with its mmproj".into(),
            )
        })?;
        let exec = &self.exec;
        let mut out = Vec::with_capacity(references.len());
        for r in references {
            let (w, h) = (r.width, r.height);
            if !w.is_multiple_of(SIZE_MULTIPLE)
                || !h.is_multiple_of(SIZE_MULTIPLE)
                || w == 0
                || h == 0
            {
                return Err(GpuModelError::Unsupported(format!(
                    "reference picture {w}x{h} is not a multiple of {SIZE_MULTIPLE}"
                )));
            }
            if r.rgba.len() != w * h * IMAGE_CHANNELS {
                return Err(GpuModelError::Unsupported(format!(
                    "reference picture {w}x{h} carries {} values, not {}",
                    r.rgba.len(),
                    w * h * IMAGE_CHANNELS
                )));
            }
            let d = exec.to_device(r.rgba)?;
            let latents = lane.encoder.encode(&d, w, h)?;
            drop(d);
            // the tower sees the picture composited over white (diffusers
            // pastes the RGBA onto a white canvas for the VL copy), planar,
            // normalised by its own mean / std
            let (mean, std) = (lane.tower.image_mean, lane.tower.image_std);
            let mut planar = vec![0f32; 3 * w * h];
            for (i, px) in r.rgba.as_chunks::<IMAGE_CHANNELS>().0.iter().enumerate() {
                let a = px[3] * 0.5 + 0.5;
                for c in 0..3 {
                    let v = (px[c] * 0.5 + 0.5) * a + (1.0 - a);
                    planar[c * w * h + i] = (v - mean[c]) / std[c];
                }
            }
            let vision = lane.tower.encode(&planar, w, h)?;
            out.push(EncodedReference {
                latents,
                lw: w / VAE_SCALE,
                lh: h / VAE_SCALE,
                vision,
            });
        }
        Ok(out)
    }

    /// Encode a prompt with reference pictures into the DiT's prefix K/V:
    /// the text encoder reads the pictures through the tower at their
    /// `<|image_pad|>` runs, and the DiT prefix takes its hidden rows as
    /// text segments with each run replaced by the picture's latents. The
    /// text-only encode when there are none.
    fn encode_prefix_vl(
        &mut self,
        ids: &[u32],
        drop: usize,
        pad_id: u32,
        refs: &[EncodedReference],
    ) -> Result<Prefix, GpuModelError> {
        if refs.is_empty() {
            return self.encode_prefix(ids, drop);
        }
        let runs = image_pad_runs(ids, pad_id);
        if runs.len() != refs.len() {
            return Err(GpuModelError::Unsupported(format!(
                "{} reference pictures but the prompt carries {} image slots",
                refs.len(),
                runs.len()
            )));
        }
        for (r, &(off, len)) in refs.iter().zip(&runs) {
            if r.vision.nx * r.vision.ny != len {
                return Err(GpuModelError::Unsupported(format!(
                    "a {}x{} picture grid against an image slot of {len} tokens",
                    r.vision.nx, r.vision.ny
                )));
            }
            if off < drop {
                return Err(GpuModelError::Unsupported(
                    "an image slot inside the system block".into(),
                ));
            }
        }
        let images: Vec<&crate::gpu_model::qwen35::vision::VisionOutput> =
            refs.iter().map(|r| &r.vision).collect();
        let hidden = self.text.encode_vl(ids, drop, &images, &runs)?;
        let total = ids.len() - drop;
        let mut segments = Vec::with_capacity(2 * refs.len() + 1);
        let mut row = 0usize;
        for (r, &(off, len)) in refs.iter().zip(&runs) {
            let start = off - drop;
            if start > row {
                segments.push(PrefixSegment::Text {
                    hidden: &hidden,
                    row0: row,
                    rows: start - row,
                });
            }
            segments.push(PrefixSegment::Image {
                latents: &r.latents,
                lw: r.lw,
                lh: r.lh,
            });
            row = start + len;
        }
        if total > row {
            segments.push(PrefixSegment::Text {
                hidden: &hidden,
                row0: row,
                rows: total - row,
            });
        }
        self.dit.encode_prefix(&segments)
    }
}

/// A reference after the editing lane's encoders: its packed latents (the
/// DiT's block) and the tower's output (the text encoder's splice).
struct EncodedReference {
    latents: CudaSlice<f32>,
    lw: usize,
    lh: usize,
    vision: crate::gpu_model::qwen35::vision::VisionOutput,
}
