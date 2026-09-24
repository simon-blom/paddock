//! The text encoder: Qwen3-VL-8B-Instruct's text stack, which the qwen3_asr
//! family already loads and runs (arch `qwen3vl`, the equal-axis M-RoPE that
//! a text-only prompt collapses to plain rope). The DiT was trained on the
//! LAST layer's residual stream BEFORE the final RMSNorm - `hidden_states
//! [-1]` with the norm neutralised in diffusers, `out_layers = {num_layers}`
//! in sd.cpp - which is exactly what `prefill_body` leaves in its scratch.
//!
//! The prompt is a raw template, not the chat template:
//!   <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
//!   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
//! and the leading system tokens are dropped from the hidden states. The
//! count is whatever the tokenizer makes of the system block, never a
//! constant - the caller tokenizes both strings [`t2i_prompt`] returns and
//! passes the difference as `drop`. An empty prompt is encoded as one space.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen3_asr::{GpuQwen3Asr, VisionSplice};
use crate::gpu_model::qwen35::vision::VisionOutput;

const SYSTEM: &str = "Comprehend and analyze the provided prompt.";

/// The system block and the full templated prompt for text-to-image. Tokenize
/// both; `drop` = the system block's token count.
pub fn t2i_prompt(prompt: &str) -> (String, String) {
    let system = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n");
    let body = if prompt.is_empty() { " " } else { prompt };
    let full = format!("{system}<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n");
    (system, full)
}

/// The editing template: every reference is `<imageN><|vision_start|>
/// <|image_pad|><|vision_end|>` (a space before the second and later ones,
/// `<imageN>` plain text), then the instruction with no space between. ONE
/// `<|image_pad|>` per picture here; the caller expands each into the
/// picture's merged-grid token count (`(h / 32) * (w / 32)`), which the
/// tokenizer's single control id makes a splice rather than a re-tokenize.
pub fn ti2i_prompt(prompt: &str, n_images: usize) -> (String, String) {
    let system = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n");
    let body = if prompt.is_empty() { " " } else { prompt };
    let mut images = String::new();
    for i in 1..=n_images {
        if i > 1 {
            images.push(' ');
        }
        images.push_str(&format!(
            "<image{i}><|vision_start|><|image_pad|><|vision_end|>"
        ));
    }
    let full =
        format!("{system}<|im_start|>user\n{images}{body}<|im_end|>\n<|im_start|>assistant\n");
    (system, full)
}

/// Where a prompt's `<|image_pad|>` runs are: `(offset, len)` per run in
/// order, as the text encoder and the DiT prefix both need them.
pub fn image_pad_runs(ids: &[u32], pad_id: u32) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == pad_id {
            let start = i;
            while i < ids.len() && ids[i] == pad_id {
                i += 1;
            }
            runs.push((start, i - start));
        } else {
            i += 1;
        }
    }
    runs
}

/// Qwen3-VL's `get_rope_index` for one prompt, axis-major `[4][ids.len()]`
/// (t, h, w, e = t): a text run advances the cursor by one per token with
/// every axis at the cursor; an image of merged grid `(nx, ny)` sits at a
/// constant t with h / w over its grid, then the cursor advances by
/// `max(nx, ny)`. `runs` are the `<|image_pad|>` runs in order, each the
/// length of its grid.
pub fn mrope_positions(n: usize, runs: &[(usize, usize)], grids: &[(usize, usize)]) -> Vec<u32> {
    let mut pos = vec![0u32; 4 * n];
    let mut cursor = 0u32;
    let mut i = 0;
    let mut run = 0;
    while i < n {
        if run < runs.len() && runs[run].0 == i {
            let (off, len) = runs[run];
            let (nx, ny) = grids[run];
            debug_assert_eq!(len, nx * ny, "an image_pad run is its grid");
            for y in 0..ny {
                for x in 0..nx {
                    let r = off + y * nx + x;
                    pos[r] = cursor;
                    pos[n + r] = cursor + y as u32;
                    pos[2 * n + r] = cursor + x as u32;
                    pos[3 * n + r] = cursor;
                }
            }
            cursor += nx.max(ny) as u32;
            i = off + len;
            run += 1;
        } else {
            for axis in 0..4 {
                pos[axis * n + i] = cursor;
            }
            cursor += 1;
            i += 1;
        }
    }
    pos
}

pub struct TextEncoder {
    exec: Arc<GpuExecutor>,
    model: GpuQwen3Asr,
    pub hidden: usize,
}

impl TextEncoder {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        if map.gguf().architecture() != Some("qwen3vl") {
            return Err(GpuModelError::Unsupported(format!(
                "qwen-image text encoder: expected architecture qwen3vl, got {:?}",
                map.gguf().architecture()
            )));
        }
        let mut model = GpuQwen3Asr::load(exec.clone(), map, max_ctx)?;
        // f16 K/V: the ASR families default to fp8, and a conditioning
        // vector is not a place to spend precision on cache bytes
        model.set_kv_dtype(KvDtype::Fp16);
        let hidden = model.hp.n_embd;
        Ok(Self {
            exec,
            model,
            hidden,
        })
    }

    pub fn weights_bytes(&self) -> u64 {
        self.model.weights_bytes
    }

    /// Run `ids` through the stack and return rows `drop..` of the final
    /// residual stream, `[ids.len() - drop][hidden]` f32 on device.
    pub fn encode(&mut self, ids: &[u32], drop: usize) -> Result<CudaSlice<f32>, GpuModelError> {
        if ids.len() <= drop {
            return Err(GpuModelError::Unsupported(format!(
                "prompt of {} tokens is all system block ({drop})",
                ids.len()
            )));
        }
        self.encode_vl(ids, drop, &[], &[])
    }

    /// The same, with reference pictures: `images[i]` (the vision tower's
    /// output, DeepStack streams included) is spliced over `runs[i]` (the
    /// i-th `<|image_pad|>` run of `ids`, `(offset, len)`), and every row is
    /// rotated by Qwen3-VL's multi-axis position. Text-only when both are
    /// empty, which is then exactly [`Self::encode`].
    pub fn encode_vl(
        &mut self,
        ids: &[u32],
        drop: usize,
        images: &[&VisionOutput],
        runs: &[(usize, usize)],
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        if ids.len() <= drop {
            return Err(GpuModelError::Unsupported(format!(
                "prompt of {} tokens is all system block ({drop})",
                ids.len()
            )));
        }
        if images.len() != runs.len() {
            return Err(GpuModelError::Unsupported(format!(
                "{} reference pictures for {} image slots in the prompt",
                images.len(),
                runs.len()
            )));
        }
        let mut splices = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for (img, &(off, len)) in images.iter().zip(runs) {
            if img.nx * img.ny != len {
                return Err(GpuModelError::Unsupported(format!(
                    "a {}x{} picture grid against an image slot of {len} tokens",
                    img.nx, img.ny
                )));
            }
            splices.push(VisionSplice {
                off,
                n_tokens: len,
                embd: &img.embd,
                deepstack: &img.deepstack,
            });
            grids.push((img.nx, img.ny));
        }
        let mrope = (!images.is_empty()).then(|| mrope_positions(ids.len(), runs, &grids));
        // every prompt is its own sequence at slot 0: rewind the position
        // counter (the cache needs no clearing - every read is bounded)
        if let Some(ds) = self.model.decode.as_mut() {
            ds.pos = 0;
        }
        self.model
            .prefill_body(ids, &[], &splices, mrope.as_deref())?;
        let rows = ids.len() - drop;
        let sc = self.model.prefill.as_ref().expect("prefill scratch");
        let mut out = self.exec.alloc(rows * self.hidden)?;
        self.exec
            .copy_region(&sc.d_x, drop * self.hidden, &mut out, 0, rows * self.hidden)?;
        Ok(out)
    }
}
