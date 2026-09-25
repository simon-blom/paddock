//! Native dense Gemma 4 SigLIP tower. BF16 weights remain BF16, residuals
//! remain F32, and patchify/pooling/standardization all execute on the GPU.
//! Ragged images share projections, but not bidirectional attention domains.
use super::*;
use paddock_engine::generator::VisionBudget;
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;

const E: usize = 1152;
const F: usize = 4304;
#[path = "vision_mlx.rs"]
mod mlx;
pub(super) use mlx::resize as resize_mlx;
pub(super) const MAX_PATCHES: usize = 10_080;
pub(super) const BUDGET: VisionBudget = VisionBudget {
    min_pixels: 70 * 2304,
    max_pixels: 280 * 2304,
    max_edge: None,
    pixels_per_token: 2304,
    min_tokens: 70,
    max_tokens: 280,
};

struct Block {
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    q_norm: Weight,
    k_norm: Weight,
    out: Weight,
    post: Weight,
    ffn_norm: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    ffn_post: Weight,
}
pub(super) struct Vision {
    mlx: bool,
    patch: Weight,
    pos: Weight,
    bias: Weight,
    scale: Weight,
    projection: Weight,
    blocks: Vec<Block>,
    eps: f32,
    quick_gelu: bool,
}
pub(super) struct Output {
    pub(super) embd: Buffer,
    pub(super) tokens: usize,
}
pub(super) struct Job {
    workspace: Buffer,
    grids: Vec<(usize, usize)>,
    rows: usize,
    x: Buffer,
    stage: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qh: Buffer,
    kh: Buffer,
    vh: Buffer,
    xy: Buffer,
    tiles: Buffer,
    tile_count: usize,
    attn: Buffer,
    delta: Buffer,
    gate: Buffer,
    up: Buffer,
    layer: usize,
    pub(super) cost: Duration,
    pub(super) gpu_seconds: f64,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(s.into())
}
fn channels(value: Option<&Value>, expected: f32) -> bool {
    matches!(value, Some(Value::Array(v)) if v.len() == 3 && v.iter().all(|v| v.as_f32() == Some(expected)))
}

pub(super) fn resize(w: usize, h: usize) -> Result<(usize, usize)> {
    // Bounds also guarantee the GPU's exact cubic polynomial fits signed i64.
    if w == 0
        || h == 0
        || w > 65_536
        || h > 65_536
        || w.checked_mul(h).and_then(|n| n.checked_mul(3)).is_none()
    {
        return Err(error(
            "invalid Gemma vision dimensions (edges must be 1..65536)",
        ));
    }
    let grid = |v: f32, mode: i32| {
        let v = v / 48.;
        (match mode {
            -1 => v.floor(),
            1 => v.ceil(),
            _ => v.round(),
        } as usize)
            .max(1)
            * 48
    };
    let (mut tw, mut th) = (grid(w as f32, 0), grid(h as f32, 0));
    if tw * th > BUDGET.max_pixels as usize {
        let b = ((w as f32 * h as f32) / BUDGET.max_pixels as f32).sqrt();
        (tw, th) = (grid(w as f32 / b, -1), grid(h as f32 / b, -1));
    } else if tw * th < BUDGET.min_pixels as usize {
        let b = (BUDGET.min_pixels as f32 / (w as f32 * h as f32)).sqrt();
        (tw, th) = (grid(w as f32 * b, 1), grid(h as f32 * b, 1));
    }
    if tw * th > BUDGET.max_pixels as usize || tw / 16 > 10240 || th / 16 > 10240 {
        return Err(error(
            "Gemma image aspect ratio cannot fit the image budget",
        ));
    }
    Ok((tw, th))
}

impl Vision {
    pub(super) fn load(device: &MetalDevice, path: &Path, width: usize) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        let u = |k: &str| meta.get(k).and_then(Value::as_u64);
        if map.gguf().architecture() != Some("clip")
            || meta
                .get("clip.vision.projector_type")
                .or_else(|| meta.get("clip.projector_type"))
                .and_then(Value::as_str)
                != Some("gemma4v")
            || u("clip.vision.embedding_length") != Some(E as u64)
            || u("clip.vision.feed_forward_length") != Some(F as u64)
            || u("clip.vision.block_count") != Some(27)
            || u("clip.vision.attention.head_count") != Some(16)
            || u("clip.vision.patch_size") != Some(16)
            || u("clip.vision.projection_dim") != Some(width as u64)
            || !matches!(meta.get("clip.has_vision_encoder"), Some(Value::Bool(true)))
            // Gemma's [-1,1] pixel conversion precedes these identity CLIP
            // channels. Refuse companions whose normalization we do not run.
            || !channels(meta.get("clip.vision.image_mean"), 0.)
            || !channels(meta.get("clip.vision.image_std"), 1.)
            || !matches!(width, 5376 | 2816)
            || map.gguf().tensors.iter().any(|t| {
                t.name.ends_with(".input_max")
                    || t.name.ends_with(".input_min")
                    || t.name.ends_with(".output_max")
                    || t.name.ends_with(".output_min")
            })
        {
            return Err(error(
                "Metal Gemma vision requires the exact 31B/26B-A4B gemma4v BF16 companion, without clipped linears",
            ));
        }
        let eps = match meta.get("clip.vision.attention.layer_norm_epsilon") {
            None => 1e-6,
            Some(v) => v
                .as_f32()
                .ok_or_else(|| error("invalid Gemma vision epsilon type"))?,
        };
        if !eps.is_finite() || eps <= 0. {
            return Err(error("invalid Gemma vision epsilon"));
        }
        let flag = |key: &str| -> Result<bool> {
            match meta.get(key) {
                None => Ok(false),
                Some(Value::Bool(v)) => Ok(*v),
                _ => Err(error(format!("invalid {key}"))),
            }
        };
        if flag("clip.use_silu")? {
            return Err(error("SiLU Gemma vision companion is not supported"));
        }
        // The shipped Unsloth companion omits clip.use_gelu. CLIP GGUF's
        // legacy default, also selected by the released same-weights oracle,
        // is QuickGELU. Google's HF config says tanh-GELU: this is an explicit
        // artifact-compatibility distinction, not a precision optimization.
        let quick_gelu = !flag("clip.use_gelu")?;
        if quick_gelu {
            tracing::warn!(
                "Gemma mmproj lacks clip.use_gelu=true: using GGUF QuickGELU compatibility; Google HF config specifies tanh-GELU"
            );
        }
        let load = |name: &str, dims: &[usize], matrix: bool| -> Result<Weight> {
            let source = Weight::load(device, &map, name, dims)?;
            if !matches!(source.ty, 0 | 1 | 30) || (matrix && dims.len() == 2 && source.ty != 30) {
                return Err(error(format!(
                    "{name}: expected canonical BF16 vision matrices"
                )));
            }
            let ty = if matrix {
                if source.ty == 30 { 30 } else { 1 }
            } else {
                0
            };
            let count: usize = dims.iter().product();
            let buffer = device.alloc(count * if matrix { 2 } else { 4 })?;
            let bad = device.upload(&0u32.to_le_bytes())?;
            let cmd = device.begin()?;
            cmd.dispatch(
                "vis_cast",
                &[&source.buffer, &buffer, &bad],
                &[count as u32, source.ty, ty],
                [count.div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
            // SAFETY: conversion/validation has completed.
            if unsafe { bad.read_u32(1)[0] } != 0 {
                return Err(error(format!("{name}: invalid vision weights")));
            }
            Ok(Weight {
                buffer,
                ty,
                k: dims[0],
                n: *dims.get(1).unwrap_or(&1),
            })
        };
        // The convolution companion is F32, unlike the BF16 transformer.
        // Keep its exact weights; the patch input alone follows IM2COL's F16 cut.
        let mut patch = load("v.patch_embd.weight", &[16, 16, 3, E], false)?;
        patch.k = 768;
        patch.n = E;
        let mut blocks = Vec::new();
        for i in 0..27 {
            let w = |s: &str, dims: &[usize], matrix| {
                load(&format!("v.blk.{i}.{s}.weight"), dims, matrix)
            };
            blocks.push(Block {
                norm: w("ln1", &[E], false)?,
                q: w("attn_q", &[E, E], true)?,
                k: w("attn_k", &[E, E], true)?,
                v: w("attn_v", &[E, E], true)?,
                q_norm: w("attn_q_norm", &[72], false)?,
                k_norm: w("attn_k_norm", &[72], false)?,
                out: w("attn_out", &[E, E], true)?,
                post: w("attn_post_norm", &[E], false)?,
                ffn_norm: w("ln2", &[E], false)?,
                gate: w("ffn_gate", &[E, F], true)?,
                up: w("ffn_up", &[E, F], true)?,
                down: w("ffn_down", &[F, E], true)?,
                ffn_post: w("ffn_post_norm", &[E], false)?,
            });
        }
        Ok(Self {
            mlx: false,
            patch,
            blocks,
            eps,
            quick_gelu,
            pos: load("v.position_embd.weight", &[E, 10240, 2], false)?,
            bias: load("v.std_bias", &[E], false)?,
            scale: load("v.std_scale", &[E], false)?,
            projection: load("mm.input_projection.weight", &[E, width], true)?,
        })
    }
    fn mm(cmd: &Commands<'_>, w: &Weight, x: &Buffer, out: &Buffer, rows: usize) {
        if w.ty == 0x101 {
            cmd.dispatch(
                "gmlx_vmm64",
                &[&w.buffer, x, out, &w.buffer],
                &[w.k as u32, w.n as u32, rows as u32, 0],
                [w.n.div_ceil(64), rows.div_ceil(64), 1],
                128,
            );
            return;
        }
        cmd.dispatch(
            if w.ty == 30 {
                "vis_bmm_fast64"
            } else {
                "gv_patch_project"
            },
            &[&w.buffer, x, out, &w.buffer],
            &[w.k as u32, w.n as u32, rows as u32, 0],
            [w.n.div_ceil(64), rows.div_ceil(64), 1],
            128,
        );
    }
    fn norm(
        &self,
        cmd: &Commands<'_>,
        x: &Buffer,
        w: &Weight,
        out: &Buffer,
        rows: usize,
        weighted: bool,
    ) {
        if self.mlx {
            cmd.dispatch(
                "gmlx_norm",
                &[x, &w.buffer, out],
                &[E as u32, if weighted { 0 } else { 2 }, self.eps.to_bits()],
                [rows, 1, 1],
                E.div_ceil(128) * 32,
            );
            return;
        }
        cmd.dispatch(
            "gv_rms",
            &[x, &w.buffer, out],
            &[E as u32, self.eps.to_bits(), u32::from(weighted)],
            [rows, 1, 1],
            256,
        );
    }
    pub(super) fn start(
        &self,
        device: &MetalDevice,
        images: &[(&[u8], usize, usize)],
    ) -> Result<Job> {
        if images.is_empty() {
            return Err(error("empty Gemma vision batch"));
        }
        let grids = images
            .iter()
            .map(|(rgb, w, h)| {
                let (tw, th) = if self.mlx {
                    resize_mlx(*w, *h)?
                } else {
                    resize(*w, *h)?
                };
                if rgb.len() != w * h * 3 {
                    return Err(error("Gemma RGB byte count mismatch"));
                }
                Ok((tw / 16, th / 16))
            })
            .collect::<Result<Vec<_>>>()?;
        let rows: usize = grids.iter().map(|(w, h)| w * h).sum();
        if rows > MAX_PATCHES {
            return Err(MetalError::Memory(
                "Gemma encoder wave exceeds 10080 patches".into(),
            ));
        }
        let mut xy = Vec::new();
        let mut tiles = Vec::new();
        let mut first = 0;
        for &(w, h) in &grids {
            for r in 0..w * h {
                xy.extend([(r % w) as u32, (r / w) as u32]);
            }
            for r in (0..w * h).step_by(32) {
                tiles.extend([
                    (first + r) as u32,
                    (w * h - r).min(32) as u32,
                    first as u32,
                    (w * h) as u32,
                ]);
            }
            first += w * h;
        }
        let a = |n| device.alloc(rows * n * 4);
        let upload =
            |v: &[u32]| device.upload(&v.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>());
        let mut job = Job {
            workspace: device.alloc(if self.mlx {
                crate::affine::workspace_bytes(E, self.projection.n, rows / 9)
            } else {
                4
            })?,
            grids,
            rows,
            x: a(E)?,
            stage: a(F)?,
            q: a(E)?,
            k: a(E)?,
            v: a(E)?,
            qh: device.alloc(rows * 1280 * 2)?,
            kh: device.alloc(rows * 1280 * 2)?,
            vh: device.alloc(rows * 1280 * 2)?,
            xy: upload(&xy)?,
            tiles: upload(&tiles)?,
            tile_count: tiles.len() / 4,
            attn: a(E)?,
            delta: a(E)?,
            gate: a(F)?,
            up: a(F)?,
            layer: 0,
            cost: Duration::ZERO,
            gpu_seconds: 0.,
        };
        // Keep every temporary alive through command completion. Coefficients
        // and horizontal u8 rows are bounded by source edges, not tower depth.
        let mut temporaries = Vec::new();
        first = 0;
        let cmd = device.begin()?;
        for ((rgb, w, h), &(pw, ph)) in images.iter().zip(&job.grids) {
            let (tw, th) = (pw * 16, ph * 16);
            let scale = (tw as f32 / *w as f32).min(th as f32 / *h as f32);
            let nw = if self.mlx {
                tw
            } else {
                ((*w as f32 * scale).ceil() as usize).clamp(1, tw)
            };
            let nh = if self.mlx {
                th
            } else {
                ((*h as f32 * scale).ceil() as usize).clamp(1, th)
            };
            let sx = 4 * w.div_ceil(nw) + 4;
            let sy = 4 * h.div_ceil(nh) + 4;
            let src = device.upload(rgb)?;
            let cx = device.alloc(nw * sx * 4)?;
            let cy = device.alloc(nh * sy * 4)?;
            let horizontal = device.alloc(nw * h * 3)?;
            for (source, target, stride, coeff) in [(*w, nw, sx, &cx), (*h, nh, sy, &cy)] {
                cmd.dispatch(
                    "gv_coeff",
                    &[coeff],
                    &[source as u32, target as u32, stride as u32],
                    [target.div_ceil(64), 1, 1],
                    64,
                );
            }
            cmd.dispatch(
                "gv_resize_h",
                &[&src, &cx, &horizontal],
                &[*w as u32, nw as u32, *h as u32, sx as u32],
                [(nw * h * 3).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                if self.mlx {
                    "gmlx_gv_patches"
                } else {
                    "gv_patches"
                },
                &[&horizontal, &cy, &job.stage],
                &[
                    nw as u32,
                    nh as u32,
                    tw as u32,
                    th as u32,
                    sy as u32,
                    first as u32,
                ],
                [(pw * ph * 768).div_ceil(256), 1, 1],
                256,
            );
            temporaries.extend([src, cx, cy, horizontal]);
            first += pw * ph;
        }
        Self::mm(&cmd, &self.patch, &job.stage, &job.x, rows);
        first = 0;
        for &(pw, ph) in &job.grids {
            cmd.dispatch(
                if self.mlx {
                    "gmlx_gv_position"
                } else {
                    "gv_position"
                },
                &[&job.x, &self.pos.buffer],
                &[pw as u32, ph as u32, first as u32, 10240],
                [(pw * ph * E).div_ceil(256), 1, 1],
                256,
            );
            first += pw * ph;
        }
        job.gpu_seconds += cmd.finish()?;
        Ok(job)
    }
    pub(super) fn step(
        &self,
        device: &MetalDevice,
        job: &mut Job,
        budget: Duration,
    ) -> Result<Option<Vec<Output>>> {
        let started = Instant::now();
        let count = if job.cost.is_zero() {
            1
        } else {
            (budget.as_secs_f64() / (job.cost.as_secs_f64() * 1.1)).floor() as usize
        }
        .clamp(1, 27);
        let end = (job.layer + count).min(27);
        let rows = job.rows;
        let cmd = device.begin()?;
        for b in &self.blocks[job.layer..end] {
            self.norm(&cmd, &job.x, &b.norm, &job.stage, rows, true);
            for (w, out) in [(&b.q, &job.q), (&b.k, &job.k), (&b.v, &job.v)] {
                Self::mm(&cmd, w, &job.stage, out, rows);
            }
            cmd.dispatch(
                if self.mlx { "gmlx_gv_qkv" } else { "gv_qkv" },
                &[
                    &job.q,
                    &job.k,
                    &job.v,
                    &b.q_norm.buffer,
                    &b.k_norm.buffer,
                    &job.xy,
                    &job.qh,
                    &job.kh,
                    &job.vh,
                ],
                &[self.eps.to_bits()],
                [16, rows, 1],
                32,
            );
            cmd.dispatch(
                if self.mlx {
                    "gmlx_gv_attention"
                } else {
                    "gv_attention"
                },
                &[&job.qh, &job.kh, &job.vh, &job.attn, &job.tiles],
                &[if self.mlx { 31 } else { 0 }],
                [16, job.tile_count, 1],
                64,
            );
            Self::mm(&cmd, &b.out, &job.attn, &job.delta, rows);
            cmd.dispatch(
                if self.mlx {
                    "gmlx_sandwich"
                } else {
                    "gemma_sandwich"
                },
                &[
                    &job.x,
                    &job.delta,
                    &b.post.buffer,
                    &b.ffn_norm.buffer,
                    &job.stage,
                ],
                &[
                    E as u32,
                    0,
                    if self.mlx { self.eps.to_bits() } else { 0 },
                    self.eps.to_bits(),
                    1f32.to_bits(),
                ],
                [rows, 1, 1],
                if self.mlx { E.div_ceil(128) * 32 } else { 256 },
            );
            Self::mm(&cmd, &b.gate, &job.stage, &job.gate, rows);
            Self::mm(&cmd, &b.up, &job.stage, &job.up, rows);
            cmd.dispatch(
                if self.mlx {
                    "gmlx_geglu"
                } else if self.quick_gelu {
                    "gv_geglu_quick"
                } else {
                    "gemma_geglu"
                },
                &[&job.gate, &job.up],
                &[(rows * F) as u32],
                [(rows * F).div_ceil(256), 1, 1],
                256,
            );
            Self::mm(&cmd, &b.down, &job.gate, &job.delta, rows);
            // Sandwich adds post-normalized FFN residual; its next RMS also
            // produces a valid scratch row (the next block rewrites it).
            cmd.dispatch(
                if self.mlx {
                    "gmlx_sandwich"
                } else {
                    "gemma_sandwich"
                },
                &[
                    &job.x,
                    &job.delta,
                    &b.ffn_post.buffer,
                    &b.norm.buffer,
                    &job.stage,
                ],
                &[
                    E as u32,
                    0,
                    if self.mlx { self.eps.to_bits() } else { 0 },
                    self.eps.to_bits(),
                    1f32.to_bits(),
                ],
                [rows, 1, 1],
                if self.mlx { E.div_ceil(128) * 32 } else { 256 },
            );
        }
        if end < 27 {
            job.gpu_seconds += cmd.finish()?;
            job.cost = started.elapsed() / (end - job.layer) as u32;
            job.layer = end;
            return Ok(None);
        }
        let mut first = 0;
        let mut pooled = 0;
        for &(pw, ph) in &job.grids {
            cmd.dispatch(
                if self.mlx { "gmlx_gv_pool" } else { "gv_pool" },
                &[&job.x, &self.bias.buffer, &self.scale.buffer, &job.attn],
                &[pw as u32, ph as u32, first as u32, pooled as u32],
                [(pw * ph / 9 * E).div_ceil(256), 1, 1],
                256,
            );
            first += pw * ph;
            pooled += pw * ph / 9;
        }
        self.norm(&cmd, &job.attn, &self.bias, &job.stage, pooled, false);
        let merged = device.alloc(pooled * self.projection.n * 4)?;
        if self.mlx {
            crate::affine::project(
                &cmd,
                &[(&self.projection, &merged)],
                &job.stage,
                pooled,
                &job.workspace,
            );
        } else {
            Self::mm(&cmd, &self.projection, &job.stage, &merged, pooled);
        }
        let bad = device.upload(&0u32.to_le_bytes())?;
        cmd.dispatch(
            "vis_finite",
            &[&merged, &bad],
            &[(pooled * self.projection.n) as u32],
            [(pooled * self.projection.n).div_ceil(256), 1, 1],
            256,
        );
        let mut outputs = Vec::new();
        first = 0;
        for &(pw, ph) in &job.grids {
            let tokens = pw * ph / 9;
            let count = tokens * self.projection.n;
            let embd = device.alloc(count * 4)?;
            copy_words(&cmd, &merged, &embd, first, 0, count);
            first += count;
            outputs.push(Output { embd, tokens });
        }
        job.gpu_seconds += cmd.finish()?;
        job.layer = end;
        // SAFETY: merger and finite validation completed before publication.
        if unsafe { bad.read_u32(1)[0] } != 0 {
            return Err(error("Gemma vision produced nonfinite embeddings"));
        }
        Ok(Some(outputs))
    }
}
