//! Native Muse Glimmer ViT-G/14. Window/global attention domains are ragged
//! descriptors, not masks. All pixel and tensor arithmetic runs on Metal.
use super::*;
use paddock_engine::generator::VisionBudget;
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::time::{Duration, Instant};

const E: usize = 1536;
const F: usize = 8960;
#[path = "muse_vision_mlx.rs"]
mod mlx;
#[cfg(test)]
#[path = "muse_vision_tests.rs"]
mod tests;
pub(super) const MAX_PATCHES: usize = 16384;
pub(super) const BUDGET: VisionBudget = VisionBudget {
    min_pixels: 784,
    max_pixels: 4096 * 784,
    max_edge: None,
    pixels_per_token: 784,
    min_tokens: 1,
    max_tokens: 4096,
};
struct Norm {
    w: Weight,
    b: Weight,
}
struct Block {
    ln1: Norm,
    ln2: Norm,
    q: Weight,
    k: Weight,
    v: Weight,
    qb: Weight,
    kb: Weight,
    vb: Weight,
    out: Weight,
    ob: Weight,
    up: Weight,
    ub: Weight,
    down: Weight,
    db: Weight,
}
pub(super) struct Vision {
    mlx: bool,
    patch: Weight,
    pos: Weight,
    pre: Norm,
    post: Norm,
    blocks: Vec<Block>,
    mm0: Weight,
    mm1: Weight,
    mm2: Weight,
    eps: f32,
}
pub(super) struct Job {
    workspace: Buffer,
    grids: Vec<(usize, usize)>,
    rows: usize,
    layer: usize,
    x: Buffer,
    stage: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qh: Buffer,
    kh: Buffer,
    vh: Buffer,
    attn: Buffer,
    up: Buffer,
    xy: Buffer,
    inverse: Buffer,
    local: Buffer,
    global: Buffer,
    local_count: usize,
    global_count: usize,
    pub(super) cost: Duration,
    pub(super) gpu_seconds: f64,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Muse vision: {}", s.into()))
}

/// The processor fits an aspect-preserving integer grid of merged 28px
/// cells. These are geometry calculations, not host pixel transformations.
pub(super) fn resize(w: usize, h: usize) -> Result<(usize, usize)> {
    if w == 0 || h == 0 || w > 65536 || h > 65536 {
        return Err(error("edges must be 1..65536"));
    }
    let (mut fh, mut fw) = (h as f64 / 28., w as f64 / 28.);
    if fh * fw > 4096. {
        fh = (4096. * h as f64 / w as f64).sqrt();
        fw = fh * w as f64 / h as f64;
    }
    let mut best: Option<(usize, usize, f64)> = None;
    for gh in [fh.floor() as usize, fh.ceil() as usize] {
        for gw in [fw.floor() as usize, fw.ceil() as usize] {
            if gh == 0 || gw == 0 || gh * gw > 4096 {
                continue;
            }
            let delta = (gh as f64 / gw as f64 - h as f64 / w as f64).abs();
            if best.is_none_or(|(bw, bh, d)| delta < d || (delta == d && gh * gw > bw * bh)) {
                best = Some((gw, gh, delta));
            }
        }
    }
    let (gw, gh) = best
        .map(|(w, h, _)| (w, h))
        .unwrap_or((fw.round().max(1.) as usize, fh.round().max(1.) as usize));
    if gw * gh > 4096 {
        return Err(error("aspect ratio cannot fit the image token ceiling"));
    }
    Ok((gw * 28, gh * 28))
}
fn upload(d: &MetalDevice, v: &[u32]) -> Result<Buffer> {
    d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
impl Vision {
    pub(super) fn load(device: &MetalDevice, path: &Path, width: usize) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        if map.gguf().architecture() != Some("clip")
            || width != 6656
            || meta
                .get("clip.vision.projector_type")
                .or_else(|| meta.get("clip.projector_type"))
                .and_then(Value::as_str)
                != Some("muse-glimmer")
            || !matches!(meta.get("clip.has_vision_encoder"), Some(Value::Bool(true)))
        {
            return Err(error(
                "expected canonical Muse Glimmer BF16 companion for width 6656",
            ));
        }
        for (key, value) in [
            ("embedding_length", E),
            ("feed_forward_length", F),
            ("block_count", 50),
            ("attention.head_count", 16),
            ("patch_size", 14),
            ("spatial_merge_size", 2),
            ("projection_dim", width),
        ] {
            if meta
                .get(&format!("clip.vision.{key}"))
                .and_then(Value::as_u64)
                != Some(value as u64)
            {
                return Err(error(format!("invalid {key}")));
            }
        }
        for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
            if !matches!(meta.get(key),Some(Value::Array(a)) if a.len()==3 && a.iter().all(|v|v.as_f32()==Some(0.5)))
            {
                return Err(error(format!("unsupported {key}")));
            }
        }
        let eps = meta
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| error("invalid epsilon"))?;
        let load = |name: &str, dims: &[usize], matrix: bool| -> Result<Weight> {
            let source = Weight::load(device, &map, name, dims)?;
            if !matches!(source.ty, 0 | 1 | 30) || (matrix && source.ty != 30) {
                return Err(error(format!("{name}: expected BF16 matrices/F32 vectors")));
            }
            let count = dims.iter().product::<usize>();
            let ty = if matrix { 30 } else { 0 };
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
            if unsafe { bad.read_u32(1)[0] } != 0 {
                return Err(error(format!("nonfinite {name}")));
            }
            Ok(Weight {
                buffer,
                ty,
                k: dims[0],
                n: *dims.get(1).unwrap_or(&1),
            })
        };
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                w: load(&format!("{name}.weight"), &[E], false)?,
                b: load(&format!("{name}.bias"), &[E], false)?,
            })
        };
        let mut patch = load("v.patch_embd.weight", &[14, 14, 3, E], false)?;
        patch.k = 588;
        patch.n = E;
        let mut blocks = Vec::new();
        for i in 0..50 {
            let w = |s: &str, dims: &[usize]| load(&format!("v.blk.{i}.{s}.weight"), dims, true);
            let b = |s: &str, n| load(&format!("v.blk.{i}.{s}.bias"), &[n], false);
            blocks.push(Block {
                ln1: norm(&format!("v.blk.{i}.ln1"))?,
                ln2: norm(&format!("v.blk.{i}.ln2"))?,
                q: w("attn_q", &[E, E])?,
                k: w("attn_k", &[E, E])?,
                v: w("attn_v", &[E, E])?,
                qb: b("attn_q", E)?,
                kb: b("attn_k", E)?,
                vb: b("attn_v", E)?,
                out: w("attn_out", &[E, E])?,
                ob: b("attn_out", E)?,
                up: w("ffn_up", &[E, F])?,
                ub: b("ffn_up", F)?,
                down: w("ffn_down", &[F, E])?,
                db: b("ffn_down", E)?,
            });
        }
        Ok(Self {
            mlx: false,
            patch,
            pos: load("v.position_embd.weight", &[E, 1024], false)?,
            pre: norm("v.pre_ln")?,
            post: norm("v.post_ln")?,
            blocks,
            mm0: load("mm.0.weight", &[6144, 4096], true)?,
            mm1: load("mm.1.weight", &[4096, 4096], true)?,
            mm2: load("mm.2.weight", &[4096, width], true)?,
            eps,
        })
    }
    fn mm(
        cmd: &Commands<'_>,
        w: &Weight,
        x: &Buffer,
        out: &Buffer,
        bias: Option<&Weight>,
        rows: usize,
        epilogue: u32,
    ) {
        if w.ty == 0x101 {
            super::mlx::dense(cmd, w, x, out, bias, rows, epilogue);
            return;
        }
        // Wide down-projections amortize their long contraction over more
        // image rows. The opposite (up) geometry regressed with this tile;
        // keep its election and small-image arithmetic unchanged.
        let bm = if w.ty == 30 && w.k == F && w.n == E && rows >= 1024 {
            128
        } else {
            64
        };
        cmd.dispatch(
            if w.ty == 30 {
                if bm == 128 { "mv_bmm128" } else { "mv_bmm64" }
            } else {
                "gv_patch_project"
            },
            &[&w.buffer, x, out, bias.map_or(&w.buffer, |b| &b.buffer)],
            &[
                w.k as u32,
                w.n as u32,
                rows as u32,
                if epilogue != 0 {
                    epilogue
                } else {
                    u32::from(bias.is_some())
                },
            ],
            [w.n.div_ceil(64), rows.div_ceil(bm), 1],
            128,
        );
    }
    fn norm(&self, cmd: &Commands<'_>, x: &Buffer, norm: &Norm, out: &Buffer, rows: usize) {
        cmd.dispatch(
            if self.mlx {
                "gmlx_layer_norm"
            } else {
                "vis_ln"
            },
            &[x, &norm.w.buffer, &norm.b.buffer, out],
            &[E as u32, self.eps.to_bits()],
            [rows, 1, 1],
            if self.mlx { E.div_ceil(128) * 32 } else { 256 },
        );
    }
    pub(super) fn start(
        &self,
        device: &MetalDevice,
        images: &[(&[u8], usize, usize)],
    ) -> Result<Job> {
        if images.is_empty() {
            return Err(error("empty batch"));
        }
        let grids = images
            .iter()
            .map(|(rgb, w, h)| {
                let (tw, th) = resize(*w, *h)?;
                if rgb.len() != w * h * 3 {
                    return Err(error("RGB byte count mismatch"));
                }
                Ok((tw / 14, th / 14))
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = grids.iter().map(|(w, h)| w * h).sum::<usize>();
        if rows > MAX_PATCHES {
            return Err(error("encoder wave exceeds 16384 patches"));
        }
        let mut perm = Vec::with_capacity(rows);
        let mut inverse = vec![0u32; rows];
        let mut xy = Vec::with_capacity(rows * 2);
        let mut local = Vec::new();
        let mut global = Vec::new();
        let mut first = 0;
        for &(w, h) in &grids {
            for wy in (0..h).step_by(32) {
                for wx in (0..w).step_by(32) {
                    let start = perm.len();
                    for y in wy..(wy + 32).min(h) {
                        for x in wx..(wx + 32).min(w) {
                            let index = first + y * w + x;
                            inverse[index] = perm.len() as u32;
                            perm.push(index as u32);
                            xy.extend([(x + 1) as u32, (y + 1) as u32]);
                        }
                    }
                    let count = perm.len() - start;
                    for at in (0..count).step_by(32) {
                        local.extend([
                            (start + at) as u32,
                            (count - at).min(32) as u32,
                            start as u32,
                            count as u32,
                        ]);
                    }
                }
            }
            for at in (0..w * h).step_by(32) {
                global.extend([
                    (first + at) as u32,
                    (w * h - at).min(32) as u32,
                    first as u32,
                    (w * h) as u32,
                ]);
            }
            first += w * h;
        }
        let a = |n| device.alloc(rows * n * 4);
        let mut job = Job {
            workspace: device.alloc(if self.mlx {
                [(6144, 4096), (4096, 4096), (4096, 6656)]
                    .into_iter()
                    .map(|(k, n)| crate::affine::workspace_bytes(k, n, rows / 4))
                    .max()
                    .expect("three Muse projectors")
            } else {
                4
            })?,
            grids,
            rows,
            layer: 0,
            x: a(E)?,
            stage: a(6144)?,
            q: a(E)?,
            k: a(E)?,
            v: a(E)?,
            qh: device.alloc(rows * E * 2)?,
            kh: device.alloc(rows * E * 2)?,
            vh: device.alloc(rows * E * 2)?,
            attn: a(E)?,
            up: a(F)?,
            xy: upload(device, &xy)?,
            inverse: upload(device, &inverse)?,
            local: upload(device, &local)?,
            global: upload(device, &global)?,
            local_count: local.len() / 4,
            global_count: global.len() / 4,
            cost: Duration::ZERO,
            gpu_seconds: 0.,
        };
        let perm = upload(device, &perm)?;
        let mut temporaries = Vec::new();
        first = 0;
        let cmd = device.begin()?;
        for ((rgb, w, h), &(pw, ph)) in images.iter().zip(&job.grids) {
            let (tw, th) = (pw * 14, ph * 14);
            let sx = 6 * w.div_ceil(tw) + 4;
            let sy = 6 * h.div_ceil(th) + 4;
            let src = device.upload(rgb)?;
            let cx = device.alloc(tw * sx * 4)?;
            let cy = device.alloc(th * sy * 4)?;
            let horizontal = device.alloc(tw * h * 3)?;
            for (source, target, stride, coeff) in [(*w, tw, sx, &cx), (*h, th, sy, &cy)] {
                cmd.dispatch(
                    "mv_coeff",
                    &[coeff],
                    &[source as u32, target as u32, stride as u32],
                    [target.div_ceil(64), 1, 1],
                    64,
                );
            }
            cmd.dispatch(
                "gv_resize_h",
                &[&src, &cx, &horizontal],
                &[*w as u32, tw as u32, *h as u32, sx as u32],
                [(tw * h * 3).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                if self.mlx {
                    "gmlx_mv_patches"
                } else {
                    "mv_patches"
                },
                &[&horizontal, &cy, &job.stage],
                &[tw as u32, th as u32, sy as u32, first as u32, *h as u32],
                [
                    (pw * ph * if self.mlx { 1176 } else { 588 }).div_ceil(256),
                    1,
                    1,
                ],
                256,
            );
            temporaries.extend([src, cx, cy, horizontal]);
            first += pw * ph;
        }
        Self::mm(&cmd, &self.patch, &job.stage, &job.x, None, rows, 0);
        first = 0;
        for &(w, h) in &job.grids {
            cmd.dispatch(
                if self.mlx {
                    "gmlx_mv_position"
                } else {
                    "mv_position"
                },
                &[&job.x, &self.pos.buffer],
                &[w as u32, h as u32, first as u32],
                [(w * h * E).div_ceil(256), 1, 1],
                256,
            );
            first += w * h;
        }
        self.norm(&cmd, &job.x, &self.pre, &job.stage, rows);
        cmd.dispatch(
            "mv_permute",
            &[&job.stage, &job.x, &perm],
            &[rows as u32],
            [(rows * E).div_ceil(256), 1, 1],
            256,
        );
        job.gpu_seconds = cmd.finish()?;
        Ok(job)
    }
    pub(super) fn step(
        &self,
        device: &MetalDevice,
        job: &mut Job,
        budget: Duration,
    ) -> Result<Option<Vec<vision::Output>>> {
        if job.layer >= 50 {
            return Err(error("completed job cannot be replayed"));
        }
        let started = Instant::now();
        let count = if job.cost.is_zero() {
            1
        } else {
            (budget.as_secs_f64() / (job.cost.as_secs_f64() * 1.1)).floor() as usize
        }
        .clamp(1, 50);
        let end = (job.layer + count).min(50);
        let rows = job.rows;
        let cmd = device.begin()?;
        for (i, b) in self.blocks.iter().enumerate().take(end).skip(job.layer) {
            self.norm(&cmd, &job.x, &b.ln1, &job.stage, rows);
            for (w, bias, out) in [
                (&b.q, &b.qb, &job.q),
                (&b.k, &b.kb, &job.k),
                (&b.v, &b.vb, &job.v),
            ] {
                Self::mm(&cmd, w, &job.stage, out, Some(bias), rows, 0);
            }
            cmd.dispatch(
                if self.mlx { "gmlx_mv_qkv" } else { "mv_qkv" },
                &[&job.q, &job.k, &job.v, &job.xy, &job.qh, &job.kh, &job.vh],
                &[rows as u32],
                [(rows * E).div_ceil(256), 1, 1],
                256,
            );
            let (tiles, count) = if i % 4 == 3 || i == 49 {
                (&job.global, job.global_count)
            } else {
                (&job.local, job.local_count)
            };
            cmd.dispatch(
                if self.mlx {
                    "gmlx_mv_attention"
                } else {
                    "mv_attention"
                },
                &[&job.qh, &job.kh, &job.vh, &job.attn, tiles],
                &[0],
                [16, count, 1],
                64,
            );
            Self::mm(&cmd, &b.out, &job.attn, &job.x, Some(&b.ob), rows, 2);
            self.norm(&cmd, &job.x, &b.ln2, &job.stage, rows);
            Self::mm(&cmd, &b.up, &job.stage, &job.up, Some(&b.ub), rows, 3);
            Self::mm(&cmd, &b.down, &job.up, &job.x, Some(&b.db), rows, 2);
        }
        let mut outputs = None;
        if end == 50 {
            self.norm(&cmd, &job.x, &self.post, &job.attn, rows);
            let (mut first, mut soft) = (0, 0);
            for &(w, h) in &job.grids {
                cmd.dispatch(
                    "mv_shuffle",
                    &[&job.attn, &job.inverse, &job.stage],
                    &[w as u32, h as u32, first as u32, soft as u32],
                    [(w * h / 4 * 6144).div_ceil(256), 1, 1],
                    256,
                );
                first += w * h;
                soft += w * h / 4;
            }
            if self.mlx {
                for (w, x, out, activate) in [
                    (&self.mm0, &job.stage, &job.up, true),
                    (&self.mm1, &job.up, &job.stage, true),
                    (&self.mm2, &job.stage, &job.up, false),
                ] {
                    crate::affine::project(&cmd, &[(w, out)], x, soft, &job.workspace);
                    if activate {
                        cmd.dispatch(
                            "gmlx_erfgelu",
                            &[out],
                            &[(soft * w.n) as u32],
                            [(soft * w.n).div_ceil(256), 1, 1],
                            256,
                        );
                    }
                }
            } else {
                Self::mm(&cmd, &self.mm0, &job.stage, &job.up, None, soft, 4);
                Self::mm(&cmd, &self.mm1, &job.up, &job.stage, None, soft, 4);
                Self::mm(&cmd, &self.mm2, &job.stage, &job.up, None, soft, 0);
            }
            let mut result = Vec::new();
            soft = 0;
            for &(w, h) in &job.grids {
                let tokens = w * h / 4;
                let embd = device.alloc(tokens * 6656 * 4)?;
                copy_words(&cmd, &job.up, &embd, soft * 6656, 0, tokens * 6656);
                result.push(vision::Output { embd, tokens });
                soft += tokens;
            }
            outputs = Some(result);
        }
        job.gpu_seconds += cmd.finish()?;
        job.cost = started.elapsed() / ((end - job.layer) as u32);
        job.layer = end;
        Ok(outputs)
    }
}
