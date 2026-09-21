//! Granite's SigLIP2 tower and eight windowed Q-Formers. F16 checkpoint
//! matrices, F32 residuals/softmax, shared conversion for Q/K/V, and MPP
//! contractions. A job yields at block/projector boundaries; no dense score
//! matrix, CPU tensor math, or persistent widened weight twin.
use super::*;
use paddock_engine::{
    generator::VisionBudget,
    granite_layout::{AnyResPlan, PackRow, TileGeom},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const E: usize = 1152;
const F: usize = 4608;
pub(super) const MAX_TILES: usize = 22;
struct Norm {
    w: Weight,
    b: Weight,
}
struct Linear {
    w: Weight,
    b: Weight,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}
struct Block {
    ln1: Norm,
    ln2: Norm,
    attn: Attention,
    up: Linear,
    down: Linear,
}
struct Projector {
    norm: Norm,
    query: Weight,
    pos: Weight,
    input: Norm,
    sa: Attention,
    sn: Norm,
    ca: Attention,
    cn: Norm,
    up: Linear,
    down: Linear,
    ffn: Norm,
    linear: Linear,
    tap: usize,
    spatial: u32,
}
pub(super) struct Vision {
    patch: Linear,
    pos: Weight,
    newline: Weight,
    blocks: Vec<Block>,
    projectors: Vec<Projector>,
    pub(super) pins: Vec<(usize, usize)>,
    eps: f32,
    width: usize,
    mean: [f32; 3],
    std: [f32; 3],
}
/// Immutable once published. Only the engine thread submits commands; readers
/// hold GPU handles, never mutable host mappings.
pub(super) struct Features {
    pub(super) streams: Vec<Buffer>,
    pub(super) tokens: usize,
}
// SAFETY: features are created after synchronous GPU completion, and every
// consumer binds streams read-only. No safe mutable mapping escapes this module.
unsafe impl Sync for Features {}
impl Features {
    pub(super) fn bytes(&self) -> usize {
        self.streams.iter().map(Buffer::len).sum()
    }
}
pub(super) struct Job {
    plans: Vec<AnyResPlan>,
    tiles: usize,
    phase: usize,
    x: Buffer,
    norm: Buffer,
    stage: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qh: Buffer,
    kh: Buffer,
    vh: Buffer,
    attn: Buffer,
    up: Buffer,
    enc: Buffer,
    state: Buffer,
    taps: Vec<Option<Buffer>>,
    tower_tiles: Buffer,
    self_tiles: Buffer,
    cross_tiles: Buffer,
    indices: Vec<Buffer>,
    outputs: Vec<Vec<Buffer>>,
    pub(super) cost: Duration,
    pub(super) gpu_seconds: f64,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Granite vision: {}", s.into()))
}
fn upload(d: &MetalDevice, v: &[u32]) -> Result<Buffer> {
    d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
impl Vision {
    pub(super) fn load(d: &MetalDevice, path: &Path, width: usize) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        if map.gguf().architecture() != Some("clip")
            || width != 2560
            || meta.get("clip.projector_type").and_then(Value::as_str) != Some("granite4_vision")
        {
            return Err(error(
                "expected Granite 4.1 vision companion for width 2560",
            ));
        }
        for (k, v) in [
            ("block_count", 27),
            ("embedding_length", E),
            ("attention.head_count", 16),
            ("feed_forward_length", 4304),
            ("image_size", 384),
            ("patch_size", 16),
            ("projection_dim", width),
            ("projector.window_side", 8),
            ("projector.query_side", 4),
        ] {
            if meta
                .get(&format!("clip.vision.{k}"))
                .and_then(Value::as_u64)
                != Some(v as u64)
            {
                return Err(error(format!("unsupported {k}")));
            }
        }
        let array = |key: &str| -> Result<Vec<i64>> {
            match meta.get(&format!("clip.vision.{key}")) {
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| v.as_i64().ok_or_else(|| error(key)))
                    .collect(),
                _ => Err(error(key)),
            }
        };
        let taps = array("feature_layer")?;
        let spatial = array("projector.spatial_offsets")?;
        if taps != [26, 20, 14, 8, 26, 26, 26, 26] || spatial != [-1, -1, -1, -1, 0, 1, 2, 3] {
            return Err(error("unqualified tower taps or spatial offsets"));
        }
        let pins = array("image_grid_pinpoints")?;
        if pins.is_empty()
            || pins.len() % 2 != 0
            || pins.iter().any(|&v| v <= 0 || v > 3840 || v % 384 != 0)
        {
            return Err(error("invalid AnyRes pinpoints"));
        }
        let pins = pins
            .chunks_exact(2)
            .map(|a| (a[1] as usize, a[0] as usize))
            .collect::<Vec<_>>();
        if pins.iter().any(|&(w, h)| w / 384 * (h / 384) > 10) {
            return Err(error("AnyRes grid exceeds 10 tiles"));
        }
        let channels = |key: &str| -> Result<[f32; 3]> {
            let Some(Value::Array(a)) = meta.get(&format!("clip.vision.{key}")) else {
                return Err(error(key));
            };
            let a = a
                .iter()
                .map(|v| {
                    v.as_f32()
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| error(key))
                })
                .collect::<Result<Vec<_>>>()?;
            a.try_into().map_err(|_| error(key))
        };
        let mean = channels("image_mean")?;
        let std = channels("image_std")?;
        if std.iter().any(|&v| v <= 0.) {
            return Err(error("invalid image standard deviation"));
        }
        let eps = meta
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| error("invalid norm epsilon"))?;
        let load = |name: &str, dims: &[usize], matrix: bool| -> Result<Weight> {
            let w = Weight::load(d, &map, name, dims)?;
            if (matrix && w.ty != 1) || (!matrix && w.ty != 0) {
                return Err(error(format!(
                    "{name}: expected original F16 matrices/F32 vectors"
                )));
            }
            Ok(w)
        };
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                w: load(&format!("{name}.weight"), &[E], false)?,
                b: load(&format!("{name}.bias"), &[E], false)?,
            })
        };
        let linear = |name: &str, k: usize, n: usize| -> Result<Linear> {
            Ok(Linear {
                w: load(&format!("{name}.weight"), &[k, n], true)?,
                b: load(&format!("{name}.bias"), &[n], false)?,
            })
        };
        let attention = |prefix: &str, q: &str, k: &str, v: &str, o: &str| -> Result<Attention> {
            Ok(Attention {
                q: linear(&format!("{prefix}.{q}"), E, E)?,
                k: linear(&format!("{prefix}.{k}"), E, E)?,
                v: linear(&format!("{prefix}.{v}"), E, E)?,
                out: linear(&format!("{prefix}.{o}"), E, E)?,
            })
        };
        let mut blocks = Vec::new();
        for i in 0..27 {
            let p = format!("v.blk.{i}");
            blocks.push(Block {
                ln1: norm(&format!("{p}.ln1"))?,
                ln2: norm(&format!("{p}.ln2"))?,
                attn: attention(&p, "attn_q", "attn_k", "attn_v", "attn_out")?,
                up: linear(&format!("{p}.ffn_up"), E, 4304)?,
                down: linear(&format!("{p}.ffn_down"), 4304, E)?,
            });
        }
        let mut projectors = Vec::new();
        for i in 0..8 {
            let p = format!("v.proj_blk.{i}");
            let ff = map
                .tensor_info(&format!("{p}.ffn_up.weight"))
                .and_then(|t| t.dims.get(1))
                .copied()
                .unwrap_or(0) as usize;
            if ff == 0 || ff > F {
                return Err(error("unsupported Q-Former FFN width"));
            }
            projectors.push(Projector {
                norm: norm(&format!("{p}.norm"))?,
                query: load(&format!("{p}.query"), &[E, 16, 1], false)?,
                pos: load(&format!("{p}.img_pos"), &[E, 64, 1], false)?,
                input: norm(&format!("{p}.post_norm"))?,
                sa: attention(
                    &p,
                    "self_attn_q",
                    "self_attn_k",
                    "self_attn_v",
                    "self_attn_out",
                )?,
                sn: norm(&format!("{p}.self_attn_norm"))?,
                ca: attention(
                    &p,
                    "cross_attn_q",
                    "cross_attn_k",
                    "cross_attn_v",
                    "cross_attn_out",
                )?,
                cn: norm(&format!("{p}.cross_attn_norm"))?,
                up: linear(&format!("{p}.ffn_up"), E, ff)?,
                down: linear(&format!("{p}.ffn_down"), ff, E)?,
                ffn: norm(&format!("{p}.ffn_norm"))?,
                linear: linear(&format!("{p}.linear"), E, width)?,
                tap: taps[i] as usize,
                spatial: spatial[i] as u32,
            });
        }
        let mut pw = load("v.patch_embd.weight", &[16, 16, 3, E], true)?;
        pw.k = 768;
        pw.n = E;
        Ok(Self {
            patch: Linear {
                w: pw,
                b: load("v.patch_embd.bias", &[E], false)?,
            },
            pos: load("v.position_embd.weight", &[E, 576], false)?,
            newline: load("v.image_newline", &[width], false)?,
            blocks,
            projectors,
            pins,
            eps,
            width,
            mean,
            std,
        })
    }
    pub(super) fn plan(&self, w: usize, h: usize) -> Result<AnyResPlan> {
        if w == 0 || h == 0 || w > 16384 || h > 16384 {
            return Err(error("image edges must be 1..16384"));
        }
        AnyResPlan::new(
            w,
            h,
            TileGeom {
                image_size: 384,
                tokens_side: 12,
            },
            &self.pins,
        )
        .ok_or_else(|| error("invalid image plan"))
    }
    pub(super) fn budget(&self) -> VisionBudget {
        VisionBudget {
            min_pixels: 384 * 384,
            max_pixels: 10 * 384 * 384,
            max_edge: Some(3840),
            pixels_per_token: 1024,
            min_tokens: 144,
            max_tokens: self
                .pins
                .iter()
                .filter_map(|&(w, h)| self.plan(w, h).ok())
                .map(|p| p.n_tokens() as u32)
                .max()
                .unwrap_or(1596),
        }
    }
    fn ln(cmd: &Commands<'_>, norm: &Norm, input: &Buffer, out: &Buffer, rows: usize, eps: f32) {
        cmd.dispatch(
            "vis_ln",
            &[input, &norm.w.buffer, &norm.b.buffer, out],
            &[E as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    fn half(cmd: &Commands<'_>, input: &Buffer, out: &Buffer, count: usize) {
        cmd.dispatch(
            "grv_half",
            &[input, out],
            &[count as u32],
            [count.div_ceil(256), 1, 1],
            256,
        );
    }
    fn mm(cmd: &Commands<'_>, w: &Linear, input: &Buffer, out: &Buffer, rows: usize, mode: u32) {
        cmd.dispatch(
            "vis_mm64",
            &[&w.w.buffer, input, out, &w.b.buffer],
            &[w.w.k as u32, w.w.n as u32, rows as u32, mode],
            [w.w.n.div_ceil(64), rows.div_ceil(64), 1],
            128,
        );
    }
    fn qkv(
        cmd: &Commands<'_>,
        a: &Attention,
        j: &Job,
        input: &Buffer,
        encoder: &Buffer,
        qr: usize,
        kr: usize,
        tower: bool,
    ) {
        Self::half(cmd, input, &j.stage, qr * E);
        Self::mm(cmd, &a.q, &j.stage, &j.q, qr, 1);
        if !std::ptr::eq(input, encoder) {
            Self::half(cmd, encoder, &j.stage, kr * E);
        }
        Self::mm(cmd, &a.k, &j.stage, &j.k, kr, 1);
        Self::mm(cmd, &a.v, &j.stage, &j.v, kr, 1);
        for (src, dst, rows) in [(&j.q, &j.qh, qr), (&j.k, &j.kh, kr), (&j.v, &j.vh, kr)] {
            if tower {
                cmd.dispatch(
                    "grv_heads",
                    &[src, dst],
                    &[rows as u32],
                    [(rows * 16 * 80).div_ceil(256), 1, 1],
                    256,
                );
            } else {
                Self::half(cmd, src, dst, rows * E);
            }
        }
    }
    fn attend(cmd: &Commands<'_>, j: &Job, tiles: &Buffer, tower: bool) {
        cmd.dispatch(
            if tower {
                "vis_attention"
            } else {
                "grv_qattention"
            },
            &[&j.qh, &j.kh, &j.vh, &j.attn, tiles],
            &[0],
            [if tower { 16 } else { 18 }, tiles.len() / 16, 1],
            64,
        );
    }
    pub(super) fn start(&self, d: &MetalDevice, images: &[(&[u8], usize, usize)]) -> Result<Job> {
        let plans = images
            .iter()
            .map(|&(_, w, h)| self.plan(w, h))
            .collect::<Result<Vec<_>>>()?;
        let tiles = plans.iter().map(AnyResPlan::n_tiles).sum::<usize>();
        if tiles == 0 || tiles > MAX_TILES {
            return Err(error("invalid batched tile count"));
        }
        let rows = tiles * 576;
        let a = || d.alloc(rows * E * 4);
        let h = || d.alloc(rows * 16 * 80 * 2);
        let descriptors = |ql: usize, kl: usize, batch: usize| -> Result<Buffer> {
            upload(
                d,
                &(0..batch)
                    .flat_map(|b| {
                        (0..ql).step_by(32).flat_map(move |q| {
                            [
                                (b * ql + q) as u32,
                                (ql - q).min(32) as u32,
                                (b * kl) as u32,
                                kl as u32,
                            ]
                        })
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let mut indices = Vec::new();
        let mut tilebase = 0;
        for p in &plans {
            indices.push(upload(
                d,
                &p.rows()
                    .iter()
                    .map(|r| match r {
                        PackRow::Feature { tile, idx } => ((tilebase + tile) * 144 + idx) as u32,
                        PackRow::Newline => u32::MAX,
                    })
                    .collect::<Vec<_>>(),
            )?);
            tilebase += p.n_tiles();
        }
        let mut j = Job {
            outputs: (0..plans.len()).map(|_| Vec::new()).collect(),
            plans,
            tiles,
            phase: 0,
            x: a()?,
            norm: a()?,
            stage: d.alloc(rows * F * 2)?,
            q: a()?,
            k: a()?,
            v: a()?,
            qh: h()?,
            kh: h()?,
            vh: h()?,
            attn: a()?,
            up: d.alloc(rows * F * 4)?,
            enc: a()?,
            state: a()?,
            taps: (0..27).map(|_| None).collect(),
            tower_tiles: descriptors(576, 576, tiles)?,
            self_tiles: descriptors(16, 16, tiles * 9)?,
            cross_tiles: descriptors(16, 64, tiles * 9)?,
            indices,
            cost: Duration::ZERO,
            gpu_seconds: 0.,
        };
        let patches = d.alloc(rows * 768 * 2)?;
        let mut first = 0;
        for (&(rgb, w, h), p) in images.iter().zip(&j.plans) {
            if w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
                return Err(error("RGB byte count mismatch"));
            }
            let src = d.upload(rgb)?;
            for (rw, rh, cw, ch, offset) in [
                (384, 384, 384, 384, first),
                (p.resized.0, p.resized.1, p.best.0, p.best.1, first + 1),
            ] {
                let sx = 4 * w.div_ceil(rw) + 4;
                let sy = 4 * h.div_ceil(rh) + 4;
                let cx = d.alloc(rw * sx * 4)?;
                let cy = d.alloc(rh * sy * 4)?;
                let temp = d.alloc(rw * h * 3)?;
                let cmd = d.begin()?;
                for (c, from, to, stride) in [(&cx, w, rw, sx), (&cy, h, rh, sy)] {
                    cmd.dispatch(
                        "gv_coeff",
                        &[c],
                        &[from as u32, to as u32, stride as u32],
                        [to.div_ceil(256), 1, 1],
                        256,
                    );
                }
                cmd.dispatch(
                    "gv_resize_h",
                    &[&src, &cx, &temp],
                    &[w as u32, rw as u32, h as u32, sx as u32],
                    [(rw * h * 3).div_ceil(256), 1, 1],
                    256,
                );
                cmd.dispatch(
                    "grv_patches",
                    &[&temp, &cy, &patches],
                    &[
                        rw as u32,
                        rh as u32,
                        cw as u32,
                        ch as u32,
                        sy as u32,
                        offset as u32,
                        self.mean[0].to_bits(),
                        self.mean[1].to_bits(),
                        self.mean[2].to_bits(),
                        self.std[0].to_bits(),
                        self.std[1].to_bits(),
                        self.std[2].to_bits(),
                    ],
                    [((cw / 16) * (ch / 16) * 768).div_ceil(256), 1, 1],
                    256,
                );
                j.gpu_seconds += cmd.finish()?;
            }
            first += p.n_tiles();
        }
        let cmd = d.begin()?;
        Self::mm(&cmd, &self.patch, &patches, &j.x, rows, 1);
        cmd.dispatch(
            "grv_position",
            &[&j.x, &self.pos.buffer],
            &[rows as u32],
            [(rows * E).div_ceil(256), 1, 1],
            256,
        );
        j.gpu_seconds += cmd.finish()?;
        Ok(j)
    }
    pub(super) fn step(&self, d: &MetalDevice, j: &mut Job) -> Result<Option<Vec<Arc<Features>>>> {
        let started = Instant::now();
        let rows = j.tiles * 576;
        let qr = j.tiles * 144;
        let cmd = d.begin()?;
        if j.phase < 27 {
            let b = &self.blocks[j.phase];
            Self::ln(&cmd, &b.ln1, &j.x, &j.norm, rows, self.eps);
            Self::qkv(&cmd, &b.attn, j, &j.norm, &j.norm, rows, rows, true);
            Self::attend(&cmd, j, &j.tower_tiles, true);
            Self::half(&cmd, &j.attn, &j.stage, rows * E);
            Self::mm(&cmd, &b.attn.out, &j.stage, &j.x, rows, 2);
            Self::ln(&cmd, &b.ln2, &j.x, &j.norm, rows, self.eps);
            Self::half(&cmd, &j.norm, &j.stage, rows * E);
            Self::mm(&cmd, &b.up, &j.stage, &j.up, rows, 3);
            Self::mm(&cmd, &b.down, &j.up, &j.x, rows, 2);
            if self.projectors.iter().any(|p| p.tap == j.phase) {
                let tap = d.alloc(rows * E * 4)?;
                cmd.dispatch(
                    "spec_copy_words",
                    &[&j.x, &tap],
                    &[0, 0, (rows * E) as u32],
                    [(rows * E).div_ceil(256), 1, 1],
                    256,
                );
                j.taps[j.phase] = Some(tap);
            }
        } else {
            let p = &self.projectors[j.phase - 27];
            Self::ln(
                &cmd,
                &p.norm,
                j.taps[p.tap].as_ref().expect("tap recorded"),
                &j.norm,
                rows,
                self.eps,
            );
            cmd.dispatch(
                "grv_window",
                &[&j.norm, &p.pos.buffer, &j.enc],
                &[j.tiles as u32, 8, 0],
                [(rows * E).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "grv_window",
                &[&j.norm, &p.query.buffer, &j.state],
                &[j.tiles as u32, 4, p.spatial],
                [(qr * E).div_ceil(256), 1, 1],
                256,
            );
            Self::ln(&cmd, &p.input, &j.state, &j.state, qr, 1e-12);
            Self::qkv(&cmd, &p.sa, j, &j.state, &j.state, qr, qr, false);
            Self::attend(&cmd, j, &j.self_tiles, false);
            Self::half(&cmd, &j.attn, &j.stage, qr * E);
            Self::mm(&cmd, &p.sa.out, &j.stage, &j.state, qr, 2);
            Self::ln(&cmd, &p.sn, &j.state, &j.state, qr, 1e-12);
            Self::qkv(&cmd, &p.ca, j, &j.state, &j.enc, qr, rows, false);
            Self::attend(&cmd, j, &j.cross_tiles, false);
            Self::half(&cmd, &j.attn, &j.stage, qr * E);
            Self::mm(&cmd, &p.ca.out, &j.stage, &j.state, qr, 2);
            Self::ln(&cmd, &p.cn, &j.state, &j.state, qr, 1e-12);
            Self::half(&cmd, &j.state, &j.stage, qr * E);
            Self::mm(&cmd, &p.up, &j.stage, &j.up, qr, 1);
            cmd.dispatch(
                "mv_gelu",
                &[&j.up],
                &[(qr * p.up.w.n) as u32],
                [(qr * p.up.w.n).div_ceil(256), 1, 1],
                256,
            );
            Self::half(&cmd, &j.up, &j.stage, qr * p.up.w.n);
            Self::mm(&cmd, &p.down, &j.stage, &j.state, qr, 2);
            Self::ln(&cmd, &p.ffn, &j.state, &j.state, qr, 1e-12);
            cmd.dispatch(
                "grv_unwindow",
                &[&j.state, &j.norm],
                &[j.tiles as u32],
                [(qr * E).div_ceil(256), 1, 1],
                256,
            );
            Self::half(&cmd, &j.norm, &j.stage, qr * E);
            let projected = d.alloc(qr * self.width * 4)?;
            Self::mm(&cmd, &p.linear, &j.stage, &projected, qr, 1);
            for ((plan, indices), outputs) in j.plans.iter().zip(&j.indices).zip(&mut j.outputs) {
                let output = d.alloc(plan.n_tokens() * self.width * 4)?;
                cmd.dispatch(
                    "grv_pack",
                    &[&projected, &self.newline.buffer, indices, &output],
                    &[self.width as u32, plan.n_tokens() as u32],
                    [(plan.n_tokens() * self.width).div_ceil(256), 1, 1],
                    256,
                );
                outputs.push(output);
            }
        }
        j.gpu_seconds += cmd.finish()?;
        j.cost = started.elapsed();
        j.phase += 1;
        if j.phase != 35 {
            return Ok(None);
        }
        let bad = d.upload(&0u32.to_le_bytes())?;
        let cmd = d.begin()?;
        for streams in &j.outputs {
            for s in streams {
                cmd.dispatch(
                    "vis_finite",
                    &[s, &bad],
                    &[(s.len() / 4) as u32],
                    [(s.len() / 4).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
        cmd.finish()?;
        if unsafe { bad.read_u32(1)[0] } != 0 {
            return Err(error("nonfinite projector features"));
        }
        Ok(Some(
            std::mem::take(&mut j.outputs)
                .into_iter()
                .zip(&j.plans)
                .map(|(streams, p)| {
                    Arc::new(Features {
                        streams,
                        tokens: p.n_tokens(),
                    })
                })
                .collect(),
        ))
    }
}
