//! Packed NaViT: independent ragged attention domains, shared BF16 projections,
//! F32 residuals and split-Q online softmax. GPU resize/patchify/interpolation.
//! One step is one tower block (or projector), permitting decode between steps.
use super::*;
use paddock_engine::generator::VisionBudget;
use paddock_models::{gguf::Value, mapped::MappedGguf};
const E: usize = 1152;
const F: usize = 4304;
pub(super) const MAX_PATCHES: usize = 16384;
pub(super) const BUDGET: VisionBudget = VisionBudget {
    min_pixels: 112896,
    max_pixels: 1605632,
    max_edge: None,
    pixels_per_token: 784,
    min_tokens: 144,
    max_tokens: 2048,
};
#[derive(Clone)]
pub(super) struct Input {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub tw: usize,
    pub th: usize,
}
impl Input {
    pub fn patches(&self) -> usize {
        self.tw / 14 * (self.th / 14)
    }
}
struct Norm {
    w: Weight,
    b: Weight,
}
struct Linear {
    w: Weight,
    b: Weight,
}
struct Block {
    ln1: Norm,
    ln2: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
}
pub(super) struct Vision {
    patch: Linear,
    pos: Weight,
    blocks: Vec<Block>,
    post: Norm,
    pre: Norm,
    up: Linear,
    down: Linear,
}
pub(super) struct Job {
    sizes: Vec<usize>,
    rows: usize,
    layer: usize,
    x: Buffer,
    n: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qh: Buffer,
    kh: Buffer,
    vh: Buffer,
    attn: Buffer,
    up: Buffer,
    xy: Buffer,
    tiles: Buffer,
    output: Buffer,
}
pub(super) fn resize(w: usize, h: usize, min: usize, max: usize) -> Result<(usize, usize)> {
    if w == 0 || h == 0 || w > 16384 || h > 16384 || min < 784 || min > max || max > 1605632 {
        return Err(error("image dimensions/budget outside implemented limits"));
    }
    let (mut h, mut w) = (h as f64, w as f64);
    if h < 28. {
        w = (w * 28. / h).round_ties_even();
        h = 28.;
    }
    if w < 28. {
        h = (h * 28. / w).round_ties_even();
        w = 28.;
    }
    if h.max(w) / h.min(w) > 200. {
        return Err(error("image aspect ratio exceeds 200"));
    }
    let (mut th, mut tw) = (
        (h / 28.).round_ties_even() * 28.,
        (w / 28.).round_ties_even() * 28.,
    );
    if th * tw > max as f64 {
        let b = (h * w / max as f64).sqrt();
        th = (h / b / 28.).floor() * 28.;
        tw = (w / b / 28.).floor() * 28.;
    } else if th * tw < min as f64 {
        let b = (min as f64 / (h * w)).sqrt();
        th = (h * b / 28.).ceil() * 28.;
        tw = (w * b / 28.).ceil() * 28.;
    }
    if tw < 28. || th < 28. || tw * th > 1605632. {
        return Err(error("resized image exceeds native patch budget"));
    }
    Ok((tw as usize, th as usize))
}
impl Vision {
    pub(super) fn load(d: &MetalDevice, path: &Path) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        if map.gguf().architecture() != Some("clip")
            || meta.get("clip.projector_type").and_then(Value::as_str) != Some("paddleocr")
            || !matches!(meta.get("clip.has_vision_encoder"), Some(Value::Bool(true)))
            || !matches!(meta.get("clip.use_gelu"), Some(Value::Bool(true)))
        {
            return Err(error("requires PaddleOCR vision companion"));
        }
        for (key, v) in [
            ("embedding_length", E),
            ("feed_forward_length", F),
            ("block_count", 27),
            ("attention.head_count", 16),
            ("patch_size", 14),
            ("projection_dim", WIDTH),
            ("image_min_pixels", 112896),
            ("image_max_pixels", 1003520),
        ] {
            if meta
                .get(&format!("clip.vision.{key}"))
                .and_then(Value::as_u64)
                != Some(v as u64)
            {
                return Err(error(format!("unsupported tower {key}")));
            }
        }
        if meta
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            != Some(1e-6)
        {
            return Err(error("unsupported tower epsilon"));
        }
        for key in ["image_mean", "image_std"] {
            if !matches!(meta.get(&format!("clip.vision.{key}")),Some(Value::Array(a)) if a.len()==3 && a.iter().all(|x|x.as_f32()==Some(0.5)))
            {
                return Err(error("unsupported image normalization"));
            }
        }
        let mut schema = vec![
            ("v.patch_embd.weight".into(), vec![14, 14, 3, E], 0),
            ("v.patch_embd.bias".into(), vec![E], 0),
            ("v.position_embd.weight".into(), vec![E, 729], 0),
        ];
        for i in 0..27 {
            for n in ["ln1", "ln2"] {
                for suffix in ["weight", "bias"] {
                    schema.push((format!("v.blk.{i}.{n}.{suffix}"), vec![E], 0));
                }
            }
            for (n, k, out) in [
                ("attn_q", E, E),
                ("attn_k", E, E),
                ("attn_v", E, E),
                ("attn_out", E, E),
                ("ffn_up", E, F),
                ("ffn_down", F, E),
            ] {
                schema.push((format!("v.blk.{i}.{n}.weight"), vec![k, out], 30));
                schema.push((format!("v.blk.{i}.{n}.bias"), vec![out], 0));
            }
        }
        for n in ["v.post_ln", "mm.input_norm"] {
            for suffix in ["weight", "bias"] {
                schema.push((format!("{n}.{suffix}"), vec![E], 0));
            }
        }
        for (n, k, out) in [("mm.1", 4 * E, 4 * E), ("mm.2", 4 * E, WIDTH)] {
            schema.push((format!("{n}.weight"), vec![k, out], 30));
            schema.push((format!("{n}.bias"), vec![out], 0));
        }
        let payload = load::validate(&map, &schema)?;
        // Reserve a bounded maximum wave plus image-output residency. Actual
        // allocations use live dimensions and remain subject to the same grant.
        if d.allocated_bytes() + payload + Self::workspace_bound() > d.budget_bytes() {
            return Err(MetalError::Memory(
                "PaddleOCR tower + workspace exceed grant".into(),
            ));
        }
        let w = |n: &str, s: &[usize]| Weight::load(d, &map, n, s);
        let norm = |n: &str| -> Result<Norm> {
            Ok(Norm {
                w: w(&format!("{n}.weight"), &[E])?,
                b: w(&format!("{n}.bias"), &[E])?,
            })
        };
        let linear = |n: &str, k: usize, out: usize| -> Result<Linear> {
            Ok(Linear {
                w: w(&format!("{n}.weight"), &[k, out])?,
                b: w(&format!("{n}.bias"), &[out])?,
            })
        };
        let mut patch = w("v.patch_embd.weight", &[14, 14, 3, E])?;
        patch.k = 588;
        patch.n = E;
        let patch = Linear {
            w: patch,
            b: w("v.patch_embd.bias", &[E])?,
        };
        let mut blocks = Vec::new();
        for i in 0..27 {
            let n = |suffix: &str| format!("v.blk.{i}.{suffix}");
            blocks.push(Block {
                ln1: norm(&n("ln1"))?,
                ln2: norm(&n("ln2"))?,
                q: linear(&n("attn_q"), E, E)?,
                k: linear(&n("attn_k"), E, E)?,
                v: linear(&n("attn_v"), E, E)?,
                o: linear(&n("attn_out"), E, E)?,
                up: linear(&n("ffn_up"), E, F)?,
                down: linear(&n("ffn_down"), F, E)?,
            });
        }
        Ok(Self {
            patch,
            pos: w("v.position_embd.weight", &[E, 729])?,
            blocks,
            post: norm("v.post_ln")?,
            pre: norm("mm.input_norm")?,
            up: linear("mm.1", 4 * E, 4 * E)?,
            down: linear("mm.2", 4 * E, WIDTH)?,
        })
    }
    pub(super) fn workspace_bound() -> u64 {
        // Includes slabs, patch staging, output copy, two coefficient tables,
        // original RGB and horizontal u8 temporary at the largest input edges.
        (MAX_PATCHES * (6 * E * 4 + 3 * 1280 * 2 + F * 4 + 588 * 2 + 8 + 16 + WIDTH * 2)
            + 2 * 16384 * 16384 * 3
            + 64 * 1024 * 1024
            + 16 * (MAX_PATCHES / 4) * WIDTH * 4) as u64
    }
    fn ln(cmd: &Commands<'_>, w: &Norm, x: &Buffer, y: &Buffer, rows: usize, eps: f32) {
        cmd.dispatch(
            "vis_ln",
            &[x, &w.w.buffer, &w.b.buffer, y],
            &[E as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    fn mm(cmd: &Commands<'_>, w: &Linear, x: &Buffer, y: &Buffer, rows: usize, mode: u32) {
        cmd.dispatch(
            "vis_bmm32",
            &[&w.w.buffer, x, y, &w.b.buffer],
            &[w.w.k as u32, w.w.n as u32, rows as u32, mode],
            [w.w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
    pub(super) fn start(&self, d: &MetalDevice, images: &[Input]) -> Result<Job> {
        let rows = images.iter().map(Input::patches).sum::<usize>();
        if rows == 0 || rows > MAX_PATCHES {
            return Err(error("tower wave exceeds patch budget"));
        }
        let a = || d.alloc(rows * E * 4);
        let h = || d.alloc((rows + 64) * 1280 * 2);
        let mut xy = Vec::new();
        let mut tiles = Vec::new();
        let mut offset = 0;
        for image in images {
            let pw = image.tw / 14;
            let count = image.patches();
            for row in 0..count {
                xy.extend([
                    (row / 4 % (pw / 2) * 2 + row % 2) as u32,
                    (row / 4 / (pw / 2) * 2 + row % 4 / 2) as u32,
                ]);
            }
            for q in (0..count).step_by(32) {
                tiles.extend([
                    (offset + q) as u32,
                    (count - q).min(32) as u32,
                    offset as u32,
                    count as u32,
                ]);
            }
            offset += count;
        }
        let upload =
            |x: &[u32]| d.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let j = Job {
            sizes: images.iter().map(Input::patches).collect(),
            rows,
            layer: 0,
            x: a()?,
            n: a()?,
            q: a()?,
            k: a()?,
            v: a()?,
            attn: a()?,
            qh: h()?,
            kh: h()?,
            vh: h()?,
            up: d.alloc(rows * F * 4)?,
            output: d.alloc(rows / 4 * WIDTH * 4)?,
            xy: upload(&xy)?,
            tiles: upload(&tiles)?,
        };
        let patches = d.alloc(rows * 588 * 2)?;
        offset = 0;
        for image in images {
            let Input { rgb, w, h, tw, th } = image;
            let sx = 4 * w.div_ceil(*tw) + 4;
            let sy = 4 * h.div_ceil(*th) + 4;
            let cx = d.alloc(tw * sx * 4)?;
            let cy = d.alloc(th * sy * 4)?;
            let src = d.upload(rgb)?;
            let temp = d.alloc(tw * h * 3)?;
            let cmd = d.begin()?;
            for (buf, from, to, stride) in [(&cx, *w, *tw, sx), (&cy, *h, *th, sy)] {
                cmd.dispatch(
                    "gv_coeff",
                    &[buf],
                    &[from as u32, to as u32, stride as u32],
                    [to.div_ceil(256), 1, 1],
                    256,
                );
            }
            cmd.dispatch(
                "gv_resize_h",
                &[&src, &cx, &temp],
                &[*w as u32, *tw as u32, *h as u32, sx as u32],
                [(tw * h * 3).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "pocr_patches",
                &[&temp, &cy, &patches],
                &[*tw as u32, *th as u32, sy as u32, offset as u32],
                [(image.patches() * 588).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
            offset += image.patches();
        }
        let cmd = d.begin()?;
        cmd.dispatch(
            "pocr_patch_mm",
            &[&self.patch.w.buffer, &patches, &j.x, &self.patch.b.buffer],
            &[588, E as u32, rows as u32, 1],
            [E.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
        offset = 0;
        for image in images {
            cmd.dispatch(
                "pocr_position",
                &[&j.x, &self.pos.buffer],
                &[
                    (image.tw / 14) as u32,
                    (image.th / 14) as u32,
                    offset as u32,
                ],
                [(image.patches() * E).div_ceil(256), 1, 1],
                256,
            );
            offset += image.patches();
        }
        cmd.finish()?;
        #[cfg(test)]
        assert!(
            unsafe { j.x.read_f32(0, rows * E) }
                .iter()
                .all(|x| x.is_finite()),
            "nonfinite tower patch input"
        );
        Ok(j)
    }
    pub(super) fn step(&self, d: &MetalDevice, j: &mut Job) -> Result<Option<Vec<Buffer>>> {
        let cmd = d.begin()?;
        let rows = j.rows;
        if j.layer < 27 {
            let l = &self.blocks[j.layer];
            Self::ln(&cmd, &l.ln1, &j.x, &j.n, rows, 1e-6);
            for (w, x, y, rotate) in [
                (&l.q, &j.q, &j.qh, 1),
                (&l.k, &j.k, &j.kh, 1),
                (&l.v, &j.v, &j.vh, 0),
            ] {
                Self::mm(&cmd, w, &j.n, x, rows, 1);
                cmd.dispatch(
                    "pocr_heads",
                    &[x, y, &j.xy],
                    &[rows as u32, rotate],
                    [((rows + 64) * 1280).div_ceil(256), 1, 1],
                    256,
                );
            }
            cmd.dispatch(
                "pocr_attention",
                &[&j.qh, &j.kh, &j.vh, &j.attn, &j.tiles],
                &[0],
                [16, j.tiles.len() / 16, 1],
                64,
            );
            Self::mm(&cmd, &l.o, &j.attn, &j.x, rows, 2);
            Self::ln(&cmd, &l.ln2, &j.x, &j.n, rows, 1e-6);
            Self::mm(&cmd, &l.up, &j.n, &j.up, rows, 3);
            Self::mm(&cmd, &l.down, &j.up, &j.x, rows, 2);
            cmd.finish()?;
            #[cfg(test)]
            for (name, buffer, count) in [
                ("x", &j.x, rows * E),
                ("norm", &j.n, rows * E),
                ("q", &j.q, rows * E),
                ("attn", &j.attn, rows * E),
                ("up", &j.up, rows * F),
            ] {
                assert!(
                    unsafe { buffer.read_f32(0, count) }
                        .iter()
                        .all(|x| x.is_finite()),
                    "nonfinite tower layer {} buffer {name}",
                    j.layer
                );
            }
            j.layer += 1;
            return Ok(None);
        }
        Self::ln(&cmd, &self.post, &j.x, &j.n, rows, 1e-6);
        Self::ln(&cmd, &self.pre, &j.n, &j.x, rows, 1e-5);
        // Merged patch order means [4,E] -> [4E] is a view, not a transpose.
        Self::mm(&cmd, &self.up, &j.x, &j.up, rows / 4, 1);
        cmd.dispatch(
            "pocr_gelu",
            &[&j.up],
            &[(rows * E) as u32],
            [(rows * E).div_ceil(256), 1, 1],
            256,
        );
        Self::mm(&cmd, &self.down, &j.up, &j.output, rows / 4, 1);
        let mut out = Vec::new();
        let mut offset = 0;
        for &size in &j.sizes {
            let b = d.alloc(size / 4 * WIDTH * 4)?;
            cmd.dispatch(
                "pocr_extract",
                &[&j.output, &b],
                &[offset as u32, (size / 4) as u32],
                [(size / 4 * WIDTH).div_ceil(256), 1, 1],
                256,
            );
            offset += size / 4;
            out.push(b);
        }
        cmd.finish()?;
        #[cfg(test)]
        assert!(
            unsafe { j.output.read_f32(0, rows / 4 * WIDTH) }
                .iter()
                .all(|x| x.is_finite()),
            "nonfinite projector"
        );
        Ok(Some(out))
    }
}
