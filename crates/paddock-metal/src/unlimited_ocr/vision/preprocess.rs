use super::*;
pub(super) fn position(
    d: &MetalDevice,
    w: &Weight,
    src: usize,
    dst: usize,
    ch: usize,
    cls: usize,
) -> Result<Buffer> {
    let temp = d.alloc(src * dst * ch * 4)?;
    let out = d.alloc(dst * dst * ch * 4)?;
    let c = d.begin()?;
    c.dispatch(
        "uov_resize_pos",
        &[&w.buffer, &temp],
        &[src as u32, dst as u32, ch as u32, cls as u32, 0],
        [(src * dst * ch).div_ceil(256), 1, 1],
        256,
    );
    c.dispatch(
        "uov_resize_pos",
        &[&temp, &out],
        &[src as u32, dst as u32, ch as u32, 0, 1],
        [(dst * dst * ch).div_ceil(256), 1, 1],
        256,
    );
    c.finish()?;
    Ok(out)
}
pub(super) fn tiles(d: &MetalDevice, n: usize, images: usize) -> Result<Buffer> {
    let mut v = Vec::new();
    for image in 0..images {
        for q in (0..n).step_by(32) {
            for x in [image * n + q, (n - q).min(32), image * n, n] {
                v.extend_from_slice(&(x as u32).to_le_bytes());
            }
        }
    }
    d.upload(&v)
}
impl Vision {
    pub(in crate::unlimited_ocr) fn start(&self, d: &MetalDevice, input: Input) -> Result<Job> {
        let source = d.upload(&input.rgb)?;
        let output = d.alloc(input.tokens() * WIDTH * 4)?;
        let c = d.begin()?;
        c.dispatch(
            "uov_separators",
            &[&self.nl.buffer, &self.separator.buffer, &output],
            &[
                input.rows as u32,
                input.cols as u32,
                (input.rows * 10 * (input.cols * 10 + 1)) as u32,
            ],
            [((input.rows * 10 + 17) * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        c.finish()?;
        Ok(Job {
            input,
            source,
            output: Some(output),
            next: 0,
            global_done: false,
            wave: None,
        })
    }
    pub(super) fn prepare(&self, d: &MetalDevice, j: &Job) -> Result<Wave> {
        let local = j.next < j.input.rows * j.input.cols;
        let count = if local {
            (j.input.rows * j.input.cols - j.next).min(4)
        } else {
            1
        };
        let px = if local { 640 } else { 1024 };
        let grid = px / 16;
        let n = count * grid * grid;
        let part = count * grid.div_ceil(14).pow(2) * 196;
        let maxrows = part.max(n);
        let tokens = (grid / 4).pow(2);
        let nc = count * (tokens + 1);
        let a = |n: usize| d.alloc(n * 4);
        let w = Wave {
            count,
            grid,
            first: j.next,
            layer: 0,
            x: a(n * 768)?,
            n: a(maxrows * 1024)?,
            part: a(maxrows * 768)?,
            qkv: a(maxrows * 2304)?,
            q: d.alloc((maxrows + 64) * 1024 * 2)?,
            k: d.alloc((maxrows + 64) * 1024 * 2)?,
            v: d.alloc((maxrows + 64) * 1024 * 2)?,
            attn: a(maxrows * 768)?,
            ff: a(maxrows * 3072)?,
            rh: a(maxrows * 12 * grid)?,
            rw: a(maxrows * 12 * grid)?,
            gather: a(n * 9 * 256)?,
            neck: a(n * 256)?,
            neck_next: a(n * 256)?,
            sam: a(count * tokens * 1024)?,
            cx: a(nc * 1024)?,
            concat: a(count * tokens * 2048)?,
            projected: a(count * tokens * 1280)?,
            sam_pos: position(d, &self.sam_pos, 64, grid, 768, 0)?,
            clip_pos: position(d, &self.clip_pos, 16, grid / 4, 1024, 1)?,
            global_tiles: tiles(d, grid * grid, count)?,
            window_tiles: tiles(d, 196, count * grid.div_ceil(14).pow(2))?,
            clip_tiles: tiles(d, tokens + 1, count)?,
        };
        let (tw, th, ox, oy) = if local {
            (j.input.cols * 640, j.input.rows * 640, 0, 0)
        } else {
            let (tw, th) = if j.input.w >= j.input.h {
                (
                    1024,
                    (j.input.h as f64 / j.input.w as f64 * 1024.).round_ties_even() as usize,
                )
            } else {
                (
                    (j.input.w as f64 / j.input.h as f64 * 1024.).round_ties_even() as usize,
                    1024,
                )
            };
            (
                tw,
                th,
                ((1024 - tw) as f64 * 0.5).round_ties_even() as usize,
                ((1024 - th) as f64 * 0.5).round_ties_even() as usize,
            )
        };
        if tw == 0 || th == 0 {
            return Err(error("image aspect too extreme for global view"));
        }
        let sx = 4 * j.input.w.div_ceil(tw) + 4;
        let sy = 4 * j.input.h.div_ceil(th) + 4;
        let cx = a(tw * sx)?;
        let cy = a(th * sy)?;
        let temp = d.alloc(tw * j.input.h * 3)?;
        let patches = d.alloc(n * 768 * 2)?;
        let c = d.begin()?;
        for (buf, from, to, stride) in [(&cx, j.input.w, tw, sx), (&cy, j.input.h, th, sy)] {
            c.dispatch(
                "gv_coeff",
                &[buf],
                &[from as u32, to as u32, stride as u32],
                [to.div_ceil(256), 1, 1],
                256,
            );
        }
        c.dispatch(
            "gv_resize_h",
            &[&j.source, &cx, &temp],
            &[j.input.w as u32, tw as u32, j.input.h as u32, sx as u32],
            [(tw * j.input.h * 3).div_ceil(256), 1, 1],
            256,
        );
        for v in 0..count {
            let view = j.next + v;
            let (col, row) = if local {
                (view % j.input.cols, view / j.input.cols)
            } else {
                (0, 0)
            };
            c.dispatch(
                "uov_patches",
                &[&temp, &cy, &patches],
                &[
                    tw as u32,
                    th as u32,
                    px as u32,
                    sy as u32,
                    col as u32,
                    row as u32,
                    ox as u32,
                    oy as u32,
                    (!local) as u32,
                    (v * grid * grid) as u32,
                ],
                [(grid * grid * 768).div_ceil(256), 1, 1],
                256,
            );
        }
        c.dispatch(
            "vis_mm32",
            &[&self.patch.w.buffer, &patches, &w.x, &self.patch.b.buffer],
            &[768, 768, n as u32, 1],
            [12, n.div_ceil(32), 1],
            128,
        );
        c.dispatch(
            "uov_add_pos",
            &[&w.x, &w.sam_pos],
            &[n as u32, 768, (grid * grid) as u32],
            [(n * 768).div_ceil(256), 1, 1],
            256,
        );
        c.finish()?;
        Ok(w)
    }
}
