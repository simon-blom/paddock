use super::*;
impl Vision {
    pub(in crate::unlimited_ocr) fn step(
        &self,
        d: &MetalDevice,
        j: &mut Job,
    ) -> Result<Option<Buffer>> {
        if j.wave.is_none() {
            j.wave = Some(self.prepare(d, j)?);
            return Ok(None);
        }
        let w = j.wave.as_mut().expect("prepared");
        let n = w.count * w.grid * w.grid;
        let c = d.begin()?;
        if w.layer < 12 {
            let l = &self.sam[w.layer];
            let global = w.layer % 3 == 2;
            let side = if global { w.grid } else { 14 };
            let rows = if global {
                n
            } else {
                w.count * w.grid.div_ceil(14).pow(2) * 196
            };
            Self::norm(&c, &l.ln1, &w.x, &w.n, n, 1e-6);
            if !global {
                c.dispatch(
                    "uov_partition",
                    &[&w.n, &w.part],
                    &[w.grid as u32, w.count as u32],
                    [(rows * 768).div_ceil(256), 1, 1],
                    256,
                );
            }
            Self::linear(
                &c,
                &l.qkv,
                if global { &w.n } else { &w.part },
                &w.qkv,
                rows,
                1,
            );
            c.dispatch(
                "uov_heads",
                &[&w.qkv, &w.q, &w.k, &w.v],
                &[rows as u32, 768],
                [((rows + 64) * 768).div_ceil(256), 1, 1],
                256,
            );
            let (rh, rw) = l.relative.as_ref().expect("SAM relative weights");
            c.dispatch(
                "uov_relative",
                &[&w.qkv, &rh.buffer, &rw.buffer, &w.rh, &w.rw],
                &[rows as u32, side as u32, rh.n as u32],
                [12 * side, rows, 1],
                32,
            );
            let tiles = if global {
                &w.global_tiles
            } else {
                &w.window_tiles
            };
            c.dispatch(
                "uov_sam_attention",
                &[&w.q, &w.k, &w.v, &w.attn, tiles, &w.rh, &w.rw],
                &[0, side as u32],
                [12, tiles.len() / 16, 1],
                64,
            );
            if !global {
                c.dispatch(
                    "uov_unpartition",
                    &[&w.attn, &w.part],
                    &[w.grid as u32, w.count as u32],
                    [(n * 768).div_ceil(256), 1, 1],
                    256,
                );
            }
            Self::linear(
                &c,
                &l.out,
                if global { &w.attn } else { &w.part },
                &w.x,
                n,
                2,
            );
            Self::norm(&c, &l.ln2, &w.x, &w.n, n, 1e-6);
            Self::linear(&c, &l.up, &w.n, &w.ff, n, 1);
            c.dispatch(
                "uov_activation",
                &[&w.ff],
                &[(n * 3072) as u32, 0],
                [(n * 3072).div_ceil(256), 1, 1],
                256,
            );
            Self::linear(&c, &l.down, &w.ff, &w.x, n, 2);
        } else if w.layer == 12 {
            Self::plain(&c, &self.neck0, &w.x, &w.neck, n);
            Self::norm(&c, &self.neck1, &w.neck, &w.n, n, 1e-6);
            c.dispatch(
                "uov_conv_rows",
                &[&w.n, &w.gather],
                &[w.grid as u32, 256, 1, w.count as u32],
                [(n * 9 * 256).div_ceil(256), 1, 1],
                256,
            );
            Self::plain(&c, &self.neck2, &w.gather, &w.neck, n);
            Self::norm(&c, &self.neck3, &w.neck, &w.n, n, 1e-6);
            c.dispatch(
                "uov_conv_rows",
                &[&w.n, &w.gather],
                &[w.grid as u32, 256, 2, w.count as u32],
                [(n / 4 * 9 * 256).div_ceil(256), 1, 1],
                256,
            );
            Self::plain(&c, &self.net2, &w.gather, &w.neck_next, n / 4);
            c.dispatch(
                "uov_conv_rows",
                &[&w.neck_next, &w.gather],
                &[(w.grid / 2) as u32, 512, 2, w.count as u32],
                [(n / 16 * 9 * 512).div_ceil(256), 1, 1],
                256,
            );
            Self::plain(&c, &self.net3, &w.gather, &w.sam, n / 16);
            let tokens = (w.grid / 4).pow(2);
            let nc = w.count * (tokens + 1);
            c.dispatch(
                "uov_clip_embed",
                &[
                    &w.sam,
                    &self.cls.buffer,
                    &w.clip_pos,
                    &self.clip_pos.buffer,
                    &w.cx,
                ],
                &[tokens as u32, w.count as u32],
                [(nc * 1024).div_ceil(256), 1, 1],
                256,
            );
            Self::norm(&c, &self.pre, &w.cx, &w.n, nc, 1e-5);
            // Distinct source/destination avoids an in-place LayerNorm alias.
            c.dispatch(
                "uov_copy",
                &[&w.n, &w.cx],
                &[(nc * 1024) as u32],
                [(nc * 1024).div_ceil(256), 1, 1],
                256,
            );
        } else if w.layer < 37 {
            let l = &self.clip[w.layer - 13];
            let tokens = (w.grid / 4).pow(2);
            let rows = w.count * (tokens + 1);
            Self::norm(&c, &l.ln1, &w.cx, &w.n, rows, 1e-5);
            Self::linear(&c, &l.qkv, &w.n, &w.qkv, rows, 1);
            c.dispatch(
                "uov_heads",
                &[&w.qkv, &w.q, &w.k, &w.v],
                &[rows as u32, 1024],
                [((rows + 64) * 1024).div_ceil(256), 1, 1],
                256,
            );
            c.dispatch(
                "uov_clip_attention",
                &[&w.q, &w.k, &w.v, &w.attn, &w.clip_tiles],
                &[0],
                [16, w.clip_tiles.len() / 16, 1],
                64,
            );
            Self::linear(&c, &l.out, &w.attn, &w.cx, rows, 2);
            Self::norm(&c, &l.ln2, &w.cx, &w.n, rows, 1e-5);
            Self::linear(&c, &l.up, &w.n, &w.ff, rows, 1);
            c.dispatch(
                "uov_activation",
                &[&w.ff],
                &[(rows * 4096) as u32, 1],
                [(rows * 4096).div_ceil(256), 1, 1],
                256,
            );
            Self::linear(&c, &l.down, &w.ff, &w.cx, rows, 2);
        } else {
            let side = w.grid / 4;
            let tokens = side * side;
            c.dispatch(
                "uov_concat",
                &[&w.cx, &w.sam, &w.concat],
                &[tokens as u32, w.count as u32],
                [(tokens * w.count * 2048).div_ceil(256), 1, 1],
                256,
            );
            Self::linear(
                &c,
                &self.projector,
                &w.concat,
                &w.projected,
                tokens * w.count,
                1,
            );
            let local = w.grid == 40;
            let offset = if local {
                0
            } else {
                j.input.rows * 10 * (j.input.cols * 10 + 1)
            };
            c.dispatch(
                "uov_assemble",
                &[&w.projected, j.output.as_ref().expect("owned output")],
                &[
                    side as u32,
                    w.count as u32,
                    if local { w.first as u32 } else { 0 },
                    if local { j.input.cols as u32 } else { 1 },
                    0,
                    offset as u32,
                ],
                [(tokens * w.count * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        }
        c.finish()?;
        #[cfg(test)]
        {
            let (buf, count) = if w.layer < 12 {
                (&w.x, n * 768)
            } else if w.layer < 37 {
                (&w.cx, w.count * ((w.grid / 4).pow(2) + 1) * 1024)
            } else {
                (&w.projected, w.count * (w.grid / 4).pow(2) * 1280)
            };
            assert!(
                unsafe { buf.read_f32(0, count) }
                    .iter()
                    .all(|x| x.is_finite()),
                "nonfinite DeepEncoder stage {}",
                w.layer
            );
        }
        w.layer += 1;
        if w.layer == 38 {
            if w.grid == 40 {
                j.next += w.count;
            } else {
                j.global_done = true;
            }
            j.wave = None;
            if j.global_done {
                return Ok(j.output.take());
            }
        }
        Ok(None)
    }
}
