use super::*;
impl Tower {
    pub(super) fn start(&self, d: &MetalDevice, inputs: &[MelFeatures]) -> Result<Job> {
        if inputs.is_empty() || inputs.len() > MAX_CLIPS {
            return Err(error("audio wave requires 1..16 clips"));
        }
        let mut rows = 0;
        let mut sizes = Vec::new();
        let mut clips = Vec::new();
        let mut tiles = Vec::new();
        let mut indices = Vec::new();
        let mut features = Vec::new();
        for m in inputs {
            validate(m)?;
            if rows + m.n_frames > MAX_FRAMES {
                return Err(error("packed wave exceeds 6000 encoder frames"));
            }
            features.extend(m.data.iter().map(|v| v.to_bits()));
            for _ in 0..m.n_frames {
                clips.extend([rows as u32, (rows + m.n_frames) as u32]);
            }
            for b in (0..m.n_frames).step_by(200) {
                let n = (m.n_frames - b).min(200);
                for q in (0..n).step_by(32) {
                    tiles.extend([
                        (rows + b + q) as u32,
                        (n - q).min(32) as u32,
                        (rows + b) as u32,
                        n as u32,
                    ]);
                }
            }
            let pad = m.n_frames.div_ceil(15) * 15;
            indices.extend((0..pad).map(|i| {
                if i < m.n_frames {
                    (rows + i) as u32
                } else {
                    u32::MAX
                }
            }));
            sizes.push(pad / 5);
            rows += m.n_frames;
        }
        if d.allocated_bytes() + WORKSPACE > d.budget_bytes() {
            return Err(MetalError::Memory(
                "Granite Speech wave exceeds grant".into(),
            ));
        }
        let padded = indices.len();
        let queries = padded / 5;
        let capacity = rows.max(padded);
        let a = |n: usize| d.alloc(capacity * n * 4);
        let j = Job {
            sizes,
            rows,
            padded,
            queries,
            phase: 0,
            tiles: upload(d, &tiles)?,
            clips: upload(d, &clips)?,
            indices: upload(d, &indices)?,
            x: a(E)?,
            norm: a(2048)?,
            tmp: a(E)?,
            up: a(F)?,
            glu: a(2048)?,
            qkv: a(3 * E)?,
            q: a(E)?,
            k: a(E)?,
            v: a(E)?,
            qr: a(8 * 401)?,
            attn: a(E)?,
            ctc: a(348)?,
            tap: if self.plus { a(E)? } else { d.alloc(4)? },
            enc: d.alloc(padded * E * if self.plus { 8 } else { 4 })?,
            query: d.alloc(queries * E * 4)?,
        };
        let input = upload(d, &features)?;
        let c = d.begin()?;
        Self::linear(&c, &self.input, &input, &j.x, rows, 1);
        c.finish()?;
        Ok(j)
    }
    pub(super) fn norm(
        c: &Commands<'_>,
        n: &Norm,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
        eps: f32,
    ) {
        c.dispatch(
            "vis_ln",
            &[x, &n.w.buffer, &n.b.buffer, out],
            &[E as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    pub(super) fn plain(
        c: &Commands<'_>,
        w: &Weight,
        b: &Buffer,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
        epilogue: u32,
    ) {
        c.dispatch(
            if w.ty == 1 { "qasr_half_mm" } else { "gs_fmm" },
            &[&w.buffer, x, out, b],
            &[w.k as u32, w.n as u32, rows as u32, epilogue],
            [w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
    fn linear(c: &Commands<'_>, w: &Linear, x: &Buffer, out: &Buffer, rows: usize, epilogue: u32) {
        Self::plain(c, &w.w, &w.b.buffer, x, out, rows, epilogue);
    }
    fn residual(c: &Commands<'_>, x: &Buffer, y: &Buffer, rows: usize, scale: f32) {
        c.dispatch(
            "gs_residual",
            &[x, y],
            &[(rows * E) as u32, scale.to_bits()],
            [(rows * E).div_ceil(256), 1, 1],
            256,
        );
    }
    fn silu(c: &Commands<'_>, x: &Buffer, n: usize) {
        c.dispatch("gs_silu", &[x], &[n as u32], [n.div_ceil(256), 1, 1], 256);
    }
    fn project_attention(c: &Commands<'_>, a: &Attention, j: &Job, cross: bool) {
        let n = j.queries;
        let keys = if cross { j.padded } else { n };
        let input = if cross { &j.enc } else { &j.query };
        Self::linear(c, &a.q, &j.query, &j.q, n, 1);
        Self::linear(c, &a.k, input, &j.k, keys, 1);
        Self::linear(c, &a.v, input, &j.v, keys, 1);
        c.dispatch(
            "gs_qattention",
            &[&j.q, &j.k, &j.v, &j.attn],
            &[n as u32, if cross { 15 } else { 3 }],
            [16, n.div_ceil(4), 1],
            128,
        );
        Self::linear(c, &a.out, &j.attn, &j.query, n, 2);
        Self::norm(c, &a.norm, &j.query, &j.query, n, 1e-12);
    }
    pub(super) fn step(&self, d: &MetalDevice, j: &mut Job) -> Result<Option<Vec<Buffer>>> {
        if j.phase > 51 {
            return Err(error("audio job already completed"));
        }
        let c = d.begin()?;
        let r = j.rows;
        if j.phase < 48 {
            let layer = j.phase / 3;
            let block = &self.blocks[layer];
            match j.phase % 3 {
                0 => {
                    Self::norm(&c, &block.ff1, &j.x, &j.norm, r, 1e-5);
                    Self::linear(&c, &block.up1, &j.norm, &j.up, r, 1);
                    Self::silu(&c, &j.up, r * F);
                    Self::linear(&c, &block.down1, &j.up, &j.tmp, r, 1);
                    Self::residual(&c, &j.x, &j.tmp, r, 0.5);
                    Self::norm(&c, &block.attn, &j.x, &j.norm, r, 1e-5);
                    Self::plain(&c, &block.qkv, &j.tmp, &j.norm, &j.qkv, r, 0);
                    c.dispatch(
                        "gs_split",
                        &[&j.qkv, &j.q, &j.k, &j.v],
                        &[r as u32],
                        [(r * E).div_ceil(256), 1, 1],
                        256,
                    );
                    Self::plain(&c, &block.rel, &j.tmp, &j.q, &j.qr, r * 8, 0);
                    c.dispatch(
                        "gs_attention",
                        &[&j.q, &j.k, &j.v, &j.attn, &j.tiles, &j.qr],
                        &[0],
                        [8, j.tiles.len() / 16, 1],
                        64,
                    );
                    Self::linear(&c, &block.out, &j.attn, &j.x, r, 2);
                }
                1 => {
                    Self::norm(&c, &block.conv, &j.x, &j.norm, r, 1e-5);
                    Self::linear(&c, &block.pw1, &j.norm, &j.up, r, 1);
                    c.dispatch(
                        "gs_glu",
                        &[&j.up, &j.glu],
                        &[r as u32, 2048],
                        [(r * 2048).div_ceil(256), 1, 1],
                        256,
                    );
                    c.dispatch(
                        "gs_depthwise",
                        &[
                            &j.glu,
                            &block.dw.buffer,
                            &block.bn.w.buffer,
                            &block.bn.b.buffer,
                            &j.clips,
                            &j.norm,
                        ],
                        &[r as u32, 2048],
                        [(r * 2048).div_ceil(256), 1, 1],
                        256,
                    );
                    Self::linear(&c, &block.pw2, &j.norm, &j.x, r, 2);
                }
                _ => {
                    Self::norm(&c, &block.ff2, &j.x, &j.norm, r, 1e-5);
                    Self::linear(&c, &block.up2, &j.norm, &j.up, r, 1);
                    Self::silu(&c, &j.up, r * F);
                    Self::linear(&c, &block.down2, &j.up, &j.tmp, r, 1);
                    Self::residual(&c, &j.x, &j.tmp, r, 0.5);
                    Self::norm(&c, &block.post, &j.x, &j.x, r, 1e-5);
                    if self.plus && layer == 2 {
                        c.dispatch(
                            "spec_copy_words",
                            &[&j.x, &j.tap],
                            &[0, 0, (r * E) as u32],
                            [(r * E).div_ceil(256), 1, 1],
                            256,
                        );
                    }
                    if layer == 7 {
                        Self::linear(&c, &self.ctc, &j.x, &j.ctc, r, 1);
                        c.dispatch("gs_softmax", &[&j.ctc], &[348], [r, 1, 1], 32);
                        Self::linear(&c, &self.mid, &j.ctc, &j.x, r, 2);
                    }
                }
            }
        } else if j.phase == 48 {
            c.dispatch(
                "gs_windows",
                &[&j.x, &j.tap, &j.indices, &j.enc],
                &[j.padded as u32, if self.plus { 2 } else { 1 }],
                [
                    (j.padded * E * if self.plus { 2 } else { 1 }).div_ceil(256),
                    1,
                    1,
                ],
                256,
            );
            c.dispatch(
                "gs_queries",
                &[&self.queries, &j.query],
                &[j.queries as u32],
                [(j.queries * E).div_ceil(256), 1, 1],
                256,
            );
        } else if j.phase < 51 {
            let p = &self.projectors[j.phase - 49];
            let n = j.queries;
            Self::project_attention(&c, &p.sa, j, false);
            Self::project_attention(&c, &p.ca, j, true);
            Self::linear(&c, &p.up, &j.query, &j.up, n, 1);
            c.dispatch(
                "uov_activation",
                &[&j.up],
                &[(n * F) as u32, 0],
                [(n * F).div_ceil(256), 1, 1],
                256,
            );
            Self::linear(&c, &p.down, &j.up, &j.query, n, 2);
            Self::norm(&c, &p.norm, &j.query, &j.query, n, 1e-12);
        } else {
            let out = d.alloc(j.queries * 2048 * 4)?;
            Self::linear(&c, &self.output, &j.query, &out, j.queries, 1);
            let mut result = Vec::new();
            let mut offset = 0;
            for &n in &j.sizes {
                let buf = d.alloc(n * 2048 * 4)?;
                c.dispatch(
                    "qasr_extract",
                    &[&out, &buf],
                    &[(n * 2048) as u32, (offset * 2048) as u32],
                    [(n * 2048).div_ceil(256), 1, 1],
                    256,
                );
                offset += n;
                result.push(buf);
            }
            c.finish()?;
            j.phase += 1;
            return Ok(Some(result));
        }
        c.finish()?;
        j.phase += 1;
        Ok(None)
    }
}
