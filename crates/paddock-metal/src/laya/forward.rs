use super::*;

struct Plan {
    ids: Vec<u32>,
    meta: Vec<u32>,
    tiles: Vec<u32>,
    indices: Vec<u32>,
    offsets: Vec<u32>,
    markers: usize,
}
impl Plan {
    fn new(cfg: &LayaConfig, ws: &Workspace, seqs: &[LayaSeq<'_>]) -> Result<Self> {
        let mut p = Self {
            ids: vec![],
            meta: vec![],
            tiles: vec![],
            indices: vec![],
            offsets: vec![0],
            markers: 0,
        };
        if seqs.len() > ws.sequences {
            return Err(error("too many sequences in one pass"));
        }
        let mut cls = vec![];
        for s in seqs {
            let rows = s.ids.len();
            let base = p.ids.len();
            if rows == 0
                || rows > cfg.max_len
                || s.markers.is_empty()
                || s.qtype > 2
                || s.ids.iter().any(|&id| id as usize >= cfg.encoder.vocab)
                || s.markers.iter().any(|&m| m as usize >= rows)
            {
                return Err(error(
                    "invalid sequence length, vocabulary id, marker or question type",
                ));
            }
            if base + rows > ws.rows || p.indices.len() + s.markers.len() + seqs.len() > ws.gather {
                return Err(error("packed pass exceeds workspace capacity"));
            }
            cls.push(base as u32);
            p.ids.extend_from_slice(s.ids);
            for pos in 0..rows {
                p.meta.extend([pos as u32, s.qtype]);
            }
            for first in (0..rows).step_by(QUERY_TILE) {
                p.tiles.extend([
                    (base + first) as u32,
                    rows.saturating_sub(first).min(QUERY_TILE) as u32,
                    base as u32,
                    rows as u32,
                ]);
            }
            p.indices.extend(s.markers.iter().map(|&i| base as u32 + i));
            p.offsets.push(p.indices.len() as u32);
        }
        p.markers = p.indices.len();
        p.indices.extend(cls);
        if p.tiles.len() / 4 > ws.tile_count {
            return Err(error("too many attention tiles"));
        }
        Ok(p)
    }
}

impl Model {
    fn norm(
        &self,
        c: &Commands<'_>,
        ws: &Workspace,
        x: &Buffer,
        bias: &Buffer,
        w: &Buffer,
        b: &Buffer,
        rows: usize,
        eps: f32,
    ) {
        c.dispatch(
            "laya_norm",
            &[x, &ws.proj, bias, w, b, &ws.n],
            &[self.cfg.encoder.hidden as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    fn attention(
        &self,
        c: &Commands<'_>,
        ws: &Workspace,
        rows: usize,
        tiles: usize,
        rope: Option<&Buffer>,
        bias: &Buffer,
        window: usize,
    ) {
        let d = self.cfg.encoder.hidden;
        c.dispatch(
            "laya_qkv",
            &[
                &ws.wide,
                &ws.meta,
                rope.unwrap_or(&self.rope_g),
                bias,
                &ws.q,
                &ws.k,
                &ws.v,
            ],
            &[d as u32, rows as u32, u32::from(rope.is_some())],
            [(rows * d).div_ceil(256), 1, 1],
            256,
        );
        c.dispatch(
            if self.cfg.encoder.n_heads == 12 {
                "laya_attention12"
            } else {
                "laya_attention16"
            },
            &[&ws.q, &ws.k, &ws.v, &ws.att, &ws.tiles],
            &[1, window as u32],
            [self.cfg.encoder.n_heads, tiles, 1],
            64,
        );
    }
    pub(super) fn forward(
        &self,
        device: &MetalDevice,
        ws: &mut Workspace,
        seqs: &[LayaSeq<'_>],
    ) -> Result<LayaOut> {
        let p = Plan::new(&self.cfg, ws, seqs)?;
        let rows = p.ids.len();
        let nm = p.markers;
        let g = p.indices.len();
        let nq = seqs.len();
        if rows == 0 {
            return Ok(LayaOut {
                logits: vec![],
                offsets: vec![0],
                act: vec![],
                n_act: self.cfg.n_act,
                rows: 0,
            });
        }
        // The owning decision thread completed the preceding pass before any
        // upload. These index buffers stay immutable until this pass finishes.
        unsafe {
            ws.ids.write_u32(&p.ids);
            ws.meta.write_u32(&p.meta);
            ws.tiles.write_u32(&p.tiles);
            ws.indices.write_u32(&p.indices);
            ws.offsets.write_u32(&p.offsets);
        }
        let e = &self.cfg.encoder;
        let d = e.hidden;
        let eps = e.eps;
        let z = &self.zeros;
        let c = device.begin()?;
        c.dispatch(
            "laya_embed",
            &[&self.emb, &ws.ids, &self.emb_norm, &ws.x, &ws.n],
            &[d as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
        for (i, l) in self.layers.iter().enumerate() {
            l.qkv.run(&c, &ws.n, &ws.wide, z, rows, 0);
            self.attention(
                &c,
                ws,
                rows,
                p.tiles.len() / 4,
                Some(if l.global { &self.rope_g } else { &self.rope_l }),
                z,
                if l.global { 0 } else { e.window },
            );
            l.out.run(&c, &ws.att, &ws.proj, z, rows, 0);
            self.norm(&c, ws, &ws.x, z, &l.mlp_norm, z, rows, eps);
            l.up.run(&c, &ws.n, &ws.wide, z, rows, 4);
            l.down.run(&c, &ws.wide, &ws.proj, z, rows, 0);
            let next = self
                .layers
                .get(i + 1)
                .and_then(|l| l.attn_norm.as_ref())
                .unwrap_or(&self.final_norm);
            self.norm(&c, ws, &ws.x, z, next, z, rows, eps);
        }
        let h = &self.head[0];
        c.dispatch(
            "laya_head_entry",
            &[
                &ws.x,
                &self.final_norm,
                &self.type_emb,
                &ws.meta,
                &h.n1.w,
                &h.n1.b,
                &ws.n,
            ],
            &[d as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
        for (i, h) in self.head.iter().enumerate() {
            h.qkv.run(&c, &ws.n, &ws.wide, z, rows, 0);
            self.attention(&c, ws, rows, p.tiles.len() / 4, None, &h.qb, 0);
            let last = i + 1 == self.head.len();
            let (count, x, input) = if last {
                c.dispatch(
                    "laya_gather",
                    &[&ws.x, &ws.att, &ws.indices, &ws.xg, &ws.n],
                    &[d as u32, g as u32],
                    [(g * d).div_ceil(256), 1, 1],
                    256,
                );
                (g, &ws.xg, &ws.n)
            } else {
                (rows, &ws.x, &ws.att)
            };
            h.out.run(&c, input, &ws.proj, z, count, 0);
            self.norm(&c, ws, x, &h.ob, &h.n2.w, &h.n2.b, count, 1e-5);
            h.up.run(&c, &ws.n, &ws.wide, &h.ub, count, 2);
            h.down.run(&c, &ws.wide, &ws.proj, z, count, 0);
            let next = if last {
                &self.score_norm
            } else {
                &self.head[i + 1].n1
            };
            self.norm(&c, ws, x, &h.db, &next.w, &next.b, count, 1e-5);
        }
        self.score.run(&c, &ws.n, &ws.att, &self.score_bias, nm, 3);
        c.dispatch(
            "laya_score",
            &[&ws.att, &self.score_w, &ws.logits],
            &[d as u32, self.score_b.to_bits()],
            [nm, 1, 1],
            32,
        );
        c.dispatch(
            "laya_act",
            &[
                &ws.logits,
                &ws.offsets,
                &ws.xg,
                &self.act_w0,
                &self.act_b0,
                &self.act_w2,
                &self.act_b2,
                &ws.act,
            ],
            &[d as u32, nm as u32, self.cfg.n_act as u32],
            [nq, 1, 1],
            256,
        );
        c.finish()?;
        // Completion fences GPU writes before host reads. Only the output
        // logits/probabilities cross this boundary, never model activations.
        let (logits, act) = unsafe {
            (
                ws.logits.read_f32(0, nm),
                ws.act.read_f32(0, nq * self.cfg.n_act),
            )
        };
        if logits.iter().chain(&act).any(|x| !x.is_finite()) {
            return Err(error("nonfinite decision output"));
        }
        Ok(LayaOut {
            logits,
            act,
            offsets: p.offsets.iter().map(|&x| x as usize).collect(),
            n_act: self.cfg.n_act,
            rows,
        })
    }
}
