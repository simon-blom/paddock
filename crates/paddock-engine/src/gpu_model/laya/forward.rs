//! One pass: packed question sequences in, option logits and act
//! probabilities out. The op train is in the module note; this file is the
//! host side of it - the pass's index planes, built once, uploaded once - and
//! the launch order.

use super::{ACT_HIDDEN, GpuLaya, GpuModelError, LayaOut, LayaSeq, LayaWorkspace};
use crate::gpu::{ENC_ATTN_QTILE, ENC_ATTN_TILE_SHIFT};

/// The index planes of one pass, built on the host.
struct Plan {
    rows: usize,
    ids: Vec<u32>,
    rtype: Vec<u32>,
    cu: Vec<u32>,
    tiles: Vec<u32>,
    /// the rows the head's last layer continues on: every marker (sequence
    /// order, option order), then every sequence's [CLS] row
    gidx: Vec<u32>,
    qoff: Vec<u32>,
    n_markers: usize,
}

impl Plan {
    fn build(seqs: &[LayaSeq], max_len: usize) -> Result<Self, GpuModelError> {
        let rows: usize = seqs.iter().map(|s| s.ids.len()).sum();
        let n_markers: usize = seqs.iter().map(|s| s.markers.len()).sum();
        let mut p = Plan {
            rows,
            ids: Vec::with_capacity(rows),
            rtype: Vec::with_capacity(rows),
            cu: Vec::with_capacity(seqs.len() + 1),
            tiles: Vec::new(),
            gidx: Vec::with_capacity(n_markers + seqs.len()),
            qoff: Vec::with_capacity(seqs.len() + 1),
            n_markers,
        };
        let (mut base, mut mk) = (0u32, 0u32);
        for (s, q) in seqs.iter().enumerate() {
            let len = q.ids.len();
            if len == 0 || len > max_len {
                return Err(GpuModelError::ContextExceeded {
                    got: len,
                    max: max_len,
                });
            }
            if q.markers.is_empty() {
                return Err(GpuModelError::Unsupported(format!(
                    "laya: sequence {s} has no option markers"
                )));
            }
            if q.qtype > 2 {
                return Err(GpuModelError::Unsupported(format!(
                    "laya: question type {} (0 choice, 1 score, 2 noul)",
                    q.qtype
                )));
            }
            p.cu.push(base);
            p.qoff.push(mk);
            p.ids.extend_from_slice(q.ids);
            p.rtype.extend(std::iter::repeat_n(q.qtype, len));
            for t in 0..len.div_ceil(ENC_ATTN_QTILE) {
                p.tiles.push(((s as u32) << ENC_ATTN_TILE_SHIFT) | t as u32);
            }
            for &m in q.markers {
                if m as usize >= len {
                    return Err(GpuModelError::Unsupported(format!(
                        "laya: sequence {s}: marker {m} past its {len} tokens"
                    )));
                }
                p.gidx.push(base + m);
            }
            base += len as u32;
            mk += q.markers.len() as u32;
        }
        p.cu.push(base);
        p.qoff.push(mk);
        // the [CLS] rows ride after the markers: the act head reads them there
        p.gidx.extend(p.cu[..seqs.len()].iter().copied());
        Ok(p)
    }
}

impl GpuLaya {
    /// Run `seqs` through the model in one pass. Every sequence must fit the
    /// checkpoint's `max_len` and the pass the workspace.
    pub fn forward(
        &self,
        ws: &mut LayaWorkspace,
        seqs: &[LayaSeq],
    ) -> Result<LayaOut, GpuModelError> {
        let exec = self.exec.clone();
        let e = &self.cfg.encoder;
        let (d, hd, heads, eps) = (e.hidden, e.head_dim(), e.n_heads, e.eps);
        if seqs.is_empty() {
            return Ok(LayaOut {
                logits: Vec::new(),
                offsets: vec![0],
                act: Vec::new(),
                n_act: self.cfg.n_act,
                rows: 0,
            });
        }
        let (wd, wide) = self.plane_dims();
        if wd > ws.d || wide > ws.wide {
            return Err(GpuModelError::Unsupported(format!(
                "laya: the workspace was sized for planes {}x{} and this checkpoint needs {wd}x{wide}",
                ws.d, ws.wide
            )));
        }
        let p = Plan::build(seqs, self.cfg.max_len)?;
        let (rows, nq, nm) = (p.rows, seqs.len(), p.n_markers);
        let g = p.gidx.len();
        if rows > ws.rows_cap || nq > ws.seq_cap || g > ws.gather_cap || p.tiles.len() > ws.tile_cap
        {
            return Err(GpuModelError::BatchTooLarge {
                got: rows,
                max: ws.rows_cap,
            });
        }
        exec.upload_u32(&p.ids, &mut ws.ids)?;
        exec.upload_u32(&p.rtype, &mut ws.rtype)?;
        exec.upload_u32(&p.cu, &mut ws.cu)?;
        exec.upload_u32(&p.tiles, &mut ws.tiles)?;
        exec.upload_u32(&p.gidx, &mut ws.gidx)?;
        exec.upload_u32(&p.qoff, &mut ws.qoff)?;
        let n_tiles = p.tiles.len();

        // ---- ModernBERT ----
        exec.enc_embed_ln(
            &self.emb,
            &ws.ids,
            &self.emb_norm,
            None,
            &mut ws.x,
            &mut ws.n16,
            rows,
            d,
            eps,
        )?;
        let n_layer = self.layers.len();
        for li in 0..n_layer {
            let l = &self.layers[li];
            exec.matvec_batch_f16_h(&l.wqkv, &ws.n16, &mut ws.wide16, rows)?;
            let (rope, window) = if l.global {
                (&self.rope_g, 0)
            } else {
                (&self.rope_l, e.window)
            };
            exec.enc_attn_h(
                &ws.wide16,
                &ws.cu,
                &ws.tiles,
                n_tiles,
                Some((&rope.0, &rope.1)),
                None,
                &mut ws.att,
                rows,
                heads,
                hd,
                window,
            )?;
            exec.matvec_batch_f16_h(&l.wo, &ws.att, &mut ws.proj, rows)?;
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &self.zeros,
                &self.ones,
                &l.mlp_norm,
                &self.zeros,
                &mut ws.n16,
                rows,
                d,
                eps,
            )?;
            exec.matvec_batch_f16_h_geglu(&l.wi, &ws.n16, &mut ws.wide16, rows)?;
            exec.matvec_batch_f16_h(&l.wo2, &ws.wide16, &mut ws.proj, rows)?;
            // the seam lands the next layer's pre-norm; after the last layer
            // that norm is final_norm, which the head entry recomputes in f32
            // from the residual - the half it lands here is never read
            let next = self
                .layers
                .get(li + 1)
                .and_then(|n| n.attn_norm.as_ref())
                .unwrap_or(&self.final_norm);
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &self.zeros,
                &self.ones,
                next,
                &self.zeros,
                &mut ws.n16,
                rows,
                d,
                eps,
            )?;
        }

        // ---- the decision head ----
        let h0 = &self.head[0];
        exec.laya_head_entry(
            &mut ws.x,
            &self.final_norm,
            &self.temb,
            &ws.rtype,
            &h0.n1w,
            &h0.n1b,
            &mut ws.n16,
            rows,
            d,
            eps,
        )?;
        let hl = self.head.len();
        for (hi, h) in self.head.iter().enumerate() {
            exec.matvec_batch_f16_h(&h.in_w, &ws.n16, &mut ws.wide16, rows)?;
            exec.enc_attn_h(
                &ws.wide16,
                &ws.cu,
                &ws.tiles,
                n_tiles,
                None,
                Some(&h.in_b),
                &mut ws.att,
                rows,
                heads,
                hd,
                0,
            )?;
            if hi + 1 < hl {
                let n = &self.head[hi + 1];
                exec.matvec_batch_f16_h(&h.out_w, &ws.att, &mut ws.proj, rows)?;
                exec.dp_res_ls_ln_h(
                    &mut ws.x,
                    &ws.proj,
                    &h.out_b,
                    &self.ones,
                    &h.n2w,
                    &h.n2b,
                    &mut ws.n16,
                    rows,
                    d,
                    eps,
                )?;
                exec.matvec_batch_f16_h_relu(&h.l1w, &ws.n16, &mut ws.wide16, Some(&h.l1b), rows)?;
                exec.matvec_batch_f16_h(&h.l2w, &ws.wide16, &mut ws.proj, rows)?;
                exec.dp_res_ls_ln_h(
                    &mut ws.x,
                    &ws.proj,
                    &h.l2b,
                    &self.ones,
                    &n.n1w,
                    &n.n1b,
                    &mut ws.n16,
                    rows,
                    d,
                    eps,
                )?;
                continue;
            }
            // The last layer: its keys needed every row, nothing after it does.
            // Continue on the markers + [CLS] rows only (g of them).
            exec.gather_rows_f16(&ws.att, &ws.gidx, &mut ws.n16, g, d)?;
            exec.gather_rows_f32(&ws.x, &ws.gidx, &mut ws.xg, g, d)?;
            exec.matvec_batch_f16_h(&h.out_w, &ws.n16, &mut ws.proj, g)?;
            exec.dp_res_ls_ln_h(
                &mut ws.xg,
                &ws.proj,
                &h.out_b,
                &self.ones,
                &h.n2w,
                &h.n2b,
                &mut ws.n16,
                g,
                d,
                eps,
            )?;
            exec.matvec_batch_f16_h_relu(&h.l1w, &ws.n16, &mut ws.wide16, Some(&h.l1b), g)?;
            exec.matvec_batch_f16_h(&h.l2w, &ws.wide16, &mut ws.proj, g)?;
            // this seam's "next norm" is the scorer's own LayerNorm
            exec.dp_res_ls_ln_h(
                &mut ws.xg,
                &ws.proj,
                &h.l2b,
                &self.ones,
                &self.s0w,
                &self.s0b,
                &mut ws.n16,
                g,
                d,
                eps,
            )?;
        }

        // ---- scorer: Linear(d, d) + GELU, then Linear(d, 1), at the markers ----
        exec.matvec_batch_f16_h_gelu(&self.s1w, &ws.n16, &mut ws.att, &self.s1b, nm)?;
        exec.laya_rowdot(&ws.att, &self.s3w, self.s3b, &mut ws.logits, nm, d)?;
        // ---- act head, off the [CLS] rows behind the markers ----
        exec.laya_act_head(
            &ws.logits,
            &ws.qoff,
            &ws.xg,
            nm,
            &self.a0w,
            &self.a0b,
            &self.a2w,
            &self.a2b,
            &mut ws.act,
            nq,
            d,
            ACT_HIDDEN,
            self.cfg.n_act,
        )?;

        let logits = exec.to_host_len(&ws.logits, nm)?;
        let act = exec.to_host_len(&ws.act, nq * self.cfg.n_act)?;
        if let Some(bad) = logits.iter().position(|v| !v.is_finite()) {
            return Err(GpuModelError::Unsupported(format!(
                "laya: non-finite logit at marker {bad} - an activation left f16's range"
            )));
        }
        Ok(LayaOut {
            logits,
            offsets: p.qoff.iter().map(|&o| o as usize).collect(),
            act,
            n_act: self.cfg.n_act,
            rows,
        })
    }
}
