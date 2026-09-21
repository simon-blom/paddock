use super::projection::project;
use super::*;

impl Laguna {
    pub(super) fn execute(
        &mut self,
        rows: &[(usize, u32, u32)],
        output_rows: &[usize],
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.execute_inner(rows, output_rows))
    }
    fn execute_inner(
        &mut self,
        rows: &[(usize, u32, u32)],
        output_rows: &[usize],
    ) -> Result<Vec<f32>> {
        let m = rows.len();
        let Geometry {
            width, ff, active, ..
        } = self.geometry;
        if m == 0
            || m > CHUNK
            || output_rows.len() > self.slots.len()
            || output_rows.iter().any(|&r| r >= m)
        {
            return Err(MetalError::Model(
                "invalid Laguna execution/output rows".into(),
            ));
        }
        let mut lengths = self
            .slots
            .iter()
            .map(|s| s.history.len())
            .collect::<Vec<_>>();
        for &(slot, token, pos) in rows {
            if slot >= self.slots.len()
                || token as usize >= VOCAB
                || pos as usize >= self.context
                || pos as usize != lengths[slot]
            {
                return Err(MetalError::Model(format!(
                    "invalid Laguna row slot={slot} token={token} pos={pos}"
                )));
            }
            lengths[slot] += 1;
        }
        for &(slot, _, pos) in rows {
            while self.slots[slot]
                .table
                .ensure(pos as usize, &mut self.pool)
                .is_err()
            {
                if self.radix.evict_lru(&mut self.pool).is_none() {
                    return Err(MetalError::Memory("Laguna KV pool exhausted".into()));
                }
            }
        }
        let mut pages = vec![0; self.slots.len() * self.page_stride];
        for (i, s) in self.slots.iter().enumerate() {
            pages[i * self.page_stride..i * self.page_stride + s.table.blocks().len()]
                .copy_from_slice(s.table.blocks());
        }
        let mut tiles = Vec::new();
        let mut decodes = Vec::new();
        let mut first = 0;
        while first < m {
            let mut end = first + 1;
            while end < m && rows[end].0 == rows[first].0 {
                end += 1;
            }
            if end - first >= 16 {
                for offset in (first..end).step_by(32) {
                    tiles.extend([offset as u32, (end - offset).min(32) as u32]);
                }
            } else {
                decodes.extend((first..end).map(|i| i as u32));
            }
            first = end;
        }
        let s = &self.scratch;
        // SAFETY: all graph submissions finish before this next write. Slots
        // and physical pages were fully validated above; no GPU count readback.
        unsafe {
            s.ids
                .write_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
            s.meta.write_u32(
                &rows
                    .iter()
                    .flat_map(|r| [r.0 as u32, r.2])
                    .collect::<Vec<_>>(),
            );
            s.pages.write_u32(&pages);
            s.output_rows
                .write_u32(&output_rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
            s.attention_tiles.write_u32(&tiles);
            s.decode_rows.write_u32(&decodes);
        }
        let cmd = self.device.begin()?;
        cmd.dispatch(
            "embed",
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[width as u32, m as u32, self.embedding.ty, 1f32.to_bits()],
            [(m * width).div_ceil(256), 1, 1],
            256,
        );
        let norm = |w: &Weight| {
            cmd.dispatch(
                "rms",
                &[&s.x, &w.buffer, &s.norm],
                &[width as u32, w.ty, self.eps.to_bits()],
                [m, 1, 1],
                256,
            )
        };
        let residual = || {
            cmd.dispatch(
                "residual",
                &[&s.x, &s.delta],
                &[(m * width) as u32, 1f32.to_bits()],
                [(m * width).div_ceil(256), 1, 1],
                256,
            )
        };
        let grouped = m >= 16;
        let tile = if m * active >= EXPERTS * 32 { 32 } else { 16 };
        let tile_cap = (m * active).div_ceil(tile) + EXPERTS;
        for (index, l) in self.layers.iter().enumerate() {
            norm(&l.norm);
            project(
                &cmd,
                &[(&l.q, &s.q), (&l.k, &s.k), (&l.v, &s.v)],
                &s.norm,
                m,
            );
            project(&cmd, &[(&l.gate, &s.gate)], &s.norm, m);
            let swa = !index.is_multiple_of(4);
            let rotary = if swa {
                [
                    10000f32.to_bits(),
                    1f32.to_bits(),
                    0,
                    1f32.to_bits(),
                    1f32.to_bits(),
                ]
            } else {
                self.rope
            };
            for (heads, input, w) in [(l.heads, &s.q, &l.qnorm), (8, &s.k, &l.knorm)] {
                let mut p = vec![heads as u32, if swa { 128 } else { 64 }];
                p.extend(rotary);
                p.push(self.eps.to_bits());
                cmd.dispatch(
                    "laguna_qnorm_rope",
                    &[input, &w.buffer, &s.meta],
                    &p,
                    [heads, m, 1],
                    32,
                );
            }
            cmd.dispatch(
                "laguna_store",
                &[&s.k, &s.v, &l.keys, &l.values, &s.meta, &s.pages],
                &[m as u32, self.page_stride as u32],
                [(m * KVWIDTH).div_ceil(256), 1, 1],
                256,
            );
            let ap = [
                l.heads as u32,
                8,
                self.page_stride as u32,
                if swa { 512 } else { 0 },
                0,
                SPLITS as u32,
            ];
            if !tiles.is_empty() {
                cmd.dispatch(
                    "laguna_prefill",
                    &[
                        &s.q,
                        &l.keys,
                        &l.values,
                        &s.meta,
                        &s.pages,
                        &s.attn,
                        &s.attention_tiles,
                    ],
                    &ap,
                    [l.heads, tiles.len() / 2, 1],
                    128,
                );
            }
            if !decodes.is_empty() {
                // Fixed per-query partitioning: batching or a neighbouring
                // long prompt must not change a request's split boundaries.
                cmd.dispatch(
                    match l.heads / 8 {
                        6 => "laguna_decode6",
                        8 => "laguna_decode8",
                        9 => "laguna_decode9",
                        _ => unreachable!("validated Laguna GQA"),
                    },
                    &[
                        &s.q,
                        &l.keys,
                        &l.values,
                        &s.meta,
                        &s.pages,
                        &s.decode_rows,
                        &s.parts,
                    ],
                    &ap,
                    [8, decodes.len(), SPLITS],
                    128,
                );
                cmd.dispatch(
                    "gemma_merge",
                    &[&s.parts, &s.attn, &s.decode_rows],
                    &[l.heads as u32, SPLITS as u32, 128],
                    [decodes.len() * l.heads, 1, 1],
                    32,
                );
            }
            cmd.dispatch(
                "laguna_gate",
                &[&s.attn, &s.gate],
                &[l.heads as u32, m as u32],
                [(m * l.heads * 128).div_ceil(256), 1, 1],
                256,
            );
            project(&cmd, &[(&l.o, &s.delta)], &s.attn, m);
            residual();
            norm(&l.post);
            project(&cmd, &[(&l.fg, &s.fg), (&l.fu, &s.fu)], &s.norm, m);
            cmd.dispatch(
                "swiglu",
                &[&s.fg, &s.fu],
                &[(m * l.fg.n) as u32],
                [(m * l.fg.n).div_ceil(256), 1, 1],
                256,
            );
            project(&cmd, &[(&l.fd, &s.delta)], &s.fg, m);
            if let Some(e) = &l.experts {
                // Router F32 is never rounded to half at the discontinuous
                // top-k boundary. Selection bias is not a mixture weight.
                cmd.dispatch(
                    "linear",
                    &[&e.router.buffer, &s.norm, &s.router],
                    &[width as u32, EXPERTS as u32, m as u32, 0, 1f32.to_bits()],
                    [EXPERTS.div_ceil(4), m, 1],
                    128,
                );
                cmd.dispatch(
                    expert_kernel(active, "laguna_route"),
                    &[&s.router, &e.bias.buffer, &s.picks, &s.probabilities],
                    &[active as u32],
                    [m, 1, 1],
                    32,
                );
                let gp = [
                    width as u32,
                    ff as u32,
                    m as u32,
                    e.gate.ty,
                    e.up.ty,
                    active as u32,
                ];
                let dp = [
                    ff as u32,
                    width as u32,
                    m as u32,
                    e.down.ty,
                    e.down.ty,
                    active as u32,
                ];
                if grouped {
                    cmd.dispatch(
                        "moe_align",
                        &[&s.picks, &s.lists, &s.counts],
                        &[(m * active) as u32],
                        [EXPERTS, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        "moe_tiles",
                        &[&s.counts, &s.tiles],
                        &[EXPERTS as u32, tile as u32],
                        [1, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        if tile == 32 {
                            expert_kernel(active, "laguna_gu_grouped32")
                        } else {
                            expert_kernel(active, "laguna_gu_grouped16")
                        },
                        &[
                            &e.gate.buffer,
                            &e.up.buffer,
                            &s.norm,
                            &s.lists,
                            &s.counts,
                            &s.tiles,
                            &s.gu,
                        ],
                        &gp,
                        [ff.div_ceil(32) * 2, tile_cap, 1],
                        128,
                    );
                } else {
                    cmd.dispatch(
                        expert_kernel(active, "laguna_gu_decode"),
                        &[&e.gate.buffer, &e.up.buffer, &s.norm, &s.picks, &s.gu],
                        &gp,
                        [ff.div_ceil(4), m * active, 1],
                        128,
                    );
                }
                cmd.dispatch(
                    "qmoe_swiglu",
                    &[&s.gu],
                    &[ff as u32, (m * active) as u32],
                    [(m * active * ff).div_ceil(256), 1, 1],
                    256,
                );
                if grouped {
                    cmd.dispatch(
                        if tile == 32 {
                            expert_kernel(active, "laguna_down_grouped32")
                        } else {
                            expert_kernel(active, "laguna_down_grouped16")
                        },
                        &[
                            &e.down.buffer,
                            &e.down.buffer,
                            &s.gu,
                            &s.lists,
                            &s.counts,
                            &s.tiles,
                            &s.expert_out,
                        ],
                        &dp,
                        [width.div_ceil(32), tile_cap, 1],
                        128,
                    );
                } else {
                    cmd.dispatch(
                        "laguna_down_decode",
                        &[&e.down.buffer, &s.gu, &s.picks, &s.expert_out],
                        &dp,
                        [width.div_ceil(4), m * active, 1],
                        128,
                    );
                }
                cmd.dispatch(
                    expert_kernel(active, "laguna_fold"),
                    &[&s.expert_out, &s.probabilities, &s.delta],
                    &[width as u32, m as u32, active as u32],
                    [(m * width).div_ceil(256), 1, 1],
                    256,
                );
            }
            residual();
        }
        if !output_rows.is_empty() {
            cmd.dispatch(
                "rms_selected",
                &[&s.x, &self.output_norm.buffer, &s.output_rows, &s.norm],
                &[width as u32, self.output_norm.ty, self.eps.to_bits()],
                [output_rows.len(), 1, 1],
                256,
            );
            project(&cmd, &[(&self.head, &s.logits)], &s.norm, output_rows.len());
        }
        self.last_gpu_seconds = cmd.finish()?;
        for &(slot, token, _) in rows {
            self.slots[slot].history.push(token);
        }
        // SAFETY: this command buffer completed before reading selected rows.
        Ok(unsafe { s.logits.read_f32(0, output_rows.len() * VOCAB) })
    }
}
