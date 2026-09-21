use super::*;
use crate::weights::projections;

impl GptOss {
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
        if m == 0
            || m > CHUNK
            || output_rows.len() > self.slots.len()
            || output_rows.iter().any(|&r| r >= m)
        {
            return Err(MetalError::Model(
                "invalid GPT-OSS execution/output rows".into(),
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
                    "invalid GPT-OSS row slot={slot} token={token} pos={pos}"
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
                    return Err(MetalError::Memory("GPT-OSS KV pool exhausted".into()));
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
        let depth = decodes
            .iter()
            .map(|&r| rows[r as usize].2 as usize + 1)
            .max()
            .unwrap_or(1);
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
            &[WIDTH as u32, m as u32, self.embedding.ty, 1f32.to_bits()],
            [(m * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        let norm = |w: &Weight| {
            cmd.dispatch(
                "rms",
                &[&s.x, &w.buffer, &s.norm],
                &[WIDTH as u32, w.ty, self.eps.to_bits()],
                [m, 1, 1],
                256,
            )
        };
        let grouped = m >= 16;
        // Larger cohorts amortize each expert tile across 32 rows. Keep the
        // narrower tile when average expert occupancy cannot fill it.
        let moe_tile = if m * 4 >= self.experts * 32 { 32 } else { 16 };
        let tile_cap = (m * 4).div_ceil(moe_tile) + self.experts;
        for (index, l) in self.layers.iter().enumerate() {
            norm(&l.norm);
            projections(
                &cmd,
                &[(&l.q, &s.q), (&l.k, &s.k), (&l.v, &s.v)],
                &s.norm,
                m,
                &s.gemm,
            );
            let mut rp = vec![
                QWIDTH as u32,
                KVWIDTH as u32,
                m as u32,
                self.page_stride as u32,
            ];
            rp.extend(self.rope);
            cmd.dispatch(
                "oss_rope_store",
                &[
                    &s.q,
                    &s.k,
                    &s.v,
                    &l.qb.buffer,
                    &l.kb.buffer,
                    &l.vb.buffer,
                    &l.keys,
                    &l.values,
                    &s.meta,
                    &s.pages,
                ],
                &rp,
                [(m * (QWIDTH + KVWIDTH) / 2).div_ceil(256), 1, 1],
                256,
            );
            let window = if index.is_multiple_of(2) { 128 } else { 0 };
            if !tiles.is_empty() {
                cmd.dispatch(
                    "linear_input",
                    &[&s.q, &s.qhalf],
                    &[QWIDTH as u32, 0, m as u32],
                    [(m * QWIDTH).div_ceil(256), 1, 1],
                    256,
                );
                cmd.dispatch(
                    "oss_attention_prefill",
                    &[
                        &s.qhalf,
                        &l.keys,
                        &l.values,
                        &s.meta,
                        &s.pages,
                        &s.attn,
                        &s.attention_tiles,
                        &l.sinks.buffer,
                    ],
                    &[64, 8, self.page_stride as u32, 0.125f32.to_bits(), window],
                    [64, tiles.len() / 2, 1],
                    128,
                );
            }
            if !decodes.is_empty() {
                let length = if window > 0 { depth.min(128) } else { depth };
                let splits = length
                    .div_ceil(128)
                    .max(8usize.div_ceil(decodes.len()))
                    .clamp(1, SPLITS);
                cmd.dispatch(
                    "oss_attention_decode",
                    &[
                        &s.q,
                        &l.keys,
                        &l.values,
                        &s.meta,
                        &s.pages,
                        &s.decode_rows,
                        &l.sinks.buffer,
                        if splits == 1 { &s.attn } else { &s.parts },
                    ],
                    &[
                        64,
                        8,
                        self.page_stride as u32,
                        0.125f32.to_bits(),
                        decodes.len() as u32,
                        splits as u32,
                        window,
                    ],
                    [8, decodes.len(), splits],
                    128,
                );
                if splits > 1 {
                    cmd.dispatch(
                        "attention_gqa_merge64",
                        &[&s.parts, &s.attn, &s.decode_rows],
                        &[splits as u32, 64],
                        [decodes.len() * 64, 1, 1],
                        32,
                    );
                }
            }
            l.o.linear(&cmd, &s.attn, &s.delta, m, 1., &s.gemm);
            cmd.dispatch(
                "oss_bias_residual",
                &[&s.x, &s.delta, &l.ob.buffer],
                &[WIDTH as u32, m as u32],
                [(m * WIDTH).div_ceil(256), 1, 1],
                256,
            );
            norm(&l.post);
            // Routing discontinuities amplify tiny rounding errors. Keep the
            // small F32 router in F32 at every row count; never silently cast
            // it through the dense W8A16 prefill election.
            cmd.dispatch(
                "linear",
                &[&l.router.buffer, &s.norm, &s.router],
                &[
                    WIDTH as u32,
                    self.experts as u32,
                    m as u32,
                    0,
                    1f32.to_bits(),
                ],
                [self.experts.div_ceil(4), m, 1],
                128,
            );
            cmd.dispatch(
                "moe_route",
                &[&s.router, &l.router_bias.buffer, &s.picks, &s.probabilities],
                &[self.experts as u32],
                [m, 1, 1],
                32,
            );
            if grouped {
                cmd.dispatch(
                    "moe_align",
                    &[&s.picks, &s.lists, &s.counts],
                    &[(m * 4) as u32],
                    [self.experts, 1, 1],
                    256,
                );
                cmd.dispatch(
                    "moe_tiles",
                    &[&s.counts, &s.tiles],
                    &[self.experts as u32, moe_tile as u32],
                    [1, 1, 1],
                    128,
                );
                cmd.dispatch(
                    if moe_tile == 32 {
                        "moe_gu_grouped32"
                    } else {
                        "moe_gu_grouped"
                    },
                    &[
                        &l.gate, &l.up, &s.norm, &s.lists, &s.counts, &s.tiles, &s.gu,
                    ],
                    &[WIDTH as u32, WIDTH as u32, m as u32],
                    [WIDTH.div_ceil(MOE_NTILE) * 2, tile_cap, 1],
                    128,
                );
            } else {
                cmd.dispatch(
                    "moe_gu_decode",
                    &[&l.gate, &l.up, &s.norm, &s.picks, &s.gu],
                    &[WIDTH as u32, WIDTH as u32, m as u32],
                    [WIDTH.div_ceil(4), m * 4, 1],
                    128,
                );
            }
            cmd.dispatch(
                "moe_swiglu",
                &[&s.gu, &l.gate_bias.buffer, &l.up_bias.buffer, &s.picks],
                &[WIDTH as u32, m as u32],
                [(m * 4 * WIDTH).div_ceil(256), 1, 1],
                256,
            );
            if grouped {
                cmd.dispatch(
                    if moe_tile == 32 {
                        "moe_down_grouped32"
                    } else {
                        "moe_down_grouped"
                    },
                    &[
                        &l.down,
                        &l.down,
                        &s.gu,
                        &s.lists,
                        &s.counts,
                        &s.tiles,
                        &s.expert_out,
                    ],
                    &[WIDTH as u32, WIDTH as u32, m as u32],
                    [WIDTH.div_ceil(MOE_NTILE), tile_cap, 1],
                    128,
                );
            } else {
                cmd.dispatch(
                    "moe_down_decode",
                    &[&l.down, &s.gu, &s.picks, &s.expert_out],
                    &[WIDTH as u32, WIDTH as u32, m as u32],
                    [WIDTH.div_ceil(4), m * 4, 1],
                    128,
                );
            }
            cmd.dispatch(
                "moe_fold",
                &[
                    &s.expert_out,
                    &l.down_bias.buffer,
                    &s.picks,
                    &s.probabilities,
                    &s.x,
                ],
                &[WIDTH as u32, m as u32],
                [(m * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        }
        if !output_rows.is_empty() {
            cmd.dispatch(
                "rms_selected",
                &[&s.x, &self.output_norm.buffer, &s.output_rows, &s.norm],
                &[WIDTH as u32, self.output_norm.ty, self.eps.to_bits()],
                [output_rows.len(), 1, 1],
                256,
            );
            self.head
                .linear(&cmd, &s.norm, &s.logits, output_rows.len(), 1., &s.gemm);
        }
        self.last_gpu_seconds = cmd.finish()?;
        for &(slot, token, _) in rows {
            self.slots[slot].history.push(token);
        }
        // SAFETY: completed command buffer exposes only the selected rows.
        Ok(unsafe { s.logits.read_f32(0, output_rows.len() * VOCAB) })
    }
}
