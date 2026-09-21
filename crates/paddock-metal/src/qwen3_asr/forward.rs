use super::*;
impl Qwen3Asr {
    pub(super) fn execute(
        &mut self,
        rows: &[(usize, u32, u32)],
        outputs: &[usize],
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.execute_inner(rows, outputs))
    }
    fn execute_inner(&mut self, rows: &[(usize, u32, u32)], outputs: &[usize]) -> Result<Vec<f32>> {
        let m = rows.len();
        if m == 0
            || m > CHUNK
            || outputs.len() > self.slots.len()
            || outputs.iter().any(|&r| r >= m)
        {
            return Err(error("invalid execution rows"));
        }
        let mut lengths = self
            .slots
            .iter()
            .map(|s| s.history.len())
            .collect::<Vec<_>>();
        for &(slot, token, pos) in rows {
            if slot >= lengths.len()
                || token as usize >= VOCAB
                || pos as usize >= self.context
                || pos as usize != lengths[slot]
            {
                return Err(error(format!(
                    "invalid row slot={slot} token={token} pos={pos}"
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
                    return Err(MetalError::Memory("Qwen3-ASR KV pool exhausted".into()));
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
                for at in (first..end).step_by(32) {
                    tiles.extend([at as u32, (end - at).min(32) as u32]);
                }
            } else {
                decodes.extend((first..end).map(|i| i as u32));
            }
            first = end;
        }
        let s = &self.scratch;
        // SAFETY: previous submissions completed, all slots and lengths were
        // validated before modifying the shared descriptor slabs.
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
                .write_u32(&outputs.iter().map(|&r| r as u32).collect::<Vec<_>>());
            s.decode_rows.write_u32(&decodes);
            s.tiles.write_u32(&tiles);
        }
        let cmd = self.device.begin()?;
        cmd.dispatch(
            "embed",
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[WIDTH as u32, m as u32, 8, 1f32.to_bits()],
            [(m * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        for (slot, seq) in self.slots.iter().enumerate() {
            if let Some(layout) = &seq.mm {
                for audio in &layout.audio {
                    if rows.iter().any(|r| {
                        r.0 == slot
                            && (r.2 as usize) >= audio.offset
                            && (r.2 as usize) < audio.offset + audio.tokens
                    }) {
                        cmd.dispatch(
                            "vis_inject",
                            &[&audio.embd, &s.meta, &s.x],
                            &[
                                WIDTH as u32,
                                m as u32,
                                slot as u32,
                                audio.offset as u32,
                                audio.tokens as u32,
                            ],
                            [(m * WIDTH).div_ceil(256), 1, 1],
                            256,
                        );
                    }
                }
            }
        }
        let norm = |w: &Weight| {
            cmd.dispatch(
                "rms",
                &[&s.x, &w.buffer, &s.norm],
                &[WIDTH as u32, w.ty, 1e-6f32.to_bits()],
                [m, 1, 1],
                256,
            )
        };
        let residual = || {
            cmd.dispatch(
                "residual",
                &[&s.x, &s.delta],
                &[(m * WIDTH) as u32, 1f32.to_bits()],
                [(m * WIDTH).div_ceil(256), 1, 1],
                256,
            )
        };
        for l in &self.layers {
            norm(&l.norm);
            for (w, y) in [(&l.q, &s.q), (&l.k, &s.k), (&l.v, &s.v)] {
                project(&cmd, w, &s.norm, y, m);
            }
            for (x, w, heads) in [(&s.q, &l.qnorm, 16), (&s.k, &l.knorm, 8)] {
                cmd.dispatch(
                    "laguna_qnorm_rope",
                    &[x, &w.buffer, &s.meta],
                    &[
                        heads,
                        128,
                        1000000f32.to_bits(),
                        1f32.to_bits(),
                        0,
                        1f32.to_bits(),
                        1f32.to_bits(),
                        1e-6f32.to_bits(),
                    ],
                    [heads as usize, m, 1],
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
            let ap = [16, 8, self.page_stride as u32, 0, 0, SPLITS as u32];
            if !tiles.is_empty() {
                cmd.dispatch(
                    "laguna_prefill",
                    &[
                        &s.q, &l.keys, &l.values, &s.meta, &s.pages, &s.attn, &s.tiles,
                    ],
                    &ap,
                    [16, tiles.len() / 2, 1],
                    128,
                );
            }
            if !decodes.is_empty() {
                cmd.dispatch(
                    "qasr_decode",
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
                    &[16, SPLITS as u32, 128],
                    [decodes.len() * 16, 1, 1],
                    32,
                );
            }
            project(&cmd, &l.o, &s.attn, &s.delta, m);
            residual();
            norm(&l.post);
            project(&cmd, &l.gate, &s.norm, &s.gate, m);
            project(&cmd, &l.up, &s.norm, &s.up, m);
            cmd.dispatch(
                "swiglu",
                &[&s.gate, &s.up],
                &[(m * FF) as u32],
                [(m * FF).div_ceil(256), 1, 1],
                256,
            );
            project(&cmd, &l.down, &s.gate, &s.delta, m);
            residual();
        }
        if !outputs.is_empty() {
            cmd.dispatch(
                "rms_selected",
                &[&s.x, &self.output_norm.buffer, &s.output_rows, &s.norm],
                &[WIDTH as u32, 0, 1e-6f32.to_bits()],
                [outputs.len(), 1, 1],
                256,
            );
            project(&cmd, &self.head, &s.norm, &s.logits, outputs.len());
        }
        self.last_gpu_seconds = cmd.finish()?;
        for &(slot, token, _) in rows {
            self.slots[slot].history.push(token);
        }
        // SAFETY: output producer has finished; no mutable mapping escapes.
        let logits = unsafe { s.logits.read_f32(0, outputs.len() * VOCAB) };
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite output logits; request cannot be sampled"));
        }
        Ok(logits)
    }
}
