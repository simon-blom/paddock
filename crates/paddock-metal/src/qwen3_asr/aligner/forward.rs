use super::*;
impl Qwen3Aligner {
    pub(super) fn execute(
        &mut self,
        reqs: &[&AlignReq],
        canceled: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<std::result::Result<Vec<u32>, String>>> {
        if reqs.is_empty() || reqs.len() > 4 {
            return Err(error("alignment batch requires 1..4 requests"));
        }
        let mut rows = 0usize;
        for r in reqs {
            self.validate_request(r)?;
            rows = rows
                .checked_add(r.ids.len())
                .ok_or_else(|| error("row overflow"))?;
        }
        if rows > self.context {
            return Err(error("alignment packed rows exceed workspace capacity"));
        }
        let checkpoint = || {
            if (0..reqs.len()).all(canceled) {
                Err(error("alignment canceled"))
            } else {
                Ok(())
            }
        };
        checkpoint()?;
        let mut audio_job = self.tower.start_refs(
            &self.device,
            &reqs.iter().map(|r| &r.mel).collect::<Vec<_>>(),
        )?;
        let audio = loop {
            checkpoint()?;
            if let Some(out) = self.tower.step(&self.device, &mut audio_job)? {
                break out;
            }
        };
        drop(audio_job);
        checkpoint()?;
        let mut ids = Vec::with_capacity(rows);
        let mut meta = Vec::with_capacity(rows * 2);
        let mut tiles = Vec::new();
        let mut timestamps = Vec::new();
        let mut splices = Vec::new();
        for r in reqs {
            let first = ids.len();
            ids.extend_from_slice(&r.ids);
            meta.extend((0..r.ids.len()).flat_map(|i| [first as u32, i as u32]));
            for at in (0..r.ids.len()).step_by(32) {
                tiles.extend([
                    (first + at) as u32,
                    (r.ids.len() - at).min(32) as u32,
                    first as u32,
                    r.ids.len() as u32,
                ]);
            }
            timestamps.extend(r.ts_rows.iter().map(|&i| (first + i) as u32));
            splices.push((first + r.splice_at, r.n_audio));
        }
        let upload = |x: &[u32]| -> Result<Buffer> {
            let b = self.device.alloc(std::mem::size_of_val(x))?;
            // Fresh metadata allocation, no pending producer/consumer.
            unsafe {
                b.write_u32(x);
            }
            Ok(b)
        };
        let ids = upload(&ids)?;
        let meta = upload(&meta)?;
        let nt = tiles.len() / 4;
        let tiles = upload(&tiles)?;
        let s = &self.scratch;
        let c = self.device.begin()?;
        c.dispatch(
            "embed",
            &[&self.embedding.buffer, &ids, &s.x],
            &[WIDTH as u32, rows as u32, 30, 1f32.to_bits()],
            [(rows * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        for (a, (offset, n)) in audio.iter().zip(splices) {
            c.dispatch(
                "qalign_inject",
                &[a, &s.x],
                &[(n * WIDTH) as u32, (offset * WIDTH) as u32],
                [(n * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        }
        c.finish()?;
        drop(audio);
        #[cfg(test)]
        trace("text-0", &s.x, rows * WIDTH);
        // The index is consumed only by the test-only layer/operation trace.
        #[allow(clippy::unused_enumerate_index)]
        for (_index, l) in self.layers.iter().enumerate() {
            #[cfg(test)]
            reference_input(&format!("text-{_index}"), &s.x, rows * WIDTH);
            checkpoint()?;
            let c = self.device.begin()?;
            let norm = |c: &Commands<'_>, w: &Weight| {
                c.dispatch(
                    "qalign_rms",
                    &[&s.x, &w.buffer, &s.norm, &ids],
                    &[WIDTH as u32, 0],
                    [rows, 1, 1],
                    256,
                )
            };
            let residual = |c: &Commands<'_>| {
                c.dispatch(
                    "qalign_residual",
                    &[&s.x, &s.delta],
                    &[(rows * WIDTH) as u32, 1f32.to_bits()],
                    [(rows * WIDTH).div_ceil(256), 1, 1],
                    256,
                )
            };
            norm(&c, &l.norm);
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-norm", &s.norm, rows * WIDTH)?;
            for (w, out) in [(&l.q, &s.q), (&l.k, &s.k), (&l.v, &s.v)] {
                project(&c, w, &s.norm, out, rows);
            }
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-q", &s.q, rows * 2048)?;
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-k", &s.k, rows * WIDTH)?;
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-v", &s.v, rows * WIDTH)?;
            for (x, w, out, h, store) in [
                (&s.q, &l.qnorm, &s.qh, 16, 0),
                (&s.k, &l.knorm, &s.kh, 8, 1),
            ] {
                c.dispatch(
                    "qalign_head_rope",
                    &[x, &w.buffer, &meta, out, &s.v, &s.vh],
                    &[h, 0, 1e6f32.to_bits(), 1e-6f32.to_bits(), store],
                    [h as usize, rows, 1],
                    32,
                );
            }
            #[cfg(test)]
            let c = if _index == 27 && std::env::var_os("PADDOCK_QALIGN_TRACE").is_some() {
                c.finish()?;
                trace_bf16(&self.device, "op-qrope", &s.qh, rows * 2048)?;
                trace_bf16(&self.device, "op-krope", &s.kh, rows * 1024)?;
                self.device.begin()?
            } else {
                c
            };
            c.dispatch(
                "qalign_text_attention",
                &[&s.qh, &s.kh, &s.vh, &s.attn, &tiles],
                &[30],
                [16, nt, 1],
                64,
            );
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-attn", &s.attn, rows * 2048)?;
            project(&c, &l.o, &s.attn, &s.delta, rows);
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-o", &s.delta, rows * WIDTH)?;
            residual(&c);
            norm(&c, &l.post);
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-post", &s.norm, rows * WIDTH)?;
            project(&c, &l.gate, &s.norm, &s.gate, rows);
            project(&c, &l.up, &s.norm, &s.up, rows);
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-gate", &s.gate, rows * FF)?;
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-up", &s.up, rows * FF)?;
            c.dispatch(
                "qalign_swiglu",
                &[&s.gate, &s.up, &s.attn],
                &[(rows * FF) as u32],
                [(rows * FF).div_ceil(256), 1, 1],
                256,
            );
            project(&c, &l.down, &s.attn, &s.delta, rows);
            #[cfg(test)]
            let c = trace_stage(&self.device, c, _index, "op-down", &s.delta, rows * WIDTH)?;
            residual(&c);
            c.finish()?;
            #[cfg(test)]
            trace(&format!("text-{}", _index + 1), &s.x, rows * WIDTH);
        }
        let mut bins = Vec::with_capacity(timestamps.len());
        // Bound score scratch by 128 selected rows, not transcript length *
        // classes. Argmax validates every logit and resolves exact ties low.
        for ts in timestamps.chunks(HEAD_ROWS) {
            #[cfg(test)]
            reference_input("text-28", &s.x, rows * WIDTH);
            checkpoint()?;
            let selected = upload(ts)?;
            let out = self.device.alloc(ts.len() * 4)?;
            let c = self.device.begin()?;
            c.dispatch(
                "qalign_rms",
                &[&s.x, &self.final_norm.buffer, &s.selected, &selected],
                &[WIDTH as u32, 1],
                [ts.len(), 1, 1],
                256,
            );
            project(&c, &self.score, &s.selected, &s.logits, ts.len());
            c.dispatch(
                "qalign_argmax",
                &[&s.logits, &out],
                &[LABELS as u32],
                [ts.len(), 1, 1],
                256,
            );
            c.finish()?;
            // Only time-bin indices cross the completion fence, no host head.
            let b = unsafe { out.read_u32(ts.len()) };
            if b.iter().any(|&i| i as usize >= LABELS) {
                return Err(error("nonfinite timestamp logits"));
            }
            bins.extend(b);
        }
        let mut offset = 0;
        Ok(reqs
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let end = offset + r.ts_rows.len();
                let result = if canceled(i) {
                    Err("alignment canceled".into())
                } else {
                    Ok(bins[offset..end].to_vec())
                };
                offset = end;
                result
            })
            .collect())
    }
}
