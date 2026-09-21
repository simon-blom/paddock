//! Transactional target verification shared by MTP and block drafters.
//! Recurrent rollback stores rank-one updates (not per-token states), while
//! paged attention masks speculative KV by the committed position.
use super::*;

pub(super) const BLOCK: usize = 8;
pub(super) struct Verify {
    pub rows: usize,
    pub live: usize,
    pub updates: Buffer,
    pub conv_input: Buffer,
    pub logits: Buffer,
    pub hidden: Buffer,
    pub picks: Buffer,
    pub round: Option<Vec<(usize, usize, Vec<u32>)>>,
}

impl Qwen35 {
    /// A zero-proposal round is ordinary decode, not a rollback transaction.
    /// Retain device argmax without copying logits to the CPU or writing the
    /// recurrent verification journal. Decode itself commits state/history.
    pub(super) fn decode_picks(&mut self, reqs: &[(usize, usize, Vec<u32>)]) -> Result<Vec<u32>> {
        self.require_committed()?;
        let mut seen = vec![false; self.slots.len()];
        let mut rows = Vec::with_capacity(reqs.len());
        for (slot, pos, chunk) in reqs {
            if *slot >= seen.len()
                || seen[*slot]
                || *pos == 0
                || *pos >= self.context
                || chunk.len() != 1
                || *pos != self.slots[*slot].history.len()
                || self.pending.iter().any(|p| p.slot == *slot)
            {
                return Err(MetalError::Model("invalid direct greedy decode row".into()));
            }
            seen[*slot] = true;
            rows.push((*slot, chunk[0], *pos as u32));
        }
        if rows.is_empty() {
            return Err(MetalError::Model("empty direct greedy decode".into()));
        }
        self.ensure_verify()?;
        self.lookup.cancel();
        self.greedy_output = true;
        let result = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>());
        self.greedy_output = false;
        result?;
        // SAFETY: execute waited for both normal decode and device argmax.
        Ok(unsafe {
            self.spec
                .as_ref()
                .expect("ensure_verify initialized the argmax buffers")
                .picks
                .read_u32(rows.len())
        })
    }

    pub(super) fn require_committed(&self) -> Result<()> {
        if self.spec.as_ref().is_some_and(|s| s.round.is_some()) {
            return Err(MetalError::Model(
                "speculative verification requires commit before another forward".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn ensure_verify(&mut self) -> Result<()> {
        let g = self.geometry;
        if self.spec.is_none() {
            let rows = (self.slots.len() * BLOCK).min(CHUNK);
            self.spec = Some(Verify {
                rows,
                live: 0,
                updates: self
                    .device
                    .alloc(g.linear_layers() * rows * g.value_heads * 257 * 4)?,
                conv_input: self.device.alloc(g.linear_layers() * rows * g.conv() * 4)?,
                logits: self.device.alloc(rows * self.vocab * 4)?,
                hidden: self.device.alloc(rows * self.width * 4)?,
                picks: self.device.alloc(rows * 4)?,
                round: None,
            });
        }
        Ok(())
    }

    pub(super) fn verify(&mut self, reqs: &[(usize, usize, Vec<u32>)]) -> Result<Vec<f32>> {
        self.verify_inner(reqs, false)
    }
    pub(super) fn verify_picks(&mut self, reqs: &[(usize, usize, Vec<u32>)]) -> Result<Vec<u32>> {
        self.verify_inner(reqs, true)?;
        // SAFETY: verification completed, including GPU argmax.
        Ok(unsafe {
            self.spec
                .as_ref()
                .expect("verification buffers allocated")
                .picks
                .read_u32(reqs.iter().map(|r| r.2.len()).sum())
        })
    }
    fn verify_inner(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        greedy: bool,
    ) -> Result<Vec<f32>> {
        if self.bonsai.is_some() || self.ternary.is_some() {
            return Err(MetalError::Model(
                "Bonsai speculative verification needs a validated rollback contract for this checkpoint".into(),
            ));
        }
        self.require_committed()?;
        self.ensure_verify()?;
        let mut seen = vec![false; self.slots.len()];
        let mut rows = Vec::new();
        for (slot, pos, chunk) in reqs {
            if *slot >= seen.len()
                || seen[*slot]
                || *pos == 0
                || *pos != self.slots[*slot].history.len()
                || chunk.is_empty()
                || chunk.len() > BLOCK
                || self.pending.iter().any(|p| p.slot == *slot)
            {
                return Err(MetalError::Model(
                    "invalid speculative request/slot/position".into(),
                ));
            }
            seen[*slot] = true;
            rows.extend(
                chunk
                    .iter()
                    .enumerate()
                    .map(|(j, &t)| (*slot, t, (*pos + j) as u32)),
            );
        }
        if rows.is_empty()
            || rows.len()
                > self
                    .spec
                    .as_ref()
                    .expect("verification buffers allocated")
                    .rows
        {
            return Err(MetalError::Model("speculative row budget exceeded".into()));
        }
        self.spec
            .as_mut()
            .expect("verification buffers allocated")
            .live = reqs.len();
        self.verifying = true;
        self.greedy_output = greedy;
        let out = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>());
        self.verifying = false;
        self.greedy_output = false;
        let out = out?;
        self.spec
            .as_mut()
            .expect("verification buffers allocated")
            .round = Some(reqs.to_vec());
        Ok(out)
    }

    pub(super) fn commit_verify(&mut self, committed: &[u32]) -> Result<()> {
        let g = self.geometry;
        let v = self
            .spec
            .as_ref()
            .ok_or_else(|| MetalError::Model("no verification transaction".into()))?;
        let reqs = v
            .round
            .as_ref()
            .ok_or_else(|| MetalError::Model("no verification transaction".into()))?;
        if committed.len() != reqs.len()
            || committed
                .iter()
                .zip(reqs)
                .any(|(&n, r)| n == 0 || n as usize > r.2.len())
        {
            return Err(MetalError::Model(
                "invalid speculative commit counts".into(),
            ));
        }
        let mut spans = Vec::new();
        let mut base = 0;
        for ((slot, _, chunk), &n) in reqs.iter().zip(committed) {
            spans.extend([base as u32, n, *slot as u32, 0]);
            base += chunk.len();
        }
        // The verify upload left row metadata intact; acceptance changes only
        // span lengths. Zero checkpoint destinations forbid speculative prefix
        // publication. All rejection positions are tested, including page edges.
        unsafe {
            self.scratch.spans.write_u32(&spans);
            self.scratch
                .checkpoint_spans
                .write_u32(&vec![0; reqs.len() * 4]);
        }
        let cmd = self.device.begin()?;
        cmd.dispatch(
            if self.mlx {
                "mlx_dn_verify_commit"
            } else {
                "dn_verify_commit"
            },
            &[&v.updates, &self.state, &self.scratch.spans],
            &[g.value_heads as u32, self.state_slots as u32, v.rows as u32],
            [
                if self.mlx { 32 } else { 8 },
                g.linear_layers() * g.value_heads,
                reqs.len(),
            ],
            128,
        );
        cmd.dispatch(
            "dn_verify_conv_commit",
            &[&v.conv_input, &self.conv, &self.scratch.spans],
            &[g.conv() as u32, self.state_slots as u32, v.rows as u32],
            [g.conv().div_ceil(256), g.linear_layers(), reqs.len()],
            256,
        );
        cmd.finish()?;
        if self.mtp.is_some() {
            // Pack only accepted hidden/token rows before draft-KV catch-up.
            // Rejected hidden states never reach the persistent MTP cursor.
            let mut ids = Vec::new();
            let mut meta = Vec::new();
            let mut bounds = Vec::new();
            let mut gather = Vec::new();
            let mut base = 0;
            for ((slot, pos, chunk), &n) in reqs.iter().zip(committed) {
                let first = ids.len();
                let end = first + n as usize;
                for (j, &t) in chunk[..n as usize].iter().enumerate() {
                    ids.push(t);
                    meta.extend([*slot as u32, (*pos + j) as u32]);
                    bounds.extend([first as u32, end as u32]);
                    gather.push((base + j) as u32);
                }
                base += chunk.len();
            }
            let m = ids.len();
            let s = &self.scratch;
            unsafe {
                s.ids.write_u32(&ids);
                s.meta.write_u32(&meta);
                s.mrope.write_u32(
                    &meta
                        .chunks_exact(2)
                        .flat_map(|r| self.rope_position(r[0] as usize, r[1] as usize))
                        .collect::<Vec<_>>(),
                );
                s.limits
                    .write_u32(&meta.chunks_exact(2).map(|r| r[1]).collect::<Vec<_>>());
                s.bounds.write_u32(&bounds);
                s.outputs.write_u32(&gather);
                s.decode_rows.write_u32(&(0..m as u32).collect::<Vec<_>>());
                s.checkpoint_rows.write_u32(&vec![0; m]);
            }
            let cmd = self.device.begin()?;
            cmd.dispatch(
                "mtp_hidden",
                &[&v.hidden, &v.hidden, &s.meta, &s.outputs, &s.x],
                &[self.width as u32, m as u32, 2],
                [(m * self.width).div_ceil(256), 1, 1],
                256,
            );
            self.mtp_catchup(
                &cmd,
                &s.x,
                m,
                0,
                m,
                meta.chunks(2)
                    .map(|r| r[1] as usize + 1)
                    .max()
                    .expect("nonempty accepted cohort"),
            );
            cmd.finish()?;
        }
        let reqs = self
            .spec
            .as_mut()
            .expect("verification buffers allocated")
            .round
            .take()
            .expect("open verification transaction");
        for ((slot, _, chunk), &n) in reqs.into_iter().zip(committed) {
            self.slots[slot]
                .history
                .extend_from_slice(&chunk[..n as usize]);
        }
        self.lookup
            .commit(committed.iter().map(|&n| n as usize).sum());
        Ok(())
    }
}
