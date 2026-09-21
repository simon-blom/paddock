//! Target verification is a transaction over paged global KV and SWA rings.
//! Rejected keys remain inaccessible above the committed position. The ring's
//! append slack preserves the entire old window, so rollback needs no replay
//! or per-token KV snapshots. Only accepted final hidden states seed MTP.
use super::*;

pub(super) const BLOCK: usize = 8;
pub(super) struct Verify {
    pub rows: usize,
    pub logits: Buffer,
    pub hidden: Buffer,
    pub picks: Buffer,
    pub round: Option<Vec<(usize, usize, Vec<u32>)>>,
}

impl Gemma4 {
    pub(super) fn verify_block(&self) -> usize {
        if self.muse { dflash::BLOCK } else { BLOCK }
    }
    pub(super) fn require_committed(&self) -> Result<()> {
        if self.spec.as_ref().is_some_and(|v| v.round.is_some()) {
            return Err(MetalError::Model(
                "Gemma verification requires commit before another forward".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn ensure_verify(&mut self) -> Result<()> {
        if self.spec.is_none() {
            let rows = (self.slots.len() * self.verify_block()).min(CHUNK);
            let verify = Verify {
                rows,
                logits: self.device.alloc(rows * self.vocab * 4)?,
                hidden: self.device.alloc(rows * self.width * 4)?,
                picks: self.device.alloc(rows * 4)?,
                round: None,
            };
            // Gemma's unit attention scale amplifies Q/probability rounding.
            // Verify uses the F32 grouped decode kernel for each causal row,
            // with a bounded split workspace, instead of the F16 MPP route.
            let hd = self
                .layers
                .iter()
                .map(Layer::hd)
                .max()
                .expect("nonempty graph");
            let parts = self.device.alloc(rows * HEADS * SPLITS * (hd + 2) * 4)?;
            self.scratch.parts = parts;
            self.spec = Some(verify);
        }
        Ok(())
    }

    pub(super) fn verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        greedy: bool,
    ) -> Result<Vec<f32>> {
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
                || chunk.len() > self.verify_block()
                || pos
                    .checked_add(chunk.len())
                    .is_none_or(|end| end > self.context)
                || self.pending.iter().any(|p| p.slot == *slot)
                || self.image_slot_pending(*slot)
            {
                return Err(MetalError::Model(
                    "invalid Gemma speculative slot/position/chunk".into(),
                ));
            }
            seen[*slot] = true;
            rows.extend(
                chunk
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| (*slot, t, (*pos + i) as u32)),
            );
        }
        if rows.is_empty() || rows.len() > self.spec.as_ref().expect("verify allocated").rows {
            return Err(MetalError::Model(
                "Gemma speculative row budget exceeded".into(),
            ));
        }
        self.verifying = true;
        self.greedy_verify = greedy;
        let out = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>());
        self.verifying = false;
        self.greedy_verify = false;
        let out = out?;
        self.spec.as_mut().expect("verify allocated").round = Some(reqs.to_vec());
        Ok(out)
    }

    pub(super) fn verify_picks(&mut self, reqs: &[(usize, usize, Vec<u32>)]) -> Result<Vec<u32>> {
        self.verify(reqs, true)?;
        Ok(unsafe {
            self.spec
                .as_ref()
                .expect("verify allocated")
                .picks
                .read_u32(reqs.iter().map(|r| r.2.len()).sum())
        })
    }

    pub(super) fn commit_verify(&mut self, counts: &[u32]) -> Result<()> {
        let v = self
            .spec
            .as_ref()
            .ok_or_else(|| MetalError::Model("no Gemma verification transaction".into()))?;
        let reqs = v
            .round
            .as_ref()
            .ok_or_else(|| MetalError::Model("no Gemma verification transaction".into()))?;
        if counts.len() != reqs.len()
            || counts
                .iter()
                .zip(reqs)
                .any(|(&n, r)| n == 0 || n as usize > r.2.len())
        {
            return Err(MetalError::Model(
                "invalid Gemma speculative commit counts".into(),
            ));
        }
        if let Some(d) = &self.mtp {
            let cmd = self.device.begin()?;
            let mut first = 0;
            for ((slot, _, chunk), &n) in reqs.iter().zip(counts) {
                copy_words(
                    &cmd,
                    &v.hidden,
                    &d.pending,
                    (first + n as usize - 1) * self.width,
                    slot * self.width,
                    self.width,
                );
                first += chunk.len();
            }
            cmd.finish()?;
        }
        let reqs = self
            .spec
            .as_mut()
            .expect("verify allocated")
            .round
            .take()
            .expect("open transaction");
        if let Some(d) = &mut self.dflash {
            d.budget.observe(reqs.iter().map(|r| r.2.len()), counts);
        }
        for ((slot, pos, chunk), &n) in reqs.into_iter().zip(counts) {
            self.slots[slot]
                .history
                .extend_from_slice(&chunk[..n as usize]);
            if let Some(d) = &mut self.mtp {
                d.cursor[slot] = Some(pos + n as usize);
            }
        }
        Ok(())
    }
}
