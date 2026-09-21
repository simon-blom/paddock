use super::*;

// Layer yields, not token chunks, now bound Muse image-prefill pauses.
// Keeping the old 128-row text cap repeats a 30B weight walk for every tiny
// image chunk. Use the existing scratch/ring-slack capacity for the image
// at the FIFO head; leave text-only and Gemma elections unchanged. The
// caller's token budget still applies, and every live decode row is reserved.
fn image_phase_cap(muse: bool, image_head: bool, decodes: usize, capacity: usize) -> usize {
    if muse && image_head {
        capacity
    } else {
        crate::schedule::row_cap(decodes, capacity.min(CHUNK))
    }
}

// Give an idle, underfilled wide server one useful matrix tile before
// revisiting admission. A reused framing page is still a cold prompt body;
// offset==0 would miss the usual four-token Gemma header reuse. Continuations
// and substantial prefix hits retain the full grant. No timer or held logits.
fn admission_cap(cap: usize, decodes: usize, slots: usize, pending: &VecDeque<Pending>) -> usize {
    if decodes == 0
        && slots > 1
        && !pending.is_empty()
        && pending.len() < slots
        // Long cohorts already yield at a natural chunk boundary. Another
        // tiny first pass adds a whole model walk without preventing an
        // early completion, and regressed the long-prompt timing probe.
        && pending.iter().map(|p| p.work).sum::<usize>() <= cap
        && pending
            .iter()
            .all(|p| p.offset <= BLOCK_TOKENS && p.tokens.len() - p.offset == p.work)
    {
        cap.min(32)
    } else {
        cap
    }
}

fn draft_depth(live: usize, requested: usize) -> usize {
    // Metal's narrow verifier has a real cost slope, unlike a flat CUDA
    // matrix rung. Cap a single chain at three: four-row fused projections
    // earned the measured short-prompt win over eight-row verification.
    // Wide rounds keep their row reuse; low acceptance still shortens chains
    // in the shared service. This is a measured prior, not a hardware ceiling.
    requested.min(if live == 1 { 3 } else { spec::BLOCK - 1 })
}

impl Gemma4 {
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        let reused = self.prepare(slot, tokens)?;
        let mut last = Vec::new();
        for chunk in tokens[reused..].chunks(CHUNK) {
            let pos = self.slots[slot].history.len();
            let rows: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, (pos + i) as u32))
                .collect();
            let outputs = if pos + chunk.len() == tokens.len() {
                vec![chunk.len() - 1]
            } else {
                Vec::new()
            };
            last = self.execute(&rows, &outputs)?;
        }
        self.publish(slot)?;
        Ok(last)
    }
}
impl Generator for Gemma4 {
    fn reset(&mut self) {
        if let Some(d) = &mut self.dflash {
            d.budget = Default::default();
        }
        if let Some(v) = &mut self.spec {
            v.round = None;
        }
        if let Some(d) = &mut self.mtp {
            d.cursor.fill(None);
        }
        self.pending.clear();
        self.prefill_phase = None;
        self.encoding.clear();
        for s in &mut self.slots {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.reused = 0;
            s.mm = None;
        }
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn vision_budget(&self) -> Option<paddock_engine::generator::VisionBudget> {
        self.vision.as_ref().map(|_| {
            if self.muse {
                muse_vision::BUDGET
            } else {
                vision::BUDGET
            }
        })
    }
    fn supports_mm_slots(&self) -> bool {
        self.vision.is_some()
    }
    fn supports_chunked_multimodal(&self) -> bool {
        self.vision.is_some()
    }
    fn forward_multimodal(
        &mut self,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<Option<(Vec<f32>, usize)>, GenError> {
        Ok(Some(self.prefill_images(0, chunks)?))
    }
    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<(Vec<f32>, usize), GenError> {
        Ok(self.prefill_images(slot, chunks)?)
    }
    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<paddock_engine::service::MmChunk>)>,
    ) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        self.admit_images(items)
    }
    fn encode_step(&mut self) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        self.step_images()
    }
    fn encoding_pending(&self) -> bool {
        !self.encoding.is_empty()
    }
    fn spec_capable(&self) -> bool {
        self.mtp.is_some() || self.dflash.is_some()
    }
    fn spec_block_width(&self) -> Option<usize> {
        self.dflash.as_ref().map(|_| dflash::BLOCK)
    }
    fn spec_fixed_draft_depth(&self) -> Option<usize> {
        self.dflash.as_ref().map(|_| dflash::BLOCK - 1)
    }
    fn spec_batch_draft_budget(&self, live: usize) -> Option<usize> {
        self.dflash.as_ref().map(|d| d.budget.depth(live))
    }
    fn spec_deferred(&self) -> bool {
        // A 16-row Muse verify exceeds the encoder's 160 ms shared quantum.
        // Subtracting it leaves just one tower layer per scheduler round:
        // measured 23 s image TTFT for <1 s of actual tower GPU work. Keep
        // ordinary decoders moving during encoding, then resume full blocks.
        // Their dense forwards still append every DFlash conditioning tap.
        self.dflash.is_some() && !self.encoding.is_empty()
    }
    fn spec_k_miss_floor(&self) -> Option<usize> {
        self.dflash.as_ref().map(|_| dflash::BLOCK - 1)
    }
    fn spec_draft_kv_space(&self) -> bool {
        true
    }
    fn spec_live_cap(&self) -> usize {
        self.slots.len().min(CHUNK / self.verify_block())
    }
    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> std::result::Result<bool, GenError> {
        Ok((self.dflash.is_some()
            || self.mtp.as_ref().is_some_and(|d| {
                d.cursor.get(slot).copied().flatten() == Some(want_pos as usize + 1)
            }))
            && self
                .slots
                .get(slot)
                .is_some_and(|s| s.history.len() == want_pos as usize + 1)
            && !self.pending.iter().any(|p| p.slot == slot)
            && !self.image_slot_pending(slot))
    }
    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> std::result::Result<Option<Vec<Vec<u32>>>, GenError> {
        if self.dflash.is_some() {
            return Ok(self.dflash_draft(pendings, k)?);
        }
        Ok(self.mtp_draft(pendings, draft_depth(pendings.len(), k))?)
    }
    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> std::result::Result<Option<Vec<u32>>, GenError> {
        if !self.spec_capable()
            || reqs
                .iter()
                .any(|(_, p, c)| p.checked_add(c.len()).is_none_or(|n| n > self.context))
        {
            return Ok(None);
        }
        let picks = self.verify_picks(reqs)?;
        let mut counts = Vec::new();
        let mut base = 0;
        for (_, _, chunk) in reqs {
            counts.push(
                (1 + chunk[1..]
                    .iter()
                    .zip(&picks[base..])
                    .take_while(|(a, b)| a == b)
                    .count()) as u32,
            );
            base += chunk.len();
        }
        self.commit_verify(&counts)?;
        Ok(Some(picks))
    }
    fn forward_spec_verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> std::result::Result<Option<Vec<f32>>, GenError> {
        // The shared service applies the target sampler before committing.
        // Greedy proposals plus sample-and-match are lossless; canonical p/q
        // rejection sampling remains a separate acceptance/performance gate.
        if !self.spec_capable()
            || reqs
                .iter()
                .any(|(_, p, c)| p.checked_add(c.len()).is_none_or(|n| n > self.context))
        {
            return Ok(None);
        }
        Ok(Some(self.verify(reqs, false)?))
    }
    fn spec_commit(&mut self, committed: &[u32]) -> std::result::Result<(), GenError> {
        Ok(self.commit_verify(committed)?)
    }
    fn max_context(&self) -> usize {
        self.context
    }
    fn enable_batch(&mut self, max: usize) -> std::result::Result<usize, GenError> {
        Ok(max.min(self.slots.len()))
    }
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weight_bytes)
    }
    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.kv_bytes)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn forward(&mut self, t: u32) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.execute(&[(0, t, self.slots[0].history.len() as u32)], &[0])?)
    }
    fn forward_prefill_stream(&mut self, t: &[u32]) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(0, t)?)
    }
    fn forward_prefill(&mut self, s: usize, t: &[u32]) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(s, t)?)
    }
    fn forward_batch(&mut self, t: &[u32], p: &[u32]) -> std::result::Result<Vec<f32>, GenError> {
        if t.len() != p.len() || t.len() > self.slots.len() {
            return Err(GenError::Backend("Gemma batch shape mismatch".into()));
        }
        let rows: Vec<_> = t
            .iter()
            .zip(p)
            .enumerate()
            .filter(|(_, (_, p))| **p != 0)
            .map(|(s, (&t, &p))| (s, t, p))
            .collect();
        let mut out = vec![0.; t.len() * self.vocab];
        if !rows.is_empty() {
            let logits = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>())?;
            for (i, r) in rows.iter().enumerate() {
                out[r.0 * self.vocab..(r.0 + 1) * self.vocab]
                    .copy_from_slice(&logits[i * self.vocab..(i + 1) * self.vocab]);
            }
        }
        Ok(out)
    }
    fn take_prefill_reused(&mut self, s: usize) -> usize {
        self.slots
            .get_mut(s)
            .map_or(0, |s| std::mem::take(&mut s.reused))
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(self.pool.free_blocks())
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for i in 0..self.slots.len() {
            if !occupied.get(i).copied().unwrap_or(false)
                && !self.pending.iter().any(|p| p.slot == i)
                && !self.image_slot_pending(i)
            {
                // Completed prefill already published its checkpoint. Releasing a
                // request must not submit GPU work through an infallible trait hook.
                self.slots[i].table.clear(&mut self.pool);
                self.slots[i].history.clear();
                self.slots[i].mm = None;
                if let Some(d) = &mut self.mtp {
                    d.cursor[i] = None;
                }
            }
        }
    }
    fn supports_chunked_prefill(&self) -> bool {
        true
    }
    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> std::result::Result<(), GenError> {
        if self.pending.iter().any(|p| p.slot == slot) || self.image_slot_pending(slot) {
            return Err(GenError::Backend("Gemma slot already prefilling".into()));
        }
        let n = self.prepare(slot, &tokens)?;
        self.pending.push_back(Pending {
            slot,
            work: tokens.len() - n,
            tokens,
            offset: n,
        });
        Ok(())
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        // A service abort may arrive after verification. Discarding the
        // entire uncommitted transaction leaves every other slot's cursor
        // intact; speculative writes are masked by those old cursors.
        if let Some(v) = &mut self.spec {
            v.round = None;
        }
        // Restart the other members of a partially executed cohort at layer
        // zero. Their committed histories/grants have not moved, and their
        // private partial KV is overwritten before it can be attended to.
        if self
            .prefill_phase
            .as_ref()
            .is_some_and(|p| p.contains(slot))
        {
            self.prefill_phase = None;
        }
        self.pending.retain(|p| p.slot != slot);
        self.abort_images(slot);
        if let Some(s) = self.slots.get_mut(slot) {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.reused = 0;
            s.mm = None;
            if let Some(d) = &mut self.mtp {
                d.cursor[slot] = None;
            }
        }
        true
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> std::result::Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        if decodes.len() > self.slots.len() {
            return Err(GenError::Backend("Gemma decode width exceeds grant".into()));
        }
        let mut seen = vec![false; self.slots.len()];
        for &(s, _, _) in decodes {
            if s >= seen.len()
                || seen[s]
                || self.pending.iter().any(|p| p.slot == s)
                || self.image_slot_pending(s)
            {
                return Err(GenError::Backend(
                    "duplicate/prefilling Gemma decode slot".into(),
                ));
            }
            seen[s] = true;
        }
        if self.prefill_phase.is_some() {
            return self.advance_prefill_phase(decodes, budget);
        }
        let mut rows = decodes.to_vec();
        let mut complete = Vec::new();
        let cap = admission_cap(
            image_phase_cap(
                self.muse,
                self.pending
                    .front()
                    .is_some_and(|p| self.slots[p.slot].mm.is_some()),
                decodes.len(),
                self.scratch.rows,
            ),
            decodes.len(),
            self.slots.len(),
            &self.pending,
        );
        let mut grants = crate::schedule::grants(
            &self
                .pending
                .iter()
                .map(|p| (p.tokens.len() - p.offset, p.work))
                .collect::<Vec<_>>(),
            budget.min(cap.saturating_sub(rows.len())),
            decodes.is_empty(),
        );
        // An image is an indivisible bidirectional unit at each language
        // layer. Never manufacture a partial image's KV under a causal mask.
        // Reserve every decode row; one image can exceed the soft token
        // allowance, but never the fixed CHUNK scratch/ring-slack bound.
        if budget > 0 && !self.muse {
            let mut capacity = CHUNK - rows.len();
            for (p, n) in self.pending.iter().zip(&mut grants) {
                *n = if let Some(mm) = &self.slots[p.slot].mm {
                    mm.grant(p.offset, (*n).min(capacity), capacity)
                } else {
                    (*n).min(capacity)
                };
                capacity -= *n;
            }
            if grants.iter().all(|&n| n == 0)
                && let Some((p, n)) = self.pending.front().zip(grants.first_mut())
                && let Some(mm) = &self.slots[p.slot].mm
            {
                *n = mm.grant(p.offset, 1, CHUNK - rows.len());
            }
        }
        for (p, &n) in self.pending.iter().zip(&grants) {
            for i in p.offset..p.offset + n {
                rows.push((p.slot, p.tokens[i], i as u32));
            }
            if n > 0 && p.offset + n == p.tokens.len() {
                complete.push((p.slot, rows.len() - 1, p.tokens.len()));
            }
        }
        if rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let outputs: Vec<_> = (0..decodes.len())
            .chain(complete.iter().map(|p| p.1))
            .collect();
        if !decodes.is_empty()
            && rows.len() - decodes.len() >= 64
            // The text framing of an image request can also exceed a decode
            // quantum. Include it, while retaining the qualified text-only
            // scheduler and its projection shapes unchanged.
            && rows[decodes.len()..].iter().any(|r| self.slots[r.0].mm.is_some())
        {
            self.begin_prefill_phase(&rows[decodes.len()..], &complete, &grants, decodes.len())?;
            return self.advance_prefill_phase(decodes, budget);
        }
        let logits = self.execute(&rows, &outputs)?;
        for (p, n) in self.pending.iter_mut().zip(grants) {
            p.offset += n;
        }
        for &(s, _, _) in &complete {
            self.publish(s)?;
        }
        self.pending.retain(|p| p.offset < p.tokens.len());
        let done = complete
            .iter()
            .enumerate()
            .map(|(i, &(s, _, n))| {
                let r = decodes.len() + i;
                (s, logits[r * self.vocab..(r + 1) * self.vocab].to_vec(), n)
            })
            .collect();
        Ok((logits[..decodes.len() * self.vocab].to_vec(), done))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn muse_image_phase_expansion_preserves_other_routes_and_token_budgets() {
        assert_eq!(image_phase_cap(true, true, 3, 512), 512);
        for (muse, image) in [(false, false), (false, true), (true, false)] {
            assert_eq!(image_phase_cap(muse, image, 3, 512), 128);
        }
        assert_eq!(image_phase_cap(true, true, 0, 512), 512);
        assert_eq!(image_phase_cap(true, true, 511, 512), 512);
        assert_eq!(image_phase_cap(true, true, 3, muse::IMAGE_CHUNK), 2048);
        assert_eq!(image_phase_cap(true, true, 0, muse::IMAGE_CHUNK), 2048);
        assert_eq!(image_phase_cap(true, false, 0, muse::IMAGE_CHUNK), CHUNK);
        assert_eq!(image_phase_cap(true, false, 3, muse::IMAGE_CHUNK), 128);
        for budget in [0usize, 1, 31, 128, 512, 8192] {
            let cap = image_phase_cap(true, true, 3, 512);
            let grants =
                crate::schedule::grants(&[(2000, 2000), (30, 30)], budget.min(cap - 3), false);
            assert_eq!(grants, [budget.min(509), 0]);
            assert!(grants.iter().sum::<usize>() + 3 <= 512);
        }
    }

    #[test]
    fn first_quantum_includes_framing_reuse_but_not_resumed_work() {
        let pending = |offset, work| {
            VecDeque::from([Pending {
                slot: 0,
                tokens: vec![1; 120],
                offset,
                work,
            }])
        };
        for reused in [0, 4, BLOCK_TOKENS] {
            let p = pending(reused, 120 - reused);
            assert_eq!(admission_cap(512, 0, 4, &p), 32);
            assert_eq!(admission_cap(17, 0, 4, &p), 17);
            assert_eq!(admission_cap(512, 0, 1, &p), 512);
            assert_eq!(admission_cap(128, 1, 4, &p), 128);
        }
        assert_eq!(admission_cap(512, 0, 4, &pending(17, 103)), 512);
        assert_eq!(admission_cap(512, 0, 4, &pending(4, 120)), 512);
        let long = VecDeque::from([Pending {
            slot: 0,
            tokens: vec![1; 800],
            offset: 4,
            work: 796,
        }]);
        assert_eq!(admission_cap(512, 0, 4, &long), 512);
        assert_eq!(admission_cap(512, 0, 4, &VecDeque::new()), 512);
        let mut full = VecDeque::new();
        for slot in 0..4 {
            full.push_back(Pending {
                slot,
                tokens: vec![1; 120],
                offset: 0,
                work: 120,
            });
        }
        assert_eq!(admission_cap(512, 0, 4, &full), 512);
    }

    #[test]
    fn serving_depth_respects_requested_work_and_narrow_cost() {
        for requested in 0..16 {
            assert_eq!(draft_depth(1, requested), requested.min(3));
            for live in 2..=4 {
                assert_eq!(draft_depth(live, requested), requested.min(7));
            }
        }
    }
}
