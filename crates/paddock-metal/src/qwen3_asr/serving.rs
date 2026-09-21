use super::*;
use paddock_engine::generator::{GenError, Generator};
impl Generator for Qwen3Asr {
    fn supports_mm_slots(&self) -> bool {
        self.audio.is_some()
    }
    fn supports_chunked_multimodal(&self) -> bool {
        self.audio.is_some()
    }
    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<paddock_engine::service::MmChunk>)>,
    ) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        self.admit_audio(items)
    }
    fn encode_step(&mut self) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        self.encode_audio()
    }
    fn encoding_pending(&self) -> bool {
        !self.encoding.is_empty()
    }
    fn forward_multimodal(
        &mut self,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<Option<(Vec<f32>, usize)>, GenError> {
        self.prefill_audio(0, chunks.to_vec()).map(Some)
    }
    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<(Vec<f32>, usize), GenError> {
        self.prefill_audio(slot, chunks.to_vec())
    }
    fn reset(&mut self) {
        self.pending.clear();
        self.encoding.clear();
        for s in &mut self.slots {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.reused = 0;
            s.mm = None;
        }
    }
    fn vocab(&self) -> usize {
        VOCAB
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
    fn forward(&mut self, token: u32) -> std::result::Result<Vec<f32>, GenError> {
        if self.pending.iter().any(|p| p.slot == 0) || self.encoding.iter().any(|p| p.owns(0)) {
            return Err(GenError::Backend("slot already prefilling".into()));
        }
        Ok(self.execute(&[(0, token, self.slots[0].history.len() as u32)], &[0])?)
    }
    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(0, tokens)?)
    }
    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(slot, tokens)?)
    }
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        if tokens.len() != positions.len() || tokens.len() > self.slots.len() {
            return Err(GenError::Backend("Qwen3Asr batch shape mismatch".into()));
        }
        let rows = tokens
            .iter()
            .zip(positions)
            .enumerate()
            .filter(|(_, (_, p))| **p != 0)
            .map(|(i, (&t, &p))| (i, t, p))
            .collect::<Vec<_>>();
        let mut out = vec![0.; tokens.len() * VOCAB];
        if !rows.is_empty() {
            let logits = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>())?;
            for (i, r) in rows.iter().enumerate() {
                out[r.0 * VOCAB..(r.0 + 1) * VOCAB]
                    .copy_from_slice(&logits[i * VOCAB..(i + 1) * VOCAB]);
            }
        }
        Ok(out)
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        std::mem::take(&mut self.slots[slot].reused)
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(self.pool.free_blocks() + self.radix.evictable_blocks(&self.pool))
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for i in 0..self.slots.len() {
            if !occupied.get(i).copied().unwrap_or(false)
                && !self.pending.iter().any(|p| p.slot == i)
                && !self.encoding.iter().any(|p| p.owns(i))
            {
                if !self.slots[i].history.is_empty() {
                    self.publish(i);
                }
                self.slots[i].table.clear(&mut self.pool);
                self.slots[i].history.clear();
                self.slots[i].reused = 0;
                self.slots[i].mm = None;
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
        if self.pending.iter().any(|p| p.slot == slot) || self.encoding.iter().any(|p| p.owns(slot))
        {
            return Err(GenError::Backend("slot already prefilling".into()));
        }
        let reused = self.prepare(slot, &tokens)?;
        self.pending.push_back(Pending {
            slot,
            work: tokens.len() - reused,
            tokens,
            offset: reused,
        });
        Ok(())
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.pending.retain(|p| p.slot != slot);
        for e in &mut self.encoding {
            e.abort(slot);
        }
        self.encoding.retain(|p| !p.empty());
        if let Some(s) = self.slots.get_mut(slot) {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.reused = 0;
            s.mm = None;
        }
        true
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> std::result::Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        if decodes.len() > self.slots.len()
            || decodes
                .iter()
                .enumerate()
                .any(|(i, r)| decodes[..i].iter().any(|s| s.0 == r.0))
            || decodes.iter().any(|r| {
                self.pending.iter().any(|p| p.slot == r.0)
                    || self.encoding.iter().any(|p| p.owns(r.0))
            })
        {
            return Err(GenError::Backend(
                "invalid Qwen3Asr mixed decode slots".into(),
            ));
        }
        let mut rows = decodes.to_vec();
        let mut complete = Vec::new();
        let cap = crate::schedule::row_cap(decodes.len(), CHUNK);
        let advances = crate::schedule::grants(
            &self
                .pending
                .iter()
                .map(|p| (p.tokens.len() - p.offset, p.work))
                .collect::<Vec<_>>(),
            budget.min(cap.saturating_sub(rows.len())),
            decodes.is_empty(),
        );
        for (pending, &n) in self.pending.iter().zip(&advances) {
            for i in pending.offset..pending.offset + n {
                rows.push((pending.slot, pending.tokens[i], i as u32));
            }
            if n > 0 && pending.offset + n == pending.tokens.len() {
                complete.push((pending.slot, rows.len() - 1, pending.tokens.len()));
            }
        }
        if rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let outputs = (0..decodes.len())
            .chain(complete.iter().map(|r| r.1))
            .collect::<Vec<_>>();
        let logits = self.execute(&rows, &outputs)?;
        for (p, n) in self.pending.iter_mut().zip(advances) {
            p.offset += n;
        }
        let done = complete
            .iter()
            .enumerate()
            .map(|(i, &(slot, _, n))| {
                let r = decodes.len() + i;
                (slot, logits[r * VOCAB..(r + 1) * VOCAB].to_vec(), n)
            })
            .collect();
        for &(slot, _, _) in &complete {
            self.publish(slot);
        }
        self.pending.retain(|p| p.offset < p.tokens.len());
        Ok((logits[..decodes.len() * VOCAB].to_vec(), done))
    }
}
