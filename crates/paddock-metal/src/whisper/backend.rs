use super::*;
impl WhisperBackend for Whisper {
    fn prepare_batch(&mut self, cap: usize) -> std::result::Result<(), String> {
        self.prepare(cap).map_err(|e| e.to_string())
    }
    fn time_scale(&self) -> TimeScale {
        TimeScale {
            begin: 50365,
            precision: 0.02,
            window_s: 30.,
        }
    }
    fn languages(&self) -> Vec<String> {
        self.langs.iter().map(|(c, _)| c.clone()).collect()
    }
    fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn contract_tokens(&self) -> (u32, u32) {
        (50258, 50257)
    }
    fn prompt_tail(&self) -> (u32, u32) {
        (50360, 50364)
    }
    fn sot_prev_token(&self) -> u32 {
        50362
    }
    fn text_ctx(&self) -> usize {
        self.ctx
    }
    fn lang_token(&self, code: &str) -> Option<u32> {
        self.langs.iter().find(|(c, _)| c == code).map(|(_, i)| *i)
    }
    fn enc_batch_cap(&self) -> usize {
        self.capacity.min(ENC_BATCH)
    }
    fn encode_into_batch(
        &mut self,
        slots: &[usize],
        mels: &[&MelFeatures],
    ) -> std::result::Result<(), String> {
        self.encode_wave(slots, mels).map_err(|e| e.to_string())
    }
    fn step_batch(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        pos: &[u32],
        rules: Option<&[u32]>,
    ) -> std::result::Result<StepOut, String> {
        self.step(slots, tokens, pos, rules)
            .map_err(|e| e.to_string())
    }
    fn logits_row(&mut self, row: usize) -> std::result::Result<Vec<f32>, String> {
        if row >= self.last_rows {
            return Err("Whisper: logits row is not live".into());
        }
        // SAFETY: step() completed its command buffer and bounded the row.
        Ok(unsafe {
            self.scratch
                .as_ref()
                .expect("successful step owns scratch")
                .logits
                .read_f32(row * V, V)
        })
    }
    fn language_posterior(&self, logits: &[f32]) -> std::result::Result<Vec<LangProb>, String> {
        paddock_engine::whisper::posterior_over(&self.langs, logits)
    }
    fn supports_word_times(&self) -> bool {
        true
    }
    fn token_boundaries(
        &mut self,
        slot: usize,
        lang: u32,
        tokens: &[u32],
        samples: usize,
    ) -> std::result::Result<Vec<f32>, String> {
        self.align_tokens(slot, lang, tokens, samples)
            .map_err(|e| e.to_string())
    }
}
