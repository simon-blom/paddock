//! Adapter only: CUDA math and scheduling elections stay unchanged.
use super::GpuWhisper;
use crate::whisper::{LangProb, StepOut, TimeScale, WhisperBackend};
impl WhisperBackend for GpuWhisper {
    fn prepare_batch(&mut self, cap: usize) -> Result<(), String> {
        GpuWhisper::prepare_batch(self, cap).map_err(|e| e.to_string())
    }
    fn time_scale(&self) -> TimeScale {
        GpuWhisper::time_scale(self)
    }
    fn languages(&self) -> Vec<String> {
        GpuWhisper::languages(self)
    }
    fn weights_bytes(&self) -> u64 {
        GpuWhisper::weights_bytes(self)
    }
    fn device_mem_used(&self) -> Option<u64> {
        GpuWhisper::device_mem_used(self)
    }
    fn contract_tokens(&self) -> (u32, u32) {
        GpuWhisper::contract_tokens(self)
    }
    fn prompt_tail(&self) -> (u32, u32) {
        GpuWhisper::prompt_tail(self)
    }
    fn sot_prev_token(&self) -> u32 {
        GpuWhisper::sot_prev_token(self)
    }
    fn text_ctx(&self) -> usize {
        GpuWhisper::text_ctx(self)
    }
    fn lang_token(&self, code: &str) -> Option<u32> {
        GpuWhisper::lang_token(self, code)
    }
    fn enc_batch_cap(&self) -> usize {
        GpuWhisper::enc_batch_cap(self)
    }
    fn encode_into_batch(
        &mut self,
        slots: &[usize],
        mels: &[&crate::audio::MelFeatures],
    ) -> Result<(), String> {
        GpuWhisper::encode_into_batch(self, slots, mels).map_err(|e| e.to_string())
    }
    fn enc_overlap(&self) -> bool {
        GpuWhisper::enc_overlap(self)
    }
    fn set_enc_inflight(&mut self, on: bool) {
        GpuWhisper::set_enc_inflight(self, on)
    }
    fn encode_sync(&mut self) -> Result<(), String> {
        GpuWhisper::encode_sync(self).map_err(|e| e.to_string())
    }
    fn step_batch(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        pos: &[u32],
        rules: Option<&[u32]>,
    ) -> Result<StepOut, String> {
        GpuWhisper::step_batch(self, slots, tokens, pos, rules).map_err(|e| e.to_string())
    }
    fn logits_row(&mut self, row: usize) -> Result<Vec<f32>, String> {
        GpuWhisper::logits_row(self, row).map_err(|e| e.to_string())
    }
    fn language_posterior(&self, logits: &[f32]) -> Result<Vec<LangProb>, String> {
        GpuWhisper::language_posterior(self, logits).map_err(|e| e.to_string())
    }
    fn token_boundaries(
        &mut self,
        slot: usize,
        lang: u32,
        tokens: &[u32],
        samples: usize,
    ) -> Result<Vec<f32>, String> {
        GpuWhisper::token_boundaries(self, slot, lang, tokens, samples).map_err(|e| e.to_string())
    }
    fn supports_word_times(&self) -> bool {
        true
    }
}
