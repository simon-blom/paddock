//! Backend-neutral Whisper contracts and transcript postprocessing.
//! GPU model arithmetic belongs to the CUDA/Metal implementations, not here.
pub mod align;
pub mod segments;
pub use segments::{Segment, split_segments, ts_state};
/// One language the checkpoint can name, with what the model thought of it at
/// the `<|startoftranscript|>` step.
///
/// `p` is renormalised over the language tokens only - see
/// `GpuWhisper::language_posterior` for why that is whisper's own rule and
/// what it costs (nothing here can say "no speech").
#[derive(Clone, Debug, PartialEq)]
pub struct LangProb {
    /// bare code as the checkpoint spells it ("sv", "jw", "yue")
    pub code: String,
    pub id: u32,
    pub p: f32,
}

/// The softmax over a checkpoint's language tokens, best first.
///
/// Free-standing so the arithmetic can be tested without a GPU - it is the
/// half of `GpuWhisper::language_posterior` that can actually be wrong (see
/// there for what the normalisation means and what it cannot say).
pub fn posterior_over(langs: &[(String, u32)], logits: &[f32]) -> Result<Vec<LangProb>, String> {
    if langs.is_empty() {
        return Err("whisper: checkpoint has no language map".into());
    }
    let mut out = Vec::with_capacity(langs.len());
    for (code, id) in langs {
        // A short row is a caller bug, not a language with no logit - refuse
        // rather than score the survivors against a truncated denominator.
        let l = *logits.get(*id as usize).ok_or_else(|| {
            format!(
                "whisper: language token {id} is past the {} logits given",
                logits.len()
            )
        })?;
        out.push(LangProb {
            code: code.clone(),
            id: *id,
            p: l,
        });
    }
    // log-sum-exp with the max pulled out: whisper's logits reach ±30 and a
    // bare exp() there loses the tail to zero
    let max = out.iter().fold(f32::NEG_INFINITY, |m, e| m.max(e.p));
    let mut z = 0.0f32;
    for e in &mut out {
        e.p = (e.p - max).exp();
        z += e.p;
    }
    if z > 0.0 {
        for e in &mut out {
            e.p /= z;
        }
    }
    // stable, so an exact tie keeps the checkpoint's own order
    out.sort_by(|a, b| b.p.total_cmp(&a.p));
    Ok(out)
}

/// Languages written without inter-word spaces - joining their windows with
/// a space would insert one where the script has none. Same set vLLM carries.
const NO_SPACE_LANGS: [&str; 2] = ["ja", "zh"];

/// Join a long-form clip's per-window transcripts into one transcript.
///
/// Whisper's encoder is fixed at 30 s, so anything longer decodes as
/// independent windows and the seam between them is ours to set. Whether the
/// model emits its usual leading space on a window's first token is not
/// reliable - the Danish gate caught Røst emitting none, which concatenated
/// straight into `...fødselsdagØerne`, two words fused into one. So each
/// window is trimmed and rejoined with exactly one space (none for scripts
/// that don't use them). That is also what vLLM does server-side, so
/// long-form transcripts agree with the reference regardless of what the
/// model did at the seam.
pub fn join_windows<S: AsRef<str>>(parts: &[S], language: &str) -> String {
    let sep = if NO_SPACE_LANGS.contains(&language) {
        ""
    } else {
        " "
    };
    let mut out = String::new();
    for p in parts {
        let p = p.as_ref().trim();
        if p.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push_str(sep);
        }
        out.push_str(p);
    }
    out
}

/// How whisper's timestamp tokens map to seconds, and how long a window is -
/// both derived from the checkpoint's own mel geometry rather than the usual
/// hardcoded 0.02/30.0, because a fine-tune that changed the hop would move
/// every timestamp we report.
#[derive(Clone, Copy, Debug)]
pub struct TimeScale {
    /// id of `<|0.00|>`; ids at or above it are timestamps
    pub begin: u32,
    /// seconds per timestamp step - hop/sr doubled for the conv stem's 2x
    /// time downsample (160/16000*2 = 0.02 s on every released whisper)
    pub precision: f32,
    /// one encoder window in seconds (30.0), the offset between windows
    pub window_s: f32,
}

impl TimeScale {
    /// Seconds for a timestamp token, offset into the clip by its window.
    /// Callers have already checked `is_timestamp`.
    pub fn seconds(&self, id: u32, window: usize) -> f32 {
        (id.saturating_sub(self.begin)) as f32 * self.precision + window as f32 * self.window_s
    }

    pub fn is_timestamp(&self, id: u32) -> bool {
        id >= self.begin
    }
}

#[derive(Default)]
pub struct StepOut {
    pub next: Vec<u32>,
    pub logprob: Vec<f32>,
    pub nospeech: Vec<f32>,
    pub runner_up: Vec<Option<(u32, f32)>>,
}

/// Timestamp grammar transport shared by the scheduler and GPU packs.
pub mod ts_flags {
    pub const ON: u32 = 1;
    pub const BEGIN: u32 = 2;
    pub const LAST: u32 = 4;
    pub const PENULT: u32 = 8;
    pub const HAVE: u32 = 16;
}

/// One device-owned encoder-decoder. Calls are serialized by the transcriber.
/// Slot caches never move when active rows compact; admission must validate
/// the entire wave before changing any live slot.
pub trait WhisperBackend: Send {
    fn prepare_batch(&mut self, cap: usize) -> Result<(), String>;
    fn time_scale(&self) -> TimeScale;
    fn languages(&self) -> Vec<String>;
    fn weights_bytes(&self) -> u64;
    fn device_mem_used(&self) -> Option<u64>;
    fn contract_tokens(&self) -> (u32, u32);
    fn prompt_tail(&self) -> (u32, u32);
    fn sot_prev_token(&self) -> u32;
    fn text_ctx(&self) -> usize;
    fn lang_token(&self, code: &str) -> Option<u32>;
    fn enc_batch_cap(&self) -> usize;
    fn encode_into_batch(
        &mut self,
        slots: &[usize],
        mels: &[&crate::audio::MelFeatures],
    ) -> Result<(), String>;
    fn enc_overlap(&self) -> bool {
        false
    }
    fn set_enc_inflight(&mut self, _on: bool) {}
    fn encode_sync(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn step_batch(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        pos: &[u32],
        rules: Option<&[u32]>,
    ) -> Result<StepOut, String>;
    fn logits_row(&mut self, row: usize) -> Result<Vec<f32>, String>;
    fn language_posterior(&self, logits: &[f32]) -> Result<Vec<LangProb>, String>;
    fn supports_word_times(&self) -> bool {
        false
    }
    fn token_boundaries(
        &mut self,
        _slot: usize,
        _lang: u32,
        _tokens: &[u32],
        _samples: usize,
    ) -> Result<Vec<f32>, String> {
        Err("word alignment is not implemented on this Whisper backend".into())
    }
}
