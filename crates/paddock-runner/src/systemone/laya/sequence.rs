//! Laya's token sequences, built exactly as the reference builds them
//! (`laya/common.py::build_sequence`), because the ids are the model's input
//! and a different id is a different question:
//!
//!   [CLS] "<type> question: <instructions>" [SEP]
//!   [MASK] " <option 0>" [MASK] " <option 1>" ... [SEP]
//!   <state> [SEP]
//!
//! Every option is `[MASK]` + at most 48 tokens of its text; when the options
//! overflow `head_max_len` each is cut to an equal share (never under 4
//! tokens); the question keeps whatever the options leave, never under 8.
//! The state fills the rest. The literal mask token is replaced by a space in
//! every piece of caller text, so no caller can plant a marker.
//!
//! One deliberate departure: the reference cuts a state that does not fit -
//! silently, to its first `room` tokens (its last, for a conversation list).
//! Here a state that does not fit is READ IN WINDOWS instead - `room` tokens
//! each, half overlapping, the reference's own `predict_long` aggregation
//! choosing the answer - and the response says it did. A state that fits is
//! one sequence, token for token the reference's. Windows are slices of the
//! state's own ids, not decode-and-re-encode (which is how `predict_long`
//! makes them, and which moves BPE boundaries by a token or two).

use paddock_models::laya::LayaConfig;
use paddock_tokenizer::GgufTokenizer;

use super::question::{Kind, Question};

/// Most tokens of one option's text (the reference's `max_length=48`).
const OPTION_TOKENS: usize = 48;

/// A checkpoint's tokenizer and the special tokens a sequence is built from.
pub struct LayaTok {
    tok: GgufTokenizer,
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    /// the mask token's text, scrubbed from every piece of caller text
    pub mask_text: String,
}

impl LayaTok {
    pub fn load(cfg: &LayaConfig) -> Result<Self, String> {
        let dir = cfg.dir.join("tokenizer");
        let tok =
            GgufTokenizer::from_hf_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let tc: serde_json::Value = serde_json::from_slice(
            &std::fs::read(cfg.tokenizer_config_path())
                .map_err(|e| format!("{}: {e}", cfg.tokenizer_config_path().display()))?,
        )
        .map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let special = |k: &str| -> Result<(String, u32), String> {
            let s = match tc.get(k) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Object(o)) => o
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                _ => return Err(format!("tokenizer_config.json: no {k}")),
            };
            let id = tok
                .token_to_id(&s)
                .ok_or_else(|| format!("tokenizer: {k} {s:?} is not in the vocabulary"))?;
            Ok((s, id))
        };
        let (_, cls) = special("cls_token")?;
        let (_, sep) = special("sep_token")?;
        let (mask_text, mask) = special("mask_token")?;
        Ok(Self {
            tok,
            cls,
            sep,
            mask,
            mask_text,
        })
    }

    /// `tok(text, add_special_tokens=False)["input_ids"]`, mask scrubbed.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let clean = text.replace(&self.mask_text, " ");
        self.tok.encode(&clean).map_err(|e| e.to_string())
    }
}

/// A question's sequence minus the state: everything up to and including the
/// [SEP] the state follows, with its markers.
pub struct Prefix {
    pub ids: Vec<u32>,
    pub markers: Vec<u32>,
    /// every option had to be cut to fit `head_max_len`
    pub options_cut: Option<usize>,
}

impl Prefix {
    pub fn build(tok: &LayaTok, q: &Question, head_max_len: usize) -> Result<Self, String> {
        let head = tok.encode(&format!("{} question: {}", q.kind.name(), q.ins))?;
        let mut opts: Vec<Vec<u32>> = Vec::with_capacity(q.options.len());
        for o in &q.options {
            let mut ids = tok.encode(&format!(" {o}"))?;
            ids.truncate(OPTION_TOKENS);
            let mut v = Vec::with_capacity(ids.len() + 1);
            v.push(tok.mask);
            v.extend(ids);
            opts.push(v);
        }
        let used: usize = opts.iter().map(Vec::len).sum();
        let mut budget = head_max_len as i64 - used as i64;
        let mut options_cut = None;
        if budget < 16 {
            let per = 4usize.max(head_max_len.saturating_sub(16) / opts.len().max(1));
            for o in &mut opts {
                o.truncate(per);
            }
            options_cut = Some(per);
            budget = head_max_len as i64 - opts.iter().map(Vec::len).sum::<usize>() as i64;
        }
        let keep = 8i64.max(budget) as usize;
        let mut ids = Vec::with_capacity(head.len().min(keep) + used + 3);
        ids.push(tok.cls);
        ids.extend_from_slice(&head[..head.len().min(keep)]);
        ids.push(tok.sep);
        let mut markers = Vec::with_capacity(opts.len());
        for o in &opts {
            markers.push(ids.len() as u32);
            ids.extend_from_slice(o);
        }
        ids.push(tok.sep);
        Ok(Self {
            ids,
            markers,
            options_cut,
        })
    }

    /// State tokens a sequence can carry after this prefix (and before the
    /// closing [SEP]).
    pub fn room(&self, max_len: usize) -> usize {
        max_len.saturating_sub(self.ids.len() + 1)
    }

    /// The full sequence over `state` (which must fit `room`).
    pub fn with_state(&self, state: &[u32], sep: u32) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.ids.len() + state.len() + 1);
        ids.extend_from_slice(&self.ids);
        ids.extend_from_slice(state);
        ids.push(sep);
        ids
    }
}

/// How a question reads its state: one sequence, or windows over it.
pub struct Plan {
    /// token ranges of the state, one per sequence
    pub windows: Vec<(usize, usize)>,
}

impl Plan {
    /// The reference's single sequence when the state fits; otherwise
    /// `room`-token windows with a half-window stride (`predict_long`'s).
    pub fn for_state(n_state: usize, room: usize) -> Self {
        if n_state <= room {
            return Self {
                windows: vec![(0, n_state)],
            };
        }
        let step = (room / 2).max(1);
        let mut windows = Vec::new();
        let mut i = 0usize;
        while i < n_state {
            let end = (i + room).min(n_state);
            windows.push((i, end));
            if i + room >= n_state {
                break;
            }
            i += step;
        }
        Self { windows }
    }
}

/// The option count a question's markers must equal (a noul is always 2).
pub fn option_count(q: &Question) -> usize {
    match q.kind {
        Kind::Noul => 2,
        _ => q.options.len(),
    }
}
