//! A Laya decision-model checkpoint (ConvAI Innovations, Apache-2.0): a
//! ModernBERT encoder under a small trained head that scores each option of
//! a typed question at its own `[MASK]` - the open model built to answer
//! TypeSafe Jev's `/v1/systemone` shape.
//!
//! A checkpoint directory holds `rl_agent_config.json` (the head: its depth,
//! the token budgets, the fitted temperatures), `encoder/config.json` (a
//! stock transformers ModernBERT config), `tokenizer/tokenizer.json` and one
//! `model.safetensors` (f16, 206 tensors for the English checkpoint). The
//! published repo bundles three: the English checkpoint at the root and
//! `multilingual/` + `typed-decisions/` beside it - [`LayaBundle`] is that
//! directory, and it is what the runner serves: the language router between
//! the checkpoints is part of the model's published behaviour.
//!
//! Same parsing stance as every safetensors family here: each field the
//! engine consumes is validated present, and a switch the reference has and
//! this engine does not build is refused by name. ModernBERT's failure mode
//! is silence - a biased norm, a different window convention or one rope
//! theta for both layer kinds all load and return plausible-looking logits.

use std::path::{Path, PathBuf};

use crate::safetensors::StError;

/// The encoder's shape, as the engine consumes it.
#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    pub hidden: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    /// the GLU's width F: `Wi` is `[2F, hidden]`
    pub intermediate: usize,
    pub vocab: usize,
    /// per layer: true = full (global) attention, false = the local window
    pub global: Vec<bool>,
    /// the local layers' half window: keys with `|i - j| <= window` - HF's
    /// `sliding_window`, `local_attention / 2`, inclusive on both sides
    pub window: usize,
    pub rope_theta_global: f32,
    pub rope_theta_local: f32,
    pub eps: f32,
    pub max_position: usize,
}

impl ModernBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }

    fn read(path: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| StError::Header(format!("{}: {e}", path.display())))?;
        let bad = |m: String| StError::Header(format!("laya encoder config: {m}"));
        let getu = |k: &str| {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("missing {k}")))
        };
        let getf = |k: &str| {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| bad(format!("missing {k}")))
        };
        let gets = |k: &str| v.get(k).and_then(serde_json::Value::as_str);
        let flag = |k: &str| v.get(k).and_then(serde_json::Value::as_bool);

        if gets("model_type") != Some("modernbert") {
            return Err(bad(format!(
                "model_type {:?} (want modernbert)",
                gets("model_type")
            )));
        }
        // each a switch the transformers ModernBERT has and this graph does not
        for (k, want) in [
            ("norm_bias", false),
            ("attention_bias", false),
            ("mlp_bias", false),
        ] {
            if flag(k).unwrap_or(false) != want {
                return Err(bad(format!(
                    "{k} true - only the bias-free ModernBERT is built"
                )));
            }
        }
        let act = gets("hidden_activation").unwrap_or("gelu");
        if act != "gelu" {
            return Err(bad(format!(
                "hidden_activation {act:?} - only the exact (erf) GELU GLU is built"
            )));
        }
        let hidden = getu("hidden_size")?;
        let n_layer = getu("num_hidden_layers")?;
        let n_heads = getu("num_attention_heads")?;
        if n_heads == 0 || !hidden.is_multiple_of(n_heads) {
            return Err(bad(format!("hidden {hidden} over {n_heads} heads")));
        }
        let hd = hidden / n_heads;
        if ![32, 64, 96, 128].contains(&hd) {
            return Err(bad(format!(
                "head dim {hd} - the attention kernel takes 32, 64, 96 or 128"
            )));
        }
        // layer_types (transformers >= 5) or the global_attn_every_n_layers
        // rule it replaced; both are accepted and must agree when both exist
        let every = v
            .get("global_attn_every_n_layers")
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize);
        let global: Vec<bool> = match v.get("layer_types").and_then(serde_json::Value::as_array) {
            Some(types) => {
                if types.len() != n_layer {
                    return Err(bad(format!(
                        "{} layer_types for {n_layer} layers",
                        types.len()
                    )));
                }
                types
                    .iter()
                    .map(|t| match t.as_str() {
                        Some("full_attention") => Ok(true),
                        Some("sliding_attention") => Ok(false),
                        other => Err(bad(format!("layer type {other:?}"))),
                    })
                    .collect::<Result<_, _>>()?
            }
            None => {
                let every = every.ok_or_else(|| bad("missing layer_types".into()))?;
                (0..n_layer).map(|i| i % every.max(1) == 0).collect()
            }
        };
        if let Some(every) = every
            && global
                .iter()
                .enumerate()
                .any(|(i, &g)| g != (i % every.max(1) == 0))
        {
            return Err(bad(
                "layer_types disagree with global_attn_every_n_layers".into()
            ));
        }
        let local = getu("local_attention")?;
        if local < 2 || !local.is_multiple_of(2) {
            return Err(bad(format!("local_attention {local}")));
        }
        // rope: transformers 5's per-kind rope_parameters, or 4.x's two thetas.
        // mmBERT is why this matters: both of its thetas are 160000, and a
        // reader that falls back to 4.x's default local 10000 runs the wrong base.
        let theta = |kind: &str, legacy: &str, default: f64| -> Result<f32, StError> {
            if let Some(p) = v.get("rope_parameters") {
                let per = p.get(kind);
                if let Some(t) = per
                    .and_then(|x| x.get("rope_type"))
                    .and_then(serde_json::Value::as_str)
                    && t != "default"
                {
                    return Err(bad(format!("{kind} rope_type {t:?} - only default rope")));
                }
                if let Some(x) = per
                    .and_then(|x| x.get("rope_theta"))
                    .or_else(|| p.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
                {
                    return Ok(x as f32);
                }
            }
            Ok(v.get(legacy)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(default) as f32)
        };
        Ok(Self {
            hidden,
            n_layer,
            n_heads,
            intermediate: getu("intermediate_size")?,
            vocab: getu("vocab_size")?,
            global,
            window: local / 2,
            rope_theta_global: theta("full_attention", "global_rope_theta", 160000.0)?,
            rope_theta_local: theta("sliding_attention", "local_rope_theta", 10000.0)?,
            eps: v
                .get("norm_eps")
                .and_then(serde_json::Value::as_f64)
                .map_or_else(|| getf("layer_norm_eps"), Ok)? as f32,
            max_position: getu("max_position_embeddings")?,
        })
    }
}

/// Question types in the head's `type_emb` order.
pub const QTYPE_CHOICE: u32 = 0;
pub const QTYPE_SCORE: u32 = 1;
pub const QTYPE_NOUL: u32 = 2;

/// The fitted temperature a checkpoint may ship outside this range is not
/// applied as shipped: the English checkpoint's `choice:11+` bucket is 0.1006,
/// which SHARPENS the logits ~10x and publishes a 0.24 top probability as
/// 0.99. The reference implementation clamps to the same bounds.
pub const TEMP_MIN: f32 = 0.5;
pub const TEMP_MAX: f32 = 5.0;

/// One Laya checkpoint: the head's config and the encoder under it.
#[derive(Debug, Clone)]
pub struct LayaConfig {
    pub dir: PathBuf,
    /// `model_name` (rl-agent, laya-typed-decisions)
    pub name: String,
    pub encoder: ModernBertConfig,
    pub head_layers: usize,
    /// act head outputs: act + one per `act_costs` entry (escalate)
    pub n_act: usize,
    /// whole-sequence token budget, and the question+options share of it
    pub max_len: usize,
    pub head_max_len: usize,
    /// per question type, as shipped
    pub temperature: [f32; 3],
    /// `"choice:3-5"`-style buckets, as shipped
    pub temperature_by_options: Vec<(String, f32)>,
}

impl LayaConfig {
    /// Whether `dir` is a Laya checkpoint - the runner's cheap probe.
    pub fn is_ours(dir: &Path) -> bool {
        dir.join("rl_agent_config.json").is_file()
            && dir.join("encoder").join("config.json").is_file()
            && dir.join("model.safetensors").is_file()
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let path = dir.join("rl_agent_config.json");
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)
            .map_err(|e| StError::Header(format!("{}: {e}", path.display())))?;
        let bad = |m: String| StError::Header(format!("laya rl_agent_config.json: {m}"));
        let getu = |k: &str| {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("missing {k}")))
        };
        let encoder = ModernBertConfig::read(&dir.join("encoder").join("config.json"))?;
        let head_layers = getu("head_layers")?;
        let max_len = getu("max_len")?;
        let head_max_len = getu("head_max_len")?;
        if head_max_len + 8 > max_len || max_len > encoder.max_position {
            return Err(bad(format!(
                "max_len {max_len} / head_max_len {head_max_len} against {} positions",
                encoder.max_position
            )));
        }
        let n_act = v
            .get("act_costs")
            .and_then(serde_json::Value::as_object)
            .map_or(0, |m| m.len())
            + 1;
        let t = v
            .get("temperature")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|x| x.as_f64().map(|x| x as f32))
                    .collect::<Option<Vec<_>>>()
            });
        let temperature = match t {
            None => [1.0; 3],
            Some(Some(t)) if t.len() == 3 => [t[0], t[1], t[2]],
            Some(_) => return Err(bad("temperature: three numbers".into())),
        };
        let temperature_by_options = v
            .get("temperature_by_options")
            .and_then(serde_json::Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, x)| x.as_f64().map(|x| (k.clone(), x as f32)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            dir: dir.to_path_buf(),
            name: v
                .get("model_name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("laya")
                .to_owned(),
            encoder,
            head_layers,
            n_act,
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
        })
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.join("tokenizer").join("tokenizer.json")
    }

    pub fn tokenizer_config_path(&self) -> PathBuf {
        self.dir.join("tokenizer").join("tokenizer_config.json")
    }

    pub fn weights_path(&self) -> PathBuf {
        self.dir.join("model.safetensors")
    }

    /// The temperature a question's logits are divided by: its
    /// `type:size` bucket if the checkpoint fitted one, else its type's -
    /// clamped to [`TEMP_MIN`, `TEMP_MAX`] (a NaN or infinity is 1). `k` is the
    /// question's option count.
    pub fn temperature_for(&self, qtype: u32, k: usize) -> f32 {
        let name = match qtype {
            QTYPE_CHOICE => "choice",
            QTYPE_SCORE => "score",
            _ => "noul",
        };
        let size = match k {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let bucket = format!("{name}:{size}");
        let t = self
            .temperature_by_options
            .iter()
            .find(|(b, _)| *b == bucket)
            .map_or(self.temperature[qtype.min(2) as usize], |(_, t)| *t);
        if t.is_finite() {
            t.clamp(TEMP_MIN, TEMP_MAX)
        } else {
            1.0
        }
    }

    /// The shipped temperatures outside the applied range, for the load log
    /// (the reference warns about the same entries).
    pub fn clamped_temperatures(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (b, t) in &self.temperature_by_options {
            if !(TEMP_MIN..=TEMP_MAX).contains(t) {
                out.push(format!("{b}={t}"));
            }
        }
        for (i, t) in self.temperature.iter().enumerate() {
            if !(TEMP_MIN..=TEMP_MAX).contains(t) {
                out.push(format!("temperature[{i}]={t}"));
            }
        }
        out
    }
}

/// The checkpoints the published repo bundles, by the router's names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Checkpoint {
    English,
    Multilingual,
    TypedDecisions,
}

impl Checkpoint {
    pub const ALL: [Checkpoint; 3] = [
        Checkpoint::English,
        Checkpoint::Multilingual,
        Checkpoint::TypedDecisions,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Checkpoint::English => "english",
            Checkpoint::Multilingual => "multilingual",
            Checkpoint::TypedDecisions => "typed-decisions",
        }
    }

    /// The directory inside the bundle ("" = its root).
    pub fn subdir(self) -> &'static str {
        match self {
            Checkpoint::English => "",
            Checkpoint::Multilingual => "multilingual",
            Checkpoint::TypedDecisions => "typed-decisions",
        }
    }

    /// A checkpoint by any name the reference router accepts: its own, an
    /// alias, or a published Hugging Face id. `None` is "let the router
    /// choose" - which is also what a Jev client's own model id means.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "english" | "en" | "laya" | "default" => Some(Checkpoint::English),
            "multilingual"
            | "multi"
            | "ml"
            | "laya-multilingual"
            | "convaiinnovations/laya-multilingual" => Some(Checkpoint::Multilingual),
            "typed-decisions"
            | "typed"
            | "typed_decisions"
            | "laya-typed-decisions"
            | "decisions"
            | "convaiinnovations/laya-typed-decisions" => Some(Checkpoint::TypedDecisions),
            _ => None,
        }
    }
}

/// A bundle directory: the English checkpoint at its root, the others in
/// their subdirectories when present.
#[derive(Debug, Clone)]
pub struct LayaBundle {
    pub root: PathBuf,
    pub checkpoints: Vec<(Checkpoint, LayaConfig)>,
}

impl LayaBundle {
    /// Whether `dir` is a Laya bundle (its root is a checkpoint).
    pub fn is_ours(dir: &Path) -> bool {
        LayaConfig::is_ours(dir)
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let mut checkpoints = Vec::new();
        for c in Checkpoint::ALL {
            let d = if c.subdir().is_empty() {
                dir.to_path_buf()
            } else {
                dir.join(c.subdir())
            };
            if LayaConfig::is_ours(&d) {
                checkpoints.push((c, LayaConfig::read(&d)?));
            }
        }
        if !checkpoints.iter().any(|(c, _)| *c == Checkpoint::English) {
            return Err(StError::Header(format!(
                "{}: not a Laya bundle - its root holds no checkpoint",
                dir.display()
            )));
        }
        Ok(Self {
            root: dir.to_path_buf(),
            checkpoints,
        })
    }

    pub fn get(&self, c: Checkpoint) -> Option<&LayaConfig> {
        self.checkpoints
            .iter()
            .find(|(k, _)| *k == c)
            .map(|(_, cfg)| cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(tbo: &[(&str, f32)], t: [f32; 3]) -> LayaConfig {
        LayaConfig {
            dir: PathBuf::new(),
            name: "t".into(),
            encoder: ModernBertConfig {
                hidden: 64,
                n_layer: 1,
                n_heads: 1,
                intermediate: 64,
                vocab: 8,
                global: vec![true],
                window: 64,
                rope_theta_global: 160000.0,
                rope_theta_local: 10000.0,
                eps: 1e-5,
                max_position: 512,
            },
            head_layers: 2,
            n_act: 2,
            max_len: 512,
            head_max_len: 192,
            temperature: t,
            temperature_by_options: tbo.iter().map(|(b, x)| (b.to_string(), *x)).collect(),
        }
    }

    #[test]
    fn temperatures_pick_the_bucket_then_the_type_and_clamp() {
        // the English checkpoint's shipped buckets
        let c = cfg(
            &[
                ("choice:3-5", 1.76),
                ("choice:11+", 0.1006),
                ("noul:2", 1.98),
            ],
            [1.64, 1.25, 1.98],
        );
        assert_eq!(c.temperature_for(QTYPE_CHOICE, 3), 1.76);
        // no 6-10 bucket here: the type's own
        assert_eq!(c.temperature_for(QTYPE_CHOICE, 7), 1.64);
        // the sharpening bucket is clamped, never applied as shipped
        assert_eq!(c.temperature_for(QTYPE_CHOICE, 30), TEMP_MIN);
        assert_eq!(c.temperature_for(QTYPE_NOUL, 2), 1.98);
        assert_eq!(c.temperature_for(QTYPE_SCORE, 3), 1.25);
        assert_eq!(
            c.clamped_temperatures(),
            vec!["choice:11+=0.1006".to_string()]
        );
        let nan = cfg(&[], [f32::NAN, 1.0, 1.0]);
        assert_eq!(nan.temperature_for(QTYPE_CHOICE, 3), 1.0);
    }

    #[test]
    fn checkpoint_names_take_the_routers_aliases() {
        assert_eq!(Checkpoint::parse("EN"), Some(Checkpoint::English));
        assert_eq!(
            Checkpoint::parse("convaiinnovations/laya-multilingual"),
            Some(Checkpoint::Multilingual)
        );
        assert_eq!(
            Checkpoint::parse("typed_decisions"),
            Some(Checkpoint::TypedDecisions)
        );
        // a Jev client's own model id means "let the router choose"
        assert_eq!(Checkpoint::parse("jev-1"), None);
        // the root bundle id is "choose", not a pin to English
        assert_eq!(Checkpoint::parse("convaiinnovations/laya"), None);
    }
}
