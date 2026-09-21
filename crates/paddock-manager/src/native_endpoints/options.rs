//! The native Advanced form and its write allowlist share this schema. Never
//! project arbitrary TOML: runner/search/MCP credentials remain Rust-owned.
use serde_json::{Value, json};

struct Field {
    key: &'static str,
    label: &'static str,
    group: &'static str,
    kind: &'static str,
    min: f64,
    max: f64,
    default: &'static str,
    help: &'static str,
    capability: &'static str,
}

const FIELDS: &[Field] = &[
    Field {
        key: "max_tokens",
        label: "Default output limit",
        group: "Generation defaults",
        kind: "integer",
        min: 1.,
        max: 1_048_576.,
        default: "Model default",
        help: "Used when a request omits its output limit. Context still bounds the reply.",
        capability: "chat",
    },
    Field {
        key: "temp",
        label: "Temperature",
        group: "Generation defaults",
        kind: "number",
        min: 0.,
        max: 2.,
        default: "Model default",
        help: "0 is greedy decoding. Requests can override this default.",
        capability: "chat",
    },
    Field {
        key: "top_p",
        label: "Top P",
        group: "Generation defaults",
        kind: "number",
        min: 0.,
        max: 1.,
        default: "Model default",
        help: "Nucleus sampling probability. 1 leaves the distribution unchanged.",
        capability: "chat",
    },
    Field {
        key: "top_k",
        label: "Top K",
        group: "Generation defaults",
        kind: "integer",
        min: 0.,
        max: 1_000_000.,
        default: "Model default",
        help: "Keep this many token candidates. 0 disables Top K.",
        capability: "chat",
    },
    Field {
        key: "min_p",
        label: "Min P",
        group: "Generation defaults",
        kind: "number",
        min: 0.,
        max: 1.,
        default: "Model default",
        help: "Minimum probability relative to the most likely token. 0 disables it.",
        capability: "chat",
    },
    Field {
        key: "repeat_penalty",
        label: "Repetition penalty",
        group: "Generation defaults",
        kind: "number",
        min: 0.01,
        max: 10.,
        default: "Model default",
        help: "1 disables the penalty. Values above 1 discourage repeated tokens.",
        capability: "chat",
    },
    Field {
        key: "repeat_last_n",
        label: "Repetition window",
        group: "Generation defaults",
        kind: "integer",
        min: 0.,
        max: 1_048_576.,
        default: "64 tokens",
        help: "Number of previous tokens inspected by the repetition penalty.",
        capability: "chat",
    },
    Field {
        key: "seed",
        label: "Random seed",
        group: "Generation defaults",
        kind: "integer",
        min: 0.,
        max: 4_294_967_295.,
        default: "Random per request",
        help: "Pins the seed only when the request does not supply one.",
        capability: "chat",
    },
    Field {
        key: "pdf_max_pages",
        label: "PDF page limit",
        group: "Document input",
        kind: "integer",
        min: 1.,
        max: 10_000.,
        default: "20 pages",
        help: "Maximum rasterized pages per PDF. Truncation is reported to the caller.",
        capability: "vision",
    },
    Field {
        key: "pdf_page_long_edge",
        label: "PDF rendering size",
        group: "Document input",
        kind: "integer",
        min: 128.,
        max: 8192.,
        default: "1568 pixels",
        help: "Long edge of a rasterized PDF page. Higher values cost more image processing time.",
        capability: "vision",
    },
    Field {
        key: "vad_gate",
        label: "Skip non-speech windows",
        group: "Speech input",
        kind: "boolean",
        min: 0.,
        max: 0.,
        default: "Off",
        help: "Voice-activity gating can change transcript content. Disabled unless explicitly selected.",
        capability: "asr",
    },
    Field {
        key: "served_model_name",
        label: "API model name",
        group: "API & admission",
        kind: "text",
        min: 0.,
        max: 256.,
        default: "Checkpoint name",
        help: "The model ID exposed to API clients. Changing it can require client configuration changes.",
        capability: "",
    },
    Field {
        key: "concurrency_limit",
        label: "In-flight request limit",
        group: "API & admission",
        kind: "integer",
        min: 1.,
        max: 65536.,
        default: "No explicit limit",
        help: "Admission cap, not batching width. Excess requests receive an overloaded response.",
        capability: "",
    },
    Field {
        key: "max_output_ceiling",
        label: "Hard output ceiling",
        group: "API & admission",
        kind: "integer",
        min: 1.,
        max: 1_048_576.,
        default: "No additional ceiling",
        help: "Caps output even when the client requests more tokens.",
        capability: "chat",
    },
    Field {
        key: "no_events",
        label: "Disable request event history",
        group: "Diagnostics",
        kind: "boolean",
        min: 0.,
        max: 0.,
        default: "Off",
        help: "Turns off the runner's bounded in-memory request event ring.",
        capability: "",
    },
    Field {
        key: "no_metrics",
        label: "Disable Prometheus metrics",
        group: "Diagnostics",
        kind: "boolean",
        min: 0.,
        max: 0.,
        default: "Off",
        help: "Turns off /metrics independently of request event history.",
        capability: "",
    },
];

pub(super) fn projection(doc: &toml::Value) -> Value {
    Value::Array(
        FIELDS
            .iter()
            .map(|f| {
                json!({
                    "id":f.key, "label":f.label, "group":f.group, "kind":f.kind,
                    "minimum":f.min, "maximum":f.max, "placeholder":f.default,
                    "help":f.help, "capability":f.capability, "value":doc.get(f.key),
                })
            })
            .collect(),
    )
}

pub(super) fn validate(key: &str, value: &Value) -> Result<Option<toml::Value>, String> {
    let f = FIELDS
        .iter()
        .find(|f| f.key == key)
        .ok_or("This runtime option is not editable in the native app.")?;
    if value.is_null() {
        return Ok(None);
    }
    let valid = match f.kind {
        "boolean" => value.as_bool().map(toml::Value::Boolean),
        "text" => value
            .as_str()
            .filter(|v| {
                !v.trim().is_empty()
                    && v.len() <= f.max as usize
                    && !v.chars().any(char::is_control)
            })
            .map(|v| toml::Value::String(v.into())),
        "integer" => value
            .as_i64()
            .filter(|v| (*v as f64) >= f.min && (*v as f64) <= f.max)
            .map(toml::Value::Integer),
        _ => value
            .as_f64()
            .filter(|v| v.is_finite() && *v >= f.min && *v <= f.max)
            .map(toml::Value::Float),
    };
    valid.map(Some).ok_or_else(|| {
        format!(
            "Invalid {}. Use the range shown in Advanced or restore its default.",
            f.label
        )
    })
}

pub(super) fn patch(
    doc: &mut toml_edit::DocumentMut,
    expected: &mut toml::Value,
    values: &serde_json::Map<String, Value>,
) -> Result<(), String> {
    if values.is_empty() || values.len() > FIELDS.len() {
        return Err("Select runtime options to change.".into());
    }
    for (key, value) in values {
        match validate(key, value)? {
            Some(value) => {
                expected
                    .as_table_mut()
                    .ok_or("Invalid configuration.")?
                    .insert(key.clone(), value.clone());
                let text = toml::to_string(&toml::Table::from_iter([(key.clone(), value)]))
                    .map_err(|_| "Cannot prepare runtime option.")?;
                let field: toml_edit::DocumentMut =
                    text.parse().map_err(|_| "Cannot prepare runtime option.")?;
                doc[key] = field[key].clone();
            }
            None => {
                expected
                    .as_table_mut()
                    .ok_or("Invalid configuration.")?
                    .remove(key);
                doc.remove(key);
            }
        }
    }
    Ok(())
}
