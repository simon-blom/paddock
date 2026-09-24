//! `POST /v1/systemone` - structured decisions on a block-diffusion model
//! (DiffusionGemma), in the Jev shape vLLM's structured-diffusion example
//! server speaks: `state` + a map of `questions` in, a typed probabilistic
//! answer per question out - `noul` (a yes/no probability), `choice` (one
//! option with a distribution over all of them), `score` (an ordered level
//! set with a fractional score).
//!
//! How it reads: the questions and their allowed labels go into a system
//! turn, the state into the user turn, both rendered through the model's own
//! chat template with thinking off; the response template (`id: label` per
//! question, first label everywhere) is tokenized and seeded into a canvas
//! behind the empty thought channel the model opens its answers with, with
//! ONLY the answer slots left as noise; one forward at temperature 1 gives
//! every slot its full-vocab distribution, from which each question's label
//! probabilities are read (`Generator::canvas_read`). Nothing is generated,
//! nothing committed. Labels must be single tokens that change exactly one
//! template position (the example server's `resolve_template` rule) - a
//! label set that does not is refused by name.
//!
//! What the answer carries beyond the example server: the mass the model
//! put OUTSIDE the label set at each slot (`1 - sum` of the raw label
//! probabilities), read from the same normalized plane - the calibration
//! signal a top-k logprobs read cannot see.
//!
//! Resampling (`samples`): `"auto"` (default) reads once and, when any
//! question's slot entropy is above `auto_threshold` (0.1), re-reads with
//! fresh slot noise up to `auto_max` (4) times in all; an integer reads that
//! many times. Probabilities average over the reads; `agreement` is the
//! share of reads that picked the reported label, and the diagnostics list
//! every read's pick and confidence per question, so a caller can see WHICH
//! reads disagreed and not only that some did.

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use paddock_engine::generator::CanvasReadOut;
use paddock_engine::service::{CanvasReadRequest, GenRequest, TokenEvent};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc::unbounded_channel;

use paddock_api::ErrorBody;

use crate::routes::AppState;
use crate::serving::ServingModel;

/// Questions per request - the example server's cap; a canvas holds a few
/// dozen `id: label` lines before the model stops tracking them.
pub const MAX_QUESTIONS: usize = 64;
/// Reads per question, fixed or auto.
pub const MAX_SAMPLES: usize = 32;
/// The canvas grows in steps of this many positions.
const CANVAS_STEP: usize = 16;
/// `<turn|>` closes the model's answer inside the canvas.
const TURN_CLOSE: u32 = 106;
/// `<pad>` fills the canvas past the template.
const PAD: u32 = 0;

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

fn invalid(msg: impl Into<String>) -> Response {
    err(StatusCode::UNPROCESSABLE_ENTITY, "validation_error", msg)
}

/// One question, validated: its labels in template order, and what each
/// label stands for (the option name / level / yes-no) for the answer.
struct Question {
    id: String,
    kind: Kind,
    instructions: String,
    /// the single-token label text as it appears in the template ("yes",
    /// "A", "3"), in order
    labels: Vec<String>,
    /// what each label means to the caller: option names, level names, or
    /// "yes"/"no"
    names: Vec<String>,
    /// per-option descriptions for the system turn (empty where none)
    descriptions: Vec<Option<String>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Noul,
    Choice,
    Score,
}

/// `questions` as Jev sends them: a map of id -> {type, instructions,
/// criteria}. `noul` (also `bool`/`boolean`) takes an optional
/// `{true: .., false: ..}` description pair; `choice` a non-empty map of
/// option name -> description; `score` an ordered list of level names.
fn parse_questions(body: &Map<String, Value>) -> Result<Vec<Question>, String> {
    let Some(qs) = body.get("questions").and_then(Value::as_object) else {
        return Err("questions: needs a non-empty map of id -> question".into());
    };
    if qs.is_empty() {
        return Err("questions: needs a non-empty map of id -> question".into());
    }
    if qs.len() > MAX_QUESTIONS {
        return Err(format!(
            "questions: {} given, at most {MAX_QUESTIONS} per request",
            qs.len()
        ));
    }
    let mut out = Vec::with_capacity(qs.len());
    for (qid, q) in qs {
        let Some(q) = q.as_object() else {
            return Err(format!("question {qid:?}: must be an object"));
        };
        if qid.is_empty() || qid.chars().any(|c| c.is_whitespace() || c == ':') {
            return Err(format!(
                "question {qid:?}: ids are single words without ':' (they are written into the \
                 answer template)"
            ));
        }
        let instructions = match q.get("instructions") {
            None => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        };
        let kind = q.get("type").and_then(Value::as_str).unwrap_or("");
        let crit = q.get("criteria");
        let (kind, labels, names, descriptions) = match kind {
            "noul" | "bool" | "boolean" => {
                let (dt, df) = match crit {
                    None | Some(Value::Null) => (None, None),
                    Some(Value::Object(c)) => (
                        c.get("true").and_then(Value::as_str).map(str::to_owned),
                        c.get("false").and_then(Value::as_str).map(str::to_owned),
                    ),
                    Some(_) => {
                        return Err(format!(
                            "question {qid:?}: noul criteria must be an object with true and false"
                        ));
                    }
                };
                (
                    Kind::Noul,
                    vec!["yes".to_owned(), "no".to_owned()],
                    vec!["yes".to_owned(), "no".to_owned()],
                    vec![dt, df],
                )
            }
            "choice" => {
                let Some(Value::Object(c)) = crit else {
                    return Err(format!(
                        "question {qid:?}: choice criteria must map option names to descriptions"
                    ));
                };
                if c.is_empty() {
                    return Err(format!(
                        "question {qid:?}: choice criteria must map option names to descriptions"
                    ));
                }
                let names: Vec<String> = c.keys().cloned().collect();
                let descriptions = c
                    .values()
                    .map(|d| {
                        d.as_str()
                            .map(str::to_owned)
                            .or_else(|| Some(d.to_string()))
                    })
                    .collect();
                let labels = (0..names.len())
                    .map(|i| char::from(b'A' + i as u8).to_string())
                    .collect();
                (Kind::Choice, labels, names, descriptions)
            }
            "score" => {
                let Some(Value::Array(levels)) = crit else {
                    return Err(format!(
                        "question {qid:?}: score criteria must be an ordered list of levels"
                    ));
                };
                let names: Vec<String> = levels
                    .iter()
                    .map(|l| {
                        l.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| l.to_string())
                    })
                    .collect();
                let labels: Vec<String> = if names.len() <= 9 {
                    (0..names.len()).map(|i| (i + 1).to_string()).collect()
                } else {
                    (0..names.len())
                        .map(|i| char::from(b'A' + i as u8).to_string())
                        .collect()
                };
                let descriptions = vec![None; names.len()];
                (Kind::Score, labels, names, descriptions)
            }
            other => return Err(format!("question {qid:?}: unknown type {other:?}")),
        };
        if names.len() < 2 || names.len() > 26 {
            return Err(format!(
                "question {qid:?}: {} alternatives; a question takes 2 to 26",
                names.len()
            ));
        }
        for key in ["depends_on", "ask_if", "alone"] {
            if q.contains_key(key) {
                return Err(format!(
                    "question {qid:?}: `{key}` (conditional questions) is not served yet"
                ));
            }
        }
        out.push(Question {
            id: qid.clone(),
            kind,
            instructions,
            labels,
            names,
            descriptions,
        });
    }
    Ok(out)
}

/// The system turn: what to answer and in which form. The model reads its
/// own answer template, so the labels named here are the template's.
fn system_text(qs: &[Question]) -> String {
    let mut s = String::from(
        "Answer a fixed set of questions about the state the user provides. Each question \
         lists its allowed answers; reply with exactly one label per question.\n\nQuestions:\n",
    );
    for q in qs {
        s.push_str(&format!("- {}: {}\n", q.id, q.instructions.trim()));
        for (i, label) in q.labels.iter().enumerate() {
            let name = &q.names[i];
            match (&q.descriptions[i], q.kind) {
                (Some(d), _) if name != label => {
                    s.push_str(&format!("    {label} = {name}: {d}\n"))
                }
                (Some(d), _) => s.push_str(&format!("    {label}: {d}\n")),
                (None, Kind::Noul) => s.push_str(&format!("    {label}\n")),
                (None, _) => s.push_str(&format!("    {label} = {name}\n")),
            }
        }
    }
    s.push_str("\nReply with one line per question, in this order, formatted as \"id: label\".");
    s
}

/// `id: label` per question, joined by newlines, with `labels[i]` picked
/// for question i.
fn template_text(qs: &[Question], pick: &[usize]) -> String {
    qs.iter()
        .zip(pick)
        .map(|(q, &p)| format!("{}: {}", q.id, q.labels[p]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Where each question's label sits in the tokenized template, and the
/// token id of each of its labels there. The example server's rule: every
/// label of a question must tokenize the template to the same length as
/// the base and differ from it at exactly one position, the same position
/// for all of that question's labels.
struct Slot {
    pos: usize,
    label_ids: Vec<u32>,
}

fn resolve_slots(model: &ServingModel, qs: &[Question]) -> Result<(Vec<u32>, Vec<Slot>), String> {
    let base_pick = vec![0usize; qs.len()];
    let base = model
        .tokenizer
        .encode(&template_text(qs, &base_pick))
        .map_err(|e| e.to_string())?;
    let mut slots = Vec::with_capacity(qs.len());
    for (qi, q) in qs.iter().enumerate() {
        let mut pos: Option<usize> = None;
        let mut ids = Vec::with_capacity(q.labels.len());
        for li in 0..q.labels.len() {
            let mut pick = base_pick.clone();
            pick[qi] = li;
            let ids_li = model
                .tokenizer
                .encode(&template_text(qs, &pick))
                .map_err(|e| e.to_string())?;
            if ids_li.len() != base.len() {
                return Err(format!(
                    "question {:?}: label {:?} is not a single token in the answer template",
                    q.id, q.labels[li]
                ));
            }
            let diff: Vec<usize> = (0..base.len()).filter(|&i| ids_li[i] != base[i]).collect();
            let p = match (diff.as_slice(), pos) {
                ([], Some(p)) => p, // the base label itself
                ([], None) if li == 0 => {
                    // the base pick: its position is whatever the others differ at
                    ids.push(0); // patched once pos is known
                    continue;
                }
                ([p], _) => *p,
                _ => {
                    return Err(format!(
                        "question {:?}: label {:?} changes {} template positions, not one",
                        q.id,
                        q.labels[li],
                        diff.len()
                    ));
                }
            };
            match pos {
                None => pos = Some(p),
                Some(q_pos) if q_pos != p => {
                    return Err(format!(
                        "question {:?}: its labels do not share one template position",
                        q.id
                    ));
                }
                _ => {}
            }
            ids.push(ids_li[p]);
        }
        let Some(p) = pos else {
            return Err(format!(
                "question {:?}: its labels tokenize identically",
                q.id
            ));
        };
        // the base label's id at the shared position
        ids[0] = base[p];
        slots.push(Slot {
            pos: p,
            label_ids: ids,
        });
    }
    Ok((base, slots))
}

/// The prompt: system + user rendered through the model's own template with
/// thinking off, tokenized, BOS-led the way the chat path does it.
fn prompt_ids(model: &ServingModel, sys: &str, state: &str) -> Result<Vec<u32>, String> {
    let template = model
        .chat_template
        .as_deref()
        .ok_or("this model has no chat template")?;
    let messages = vec![
        json!({"role": "system", "content": sys}),
        json!({"role": "user", "content": state}),
    ];
    let kwargs = json!({"enable_thinking": false});
    let text = crate::chat_template::render_with_specials(
        template,
        &messages,
        None,
        Some(&kwargs),
        &model.template_specials(),
    )?;
    let mut ids = model.tokenizer.encode(&text).map_err(|e| e.to_string())?;
    if let Some(bos) = model.bos
        && ids.first() != Some(&bos)
    {
        ids.insert(0, bos);
    }
    Ok(ids)
}

/// The empty thought channel the model opens every answer with when
/// thinking is off - seeded ahead of the template so the answer line is
/// read as an answer line (measured: without it the slot wants a newline).
fn scaffold(model: &ServingModel) -> Result<Vec<u32>, String> {
    let mut s = model
        .tokenizer
        .encode("<|channel>thought\n")
        .map_err(|e| e.to_string())?;
    s.extend(
        model
            .tokenizer
            .encode("<channel|>")
            .map_err(|e| e.to_string())?,
    );
    Ok(s)
}

/// One read's per-question outcome.
struct Read {
    /// normalized over the question's labels
    probs: Vec<f32>,
    /// raw label mass before normalizing (1 - this = outside the label set)
    mass: f32,
    entropy: f32,
}

/// Run one read on the engine: seed the canvas (template behind the
/// scaffold, `<turn|>`, pad; noise at the slots), submit, collect.
async fn read_once(
    model: &ServingModel,
    prompt: &[u32],
    head: &[u32],
    template: &[u32],
    slots: &[Slot],
    union: &[u32],
    width: usize,
    noise_seed: u64,
) -> Result<Vec<Read>, Box<Response>> {
    let mut canvas = head.to_vec();
    canvas.extend_from_slice(template);
    canvas.push(TURN_CLOSE);
    canvas.resize(width, PAD);
    // per-read slot noise: uniform ids from a splitmix64 stream on the seed
    let mut x = noise_seed ^ 0x9E37_79B9_7F4A_7C15;
    let vocab = model.tokenizer.vocab_size().max(1) as u64;
    for s in slots {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        canvas[head.len() + s.pos] = (z % vocab) as u32;
    }
    let (reply_tx, reply_rx) = std::sync::mpsc::channel::<CanvasReadOut>();
    let (tx, mut rx) = unbounded_channel();
    let req = GenRequest {
        prompt: prompt.to_vec(),
        max_tokens: 1,
        sampler: Default::default(),
        stop_tokens: Vec::new(),
        events: tx,
        mm_chunks: None,
        constraint: None,
        logprobs: None,
        submitted: None,
        canvas_read: Some(CanvasReadRequest {
            canvas,
            label_ids: union.to_vec(),
            reply: reply_tx,
        }),
    };
    if let Err(e) = model.engine.submit(req) {
        return Err(Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            e,
        )));
    }
    loop {
        match rx.recv().await {
            Some(TokenEvent::Done(..)) => break,
            Some(TokenEvent::Error(e)) => return Err(Box::new(crate::chat::engine_err(&e))),
            Some(_) => {}
            None => {
                return Err(Box::new(err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "the engine closed the read without answering",
                )));
            }
        }
    }
    let Ok(out) = reply_rx.try_recv() else {
        return Err(Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "the engine finished the read without a distribution",
        )));
    };
    let k = union.len();
    Ok(slots
        .iter()
        .map(|s| {
            let pos = head.len() + s.pos;
            let raw: Vec<f32> = s
                .label_ids
                .iter()
                .map(|id| {
                    let j = union.iter().position(|u| u == id).expect("label in union");
                    out.probs.get(pos * k + j).copied().unwrap_or(0.0)
                })
                .collect();
            let mass: f32 = raw.iter().sum();
            let probs = if mass > 0.0 {
                raw.iter().map(|p| p / mass).collect()
            } else {
                vec![1.0 / raw.len() as f32; raw.len()]
            };
            Read {
                probs,
                mass,
                entropy: out.entropy.get(pos).copied().unwrap_or(f32::NAN),
            }
        })
        .collect())
}

pub async fn handle(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let t0 = Instant::now();
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no chat model is loaded",
        );
    };
    let width_cap = model.engine.canvas_width();
    if width_cap == 0 {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "structured decisions need a block-diffusion model (DiffusionGemma); this model \
             generates one token at a time",
        );
    }
    let Some(body) = body.as_object() else {
        return invalid("the body must be a JSON object");
    };
    let qs = match parse_questions(body) {
        Ok(q) => q,
        Err(e) => return invalid(e),
    };
    let state_text = match body.get("state") {
        None | Some(Value::Null) => return invalid("state: required"),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    };
    if body.get("images").is_some_and(|v| !v.is_null()) {
        return invalid("images are not served on this endpoint yet");
    }
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(42);
    if let Some(s) = body.get("steps").and_then(Value::as_u64)
        && s != 1
    {
        return invalid("steps: only 1 (a single denoising read) is served");
    }
    // samples: "auto" | N
    let (auto, fixed) = match body.get("samples") {
        None | Some(Value::Null) => (true, 1usize),
        Some(Value::String(s)) if s == "auto" => (true, 1),
        Some(v) => match v.as_u64() {
            Some(n) if (1..=MAX_SAMPLES as u64).contains(&n) => (false, n as usize),
            _ => return invalid(format!("samples: \"auto\" or 1..={MAX_SAMPLES}")),
        },
    };
    let auto_threshold = body
        .get("auto_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.1) as f32;
    let auto_max = body
        .get("auto_max")
        .and_then(Value::as_u64)
        .map_or(4usize, |n| (n as usize).clamp(1, MAX_SAMPLES));

    let sys = system_text(&qs);
    let prompt = match prompt_ids(model, &sys, &state_text) {
        Ok(p) => p,
        Err(e) => return invalid(e),
    };
    let (template, slots) = match resolve_slots(model, &qs) {
        Ok(v) => v,
        Err(e) => return invalid(e),
    };
    let head = match scaffold(model) {
        Ok(h) => h,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
    };
    let need = head.len() + template.len() + 1;
    let width = need.next_multiple_of(CANVAS_STEP);
    if width > width_cap {
        return invalid(format!(
            "the answer template needs {need} canvas positions; this model reads up to \
             {width_cap} - ask fewer questions per call"
        ));
    }
    if prompt.len() + width > state.max_ctx {
        return invalid(format!(
            "the prompt is {} tokens and the canvas {width}; the window is {}",
            prompt.len(),
            state.max_ctx
        ));
    }
    let mut union: Vec<u32> = Vec::new();
    for s in &slots {
        for &id in &s.label_ids {
            if !union.contains(&id) {
                union.push(id);
            }
        }
    }

    // the reads: one, then more when the entropy says so (auto) or as asked
    let mut reads: Vec<Vec<Read>> = Vec::new();
    let mut n_reads = 0usize;
    loop {
        let r = match read_once(
            model,
            &prompt,
            &head,
            &template,
            &slots,
            &union,
            width,
            seed.wrapping_add(n_reads as u64 * 7919),
        )
        .await
        {
            Ok(r) => r,
            Err(resp) => return *resp,
        };
        reads.push(r);
        n_reads += 1;
        let more = if auto {
            n_reads < auto_max
                && reads[0]
                    .iter()
                    .any(|q| q.entropy.is_nan() || q.entropy > auto_threshold)
        } else {
            n_reads < fixed
        };
        if !more {
            break;
        }
    }

    // fold the reads per question
    let mut answers = Map::new();
    let mut diag_q = Vec::with_capacity(qs.len());
    for (qi, q) in qs.iter().enumerate() {
        let n = q.labels.len();
        let mut mean = vec![0f32; n];
        let mut picks = vec![0usize; n];
        let mut mass = 0f32;
        let mut entropy = 0f32;
        // per read, what this slot picked and how surely - the diagnostics
        // carry it so a resampled read can be shown read by read (which
        // reads disagreed, and at what confidence) instead of only as an
        // agreement ratio
        let mut per_read = Vec::with_capacity(reads.len());
        for r in &reads {
            let rq = &r[qi];
            for (m, p) in mean.iter_mut().zip(&rq.probs) {
                *m += p;
            }
            let top = rq
                .probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i);
            picks[top] += 1;
            mass += rq.mass;
            entropy += rq.entropy;
            per_read.push(json!({
                "pick": q.names[top],
                "confidence": rq.probs[top],
                "entropy": rq.entropy,
            }));
        }
        let k = reads.len() as f32;
        for m in mean.iter_mut() {
            *m /= k;
        }
        mass /= k;
        entropy /= k;
        let top = mean
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i);
        let confidence = mean[top];
        let agreement = picks[top] as f32 / k;
        let probabilities: Map<String, Value> = q
            .names
            .iter()
            .zip(&mean)
            .map(|(name, p)| (name.clone(), json!(p)))
            .collect();
        let answer = match q.kind {
            Kind::Noul => json!({
                "type": "noul",
                "noul": mean[0],
                "confidence": confidence,
                "agreement": agreement,
                "outside": 1.0 - mass,
            }),
            Kind::Choice => json!({
                "type": "choice",
                "choice": q.names[top],
                "probabilities": probabilities,
                "confidence": confidence,
                "agreement": agreement,
                "outside": 1.0 - mass,
            }),
            Kind::Score => {
                let score: f32 = mean.iter().enumerate().map(|(i, p)| i as f32 * p).sum();
                let legend: Map<String, Value> = q
                    .names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (i.to_string(), json!(name)))
                    .collect();
                let probs_idx: Map<String, Value> = mean
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i.to_string(), json!(p)))
                    .collect();
                json!({
                    "type": "score",
                    "score": score,
                    "level": q.names[top],
                    "legend": legend,
                    "probabilities": probs_idx,
                    "confidence": confidence,
                    "agreement": agreement,
                    "outside": 1.0 - mass,
                })
            }
        };
        answers.insert(q.id.clone(), answer);
        diag_q.push(json!({
            "id": q.id,
            "label": q.labels[top],
            "position": head.len() + slots[qi].pos,
            "entropy": entropy,
            "label_mass": mass,
            "reads": per_read,
        }));
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;
    (
        StatusCode::OK,
        Json(json!({
            "model": model.id,
            "answers": answers,
            "usage": {"input_tokens": prompt.len(), "output_tokens": 0},
            "diagnostics": {
                "reads": n_reads,
                "canvas": width,
                "questions": diag_q,
                "timing": {"total_ms": total_ms},
            },
        })),
    )
        .into_response()
}
