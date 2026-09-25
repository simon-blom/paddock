//! A Laya question: what `questions` says, validated by the reference's own
//! rules (`Agent._check_question`) and rendered into the option texts its
//! sequences carry (`render_options`) - plus the staging fields every
//! `/v1/systemone` backend takes (`depends_on`, `ask_if`, `alone`).
//!
//! Laya's rules are not DiffusionGemma's: a choice takes one to a hundred
//! options, as a label -> description map or a bare list whose labels may be
//! numbers, booleans or null; a score level must be described; a noul can be
//! relabelled. Each refusal names the question and what to fix, as the
//! reference's do.

use super::pyjson::PyVal;

/// Per-request limits, the reference server's (`laya/serve.py`): one request
/// is one forward pass per question over the whole state, so these bound the
/// work a single body can ask for.
pub const MAX_QUESTIONS: usize = 64;
pub const MAX_CHOICE_OPTIONS: usize = 100;
pub const MAX_SCORE_LEVELS: usize = 32;
pub const MAX_TOTAL_OPTIONS: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Choice,
    Score,
    Noul,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Score => "score",
            Kind::Noul => "noul",
        }
    }
    /// the head's type-embedding row
    pub fn qtype(self) -> u32 {
        match self {
            Kind::Choice => paddock_models::laya::QTYPE_CHOICE,
            Kind::Score => paddock_models::laya::QTYPE_SCORE,
            Kind::Noul => paddock_models::laya::QTYPE_NOUL,
        }
    }
}

pub struct Question {
    pub id: String,
    pub kind: Kind,
    /// the instructions as the sequence spells them (`str(ins)`, or the
    /// `json.dumps` of a non-string)
    pub ins: String,
    /// option texts in label order - a noul is always [false, true]
    pub options: Vec<String>,
    /// a choice's labels as the caller wrote them (the answer echoes these)
    pub labels: Vec<PyVal>,
    /// a score's levels as the caller wrote them (the answer's legend)
    pub levels: Vec<PyVal>,
    /// what each option answers as for `ask_if`: `str(label)`, a level's
    /// rendered text, or yes / no (the other backend's vocabulary, so a
    /// condition reads the same on either)
    pub names: Vec<String>,
    pub depends_on: Vec<String>,
    pub ask_if: Vec<(String, Vec<String>)>,
}

/// `render_criterion`: a string as it stands, anything else as compact JSON.
fn render_criterion(v: &PyVal) -> String {
    match v {
        PyVal::Str(s) => s.clone(),
        other => other.dumps(),
    }
}

/// `v is None or v == ""`.
fn undescribed(v: &PyVal) -> bool {
    matches!(v, PyVal::Null) || matches!(v, PyVal::Str(s) if s.is_empty())
}

/// Parse and render one question. `order`-sensitive by construction: the
/// PyVal keeps the caller's key order, and a choice's option order is which
/// marker is which.
pub fn parse(qid: &str, q: &PyVal) -> Result<Question, String> {
    let PyVal::Dict(_) = q else {
        return Err(format!("question {qid:?}: definition must be an object"));
    };
    let kind = match q.get("type") {
        Some(PyVal::Str(t)) if t == "choice" => Kind::Choice,
        Some(PyVal::Str(t)) if t == "score" => Kind::Score,
        Some(PyVal::Str(t)) if t == "noul" => Kind::Noul,
        other => {
            return Err(format!(
                "question {qid:?}: unknown type {}; use one of choice, noul, score",
                other.map_or("None".into(), PyVal::dumps)
            ));
        }
    };
    let Some(ins) = q.get("instructions") else {
        return Err(format!(
            "question {qid:?}: no 'instructions'; add the text the model should answer"
        ));
    };
    let ins = match ins {
        PyVal::Str(s) => s.clone(),
        other => other.dumps(),
    };
    let crit = q.get("criteria").filter(|c| !matches!(c, PyVal::Null));
    if q.get("labels").is_some() && kind != Kind::Noul {
        return Err(format!(
            "question {qid:?}: 'labels' is only supported for noul questions"
        ));
    }
    let (options, labels, levels, names) = match kind {
        Kind::Choice => {
            // a list of labels is a map with no descriptions
            let pairs: Vec<(PyVal, PyVal)> = match crit {
                Some(PyVal::Dict(kv)) => kv
                    .iter()
                    .map(|(k, v)| (PyVal::Str(k.clone()), v.clone()))
                    .collect(),
                Some(PyVal::List(items)) => {
                    for (i, l) in items.iter().enumerate() {
                        if matches!(l, PyVal::List(_) | PyVal::Dict(_)) {
                            return Err(format!(
                                "question {qid:?}: choice label {i} is a {}; a label is rendered \
                                 as option text and used as the answer key, so it must be a \
                                 scalar (a string, number or None), got {}",
                                if matches!(l, PyVal::List(_)) {
                                    "list"
                                } else {
                                    "dict"
                                },
                                l.dumps()
                            ));
                        }
                    }
                    // a repeated list label is one dict key in the reference
                    let mut seen: Vec<(PyVal, PyVal)> = Vec::new();
                    for l in items {
                        if !seen.iter().any(|(k, _)| k.py_str() == l.py_str()) {
                            seen.push((l.clone(), PyVal::Null));
                        }
                    }
                    seen
                }
                _ => {
                    return Err(format!(
                        "question {qid:?}: a choice question takes 'criteria' as a dict of label \
                         -> description, or a list of labels"
                    ));
                }
            };
            if pairs.is_empty() {
                return Err(format!(
                    "question {qid:?}: a choice question needs at least one criterion"
                ));
            }
            if pairs.len() > MAX_CHOICE_OPTIONS {
                return Err(format!(
                    "too many choice options for {qid:?} ({} > {MAX_CHOICE_OPTIONS})",
                    pairs.len()
                ));
            }
            let options = pairs
                .iter()
                .map(|(k, v)| {
                    if undescribed(v) {
                        k.py_str()
                    } else {
                        format!("{}: {}", k.py_str(), render_criterion(v))
                    }
                })
                .collect();
            let names = pairs.iter().map(|(k, _)| k.py_str()).collect();
            let labels = pairs.into_iter().map(|(k, _)| k).collect();
            (options, labels, Vec::new(), names)
        }
        Kind::Score => {
            let Some(PyVal::List(levels)) = crit else {
                return Err(format!(
                    "question {qid:?}: a score question takes 'criteria' as a list of level \
                     descriptions, index 0 first"
                ));
            };
            if levels.is_empty() {
                return Err(format!(
                    "question {qid:?}: a score question needs at least one level"
                ));
            }
            if let Some(i) = levels.iter().position(|l| matches!(l, PyVal::Null)) {
                return Err(format!(
                    "question {qid:?}: score level {i} is null; give every level a \
                     description, index 0 first"
                ));
            }
            if levels.len() > MAX_SCORE_LEVELS {
                return Err(format!(
                    "too many score levels for {qid:?} ({} > {MAX_SCORE_LEVELS})",
                    levels.len()
                ));
            }
            let options = levels
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect();
            let names = levels.iter().map(render_criterion).collect();
            (options, Vec::new(), levels.clone(), names)
        }
        Kind::Noul => {
            let (mut t, mut f) = (None, None);
            match crit {
                None => {}
                Some(PyVal::Dict(kv)) => {
                    // keys lower-cased, as `str(k).lower()`: "True" is "true"
                    let keys: Vec<String> = kv.iter().map(|(k, _)| k.to_lowercase()).collect();
                    if let Some(bad) = keys.iter().find(|k| *k != "true" && *k != "false") {
                        let mut sorted = keys.clone();
                        sorted.sort();
                        let _ = bad;
                        return Err(format!(
                            "question {qid:?}: a noul question takes 'criteria' keyed only \
                             'true'/'false' (either or both, and omitted is fine), got {sorted:?}"
                        ));
                    }
                    for ((_, v), k) in kv.iter().zip(&keys) {
                        if k == "true" {
                            t = Some(v.clone());
                        } else {
                            f = Some(v.clone());
                        }
                    }
                }
                Some(_) => {
                    return Err(format!(
                        "question {qid:?}: a noul question takes 'criteria' as a dict with \
                         optional 'true'/'false' descriptions, or omits it"
                    ));
                }
            }
            let (fl, tl) = match q.get("labels") {
                None => ("false".to_owned(), "true".to_owned()),
                Some(PyVal::Dict(kv)) => {
                    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v);
                    let bad = || {
                        format!(
                            "question {qid:?}: noul labels must map exactly 'false' and 'true' \
                             to distinct non-empty strings"
                        )
                    };
                    if kv.len() != 2 {
                        return Err(bad());
                    }
                    match (get("false"), get("true")) {
                        (Some(PyVal::Str(a)), Some(PyVal::Str(b))) => {
                            let (a, b) = (a.trim().to_owned(), b.trim().to_owned());
                            if a.is_empty() || b.is_empty() || a == b {
                                return Err(bad());
                            }
                            (a, b)
                        }
                        _ => return Err(bad()),
                    }
                }
                Some(_) => {
                    return Err(format!(
                        "question {qid:?}: noul labels must map exactly 'false' and 'true' to \
                         distinct non-empty strings"
                    ));
                }
            };
            let desc = |v: &Option<PyVal>, dflt: &str| match v {
                Some(v) if !undescribed(v) => render_criterion(v),
                _ => dflt.to_owned(),
            };
            let options = vec![
                format!("{fl}: {}", desc(&f, "no, the statement does not hold")),
                format!("{tl}: {}", desc(&t, "yes, the statement holds")),
            ];
            (
                options,
                Vec::new(),
                Vec::new(),
                vec!["no".to_owned(), "yes".to_owned()],
            )
        }
    };

    let mut depends_on: Vec<String> = match q.get("depends_on") {
        None | Some(PyVal::Null) => Vec::new(),
        Some(PyVal::List(a)) if a.iter().all(|v| matches!(v, PyVal::Str(_))) => a
            .iter()
            .map(|v| v.as_str().expect("checked").to_owned())
            .collect(),
        Some(_) => {
            return Err(format!(
                "question {qid:?}: depends_on must be a list of question ids"
            ));
        }
    };
    let ask_if: Vec<(String, Vec<String>)> = match q.get("ask_if") {
        None | Some(PyVal::Null) => Vec::new(),
        Some(PyVal::Dict(kv)) => {
            let mut v = Vec::with_capacity(kv.len());
            for (dep, vals) in kv {
                let PyVal::List(list) = vals else {
                    return Err(format!(
                        "question {qid:?}: ask_if must map a question id to a non-empty list of \
                         its answers"
                    ));
                };
                if list.is_empty() {
                    return Err(format!(
                        "question {qid:?}: ask_if must map a question id to a non-empty list of \
                         its answers"
                    ));
                }
                v.push((dep.clone(), list.iter().map(PyVal::py_str).collect()));
            }
            v
        }
        Some(_) => {
            return Err(format!(
                "question {qid:?}: ask_if must map a question id to a non-empty list of its \
                 answers"
            ));
        }
    };
    for (dep, _) in &ask_if {
        if !depends_on.contains(dep) {
            depends_on.push(dep.clone());
        }
    }
    match q.get("alone") {
        None | Some(PyVal::Null) | Some(PyVal::Bool(_)) => {}
        Some(_) => return Err(format!("question {qid:?}: alone must be true or false")),
    }
    Ok(Question {
        id: qid.to_owned(),
        kind,
        ins,
        options,
        labels,
        levels,
        names,
        depends_on,
        ask_if,
    })
}

/// Every question of a body, in the caller's order, with the cross-question
/// checks (conditions name real questions and answers they can give) and the
/// reference server's size limits.
pub fn parse_all(questions: &PyVal) -> Result<Vec<Question>, String> {
    let PyVal::Dict(kv) = questions else {
        return Err("'questions' must be an object".into());
    };
    if kv.is_empty() {
        return Err("questions: needs a non-empty map of id -> question".into());
    }
    if kv.len() > MAX_QUESTIONS {
        return Err(format!(
            "too many questions ({} > {MAX_QUESTIONS})",
            kv.len()
        ));
    }
    let out = kv
        .iter()
        .map(|(id, q)| parse(id, q))
        .collect::<Result<Vec<_>, _>>()?;
    let total: usize = out
        .iter()
        .filter(|q| q.kind != Kind::Noul)
        .map(|q| q.options.len())
        .sum();
    if total > MAX_TOTAL_OPTIONS {
        return Err(format!(
            "too many answer options across questions ({total} > {MAX_TOTAL_OPTIONS})"
        ));
    }
    for q in &out {
        for dep in &q.depends_on {
            if dep == &q.id || !out.iter().any(|o| &o.id == dep) {
                return Err(format!(
                    "question {:?}: depends on unknown question {dep:?}",
                    q.id
                ));
            }
        }
        for (dep, vals) in &q.ask_if {
            let names = &out.iter().find(|o| &o.id == dep).expect("checked").names;
            if let Some(v) = vals.iter().find(|v| !names.contains(v)) {
                return Err(format!(
                    "question {:?}: ask_if value {v:?} for {dep:?} must be one of {names:?}",
                    q.id
                ));
            }
        }
    }
    schedule(&out.iter().collect::<Vec<_>>())?;
    Ok(out)
}

/// Questions in stages by their dependencies, declaration order kept; a
/// dependency outside `qs` counts as settled. Refuses a cycle. Same rule as
/// the canvas backend's scheduler.
pub fn schedule<'a>(qs: &[&'a Question]) -> Result<Vec<Vec<&'a Question>>, String> {
    let ids: Vec<&str> = qs.iter().map(|q| q.id.as_str()).collect();
    let mut done: Vec<&str> = Vec::new();
    let mut pending: Vec<&'a Question> = qs.to_vec();
    let mut levels = Vec::new();
    while !pending.is_empty() {
        let level: Vec<&'a Question> = pending
            .iter()
            .copied()
            .filter(|q| {
                q.depends_on
                    .iter()
                    .all(|d| done.contains(&d.as_str()) || !ids.contains(&d.as_str()))
            })
            .collect();
        if level.is_empty() {
            return Err(format!(
                "questions: a dependency cycle among {}",
                pending
                    .iter()
                    .map(|q| q.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        done.extend(level.iter().map(|q| q.id.as_str()));
        pending.retain(|q| !level.iter().any(|l| l.id == q.id));
        levels.push(level);
    }
    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::super::pyjson::parse as pj;
    use super::*;

    #[test]
    fn options_render_as_the_reference_does() {
        let q = parse(
            "d",
            &pj(r#"{"type":"choice","instructions":"x","criteria":{"billing":"charges","sales":null,"x":""}}"#)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(q.options, ["billing: charges", "sales", "x"]);
        let q = parse(
            "s",
            &pj(r#"{"type":"choice","instructions":{"a":1},"criteria":[1, 2.5, true, null]}"#)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(q.options, ["1", "2.5", "True", "None"]);
        assert_eq!(q.ins, "{\"a\": 1}");
        let q = parse(
            "l",
            &pj(r#"{"type":"score","instructions":"x","criteria":["low",{"desc":"mid","w":0.5}]}"#)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            q.options,
            ["level 0: low", "level 1: {\"desc\": \"mid\", \"w\": 0.5}"]
        );
        let q = parse(
            "n",
            &pj(r#"{"type":"noul","instructions":"x","criteria":{"TRUE":"passes"},"labels":{"true":" A ","false":"B"}}"#)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            q.options,
            ["B: no, the statement does not hold", "A: passes"]
        );
    }

    #[test]
    fn refusals_name_the_question() {
        let e = |s: &str| parse("q", &pj(s).unwrap()).err().unwrap();
        assert!(e(r#"{"type":"bool","instructions":"x"}"#).contains("unknown type"));
        assert!(e(r#"{"type":"noul"}"#).contains("no 'instructions'"));
        assert!(
            e(r#"{"type":"score","instructions":"x","criteria":["a",null]}"#)
                .contains("level 1 is null")
        );
        assert!(
            e(r#"{"type":"noul","instructions":"x","criteria":{"yes":"a"}}"#)
                .contains("keyed only")
        );
        assert!(e(r#"{"type":"choice","instructions":"x","criteria":[[1]]}"#).contains("scalar"));
        assert!(
            e(r#"{"type":"choice","instructions":"x","criteria":{}}"#).contains("at least one")
        );
    }
}
