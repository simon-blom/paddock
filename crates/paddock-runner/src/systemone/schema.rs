//! The question schema of a structured read: what `questions` says, in the
//! order the caller wrote it, and how the questions run - in stages by their
//! dependencies, in chunks by the canvas.

use serde::Deserialize;
use serde::de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

/// Questions per request - the example server's cap; past it a decision is
/// many decisions and should be sent as such.
pub const MAX_QUESTIONS: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

/// One question, validated: its labels in template order, and what each
/// label stands for (the option name / level / yes-no) for the answer.
pub struct Question {
    pub id: String,
    pub kind: Kind,
    pub instructions: String,
    /// the single-token label as it is written into the template: yes/no, a
    /// letter per option (any option name then works - the name lives in the
    /// system turn), a digit per level (letters past nine)
    pub labels: Vec<String>,
    /// what each label means to the caller: option names, level names, or
    /// "yes"/"no" - the answer's vocabulary and `ask_if`'s
    pub names: Vec<String>,
    /// per-option descriptions for the system turn (None where none)
    pub descriptions: Vec<Option<String>>,
    /// questions whose answers this one is read with - `ask_if`'s keys are
    /// folded in, so a condition always runs after what it tests
    pub depends_on: Vec<String>,
    /// asked only when each named question's answer is among the listed
    /// names; otherwise its answer is null
    pub ask_if: Vec<(String, Vec<String>)>,
    /// a read of its own: never shares a canvas with other questions
    pub alone: bool,
}

// ── key order ───────────────────────────────────────────────────────────────
// serde_json's map sorts its keys, and a question map's order is not
// cosmetic: it is the order the questions are asked in (the template lists
// them that way), and a choice's option order decides which option is A.
// Parsing the body into Value alphabetised both. This second, tiny pass over
// the same bytes records the keys in document order and nothing else.

/// The keys of a JSON object in document order; empty for anything else.
struct Keys(Vec<String>);

impl<'de> Deserialize<'de> for Keys {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Keys;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Keys, A::Error> {
                let mut keys = Vec::new();
                while let Some(k) = m.next_key::<String>()? {
                    m.next_value::<IgnoredAny>()?;
                    keys.push(k);
                }
                Ok(Keys(keys))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<Keys, A::Error> {
                while s.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Keys(Vec::new()))
            }
            fn visit_str<E>(self, _: &str) -> Result<Keys, E> {
                Ok(Keys(Vec::new()))
            }
            fn visit_bool<E>(self, _: bool) -> Result<Keys, E> {
                Ok(Keys(Vec::new()))
            }
            fn visit_i64<E>(self, _: i64) -> Result<Keys, E> {
                Ok(Keys(Vec::new()))
            }
            fn visit_u64<E>(self, _: u64) -> Result<Keys, E> {
                Ok(Keys(Vec::new()))
            }
            fn visit_f64<E>(self, _: f64) -> Result<Keys, E> {
                Ok(Keys(Vec::new()))
            }
            fn visit_unit<E>(self) -> Result<Keys, E> {
                Ok(Keys(Vec::new()))
            }
            fn visit_none<E>(self) -> Result<Keys, E> {
                Ok(Keys(Vec::new()))
            }
        }
        d.deserialize_any(V)
    }
}

#[derive(Deserialize, Default)]
struct QuestionKeys {
    #[serde(default)]
    criteria: Option<Keys>,
}

/// `questions` in document order: each id with its criteria's key order.
struct QuestionsInOrder(Vec<(String, Vec<String>)>);

impl<'de> Deserialize<'de> for QuestionsInOrder {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = QuestionsInOrder;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map of question id -> question")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<QuestionsInOrder, A::Error> {
                let mut out = Vec::new();
                while let Some(k) = m.next_key::<String>()? {
                    // read straight from the bytes: going through Value here
                    // would sort the criteria keys again. A malformed
                    // question fails this pass, which then yields no order;
                    // the real parse refuses it with its own message.
                    let q = m.next_value::<QuestionKeys>()?;
                    out.push((k, q.criteria.map(|c| c.0).unwrap_or_default()));
                }
                Ok(QuestionsInOrder(out))
            }
        }
        d.deserialize_map(V)
    }
}

#[derive(Deserialize)]
struct BodyKeys {
    #[serde(default)]
    questions: Option<QuestionsInOrder>,
}

/// Document order of the questions and of each question's criteria keys,
/// read from the raw body. A body whose `questions` is not a map yields no
/// order (the real parse refuses it with the useful message).
pub struct Order(Vec<(String, Vec<String>)>);

impl Order {
    pub fn read(raw: &[u8]) -> Self {
        Self(
            serde_json::from_slice::<BodyKeys>(raw)
                .ok()
                .and_then(|b| b.questions)
                .map(|q| q.0)
                .unwrap_or_default(),
        )
    }

    fn criteria(&self, qid: &str) -> Option<&[String]> {
        self.0
            .iter()
            .find(|(id, _)| id == qid)
            .map(|(_, keys)| keys.as_slice())
    }
}

/// `questions` as Jev sends them: a map of id -> {type, instructions,
/// criteria}. `noul` (also `bool`/`boolean`) takes an optional
/// `{true: .., false: ..}` description pair; `choice` a non-empty map of
/// option name -> description; `score` an ordered list of level names.
/// Extensions the example server reads too: `depends_on`, `ask_if`, `alone`.
pub fn parse_questions(body: &Map<String, Value>, order: &Order) -> Result<Vec<Question>, String> {
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
    // the caller's order; any id the order pass did not see keeps the map's
    let mut ids: Vec<&String> = order
        .0
        .iter()
        .filter_map(|(id, _)| qs.get_key_value(id).map(|(k, _)| k))
        .collect();
    for k in qs.keys() {
        if !ids.contains(&k) {
            ids.push(k);
        }
    }
    let mut out = Vec::with_capacity(qs.len());
    for qid in ids {
        out.push(parse_one(qid, &qs[qid], order)?);
    }
    // conditions point at questions that exist, never at themselves, and
    // test answers the named question can give
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

fn parse_one(qid: &str, q: &Value, order: &Order) -> Result<Question, String> {
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
            // the options in the caller's order: option A is the one written
            // first
            let mut names: Vec<String> = order
                .criteria(qid)
                .unwrap_or(&[])
                .iter()
                .filter(|k| c.contains_key(*k))
                .cloned()
                .collect();
            for k in c.keys() {
                if !names.contains(k) {
                    names.push(k.clone());
                }
            }
            let descriptions = names
                .iter()
                .map(|n| match &c[n] {
                    Value::Null => None,
                    Value::String(s) => Some(s.clone()),
                    other => Some(other.to_string()),
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
    let mut depends_on: Vec<String> = match q.get("depends_on") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => a
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
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Object(m)) => {
            let mut v = Vec::with_capacity(m.len());
            for (dep, vals) in m {
                let Some(list) = vals.as_array().filter(|a| !a.is_empty()) else {
                    return Err(format!(
                        "question {qid:?}: ask_if must map a question id to a non-empty list \
                         of its answers"
                    ));
                };
                let names: Vec<String> = list
                    .iter()
                    .map(|x| x.as_str().map_or_else(|| x.to_string(), str::to_owned))
                    .collect();
                v.push((dep.clone(), names));
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
    let alone = match q.get("alone") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(format!("question {qid:?}: alone must be true or false")),
    };
    Ok(Question {
        id: qid.to_owned(),
        kind,
        instructions,
        labels,
        names,
        descriptions,
        depends_on,
        ask_if,
        alone,
    })
}

/// Questions in stages: a question's stage comes after the stages of
/// everything it depends on, declaration order kept within a stage. A
/// dependency outside `qs` (not asked) counts as settled. Refuses a cycle.
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

/// The answer template's shape. Up to ten questions answer as `id: label`
/// lines, which read naturally; past that each id runs straight into its
/// label, space separated - three tokens a question instead of four or
/// five, which keeps a large schema in one read (the example server's
/// measurement: the two agreed on 42 booleans, ten 26-way choices and twenty
/// 5-level scores). With no id at all the labels lose which question they
/// answer past about twenty.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Lines,
    Indexed,
}

impl Format {
    pub fn for_count(n: usize) -> Self {
        if n <= 10 { Self::Lines } else { Self::Indexed }
    }

    pub fn join(self) -> &'static str {
        match self {
            Self::Lines => "\n",
            Self::Indexed => " ",
        }
    }

    fn lead(self, id: &str) -> String {
        match self {
            Self::Lines => format!("{id}: "),
            Self::Indexed => id.to_owned(),
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Self::Lines => {
                "Reply with one line per question, in this order, formatted as \"id: label\"."
            }
            Self::Indexed => {
                "Reply on one line with each question's id immediately followed by its label, \
                 separated by single spaces."
            }
        }
    }
}

/// The system turn: what to answer and in which form. The model reads its
/// own answer template, so the labels named here are the template's; an
/// option's name and description stand beside its letter.
pub fn system_text(
    qs: &[&Question],
    instructions: Option<&str>,
    fmt: Format,
    partial: bool,
) -> String {
    let mut s = String::from(
        "Answer a fixed set of questions about the state the user provides. Each question \
         lists its allowed answers; reply with exactly one label per question.\n",
    );
    if let Some(i) = instructions.map(str::trim).filter(|i| !i.is_empty()) {
        s.push('\n');
        s.push_str(i);
        s.push('\n');
    }
    s.push_str("\nQuestions:\n");
    for q in qs {
        s.push_str(&format!("- {}: {}\n", q.id, q.instructions.trim()));
        for (i, label) in q.labels.iter().enumerate() {
            let name = &q.names[i];
            match (&q.descriptions[i], q.kind) {
                (Some(d), _) if name != label => {
                    s.push_str(&format!("    {label} = {name}: {}\n", d.trim()))
                }
                (Some(d), _) => s.push_str(&format!("    {label}: {}\n", d.trim())),
                (None, Kind::Noul) => s.push_str(&format!("    {label}\n")),
                (None, _) => s.push_str(&format!("    {label} = {name}\n")),
            }
        }
    }
    s.push('\n');
    s.push_str(fmt.instruction());
    if partial {
        s.push_str(
            " A reply may cover only some of the questions; answer every line that is present.",
        );
    }
    s
}

/// The answer text for `qs` with `picks[i]` chosen for question i.
pub fn answer_text(qs: &[&Question], picks: &[usize], fmt: Format) -> String {
    qs.iter()
        .zip(picks)
        .map(|(q, &p)| format!("{}{}", fmt.lead(&q.id), q.labels[p]))
        .collect::<Vec<_>>()
        .join(fmt.join())
}

/// `qs`, in order, split into the fewest groups whose answer templates fit
/// `rows` canvas positions (`rows_of` measures a group); a question marked
/// `alone` gets a group of its own.
pub fn chunk_groups<'a>(
    qs: &[&'a Question],
    rows: usize,
    rows_of: impl Fn(&[&'a Question]) -> usize,
) -> Vec<Vec<&'a Question>> {
    let mut groups: Vec<Vec<&'a Question>> = Vec::new();
    let mut group: Vec<&'a Question> = Vec::new();
    for &q in qs {
        if q.alone {
            if !group.is_empty() {
                groups.push(std::mem::take(&mut group));
            }
            groups.push(vec![q]);
            continue;
        }
        let mut trial = group.clone();
        trial.push(q);
        if rows_of(&trial) > rows && !group.is_empty() {
            groups.push(std::mem::replace(&mut group, vec![q]));
        } else {
            group = trial;
        }
    }
    if !group.is_empty() {
        groups.push(group);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &str) -> Result<Vec<Question>, String> {
        let v: Value = serde_json::from_str(raw).unwrap();
        parse_questions(v.as_object().unwrap(), &Order::read(raw.as_bytes()))
    }

    #[test]
    fn questions_and_options_keep_the_order_they_were_written_in() {
        let qs = parse(
            r#"{"questions": {
                "urgent": {"type": "noul"},
                "about": {"type": "choice", "criteria": {"outage": "down", "billing": null, "feature": "new"}},
                "mood": {"type": "score", "criteria": ["calm", "annoyed", "furious"]}}}"#,
        )
        .unwrap();
        let ids: Vec<&str> = qs.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["urgent", "about", "mood"]);
        assert_eq!(qs[1].names, ["outage", "billing", "feature"]);
        assert_eq!(qs[1].labels, ["A", "B", "C"]);
        assert_eq!(qs[1].descriptions[1], None);
        assert_eq!(qs[2].labels, ["1", "2", "3"]);
    }

    #[test]
    fn conditions_stage_after_what_they_test_and_cycles_are_refused() {
        let qs = parse(
            r#"{"questions": {
                "kind": {"type": "choice", "criteria": {"bug": null, "question": null}},
                "severity": {"type": "score", "criteria": ["low", "high"], "ask_if": {"kind": ["bug"]}},
                "urgent": {"type": "noul", "depends_on": ["severity"]}}}"#,
        )
        .unwrap();
        assert_eq!(qs[1].depends_on, ["kind"]);
        let refs: Vec<&Question> = qs.iter().collect();
        let levels = schedule(&refs).unwrap();
        let names: Vec<Vec<&str>> = levels
            .iter()
            .map(|l| l.iter().map(|q| q.id.as_str()).collect())
            .collect();
        assert_eq!(names, [vec!["kind"], vec!["severity"], vec!["urgent"]]);
        assert!(parse(r#"{"questions": {"a": {"type": "noul", "depends_on": ["b"]}, "b": {"type": "noul", "depends_on": ["a"]}}}"#).is_err());
        assert!(parse(r#"{"questions": {"a": {"type": "noul", "ask_if": {"b": ["maybe"]}}, "b": {"type": "noul"}}}"#).is_err());
        assert!(
            parse(r#"{"questions": {"a": {"type": "noul", "depends_on": ["nope"]}}}"#).is_err()
        );
    }

    #[test]
    fn chunks_fill_to_the_row_budget_and_alone_reads_alone() {
        let qs = parse(
            r#"{"questions": {"a": {"type": "noul"}, "b": {"type": "noul"}, "c": {"type": "noul", "alone": true}, "d": {"type": "noul"}}}"#,
        )
        .unwrap();
        let refs: Vec<&Question> = qs.iter().collect();
        let groups = chunk_groups(&refs, 2, |g| g.len());
        let ids: Vec<Vec<&str>> = groups
            .iter()
            .map(|g| g.iter().map(|q| q.id.as_str()).collect())
            .collect();
        assert_eq!(ids, [vec!["a", "b"], vec!["c"], vec!["d"]]);
    }

    #[test]
    fn the_indexed_format_runs_ids_into_labels() {
        let qs = parse(r#"{"questions": {"a": {"type": "noul"}, "b": {"type": "noul"}}}"#).unwrap();
        let refs: Vec<&Question> = qs.iter().collect();
        assert_eq!(answer_text(&refs, &[0, 1], Format::Lines), "a: yes\nb: no");
        assert_eq!(answer_text(&refs, &[0, 1], Format::Indexed), "ayes bno");
    }
}
