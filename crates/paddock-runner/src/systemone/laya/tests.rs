//! The runner half of Laya's parity gate: the reference's own token
//! sequences and routing decisions (the reference run's output), rebuilt
//! here from the same requests. The engine half (`paddock-engine`'s
//! `gpu_laya_golden`) takes those sequences as given; this is what makes
//! them the ones a live request produces.
//!
//! Needs `LAYA_DIR` (the bundle, with `golden/reference.json`); without it
//! the test says so and passes, like every model-backed gate here.

use paddock_models::laya::{Checkpoint, LayaBundle};

use super::pyjson::{self, PyVal};
use super::question;
use super::sequence::{LayaTok, Prefix};

fn list(v: &PyVal) -> &[PyVal] {
    match v {
        PyVal::List(items) => items,
        _ => panic!("a list"),
    }
}

fn ints(v: &PyVal) -> Vec<u32> {
    list(v)
        .iter()
        .map(|x| match x {
            PyVal::Int(d) => d.parse().expect("an id"),
            _ => panic!("an integer"),
        })
        .collect()
}

#[test]
fn sequences_and_routes_match_the_reference() {
    let Some(dir) = std::env::var_os("LAYA_DIR").map(std::path::PathBuf::from) else {
        eprintln!("SKIP: LAYA_DIR is not set");
        return;
    };
    let text = std::fs::read_to_string(dir.join("golden").join("reference.json"))
        .expect("read golden/reference.json");
    let golden = pyjson::parse(&text).expect("parse reference.json");
    let bundle = LayaBundle::read(&dir).expect("read the bundle");
    let toks: Vec<(Checkpoint, LayaTok)> = bundle
        .checkpoints
        .iter()
        .map(|(c, cfg)| (*c, LayaTok::load(cfg).expect("tokenizer")))
        .collect();

    let mut checked = 0usize;
    for (ri, r) in list(golden.get("requests").expect("requests"))
        .iter()
        .enumerate()
    {
        let req = r.get("request").expect("request");
        let state = req.get("state").expect("state");
        let want_model = r
            .get("route")
            .and_then(|x| x.get("model"))
            .and_then(PyVal::as_str)
            .unwrap();
        let want_reason = r
            .get("route")
            .and_then(|x| x.get("reason"))
            .and_then(PyVal::as_str)
            .unwrap();
        let ck = match req.get("model").and_then(PyVal::as_str) {
            Some(m) => Checkpoint::parse(m).expect("a checkpoint"),
            None => {
                let (multi, reason, _) = super::lang::route_text(state);
                assert_eq!(reason, want_reason, "request {ri}: routing reason");
                if multi {
                    Checkpoint::Multilingual
                } else {
                    Checkpoint::English
                }
            }
        };
        assert_eq!(ck.name(), want_model, "request {ri}: checkpoint");
        let cfg = bundle.get(ck).expect("loaded");
        let tok = &toks.iter().find(|(c, _)| *c == ck).expect("tokenizer").1;
        let qs = question::parse_all(req.get("questions").expect("questions")).expect("valid");
        let state_ids = tok
            .encode(&match state {
                PyVal::Str(s) => s.clone(),
                other => other.dumps(),
            })
            .expect("encode");
        let items = list(r.get("items").expect("items"));
        assert_eq!(items.len(), qs.len(), "request {ri}: question count");
        for (q, it) in qs.iter().zip(items) {
            let p = Prefix::build(tok, q, cfg.head_max_len).expect("prefix");
            let room = p.room(cfg.max_len);
            // the reference cuts a state that does not fit (its last tokens
            // for a conversation list); the builder is checked on that form -
            // how the runner then windows instead is policy, not parity
            let st: &[u32] = if state_ids.len() <= room {
                &state_ids
            } else if matches!(state, PyVal::List(_)) {
                &state_ids[state_ids.len() - room..]
            } else {
                &state_ids[..room]
            };
            let ids = p.with_state(st, tok.sep);
            assert_eq!(
                ids,
                ints(it.get("ids").unwrap()),
                "request {ri} question {}: token ids",
                q.id
            );
            assert_eq!(
                p.markers,
                ints(it.get("markers").unwrap()),
                "request {ri} question {}: markers",
                q.id
            );
            let qt = match it.get("qtype") {
                Some(PyVal::Int(d)) => d.parse::<u32>().unwrap(),
                _ => panic!("qtype"),
            };
            assert_eq!(q.kind.qtype(), qt, "request {ri} question {}: type", q.id);
            checked += 1;
        }
    }

    let mut routed = 0usize;
    for (i, t) in list(golden.get("router").expect("router"))
        .iter()
        .enumerate()
    {
        let state = t.get("state").expect("state");
        let (multi, reason, _) = super::lang::route_text(state);
        let want = t.get("model").and_then(PyVal::as_str).unwrap();
        let got = if multi { "multilingual" } else { "english" };
        assert_eq!(
            (got, reason.as_str()),
            (want, t.get("reason").and_then(PyVal::as_str).unwrap()),
            "router text {i}: {}",
            state.dumps()
        );
        routed += 1;
    }
    eprintln!("laya parity: {checked} sequences bit-identical, {routed} router texts agree");
}
