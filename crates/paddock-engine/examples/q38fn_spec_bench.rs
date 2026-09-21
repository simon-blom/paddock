//! Flash-Next decode / verify timing harness: one model load, then the pieces
//! a serving round is built from, each timed on its own and each inside its
//! own cuProfiler window - so
//!
//!   nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat \
//!       --cuda-graph-trace=node -t cuda target/release/examples/q38fn_spec_bench ...
//!
//! writes one report per phase, in phase order (prefill, decode, [eager],
//! then one per verify width).
//!
//! Usage:
//!   q38fn_spec_bench --gguf <shard 1> [--mtp <head gguf>] [--pack <so>]
//!       [--prompt 1024] [--maxctx 8192] [--ticks 64] [--eager]
//!       [--widths 2,3,4,6,8] [--rounds 24] [--agree N] [--code]
//!       [--golden FILE --depth K --steps 128 [--warm]]  (a cudafast track leg)
//!
//! `--code` runs every phase on a short code prompt instead of the repeated
//! prose (whose continuation is a copy, which flatters draft acceptance).
//!
//! `--agree N` (after the timing phases): greedy-decode N tokens plainly, then
//! the same prompt again through spec rounds at every width, and report where
//! each spec stream first leaves the plain one - the exactness gate a verify
//! change has to keep.
//!
//! A BARE-LOOP instrument (no HTTP, no scheduler): it prices kernels, graphs
//! and round shapes. Its numbers are never a serving cell.
// A development probe: stop where a failure happens rather than report it.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Instant;

use paddock_engine::generator::{Generator, RowSample};
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;
use paddock_engine::sampler::DevicePlan;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// The prompt the serving A/B (`cells.py`-style) uses, so a harness number and
/// a serving number describe the same text.
const PARA: &str = "The city of Stockholm is built on fourteen islands where Lake Malaren meets the Baltic Sea. Its old town keeps its medieval street plan, with narrow cobbled lanes, the royal palace, and the cathedral where Swedish monarchs were crowned. During the seventeenth century Sweden became a great power, and the capital grew with new districts, a navy yard, and the warship Vasa, which sank on its maiden voyage in 1628 and was raised in 1961. ";

/// A prompt whose continuation is not a copy of itself.
const CODE: &str = "Write a Python implementation of a thread-safe LRU cache with per-entry TTL expiry. Include type hints, docstrings, and a set of unittest test cases.";

fn window(on: bool) {
    // SAFETY: plain driver calls with no arguments; a failure only means the
    // profiler is absent, which is the normal un-profiled run
    unsafe {
        if on {
            let _ = cudarc::driver::sys::cuProfilerStart();
        } else {
            let _ = cudarc::driver::sys::cuProfilerStop();
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Row-exactness oracle: per-layer dumps (`PADDOCK_Q38FN_DUMP`) of two plain
/// decode ticks and of one 2-row verify over the same two tokens from the same
/// state, written to `<dir>/{dec0,dec1,ver}`. Verify row 0 must equal tick 0
/// bit for bit and row 1 tick 1; the comparison script names the first tensor
/// in walk order that does not.
fn exact_probe(m: &mut Qwen4ExpGpu, ids: &[u32], dir: &str) {
    // SAFETY: single-threaded probe; the engine reads the variable per walk
    let arm =
        |sub: &str| unsafe { std::env::set_var("PADDOCK_Q38FN_DUMP", format!("{dir}/{sub}")) };
    let disarm = || unsafe { std::env::remove_var("PADDOCK_Q38FN_DUMP") };
    // dumps read device memory mid-walk, which a captured replay cannot do
    m.set_graph_capture(false);
    let logits = m.forward_prefill(0, ids).unwrap();
    let t0 = argmax(&logits);
    let p = m.slot_position(0);
    arm("dec0");
    let t1 = tick(m, t0);
    disarm();
    arm("dec1");
    let t2 = tick(m, t1);
    disarm();
    // the same prompt again: the prefix cache resumes it bit-identically
    let logits = m.forward_prefill(0, ids).unwrap();
    assert_eq!(
        argmax(&logits),
        t0,
        "re-prefill must reproduce the first token"
    );
    assert_eq!(m.slot_position(0), p);
    arm("ver");
    let picks = m
        .forward_spec_batch(&[(0, p, vec![t0, t1])])
        .unwrap()
        .unwrap();
    disarm();
    println!(
        "exact: pos {p}, tokens t0={t0} t1={t1} t2={t2}; verify picks {picks:?} (decode gave [{t1}, {t2}])"
    );
}

/// The cudafast track's single-stream leg on one of its public goldens
/// (`--golden FILE --depth K [--steps N]`): the golden's raw prompt ids, the
/// prefill window up to the seed token, then an N-step decode window (128 on
/// the track), serial at depth 0 or through MTP rounds at depth K. Both windows
/// run COLD and once per process - the track loads the model once per leg and
/// runs its prompt once, so graph capture and first-touch costs land inside
/// the windows here exactly as they do there. A round that overshoots the
/// window is timed whole (the tokens are computed either way) and the output
/// is truncated to N. Every token is checked against the golden.
///
/// `--warm` adds one UNTIMED pass first on the prompt rotated by one token -
/// the same trigrams, so the same host-mapped n-gram table pages, and the
/// same walk widths and graph shapes, but no prefix the cache could resume the
/// timed prompt from - then resets the slot. That separates the engine from a
/// cold page cache (a download or another model evicts the table's pages, and
/// the prefill window then waits on the disk) without leaving the protocol's
/// one-prompt-per-leg shape.
fn golden_leg(
    m: &mut Qwen4ExpGpu,
    golden: &serde_json::Value,
    depth: usize,
    steps: usize,
    warm: bool,
) {
    let ids_of = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .expect("golden token list")
            .iter()
            .map(|t| t.as_u64().expect("token id") as u32)
            .collect()
    };
    let b = &golden["benchmark"];
    let ids = ids_of(&b["prefill_prompt_tokens"]);
    let want_seed = b["expected_prefill_token"]
        .as_u64()
        .expect("expected_prefill_token") as u32;
    let want = ids_of(&b["expected_decode_tokens"]);
    let steps = steps.min(want.len());

    if warm {
        let mut w = Vec::with_capacity(ids.len());
        w.push(ids[ids.len() - 1]);
        w.extend_from_slice(&ids[..ids.len() - 1]);
        let t = Instant::now();
        let logits = m.forward_prefill(0, &w).unwrap();
        let mut p = argmax(&logits);
        for _ in 0..8 {
            p = if depth == 0 {
                tick(m, p)
            } else {
                mtp_round(m, p, depth).0
            };
        }
        m.reset();
        println!(
            "golden    warm-up pass (untimed window): {:.1} s",
            t.elapsed().as_secs_f64()
        );
    }

    window(true);
    let t = Instant::now();
    let logits = m.forward_prefill(0, &ids).unwrap();
    let seed = argmax(&logits);
    let t_pre = t.elapsed().as_secs_f64();
    window(false);

    let mut out: Vec<u32> = Vec::with_capacity(steps + depth + 1);
    let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let mut pending = seed;
    window(true);
    let t = Instant::now();
    while out.len() < steps {
        if depth == 0 {
            pending = tick(m, pending);
            out.push(pending);
            continue;
        }
        let (next, emitted, d, a) = mtp_round(m, pending, depth);
        out.extend_from_slice(&emitted);
        pending = next;
        rounds += 1;
        drafted += d;
        accepted += a;
    }
    let t_dec = t.elapsed().as_secs_f64();
    window(false);
    let produced = out.len();
    out.truncate(steps);

    println!(
        "golden    prefill {} tok: {:8.1} ms  {:7.1} tok/s  seed {seed} ({})",
        ids.len(),
        t_pre * 1e3,
        ids.len() as f64 / t_pre,
        if seed == want_seed {
            "matches the golden".to_string()
        } else {
            format!("golden says {want_seed}")
        }
    );
    let spec = if depth == 0 {
        String::new()
    } else {
        format!(
            "  | {rounds} rounds, {:.2} tokens/round, {accepted}/{drafted} drafts accepted, {produced} tokens computed",
            produced as f64 / rounds.max(1) as f64
        )
    };
    println!(
        "golden    decode {steps} steps depth {depth}: {:8.1} ms  {:6.2} tok/s  {:.2} ms/token{spec}",
        t_dec * 1e3,
        steps as f64 / t_dec,
        t_dec * 1e3 / steps as f64
    );
    match out.iter().zip(&want).position(|(x, y)| x != y) {
        None => println!("golden    tokens: all {steps} match the golden"),
        Some(i) => println!(
            "golden    tokens: first difference at step {i} (engine {} vs golden {})",
            out[i], want[i]
        ),
    }
}

/// One MTP round at `depth`: draft, verify `[pending, drafts..]`, accept the
/// matching prefix. Returns (new pending, the tokens this round emitted, drafts
/// proposed, drafts accepted).
fn mtp_round(m: &mut Qwen4ExpGpu, pending: u32, depth: usize) -> (u32, Vec<u32>, usize, usize) {
    let pos = m.slot_position(0);
    let drafts = m
        .spec_draft_batch(&[(0, pending)], depth)
        .unwrap()
        .map(|mut d| d.remove(0))
        .unwrap_or_default();
    let mut chunk = vec![pending];
    chunk.extend(drafts.iter().copied().take(depth));
    let picks = m
        .forward_spec_batch(&[(0, pos, chunk.clone())])
        .unwrap()
        .unwrap();
    let mut a = 0usize;
    while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
        a += 1;
    }
    (picks[a], picks[..=a].to_vec(), chunk.len() - 1, a)
}

/// The prefix cache's in-walk checkpoints on the golden prompt: a cold
/// prefill; an exact re-send, which must prefill cold and reproduce the cold
/// logits bit for bit; and the prompt continued by the golden's first 32
/// decode tokens, which must resume inside the first walk's checkpoints and
/// predict the golden's next token.
fn resume_leg(m: &mut Qwen4ExpGpu, golden: &serde_json::Value) {
    let ids_of = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .expect("golden token list")
            .iter()
            .map(|t| t.as_u64().expect("token id") as u32)
            .collect()
    };
    let b = &golden["benchmark"];
    let ids = ids_of(&b["prefill_prompt_tokens"]);
    let want = ids_of(&b["expected_decode_tokens"]);
    let cold = m.forward_prefill(0, &ids).unwrap();
    let r0 = m.take_prefill_reused(0);
    println!(
        "resume    cold prefill: reused {r0}, seed {}",
        argmax(&cold)
    );
    let again = m.forward_prefill(0, &ids).unwrap();
    let r1 = m.take_prefill_reused(0);
    let same = cold.len() == again.len()
        && cold
            .iter()
            .zip(&again)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("resume    exact re-send: reused {r1} (want 0), logits bit-identical: {same}");
    let k = 32.min(want.len() - 1);
    let mut ext = ids.clone();
    ext.extend_from_slice(&want[..k]);
    let cont = m.forward_prefill(0, &ext).unwrap();
    let r2 = m.take_prefill_reused(0);
    println!(
        "resume    continuation +{k}: reused {r2}, next {} (golden {})",
        argmax(&cont),
        want[k]
    );
}

fn csv(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.trim().parse().unwrap()).collect()
}

/// One greedy decode tick through the serving entry (graph replay + device
/// sampling). Returns the next pending token.
fn tick(m: &mut Qwen4ExpGpu, pending: u32) -> u32 {
    let pos = m.slot_position(0) as u32;
    let step = m
        .forward_batch_sampled(&[pending], &[pos], &[RowSample::Device(DevicePlan::Greedy)])
        .unwrap();
    step.ids[0]
}

fn main() {
    let (mut gguf, mut mtp, mut pack) = (None, None, None);
    let (mut prompt, mut maxctx, mut ticks, mut rounds) = (1024usize, 8192usize, 64usize, 24usize);
    let mut widths = vec![2usize, 3, 4, 6, 8];
    let mut eager = false;
    let mut agree = 0usize;
    let mut code = false;
    let mut exact: Option<String> = None;
    let mut golden: Option<String> = None;
    let (mut depth, mut steps) = (0usize, 128usize);
    let mut warm = false;
    let mut resume_check = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = args.next(),
            "--mtp" => mtp = args.next(),
            "--pack" => pack = args.next(),
            "--prompt" => prompt = args.next().unwrap().parse().unwrap(),
            "--maxctx" => maxctx = args.next().unwrap().parse().unwrap(),
            "--ticks" => ticks = args.next().unwrap().parse().unwrap(),
            "--rounds" => rounds = args.next().unwrap().parse().unwrap(),
            "--widths" => widths = csv(&args.next().unwrap()),
            "--eager" => eager = true,
            "--agree" => agree = args.next().unwrap().parse().unwrap(),
            "--code" => code = true,
            "--exact" => exact = args.next(),
            "--golden" => golden = args.next(),
            "--depth" => depth = args.next().unwrap().parse().unwrap(),
            "--steps" => steps = args.next().unwrap().parse().unwrap(),
            "--warm" => warm = true,
            "--resume-check" => resume_check = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let gguf = gguf.expect("--gguf <first shard> required");
    let pack = pack
        .or_else(|| std::env::var("PADDOCK_PACK").ok())
        .expect("--pack or PADDOCK_PACK required");

    let ids: Vec<u32> = {
        let map = MappedGguf::open(std::path::Path::new(&gguf)).unwrap();
        let tok = GgufTokenizer::from_gguf(map.gguf()).unwrap();
        let mut text = String::from("(request bench) ");
        let mut ids = tok.encode(&text).unwrap();
        while ids.len() < prompt {
            text.push_str(PARA);
            ids = tok
                .encode(&(text.clone() + "\nSummarize the text above in detail."))
                .unwrap();
        }
        ids.truncate(prompt);
        if code {
            // non-repetitive text for the timing rounds: acceptance there
            // reflects drafting, not the model copying its own prompt
            ids = tok.encode(CODE).unwrap();
        }
        ids
    };

    let exec = Arc::new(GpuExecutor::new(0, std::path::Path::new(&pack)).unwrap());
    let t_load = Instant::now();
    let mut m =
        Qwen4ExpGpu::load_gguf_with_slots(&exec, std::path::Path::new(&gguf), maxctx, 1).unwrap();
    if let Some(p) = &mtp {
        m.attach_mtp(std::path::Path::new(p)).unwrap();
    }
    m.enable_batch(1).unwrap();
    eprintln!("loaded in {:.1} s", t_load.elapsed().as_secs_f64());
    if let Some(dir) = &exact {
        exact_probe(&mut m, &ids, dir);
        return;
    }
    if let Some(path) = &golden {
        assert!(depth == 0 || mtp.is_some(), "--depth > 0 needs --mtp");
        let g: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read --golden"))
                .expect("golden json");
        if resume_check {
            resume_leg(&mut m, &g);
            return;
        }
        golden_leg(&mut m, &g, depth, steps, warm);
        return;
    }

    // ---- prefill (cold: the slot is fresh and the cache holds nothing) ----
    window(true);
    let t = Instant::now();
    let logits = m.forward_prefill(0, &ids).unwrap();
    let dt = t.elapsed().as_secs_f64();
    window(false);
    println!(
        "prefill   {} tok: {:8.1} ms  {:7.1} tok/s{}",
        ids.len(),
        dt * 1e3,
        ids.len() as f64 / dt,
        if mtp.is_some() {
            "  (incl. MTP head seeding)"
        } else {
            ""
        }
    );
    let mut pending = argmax(&logits);

    // ---- decode ticks, graph-replayed (the first tick at a width captures) ----
    for _ in 0..4 {
        pending = tick(&mut m, pending);
    }
    window(true);
    let t = Instant::now();
    for _ in 0..ticks {
        pending = tick(&mut m, pending);
    }
    let dt = t.elapsed().as_secs_f64();
    window(false);
    println!(
        "decode    graph: {ticks} ticks {:7.2} ms/tick  {:6.2} tok/s  (pos {})",
        dt * 1e3 / ticks as f64,
        ticks as f64 / dt,
        m.slot_position(0)
    );

    if eager {
        m.set_graph_capture(false);
        pending = tick(&mut m, pending);
        window(true);
        let t = Instant::now();
        for _ in 0..ticks {
            pending = tick(&mut m, pending);
        }
        let dt = t.elapsed().as_secs_f64();
        window(false);
        println!(
            "decode    eager: {ticks} ticks {:7.2} ms/tick  {:6.2} tok/s",
            dt * 1e3 / ticks as f64,
            ticks as f64 / dt
        );
        m.set_graph_capture(true);
    }

    if mtp.is_none() {
        return;
    }
    // ---- spec rounds per verify width: real drafts from the head ----
    for &w in &widths {
        let k = w - 1;
        let round = |m: &mut Qwen4ExpGpu, pending: u32, draft_s: &mut f64, verify_s: &mut f64| {
            let pos = m.slot_position(0);
            let t = Instant::now();
            let drafts = m
                .spec_draft_batch(&[(0, pending)], k)
                .unwrap()
                .map(|mut d| d.remove(0))
                .unwrap_or_default();
            *draft_s += t.elapsed().as_secs_f64();
            let mut chunk = vec![pending];
            chunk.extend(drafts.iter().copied().take(k));
            let t = Instant::now();
            let picks = m
                .forward_spec_batch(&[(0, pos, chunk.clone())])
                .unwrap()
                .unwrap();
            *verify_s += t.elapsed().as_secs_f64();
            let mut a = 0usize;
            while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
                a += 1;
            }
            (picks[a], a + 1, chunk.len())
        };
        let (mut ds, mut vs) = (0f64, 0f64);
        for _ in 0..2 {
            pending = round(&mut m, pending, &mut ds, &mut vs).0;
        }
        let (mut ds, mut vs, mut committed, mut rows) = (0f64, 0f64, 0usize, 0usize);
        window(true);
        let t = Instant::now();
        for _ in 0..rounds {
            let (p, c, r) = round(&mut m, pending, &mut ds, &mut vs);
            pending = p;
            committed += c;
            rows += r;
        }
        let dt = t.elapsed().as_secs_f64();
        window(false);
        println!(
            "spec  w={w} (depth {k}): {rounds} rounds, {:.2} rows  draft {:6.2} ms  verify {:6.2} ms  \
             {:.2} tok/round  -> {:6.2} tok/s  (pos {})",
            rows as f64 / rounds as f64,
            ds * 1e3 / rounds as f64,
            vs * 1e3 / rounds as f64,
            committed as f64 / rounds as f64,
            committed as f64 / dt,
            m.slot_position(0)
        );
    }

    if agree == 0 {
        return;
    }
    // ---- greedy agreement: spec streams against the plain decode stream ----
    // Two prompts: the timing prompt (repetitive prose - its continuation is
    // largely a copy, so agreement there is weak evidence) and a short code
    // prompt that is not. Both legs re-prefill the same prompt; the prefix
    // cache resumes it, and a same-prompt resume is bit-identical to the cold
    // walk (the prefix gate).
    let code_ids: Vec<u32> = {
        let map = MappedGguf::open(std::path::Path::new(&gguf)).unwrap();
        let tok = GgufTokenizer::from_gguf(map.gguf()).unwrap();
        tok.encode(CODE).unwrap()
    };
    for (label, ids) in [("prose", &ids), ("code", &code_ids)] {
        let logits = m.forward_prefill(0, ids).unwrap();
        let mut p = argmax(&logits);
        let mut plain = vec![p];
        while plain.len() < agree {
            p = tick(&mut m, p);
            plain.push(p);
        }
        for &w in &widths {
            let k = w - 1;
            let logits = m.forward_prefill(0, ids).unwrap();
            let mut pending = argmax(&logits);
            let mut stream: Vec<u32> = Vec::with_capacity(agree + w);
            while stream.len() < agree {
                let pos = m.slot_position(0);
                let drafts = m
                    .spec_draft_batch(&[(0, pending)], k)
                    .unwrap()
                    .map(|mut d| d.remove(0))
                    .unwrap_or_default();
                let mut chunk = vec![pending];
                chunk.extend(drafts.iter().copied().take(k));
                let picks = m
                    .forward_spec_batch(&[(0, pos, chunk.clone())])
                    .unwrap()
                    .unwrap();
                let mut a = 0usize;
                while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
                    a += 1;
                }
                stream.extend_from_slice(&chunk[..=a]);
                pending = picks[a];
            }
            stream.truncate(agree);
            match plain.iter().zip(&stream).position(|(x, y)| x != y) {
                None => println!("agree {label} w={w} (depth {k}): all {agree} tokens identical"),
                Some(i) => println!(
                    "agree {label} w={w} (depth {k}): FIRST DIVERGENCE at token {i} of {agree} (plain {} vs spec {})",
                    plain[i], stream[i]
                ),
            }
        }
    }
}
