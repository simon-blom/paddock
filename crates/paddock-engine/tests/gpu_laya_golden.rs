//! Laya (ModernBERT + the decision head) against its own reference.
//!
//! There is no llama.cpp for a model like this, so the oracle is the model's
//! own package - a reference run drives Laya's `Router` and
//! `Agent` unmodified on the GPU and records, for a battery that walks every
//! sequence-building and routing branch, the exact token sequences it built
//! and the forward's raw outputs at fp32, fp16 autocast and bf16 autocast
//! (the last is what `laya-serve` runs by default). This gate feeds those
//! same sequences through the port.
//!
//! What "matching" means. The port is f16 weights (the checkpoint's own),
//! f16 GEMM operands and landings, f32 residual / norms / accumulate - fp16
//! autocast's class - so the target is fp32 within that class's noise. On
//! the battery fp16 autocast itself sits at 0.015 logit / 0.0013 probability
//! from fp32 and bf16 at 0.076 / 0.016. The gates:
//!   - every question's answer (argmax) equals fp32's. A wrong rope base, a
//!     window off by one, the GLU halves swapped or a norm with a phantom
//!     bias all load cleanly and land here near chance.
//!   - option logits within 0.05 of fp32 and the option distribution within
//!     0.005 - between fp16's noise and bf16's;
//!   - the act head's probabilities within 0.01.
//!
//! And one bit-exact leg: a request's outputs are a function of the request,
//! not of what else shared its pass.
//!
//! Needs the bundle under a model root as `laya/` (or `LAYA_DIR`) with the
//! reference at `laya/golden/reference.json`.

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::laya::{GpuLaya, LayaSeq, LayaWorkspace};
use paddock_models::laya::{Checkpoint, LayaBundle};

struct Item {
    ids: Vec<u32>,
    markers: Vec<u32>,
    qtype: u32,
}

struct Req {
    model: Checkpoint,
    items: Vec<Item>,
    /// per precision: per question, its option logits and act probabilities
    fp32: (Vec<Vec<f32>>, Vec<Vec<f32>>),
    fp16: Vec<Vec<f32>>,
    bf16: Vec<Vec<f32>>,
}

fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("a number list")
        .iter()
        .map(|x| x.as_f64().expect("a number") as f32)
        .collect()
}

fn rows(v: &serde_json::Value) -> Vec<Vec<f32>> {
    v.as_array()
        .expect("a list of rows")
        .iter()
        .map(f32s)
        .collect()
}

fn read(path: &std::path::Path) -> Vec<Req> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("read reference.json"))
            .expect("parse reference.json");
    v["requests"]
        .as_array()
        .expect("requests")
        .iter()
        .map(|r| {
            let model = Checkpoint::parse(r["route"]["model"].as_str().expect("route.model"))
                .expect("a checkpoint name");
            let items = r["items"]
                .as_array()
                .expect("items")
                .iter()
                .map(|it| Item {
                    ids: f32s(&it["ids"]).into_iter().map(|x| x as u32).collect(),
                    markers: f32s(&it["markers"]).into_iter().map(|x| x as u32).collect(),
                    qtype: it["qtype"].as_u64().expect("qtype") as u32,
                })
                .collect();
            let p = &r["precisions"];
            Req {
                model,
                items,
                fp32: (rows(&p["fp32"]["logits"]), rows(&p["fp32"]["act"])),
                fp16: rows(&p["fp16"]["logits"]),
                bf16: rows(&p["bf16"]["logits"]),
            }
        })
        .collect()
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

/// (max |d logit|, max |d prob|, argmax flips) of `ours` against `refs`.
fn diff(ours: &[&[f32]], refs: &[Vec<f32>]) -> (f32, f32, usize) {
    let (mut dl, mut dp, mut flips) = (0f32, 0f32, 0usize);
    for (a, b) in ours.iter().zip(refs) {
        assert_eq!(a.len(), b.len(), "option counts differ");
        for (x, y) in a.iter().zip(b) {
            dl = dl.max((x - y).abs());
        }
        for (x, y) in softmax(a).iter().zip(softmax(b)) {
            dp = dp.max((x - y).abs());
        }
        flips += usize::from(argmax(a) != argmax(b));
    }
    (dl, dp, flips)
}

fn say(msg: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{msg}");
}

#[test]
fn laya_reproduces_its_reference() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("LAYA_DIR", &["laya"]) else {
        return;
    };
    let golden = dir.join("golden").join("reference.json");
    if !golden.exists() {
        common::missing(&format!(
            "the Laya reference is not at {}",
            golden.display()
        ));
        return;
    }
    if !exec.has_text_encoder() {
        common::missing("this pack predates the text-encoder lane (slots 671-678)");
        return;
    }
    let reqs = read(&golden);
    let bundle = LayaBundle::read(&dir).expect("read the Laya bundle");
    let models: Vec<(Checkpoint, GpuLaya)> = bundle
        .checkpoints
        .iter()
        .map(|(c, cfg)| {
            (
                *c,
                GpuLaya::load(Arc::clone(&exec), cfg).expect("load a checkpoint"),
            )
        })
        .collect();
    let (d, wide) = models
        .iter()
        .map(|(_, m)| m.plane_dims())
        .fold((0, 0), |a, b| (a.0.max(b.0), a.1.max(b.1)));
    let mut ws = LayaWorkspace::new(&exec, d, wide, 16384, 256).expect("workspace");
    let model = |c: Checkpoint| &models.iter().find(|(k, _)| *k == c).expect("loaded").1;

    let mut worst = (0f32, 0f32, 0usize, 0f32);
    let mut outs = Vec::new();
    for (ri, r) in reqs.iter().enumerate() {
        let seqs: Vec<LayaSeq> = r
            .items
            .iter()
            .map(|it| LayaSeq {
                ids: &it.ids,
                markers: &it.markers,
                qtype: it.qtype,
            })
            .collect();
        let out = model(r.model).forward(&mut ws, &seqs).expect("forward");
        let ours: Vec<&[f32]> = (0..seqs.len())
            .map(|s| &out.logits[out.offsets[s]..out.offsets[s + 1]])
            .collect();
        let (dl, dp, flips) = diff(&ours, &r.fp32.0);
        let (dl16, _, _) = diff(&ours, &r.fp16);
        let (dlb, _, _) = diff(&ours, &r.bf16);
        let mut da = 0f32;
        for (s, want) in r.fp32.1.iter().enumerate() {
            for (a, x) in want.iter().enumerate() {
                da = da.max((out.act[s * out.n_act + a] - x).abs());
            }
        }
        say(&format!(
            "request {ri:2} ({:16} {:2} questions, {:4} tokens): vs fp32 logit {dl:.4} prob \
             {dp:.5} act {da:.4} flips {flips} | vs fp16 {dl16:.4} vs bf16 {dlb:.4}",
            r.model.name(),
            seqs.len(),
            out.rows
        ));
        worst = (
            worst.0.max(dl),
            worst.1.max(dp),
            worst.2 + flips,
            worst.3.max(da),
        );
        outs.push(out);
    }
    say(&format!(
        "worst vs fp32: logit {:.4}  prob {:.5}  act {:.4}  argmax flips {}",
        worst.0, worst.1, worst.3, worst.2
    ));
    assert_eq!(worst.2, 0, "answers that differ from fp32");
    assert!(worst.0 < 0.05, "logit max abs diff {}", worst.0);
    assert!(worst.1 < 0.005, "probability max abs diff {}", worst.1);
    assert!(worst.3 < 0.01, "act probability max abs diff {}", worst.3);

    // ---- batch invariance: every English request in ONE pass, bit-exact ----
    let english: Vec<usize> = (0..reqs.len())
        .filter(|&i| reqs[i].model == Checkpoint::English)
        .collect();
    let seqs: Vec<LayaSeq> = english
        .iter()
        .rev() // a different order, so no request sits where it sat alone
        .flat_map(|&i| {
            reqs[i].items.iter().map(|it| LayaSeq {
                ids: &it.ids,
                markers: &it.markers,
                qtype: it.qtype,
            })
        })
        .collect();
    let packed = model(Checkpoint::English)
        .forward(&mut ws, &seqs)
        .expect("packed forward");
    let mut s = 0usize;
    for &i in english.iter().rev() {
        let alone = &outs[i];
        for q in 0..reqs[i].items.len() {
            let a = &alone.logits[alone.offsets[q]..alone.offsets[q + 1]];
            let b = &packed.logits[packed.offsets[s]..packed.offsets[s + 1]];
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "request {i} question {q}: logits moved when packed with {} other sequences",
                seqs.len() - 1
            );
            let (aa, bb) = (
                &alone.act[q * alone.n_act..(q + 1) * alone.n_act],
                &packed.act[s * packed.n_act..(s + 1) * packed.n_act],
            );
            assert_eq!(aa, bb, "request {i} question {q}: act moved when packed");
            s += 1;
        }
    }
    say(&format!(
        "batch invariance: {} sequences / {} tokens in one pass, bit-identical to alone",
        seqs.len(),
        packed.rows
    ));
}
