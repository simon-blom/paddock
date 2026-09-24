//! Flash-Next QSA, rung R1 (the Flash-Next QSA design note):
//! the indexer's compressed key cache on the real checkpoint. Heavy:
//! `QWEN38FN_GGUF=<first shard>` names the file, as in gpu_qwen4exp_gguf.rs;
//! on a unified-memory box set `QWEN38FN_MOE_DEVICE=1`.
//!
//! The kernels are op-gated against the reference (gpu_qwen4exp_ops.rs);
//! this gates the WIRING: every walk shape the lane serves must leave the
//! same block keys - a prompt walked whole, the same tokens decoded one at a
//! time (the block a decode row closes reads its first keys off the ring
//! earlier walks filed), odd-sized chunked spans (blocks straddle spans), and
//! a prefix-cache resume (the side pages must hand back the keys they were
//! given, bit for bit). One test, two loads in sequence: a second resident
//! copy of this model does not fit the boxes it runs on.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b })
}

fn rel(x: &[f32], y: &[f32]) -> f32 {
    let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
    let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn load(
    exec: &std::sync::Arc<paddock_engine::gpu::GpuExecutor>,
    path: &std::path::Path,
) -> Qwen4ExpGpu {
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(exec, path, 4096, 3).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");
    m
}

/// Drive `slot`'s queued prompt to its end through riderless mixed ticks of
/// `budget` rows (spans that end mid-block).
fn chunk(m: &mut Qwen4ExpGpu, slot: usize, budget: usize) -> Vec<f32> {
    for _ in 0..10_000 {
        let (_, finished) = m.forward_mixed(&[], budget).expect("mixed tick");
        if let Some((_, l, _)) = finished.into_iter().find(|f| f.0 == slot) {
            return l;
        }
    }
    panic!("prefill never finished");
}

const HD: usize = 128;

/// The attention layers this checkpoint has (every 4th of 48).
fn attn_layers() -> Vec<usize> {
    (3..48).step_by(4).collect()
}

/// Per-block rel L2 of `got` against `want` ([blocks, HD] each).
fn block_rels(got: &[f32], want: &[f32]) -> Vec<f32> {
    got.chunks(HD)
        .zip(want.chunks(HD))
        .map(|(g, w)| rel(g, w))
        .collect()
}

#[test]
fn gguf_qsa_index_cache_agrees_across_walks() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    if !exec.has_qsa_indexer() {
        common::missing("pack has no QSA indexer kernels (rebuild packs/cuda)");
        return;
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let base = tok
        .encode(
            "The reference manual describes a distributed consensus protocol in which \
             every participant maintains a monotonically increasing term counter and \
             exchanges signed heartbeat messages over authenticated channels. ",
        )
        .expect("enc");
    // 221 tokens: 55 blocks close in the prompt, token 220 is the tail
    let a: Vec<u32> = base.iter().copied().cycle().take(221).collect();
    let layers = attn_layers();

    // ---- load A, prefix cache off: whole / decoded / chunked walks
    let full = {
        unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
        let mut m = load(&exec, &path);
        unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };
        // slot 0: the prompt, then three decoded tokens - the third closes
        // block 55 from 220 (the prompt walk's ring entry) and 221-222 (the
        // decode walks' ones)
        let l = m.prefill_slot(0, &a).expect("prefill");
        let mut toks = vec![argmax(&l) as u32];
        for _ in 0..2 {
            let l = m
                .decode_step_batch(&[(0, *toks.last().unwrap())])
                .expect("decode");
            toks.push(argmax(&l[0]) as u32);
        }
        m.decode_step_batch(&[(0, toks[2])]).expect("decode");
        assert_eq!(m.slot_position(0), 224);
        let mut full = a.clone();
        full.extend(&toks);
        // slot 1: the same 224 tokens in one walk; slot 2: in 37-row spans
        m.prefill_slot(1, &full).expect("prefill full");
        m.prefill_begin(2, full.clone()).expect("prefill_begin");
        chunk(&mut m, 2, 37);
        for &li in &layers {
            let k = |s| {
                m.qsa_index_keys(li, s, 0, 56)
                    .expect("read")
                    .expect("attention layer has an index cache")
            };
            let (k0, k1, k2) = (k(0), k(1), k(2));
            assert!(
                k0.iter().chain(&k1).chain(&k2).all(|v| v.is_finite()),
                "layer {li}: a written block key is not finite"
            );
            let (r0, r2) = (block_rels(&k0, &k1), block_rels(&k2, &k1));
            let worst = |r: &[f32]| r.iter().copied().fold(0.0f32, f32::max);
            // What a WRONG key looks like here: the distance between one
            // block's key and the next one's in the same walk - a pooling
            // source error (a stale ring entry, a neighbour's raw key, a span
            // boundary read from the wrong side) lands at that scale.
            let mut nb: Vec<f32> = k1
                .chunks(HD)
                .zip(k1.chunks(HD).skip(1))
                .map(|(x, y)| rel(y, x))
                .collect();
            nb.sort_by(f32::total_cmp);
            let median_nb = nb[nb.len() / 2];
            eprintln!(
                "layer {li:2}: decoded-vs-whole worst {:.2e} (block 55 {:.2e}), \
                 chunked-vs-whole worst {:.2e}; neighbouring blocks' median {median_nb:.2e}",
                worst(&r0),
                r0[55],
                worst(&r2)
            );
            // The walk geometry moves the activations under the keys: ~1e-2
            // at the first attention layer, then more with every layer (37-row
            // spans move this lane's logits ~2e-1 - the chunked gate's
            // control), until a span boundary's own drift is as big as a wrong
            // key's. The pooling is the same code in every layer, so the first
            // layer is where it is judged: every block - the one decode closed
            // off the ring, the ones straddling a span boundary - within a
            // tenth of the distance to a wrong key. Deeper layers are printed
            // for the record.
            if li == layers[0] {
                assert!(
                    worst(&r0).max(worst(&r2)) < 0.1 * median_nb,
                    "layer {li}: walk shapes disagree on block keys before the geometry could"
                );
            }
        }
        full
    };

    // ---- load B, prefix cache on: a resume hands the keys back
    let mut m = load(&exec, &path);
    m.prefill_slot(0, &a).expect("prefill");
    let resumed = {
        m.prefill_slot(1, &full).expect("prefill resumed");
        m.take_prefill_reused(1)
    };
    eprintln!("prefix resume at {resumed}");
    assert!(
        resumed >= 16,
        "the full prompt did not resume off the cached one"
    );
    let copied = resumed / 4;
    for &li in &layers {
        let k0 = m
            .qsa_index_keys(li, 0, 0, copied)
            .expect("read")
            .expect("cache");
        let k1 = m
            .qsa_index_keys(li, 1, 0, copied)
            .expect("read")
            .expect("cache");
        assert!(
            k0 == k1,
            "layer {li}: the resumed slot's first {copied} block keys are not the published ones"
        );
    }
}
