//! Flash-Next speculation rounds (heavy: the real GGUF + the MTP head).
//!
//! `QWEN38FN_GGUF` = shard 1 of the model, `QWEN38FN_MTP` = the draft head
//! (e.g. `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`); skipped without either.

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

/// The HOST-SAMPLED verify round (`forward_spec_verify` + `spec_commit`) -
/// the round every tool-carrying request speculates through, since a
/// constraint keeps a slot out of the greedy and device rounds - against the
/// greedy device round (`forward_spec_batch`). Two slots walk one prompt cold,
/// so they start bit-identical; slot 0 then speculates through the greedy
/// round and slot 1 through the sampled one with the host taking each row's
/// argmax. They must draft the same, pick the same, commit the same, and
/// leave the same carried state (the next decode's logits, bit for bit). The
/// prompt is past the 2051-token window, so the verify walks attend through
/// QSA.
#[test]
fn gguf_sampled_verify_commits_what_the_greedy_round_does() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(mtp) = std::env::var_os("QWEN38FN_MTP").map(std::path::PathBuf::from) else {
        common::missing("QWEN38FN_MTP");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let base = tok
        .encode(
            "The dispatcher's log lists each courier, the parcels they carried, the depot \
             they left from and the time they signed the handover sheet. ",
        )
        .expect("encode");
    let p: Vec<u32> = base.iter().copied().cycle().take(2200).collect();

    // the prefix cache off: both slots walk the prompt cold, in one walk each
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 2).expect("load gguf");
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(3 << 30))
        .expect("cache");
    m.attach_mtp(&mtp).expect("attach mtp");
    let l0 = m.prefill_slot(0, &p).expect("prefill 0");
    let l1 = m.prefill_slot(1, &p).expect("prefill 1");
    assert!(l0 == l1, "the two slots do not start bit-identical");

    let depth = 3usize;
    let mut pend = [argmax(&l0) as u32; 2];
    let (mut out0, mut out1) = (Vec::new(), Vec::new());
    let mut accepted = 0usize;
    for round in 0..16 {
        let draft = |m: &mut Qwen4ExpGpu, slot: usize, pending: u32| -> Vec<u32> {
            let d = m
                .spec_draft_batch(&[(slot, pending)], depth)
                .expect("draft")
                .map(|mut d| d.remove(0))
                .unwrap_or_default();
            let mut chunk = vec![pending];
            chunk.extend(d.iter().copied().take(depth));
            chunk
        };
        // slot 0: the greedy device round
        let pos0 = m.slot_position(0);
        let c0 = draft(&mut m, 0, pend[0]);
        let picks = m
            .forward_spec_batch(&[(0, pos0, c0.clone())])
            .expect("greedy round")
            .expect("the greedy round declined");
        let mut a = 0usize;
        while a + 1 < c0.len() && c0[a + 1] == picks[a] {
            a += 1;
        }
        out0.extend_from_slice(&picks[..=a]);
        pend[0] = picks[a];
        accepted += a;

        // slot 1: the host-sampled round, the host taking each row's argmax
        let pos1 = m.slot_position(1);
        let c1 = draft(&mut m, 1, pend[1]);
        assert_eq!(c1, c0, "round {round}: the head drafted differently");
        let rows = m
            .forward_spec_verify(&[(1, pos1, c1.clone())])
            .expect("sampled round")
            .expect("the sampled round declined");
        let vocab = rows.len() / c1.len();
        let hp: Vec<u32> = rows.chunks(vocab).map(|r| argmax(r) as u32).collect();
        assert_eq!(
            hp, picks,
            "round {round}: the sampled round's rows pick differently"
        );
        let mut b = 0usize;
        while b + 1 < c1.len() && c1[b + 1] == hp[b] {
            b += 1;
        }
        m.spec_commit(&[(b + 1) as u32]).expect("commit");
        out1.extend_from_slice(&hp[..=b]);
        pend[1] = hp[b];
        assert_eq!(
            m.slot_position(1),
            m.slot_position(0),
            "round {round}: the two routes committed different rows"
        );
    }
    assert_eq!(out1, out0, "the two routes emitted different streams");
    assert!(
        accepted > 0,
        "no draft was ever accepted: nothing speculated"
    );
    // the carried state each route left, read through the next decode
    let n0 = m.decode_step_batch(&[(0, pend[0])]).expect("decode 0");
    let n1 = m.decode_step_batch(&[(1, pend[1])]).expect("decode 1");
    assert!(
        n0[0] == n1[0],
        "the two routes left different carried state"
    );
    eprintln!(
        "SAMPLED ROUND: 16 rounds at depth {depth} from position {}: {} tokens, {accepted} drafts \
         accepted; picks, commits and the next decode bit-identical to the greedy round",
        p.len(),
        out0.len()
    );
}
