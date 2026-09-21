//! Expensive state-machine tests use the real GGUF entirely on Metal. They
//! are opt-in so `cargo test` never downloads a checkpoint as a side effect.
use super::Granite;
use paddock_engine::generator::Generator;

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // MPP rounds tile operands to F16 while single-row GEMV accumulates the
    // dequantized operands in F32. Compare GPU paths without hiding NaNs.
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    assert!(error < 0.08, "GPU schedule max logit difference {error}");
    eprintln!("GPU schedule max logit difference: {error}");
}

#[test]
#[ignore = "requires PADDOCK_METAL_TEST_MODEL pointing to Granite Q8 GGUF and an M5"]
fn granite_prefix_holes_mixed_prefill_and_cancellation() {
    let path = std::env::var("PADDOCK_METAL_TEST_MODEL").expect("set model path");
    let mut model = Granite::load(std::path::Path::new(&path), 128, 3, None).unwrap();
    let p: Vec<u32> = (1..=49).collect();
    let baseline = model.forward_prefill(0, &p).unwrap();
    let next = model.forward(12366).unwrap();
    model.reset();

    model.prefill_begin(2, p.clone()).unwrap();
    let (decode, done) = model.forward_mixed(&[], 8).unwrap();
    assert!(decode.is_empty());
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].0, 2);
    assert_eq!(done[0].2, p.len());
    assert_eq!(model.take_prefill_reused(2), 48);
    close(&baseline, &done[0].1);
    let batch = model.forward_batch(&[0, 0, 12366], &[0, 0, 49]).unwrap();
    assert!(batch[..2 * model.vocab()].iter().all(|v| *v == 0.0));
    close(&next, &batch[2 * model.vocab()..]);

    let p2: Vec<u32> = (300..325).collect();
    model.prefill_begin(1, p2.clone()).unwrap();
    let (decode, done) = model.forward_mixed(&[(2, 13, 50)], 7).unwrap();
    assert_eq!(decode.len(), model.vocab());
    assert!(done.is_empty());
    assert!(model.prefill_abort(1));
    model.prefill_begin(1, p2).unwrap();
    let mut done = Vec::new();
    for _ in 0..4 {
        done.extend(model.forward_mixed(&[], 7).unwrap().1);
    }
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].2, 25);
    model.release_inactive_slots(&[false, false, false]);
    assert!(
        model
            .forward_batch(&[0, 0, 0], &[0, 0, 0])
            .unwrap()
            .iter()
            .all(|v| *v == 0.0)
    );
    assert!(model.forward_mixed(&[(3, 1, 0)], 1).is_err());
    assert!(model.forward_prefill(0, &vec![1; 129]).is_err());
}

fn top(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("nonempty vocabulary")
        .0 as u32
}

#[test]
#[ignore = "requires PADDOCK_METAL_TEST_MODEL pointing to Granite Q8 GGUF and an M5"]
fn granite_full_four_row_decode_and_selected_completion_heads() {
    let path = std::env::var("PADDOCK_METAL_TEST_MODEL").expect("set model path");
    let path = std::path::Path::new(&path);
    let prompts: Vec<Vec<u32>> = [(201, 238), (401, 426), (801, 842), (1001, 1034)]
        .into_iter()
        .map(|(a, b)| (a..b).collect())
        .collect();
    let mut model = Granite::load(path, 640, 4, None).unwrap();
    let mut reference = Vec::new();
    for prompt in &prompts {
        let logits = model.forward_prefill(0, prompt).unwrap();
        let next = model.forward(top(&logits)).unwrap();
        reference.push((logits, next));
    }
    drop(model);
    // Fresh physical/radix state: all four requests finish in the same cold
    // pass. The head must return four selected rows, in scheduler order.
    let mut model = Granite::load(path, 640, 4, None).unwrap();
    for (slot, prompt) in prompts.iter().enumerate() {
        model.prefill_begin(slot, prompt.clone()).unwrap();
    }
    let (decodes, done) = model.forward_mixed(&[], 512).unwrap();
    assert!(decodes.is_empty());
    assert_eq!(done.len(), 4);
    let mut tokens = Vec::new();
    for (slot, logits, n) in done {
        assert_eq!(n, prompts[slot].len());
        close(&reference[slot].0, &logits);
        assert_eq!(top(&reference[slot].0), top(&logits));
        tokens.push(top(&logits));
    }
    let positions: Vec<_> = prompts.iter().map(|p| p.len() as u32).collect();
    let logits = model.forward_batch(&tokens, &positions).unwrap();
    for (slot, row) in logits.chunks_exact(model.vocab()).enumerate() {
        close(&reference[slot].1, row);
        assert_eq!(top(&reference[slot].1), top(row));
    }
    model.release_inactive_slots(&[false; 4]);
    let long: Vec<u32> = (2000..2600).collect();
    model.prefill_begin(0, long.clone()).unwrap();
    let (decodes, done) = model.forward_mixed(&[], 512).unwrap();
    assert!(decodes.is_empty() && done.is_empty());
    let (decodes, done) = model.forward_mixed(&[], 512).unwrap();
    assert!(decodes.is_empty());
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].2, 600);
    let cached = model.forward_prefill(0, &long).unwrap();
    assert_eq!(model.take_prefill_reused(0), 592);
    close(&done[0].1, &cached);
    assert_eq!(top(&done[0].1), top(&cached));

    // Similar-sized long cold prompts advance together across ragged query
    // boundaries. Compare every completion with an independent serial GPU
    // pass; no logits may be delayed after its last prompt row is executed.
    let prompts: Vec<Vec<u32>> = [537, 552, 560, 569]
        .into_iter()
        .enumerate()
        .map(|(slot, n)| (0..n).map(|i| 4000 + (slot * 700 + i) as u32).collect())
        .collect();
    drop(model);
    let mut model = Granite::load(path, 640, 4, None).unwrap();
    let reference: Vec<_> = prompts
        .iter()
        .map(|p| model.forward_prefill(0, p).unwrap())
        .collect();
    drop(model);
    let mut model = Granite::load(path, 640, 4, None).unwrap();
    for (slot, p) in prompts.iter().enumerate() {
        model.prefill_begin(slot, p.clone()).unwrap();
    }
    for _ in 0..4 {
        let (decode, done) = model.forward_mixed(&[], 512).unwrap();
        assert!(decode.is_empty() && done.is_empty());
    }
    let (decode, done) = model.forward_mixed(&[], 512).unwrap();
    assert!(decode.is_empty());
    assert_eq!(done.len(), 4);
    for (slot, logits, n) in done {
        assert_eq!(n, prompts[slot].len());
        close(&reference[slot], &logits);
        assert_eq!(top(&reference[slot]), top(&logits));
    }
}
