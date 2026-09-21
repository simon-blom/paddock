use super::*;
use paddock_engine::{
    generator::{Generator, MmAdmit},
    service::MmChunk,
};
const MODEL: &str = concat!(
    env!("HOME"),
    "/paddock/models/PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF.gguf"
);
const TOWER: &str = concat!(
    env!("HOME"),
    "/paddock/models/PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
);
fn model() -> PaddleOcr {
    PaddleOcr::load(
        Path::new(&std::env::var("PADDOCK_TEST_PADDLEOCR").unwrap_or_else(|_| MODEL.into())),
        4096,
        4,
        None,
    )
    .unwrap()
}
fn delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
#[test]
fn resize_checkpoint_rounding_and_budgets() {
    for (w, h, tw, th) in [
        (700, 500, 700, 504),
        (300, 900, 308, 896),
        (1400, 900, 1232, 784),
        (100, 80, 392, 308),
        (350, 350, 336, 336),
        (406, 406, 392, 392),
        (378, 378, 392, 392),
        (28, 28, 336, 336),
    ] {
        assert_eq!(vision::resize(w, h, 112896, 1003520).unwrap(), (tw, th));
    }
    assert_eq!(
        vision::resize(1400, 900, 112896, 1605632).unwrap(),
        (1400, 896)
    );
    for (w, h, min, max) in [
        (0, 1, 784, 1003520),
        (1, 0, 784, 1003520),
        (30000, 30, 784, 1003520),
        (30, 30, 0, 1003520),
        (30, 30, 1000, 999),
        (30, 30, 784, 1605633),
    ] {
        assert!(vision::resize(w, h, min, max).is_err());
    }
}
#[test]
#[ignore = "requires elected R2 BF16 checkpoint and Apple10 GPU"]
fn decoder_lifecycle_and_mixed_rows() {
    let mut m = model();
    assert!(PaddleOcr::load(Path::new(MODEL), 0, 4, None).is_err());
    assert!(PaddleOcr::load(Path::new(MODEL), 4096, 17, None).is_err());
    assert!(PaddleOcr::load(Path::new(MODEL), 4096, 4, Some(1024)).is_err());
    let tokens = (0..65).map(|i| 110 + (i * 7 % 137)).collect::<Vec<_>>();
    let full = m.forward_prefill(0, &tokens).unwrap();
    let bytes = m.device.allocated_bytes();
    m.reset();
    // No radix shortcut: execute each row through narrow SIMD attention.
    let mut scan = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        scan = m.execute(&[(0, token, i as u32)], &[0]).unwrap();
    }
    let diff = delta(&full, &scan);
    eprintln!("Paddle whole/scan logit max={diff}");
    assert!(diff < 0.05);
    m.reset();
    m.prefill_begin(0, tokens.clone()).unwrap();
    assert!(m.prefill_begin(0, tokens.clone()).is_err());
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 7).unwrap().1);
    }
    assert_eq!(done.len(), 1);
    assert!(delta(&full, &done[0].1) < 0.05);
    let seq = vec![127, 131, 137, 139, 149];
    let first = m.forward_prefill(1, &seq).unwrap();
    assert!(first.iter().all(|v| v.is_finite()));
    let dense = m.execute(&[(1, 151, 5)], &[0]).unwrap();
    m.forward_prefill(1, &seq).unwrap();
    m.prefill_begin(2, (0..127).map(|i| 150 + i % 19).collect())
        .unwrap();
    let mixed = m.forward_mixed(&[(1, 151, 5)], 128).unwrap().0;
    eprintln!("Paddle mixed logit max={}", delta(&dense, &mixed));
    assert!(delta(&dense, &mixed) < 0.05);
    assert!(m.prefill_abort(2));
    m.release_inactive_slots(&[false, false, false, false]);
    assert!(m.slots.iter().all(|s| s.history.is_empty()));
    assert_eq!(bytes, m.device.allocated_bytes());
    assert!(m.execute(&[(4, 1, 0)], &[0]).is_err());
    assert!(m.execute(&[(0, VOCAB as u32, 0)], &[0]).is_err());
    assert!(m.execute(&[(0, 1, 1)], &[0]).is_err());
    assert!(m.forward_batch(&[1], &[]).is_err());
}
fn image(w: usize, h: usize, seed: usize) -> Vec<MmChunk> {
    vec![
        MmChunk::Text(vec![100273, 100274, 42]),
        MmChunk::Image {
            rgb: (0..w * h * 3)
                .map(|i| ((i * 131 + seed) % 256) as u8)
                .collect(),
            w,
            h,
        },
        MmChunk::Text(vec![100276, 55, 100277]),
    ]
}
#[test]
#[ignore = "requires elected R2 decoder/tower and Apple10 GPU"]
fn vision_packed_lifecycle_and_cancellation() {
    let mut m = model();
    m.attach_vision(Path::new(TOWER)).unwrap();
    assert!(m.attach_vision(Path::new(TOWER)).is_err());
    let baseline = m.device.allocated_bytes();
    let chunks = image(336, 336, 7);
    let (single, n) = m.forward_prefill_multimodal(0, &chunks).unwrap();
    assert_eq!(n, 150);
    assert!(single.iter().all(|v| v.is_finite()));
    let p = m.slots[0].mm.as_ref().unwrap();
    assert_eq!(p.position(3), [3, 3, 3, 0]);
    assert_eq!(p.position(4), [3, 3, 4, 0]);
    assert_eq!(p.position(146), [3, 14, 14, 0]);
    assert_eq!(p.position(147), [15, 15, 15, 0]);
    assert_eq!(p.position(150), [18, 18, 18, 0]);
    let last = m.execute(&[(0, 100, 150)], &[0]).unwrap();
    m.reset();
    assert_eq!(m.device.allocated_bytes(), baseline);
    let wave = m.prefill_begin_multimodal(vec![
        (0, chunks.clone()),
        (1, image(336, 392, 11)),
        (2, image(392, 336, 19)),
        (3, chunks.clone()),
    ]);
    assert!(wave.iter().all(|(_, a)| matches!(a, MmAdmit::Encoding)));
    assert_eq!(m.encoding.len(), 1);
    m.encode_step();
    m.encode_step();
    assert!(m.prefill_abort(2));
    let mut queued = Vec::new();
    while m.encoding_pending() {
        queued.extend(m.encode_step());
    }
    assert_eq!(queued.len(), 3);
    assert!(
        queued
            .iter()
            .all(|(s, a)| *s != 2 && matches!(a, MmAdmit::Queued))
    );
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 127).unwrap().1);
    }
    assert_eq!(done.len(), 3);
    for (slot, logits, count) in &done {
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "nonfinite packed slot {slot}"
        );
        if *slot == 0 || *slot == 3 {
            assert_eq!(*count, n);
            eprintln!(
                "Paddle packed image slot={slot} logit max={}",
                delta(&single, logits)
            );
            assert!(delta(&single, logits) < 0.05);
        }
    }
    let current = m.execute(&[(0, 100, 150), (3, 100, 150)], &[0, 1]).unwrap();
    assert!(delta(&last, &current[..VOCAB]) < 0.05);
    assert!(delta(&last, &current[VOCAB..]) < 0.05);
    m.release_inactive_slots(&[]);
    assert_eq!(m.device.allocated_bytes(), baseline);
    // Image placeholders must never have entered the text radix.
    assert!(m.radix.match_prefix(&vec![IMAGE; 512]).is_empty());
    m.prefill_begin_multimodal(vec![(0, chunks)]);
    m.encode_step();
    m.reset();
    assert!(!m.encoding_pending());
    assert_eq!(m.device.allocated_bytes(), baseline);
    let invalid = vec![MmChunk::Image {
        rgb: vec![0; 3],
        w: 2,
        h: 2,
    }];
    assert!(matches!(
        m.prefill_begin_multimodal(vec![(0, invalid)])[0].1,
        MmAdmit::Failed(_)
    ));
}

#[test]
#[ignore = "requires elected tower and optional exact RGB fixture"]
fn rectangular_tower_finite() {
    let mut m = model();
    m.attach_vision(Path::new(TOWER)).unwrap();
    let mut chunks = image(392, 336, 11);
    if let Ok(path) = std::env::var("PADDOCK_PADDLE_RGB")
        && let MmChunk::Image { rgb, .. } = &mut chunks[1]
    {
        *rgb = std::fs::read(path).unwrap();
    }
    let (logits, _) = m.forward_prefill_multimodal(0, &chunks).unwrap();
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
#[ignore = "requires elected tower; maximum spotting budget and live decode"]
fn spotting_budget_and_encoder_decode_coexist() {
    let mut m = model();
    m.attach_vision(Path::new(TOWER)).unwrap();
    let base = m.device.allocated_bytes();
    m.forward_prefill(1, &[101, 102, 103]).unwrap();
    let expected = m.execute(&[(1, 104, 3)], &[0]).unwrap();
    m.forward_prefill(1, &[101, 102, 103]).unwrap();
    let mut chunks = image(1400, 1120, 71);
    chunks.insert(
        0,
        MmChunk::VisionPixels {
            min_pixels: Some(112896),
            max_pixels: Some(1605632),
        },
    );
    assert!(matches!(
        m.prefill_begin_multimodal(vec![(0, chunks)])[0].1,
        MmAdmit::Encoding
    ));
    m.encode_step();
    let actual = m.forward_mixed(&[(1, 104, 3)], 0).unwrap().0;
    assert_eq!(delta(&expected, &actual), 0.);
    while m.encoding_pending() {
        assert!(
            m.encode_step()
                .iter()
                .all(|(_, a)| matches!(a, MmAdmit::Queued))
        );
    }
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], CHUNK).unwrap().1);
    }
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].2, 2006);
    assert!(done[0].1.iter().all(|v| v.is_finite()));
    m.release_inactive_slots(&[]);
    assert_eq!(m.device.allocated_bytes(), base);
}
