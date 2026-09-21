use super::*;
use paddock_engine::{generator::MmAdmit, service::MmChunk};

#[test]
#[ignore = "requires Gemma Q4 target and canonical BF16 mmproj"]
fn cold_four_image_cohort_publishes_completions_and_preserves_slot_identity() {
    let model = std::env::var("PADDOCK_GEMMA4_GGUF").expect("target");
    let mm = std::env::var("PADDOCK_GEMMA4_MMPROJ").expect("vision");
    let mut m = Gemma4::load(Path::new(&model), 2048, 4, None).unwrap();
    m.attach_vision(Path::new(&mm)).unwrap();
    let request = vec![
        MmChunk::Text(vec![846; 100]),
        MmChunk::Image {
            rgb: vec![255; 768 * 768 * 3],
            w: 768,
            h: 768,
        },
        MmChunk::Text(vec![106, 105, 4368, 107]),
    ];
    assert!(
        m.admit_images((0..4).map(|s| (s, request.clone())).collect())
            .iter()
            .all(|(_, a)| matches!(a, MmAdmit::Encoding))
    );
    while m.encoding_pending() {
        assert!(
            m.step_images()
                .iter()
                .all(|(_, a)| matches!(a, MmAdmit::Queued))
        );
    }
    let mut ticks = 0;
    let mut completions = Vec::new();
    while completions.len() < 4 {
        let (_, done) = m.forward_mixed(&[], CHUNK).unwrap();
        ticks += 1;
        assert!(ticks < 10, "cold cohort stopped making progress");
        for &(slot, _, n) in &done {
            assert_eq!(
                m.cache[slot].history.len(),
                n,
                "completion was not published immediately"
            );
        }
        completions.extend(done);
    }
    let done = completions;
    let mut slots = done.iter().map(|r| r.0).collect::<Vec<_>>();
    slots.sort_unstable();
    assert_eq!(slots, [0, 1, 2, 3]);
    for (slot, logits, n) in done {
        assert_eq!(n, 362);
        assert_eq!(m.slots[slot].history.len(), n);
        assert_eq!(
            m.cache[slot].history.len(),
            n,
            "completion must publish immediately"
        );
        assert_eq!(logits.len(), m.vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
    assert!(m.pending.is_empty() && m.prefill_phase.is_none());
}

#[test]
#[ignore = "requires canonical Gemma 4 BF16 mmproj"]
fn gemma_vision_ragged_batch_and_yield_match_individual_encodes() {
    let path = std::env::var("PADDOCK_GEMMA4_MMPROJ").expect("mmproj");
    let device = MetalDevice::new(None).unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let width = map.gguf().metadata["clip.vision.projection_dim"]
        .as_u64()
        .unwrap() as usize;
    let vision = vision::Vision::load(&device, Path::new(&path), width).unwrap();
    let images = [(256, 256), (127, 317), (768, 768)].map(|(w, h)| {
        (
            (0..w * h * 3)
                .map(|i| ((i * 73 + i / 17) % 256) as u8)
                .collect::<Vec<_>>(),
            w,
            h,
        )
    });
    let finish = |mut job: vision::Job, budget: std::time::Duration| {
        loop {
            if let Some(v) = vision.step(&device, &mut job, budget).unwrap() {
                break v;
            }
        }
    };
    let individually = images
        .iter()
        .map(|(rgb, w, h)| {
            finish(
                vision.start(&device, &[(rgb, *w, *h)]).unwrap(),
                std::time::Duration::from_secs(1),
            )
            .remove(0)
        })
        .collect::<Vec<_>>();
    let refs = images
        .iter()
        .map(|(rgb, w, h)| (&**rgb, *w, *h))
        .collect::<Vec<_>>();
    let grouped = finish(
        vision.start(&device, &refs).unwrap(),
        std::time::Duration::ZERO,
    );
    for (i, (a, b)) in individually.iter().zip(&grouped).enumerate() {
        assert_eq!(a.tokens, b.tokens);
        let expected = unsafe { a.embd.read_f32(0, a.tokens * width) };
        let actual = unsafe { b.embd.read_f32(0, b.tokens * width) };
        let error = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("Gemma ragged image {i}: max_abs={error}");
        assert!(
            error < 0.0001,
            "ragged attention/projection changed image {i}: {error}"
        );
    }
    let before = device.allocated_bytes();
    let mut cancelled = vision.start(&device, &refs).unwrap();
    assert!(
        vision
            .step(&device, &mut cancelled, std::time::Duration::ZERO)
            .unwrap()
            .is_none()
    );
    drop(cancelled);
    assert_eq!(
        device.allocated_bytes(),
        before,
        "aborted encoder scratch leaked"
    );
}

#[test]
#[ignore = "requires Gemma Q4 target and canonical BF16 mmproj"]
fn gemma_multimodal_prefix_cancellation_and_decode_riders() {
    let model = std::env::var("PADDOCK_GEMMA4_GGUF").expect("target");
    let mm = std::env::var("PADDOCK_GEMMA4_MMPROJ").expect("vision");
    let mut m = Gemma4::load(Path::new(&model), 2048, 4, None).unwrap();
    m.attach_vision(Path::new(&mm)).unwrap();
    let chunks = |blue: bool| {
        vec![
            MmChunk::Text(vec![2, 105, 2364]),
            MmChunk::Image {
                rgb: (0..256 * 256)
                    .flat_map(|_| if blue { [0, 0, 255] } else { [255, 0, 0] })
                    .collect(),
                w: 256,
                h: 256,
            },
            MmChunk::Text(vec![106, 105, 4368, 107]),
        ]
    };
    let red = chunks(false);
    let blue = chunks(true);
    let (cold, n) = m.prefill_images(0, &red).unwrap();
    let (_, same) = m.prefill_images(1, &red).unwrap();
    assert_eq!(same, n);
    assert_eq!(m.take_prefill_reused(1), n - 1);
    assert!(m.image_cache_reuses() > 0);
    let (changed, _) = m.prefill_images(1, &blue).unwrap();
    assert!(m.take_prefill_reused(1) <= 4);
    assert!(
        cold.iter().zip(changed).any(|(a, b)| (a - b).abs() > 0.1),
        "image identity was ignored"
    );
    m.reset();
    let (serial, _) = m.prefill_images(0, &red).unwrap();
    m.reset();
    let mut tokens = vec![2, 105, 2364, 107, 846, 106, 105, 4368, 107];
    let mut next = m.forward_prefill(1, &tokens).unwrap();
    m.image_cache.clear();
    m.last_gpu_seconds = 1.;
    let result = m.admit_images(vec![(0, red.clone())]);
    assert!(matches!(result[0].1, MmAdmit::Encoding));
    // Force an exhausted wall quantum, independent of GPU clock speed, so
    // cancellation lands after patch setup and before tower completion.
    assert!(m.step_images().is_empty());
    assert!(m.prefill_abort(0));
    assert!(!m.image_slot_pending(0));
    m.last_gpu_seconds = 0.;
    assert!(matches!(
        m.admit_images(vec![(0, red)])[0].1,
        MmAdmit::Encoding
    ));
    let mut image_result = None;
    let mut ticks = 0;
    while image_result.is_none() {
        let pick = next
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        let (decode, done) = m
            .forward_mixed(&[(1, pick, tokens.len() as u32)], 128)
            .unwrap();
        tokens.push(pick);
        next = decode;
        for result in m.step_images() {
            assert!(!matches!(result.1, MmAdmit::Failed(_)));
        }
        if let Some((_, v, len)) = done.into_iter().find(|r| r.0 == 0) {
            assert_eq!(len, n);
            image_result = Some(v);
        }
        ticks += 1;
        assert!(ticks < 100);
    }
    let image_result = image_result.unwrap();
    let pick = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(pick(&serial), pick(&image_result));
    m.reset();
    let replay = m.forward_prefill(0, &tokens).unwrap();
    assert_eq!(pick(&next), pick(&replay));
    let mut long = chunks(false);
    long.insert(0, MmChunk::Text(vec![846; 1000]));
    let (_, length) = m.prefill_images(0, &long).unwrap();
    assert!(length > 1024);
    let (_, again) = m.prefill_images(2, &long).unwrap();
    assert_eq!(length, again);
    assert_eq!(m.take_prefill_reused(2), length - 1);
    m.prefill_abort(0);
    m.prefill_abort(2);
}
