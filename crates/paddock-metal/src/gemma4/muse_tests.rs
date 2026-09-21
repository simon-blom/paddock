use super::*;

#[test]
#[ignore = "M5 GPU decode shape election, not a serving comparison"]
fn muse_decode_shape_election() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let q = d
        .upload(
            &(0..4 * 32 * 128)
                .flat_map(|i| (((i * 11) % 127) as f32 / 37. - 1.5).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let n = 4 * 8192 * 2 * 128;
    let bytes = (0..n)
        .flat_map(|i| {
            half::f16::from_f32(((i * 7 + i / 96) % 251) as f32 / 93. - 1.25).to_le_bytes()
        })
        .collect::<Vec<_>>();
    let k = d.upload(&bytes).unwrap();
    let v = d.upload(&bytes).unwrap();
    let pages = upload(&(0..2048).map(|i| (i * 17 + 23) % 2048).collect::<Vec<_>>());
    let rows = upload(&[0, 1, 2, 3]);
    let meta = upload(&[0; 8]);
    let parts = d.alloc(4 * 32 * 32 * 130 * 4).unwrap();
    let out = d.alloc(4 * 32 * 128 * 4).unwrap();
    for c in [1usize, 4] {
        for length in [32usize, 128, 256, 512, 1024, 2048, 4096, 8192] {
            unsafe {
                meta.write_u32(
                    &(0..4)
                        .flat_map(|s| [s, (length - 1) as u32])
                        .collect::<Vec<_>>(),
                );
            }
            let splits = length
                .div_ceil(128)
                .max(16usize.div_ceil(c))
                .clamp(1, SPLITS);
            let p = [32, 2, 512, 0, 2560, splits as u32];
            let kernels = ["muse_decode_vector", "muse_decode", "muse_decode_register"];
            let mut timings = [0.; 3];
            for repeat in 0..4 {
                for offset in 0..3 {
                    let index = (repeat + offset) % 3;
                    let cmd = d.begin().unwrap();
                    for _ in 0..32 {
                        cmd.dispatch(
                            kernels[index],
                            &[&q, &k, &v, &meta, &pages, &rows, &parts],
                            &p,
                            [2, c, splits],
                            if index == 2 { 32 } else { 128 },
                        );
                        cmd.dispatch(
                            "muse_merge",
                            &[&parts, &out, &rows],
                            &[32, splits as u32, 128],
                            [c * 32, 1, 1],
                            32,
                        );
                    }
                    let seconds = cmd.finish().unwrap();
                    if repeat > 0 {
                        timings[index] += seconds / 96.;
                    }
                }
            }
            eprintln!(
                "MUSE_DECODE_SHAPE c={c} length={length} splits={splits} vector_us={} matrix_us={} register_us={}",
                timings[0] * 1e6,
                timings[1] * 1e6,
                timings[2] * 1e6
            );
        }
    }
}

#[test]
fn muse_matrix_decode_matches_vector_with_pages_rings_holes_and_empty_splits() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let q = d
        .upload(
            &(0..4 * 32 * 128)
                .flat_map(|i| (((i * 19 + i / 128) % 127) as f32 / 37. - 1.5).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    // Slot and packed-row order are independent. Slot 1 and packed row 3
    // deliberately remain unused; the 13-token row has empty tail splits.
    let meta = upload(&[2, 12, 0, 4096, 3, 2500, 1, 0]);
    let rows = upload(&[2, 0, 1]);
    let page_stride = 384;
    let pages = upload(
        &(0..4 * page_stride)
            .map(|i| (i * 17 + 23) % (4 * page_stride))
            .collect::<Vec<_>>(),
    );
    let n = 4 * page_stride as usize * 16 * 2 * 128;
    let values = |salt| {
        (0..n)
            .flat_map(|i| {
                half::f16::from_f32(((i * salt + i / 96) % 251) as f32 / 93. - 1.25).to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let k = d.upload(&values(7)).unwrap();
    let v = d.upload(&values(13)).unwrap();
    let parts = d.alloc(3 * 32 * 32 * 130 * 4).unwrap();
    let out = d.alloc(4 * 32 * 128 * 4).unwrap();
    for (window, splits) in [(0, 1), (0, 32), (2048, 1), (2048, 16), (2048, 32)] {
        let p = [32, 2, page_stride, window, 2560, splits];
        let execute = |kernel, repeats| {
            let cmd = d.begin().unwrap();
            for _ in 0..repeats {
                cmd.dispatch(
                    kernel,
                    &[&q, &k, &v, &meta, &pages, &rows, &parts],
                    &p,
                    [2, 3, splits as usize],
                    if kernel == "muse_decode_register" {
                        32
                    } else {
                        128
                    },
                );
                cmd.dispatch(
                    "muse_merge",
                    &[&parts, &out, &rows],
                    &[32, splits, 128],
                    [3 * 32, 1, 1],
                    32,
                );
            }
            let seconds = cmd.finish().unwrap();
            (
                unsafe { out.read_f32(0, 3 * 32 * 128) },
                seconds / repeats as f64,
            )
        };
        let (check, _) = execute("muse_decode_vector", 1);
        for kernel in ["muse_decode", "muse_decode_register"] {
            let (actual, _) = execute(kernel, 1);
            let max = actual
                .iter()
                .zip(&check)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                actual.iter().all(|v| v.is_finite()) && max < 0.00002,
                "kernel={kernel} window={window} splits={splits} max={max}"
            );
            let (_, old) = execute("muse_decode_vector", 16);
            let (_, new) = execute(kernel, 16);
            eprintln!(
                "MUSE_MATRIX kernel={kernel} window={window} splits={splits} max={max} vector_us={} candidate_us={}",
                old * 1e6,
                new * 1e6
            );
        }
    }
}

#[test]
#[ignore = "requires Muse Glimmer Q8_0 GGUF"]
fn muse_decoder_smoke_and_exact_prefix_restore() {
    let path = std::env::var("PADDOCK_MUSE_GGUF").expect("target");
    let map = MappedGguf::open(Path::new(&path)).unwrap();
    for (key, value) in &map.gguf().metadata {
        if key.starts_with("muse-glimmer.") {
            eprintln!("{key}={value:?}");
        }
    }
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let ids=tok.encode("<|begin_of_text|><|start|>user<|message|>What is the capital of France? Answer with the city name only.<|eot|><|start|>assistant").unwrap();
    let mut m = Gemma4::load(Path::new(&path), 8192, 4, None).unwrap();
    assert!(m.muse);
    assert_eq!(m.layers.len(), 52);
    assert!(m.layers.iter().all(|l| l.hd() == 128 && l.kh() == 2));
    let prefix = &ids[..ids.len() - 1];
    m.forward_prefill(0, prefix).unwrap();
    let mut logits = m
        .execute(&[(0, *ids.last().unwrap(), prefix.len() as u32)], &[0])
        .unwrap();
    let restored = m.forward_prefill(2, &ids).unwrap();
    assert_eq!(m.take_prefill_reused(2), prefix.len());
    assert_eq!(logits, restored, "prefix restore changed GPU state");
    let mut generated = Vec::new();
    for i in 0..80 {
        assert!(logits.iter().all(|v| v.is_finite()));
        let id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        generated.push(id);
        if id == 200008 {
            break;
        }
        logits = m.execute(&[(0, id, (ids.len() + i) as u32)], &[0]).unwrap();
        let other = m.execute(&[(2, id, (ids.len() + i) as u32)], &[0]).unwrap();
        assert_eq!(logits, other);
    }
    let text = tok.decode(&generated, false).unwrap();
    eprintln!("MUSE_GREEDY {text}");
    assert!(text.contains("Paris"));
    // The SWA snapshot must cover a wrapped physical ring, not only the
    // initial contiguous prefix. Recompute one identical final query on both.
    m.reset();
    while m.evict() {}
    let long: Vec<_> = ids.iter().copied().cycle().take(m.ring + 83).collect();
    m.forward_prefill(0, &long[..long.len() - 1]).unwrap();
    let logits = m
        .execute(&[(0, *long.last().unwrap(), (long.len() - 1) as u32)], &[0])
        .unwrap();
    let restored = m.forward_prefill(2, &long).unwrap();
    assert_eq!(m.take_prefill_reused(2), long.len() - 1);
    assert_eq!(logits, restored, "wrapped Muse prefix changed GPU state");
    m.reset();
    while m.evict() {}
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
}

#[test]
#[ignore = "requires Muse Q8 target and canonical BF16 mmproj"]
fn muse_multimodal_causal_chunks_cache_cancellation_and_decode_riders() {
    multimodal_riders(128, false);
}

#[test]
#[ignore = "requires Muse Q8 target and canonical BF16 mmproj"]
fn muse_wide_multimodal_phase_preserves_decode_riders_and_cache_ownership() {
    multimodal_riders(CHUNK, false);
}

#[test]
#[ignore = "requires Muse Q8 target, BF16 mmproj and DFlash2 GGUF"]
fn muse_dflash_encoder_deferral_preserves_riders_and_resumes_conditioning() {
    multimodal_riders(CHUNK, true);
}

#[test]
#[ignore = "requires Muse Q8 target, BF16 mmproj and DFlash2 GGUF"]
fn muse_large_image_phase_preserves_riders_and_dflash_conditioning() {
    multimodal_riders(super::IMAGE_CHUNK, true);
}

fn multimodal_riders(budget: usize, dflash: bool) {
    use paddock_engine::{generator::MmAdmit, service::MmChunk};
    let target = std::env::var("PADDOCK_MUSE_GGUF").expect("target");
    let tower = std::env::var("PADDOCK_MUSE_MMPROJ").expect("tower");
    let map = MappedGguf::open(Path::new(&target)).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let mut m = Gemma4::load(Path::new(&target), 4096, 4, None).unwrap();
    m.attach_vision(Path::new(&tower)).unwrap();
    if dflash {
        let draft = std::env::var("PADDOCK_MUSE_DFLASH").expect("drafter");
        m.attach_dflash(Path::new(&draft)).unwrap();
    }
    assert!(!m.spec_deferred());
    let prefix = tok
        .encode("<|begin_of_text|><|start|>user<|message|>What color is the image?")
        .unwrap();
    let suffix = tok.encode("<|eot|><|start|>assistant").unwrap();
    // The large-image case must cross the 1024-row projection election as
    // well as the 512-row text chunk; 768px alone only produces 784 tokens.
    let side = if budget == muse::IMAGE_CHUNK {
        1024
    } else {
        768
    };
    let chunks = |blue| {
        vec![
            MmChunk::Text(prefix.clone()),
            MmChunk::Image {
                rgb: (0..side * side)
                    .flat_map(|_| if blue { [0, 0, 255] } else { [255, 0, 0] })
                    .collect(),
                w: side,
                h: side,
            },
            MmChunk::Text(suffix.clone()),
        ]
    };
    let red = chunks(false);
    let expected = prefix.len() + suffix.len() + 2 + if side == 1024 { 1369 } else { 784 };
    let pick = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let (serial, n) = m.prefill_images(0, &red).unwrap();
    assert_eq!(n, expected, "Muse image was truncated to a prefill chunk");
    let (hot, n) = m.prefill_images(2, &red).unwrap();
    assert_eq!(n, expected);
    assert_eq!(m.take_prefill_reused(2), n - 1);
    assert!(m.image_cache_reuses() > 0);
    assert_eq!(pick(&serial), pick(&hot));
    let (changed, _) = m.prefill_images(2, &chunks(true)).unwrap();
    assert!(m.take_prefill_reused(2) <= prefix.len() + 1);
    assert!(
        serial
            .iter()
            .zip(changed)
            .any(|(a, b)| (a - b).abs() > 0.01)
    );
    m.reset();
    while m.evict() {}
    m.image_cache.clear();
    let text=tok.encode("<|begin_of_text|><|start|>user<|message|>Explain computer systems in detail.<|eot|><|start|>assistant").unwrap();
    let mut next = m.forward_prefill(1, &text).unwrap();
    m.forward_prefill(3, &text).unwrap();
    m.last_gpu_seconds = 1.;
    assert!(matches!(
        m.admit_images(vec![(0, red.clone())])[0].1,
        MmAdmit::Encoding
    ));
    assert_eq!(m.spec_deferred(), dflash);
    assert!(m.step_images().is_empty());
    assert!(m.prefill_abort(0));
    assert!(!m.image_slot_pending(0));
    assert!(
        !m.spec_deferred(),
        "cancelled encoder left speculation deferred"
    );
    m.last_gpu_seconds = 0.;
    assert!(matches!(
        m.admit_images(vec![(0, red)])[0].1,
        MmAdmit::Encoding
    ));
    let mut completed = false;
    // Instrumentation can exhaust each wall quantum. Even then each tick
    // finishes one tower/backbone layer: bound by work, not an assumed clock.
    let max_ticks = 52 + (expected.div_ceil(budget.min(CHUNK) - 1) + 1) * m.layers.len();
    for tick in 0..max_ticks {
        assert_eq!(m.spec_deferred(), dflash && m.encoding_pending());
        let id = pick(&next);
        let pos = (text.len() + tick) as u32;
        let (decode, done) = m.forward_mixed(&[(1, id, pos)], budget).unwrap();
        let isolated = m.execute(&[(3, id, pos)], &[0]).unwrap();
        assert_eq!(
            pick(&decode),
            pick(&isolated),
            "image work corrupted a text rider at tick {tick}"
        );
        next = decode;
        for (_, admit) in m.step_images() {
            assert!(!matches!(admit, MmAdmit::Failed(_)));
        }
        if let Some((slot, logits, n)) = done.into_iter().next() {
            assert_eq!((slot, n), (0, expected));
            assert_eq!(pick(&logits), pick(&serial));
            assert!(logits.iter().all(|v| v.is_finite()));
            completed = true;
            break;
        }
    }
    assert!(completed, "cold Muse image stopped making progress");
    assert!(
        !m.spec_deferred(),
        "finished encoder left speculation deferred"
    );
    if dflash {
        let proposals = m
            .dflash_draft(&[(1, pick(&next)), (3, pick(&next))], 15)
            .unwrap()
            .expect("dense interlude must leave the drafter ready to resume");
        assert_eq!(proposals[0].len(), 15);
        assert_eq!(
            proposals[0], proposals[1],
            "image yields corrupted rider conditioning"
        );
    }
    m.reset();
    while m.evict() {}
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
    assert!(!m.encoding_pending() && m.pending.is_empty() && m.prefill_phase.is_none());
}
