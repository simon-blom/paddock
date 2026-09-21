use super::*;

fn best(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0 as u32
}

#[test]
fn muse_dflash_window_matches_independent_gpu_attention() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    // Different slots, a ring wrap, crossing the window boundary, and a
    // ragged final block. Tile order deliberately differs from row order.
    let groups = [(2usize, 2557usize, 16usize), (0, 2038, 16), (3, 4090, 3)];
    let rows: usize = groups.iter().map(|r| r.2).sum();
    let (heads, kh, ring, stride) = (32usize, 2usize, 2560usize, 257usize);
    let q = d
        .upload(
            &(0..rows * heads * 128)
                .flat_map(|i| (((i * 7 + i / 128) % 19) as f32 / 256. - 9. / 256.).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let qhalf = d.alloc((rows + 32) * heads * 128 * 2).unwrap();
    let data = |mult| {
        (0..4 * ring * kh * 128)
            .flat_map(|i| {
                half::f16::from_f32(((i * mult + i / 128) % 31) as f32 / 128. - 15. / 128.)
                    .to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let k = d.upload(&data(3)).unwrap();
    let v = d.upload(&data(11)).unwrap();
    let pages = upload(
        &(0..4)
            .flat_map(|s| (0..stride).map(move |p| (s * ring / 16 + p % (ring / 16)) as u32))
            .collect::<Vec<_>>(),
    );
    let meta = upload(
        &groups
            .iter()
            .flat_map(|&(s, p, n)| (0..n).flat_map(move |r| [s as u32, (p + r) as u32]))
            .collect::<Vec<_>>(),
    );
    let bounded = upload(
        &groups
            .iter()
            .flat_map(|&(s, p, n)| (0..n).flat_map(move |_| [s as u32, (p + n - 1) as u32]))
            .collect::<Vec<_>>(),
    );
    let windows: Vec<_> = groups
        .iter()
        .flat_map(|&(_, p, n)| (0..n).map(move |r| p + n - (p + r + 1).saturating_sub(2048)))
        .collect();
    let tiles = upload(&[16, 16, 32, 3, 0, 16]);
    let selected: Vec<_> = (0..rows).map(|r| upload(&[r as u32])).collect();
    let out = d.alloc(rows * heads * 128 * 4).unwrap();
    let old = d.alloc(out.len()).unwrap();
    let expected = d.alloc(out.len()).unwrap();
    let parts = d.alloc(rows * heads * 130 * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "attention_query",
        &[&q, &qhalf],
        &[(heads * 128) as u32, 0, rows as u32],
        [((rows + 32) * heads * 128).div_ceil(256), 1, 1],
        256,
    );
    for (kernel, dst) in [("df_attention_muse", &out), ("df_attention", &old)] {
        cmd.dispatch(
            kernel,
            &[&qhalf, &k, &v, &meta, &pages, dst, &tiles],
            &[
                heads as u32,
                kh as u32,
                stride as u32,
                (1f32 / 128f32.sqrt()).to_bits(),
            ],
            [heads, groups.len(), 1],
            128,
        );
    }
    // The vector kernel has an independent softmax and contraction. Give
    // each query its exact [lower, block-end] interval, not a CPU oracle.
    for (r, selected) in selected.iter().enumerate() {
        cmd.dispatch(
            "muse_decode_vector",
            &[&q, &k, &v, &bounded, &pages, selected, &parts],
            &[
                heads as u32,
                kh as u32,
                stride as u32,
                windows[r] as u32,
                ring as u32,
                1,
            ],
            [kh, 1, 1],
            128,
        );
        cmd.dispatch(
            "muse_merge",
            &[&parts, &expected, selected],
            &[heads as u32, 1, 128],
            [heads, 1, 1],
            32,
        );
    }
    cmd.finish().unwrap();
    let actual = unsafe { out.read_f32(0, rows * heads * 128) };
    let expected = unsafe { expected.read_f32(0, actual.len()) };
    let old = unsafe { old.read_f32(0, actual.len()) };
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        actual.iter().all(|x| x.is_finite()) && error < 0.00001,
        "window error={error}"
    );
    assert!(
        old.iter()
            .zip(&expected)
            .any(|(a, b)| (a - b).abs() > 0.00001),
        "fixture did not distinguish block-end window trimming"
    );
}

#[test]
#[ignore = "requires canonical Muse target/drafter; diagnostic phase timings, not a serving benchmark"]
fn muse_dflash_phase_profile() {
    let target = std::env::var("PADDOCK_MUSE_GGUF").unwrap();
    let draft = std::env::var("PADDOCK_MUSE_DFLASH").unwrap();
    let map = MappedGguf::open(Path::new(&target)).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let ids = tok.encode("<|begin_of_text|><|start|>user<|message|>Write the numbers from one to twenty in order.<|eot|><|start|>assistant").unwrap();
    let mut m = Gemma4::load(Path::new(&target), 4096, 4, None).unwrap();
    m.attach_dflash(Path::new(&draft)).unwrap();
    for c in [1, 4] {
        m.reset();
        let mut pending = Vec::new();
        for slot in 0..c {
            let logits = m.forward_prefill(slot, &ids).unwrap();
            pending.push((slot, best(&logits)));
        }
        for round in 0..3 {
            let started = std::time::Instant::now();
            let drafts = m.dflash_draft(&pending, 15).unwrap().unwrap();
            let draft_ms = started.elapsed().as_secs_f64() * 1000.;
            let reqs: Vec<_> = pending
                .iter()
                .zip(drafts)
                .map(|(&(s, t), draft)| {
                    (
                        s,
                        m.slots[s].history.len(),
                        std::iter::once(t).chain(draft).collect(),
                    )
                })
                .collect();
            let started = std::time::Instant::now();
            let logits = m.verify(&reqs, false).unwrap();
            let verify_ms = started.elapsed().as_secs_f64() * 1000.;
            let gpu_ms = m.last_gpu_seconds * 1000.;
            let mut counts = Vec::new();
            for (i, (_, _, chunk)) in reqs.iter().enumerate() {
                let picks: Vec<_> = logits[i * 16 * m.vocab..(i + 1) * 16 * m.vocab]
                    .chunks(m.vocab)
                    .map(best)
                    .collect();
                let n = 1 + chunk[1..]
                    .iter()
                    .zip(&picks)
                    .take_while(|(a, b)| a == b)
                    .count();
                counts.push(n as u32);
                pending[i].1 = picks[n - 1];
            }
            m.commit_verify(&counts).unwrap();
            eprintln!(
                "DF_PHASE c={c} round={round} draft_ms={draft_ms:.3} verify_ms={verify_ms:.3} verify_gpu_ms={gpu_ms:.3} accepted={counts:?}"
            );
        }
    }
}

#[test]
#[ignore = "real Muse four-request verification budget election, not a serving comparison"]
fn muse_dflash_verify_depth_election() {
    let target = std::env::var("PADDOCK_MUSE_GGUF").unwrap();
    let draft = std::env::var("PADDOCK_MUSE_DFLASH").unwrap();
    let map = MappedGguf::open(Path::new(&target)).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let mut m = Gemma4::load(Path::new(&target), 4096, 4, None).unwrap();
    m.attach_dflash(Path::new(&draft)).unwrap();
    for repeats in [1, 80] {
        let prompt = format!(
            "<|begin_of_text|><|start|>user<|message|>{}Explain how computers work.<|eot|><|start|>assistant",
            "Computers process instructions and store information in memory. ".repeat(repeats)
        );
        let ids = tok.encode(&prompt).unwrap();
        m.reset();
        while m.evict() {}
        let mut times = [Vec::new(), Vec::new()];
        type Witness = (Vec<Vec<u32>>, Vec<Vec<u32>>);
        let mut witnesses: [Option<Witness>; 2] = [None, None];
        for round in 0..7 {
            for offset in 0..2 {
                let arm = (round + offset) % 2;
                let k = [7, 15][arm];
                // Restore the same published prompt before each arm. Never
                // compare a shorter speculative history with a longer one.
                let pending: Vec<_> = (0..4)
                    .map(|slot| (slot, best(&m.forward_prefill(slot, &ids).unwrap())))
                    .collect();
                let start = std::time::Instant::now();
                let drafts = m.dflash_draft(&pending, k).unwrap().unwrap();
                let draft_ms = start.elapsed().as_secs_f64() * 1000.;
                let reqs: Vec<_> = pending
                    .iter()
                    .zip(&drafts)
                    .map(|(&(slot, t), draft)| {
                        (
                            slot,
                            m.slots[slot].history.len(),
                            std::iter::once(t).chain(draft.iter().copied()).collect(),
                        )
                    })
                    .collect();
                let start = std::time::Instant::now();
                let picks = m.verify_picks(&reqs).unwrap();
                let verify_ms = start.elapsed().as_secs_f64() * 1000.;
                let gpu_ms = m.last_gpu_seconds * 1000.;
                if round > 0 {
                    let witness = (
                        drafts.iter().map(|d| d[..7].to_vec()).collect(),
                        picks.chunks(k + 1).map(|p| p[..8].to_vec()).collect(),
                    );
                    if let Some(old) = &witnesses[arm] {
                        assert_eq!(old, &witness, "same-shape verification did not reproduce");
                    }
                    witnesses[arm] = Some(witness);
                    times[arm].push(draft_ms + verify_ms);
                }
                // Close the transaction without publishing a speculative
                // continuation as the next arm's starting cache snapshot.
                m.commit_verify(&[1; 4]).unwrap();
                eprintln!(
                    "DF_DEPTH prompt={} round={round} k={k} draft_ms={draft_ms:.3} verify_ms={verify_ms:.3} verify_gpu_ms={gpu_ms:.3}",
                    ids.len()
                );
            }
        }
        assert_eq!(
            witnesses[0], witnesses[1],
            "short verification changed a common-prefix pick"
        );
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        eprintln!(
            "DF_DEPTH_MEDIAN prompt={} k7_ms={:.3} k15_ms={:.3} ratio={:.3}",
            ids.len(),
            (times[0][2] + times[0][3]) / 2.,
            (times[1][2] + times[1][3]) / 2.,
            (times[1][2] + times[1][3]) / (times[0][2] + times[0][3])
        );
    }
}

#[test]
#[ignore = "real Muse four-request short/full verification continuation parity"]
fn muse_dflash_short_full_batch_continuations() {
    const CAP: usize = 512;
    let target = std::env::var("PADDOCK_MUSE_GGUF").unwrap();
    let draft = std::env::var("PADDOCK_MUSE_DFLASH").unwrap();
    let map = MappedGguf::open(Path::new(&target)).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let mut m = Gemma4::load(Path::new(&target), 4096, 4, None).unwrap();
    m.attach_dflash(Path::new(&draft)).unwrap();
    let prompts: Vec<_> = [
        "Write the numbers from one to one hundred in order.",
        "Explain why the sky appears blue and sunsets often appear red.",
        "Write a Python function that merges two sorted lists. Explain how it works.",
        "What is 37 times 19? Show your calculation and check your answer.",
    ]
    .iter()
    .map(|p| {
        tok.encode(&format!(
            "<|begin_of_text|><|start|>user<|message|>{p}<|eot|><|start|>assistant"
        ))
        .unwrap()
    })
    .collect();
    let mut witnesses = Vec::new();
    for k in [15, 7] {
        m.reset();
        while m.evict() {}
        let mut pending: Vec<_> = prompts
            .iter()
            .enumerate()
            .map(|(slot, ids)| (slot, best(&m.forward_prefill(slot, ids).unwrap())))
            .collect();
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); 4];
        let mut histories = prompts.clone();
        let mut accepted = 0;
        while generated.iter().any(|g| g.len() < CAP) {
            let drafts = m.dflash_draft(&pending, k).unwrap().unwrap();
            let reqs: Vec<_> = pending
                .iter()
                .zip(&drafts)
                .map(|(&(slot, t), d)| {
                    (
                        slot,
                        m.slots[slot].history.len(),
                        std::iter::once(t).chain(d.iter().copied()).collect(),
                    )
                })
                .collect();
            let picks = m.verify_picks(&reqs).unwrap();
            let mut counts = Vec::new();
            for (i, ((slot, _, chunk), picks)) in reqs.iter().zip(picks.chunks(k + 1)).enumerate() {
                let matched = 1 + drafts[i]
                    .iter()
                    .zip(picks)
                    .take_while(|(a, b)| a == b)
                    .count();
                // Keep four rows live for the shape comparison, but stop
                // publishing long blocks for a slot whose witness is done.
                // This also bounds cache growth when acceptance is skewed.
                let n = matched.min(CAP.saturating_sub(generated[*slot].len()).max(1));
                counts.push(n as u32);
                histories[*slot].extend_from_slice(&chunk[..n]);
                generated[*slot].extend_from_slice(&chunk[..n]);
                pending[i].1 = picks[n - 1];
                accepted += n - 1;
            }
            m.commit_verify(&counts).unwrap();
            for (s, history) in m.slots.iter().zip(&histories) {
                assert_eq!(&s.history, history, "ragged commit changed slot history");
            }
        }
        assert!(accepted > 0);
        for g in &mut generated {
            g.truncate(CAP);
        }
        witnesses.push(generated);
        eprintln!("DF_DEPTH_CONTINUATION k={k} accepted={accepted}");
    }
    assert_eq!(witnesses[0], witnesses[1]);
    m.reset();
    while m.evict() {}
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
}

#[test]
#[ignore = "requires canonical Muse Q8 target and DFlash2 GGUF"]
fn muse_dflash_drafts_and_verified_continuations_match_dense_gpu() {
    let target = std::env::var("PADDOCK_MUSE_GGUF").expect("target");
    let draft = std::env::var("PADDOCK_MUSE_DFLASH").expect("drafter");
    let map = MappedGguf::open(Path::new(&target)).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let ids=tokenizer.encode("<|begin_of_text|><|start|>user<|message|>Write the numbers from one to twenty in order.<|eot|><|start|>assistant").unwrap();
    let mut m = Gemma4::load(Path::new(&target), 4096, 4, None).unwrap();
    m.attach_dflash(Path::new(&draft)).unwrap();
    assert!(m.attach_dflash(Path::new(&draft)).is_err());
    assert_eq!(m.spec_block_width(), Some(16));
    assert_eq!(m.spec_fixed_draft_depth(), Some(15));
    // Compare the same final-query kernel on both paths, not a full-prompt
    // matrix prefill against a one-row decode with different accumulation.
    let prefix = &ids[..ids.len() - 1];
    m.forward_prefill(0, prefix).unwrap();
    let logits = m
        .execute(&[(0, *ids.last().unwrap(), prefix.len() as u32)], &[0])
        .unwrap();
    let restored = m.forward_prefill(2, &ids).unwrap();
    assert_eq!(m.take_prefill_reused(2), prefix.len());
    assert!(logits == restored, "snapshot restore changed target logits");
    let snapshot = m
        .dflash_draft(&[(0, best(&logits)), (2, best(&logits))], 15)
        .unwrap()
        .unwrap();
    assert_eq!(
        snapshot[0], snapshot[1],
        "conditioning snapshot changed drafts"
    );
    let shorter = m
        .dflash_draft(&[(0, best(&logits)), (2, best(&logits))], 3)
        .unwrap()
        .unwrap();
    assert_eq!(
        shorter[0],
        snapshot[0][..3],
        "consumer cap changed the noncausal noise block"
    );
    assert_eq!(shorter[1], snapshot[1][..3]);
    m.prefill_abort(2);
    let mut next = best(&logits);
    let mut history = ids.clone();
    let mut generated = Vec::new();
    let mut accepted = 0;
    let mut proposed = 0;
    for round in 0..12 {
        let pos = history.len();
        let drafts = m.dflash_draft(&[(0, next)], 15).unwrap().unwrap().remove(0);
        assert_eq!(drafts.len(), 15);
        assert!(drafts.iter().all(|&t| (t as usize) < m.vocab));
        proposed += drafts.len();
        let chunk = std::iter::once(next).chain(drafts).collect::<Vec<_>>();
        let picks = m.verify_picks(&[(0, pos, chunk.clone())]).unwrap();
        let count = 1 + chunk[1..]
            .iter()
            .zip(&picks)
            .take_while(|(a, b)| a == b)
            .count();
        // Force different rejection points even when the drafter accepts:
        // no future row can survive through its cursor, history or cache.
        let count = if round < 3 {
            count.min(round + 1)
        } else {
            count
        };
        assert!(m.execute(&[(0, next, pos as u32)], &[0]).is_err());
        m.commit_verify(&[count as u32]).unwrap();
        assert!(m.commit_verify(&[1]).is_err());
        history.extend_from_slice(&chunk[..count]);
        accepted += count - 1;
        generated.extend_from_slice(&chunk[..count]);
        next = picks[count - 1];
        assert_eq!(m.slots[0].history, history);
        // Slot 2 is an independent dense target, never a CPU oracle.
        // Replay the common prompt, then each actually committed row.
        if round == 0 {
            m.forward_prefill(2, &ids).unwrap();
        }
        for (i, &token) in chunk[..count].iter().enumerate() {
            let dense = m.execute(&[(2, token, (pos + i) as u32)], &[0]).unwrap();
            assert_eq!(
                picks[i],
                best(&dense),
                "round={round} row={i} verified target drift"
            );
        }
        if generated.len() >= 80 {
            break;
        }
    }
    eprintln!(
        "MUSE_DFLASH proposed={proposed} accepted={accepted} generated={} text={}",
        generated.len(),
        tokenizer.decode(&generated, false).unwrap()
    );
    assert!(
        accepted > 0,
        "draft graph never predicted a verified continuation"
    );
    m.reset();
    while m.evict() {}
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
}
