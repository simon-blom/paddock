use super::*;
use paddock_engine::generator::Generator;

fn greedy(m: &FlashNext, rows: usize) -> Vec<u32> {
    let cmd = m.device.begin().unwrap();
    cmd.dispatch(
        "spec_argmax",
        &[&m.scratch.logits, &m.scratch.ids],
        &[VOCAB as u32],
        [rows, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    unsafe { m.scratch.ids.read_u32(rows) }
}

#[test]
fn flash_next_mlx_gpu_margin_ties_and_teacher_score() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    let mut values = vec![0f32; 1003];
    values[9] = 4.;
    values[777] = 4.;
    values[1002] = -2.;
    let x = d
        .upload(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let ids = d.alloc(8).unwrap();
    let scores = d.alloc(12).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4b_margin",
        &[&x, &ids, &scores],
        &[1003, 1002],
        [1, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { ids.read_u32(2) }, [9, 777]);
    assert_eq!(unsafe { scores.read_f32(0, 3) }, [4., 4., -2.]);
}

#[test]
#[ignore = "111 GB cost diagnostic; external watchdog required, counters are not serving timings"]
fn flash_next_mlx_prompt_cost() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    let prompt = reference["cases"][1]["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    eprintln!("MLX_COST prefill rows={}", prompt.len());
    m.prefill(0, &prompt).unwrap();
    for token in reference["cases"][1]["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .take(3)
    {
        eprintln!("MLX_COST decode");
        m.forward(token.as_u64().unwrap() as u32).unwrap();
    }
}

#[test]
#[ignore = "111 GB GPU logit diagnostic; external memory watchdog required"]
fn flash_next_mlx_teacher_forced_margins() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    // Both the retained 128-row gate and the wide-prefill candidate can be
    // checked against their actual same-checkpoint upstream chunk contract.
    let chunk = reference["chunk"].as_u64().unwrap() as usize;
    assert!([CHUNK, 512, MLX_CHUNK].contains(&chunk));
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    m.chunk = chunk;
    for case in reference["cases"].as_array().unwrap() {
        let vector = |key: &str| {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        };
        let prompt = vector("prompt_ids");
        let tokens = vector("token_ids");
        m.reset();
        let start = std::time::Instant::now();
        m.prefill(0, &prompt).unwrap();
        let prefill_s = start.elapsed().as_secs_f64();
        let mut rows = vec![];
        for (step, &token) in tokens.iter().enumerate() {
            let cmd = m.device.begin().unwrap();
            cmd.dispatch(
                "q4b_margin",
                &[&m.scratch.logits, &m.scratch.ids, &m.scratch.delta],
                &[VOCAB as u32, token],
                [1, 1, 1],
                256,
            );
            cmd.finish().unwrap();
            // Forward has completed; both control IDs and layer-output
            // scratch are dead until the next walk. Respect the exact grant.
            let actual = unsafe { m.scratch.ids.read_u32(2) };
            let values = unsafe { m.scratch.delta.read_f32(0, 3) };
            assert!(values.iter().all(|v| v.is_finite()));
            rows.push(serde_json::json!({"step":step, "native_ids":actual,
                "native_values":values, "teacher_id":token,
                "reference":case["logits"][step], "gpu_s":m.last_gpu_seconds}));
            // Always feed the reference token, even after a mismatch, so
            // subsequent margins compare the same token history on GPU.
            if step + 1 < tokens.len() {
                m.forward(token).unwrap();
            }
        }
        eprintln!(
            "MLX_MARGINS {}",
            serde_json::json!({"case":case["name"],
            "prefill_s":prefill_s, "rows":rows})
        );
    }
}

#[test]
#[ignore = "111 GB native MLX full-model GPU gate; run under external memory watchdog"]
fn flash_next_mlx_full_generation_control() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let source =
        std::fs::read_to_string(std::env::var("PADDOCK_FLASH_NEXT_MLX_CONTROL").unwrap()).unwrap();
    let reference: serde_json::Value = source
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|v| v.get("batch_prompt_ids").is_some())
        .expect("MLX-VLM GPU batch control");
    let vectors = |key: &str| {
        reference[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let prompts = vectors("batch_prompt_ids");
    let expected = vectors("serial_token_ids");
    let started = std::time::Instant::now();
    let mut m = FlashNext::load(Path::new(&path), 256, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    eprintln!(
        "MLX_NATIVE loaded_s={} resident_bytes={allocated}",
        started.elapsed().as_secs_f64()
    );
    let mut failures = vec![];
    for (i, prompt) in prompts.iter().enumerate() {
        m.reset();
        let start = std::time::Instant::now();
        m.prefill(0, prompt).unwrap();
        eprintln!(
            "MLX_NATIVE prefill case={i} wall_s={} gpu_s={}",
            start.elapsed().as_secs_f64(),
            m.last_gpu_seconds
        );
        let mut actual = vec![];
        for _ in 0..32 {
            let token = greedy(&m, 1)[0];
            actual.push(token);
            if [248044, 248046].contains(&token) {
                break;
            }
            m.forward(token).unwrap();
            eprintln!("MLX_NATIVE decode case={i} gpu_s={}", m.last_gpu_seconds);
        }
        eprintln!(
            "MLX_NATIVE serial case={i} elapsed_s={} tokens={actual:?} expected={:?}",
            start.elapsed().as_secs_f64(),
            expected[i]
        );
        if actual != expected[i] {
            failures.push(format!("serial {i}"));
        }
    }
    m.reset();
    // One ragged prefill walk, then compact only live decode rows. GPU
    // argmax samples the identical final logits used by the serving trait.
    for (slot, prompt) in prompts.iter().enumerate() {
        m.prepare(slot, prompt).unwrap();
    }
    let rows = prompts
        .iter()
        .enumerate()
        .flat_map(|(slot, prompt)| {
            prompt
                .iter()
                .enumerate()
                .map(move |(pos, &token)| (slot, token, pos as u32))
        })
        .collect::<Vec<_>>();
    let outputs = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| (r.2 as usize + 1 == prompts[r.0].len()).then_some(i))
        .collect::<Vec<_>>();
    m.execute(&rows, &outputs).unwrap();
    let mut tokens = greedy(&m, 4);
    let mut active = (0..4).collect::<Vec<_>>();
    let mut actual = vec![vec![]; 4];
    for _ in 0..32 {
        let mut rows = vec![];
        for (&slot, &token) in active.iter().zip(&tokens) {
            actual[slot].push(token);
            if ![248044, 248046].contains(&token) {
                rows.push((slot, token, m.slots[slot].length as u32));
            }
        }
        if rows.is_empty() {
            break;
        }
        active = rows.iter().map(|r| r.0).collect();
        m.execute(&rows, &(0..rows.len()).collect::<Vec<_>>())
            .unwrap();
        tokens = greedy(&m, rows.len());
    }
    eprintln!("MLX_NATIVE ragged tokens={actual:?}");
    if actual != expected {
        failures.push("ragged".into());
    }
    m.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        m.prefill_begin(slot, prompt.clone()).unwrap();
    }
    let mut decodes = vec![];
    let mut actual = vec![vec![]; 4];
    for _ in 0..40 {
        let (_, done) = m.forward_mixed(&decodes, 33).unwrap();
        let slots = decodes
            .iter()
            .map(|r| r.0)
            .chain(done.iter().map(|r| r.0))
            .collect::<Vec<_>>();
        decodes.clear();
        if !slots.is_empty() {
            let sampled = greedy(&m, slots.len());
            for (slot, token) in slots.into_iter().zip(sampled) {
                actual[slot].push(token);
                if ![248044, 248046].contains(&token) {
                    decodes.push((slot, token, m.slots[slot].length as u32));
                }
            }
        }
        if decodes.is_empty() && m.pending.is_empty() {
            break;
        }
    }
    eprintln!("MLX_NATIVE chunked_mixed tokens={actual:?}");
    if actual != expected {
        failures.push("chunked mixed".into());
    }
    m.reset();
    m.prefill_begin(3, prompts[0].repeat(5)).unwrap();
    m.forward_mixed(&[], 31).unwrap();
    assert!(m.prefill_abort(3));
    assert_eq!(m.slots[3].length, 0);
    m.prefill(3, &prompts[0]).unwrap();
    assert_eq!(
        greedy(&m, 1),
        vec![expected[0][0]],
        "cancelled slot must reset PLE/GDN/QSA state"
    );
    let before = m.slots[3].length;
    assert!(
        m.execute(&[(3, 123, before as u32), (0, VOCAB as u32, 0)], &[0])
            .is_err()
    );
    assert_eq!(m.slots[3].length, before);
    assert!(!m.poisoned);
    let free = m.pool.free_blocks();
    let logits = unsafe { m.scratch.logits.read_f32(0, VOCAB) };
    for contracts in [vec![], vec![0], vec![MLX_CHUNK + 1], vec![1, 1]] {
        assert!(
            m.execute_contracts(&[(3, 123, before as u32)], &[0], Some(&contracts))
                .is_err()
        );
        assert_eq!(m.slots[3].length, before);
        assert_eq!(m.pool.free_blocks(), free);
        assert!(!m.poisoned);
        assert_eq!(unsafe { m.scratch.logits.read_f32(0, VOCAB) }, logits);
    }
    assert_eq!(allocated, m.device.allocated_bytes());
    assert!(
        failures.is_empty(),
        "same-checkpoint MLX generation failures: {failures:?}"
    );
}

#[test]
#[ignore = "111 GB mixed decode replay; external memory watchdog required"]
fn flash_next_mlx_mixed_decode_preserves_serial_logits() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    let ids = |case: usize, key: &str| {
        reference["cases"][case][key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let prompt = ids(0, "prompt_ids");
    let neighbour = ids(1, "prompt_ids");
    let tokens = ids(0, "token_ids");
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    let mut serial = vec![m.prefill(0, &prompt).unwrap()];
    for &token in &tokens[..tokens.len() - 1] {
        serial.push(m.forward(token).unwrap());
    }
    m.reset();
    assert_eq!(m.prefill(0, &prompt).unwrap(), serial[0]);
    m.prefill_begin(1, neighbour.clone()).unwrap();
    m.prefill_begin(2, prompt.clone()).unwrap();
    let mut mismatches = Vec::new();
    for step in 1..tokens.len() {
        // Admission, completion, cancellation, and reuse of other slots must
        // not alter this sequence's cached state or projection contraction.
        if [11, 23, 47].contains(&step) {
            m.prefill_abort(1);
            m.prefill_begin(1, neighbour.clone()).unwrap();
        }
        let (actual, _) = m
            .forward_mixed(&[(0, tokens[step - 1], m.slots[0].length as u32)], CHUNK)
            .unwrap();
        if actual != serial[step] {
            mismatches.push(step);
        }
        assert_eq!(m.device.allocated_bytes(), allocated);
    }
    eprintln!(
        "MLX_MIXED_REPLAY positions={} vocabulary={} unequal_steps={mismatches:?}",
        tokens.len(),
        VOCAB
    );
    assert!(
        mismatches.is_empty(),
        "mixed prefill changed serial decode logits"
    );
}

#[test]
fn whole_walk_status_survives_reused_layer_scratch() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let flags = d.upload(&u32::MAX.to_le_bytes()).unwrap();
    let bad = d.upload(&0u32.to_le_bytes()).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch("q4x_status", &[&flags, &bad], &[1, 512, 2], [1, 1, 1], 256);
    // The next layer overwrites its scratch with a healthy value. The
    // accumulated whole-walk status must still retain the earlier failure.
    cmd.dispatch(
        "nemo_state_copy",
        &[&flags],
        &[1, 0, u32::MAX],
        [1, 1, 1],
        256,
    );
    cmd.dispatch("q4x_status", &[&flags, &bad], &[1, 0, 4], [1, 1, 1], 256);
    cmd.finish().unwrap();
    assert_eq!(unsafe { bad.read_u32(1) }, [2]);
    assert_eq!(unsafe { flags.read_u32(1) }, [0]);
}

#[test]
#[ignore = "111 GB logical prefill replay; external memory watchdog required"]
fn flash_next_mlx_sliced_prefill_preserves_serial_logits() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    let ids = |case: usize, key: &str| {
        reference["cases"][case][key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let prompt = ids(1, "prompt_ids");
    let neighbour = ids(0, "prompt_ids");
    let tokens = ids(1, "token_ids");
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    let mut serial = vec![m.prefill(0, &prompt).unwrap()];
    for &token in &tokens[..15] {
        serial.push(m.forward(token).unwrap());
    }
    m.reset();
    m.prefill_begin(0, prompt.clone()).unwrap();
    m.prefill_begin(1, neighbour.clone()).unwrap();
    let mut tick = 0;
    let first = loop {
        let budget = [1, 3, 8, 13, 32, 63, 125, 128][tick % 8];
        let (decodes, complete) = m.forward_mixed(&[], budget).unwrap();
        assert!(decodes.is_empty());
        assert_eq!(allocated, m.device.allocated_bytes());
        if tick == 3 {
            m.prefill_abort(1);
            m.prefill_begin(1, neighbour.clone()).unwrap();
        }
        if let Some((_, logits, used)) = complete.into_iter().find(|c| c.0 == 0) {
            assert_eq!(used, prompt.len());
            break logits;
        }
        tick += 1;
        assert!(tick < 128, "sliced prefill failed to make progress");
    };
    assert_eq!(
        first, serial[0],
        "sliced prompt changed full-vocabulary completion logits"
    );
    for (i, &token) in tokens[..15].iter().enumerate() {
        assert_eq!(
            m.forward(token).unwrap(),
            serial[i + 1],
            "sliced prompt changed recurrent/cache state at {i}"
        );
    }
    eprintln!(
        "MLX_SLICED_PREFILL prompt={} positions={} vocabulary={VOCAB} ticks={} exact=true",
        prompt.len(),
        serial.len(),
        tick + 1
    );
}

#[test]
fn checked_full_model_memory_bounds() {
    for (ctx, batch) in [
        (0, 1),
        (262145, 1),
        (1, 0),
        (1, 65),
        (usize::MAX, usize::MAX),
    ] {
        assert!(FlashNext::memory(ctx, batch).is_err());
        assert!(FlashNext::memory_rows(ctx, batch, MLX_CHUNK).is_err());
    }
    for rows in [0, MLX_CHUNK + 1, usize::MAX] {
        assert!(FlashNext::memory_rows(4096, 4, rows).is_err());
    }
    for (ctx, batch) in [(1, 1), (4096, 4), (262144, 64)] {
        let (c, s) = FlashNext::memory(ctx, batch).unwrap();
        eprintln!("Flash Next context={ctx} batch={batch} cache={c} scratch={s}");
        assert!(c > 0 && s > 0);
        let (mlx_cache, mlx_scratch) = FlashNext::memory_rows(ctx, batch, MLX_CHUNK).unwrap();
        assert_eq!(
            mlx_cache, c,
            "prefill capacity must not alter cache ownership"
        );
        assert!(mlx_scratch > s);
    }
}

#[test]
fn mlx_prefill_capacity_preserves_explicit_cache_budget() {
    let weight_bytes = 1 << 20;
    for expected in [CHUNK, 512, MLX_CHUNK] {
        let (cache, scratch) = FlashNext::memory_rows(256, 4, expected).unwrap();
        let scratch = scratch + super::super::affine::WORKSPACE_BYTES as u64;
        let budget = weight_bytes + cache + scratch;
        let (device, chunk, actual_cache, actual_scratch) =
            FlashNext::mlx_device(256, 4, weight_bytes, Some(budget)).unwrap();
        assert_eq!(
            (chunk, actual_cache, actual_scratch),
            (expected, cache, scratch)
        );
        assert_eq!(device.budget_bytes(), budget);
        assert_eq!(device.allocated_bytes(), 0);
    }
    assert!(FlashNext::mlx_device(256, 4, weight_bytes, Some(1)).is_err());
    assert!(FlashNext::mlx_device(256, 4, u64::MAX, None).is_err());
}

#[test]
#[ignore = "requires elected 82GB GGUF and M5 GPU; not independent model parity"]
fn full_walk_load_reset_mixed_cancel_and_fail_closed() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MODEL").expect("model");
    let mut m = FlashNext::load(Path::new(&path), 256, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    let (cache, scratch) = FlashNext::memory(256, 4).unwrap();
    assert_eq!(m.weight_bytes, 81_950_799_360);
    assert_eq!(allocated, m.weight_bytes + cache + scratch);
    let prompt = [
        17, 42, 88, 248044, 27, 100, 101, 33, 248044, 75, 61, 19, 71, 13, 10, 99, 31,
    ];
    let first = m.prefill(0, &prompt).unwrap();
    assert_eq!(first.len(), VOCAB);
    assert!(first.iter().all(|x| x.is_finite()));
    eprintln!(
        "full walk finite; allocated={allocated}, last_gpu_ms={}",
        m.last_gpu_seconds * 1000.
    );
    let next = m.forward(123).unwrap();
    m.reset();
    let repeated = m.prefill(0, &prompt).unwrap();
    assert_eq!(
        first, repeated,
        "same-shaped reset/replay must be byte identical"
    );
    assert_eq!(next, m.forward(123).unwrap());
    // Invalid later rows must not advance an earlier valid slot or mutate its
    // GPU caches. Replay continuation against the same-shaped clean walk.
    assert!(
        m.execute(&[(0, 124, 18), (1, VOCAB as u32, 0)], &[0])
            .is_err()
    );
    assert_eq!(m.slots[0].length, 18);
    assert!(!m.poisoned);
    let continuation = m.forward(124).unwrap();
    m.reset();
    m.prefill(0, &prompt).unwrap();
    m.forward(123).unwrap();
    assert_eq!(continuation, m.forward(124).unwrap());
    m.reset();
    for i in 0..4 {
        m.prefill_begin(i, prompt.to_vec()).unwrap();
    }
    assert!(m.prefill_begin(0, prompt.to_vec()).is_err());
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        let (decode, complete) = m.forward_mixed(&[], 33).unwrap();
        assert!(decode.is_empty());
        done.extend(complete);
    }
    assert_eq!(done.len(), 4);
    for (slot, logits, work) in done {
        assert_eq!(work, prompt.len());
        assert_eq!(m.slots[slot].length, prompt.len());
        let error = logits
            .iter()
            .zip(&first)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("full walk schedule diagnostic slot={slot} max_logit_diff={error}");
        // Diagnostic arithmetic tolerance only. Independent greedy generation
        // parity is a separate same-GGUF llama.cpp gate, never replaced by this.
        assert!(
            logits
                .iter()
                .zip(&first)
                .all(|(x, y)| x.is_finite() && (x - y).abs() <= 0.02 + 0.001 * y.abs())
        );
    }
    m.prefill_begin(2, vec![17; 129]).unwrap();
    let _ = m.forward_mixed(&[(0, 123, 17)], 31).unwrap();
    assert!(m.prefill_abort(2));
    assert_eq!(m.slots[2].length, 0);
    m.prefill_begin(2, prompt.to_vec()).unwrap();
    let (_, done) = m.forward_mixed(&[], 128).unwrap();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].0, 2);
    assert_eq!(done[0].1, first);
    assert_eq!(m.device.allocated_bytes(), allocated);
    m.release_inactive_slots(&[false; 4]);
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
    // Simulate a latched whole-walk failure, not a fabricated Metal error.
    m.poisoned = true;
    m.reset();
    m.prefill_abort(0);
    assert!(m.forward(17).unwrap_err().to_string().contains("poisoned"));
    assert!(
        m.prefill_begin(0, prompt.to_vec())
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );
}
