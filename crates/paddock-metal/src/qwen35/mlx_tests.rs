use super::*;

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; packed recurrence and GQA old/new exact logits with alternating long cost"]
fn mlx_packed_kernels_long_execution() {
    mlx_long_prefill_execution(
        |baseline| {
            forward::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(baseline));
            attention::BASELINE_GQA_FOR_TEST.with(|v| v.set(baseline));
        },
        "MLX_PACKED_KERNELS_LONG",
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact packed recurrence long prefill/continuation and alternating GPU cost"]
fn mlx_packed_recurrent_long_execution() {
    mlx_long_prefill_execution(
        |baseline| forward::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_PACKED_RECURRENT_LONG",
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact packed recurrence across batch and small/large spans"]
fn mlx_packed_recurrent_model_execution() {
    mlx_projection_routes_preserve_model_execution(|baseline| {
        forward::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(baseline));
    });
}

#[test]
#[ignore = "requires target; rotate one projection family at a time under full-model cold-weight pressure"]
fn mlx_staging_family_cost() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            crate::affine::STAGING_MASK_FOR_TEST.with(|v| v.set(31));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let masks = [0, 1, 2, 4, 8, 16, 31];
    let prompt: Vec<_> = (1000..1128).collect();
    let rows: Vec<_> = (0..4)
        .flat_map(|slot| {
            prompt
                .iter()
                .enumerate()
                .map(move |(p, &t)| (slot, t, p as u32))
        })
        .collect();
    let mut expected = None;
    let mut times = vec![Vec::new(); masks.len()];
    for round in 0..8 {
        for i in 0..masks.len() {
            let route = (round + i) % masks.len();
            crate::affine::STAGING_MASK_FOR_TEST.with(|v| v.set(masks[route]));
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            for slot in 0..4 {
                model.prepare(slot, &prompt).unwrap();
            }
            let result = model.execute(&rows, &[127, 255, 383, 511]).unwrap();
            let bits: Vec<_> = result.iter().map(|v| v.to_bits()).collect();
            if let Some(reference) = &expected {
                assert!(&bits == reference, "mask={}", masks[route]);
            } else {
                expected = Some(bits);
            }
            if round > 0 {
                times[route].push(model.last_gpu_seconds * 1000.);
            }
        }
    }
    eprintln!(
        "MLX_STAGING_FAMILIES {}",
        serde_json::json!({"masks":masks,"gpu_ms":times})
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact long-prefill/continuation and alternating staging cost"]
fn mlx_wide_staging_long_execution() {
    mlx_long_prefill_execution(
        |baseline| crate::affine::BASELINE_STAGING_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_WIDE_STAGING_LONG",
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; same-shape staging election and full logits"]
fn mlx_wide_staging_model_execution() {
    mlx_projection_routes_preserve_model_execution(|baseline| {
        crate::affine::BASELINE_STAGING_FOR_TEST.with(|v| v.set(baseline))
    });
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact same-shape compact FFN comparison and GPU cost"]
fn mlx_compact_ffn_forward_cost() {
    mlx_projection_forward_cost(
        |baseline| crate::affine::BASELINE_FFN_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_COMPACT_FFN_COST",
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; full logits and continuations with compact FFN"]
fn mlx_compact_ffn_preserves_model_execution() {
    mlx_projection_routes_preserve_model_execution(|baseline| {
        crate::affine::BASELINE_FFN_FOR_TEST.with(|v| v.set(baseline))
    });
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; alternating same-shape GPU cost, NOT HTTP serving"]
fn mlx_shared_input_forward_cost() {
    mlx_projection_forward_cost(
        |baseline| crate::affine::SEPARATE_INPUTS_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_SHARED_INPUT_COST",
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; alternating same-shape GPU cost, NOT HTTP serving"]
fn mlx_bulk_store_forward_cost() {
    mlx_projection_forward_cost(
        |baseline| crate::affine::BASELINE_STORE_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_BULK_STORE_COST",
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; old/new long prefill equality and alternating cost, NOT HTTP serving"]
fn mlx_indexed_attention_long_execution() {
    mlx_long_prefill_execution(
        |baseline| attention::BASELINE_PREFILL_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_ATTENTION_LONG",
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact long prefill and continuations, alternating GPU cost"]
fn mlx_compact_ffn_long_execution() {
    mlx_long_prefill_execution(
        |baseline| crate::affine::BASELINE_FFN_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_COMPACT_FFN_LONG",
    );
}

fn mlx_long_prefill_execution(control: fn(bool), label: &str) {
    let _reset = ProjectionControlReset(control);
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let prompt: Vec<_> = (1000..4584).collect();
    for slots in [1, 4] {
        let mut gpu = [Vec::new(), Vec::new()];
        let mut wall = [Vec::new(), Vec::new()];
        let mut expected = None;
        for round in 0..3 {
            for route in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                control(route == 0);
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                for slot in 0..slots {
                    model.prepare(slot, &prompt).unwrap();
                }
                let started = std::time::Instant::now();
                let mut total_gpu = 0.;
                let n = 512 / slots;
                let mut output = Vec::new();
                for offset in (0..prompt.len()).step_by(n) {
                    let rows: Vec<_> = (0..slots)
                        .flat_map(|slot| {
                            (offset..offset + n)
                                .map(|pos| (slot, prompt[pos], pos as u32))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    let selected = if offset + n == prompt.len() {
                        (0..slots).map(|s| (s + 1) * n - 1).collect()
                    } else {
                        vec![]
                    };
                    output = model.execute(&rows, &selected).unwrap();
                    total_gpu += model.last_gpu_seconds;
                }
                gpu[route].push(total_gpu * 1000.);
                wall[route].push(started.elapsed().as_secs_f64() * 1000.);
                let mut logits = vec![output];
                for step in 0..4 {
                    logits.push(
                        model
                            .execute(
                                &(0..slots)
                                    .map(|slot| {
                                        (slot, 7000 + step as u32, (prompt.len() + step) as u32)
                                    })
                                    .collect::<Vec<_>>(),
                                &(0..slots).collect::<Vec<_>>(),
                            )
                            .unwrap(),
                    );
                }
                assert!(logits.iter().flatten().all(|v| v.is_finite()));
                let logits: Vec<Vec<u32>> = logits
                    .into_iter()
                    .map(|row| row.into_iter().map(f32::to_bits).collect())
                    .collect();
                if let Some(reference) = &expected {
                    assert!(
                        &logits == reference,
                        "{label} slots={slots} round={round} route={route}"
                    );
                } else {
                    expected = Some(logits);
                }
                eprintln!(
                    "{label}_SAMPLE slots={slots} round={round} route={route} gpu_ms={}",
                    gpu[route].last().unwrap()
                );
            }
        }
        eprintln!(
            "{label}_COST {}",
            serde_json::json!({
                "slots": slots, "prompt_tokens": prompt.len(), "wave_rows": 512,
                "routes": ["baseline", "candidate"], "gpu_ms": gpu, "wall_ms": wall,
            })
        );
    }
}

struct ProjectionControlReset(fn(bool));
impl Drop for ProjectionControlReset {
    fn drop(&mut self) {
        (self.0)(false);
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; admission arithmetic/cost diagnostic, not serving qualification"]
fn mlx_admission_arithmetic_cost() {
    let control: fn(bool) = |canonical| {
        crate::affine::BASELINE_COLD_CONTRACT_FOR_TEST.with(|v| v.set(!canonical));
    };
    let _reset = ProjectionControlReset(|_| {
        crate::affine::BASELINE_COLD_CONTRACT_FOR_TEST.with(|v| v.set(false))
    });
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let prompt: Vec<_> = (1000..1512).collect();
    for first in [32, 64, 128, 256, 512] {
        let mut gpu = [Vec::new(), Vec::new()];
        let mut reference = None;
        for round in 0..4 {
            for route in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                control(route == 1);
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                model.prepare(0, &prompt).unwrap();
                model
                    .execute(
                        &(0..first)
                            .map(|p| (0, prompt[p], p as u32))
                            .collect::<Vec<_>>(),
                        &[],
                    )
                    .unwrap();
                gpu[route].push(model.last_gpu_seconds * 1000.);
                let hidden: Vec<_> = unsafe { model.scratch.x.read_f32(0, first * model.width) }
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                if route == 0 {
                    reference = Some(hidden);
                } else {
                    let differences = hidden
                        .iter()
                        .zip(reference.as_ref().unwrap())
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!(
                        "MLX_ADMISSION_HIDDEN first={first} round={round} unequal={differences}"
                    );
                }
            }
        }
        eprintln!(
            "MLX_ADMISSION_COST {}",
            serde_json::json!({"rows":first,"routes":["baseline","canonical_prefill"],"gpu_ms":gpu})
        );
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; cold quantum arithmetic and GPU cost diagnostic, NOT a serving election"]
fn mlx_cold_quantum_diagnostic() {
    let control: fn(bool) =
        |canonical| crate::affine::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(canonical));
    let _reset = ProjectionControlReset(control);
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let initial_bytes = model.device.allocated_bytes();
    model.reserve_rows(2048).unwrap();
    let grown_bytes = model.device.allocated_bytes();
    let prompt: Vec<_> = (1000..4584).collect();
    for slots in [1, 4] {
        let mut reference: Option<Vec<f32>> = None;
        for (quantum, canonical) in [
            (512, false),
            (1024, false),
            (2048, false),
            (1024, true),
            (2048, true),
        ] {
            control(canonical);
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            for slot in 0..slots {
                model.prepare(slot, &prompt).unwrap();
            }
            let mut output = Vec::new();
            let mut gpu = 0.;
            let started = std::time::Instant::now();
            for offset in (0..prompt.len()).step_by(quantum / slots) {
                let end = (offset + quantum / slots).min(prompt.len());
                let rows: Vec<_> = (0..slots)
                    .flat_map(|slot| {
                        (offset..end)
                            .map(|pos| (slot, prompt[pos], pos as u32))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let selected = if end == prompt.len() {
                    (0..slots).map(|s| (s + 1) * (end - offset) - 1).collect()
                } else {
                    vec![]
                };
                output = model.execute(&rows, &selected).unwrap();
                gpu += model.last_gpu_seconds;
            }
            let wall = started.elapsed().as_secs_f64();
            let expected = reference.get_or_insert_with(|| output.clone());
            let differences = output
                .iter()
                .zip(expected.iter())
                .filter(|(a, b)| a != b)
                .count();
            let max_error = output
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "MLX_COLD_QUANTUM {}",
                serde_json::json!({
                    "slots":slots,"quantum":quantum,"canonical":canonical,"gpu_ms":gpu*1000.,"wall_ms":wall*1000.,
                    "logit_differences":differences,"max_error":max_error,"initial_bytes":initial_bytes,"grown_bytes":grown_bytes,
                })
            );
        }
    }
}

fn mlx_projection_forward_cost(control: fn(bool), label: &str) {
    mlx_projection_rows_cost(control, label, 512);
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact small-wave admission GPU cost"]
fn mlx_small_admission_cost() {
    for rows in [16, 32] {
        mlx_projection_rows_cost(
            |baseline| crate::affine::BASELINE_ADMISSION_FOR_TEST.with(|v| v.set(baseline)),
            "MLX_SMALL_ADMISSION_COST",
            rows,
        );
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; small admission routes preserve final logits and continuation"]
fn mlx_small_admission_model_execution() {
    mlx_projection_routes_preserve_model_execution(|baseline| {
        crate::affine::BASELINE_ADMISSION_FOR_TEST.with(|v| v.set(baseline))
    });
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact nearly-full admission-tail GPU cost"]
fn mlx_admission_tail_cost() {
    mlx_projection_rows_cost(
        |baseline| crate::affine::BASELINE_TAIL_FOR_TEST.with(|v| v.set(baseline)),
        "MLX_ADMISSION_TAIL_COST",
        480,
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; tail tile preserves complete logits and continuations"]
fn mlx_admission_tail_model_execution() {
    mlx_projection_routes_preserve_model_execution(|baseline| {
        crate::affine::BASELINE_TAIL_FOR_TEST.with(|v| v.set(baseline))
    });
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL and PADDOCK_METAL_ADMISSION_FIXTURE; deterministic full capped generations, not HTTP timing"]
fn mlx_admission_generation_replay() {
    mlx_generation_replay(false, false);
}

#[test]
#[ignore = "same fixture as admission replay; isolate fixed prefill arithmetic across arrival schedules, not a production election"]
fn mlx_admission_canonical_replay() {
    mlx_generation_replay(true, false);
}

#[test]
#[ignore = "requires target and admission fixture; old/new bulk staging must preserve all generation logits"]
fn mlx_wide_staging_generation_replay() {
    mlx_generation_replay(true, true);
}

fn mlx_generation_replay(canonical: bool, wide_staging: bool) {
    let control: fn(bool) = if wide_staging {
        |baseline| crate::affine::BASELINE_STAGING_FOR_TEST.with(|v| v.set(baseline))
    } else {
        |baseline| {
            crate::affine::BASELINE_ADMISSION_FOR_TEST.with(|v| v.set(baseline));
            crate::affine::BASELINE_TAIL_FOR_TEST.with(|v| v.set(baseline));
        }
    };
    let _reset = ProjectionControlReset(control);
    let _reset_canonical = ProjectionControlReset(|_| {
        crate::affine::BASELINE_COLD_CONTRACT_FOR_TEST.with(|v| v.set(false))
    });
    crate::affine::BASELINE_COLD_CONTRACT_FOR_TEST.with(|v| v.set(!canonical));
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let fixture = std::fs::read(std::env::var("PADDOCK_METAL_ADMISSION_FIXTURE").unwrap()).unwrap();
    let data: serde_json::Value = serde_json::from_slice(&fixture).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let stops = tokenizer.stop_ids();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    eprintln!("MLX_ADMISSION_REPLAY_FIXTURE {}", blake3::hash(&fixture));
    for count in [512, 3584] {
        let cell = data["cells"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["engine"] == "paddock-adaptive"
                    && c["round"] == 0
                    && c["c"] == 4
                    && c["prompt_tokens"] == count
                    && c["workload"] == "prose"
            })
            .unwrap();
        let prompts: Vec<_> = cell["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| tokenizer.encode(s.as_str().unwrap()).unwrap())
            .collect();
        assert!(prompts.iter().all(|p| p.len() == count));
        let mut together: Option<(Vec<Vec<u32>>, String)> = None;
        for late_peers in [false, true] {
            let mut reference = None;
            for &baseline in if canonical && !wide_staging {
                &[false][..]
            } else {
                &[true, false][..]
            } {
                control(baseline);
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                let mut traces = Vec::new();
                let mut gpu = Vec::new();
                model.prefill_begin(0, prompts[0].clone()).unwrap();
                if late_peers {
                    let (dec, done) = model.forward_mixed(&[], 512).unwrap();
                    assert!(dec.is_empty() && done.is_empty());
                    traces.push(vec![model.slots[0].history.len(), 0, 0, 0]);
                    gpu.push(model.last_gpu_seconds * 1000.);
                }
                for (slot, prompt) in prompts.iter().enumerate().skip(1) {
                    model.prefill_begin(slot, prompt.clone()).unwrap();
                }
                let mut output = vec![Vec::<u32>::new(); 4];
                let mut finished = [false; 4];
                let mut prefilled = [false; 4];
                let mut hash = blake3::Hasher::new();
                let mut ttft = [None; 4];
                let mut elapsed = gpu.iter().sum::<f64>();
                for tick in 0..1024 {
                    let decodes: Vec<_> = (0..4)
                        .filter(|&s| prefilled[s] && !finished[s])
                        .map(|s| {
                            (
                                s,
                                *output[s].last().unwrap(),
                                model.slots[s].history.len() as u32,
                            )
                        })
                        .collect();
                    let (dec, done) = model.forward_mixed(&decodes, 512).unwrap();
                    elapsed += model.last_gpu_seconds * 1000.;
                    gpu.push(model.last_gpu_seconds * 1000.);
                    traces.push(model.slots.iter().map(|s| s.history.len()).collect());
                    let results = decodes
                        .iter()
                        .enumerate()
                        .map(|(i, r)| (r.0, &dec[i * model.vocab..(i + 1) * model.vocab]))
                        .chain(done.iter().map(|(s, l, _)| (*s, l.as_slice())));
                    for (slot, logits) in results {
                        assert!(logits.iter().all(|v| v.is_finite()));
                        // F32 is plain initialized storage. Hash its bits, not a
                        // tolerance or rounded text, after GPU completion.
                        hash.update(unsafe {
                            std::slice::from_raw_parts(
                                logits.as_ptr().cast::<u8>(),
                                std::mem::size_of_val(logits),
                            )
                        });
                        let token = logits
                            .iter()
                            .enumerate()
                            .fold(0, |best, (i, v)| if *v > logits[best] { i } else { best })
                            as u32;
                        output[slot].push(token);
                        prefilled[slot] = true;
                        ttft[slot].get_or_insert(elapsed);
                        finished[slot] = stops.contains(&token) || output[slot].len() == 256;
                    }
                    if finished.iter().all(|v| *v) {
                        break;
                    }
                    assert!(tick < 1023, "replay made no bounded progress");
                }
                let digest = hash.finalize().to_hex().to_string();
                if let Some((expected, expected_hash, expected_trace)) = &reference {
                    assert!(
                        &output == expected,
                        "generation changed count={count}, late={late_peers}"
                    );
                    assert!(
                        &digest == expected_hash,
                        "logits changed count={count}, late={late_peers}"
                    );
                    assert!(&traces == expected_trace, "admission plan changed");
                } else {
                    reference = Some((output.clone(), digest.clone(), traces.clone()));
                }
                eprintln!(
                    "MLX_ADMISSION_REPLAY {}",
                    serde_json::json!({"prompt_tokens":count,"late_peers":late_peers,"baseline":baseline,"canonical_prefill":canonical,"wide_staging_control":wide_staging,"logits_blake3":digest,"tokens":output,"natural_stops":output.iter().filter(|v|stops.contains(v.last().unwrap())).count(),"gpu_ticks_ms":gpu,"gpu_first_output_ms":ttft,"history_trace":traces})
                );
                if !baseline {
                    if late_peers {
                        let (previous, previous_hash) = together.as_ref().unwrap();
                        eprintln!(
                            "MLX_ADMISSION_ARRIVAL_PARITY count={count} matched={}/4",
                            output.iter().zip(previous).filter(|(a, b)| a == b).count()
                        );
                        if canonical {
                            assert!(
                                output == *previous,
                                "canonical arrival generations differ at {count}"
                            );
                            assert!(
                                digest == *previous_hash,
                                "canonical arrival logits differ at {count}"
                            );
                        }
                    } else {
                        together = Some((output, digest));
                    }
                }
            }
        }
    }
}

fn mlx_projection_rows_cost(control: fn(bool), label: &str, total_rows: usize) {
    let _reset = ProjectionControlReset(control);
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    for slots in [1, 4] {
        let n = total_rows / slots;
        let prompt: Vec<_> = (1000..1000 + n as u32).collect();
        let rows: Vec<_> = (0..slots)
            .flat_map(|slot| {
                prompt
                    .iter()
                    .enumerate()
                    .map(move |(pos, &t)| (slot, t, pos as u32))
            })
            .collect();
        let selected: Vec<_> = (0..slots).map(|s| (s + 1) * n - 1).collect();
        let mut gpu = [Vec::new(), Vec::new()];
        let mut wall = [Vec::new(), Vec::new()];
        let mut reference = None;
        for round in 0..8 {
            for route in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                control(route == 0);
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                for slot in 0..slots {
                    model.prepare(slot, &prompt).unwrap();
                }
                let started = std::time::Instant::now();
                let output = model.execute(&rows, &selected).unwrap();
                let elapsed = started.elapsed().as_secs_f64();
                if let Some(expected) = &reference {
                    assert!(
                        &output == expected,
                        "slots={slots}, round={round}, route={route}"
                    );
                } else {
                    reference = Some(output);
                }
                if round > 0 {
                    gpu[route].push(model.last_gpu_seconds * 1000.);
                    wall[route].push(elapsed * 1000.);
                }
            }
        }
        eprintln!(
            "{label} {}",
            serde_json::json!({
                "slots": slots, "total_rows": total_rows,
                "routes": ["baseline", "candidate"], "gpu_ms": gpu, "wall_ms": wall,
            })
        );
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; old/new input preparation at identical execution shapes"]
fn mlx_shared_inputs_preserve_model_execution() {
    mlx_projection_routes_preserve_model_execution(|baseline| {
        crate::affine::SEPARATE_INPUTS_FOR_TEST.with(|v| v.set(baseline))
    });
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; old/new prefill stores at identical execution shapes"]
fn mlx_bulk_store_preserves_model_execution() {
    mlx_projection_routes_preserve_model_execution(|baseline| {
        crate::affine::BASELINE_STORE_FOR_TEST.with(|v| v.set(baseline))
    });
}

fn mlx_projection_routes_preserve_model_execution(control: fn(bool)) {
    let _reset = ProjectionControlReset(control);
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    for (slots, first_wave) in [(1, 512), (4, 512), (4, 32), (1, 256), (4, 384)] {
        let prompt: Vec<_> = (1000..1512).collect();
        let mut expected = Vec::new();
        for separate in [true, false] {
            control(separate);
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            for slot in 0..slots {
                model.prepare(slot, &prompt).unwrap();
            }
            let mut offset = 0;
            let mut logits = Vec::new();
            while offset < prompt.len() {
                let n =
                    (if offset == 0 { first_wave } else { 512 } / slots).min(prompt.len() - offset);
                let rows: Vec<_> = (0..slots)
                    .flat_map(|slot| {
                        (offset..offset + n)
                            .map(|pos| (slot, prompt[pos], pos as u32))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                offset += n;
                let selected = if offset == prompt.len() {
                    (0..slots).map(|s| (s + 1) * n - 1).collect()
                } else {
                    vec![]
                };
                logits = model.execute(&rows, &selected).unwrap();
            }
            for step in 0..5 {
                if separate {
                    expected.push(logits.clone());
                } else {
                    assert!(
                        logits == expected[step],
                        "projection route slots={slots}, first_wave={first_wave}, step={step}"
                    );
                }
                if step < 4 {
                    logits = model
                        .execute(
                            &(0..slots)
                                .map(|slot| {
                                    (slot, 7000 + step as u32, (prompt.len() + step) as u32)
                                })
                                .collect::<Vec<_>>(),
                            &(0..slots).collect::<Vec<_>>(),
                        )
                        .unwrap();
                }
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; use PADDOCK_METAL_PROFILE=1 for instrumented attribution, NOT serving timing"]
fn mlx_prefill_dispatch_diagnostic() {
    mlx_prefill_wave_diagnostic(false);
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; full 3584-token context attribution, NOT serving timing"]
fn mlx_long_prefill_dispatch_diagnostic() {
    mlx_prefill_wave_diagnostic(true);
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; full-backbone attribution, no head/sampling/HTTP; optional PADDOCK_METAL_PROFILE"]
fn mlx_full_stage_cost() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let mut reference = std::collections::BTreeMap::new();
    // First round warms every geometry. Two reversed rounds expose ordering
    // drift. Stage counters use separate encoders and are not serving timings.
    for round in 0..3 {
        let mut cases = vec![(1, 512), (1, 3584), (4, 512), (4, 3584)];
        if round == 2 {
            cases.reverse();
        }
        for (slots, length) in cases {
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            let prompt = (1000..1000 + length as u32).collect::<Vec<_>>();
            for slot in 0..slots {
                model.prepare(slot, &prompt).unwrap();
            }
            eprintln!("NATIVE_STAGE_BEGIN slots={slots} tokens={length} round={round}");
            let mut waves = Vec::new();
            let started = std::time::Instant::now();
            for offset in (0..length).step_by(512 / slots) {
                let count = (512 / slots).min(length - offset);
                let rows = (0..slots)
                    .flat_map(|slot| {
                        (offset..offset + count).map(move |p| (slot, 1000 + p as u32, p as u32))
                    })
                    .collect::<Vec<_>>();
                model.execute(&rows, &[]).unwrap();
                waves.push(model.last_gpu_seconds * 1000.);
            }
            let wall_ms = started.elapsed().as_secs_f64() * 1000.;
            // SAFETY: completed GPU command; scratch.x holds this wave's final hidden
            // rows before the final output norm (which requires selection).
            let hidden = unsafe { model.scratch.x.read_f32(0, 512 * model.width) };
            let mut hash = blake3::Hasher::new();
            for value in hidden {
                assert!(value.is_finite());
                hash.update(&value.to_bits().to_le_bytes());
            }
            let signature = hash.finalize().to_hex().to_string();
            let expected = reference
                .entry((slots, length))
                .or_insert(signature.clone());
            eprintln!(
                "NATIVE_STAGE_COST {}",
                serde_json::json!({"slots":slots,"tokens":length,"round":round,
                    "gpu_ms":waves,"wall_ms":wall_ms,"hidden_blake3":signature,
                    "exact_control": &signature == expected})
            );
            assert_eq!(&signature, expected, "repeated backbone state changed");
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; rotated cold-wave cost and exact logit qualification, not a serving bar"]
fn mlx_prefill_capacity_cost() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    model.reserve_rows(2048).unwrap();
    for slots in [1, 4] {
        let prompt: Vec<u32> = (1000..4584).collect();
        let mut reference = Vec::new();
        for round in 0..3 {
            for order in 0..3 {
                let quantum = [512, 1024, 2048][(round + order) % 3];
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                for slot in 0..slots {
                    model.prepare(slot, &prompt).unwrap();
                }
                let mut times = Vec::new();
                let mut logits = Vec::new();
                let started = std::time::Instant::now();
                for first in (0..prompt.len()).step_by(quantum / slots) {
                    let end = (first + quantum / slots).min(prompt.len());
                    let rows = (0..slots)
                        .flat_map(|slot| {
                            (first..end).map(move |p| (slot, 1000 + p as u32, p as u32))
                        })
                        .collect::<Vec<_>>();
                    let selected = if end == prompt.len() {
                        (1..=slots).map(|s| s * (end - first) - 1).collect()
                    } else {
                        vec![]
                    };
                    logits = model.execute(&rows, &selected).unwrap();
                    times.push(model.last_gpu_seconds * 1000.);
                }
                let wall = started.elapsed().as_secs_f64() * 1000.;
                let mut hashes = Vec::new();
                for step in 0..4 {
                    let mut hash = blake3::Hasher::new();
                    for value in &logits {
                        hash.update(&value.to_bits().to_le_bytes());
                    }
                    hashes.push(hash.finalize().to_hex().to_string());
                    if step < 3 {
                        let rows = (0..slots)
                            .map(|slot| (slot, 7000 + step, prompt.len() as u32 + step))
                            .collect::<Vec<_>>();
                        logits = model
                            .execute(&rows, &(0..slots).collect::<Vec<_>>())
                            .unwrap();
                    }
                }
                if reference.is_empty() {
                    reference = hashes.clone();
                }
                eprintln!(
                    "MLX_CAPACITY_COST {}",
                    serde_json::json!({
                        "slots":slots,"round":round,"quantum":quantum,"gpu_ms":times,
                        "wall_ms":wall,"exact":hashes==reference,
                        "allocated":model.device.allocated_bytes()
                    })
                );
                assert_eq!(hashes, reference, "slots={slots}, quantum={quantum}");
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; rotated admission-policy diagnostic, not an HTTP rival bar"]
fn mlx_admission_policy_cost() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            serving::SCHEDULE_PROBE.with(|v| v.set(0));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    for length in [512, 3584] {
        let mut reference = vec![Vec::new(); 4];
        for round in 0..2 {
            for order in 0..5 {
                let route = if round == 0 { order } else { 4 - order };
                serving::SCHEDULE_PROBE.with(|v| v.set(route));
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                let mut hashes = vec![Vec::new(); 4];
                let mut counts = [0; 4];
                let mut ttft = [0.; 4];
                let mut last = [0.; 4];
                let mut gaps = Vec::new();
                let started = std::time::Instant::now();
                for tick in 0..2048 {
                    if tick == 0 || tick == 1 {
                        for slot in if tick == 0 { 0..1 } else { 1..4 } {
                            model
                                .prefill_begin(
                                    slot,
                                    (0..length)
                                        .map(|p| 1000 + slot as u32 * 4000 + p as u32)
                                        .collect(),
                                )
                                .unwrap();
                        }
                    }
                    let decodes: Vec<_> = (0..4)
                        .filter(|&s| counts[s] > 0 && counts[s] < 32)
                        .map(|s| {
                            (
                                s,
                                7000 + counts[s] as u32,
                                model.slots[s].history.len() as u32,
                            )
                        })
                        .collect();
                    let (decoded, prefilled) = model.forward_mixed(&decodes, 8192).unwrap();
                    let results = decodes
                        .iter()
                        .enumerate()
                        .map(|(r, &(s, _, _))| {
                            (s, &decoded[r * model.vocab..(r + 1) * model.vocab])
                        })
                        .chain(
                            prefilled
                                .iter()
                                .map(|(s, logits, _)| (*s, logits.as_slice())),
                        );
                    let now = started.elapsed().as_secs_f64() * 1000.;
                    for (s, logits) in results {
                        if counts[s] == 0 {
                            ttft[s] = now;
                        } else {
                            gaps.push(now - last[s]);
                        }
                        last[s] = now;
                        counts[s] += 1;
                        let mut hash = blake3::Hasher::new();
                        for value in logits {
                            hash.update(&value.to_bits().to_le_bytes());
                        }
                        hashes[s].push(hash.finalize().to_hex().to_string());
                    }
                    if counts.iter().all(|n| *n == 32) {
                        break;
                    }
                    assert!(tick < 2047);
                }
                if route == 0 && round == 0 {
                    reference = hashes.clone();
                }
                gaps.sort_by(f64::total_cmp);
                eprintln!(
                    "MLX_ADMISSION_COST {}",
                    serde_json::json!({
                        "length":length,"round":round,"route":route,"ttft_ms":ttft,
                        "wall_ms":started.elapsed().as_secs_f64()*1000.,
                        "p99_ms":gaps[(gaps.len()*99/100).min(gaps.len()-1)],
                        "max_ms":gaps.last(),"exact":hashes==reference
                    })
                );
                assert_eq!(hashes, reference, "route={route}, length={length}");
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; mixed-wave GPU cost floor, not a serving or parity gate"]
fn mlx_mixed_wave_cost() {
    let _reset =
        ProjectionControlReset(|old| crate::affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(old)));
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let mut reference = std::collections::BTreeMap::new();
    for round in 0..3 {
        for decoders in [0, 1, 3] {
            for baseline in if round % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                crate::affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(baseline));
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                for slot in 0..=decoders {
                    let prompt =
                        (1000 + slot as u32 * 4000..1512 + slot as u32 * 4000).collect::<Vec<_>>();
                    model.prefill(slot, &prompt).unwrap();
                }
                // This slot remains in the prompt phase. Its peers decode, even
                // when a probe appends only one token. No phase/precision waiver.
                model.slots[decoders].prefill_end = 4096;
                let mut counts = vec![0, 1, 4, 8, 16, 32, 64, 125];
                if round == 1 {
                    counts.reverse();
                }
                for count in counts {
                    if count == 0 && decoders == 0 {
                        continue;
                    }
                    let mut rows = (0..decoders)
                        .map(|slot| (slot, 7000, model.slots[slot].history.len() as u32))
                        .collect::<Vec<_>>();
                    let prefix = model.slots[decoders].history.len();
                    rows.extend((prefix..prefix + count).map(|p| (decoders, 9000, p as u32)));
                    let selected = (0..decoders).collect::<Vec<_>>();
                    let started = std::time::Instant::now();
                    let logits = model.execute(&rows, &selected).unwrap();
                    let wall_ms = started.elapsed().as_secs_f64() * 1000.;
                    assert!(logits.iter().all(|v| v.is_finite()));
                    // Completed command, bounded row count. Compare the unfinished
                    // prompt's final hidden state too, even when there is no head.
                    let hidden = unsafe { model.scratch.x.read_f32(0, rows.len() * model.width) };
                    let mut hash = blake3::Hasher::new();
                    for value in logits.iter().chain(&hidden) {
                        assert!(value.is_finite());
                        hash.update(&value.to_bits().to_le_bytes());
                    }
                    let digest = hash.finalize();
                    assert_eq!(
                        *reference.entry((round, decoders, count)).or_insert(digest),
                        digest
                    );
                    eprintln!(
                        "MIXED_WAVE_COST {}",
                        serde_json::json!({
                            "round":round,"baseline":baseline,"decodes":decoders,"prefill":count,"prefix":prefix,
                            "heads":selected.len(),"gpu_ms":model.last_gpu_seconds*1000.,
                            "wall_ms":wall_ms,"exact":true
                        })
                    );
                }
            }
        }
    }
}

fn mlx_prefill_wave_diagnostic(full_context: bool) {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    for slots in [1, 4] {
        model.reset();
        for cache in &mut model.cache {
            cache.table.clear(&mut model.pool);
            cache.history.clear();
        }
        for slot in 0..slots {
            model
                .prepare(slot, &(1000..4584).collect::<Vec<_>>())
                .unwrap();
        }
        // No startup admission gap: all peers are already present. Capture
        // either the original two startup waves, or all waves through 3584
        // tokens per slot. Late attention/checkpoint cost must not be inferred
        // from the beginning of the prompt. This excludes final vocabulary
        // projection, sampling, HTTP and admission-arrival timing.
        let waves = if full_context { 3584 * slots / 512 } else { 2 };
        for wave in 0..waves {
            eprintln!("MLX_DISPATCH_BEGIN slots={slots} wave={wave}");
            let started = std::time::Instant::now();
            let per_slot = 512 / slots;
            let rows = (0..slots)
                .flat_map(|slot| {
                    (wave * per_slot..(wave + 1) * per_slot)
                        .map(move |pos| (slot, 1000 + pos as u32, pos as u32))
                })
                .collect::<Vec<_>>();
            model.execute(&rows, &[]).unwrap();
            eprintln!(
                "MLX_DISPATCH_END slots={slots} wave={wave} gpu_ms={} wall_ms={}",
                model.last_gpu_seconds * 1000.,
                started.elapsed().as_secs_f64() * 1000.
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; teacher-forced shape isolation, NOT a generation-parity pass"]
fn mlx_batch_shape_hidden_and_head_diagnostic() {
    struct ResetElections;
    impl Drop for ResetElections {
        fn drop(&mut self) {
            projection::CANONICAL_MLX_FOR_TEST.with(|v| v.set(false));
            crate::affine::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(false));
            projection::BASELINE_AFFINE_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = ResetElections;
    // Preserve this historical diagnostic's old single-stream reference;
    // the phase-local contract has its own complete-generation gate.
    projection::BASELINE_AFFINE_FOR_TEST.with(|v| v.set(true));
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let prompt: Vec<u32> = (1000..1512).collect();
    let mut reference = Vec::new();
    // Equal 512-row projection waves isolate cohort/chunk layout from total
    // matmul width. The 32-row admission variant then changes that width.
    for (slots, first_wave, canonical) in [
        (1, 512, false),
        (4, 512, false),
        (1, 32, false),
        (4, 512, true),
        (1, 32, true),
    ] {
        projection::CANONICAL_MLX_FOR_TEST.with(|v| v.set(canonical));
        crate::affine::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(canonical));
        model.reset();
        for cache in &mut model.cache {
            cache.table.clear(&mut model.pool);
            cache.history.clear();
        }
        for slot in 0..slots {
            model.prepare(slot, &prompt).unwrap();
        }
        let mut offset = 0;
        let mut last = Vec::new();
        while offset < prompt.len() {
            let count =
                (if offset == 0 { first_wave } else { 512 } / slots).min(prompt.len() - offset);
            let mut rows = Vec::new();
            for slot in 0..slots {
                rows.extend((offset..offset + count).map(|pos| (slot, prompt[pos], pos as u32)));
            }
            offset += count;
            let selected = if offset == prompt.len() {
                (0..slots).map(|slot| (slot + 1) * count - 1).collect()
            } else {
                vec![]
            };
            last = model.execute(&rows, &selected).unwrap();
        }
        for step in 0..3 {
            let hidden = unsafe { model.scratch.norm.read_f32(0, model.width) };
            let logits = last[..model.vocab].to_vec();
            if slots == 1 && first_wave == 512 {
                reference.push((hidden.clone(), logits.clone()));
            }
            let (expected_hidden, expected_logits) = &reference[step];
            if canonical {
                for slot in 0..slots {
                    let hidden =
                        unsafe { model.scratch.norm.read_f32(slot * model.width, model.width) };
                    assert!(
                        hidden == *expected_hidden,
                        "canonical hidden slot={slot}/{slots}, first_wave={first_wave}, step={step}"
                    );
                    assert!(
                        last[slot * model.vocab..(slot + 1) * model.vocab] == *expected_logits,
                        "canonical logits slot={slot}/{slots}, first_wave={first_wave}, step={step}"
                    );
                }
            }
            let summary = |a: &[f32], b: &[f32]| {
                serde_json::json!({
                    "unequal": a.iter().zip(b).filter(|(a,b)| a != b).count(),
                    "max_abs": a.iter().zip(b).map(|(a,b)| (a-b).abs()).fold(0f32, f32::max)
                })
            };
            let pick = |a: &[f32]| {
                a.iter()
                    .enumerate()
                    .fold(0, |best, (i, v)| if *v > a[best] { i } else { best })
            };
            eprintln!(
                "MLX_BATCH_HIDDEN {}",
                serde_json::json!({
                    "slots":slots, "first_wave":first_wave, "step":step, "canonical":canonical,
                    "gpu_ms":model.last_gpu_seconds*1000.,
                    "hidden":summary(&hidden,expected_hidden), "logits":summary(&logits,expected_logits),
                    "pick":pick(&logits), "reference_pick":pick(expected_logits)
                })
            );
            if step < 2 {
                // Identical continuation IDs on every route: token feedback
                // must not obscure the first numerical divergence.
                last = model
                    .execute(
                        &(0..slots)
                            .map(|slot| (slot, 7000 + step as u32, (prompt.len() + step) as u32))
                            .collect::<Vec<_>>(),
                        &(0..slots).collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; alternating full-model attention cost and exact c=1 preservation"]
fn mlx_stable_attention_model_cost() {
    let _reset = ProjectionControlReset(|baseline| {
        projection::BASELINE_ATTENTION_FOR_TEST.with(|v| v.set(baseline));
    });
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    for slots in [1, 4] {
        for length in [512, 3584] {
            let prompt: Vec<u32> = (1000..1000 + length as u32).collect();
            let mut reference = Vec::new();
            for round in 0..3 {
                for i in 0..2 {
                    let baseline = (round + i) % 2 == 0;
                    projection::BASELINE_ATTENTION_FOR_TEST.with(|v| v.set(baseline));
                    model.reset();
                    for cache in &mut model.cache {
                        cache.table.clear(&mut model.pool);
                        cache.history.clear();
                    }
                    for slot in 0..slots {
                        model.prepare(slot, &prompt).unwrap();
                    }
                    let quantum = 512 / slots;
                    for first in (0..length).step_by(quantum) {
                        let rows = (0..slots)
                            .flat_map(|slot| {
                                (first..first + quantum)
                                    .map(|pos| (slot, prompt[pos], pos as u32))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        model.execute(&rows, &[]).unwrap();
                    }
                    let mut times = Vec::new();
                    let mut digests = Vec::new();
                    for step in 0..16 {
                        let rows = (0..slots)
                            .map(|slot| (slot, 7000 + step as u32, (length + step) as u32))
                            .collect::<Vec<_>>();
                        let logits = model
                            .execute(&rows, &(0..slots).collect::<Vec<_>>())
                            .unwrap();
                        assert!(logits.iter().all(|v| v.is_finite()));
                        times.push(model.last_gpu_seconds * 1000.);
                        let mut hash = blake3::Hasher::new();
                        for value in &logits {
                            hash.update(&value.to_bits().to_le_bytes());
                        }
                        digests.push(hash.finalize().to_hex().to_string());
                    }
                    if slots == 1 {
                        if reference.is_empty() {
                            reference = digests.clone();
                        }
                        assert_eq!(
                            digests, reference,
                            "c=1 attention arithmetic changed at length={length}"
                        );
                    }
                    eprintln!(
                        "MLX_STABLE_ATTENTION_MODEL {}",
                        serde_json::json!({
                            "slots":slots,"length":length,"round":round,"baseline":baseline,
                            "decode_gpu_ms":times,"logits_blake3":digests
                        })
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; rotating full-model projection-contract cost, not generation qualification"]
fn mlx_projection_contract_model_cost() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            projection::CANONICAL_MLX_FOR_TEST.with(|v| v.set(false));
            projection::BASELINE_AFFINE_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let length = 512;
    let prompt: Vec<u32> = (1000..1000 + length as u32).collect();
    for round in 0..3 {
        for slots in if round % 2 == 0 { [1, 4] } else { [4, 1] } {
            for i in 0..3 {
                let route = (round + i) % 3;
                projection::CANONICAL_MLX_FOR_TEST.with(|v| v.set(route == 1));
                projection::BASELINE_AFFINE_FOR_TEST.with(|v| v.set(route != 2));
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                for slot in 0..slots {
                    model.prepare(slot, &prompt).unwrap();
                }
                let quantum = 512 / slots;
                for first in (0..length).step_by(quantum) {
                    let rows = (0..slots)
                        .flat_map(|slot| {
                            (first..first + quantum)
                                .map(|pos| (slot, prompt[pos], pos as u32))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    model.execute(&rows, &[]).unwrap();
                }
                let mut times = Vec::new();
                for step in 0..16 {
                    let rows = (0..slots)
                        .map(|slot| (slot, 7000 + step as u32, (length + step) as u32))
                        .collect::<Vec<_>>();
                    let logits = model
                        .execute(&rows, &(0..slots).collect::<Vec<_>>())
                        .unwrap();
                    assert!(logits.iter().all(|v| v.is_finite()));
                    times.push(model.last_gpu_seconds * 1000.);
                }
                eprintln!(
                    "MLX_PROJECTION_CONTRACT_COST {}",
                    serde_json::json!({"slots":slots,"round":round,"route":(["baseline","canonical","production"][route]),"decode_gpu_ms":times})
                );
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact native cache lifecycle GPU test"]
fn native_checkpoint_relocation_replay_and_abort_are_exact() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 1024, 4, None).unwrap();
    assert!(model.mlx);
    assert!(!model.spec_capable());
    let prompt: Vec<u32> = (1000..1537).collect();
    model.prepare(0, &prompt).unwrap();
    // Keep the numerical route fixed on both sides of the checkpoint. BF16
    // partial sums and one-vector vs batched bias arithmetic intentionally
    // differ across routes, so cross-route F32 tolerances are not this test.
    for (first, end) in [(0, 512), (512, 528)] {
        model
            .execute(
                &(first..end)
                    .map(|i| (0, prompt[i], i as u32))
                    .collect::<Vec<_>>(),
                &[],
            )
            .unwrap();
    }
    let suffix = |slot| {
        (528..537)
            .map(|i| (slot, prompt[i], i as u32))
            .collect::<Vec<_>>()
    };
    let reference = model.execute(&suffix(0), &[8]).unwrap();
    let reference_next = model.execute(&[(0, 11751, 537)], &[0]).unwrap();
    assert!(
        reference
            .iter()
            .chain(&reference_next)
            .all(|v| v.is_finite())
    );
    for slot in [3, 1, 2] {
        assert_eq!(model.prepare(slot, &prompt).unwrap(), 528);
        let got = model.execute(&suffix(slot), &[8]).unwrap();
        assert_eq!(reference, got, "slot relocation {slot}");
        assert_eq!(
            reference_next,
            model.execute(&[(slot, 11751, 537)], &[0]).unwrap()
        );
    }
    // A four-row step must preserve slot identity through release/hole reuse.
    let rows = (0..4).map(|s| (s, 42 + s as u32, 538)).collect::<Vec<_>>();
    let expected = model.execute(&rows, &[0, 1, 2, 3]).unwrap();
    model.release_inactive_slots(&[true, false, true, false]);
    for slot in [1, 3] {
        model.prepare(slot, &prompt).unwrap();
        model.execute(&suffix(slot), &[]).unwrap();
        model.execute(&[(slot, 11751, 537)], &[]).unwrap();
    }
    // Restore the same pre-step state of the retained slots as well, then
    // repeat the identical packed operation. No per-slot state may leak.
    for slot in [0, 2] {
        model.prepare(slot, &prompt).unwrap();
        model.execute(&suffix(slot), &[]).unwrap();
        model.execute(&[(slot, 11751, 537)], &[]).unwrap();
    }
    assert_eq!(expected, model.execute(&rows, &[0, 1, 2, 3]).unwrap());
    for c in &mut model.cache {
        c.table.clear(&mut model.pool);
        c.history.clear();
    }
    let other: Vec<u32> = (3000..3073).collect();
    model.prefill_begin(1, other.clone()).unwrap();
    let (rider, done) = model.forward_mixed(&[(2, 13, 539)], 7).unwrap();
    assert_eq!(rider.len(), model.vocab);
    assert!(done.is_empty());
    assert!(model.prefill_abort(1));
    assert!(model.pending.iter().all(|p| p.slot != 1));
    assert_eq!(model.prepare(1, &other).unwrap(), 0);
    let mut first = Vec::new();
    for slot in [1, 3] {
        if slot == 3 {
            for c in &mut model.cache {
                c.table.clear(&mut model.pool);
                c.history.clear();
            }
            assert_eq!(model.prepare(slot, &other).unwrap(), 0);
        }
        for start in (0..73).step_by(7) {
            let end = (start + 7).min(73);
            let selected = if end == 73 {
                vec![end - start - 1]
            } else {
                vec![]
            };
            let value = model
                .execute(
                    &(start..end)
                        .map(|i| (slot, other[i], i as u32))
                        .collect::<Vec<_>>(),
                    &selected,
                )
                .unwrap();
            if end == 73 {
                if slot == 1 {
                    first = value;
                } else {
                    assert_eq!(first, value);
                }
            }
        }
    }
    model.release_inactive_slots(&[false; 4]);
    assert!(
        model
            .forward_batch(&[0; 4], &[0; 4])
            .unwrap()
            .iter()
            .all(|&v| v == 0.)
    );
    assert!(model.forward_mixed(&[(4, 1, 0)], 1).is_err());
    assert!(model.forward_prefill(0, &vec![1; 1025]).is_err());
}
