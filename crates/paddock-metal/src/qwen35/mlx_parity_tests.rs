//! Complete-generation gates, not tolerance-based tensor diagnostics.
use super::*;

#[derive(Clone, Default)]
struct Generation {
    tokens: Vec<u32>,
    logits: Vec<String>,
    stopped: bool,
    cached: usize,
}

fn clear(model: &mut Qwen35, cold: bool) {
    model.reset();
    if cold {
        for cache in &mut model.cache {
            cache.table.clear(&mut model.pool);
            cache.history.clear();
        }
    }
}

fn generate(
    model: &mut Qwen35,
    prompts: &[Vec<u32>],
    active: &[usize],
    late: bool,
    stops: &[u32],
    cap: usize,
    quantum: usize,
) -> Vec<Generation> {
    let mut output = vec![Generation::default(); prompts.len()];
    let mut admitted = vec![false; prompts.len()];
    let mut ready = vec![false; prompts.len()];
    for tick in 0..1024 {
        for (order, &slot) in active.iter().enumerate() {
            if !admitted[slot] && (!late || order == 0 || tick >= 1) {
                model.prefill_begin(slot, prompts[slot].clone()).unwrap();
                output[slot].cached = model.slots[slot].reused;
                admitted[slot] = true;
            }
        }
        let decodes = active
            .iter()
            .copied()
            .filter(|&slot| ready[slot] && !output[slot].stopped && output[slot].tokens.len() < cap)
            .map(|slot| {
                (
                    slot,
                    *output[slot].tokens.last().unwrap(),
                    model.slots[slot].history.len() as u32,
                )
            })
            .collect::<Vec<_>>();
        let (decoded, prefilled) = model.forward_mixed(&decodes, quantum).unwrap();
        let results = decodes
            .iter()
            .enumerate()
            .map(|(row, &(slot, _, _))| {
                (slot, &decoded[row * model.vocab..(row + 1) * model.vocab])
            })
            .chain(
                prefilled
                    .iter()
                    .map(|(slot, logits, _)| (*slot, logits.as_slice())),
            );
        for (slot, logits) in results {
            assert!(logits.iter().all(|v| v.is_finite()));
            let mut hash = blake3::Hasher::new();
            for value in logits {
                hash.update(&value.to_bits().to_le_bytes());
            }
            let token =
                logits.iter().enumerate().fold(
                    0,
                    |best, (i, value)| if *value > logits[best] { i } else { best },
                ) as u32;
            output[slot]
                .logits
                .push(hash.finalize().to_hex().to_string());
            output[slot].tokens.push(token);
            output[slot].stopped = stops.contains(&token);
            ready[slot] = true;
        }
        if active
            .iter()
            .all(|&slot| output[slot].stopped || output[slot].tokens.len() == cap)
        {
            return output;
        }
    }
    panic!("generation made no bounded progress");
}

#[test]
#[ignore = "requires Bonsai and PADDOCK_BONSAI_REFERENCE; complete logits across batches, arrivals, hot restore and tiny prompt slices"]
fn bonsai_phase_generation_gate() {
    let path = std::env::var("PADDOCK_METAL_BONSAI_MODEL").unwrap();
    let reference = std::path::PathBuf::from(std::env::var("PADDOCK_BONSAI_REFERENCE").unwrap());
    let data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("results.json")).unwrap()).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() >= 4);
    let cases = &cases[cases.len() - 4..];
    let prompts = cases
        .iter()
        .map(|c| {
            let tokens = tokenizer.encode(c["prompt"].as_str().unwrap()).unwrap();
            assert_eq!(serde_json::json!(tokens), c["tokens"]);
            tokens
        })
        .collect::<Vec<_>>();
    let cap = data["max_new_tokens"].as_u64().unwrap() as usize;
    assert!((1..=256).contains(&cap));
    assert!(prompts.iter().all(|p| p.len() + cap <= 4096));
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let mut references = Vec::new();
    let mut hot_restores = 0;
    for slot in 0..4 {
        clear(&mut model, true);
        let mut one = generate(
            &mut model,
            &prompts,
            &[slot],
            false,
            &tokenizer.stop_ids(),
            cap,
            512,
        );
        references.push(std::mem::take(&mut one[slot]));
        if prompts[slot].len() > 48 {
            clear(&mut model, false);
            let hot = generate(
                &mut model,
                &prompts,
                &[slot],
                false,
                &tokenizer.stop_ids(),
                cap,
                512,
            );
            assert!(
                hot[slot].cached > 0,
                "hot singleton did not exercise prefix restoration"
            );
            assert_eq!(hot[slot].logits, references[slot].logits);
            assert_eq!(hot[slot].tokens, references[slot].tokens);
            hot_restores += 1;
        }
    }
    let mut failures = Vec::new();
    for (label, active, late, cold, quantum) in [
        ("cold-c4", vec![0, 1, 2, 3], false, true, 512),
        ("hot-c4", vec![3, 1, 0, 2], false, false, 512),
        ("late-c4", vec![0, 1, 2, 3], true, true, 512),
        ("q32-c4", vec![0, 1, 2, 3], false, true, 32),
        ("q13-c2", vec![3, 0], true, true, 13),
    ] {
        clear(&mut model, cold);
        let output = generate(
            &mut model,
            &prompts,
            &active,
            late,
            &tokenizer.stop_ids(),
            cap,
            quantum,
        );
        for slot in active {
            let expected = &references[slot];
            let got = &output[slot];
            let exact = got.logits == expected.logits;
            let tokens = got.tokens == expected.tokens;
            let reference_tokens = serde_json::json!(got.tokens) == cases[slot]["generated"];
            eprintln!(
                "BONSAI_PHASE_GATE {}",
                serde_json::json!({"scenario":label,"slot":slot,"exact":exact,"tokens":tokens,"reference_tokens":reference_tokens,"stopped":got.stopped,"cached":got.cached,"first_logit_difference":got.logits.iter().zip(&expected.logits).position(|(a,b)|a!=b)})
            );
            if !exact || !tokens || !reference_tokens || !got.stopped {
                failures.push((label, slot));
            }
        }
    }
    assert!(failures.is_empty(), "Bonsai phase failures: {failures:?}");
    eprintln!("BONSAI_PHASE_HOT_RESTORES {hot_restores}");
}

#[test]
#[ignore = "requires target and PADDOCK_METAL_PARITY_FIXTURE; strict natural-stop/batch/arrival/hot-restore gate, no speculation"]
fn mlx_batch_stable_complete_generation_gate() {
    complete_generation_gate(false, false);
}

#[test]
#[ignore = "requires target and PADDOCK_METAL_PARITY_FIXTURE; packed recurrence/GQA versus old-kernel complete logits through EOS"]
fn mlx_packed_kernels_complete_generation_gate() {
    complete_generation_gate(true, false);
}

#[test]
#[ignore = "requires target and PADDOCK_METAL_PARITY_FIXTURE; ragged projection baseline versus candidate through EOS"]
fn mlx_ragged_projection_complete_generation_gate() {
    complete_generation_gate(false, true);
}

fn complete_generation_gate(baseline_kernel_reference: bool, baseline_ragged_reference: bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            projection::CANONICAL_MLX_FOR_TEST.with(|v| v.set(false));
            crate::affine::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(false));
            attention::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(false));
            projection::CANONICAL_PROMPT_PHASE_FOR_TEST.with(|v| v.set(false));
            projection::BASELINE_AFFINE_FOR_TEST.with(|v| v.set(false));
            forward::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(false));
            attention::BASELINE_GQA_FOR_TEST.with(|v| v.set(false));
            crate::affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    // Default means the shipping path with no arithmetic overrides. Retain
    // the historical controls for independent numerical/cost isolation.
    let control =
        std::env::var("PADDOCK_METAL_PARITY_CONTROL").unwrap_or_else(|_| "production".into());
    assert!(["decode-only", "prefill", "full", "production"].contains(&control.as_str()));
    let baseline = control != "production";
    projection::BASELINE_AFFINE_FOR_TEST.with(|v| v.set(baseline));
    projection::CANONICAL_MLX_FOR_TEST.with(|v| v.set(baseline));
    crate::affine::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(baseline));
    attention::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(baseline && control != "decode-only"));
    projection::CANONICAL_PROMPT_PHASE_FOR_TEST.with(|v| v.set(control == "full"));
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let fixture = std::fs::read(std::env::var("PADDOCK_METAL_PARITY_FIXTURE").unwrap()).unwrap();
    let data: serde_json::Value = serde_json::from_slice(&fixture).unwrap();
    assert_eq!(data["schema"], 1);
    assert_eq!(
        std::fs::canonicalize(&path).unwrap(),
        std::fs::canonicalize(data["model"].as_str().unwrap()).unwrap()
    );
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let prompts = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let tokens = tokenizer.encode(case["prompt"].as_str().unwrap()).unwrap();
            assert_eq!(
                serde_json::json!(tokens),
                case["tokens"],
                "Rust/HF prompt tokenization differs"
            );
            assert_eq!(
                tokens.len(),
                case["prompt_tokens"].as_u64().unwrap() as usize
            );
            tokens
        })
        .collect::<Vec<_>>();
    assert_eq!(prompts.len(), 4);
    let cap = data["max_new_tokens"].as_u64().unwrap() as usize;
    assert!((1..=256).contains(&cap));
    assert!(prompts.iter().all(|p| p.len() + cap <= 4096));
    let stops = tokenizer.stop_ids();
    assert!(!stops.is_empty());
    let config = std::fs::read(Path::new(&path).join("config.json")).unwrap();
    eprintln!(
        "MLX_GENERATION_GATE_INPUT {}",
        serde_json::json!({
            "fixture_blake3":blake3::hash(&fixture).to_hex().to_string(),
            "loaded_config_blake3":blake3::hash(&config).to_hex().to_string(),
        "stop_ids":stops,"cap":cap,"speculation":false,"control":control,
        "baseline_kernel_reference":baseline_kernel_reference,
        "baseline_ragged_reference":baseline_ragged_reference
        })
    );
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let mut references = Vec::new();
    forward::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(baseline_kernel_reference));
    attention::BASELINE_GQA_FOR_TEST.with(|v| v.set(baseline_kernel_reference));
    crate::affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(baseline_ragged_reference));
    for slot in 0..4 {
        clear(&mut model, true);
        let mut one = generate(&mut model, &prompts, &[slot], false, &stops, cap, 512);
        references.push(std::mem::take(&mut one[slot]));
    }
    forward::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(false));
    attention::BASELINE_GQA_FOR_TEST.with(|v| v.set(false));
    crate::affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(false));
    let mut failures = Vec::new();
    let mut scenarios = vec![
        ("cold-c4", vec![0, 1, 2, 3], false, true, 512),
        ("hot-c4-reordered", vec![3, 1, 0, 2], false, false, 512),
        ("cold-c4-late", vec![0, 1, 2, 3], true, true, 512),
        ("cold-c2", vec![3, 0], false, true, 512),
    ];
    if std::env::var("PADDOCK_METAL_PARITY_STRESS").as_deref() == Ok("1") {
        scenarios.extend([
            ("cold-c4-q32", vec![0, 1, 2, 3], false, true, 32),
            ("cold-c4-q128-late", vec![0, 1, 2, 3], true, true, 128),
            ("cold-c2-q13", vec![3, 0], false, true, 13),
        ]);
    }
    for (label, active, late, cold, quantum) in scenarios {
        clear(&mut model, cold);
        let output = generate(&mut model, &prompts, &active, late, &stops, cap, quantum);
        for &slot in &active {
            let expected = &references[slot];
            let got = &output[slot];
            let first_logits = got
                .logits
                .iter()
                .zip(&expected.logits)
                .position(|(a, b)| a != b)
                .or_else(|| {
                    (got.logits.len() != expected.logits.len())
                        .then_some(got.logits.len().min(expected.logits.len()))
                });
            let first_token = got
                .tokens
                .iter()
                .zip(&expected.tokens)
                .position(|(a, b)| a != b)
                .or_else(|| {
                    (got.tokens.len() != expected.tokens.len())
                        .then_some(got.tokens.len().min(expected.tokens.len()))
                });
            let pass = first_logits.is_none()
                && first_token.is_none()
                && got.stopped
                && expected.stopped
                && if cold {
                    got.cached == 0
                } else {
                    got.cached > 0
                };
            eprintln!(
                "MLX_GENERATION_GATE {}",
                serde_json::json!({
                    "route":label,"slot":slot,"pass":pass,"natural_stop":got.stopped,"quantum":quantum,
                    "reference_natural_stop":expected.stopped,"cached_tokens":got.cached,
                    "first_logit_difference":first_logits,"first_token_difference":first_token,
                    "tokens":got.tokens,"reference_tokens":expected.tokens,
                    "text":tokenizer.decode(&got.tokens, false).unwrap(),
                    "reference_text":tokenizer.decode(&expected.tokens, false).unwrap()
                })
            );
            if !pass {
                failures.push(format!("{label}/slot-{slot}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "complete-generation qualification failed: {failures:?}"
    );
}
