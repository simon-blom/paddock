use super::*;
use paddock_engine::{generator::MmAdmit, service::MmChunk};
use paddock_kernels::reference::ternary::{dequant_ternary, encode_ptq1_0};

#[test]
fn ptq1_header_rejects_incompatible_rotation_before_gpu_allocation() {
    use paddock_models::{
        ggml_type::GgmlType,
        gguf::{GgufFile, TensorInfo, Value},
    };
    let header = || {
        let mut names = vec!["output.weight".to_string()];
        for layer in 0..64 {
            let mixer: &[&str] = if layer % 4 == 3 {
                &["attn_q", "attn_k", "attn_v", "attn_output"]
            } else {
                &["attn_qkv", "attn_gate", "ssm_out"]
            };
            for kind in mixer.iter().chain(&["ffn_gate", "ffn_up", "ffn_down"]) {
                names.push(format!("blk.{layer}.{kind}.weight"));
            }
        }
        let tensors = names
            .iter()
            .map(String::as_str)
            .chain(["token_embd.weight"])
            .map(|name| TensorInfo {
                name: name.into(),
                dims: vec![5120, 5120],
                ggml_type: GgmlType::Ptq1_0,
                raw_type: 143,
                offset: 0,
            })
            .collect();
        let metadata = [
            ("version", Value::U32(1)),
            ("block_size", Value::U32(1024)),
            (
                "transform",
                Value::Str("normalized-sylvester-walsh-hadamard".into()),
            ),
            ("axis", Value::Str("input-last-dimension".into())),
            ("sign_mode", Value::Str("identity".into())),
            (
                "weight_names",
                Value::Array(names.into_iter().map(Value::Str).collect()),
            ),
            (
                "inverse_weight_names",
                Value::Array(vec![Value::Str("token_embd.weight".into())]),
            ),
            ("gdn_v_grouped", Value::Bool(true)),
        ]
        .into_iter()
        .map(|(k, v)| (format!("prism.hadamard.{k}"), v))
        .collect();
        GgufFile {
            version: 3,
            alignment: 32,
            metadata,
            tensors,
            data_offset: 0,
        }
    };
    assert!(
        ternary::validate(&header(), Geometry::DENSE_27B, 0)
            .unwrap()
            .is_some()
    );
    for (key, value) in [
        ("block_size", Value::U32(512)),
        ("gdn_v_grouped", Value::Bool(false)),
        ("inverse_weight_names", Value::Array(vec![])),
        (
            "weight_names",
            Value::Array(vec![Value::Str("output.weight".into())]),
        ),
        ("future_transform", Value::Bool(true)),
    ] {
        let mut h = header();
        h.metadata.insert(format!("prism.hadamard.{key}"), value);
        assert!(
            ternary::validate(&h, Geometry::DENSE_27B, 0).is_err(),
            "{key}"
        );
    }
    assert!(ternary::validate(&header(), Geometry::DENSE_9B, 0).is_err());
    assert!(ternary::validate(&header(), Geometry::DENSE_27B, 1).is_err());
    let mut h = header();
    h.tensors[0].raw_type = 142;
    assert!(ternary::validate(&h, Geometry::DENSE_27B, 0).is_err());
    let mut h = header();
    let mut unexpected = h.tensors[0].clone();
    unexpected.name = "blk.0.ssm_alpha.weight".into();
    h.tensors.push(unexpected);
    assert!(ternary::validate(&h, Geometry::DENSE_27B, 0).is_err());
    let mut h = header();
    h.tensors.pop();
    assert!(ternary::validate(&h, Geometry::DENSE_27B, 0).is_err());
}

fn upload(device: &MetalDevice, values: &[f32]) -> Buffer {
    device
        .upload_parts(&[&values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()])
        .unwrap()
}
fn plane(k: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    for block in 0..k * n / 128 {
        let values = (0..128)
            .map(|i| ((i * 97 + block * 47 + i * i) % 3) as i8 - 1)
            .collect::<Vec<_>>();
        bytes.extend(encode_ptq1_0(
            &values,
            half::f16::from_f32((block % 29 + 1) as f32 / 113.),
        ));
    }
    let mut floats = vec![0.; k * n];
    dequant_ternary(143, &bytes, &mut floats).unwrap();
    (bytes, floats)
}

#[test]
fn ptq1_unpack_is_exact_for_every_byte_and_tail_digit() {
    let device = MetalDevice::new(None).unwrap();
    let bytes = (0..256)
        .flat_map(|b| {
            let mut block = vec![b as u8; 26];
            block.extend(half::f16::from_f32(0.37).to_le_bytes());
            block
        })
        .collect::<Vec<_>>();
    let mut expected = vec![0.; 256 * 128];
    dequant_ternary(143, &bytes, &mut expected).unwrap();
    let weights = device.upload_parts(&[&bytes]).unwrap();
    let out = device.alloc(expected.len() * 4).unwrap();
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "ptq1_unpack",
        &[&weights, &out],
        &[expected.len() as u32],
        [expected.len().div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { out.read_f32(0, expected.len()) }, expected);
}

#[test]
fn ptq1_projection_matches_f64_with_row_and_column_tails() {
    let device = MetalDevice::new(None).unwrap();
    let (k, n) = (512, 35);
    let (bytes, weights) = plane(k, n);
    let packed = device.upload_parts(&[&bytes]).unwrap();
    for rows in [1usize, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 64, 65, 129] {
        let values = (0..rows * k)
            .map(|i| ((i * 131 % 997) as f32 - 498.) / 317.)
            .collect::<Vec<_>>();
        let input = upload(&device, &values);
        let out = upload(&device, &vec![12345.; rows * n + 16]);
        let (kernel, cols, tile) = if rows <= 4 {
            (
                [
                    "ptq1_vectors1",
                    "ptq1_vectors2",
                    "ptq1_vectors3",
                    "ptq1_vectors4",
                ][rows - 1],
                4,
                rows,
            )
        } else if rows <= 16 {
            ("ptq1_mm16", 32, 16)
        } else if rows <= 32 {
            ("ptq1_mm32", 32, 32)
        } else {
            ("ptq1_mm64", 32, 64)
        };
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            kernel,
            &[&packed, &input, &out],
            &[k as u32, n as u32, rows as u32],
            [n.div_ceil(cols), rows.div_ceil(tile), 1],
            128,
        );
        cmd.finish().unwrap();
        let actual = unsafe { out.read_f32(0, rows * n + 16) };
        let mut maximum = 0f64;
        for row in 0..rows {
            for col in 0..n {
                let expected = (0..k)
                    .map(|i| values[row * k + i] as f64 * weights[col * k + i] as f64)
                    .sum::<f64>();
                maximum = maximum.max((actual[row * n + col] as f64 - expected).abs());
            }
        }
        assert!(maximum < 0.0002, "rows={rows}: {maximum}");
        assert!(actual[rows * n..].iter().all(|&v| v == 12345.));
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_PTQ1_MODEL"]
fn ptq1_model_smoke() {
    let path = std::env::var("PADDOCK_METAL_PTQ1_MODEL").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let prompt = "<|im_start|>user\nWhat is 2 + 3? Reply with just the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = tokenizer.encode(prompt).unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    assert!(model.ternary.is_some() && model.bonsai.is_none());
    assert!(
        model.weight_bytes < 6_100_000_000,
        "weights expanded: {}",
        model.weight_bytes
    );
    let start = std::time::Instant::now();
    let mut logits = model.forward_prefill(0, &tokens).unwrap();
    let ttft = start.elapsed().as_secs_f64();
    let mut ids = Vec::new();
    let mut decode = Vec::new();
    for _ in 0..64 {
        assert!(logits.iter().all(|v| v.is_finite()));
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        ids.push(next);
        if tokenizer.stop_ids().contains(&next) {
            break;
        }
        logits = model.forward(next).unwrap();
        decode.push(model.last_gpu_seconds);
    }
    let text = tokenizer.decode(&ids, false).unwrap();
    eprintln!(
        "PTQ1_SMOKE {}",
        serde_json::json!({"ids":ids,"text":text,"ttft":ttft,"decode_gpu_s":decode,"weight_bytes":model.weight_bytes,"allocated":model.device.allocated_bytes()})
    );
    assert_eq!(tokenizer.decode(&ids, true).unwrap().trim(), "5");
}

#[test]
fn ptq1_gate_preserves_bf16_values_and_f32_inputs() {
    let device = MetalDevice::new(None).unwrap();
    let (k, n, rows) = (5120, 48, 4);
    let weights = (0..k * n)
        .map(|i| half::bf16::from_f32((i as f32 % 79. - 39.) / 100.))
        .collect::<Vec<_>>();
    let bytes = weights
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let w = device.upload_parts(&[&bytes]).unwrap();
    let x = (0..k * rows)
        .map(|i| (i as f32 % 131. - 65.) / 1000.)
        .collect::<Vec<_>>();
    let input = upload(&device, &x);
    let out = upload(&device, &vec![12345.; n * rows + 16]);
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "ptq1_gate",
        &[&w, &input, &out],
        &[k as u32, n as u32, rows as u32],
        [n, rows, 1],
        256,
    );
    cmd.finish().unwrap();
    let got = unsafe { out.read_f32(0, n * rows + 16) };
    for r in 0..rows {
        for c in 0..n {
            let expected = (0..k)
                .map(|i| weights[c * k + i].to_f64() * x[r * k + i] as f64)
                .sum::<f64>();
            assert!((got[r * n + c] as f64 - expected).abs() < 0.00001);
        }
    }
    assert!(got[n * rows..].iter().all(|&v| v == 12345.));
}

#[test]
fn ptq1_embedding_and_grouped_rotation_match_cpu_butterflies() {
    let device = MetalDevice::new(None).unwrap();
    let k = 6144;
    let (bytes, floats) = plane(k, 3);
    let weights = device.upload_parts(&[&bytes]).unwrap();
    let signs = (0..k)
        .map(|i| if i % 7 < 3 { -1. } else { 1. })
        .collect::<Vec<_>>();
    let sign_buffer = upload(&device, &signs);
    let ids = device
        .upload_parts(&[&[2u32, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()])
        .unwrap();
    let out = upload(&device, &vec![12345.; k * 2 + 16]);
    let fwht = |v: &mut [f32]| {
        for block in v.chunks_exact_mut(1024) {
            for stride in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
                for first in (0..1024).step_by(stride * 2) {
                    for j in first..first + stride {
                        let (a, b) = (block[j], block[j + stride]);
                        block[j] = a + b;
                        block[j + stride] = a - b;
                    }
                }
            }
            for x in block {
                *x /= 32.;
            }
        }
    };
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "ptq1_embed",
        &[&weights, &ids, &sign_buffer, &out],
        &[k as u32, 3, 2],
        [k / 1024, 2, 1],
        256,
    );
    cmd.finish().unwrap();
    let got = unsafe { out.read_f32(0, k * 2 + 16) };
    for (r, id) in [2, 0].into_iter().enumerate() {
        let mut expected = floats[id * k..(id + 1) * k].to_vec();
        fwht(&mut expected);
        for i in 0..k {
            assert_eq!(got[r * k + i], expected[i] * signs[i]);
        }
    }
    assert!(got[k * 2..].iter().all(|&v| v == 12345.));
    let values = (0..k * 2)
        .map(|i| (i % 631) as f32 / 512.)
        .collect::<Vec<_>>();
    let input = upload(&device, &values);
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "ptq1_rotate_grouped",
        &[&input, &sign_buffer, &out],
        &[k as u32, 2],
        [k / 1024, 2, 1],
        256,
    );
    cmd.finish().unwrap();
    let got = unsafe { out.read_f32(0, k * 2 + 16) };
    for r in 0..2 {
        let mut expected = (0..k)
            .map(|i| values[r * k + ((i / 128 % 3) * 16 + i / 128 / 3) * 128 + i % 128] * signs[i])
            .collect::<Vec<_>>();
        fwht(&mut expected);
        assert_eq!(&got[r * k..(r + 1) * k], &expected);
    }
    assert!(got[k * 2..].iter().all(|&v| v == 12345.));
}

#[test]
#[ignore = "requires PADDOCK_METAL_PTQ1_MODEL and PADDOCK_PTQ1_REFERENCE"]
fn ptq1_complete_generations_batch_and_cache() {
    let path = std::env::var("PADDOCK_METAL_PTQ1_MODEL").unwrap();
    let data: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_PTQ1_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let cases = data["cases"].as_array().unwrap();
    let max_tokens = data["max_new_tokens"].as_u64().unwrap() as usize;
    assert!((1..=256).contains(&max_tokens) && !cases.is_empty() && cases.len().is_multiple_of(4));
    let prompts: Vec<_> = cases
        .iter()
        .map(|c| {
            let tokens = tokenizer.encode(c["prompt"].as_str().unwrap()).unwrap();
            assert_eq!(serde_json::json!(tokens), c["tokens"]);
            tokens
        })
        .collect();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    assert!(model.attach_vision(Path::new("/nonexistent")).is_err());
    let mut mismatches = Vec::new();
    let mut batch_mismatches = Vec::new();
    let mut serial = Vec::new();
    let mut hot_restores = 0;
    for live in [1, 4] {
        for first in (0..cases.len()).step_by(live) {
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            let mut outputs = vec![Vec::<u32>::new(); live];
            let mut logits = vec![Vec::new(); live];
            for slot in 0..live {
                model
                    .prefill_begin(slot, prompts[first + slot].clone())
                    .unwrap();
            }
            let start = std::time::Instant::now();
            while !model.pending.is_empty() {
                for (slot, values, _) in model.forward_mixed(&[], 512).unwrap().1 {
                    logits[slot] = values;
                }
            }
            let ttft = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            for step in 0..max_tokens {
                let mut decodes = Vec::new();
                for slot in 0..live {
                    if outputs[slot]
                        .last()
                        .is_some_and(|t| tokenizer.stop_ids().contains(t))
                    {
                        continue;
                    }
                    assert!(logits[slot].iter().all(|v| v.is_finite()));
                    let token = logits[slot]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32;
                    outputs[slot].push(token);
                    if !tokenizer.stop_ids().contains(&token) {
                        decodes.push((slot, token, model.slots[slot].history.len() as u32));
                    }
                }
                if decodes.is_empty() || step + 1 == max_tokens {
                    break;
                }
                let (values, _) = model.forward_mixed(&decodes, 0).unwrap();
                for (i, &(slot, _, _)) in decodes.iter().enumerate() {
                    logits[slot] = values[i * model.vocab..(i + 1) * model.vocab].to_vec();
                }
            }
            let seconds = start.elapsed().as_secs_f64();
            for (slot, output) in outputs.iter().enumerate() {
                let index = first + slot;
                eprintln!(
                    "PTQ1_REFERENCE {}",
                    serde_json::json!({"live":live,"case":index,"ttft_s":ttft,"decode_s":seconds,"generated":output,"text":tokenizer.decode(output,false).unwrap()})
                );
                if serde_json::json!(output) != cases[index]["generated"] {
                    mismatches.push((live, index));
                }
                if live == 1 {
                    serial.push(output.clone());
                } else if output != &serial[index] {
                    batch_mismatches.push(index);
                }
            }
            if live == 1 && prompts[first].len() >= 64 {
                let mut hot = model.forward_prefill(0, &prompts[first]).unwrap();
                let reused = model.take_prefill_reused(0);
                assert!(reused > 0, "expected resident cache restore");
                for (step, &expected) in outputs[0].iter().enumerate() {
                    let token = hot
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32;
                    assert_eq!(token, expected, "hot restore step {step}, reused {reused}");
                    if !tokenizer.stop_ids().contains(&token) && step + 1 < outputs[0].len() {
                        hot = model.forward(token).unwrap();
                    }
                }
                hot_restores += 1;
            }
        }
    }
    assert!(hot_restores > 0);
    eprintln!(
        "PTQ1_PARITY reference_mismatches={mismatches:?} batch_mismatches={batch_mismatches:?}"
    );
    assert!(
        batch_mismatches.is_empty(),
        "batch generation mismatch: {batch_mismatches:?}"
    );
    assert!(
        mismatches.is_empty(),
        "reference generation mismatch: {mismatches:?}"
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_PTQ1_MODEL and PADDOCK_METAL_MMPROJ"]
fn ptq1_vision_color_cohort_cache_and_text_unchanged() {
    let path = std::env::var("PADDOCK_METAL_PTQ1_MODEL").unwrap();
    let mm = std::env::var("PADDOCK_METAL_MMPROJ").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    assert!(model.ternary.is_some());
    let text=tokenizer.encode("<|im_start|>user\nWhat is 2 + 3? Reply with just the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").unwrap();
    let without = model.forward_prefill(0, &text).unwrap();
    model.reset();
    while model.evict_checkpoint() {}
    let weights = model.weight_bytes;
    model.attach_vision(Path::new(&mm)).unwrap();
    let tower = model.weight_bytes - weights;
    assert!(tower < 1_100_000_000);
    assert_eq!(
        model.forward_prefill(0, &text).unwrap(),
        without,
        "attachment changed text logits"
    );
    model.reset();
    while model.evict_checkpoint() {}
    let before = tokenizer
        .encode("<|im_start|>user\n<|vision_start|>")
        .unwrap();
    // Keep at least a full cache page of text after the image. Checkpoints
    // deliberately never split image spans; a shorter suffix can leave both
    // trailing page boundaries inside the image and has no reusable prefix.
    let after = tokenizer
        .encode("<|vision_end|>\nWhat color fills this image? Look at the full picture and answer with one color word only, without an explanation or any other text.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        .unwrap();
    let colors = [
        ([255, 0, 0], "red", 256),
        ([0, 0, 255], "blue", 288),
        ([0, 255, 0], "green", 512),
        ([255, 0, 0], "red", 1024),
    ];
    let prompts = colors
        .iter()
        .map(|&(rgb, _, side)| {
            vec![
                MmChunk::Text(before.clone()),
                MmChunk::Image {
                    rgb: (0..side * side).flat_map(|_| rgb).collect(),
                    w: side,
                    h: side,
                },
                MmChunk::Text(after.clone()),
            ]
        })
        .collect::<Vec<_>>();
    let pick = |row: &[f32]| {
        assert!(row.iter().all(|v| v.is_finite()));
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let mut serial = Vec::new();
    for (case, prompt) in prompts.iter().enumerate() {
        model.reset();
        while model.evict_checkpoint() {}
        let start = std::time::Instant::now();
        let (mut logits, _) = model.prefill_images(0, prompt).unwrap();
        let ttft = start.elapsed().as_secs_f64();
        let mut ids = Vec::new();
        for _ in 0..16 {
            let id = pick(&logits);
            ids.push(id);
            if tokenizer.stop_ids().contains(&id) {
                break;
            }
            logits = model.forward(id).unwrap();
        }
        let reply = tokenizer.decode(&ids, true).unwrap();
        eprintln!(
            "PTQ1_VISION {}",
            serde_json::json!({"case":case,"ids":ids,"text":reply,"ttft":ttft,"tower_bytes":tower,"allocated":model.device.allocated_bytes()})
        );
        assert_eq!(
            reply.trim().trim_end_matches('.').to_lowercase(),
            colors[case].1
        );
        serial.push(ids);
    }
    model.reset();
    while model.evict_checkpoint() {}
    // Drop encoder cache too: the cohort must actually traverse the tower.
    model.image_cache.clear();
    for (_, result) in model.admit_images(prompts.iter().cloned().enumerate().collect()) {
        assert!(!matches!(result, MmAdmit::Failed(_)));
    }
    let mut logits = vec![Vec::new(); 4];
    let deadline = std::time::Instant::now();
    while logits.iter().any(Vec::is_empty) {
        assert!(deadline.elapsed().as_secs() < 120);
        if model.encoding_pending() {
            for (_, result) in model.step_images() {
                assert!(!matches!(result, MmAdmit::Failed(_)));
            }
        }
        for (slot, row, _) in model.forward_mixed(&[], 512).unwrap().1 {
            logits[slot] = row;
        }
    }
    let mut outputs = vec![Vec::new(); 4];
    for _ in 0..16 {
        let mut decodes = Vec::new();
        for slot in 0..4 {
            if outputs[slot]
                .last()
                .is_some_and(|id| tokenizer.stop_ids().contains(id))
            {
                continue;
            }
            let id = pick(&logits[slot]);
            outputs[slot].push(id);
            if !tokenizer.stop_ids().contains(&id) {
                decodes.push((slot, id, model.slots[slot].history.len() as u32));
            }
        }
        if decodes.is_empty() {
            break;
        }
        let (rows, _) = model.forward_mixed(&decodes, 0).unwrap();
        for (i, &(slot, _, _)) in decodes.iter().enumerate() {
            logits[slot] = rows[i * model.vocab..(i + 1) * model.vocab].to_vec();
        }
    }
    assert_eq!(outputs, serial, "cold cohort changed complete generations");
    model.reset();
    let (mut hot, _) = model.prefill_images(0, &prompts[3]).unwrap();
    assert!(model.take_prefill_reused(0) > 0);
    for (i, &id) in serial[3].iter().enumerate() {
        assert_eq!(pick(&hot), id, "hot image-prefix step {i}");
        if i + 1 < serial[3].len() {
            hot = model.forward(id).unwrap();
        }
    }
    eprintln!("PTQ1_VISION cohort and resident image-prefix pass");
}
