use super::*;
use paddock_engine::{generator::MmAdmit, service::MmChunk};
use paddock_models::safetensors::SafetensorsFile;

#[test]
fn gemma_mlx_resize_is_not_gguf_centered_padding() {
    assert_eq!(vision::resize_mlx(256, 256).unwrap(), (768, 768));
    assert_eq!(vision::resize(256, 256).unwrap(), (432, 432));
    for (w, h) in [(1, 1), (65536, 1), (1, 65536), (1920, 1080), (601, 799)] {
        let (x, y) = vision::resize_mlx(w, h).unwrap();
        assert!(x > 0 && y > 0 && x % 48 == 0 && y % 48 == 0 && x * y <= 280 * 2304);
    }
    assert!(vision::resize_mlx(0, 256).is_err());
}

#[test]
#[ignore = "requires downloaded Gemma/Muse MLX checkpoint and GPU reference fixtures"]
fn mlx_multimodal_checkpoint_parity() {
    let path = std::env::var("PADDOCK_MM_MLX").unwrap();
    let reference = std::env::var("PADDOCK_MM_MLX_REFERENCE").unwrap();
    let root = Path::new(&reference);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let mut m = Gemma4::load(Path::new(&path), 8192, 4, None).unwrap();
    assert!(m.mlx && m.vision.is_some());
    let tok = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let mut matched = 0;
    let mut results = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        m.reset();
        while m.evict() {}
        let name = case["name"].as_str().unwrap();
        let ids: Vec<_> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected: Vec<_> = case["generated"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mut feature_max = None;
        let started = std::time::Instant::now();
        let mut logits = if let Some(rgb) = case["rgb"].as_str() {
            let rgb = std::fs::read(root.join(rgb)).unwrap();
            let w = case["w"].as_u64().unwrap() as usize;
            let h = case["h"].as_u64().unwrap() as usize;
            let tower = m.vision.as_ref().unwrap();
            let mut job = tower.start(&m.device, &[(&rgb, w, h)]).unwrap();
            let encoded = loop {
                if let Some(mut out) = tower
                    .step(&m.device, &mut job, std::time::Duration::ZERO)
                    .unwrap()
                {
                    break out.remove(0);
                }
            };
            let fixture =
                SafetensorsFile::open(&root.join(format!("{name}-features.safetensors"))).unwrap();
            let (_, bytes) = fixture.bytes("features").unwrap();
            assert_eq!(
                encoded.tokens * m.width * 4,
                bytes.len(),
                "image geometry parity"
            );
            // SAFETY: tower.step completed the GPU encoder command.
            let actual = unsafe { encoded.embd.read_f32(0, bytes.len() / 4) };
            feature_max = Some(
                actual
                    .iter()
                    .zip(bytes.chunks_exact(4))
                    .map(|(a, b)| (a - f32::from_le_bytes(b.try_into().unwrap())).abs())
                    .fold(0f32, f32::max),
            );
            let (begin, end) = m.image_markers.unwrap();
            let lo = ids.iter().position(|v| *v == begin).unwrap();
            let hi = ids.iter().rposition(|v| *v == end).unwrap();
            assert_eq!(hi - lo - 1, encoded.tokens, "expanded placeholder count");
            let chunks = vec![
                MmChunk::Text(ids[..lo].to_vec()),
                MmChunk::Image { rgb, w, h },
                MmChunk::Text(ids[hi + 1..].to_vec()),
            ];
            assert!(matches!(
                m.admit_images(vec![(0, chunks)])[0].1,
                MmAdmit::Encoding
            ));
            while m.encoding_pending() {
                assert!(
                    m.step_images()
                        .iter()
                        .all(|(_, v)| matches!(v, MmAdmit::Queued))
                );
            }
            loop {
                let (_, mut done) = m.forward_mixed(&[], CHUNK).unwrap();
                if let Some((slot, logits, n)) = done.pop() {
                    assert_eq!(slot, 0);
                    assert_eq!(n, ids.len());
                    break logits;
                }
            }
        } else {
            m.forward_prefill(0, &ids).unwrap()
        };
        let prefill = started.elapsed().as_secs_f64();
        let fixture = SafetensorsFile::open(&root.join(format!("{name}.safetensors"))).unwrap();
        let (_, bytes) = fixture.bytes("logits").unwrap();
        let oracle: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let max = logits
            .iter()
            .zip(&oracle)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mut generated = Vec::new();
        for step in 0..expected.len() {
            assert!(logits.iter().all(|v| v.is_finite()));
            let token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            generated.push(token);
            if step + 1 < expected.len() {
                logits = m
                    .execute(&[(0, token, (ids.len() + step) as u32)], &[0])
                    .unwrap();
            }
        }
        matched += usize::from(generated == expected);
        let result = serde_json::json!({"name":name,"generated":generated,"expected":expected,
            "text":tok.decode(&generated,false).unwrap(),"prefill_s":prefill,"wall_s":started.elapsed().as_secs_f64(),"max_first_logit_error":max,"feature_max_error":feature_max});
        eprintln!("MLX_PARITY {result}");
        results.push(result);
    }
    eprintln!("MLX_PARITY_SUMMARY matched={matched}/{}", results.len());
    if let Ok(output) = std::env::var("PADDOCK_MM_MLX_REPORT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        file.write_all(serde_json::to_string_pretty(&results).unwrap().as_bytes())
            .unwrap();
    }
    assert_eq!(matched, results.len(), "complete greedy stream gate");
}

#[test]
#[ignore = "requires Gemma/Muse MLX checkpoint and text reference prompts; native c=4 isolation"]
fn mlx_four_slot_admission_and_decode_identity() {
    let path = std::env::var("PADDOCK_MM_MLX").unwrap();
    let reference = std::env::var("PADDOCK_MM_MLX_REFERENCE").unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(Path::new(&reference).join("manifest.json")).unwrap(),
    )
    .unwrap();
    let prompts: Vec<Vec<u32>> = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .take(4)
        .map(|c| {
            c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        })
        .collect();
    assert_eq!(prompts.len(), 4);
    let pick = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let mut model = Gemma4::load(Path::new(&path), 8192, 4, None).unwrap();
    let mut serial = Vec::new();
    for prompt in &prompts {
        model.reset();
        while model.evict() {}
        let first = pick(&model.forward_prefill(0, prompt).unwrap());
        let second = pick(
            &model
                .execute(&[(0, first, prompt.len() as u32)], &[0])
                .unwrap(),
        );
        serial.push((first, second));
    }
    model.reset();
    while model.evict() {}
    for (slot, prompt) in prompts.iter().enumerate() {
        model.prefill_begin(slot, prompt.clone()).unwrap();
    }
    let mut completed = [false; 4];
    for _ in 0..100 {
        let (_, done) = model.forward_mixed(&[], CHUNK).unwrap();
        for (slot, logits, n) in done {
            assert!(!completed[slot]);
            completed[slot] = true;
            assert_eq!(n, prompts[slot].len());
            assert_eq!(pick(&logits), serial[slot].0, "prefill slot {slot}");
        }
        if completed.iter().all(|v| *v) {
            break;
        }
    }
    assert!(
        completed.iter().all(|v| *v),
        "admission failed to make progress"
    );
    let rows: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(slot, prompt)| (slot, serial[slot].0, prompt.len() as u32))
        .collect();
    let logits = model.execute(&rows, &[0, 1, 2, 3]).unwrap();
    for (slot, logits) in logits.chunks_exact(model.vocab).enumerate() {
        assert_eq!(pick(logits), serial[slot].1, "decode slot {slot}");
    }
    model.reset();
    model.prefill_begin(2, prompts[2].clone()).unwrap();
    assert!(model.prefill_abort(2));
    assert!(model.pending.is_empty());
}
