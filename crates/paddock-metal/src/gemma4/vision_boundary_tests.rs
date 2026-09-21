//! Test-only substitution of independently captured GPU embeddings. This
//! code is not linked into the runner and never serves as a model backend.
use super::*;
use paddock_engine::service::MmChunk;

#[test]
#[ignore = "canonical greedy-parity.py GPU boundary diagnostic only"]
fn gemma_vision_backbone_boundary_capture() {
    boundary_capture(false);
}

#[test]
#[ignore = "canonical greedy-parity.py GPU boundary diagnostic only"]
fn muse_vision_backbone_boundary_capture() {
    boundary_capture(true);
}

fn boundary_capture(muse: bool) {
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_METAL_BOUNDARY_FIXTURE").expect("fixture")).unwrap(),
    )
    .unwrap();
    let model = std::env::var(if muse {
        "PADDOCK_MUSE_GGUF"
    } else {
        "PADDOCK_GEMMA4_GGUF"
    })
    .expect("target");
    let mm = std::env::var(if muse {
        "PADDOCK_MUSE_MMPROJ"
    } else {
        "PADDOCK_GEMMA4_MMPROJ"
    })
    .expect("mmproj");
    let mut m = Gemma4::load(Path::new(&model), 4096, 4, None).unwrap();
    m.attach_vision(Path::new(&mm)).unwrap();
    let ids = |key: &str| {
        fixture[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let chunks = vec![
        MmChunk::Text(ids("before")),
        MmChunk::Image {
            rgb: std::fs::read(fixture["rgb"].as_str().unwrap()).unwrap(),
            w: fixture["w"].as_u64().unwrap() as usize,
            h: fixture["h"].as_u64().unwrap() as usize,
        },
        MmChunk::Text(ids("after")),
    ];
    let continuation = ids("continuation");
    let reference = std::fs::read(fixture["reference_embedding"].as_str().unwrap()).unwrap();
    let tokens = i32::from_le_bytes(reference[..4].try_into().unwrap()) as usize;
    let width = i32::from_le_bytes(reference[4..8].try_into().unwrap()) as usize;
    assert_eq!(width, m.width);
    assert_eq!(reference.len(), 8 + tokens * width * 4);
    let mut capture = serde_json::Map::new();
    for (name, exact_embedding, chunks_like_reference) in [
        ("native_embeddings", false, 0),
        ("reference_embeddings", true, 0),
        ("native_media_chunks", false, 1),
        ("reference_media_chunks", true, 1),
        ("native_reference_checkpoints", false, 2),
        ("reference_reference_checkpoints", true, 2),
    ] {
        // Muse's causal images may exceed CHUNK; the canonical mixed path
        // below already chunks them safely. Gemma's atomic-media shape
        // experiments are not a valid Muse execution schedule.
        if muse && chunks_like_reference > 0 {
            continue;
        }
        m.reset();
        // A reference-image replay must not adopt KV computed from the native
        // image in the preceding pass. Reset releases slots, not checkpoints.
        while m.evict() {}
        m.admit_images(vec![(0, chunks.clone())]);
        while m.image_slot_pending(0) {
            m.step_images();
        }
        if exact_embedding {
            let output = &mut m.slots[0].mm.as_mut().unwrap().images[0];
            assert_eq!(tokens, output.tokens);
            output.embd = m.device.upload(&reference[8..]).unwrap();
        }
        let mut logits = if chunks_like_reference > 0 {
            // Diagnostic only: the reference log separates text/image/text,
            // plus checkpoints after BOS and four rows before the final
            // prompt boundary. Test shape sensitivity without changing the
            // production scheduler or using external inference as a backend.
            let prompt = m.pending.pop_front().unwrap();
            assert_eq!(prompt.offset, 0);
            let image_start = ids("before").len() + 1;
            let image_end = image_start + tokens;
            let mut ends = vec![image_start, image_end, prompt.tokens.len()];
            if chunks_like_reference == 2 {
                assert!(prompt.tokens.len() - image_end > 4);
                ends.extend([1, prompt.tokens.len() - 4]);
                ends.sort_unstable();
            }
            let mut at = 0;
            let mut output = Vec::new();
            for end in ends {
                let rows = (at..end)
                    .map(|i| (0, prompt.tokens[i], i as u32))
                    .collect::<Vec<_>>();
                let selected = [end - at - 1];
                output = m
                    .execute(
                        &rows,
                        if end == prompt.tokens.len() {
                            &selected
                        } else {
                            &[]
                        },
                    )
                    .unwrap();
                at = end;
            }
            output
        } else {
            loop {
                let (_, mut done) = m.forward_mixed(&[], CHUNK).unwrap();
                if !done.is_empty() {
                    break done.remove(0).1;
                }
            }
        };
        let mut rows = Vec::new();
        for token in &continuation {
            let mut order = (0..logits.len()).collect::<Vec<_>>();
            order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            rows.push(
                order[..10]
                    .iter()
                    .map(|&i| serde_json::json!([i, logits[i]]))
                    .collect::<Vec<_>>(),
            );
            logits = m.forward(*token).unwrap();
        }
        capture.insert(name.into(), serde_json::json!(rows));
    }
    eprintln!("VISION_BACKBONE {}", serde_json::Value::Object(capture));
}
