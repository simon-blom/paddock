use super::*;
use paddock_tokenizer::GgufTokenizer;

fn pixels(w: usize, h: usize, swap: bool) -> Vec<f32> {
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let a = if x < w / 3 {
                0.
            } else if x < 2 * w / 3 {
                0.5
            } else {
                1.
            };
            let r = x as f32 / (w - 1) as f32;
            let b = y as f32 / (h - 1) as f32;
            let rgb = if swap { [b, 0.25, r] } else { [r, 0.25, b] };
            [
                rgb[0] * 2. - 1.,
                rgb[1] * 2. - 1.,
                rgb[2] * 2. - 1.,
                a * 2. - 1.,
            ]
        })
        .collect()
}
fn save(dir: &Path, name: &str, data: &[f32]) {
    assert!(data.iter().all(|v| v.is_finite()), "{name}");
    std::fs::write(
        dir.join(name),
        data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}
fn prompt(tok: &GgufTokenizer, grids: &[(usize, usize)]) -> (Vec<u32>, usize, u32) {
    let sys = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    let images = (1..=grids.len())
        .map(|i| format!("<image{i}><|vision_start|><|image_pad|><|vision_end|>"))
        .collect::<Vec<_>>()
        .join(" ");
    let instruction = std::env::var("PADDOCK_QI_EDIT_PROMPT")
        .unwrap_or_else(|_| "Keep the composition and make the colours brighter.".into());
    let full =
        format!("{sys}<|im_start|>user\n{images}{instruction}<|im_end|>\n<|im_start|>assistant\n");
    let drop = tok.encode(sys).unwrap().len();
    let pad = tok.encode("<|image_pad|>").unwrap()[0];
    let mut at = 0;
    let ids = tok
        .encode(&full)
        .unwrap()
        .into_iter()
        .flat_map(|id| {
            let n = if id == pad {
                let (w, h) = grids[at];
                at += 1;
                w * h
            } else {
                1
            };
            std::iter::repeat_n(id, n)
        })
        .collect();
    (ids, drop, pad)
}

#[test]
#[ignore = "requires Metal GPU"]
fn reference_patches_preserve_merge_order_and_alpha() {
    let exec = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let (w, h) = (64, 96);
    let host = pixels(w, h, false);
    let rgba = exec.to_device(&host).unwrap();
    for channels in [768, 1536] {
        let n = w * h / 256 * channels;
        let output = exec.alloc::<f32>(n).unwrap();
        exec.run(
            "qi_vision_patches",
            &[&rgba, &output],
            &[
                w as u32,
                h as u32,
                channels as u32,
                0.5f32.to_bits(),
                0.5f32.to_bits(),
                0.5f32.to_bits(),
                0.5f32.to_bits(),
                0.5f32.to_bits(),
                0.5f32.to_bits(),
            ],
            [n.div_ceil(256), 1, 1],
            256,
        )
        .unwrap();
        let result = unsafe { output.read_f32(0, n) };
        for (i, &v) in result.iter().enumerate() {
            let (row, d) = (i / channels, i % channels);
            let c = d / (channels / 3);
            let px = d % 256;
            let x = (row / 4 % (w / 32)) * 32 + (row % 2) * 16 + px % 16;
            let y = (row / 4 / (w / 32)) * 32 + (row % 4 / 2) * 16 + px / 16;
            let at = (y * w + x) * 4;
            let a = (host[at + 3] + 1.) * 0.5;
            let expected = ((host[at + c] + 1.) * 0.5 * a + (1. - a)) * 2. - 1.;
            assert!((v - expected).abs() < 1e-6, "{i}: {v} vs {expected}");
        }
    }
}

#[test]
#[ignore = "requires Metal GPU"]
fn vision_affine_preserves_f32_contract_and_ragged_tiles() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let (k, n) = (192usize, 37usize);
    let scale = half::bf16::from_f32(0.03125);
    let bias = half::bf16::from_f32(-0.1875);
    let codes = (0..k * n / 8)
        .map(|i| (0..8).fold(0u32, |v, j| v | (((i * 8 + j) % 16) as u32) << (j * 4)))
        .collect::<Vec<_>>();
    let mut bytes = codes
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    for v in [scale, bias] {
        for _ in 0..k * n / 64 {
            bytes.extend(v.to_bits().to_le_bytes());
        }
    }
    let w = e.device.upload(&bytes).unwrap();
    for rows in [1, 17, 33, 64] {
        let x = (0..rows * k)
            .map(|i| ((i as f32) * 0.071).sin() * 0.3)
            .collect::<Vec<_>>();
        let xb = e.to_device(&x).unwrap();
        let y = e.alloc::<f32>(rows * n).unwrap();
        e.run(
            "qi_vision_affine",
            &[&w, &xb, &y],
            &[k as u32, n as u32, rows as u32],
            [n.div_ceil(32), rows.div_ceil(32), 1],
            128,
        )
        .unwrap();
        let actual = unsafe { y.read_f32(0, rows * n) };
        for r in 0..rows {
            for c in 0..n {
                let expected = (0..k)
                    .map(|j| {
                        f64::from(x[r * k + j])
                            * f64::from(((c * k + j) % 16) as f32 * scale.to_f32() + bias.to_f32())
                    })
                    .sum::<f64>() as f32;
                assert!(
                    (actual[r * n + c] - expected).abs() < 2e-5,
                    "row {r} col {c}: {} vs {expected}",
                    actual[r * n + c]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires Metal GPU"]
fn vision_splice_and_deepstack_leave_other_tokens_untouched() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    for bf in [0, 1] {
        // This feature rounds onto a BF16 half-ULP tie at x=0.5. Rounding
        // only after addition instead of before it gives a different answer.
        let feature = 0.0019533;
        let image = e.to_device(&vec![feature; 2 * 4096]).unwrap();
        for add in [0, 1] {
            let x = e.to_device(&vec![0.5; 5 * 4096]).unwrap();
            e.run(
                "qi_text_splice",
                &[&image, &x],
                &[2, 1, add, bf],
                [32, 1, 1],
                256,
            )
            .unwrap();
            let actual = unsafe { x.read_f32(0, 5 * 4096) };
            let round = |v| {
                if bf == 1 {
                    half::bf16::from_f32(v).to_f32()
                } else {
                    v
                }
            };
            let expected = if add == 0 {
                round(feature)
            } else {
                round(0.5 + round(feature))
            };
            assert!(
                actual[..4096]
                    .iter()
                    .chain(&actual[3 * 4096..])
                    .all(|v| *v == 0.5)
            );
            assert!(actual[4096..3 * 4096].iter().all(|v| *v == expected));
        }
    }
}

#[test]
#[ignore = "requires pinned MLX checkpoint and Metal"]
fn mlx_reference_edit_conditioning_and_generation() {
    real_model(true);
}
#[test]
#[ignore = "requires Qwen-Image GGUF and vision companion, Metal"]
fn gguf_reference_edit_conditioning_and_generation() {
    real_model(false);
}

#[test]
#[ignore = "requires a raw RGBA fixture, Qwen-Image MLX VAE and Metal"]
fn reference_vae_full_resolution_diagnostic() {
    let dir = std::env::var("PADDOCK_QI_VAE_FIXTURE").expect("VAE fixture directory");
    let dir = Path::new(&dir);
    let root = std::env::var("PADDOCK_QI_MODELS").expect("models directory");
    let dims: [usize; 2] =
        serde_json::from_slice(&std::fs::read(dir.join("size.json")).unwrap()).unwrap();
    let [w, h] = dims;
    let bytes = std::fs::read(dir.join("rgba.f32")).unwrap();
    assert_eq!(bytes.len(), w * h * 16);
    let rgba = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let exec = Rc::new(Ops {
        device: MetalDevice::new(None).unwrap(),
    });
    let path = Path::new(&root).join("Qwen-Image-2.1-MLX-4bit/vae/model.safetensors");
    let encoder = vae_encode::Encoder::load(exec.clone(), &path).unwrap();
    let input = exec.to_device(&rgba).unwrap();
    let latent = encoder.encode(&input, w, h, &|| false).unwrap();
    save(dir, "latents.f32", &unsafe {
        latent.read_f32(0, latent.len() / 4)
    });
    drop(encoder);
    let decoder = vae::VaeDecoder::load(exec, &path).unwrap();
    let image = decoder.decode(&latent, w / 16, h / 16).unwrap();
    std::fs::write(dir.join("roundtrip.rgba"), image).unwrap();
}

fn real_model(mlx: bool) {
    let root = std::env::var("PADDOCK_QI_MODELS").expect("models directory");
    let root = Path::new(&root);
    let mlx_root = root.join("Qwen-Image-2.1-MLX-4bit");
    let (mut model, tok) = if mlx {
        (
            QwenImage::load_mlx(&mlx_root, 8192, None).unwrap(),
            GgufTokenizer::from_hf_dir(&mlx_root.join("processor")).unwrap(),
        )
    } else {
        let text = root.join("Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf");
        let map = paddock_models::mapped::MappedGguf::open(&text).unwrap();
        let tok = GgufTokenizer::from_gguf(map.gguf()).unwrap();
        let model = QwenImage::load_with_vision(
            &root.join("Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf"),
            &text,
            &root.join("Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors"),
            Some(&root.join("Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf")),
            8192,
            None,
        )
        .unwrap();
        (model, tok)
    };
    assert!(model.can_edit());
    let (w, h, rgba) = if let Some(dir) = std::env::var_os("PADDOCK_QI_EDIT_REFERENCE") {
        let dir = Path::new(&dir);
        let [w, h]: [usize; 2] =
            serde_json::from_slice(&std::fs::read(dir.join("size.json")).unwrap()).unwrap();
        let data = std::fs::read(dir.join("rgba.f32")).unwrap();
        assert_eq!(data.len(), w * h * 16);
        (
            w,
            h,
            data.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect(),
        )
    } else {
        (128, 192, pixels(128, 192, false))
    };
    let changed = pixels(w, h, true);
    let reference = Reference {
        rgba: &rgba,
        width: w,
        height: h,
    };
    let (ids, drop, pad) = prompt(&tok, &[(w / 32, h / 32)]);
    let req = GenerateRequest {
        prompt_ids: &ids,
        drop,
        negative: None,
        width: 128,
        height: 128,
        steps: 2,
        seed: 42,
        noise_offset: 0,
        guidance: 1.,
        references: std::slice::from_ref(&reference),
        image_pad_id: pad,
    };
    assert!(
        model
            .render(&req, 0, &mut |_, _| Ok(()), &|| true)
            .err()
            .unwrap()
            .to_string()
            .contains("went away")
    );
    if let Some(dir) = std::env::var_os("PADDOCK_QI_EDIT_FIXTURE") {
        let dir = Path::new(&dir);
        let exec = &model.exec;
        let lane = model.edit.as_ref().unwrap();
        save(dir, "rgba.f32", &rgba);
        let pixels = exec.to_device(&rgba).unwrap();
        let z = lane.encoder.encode(&pixels, w, h, &|| false).unwrap();
        let vision = lane.tower.encode(exec, &pixels, w, h, &|| false).unwrap();
        save(dir, "latents.f32", &unsafe { z.read_f32(0, z.len() / 4) });
        save(dir, "vision.f32", &unsafe {
            vision.embd.read_f32(0, vision.embd.len() / 4)
        });
        for (i, ds) in vision.deepstack.iter().enumerate() {
            save(dir, &format!("deepstack-{i}.f32"), &unsafe {
                ds.read_f32(0, ds.len() / 4)
            });
        }
        let runs = conditioning::slots(&ids, drop, pad, &[(w / 32, h / 32)]).unwrap();
        let hidden = model
            .text
            .encode_vl(exec, &ids, drop, &[&vision], &runs, &|| false)
            .unwrap();
        save(dir, "hidden.f32", &unsafe {
            hidden.read_f32(0, hidden.len() / 4)
        });
        std::fs::write(dir.join("metadata.json"),serde_json::to_vec(&serde_json::json!({"width":w,"height":h,"ids":ids,"drop":drop,"runs":runs,"positions":conditioning::text_positions(ids.len(),&runs,&[(w/32,h/32)])})).unwrap()).unwrap();
        let size = std::env::var("PADDOCK_QI_EDIT_SIZE")
            .ok()
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(256);
        let steps = std::env::var("PADDOCK_QI_EDIT_STEPS")
            .ok()
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(2);
        let n = size * size / 256;
        let noise = exec.alloc::<f32>(n * 64).unwrap();
        exec.run(
            "qi_noise",
            &[&noise],
            &[n as u32, 42, 0, 0],
            [(n * 64).div_ceil(256), 1, 1],
            256,
        )
        .unwrap();
        save(dir, "initial.f32", &unsafe { noise.read_f32(0, n * 64) });
        std::fs::write(dir.join("generation.json"),serde_json::to_vec(&serde_json::json!({"size":size,"sigmas":Schedule::new(steps,mu_for_tokens(n)).sigmas})).unwrap()).unwrap();
        if std::env::var("PADDOCK_QI_EDIT_CAPTURE_ONLY").as_deref() == Ok("conditioning") {
            return;
        }
        let start = std::time::Instant::now();
        let image = model
            .render(
                &GenerateRequest {
                    width: size,
                    height: size,
                    steps,
                    ..req
                },
                0,
                &mut |_, _| Ok(()),
                &|| false,
            )
            .unwrap();
        eprintln!(
            "reference edit {size}x{size}/{steps} steps: {:.3}s",
            start.elapsed().as_secs_f64()
        );
        std::fs::write(dir.join("generation.rgba"), image.pixels).unwrap();
        if std::env::var_os("PADDOCK_QI_EDIT_CAPTURE_ONLY").is_some() {
            return;
        }
    }
    let baseline = model
        .render(&req, 0, &mut |_, _| Ok(()), &|| false)
        .unwrap();
    let again = model
        .render(&req, 0, &mut |_, _| Ok(()), &|| false)
        .unwrap();
    assert_eq!(
        baseline.pixels, again.pixels,
        "fixed references and seed must be repeatable"
    );
    let other = Reference {
        rgba: &changed,
        width: w,
        height: h,
    };
    let altered = model
        .render(
            &GenerateRequest {
                references: std::slice::from_ref(&other),
                ..req
            },
            0,
            &mut |_, _| Ok(()),
            &|| false,
        )
        .unwrap();
    assert_ne!(
        baseline.pixels, altered.pixels,
        "reference content must affect generation"
    );
    // Guidance must use the SAME images for both prompts, and reduce exactly
    // to the conditional path when their text is identical.
    let guided = model
        .render(
            &GenerateRequest {
                negative: Some((&ids, drop)),
                guidance: 2.,
                ..req
            },
            0,
            &mut |_, _| Ok(()),
            &|| false,
        )
        .unwrap();
    assert_eq!(baseline.pixels, guided.pixels);
    let mut bad_ids = ids.clone();
    bad_ids.retain(|id| *id != pad);
    assert!(
        model
            .render(
                &GenerateRequest {
                    prompt_ids: &bad_ids,
                    ..req
                },
                0,
                &mut |_, _| Ok(()),
                &|| false
            )
            .is_err()
    );
    // Multiple differently shaped references must get distinct slots/frames.
    let second = Reference {
        rgba: &rgba,
        width: h,
        height: w,
    };
    let refs = [
        Reference {
            rgba: &rgba,
            width: w,
            height: h,
        },
        second,
    ];
    let (ids2, drop2, _) = prompt(&tok, &[(w / 32, h / 32), (h / 32, w / 32)]);
    let multi = model.render(
        &GenerateRequest {
            prompt_ids: &ids2,
            drop: drop2,
            references: &refs,
            ..req
        },
        1,
        &mut |_, _| Err("receiver closed".into()),
        &|| false,
    );
    assert!(multi.err().unwrap().to_string().contains("receiver closed"));
}
