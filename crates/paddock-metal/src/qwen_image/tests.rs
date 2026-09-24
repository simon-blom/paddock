use super::*;

#[test]
#[ignore = "requires Metal GPU"]
fn prefix_copy_preserves_bits_and_target_tail() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let k: Vec<_> = (0..37u32)
        .map(|i| 0x7f800000u32.wrapping_add(i * 0x12345))
        .collect();
    let v: Vec<_> = k.iter().map(|v| !v).collect();
    let kb = words(&e.device, &k).unwrap();
    let vb = words(&e.device, &v).unwrap();
    let joined_k = words(&e.device, &[0xdeadbeef; 89]).unwrap();
    let joined_v = words(&e.device, &[0xbaadf00d; 89]).unwrap();
    e.run(
        "qi_prefix_copy",
        &[&kb, &vb, &joined_k, &joined_v],
        &[37],
        [1, 1, 1],
        256,
    )
    .unwrap();
    let ak = unsafe { joined_k.read_u32(89) };
    let av = unsafe { joined_v.read_u32(89) };
    assert_eq!(&ak[..37], k);
    assert_eq!(&av[..37], v);
    assert!(ak[37..].iter().all(|&v| v == 0xdeadbeef));
    assert!(av[37..].iter().all(|&v| v == 0xbaadf00d));
}

#[test]
#[ignore = "requires Metal GPU"]
fn text_rope_preserves_intermediate_bf16_rounding() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let rows = 31;
    let input = e.to_device(&vec![1f32; rows * 128]).unwrap();
    let mut weights = vec![1f32; 128];
    weights[17] = 9.4375;
    weights[81] = -1.359375;
    let norm = e.to_device(&weights).unwrap();
    let meta = words(
        &e.device,
        &(0..rows).flat_map(|i| [0, i as u32]).collect::<Vec<_>>(),
    )
    .unwrap();
    let out = e.device.alloc(rows * 128 * 2).unwrap();
    e.run(
        "qi_mlx_text_rope",
        &[&input, &norm, &meta, &out, &input, &out],
        &[1, 0, 5000000f32.to_bits(), 1e-6f32.to_bits(), 0],
        [1, rows, 1],
        32,
    )
    .unwrap();
    let actual: Vec<_> = unsafe { out.read_u32(rows * 64) }
        .into_iter()
        .flat_map(|v| [v as u16, (v >> 16) as u16])
        .map(|v| half::f16::from_bits(v).to_f32())
        .collect();
    let bf = |v| half::bf16::from_f32(v).to_f32();
    // If the compiler contracts away the product's BF16 materialization,
    // this becomes 9.5 instead of 9.4375. This is not a tolerance gate.
    assert_eq!(actual[4 * 128 + 17], 9.4375);
    for row in 0..rows {
        for d in 0..128 {
            let angle = row as f32 * 5000000f32.powf(-((d % 64) as f32) / 64.);
            let a = bf(weights[d] / (1f32 + 1e-6).sqrt());
            let b = bf(weights[(d + 64) % 128] / (1f32 + 1e-6).sqrt());
            let expected =
                bf(bf(a * bf(angle.cos())) + bf((if d < 64 { -b } else { b }) * bf(angle.sin())));
            assert!(
                (actual[row * 128 + d] - expected).abs() <= 0.00001,
                "row {row}, d {d}: {} != {expected}",
                actual[row * 128 + d]
            );
        }
    }
}

#[test]
#[ignore = "requires Metal GPU"]
fn text_swiglu_materializes_bf16_contract() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let bf = |v| half::bf16::from_f32(v).to_f32();
    let gate: Vec<_> = (0..1024).map(|i| bf((i as f32 - 512.) / 43.)).collect();
    let up: Vec<_> = (0..1024)
        .map(|i| bf((i as f32 * 0.31).sin() * 5.))
        .collect();
    let g = e.to_device(&gate).unwrap();
    let u = e.to_device(&up).unwrap();
    e.run("qi_mlx_swiglu", &[&g, &u], &[1024], [4, 1, 1], 256)
        .unwrap();
    let actual = unsafe { g.read_f32(0, 1024) };
    for i in 0..1024 {
        let v = gate[i];
        // CPU libm and Metal exp can straddle a BF16 midpoint by one F32
        // ULP (exp(6.84375) = 938.000047832). Propagate those adjacent exp
        // values through the exact rounding contract, not an output tolerance.
        let exp = (f64::from(v.abs()).exp() as f32).to_bits();
        let candidates = [exp - 1, exp, exp + 1].map(|bits| {
            let tail = bf(1. / bf(1. + bf(f32::from_bits(bits))));
            let sigmoid = if v < 0. { tail } else { bf(1. - tail) };
            bf(bf(v * sigmoid) * up[i])
        });
        assert!(
            candidates.contains(&actual[i]),
            "input {v}, up {}, output {}, allowed {candidates:?}",
            up[i],
            actual[i]
        );
    }
}

#[test]
#[ignore = "requires Metal GPU"]
fn text_f32_attention_matches_scalar_causal_gqa() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let rows = 37;
    let make = |n: usize, phase: f32| {
        (0..n)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.037 + phase).sin()).to_f32())
            .collect::<Vec<_>>()
    };
    let q = make(rows * 32 * 128, 0.1);
    let k = make(rows * 8 * 128, 0.7);
    let v = make(rows * 8 * 128, 1.1);
    let upload = |values: &[f32]| {
        e.f16_to_device(
            &values
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (qb, kb, vb) = (upload(&q), upload(&k), upload(&v));
    let out = e.device.alloc(rows * 32 * 128 * 4).unwrap();
    e.run(
        "qi_text_attention",
        &[&qb, &kb, &vb, &out],
        &[],
        [32, rows, 1],
        32,
    )
    .unwrap();
    let actual = unsafe { out.read_f32(0, rows * 32 * 128) };
    for row in 0..rows {
        for h in 0..32 {
            let scores: Vec<_> = (0..=row)
                .map(|t| {
                    (0..128)
                        .map(|d| {
                            f64::from(q[(row * 32 + h) * 128 + d])
                                * f64::from(k[(t * 8 + h / 4) * 128 + d])
                        })
                        .sum::<f64>()
                        / 128f64.sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probs: Vec<_> = scores.iter().map(|v| (v - max).exp()).collect();
            let sum = probs.iter().sum::<f64>();
            for d in 0..128 {
                let expected = (probs
                    .iter()
                    .enumerate()
                    .map(|(t, p)| p * f64::from(v[(t * 8 + h / 4) * 128 + d]))
                    .sum::<f64>()
                    / sum) as f32;
                let rounded = half::bf16::from_f32(expected);
                let ulp = (half::bf16::from_bits(rounded.to_bits() ^ 1).to_f32()
                    - rounded.to_f32())
                .abs();
                assert!(
                    (actual[(row * 32 + h) * 128 + d] - expected).abs() <= ulp * 0.501 + 2e-6,
                    "row {row} head {h} col {d}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires pinned MLX checkpoint; isolated denoising timing"]
fn mlx_denoising_benchmark() {
    let root = std::env::var("PADDOCK_QI_MLX").unwrap();
    let fixture = std::env::var("PADDOCK_QI_FIXTURE").unwrap();
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let dit = dit::Dit::load_mlx(&e.device, &Path::new(&root).join("transformer")).unwrap();
    let data = std::fs::read(Path::new(&fixture).join("hidden.f32")).unwrap();
    let rows = data.len() / 4 / 4096;
    let hidden = e.device.upload(&data).unwrap();
    let sizes = std::env::var("PADDOCK_QI_BENCH_SIZE")
        .map(|v| vec![v.parse::<usize>().expect("benchmark size")])
        .unwrap_or_else(|_| vec![512, 1024]);
    let iterations = std::env::var("PADDOCK_QI_BENCH_ITERATIONS")
        .map(|v| v.parse::<usize>().expect("benchmark iterations"))
        .unwrap_or(5);
    assert!(iterations > 0);
    for size in sizes {
        assert!((32..=2752).contains(&size) && size.is_multiple_of(32));
        let n = size * size / 256;
        let prefix = dit.prefix(&e, &hidden, rows, &|| false).unwrap();
        assert_eq!(prefix.storage_bytes(), rows * 4096 * 2 * 2 * 32);
        let sc = dit::Scratch::new(&e.device, n, prefix.len).unwrap();
        let x = e
            .to_device(
                &(0..n * 64)
                    .map(|i| (i as f32 * 0.017).sin())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let out = e.device.alloc(n * 64 * 4).unwrap();
        for iteration in 0..iterations {
            let now = std::time::Instant::now();
            dit.step(
                &e,
                &prefix,
                &x,
                &sc,
                0.75,
                size / 16,
                size / 16,
                &out,
                &|| false,
            )
            .unwrap();
            eprintln!(
                "DiT size={size} iteration={iteration} seconds={:.6}",
                now.elapsed().as_secs_f64()
            );
        }
    }
}

#[test]
#[ignore = "requires pinned Qwen-Image MLX checkpoint and Metal"]
fn mlx_image_grouped_qkv_is_bit_identical() {
    let root = std::env::var("PADDOCK_QI_MLX").unwrap();
    let source = mlx::Source::open(&Path::new(&root).join("transformer")).unwrap();
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let weights: Vec<_> = ["q", "k", "v"]
        .iter()
        .map(|n| {
            source
                .weight(
                    &e.device,
                    &format!("transformer_blocks.0.attn.to_{n}.weight"),
                    &[4096, 4096],
                )
                .unwrap()
        })
        .collect();
    for rows in [17, 65, 512, 4096] {
        let x = e
            .to_device(
                &(0..rows * 4096)
                    .map(|i| half::bf16::from_f32((i as f32 * 0.037).sin()).to_f32())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let expected: Vec<_> = (0..3)
            .map(|_| e.device.alloc(rows * 4096 * 4).unwrap())
            .collect();
        let actual: Vec<_> = (0..3)
            .map(|_| e.device.alloc(rows * 4096 * 4).unwrap())
            .collect();
        let scratch = e.device.alloc(mlx::workspace(rows)).unwrap();
        for iteration in 0..3 {
            for grouped in [false, true] {
                let now = std::time::Instant::now();
                let cmd = e.device.begin().unwrap();
                if grouped {
                    let planes: Vec<_> = weights.iter().zip(&actual).collect();
                    mlx::project(&cmd, &planes, &x, rows, &scratch);
                } else {
                    // The previous path also shared the input conversion;
                    // only projection dispatch grouping differs here.
                    let planes: Vec<_> = weights.iter().zip(&expected).collect();
                    mlx::project_separate(&cmd, &planes, &x, rows, &scratch);
                }
                cmd.finish().unwrap();
                eprintln!(
                    "QKV M={rows} grouped={grouped} iteration={iteration} ms={:.3}",
                    now.elapsed().as_secs_f64() * 1000.
                );
            }
        }
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(unsafe { a.read_f32(0, rows * 4096) }, unsafe {
                b.read_f32(0, rows * 4096)
            });
        }
    }
}

#[test]
#[ignore = "requires pinned Qwen-Image MLX checkpoint and Metal"]
fn mlx_image_projection_store_matches_baseline() {
    let root = std::env::var("PADDOCK_QI_MLX").unwrap();
    let source = mlx::Source::open(&Path::new(&root).join("transformer")).unwrap();
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    for (name, k, n) in [
        ("attn.to_q", 4096, 4096),
        ("img_mlp.gate_layer", 4096, 12288),
        ("img_mlp.out", 12288, 4096),
    ] {
        let w = source
            .weight(
                &e.device,
                &format!("transformer_blocks.0.{name}.weight"),
                &[k, n],
            )
            .unwrap();
        for rows in [512, 1024, 4096] {
            let x = e
                .to_device(
                    &(0..rows * k)
                        .map(|i| half::bf16::from_f32((i as f32 * 0.037).sin()).to_f32())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let baseline = e.device.alloc(rows * n * 4).unwrap();
            let output = e.device.alloc(rows * n * 4).unwrap();
            let scratch = e.device.alloc(mlx::workspace(rows)).unwrap();
            for iteration in 0..4 {
                for fast in [false, true] {
                    let now = std::time::Instant::now();
                    let cmd = e.device.begin().unwrap();
                    if fast {
                        mlx::project(&cmd, &[(&w, &output)], &x, rows, &scratch);
                    } else {
                        crate::affine::project(&cmd, &[(&w, &baseline)], &x, rows, &scratch);
                    }
                    cmd.finish().unwrap();
                    eprintln!(
                        "image projection {name} M={rows} fast={fast} iteration={iteration} ms={:.3}",
                        now.elapsed().as_secs_f64() * 1000.
                    );
                }
            }
            let a = unsafe { baseline.read_f32(0, rows * n) };
            let b = unsafe { output.read_f32(0, rows * n) };
            assert!(
                a.iter().zip(&b).all(|(a, b)| a == b),
                "{name} M={rows} differs"
            );
            for prefix_rows in [1, 7, 17, 31, 65] {
                let cmd = e.device.begin().unwrap();
                mlx::project(&cmd, &[(&w, &output)], &x, prefix_rows, &scratch);
                cmd.finish().unwrap();
                let prefix = unsafe { output.read_f32(0, prefix_rows * n) };
                assert_eq!(
                    &b[..prefix_rows * n],
                    prefix,
                    "cached prefix {name}, rows={prefix_rows} vs joint={rows}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires Metal device"]
fn contiguous_attention_matches_scalar_ragged_contract() {
    use half::f16;
    let exec = Ops {
        device: MetalDevice::new(Some(1 << 30)).unwrap(),
    };
    let (rows, keys, heads) = (67, 259, 2);
    let make = |n, phase: f32| {
        (0..n)
            .map(|i| f16::from_f32((i as f32 * 0.071 + phase).sin() * 0.4))
            .collect::<Vec<_>>()
    };
    let q = make(rows * heads * 128, 0.1);
    let k = make(keys * heads * 128, 0.5);
    let v = make(keys * heads * 128, 0.9);
    let qb = exec.f16_to_device(&q).unwrap();
    let kb = exec.f16_to_device(&k).unwrap();
    let vb = exec.f16_to_device(&v).unwrap();
    let output = exec.alloc::<f32>(rows * heads * 128).unwrap();
    for (kernel, tile) in [("qi_attention", 32), ("qi_attention_deep", 16)] {
        let t = words(
            &exec.device,
            &(0..rows)
                .step_by(32)
                .flat_map(|i| [i as u32, (rows - i).min(32) as u32])
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for mode in 0..3 {
            let ends: Vec<_> = (0..rows)
                .map(|i| match mode {
                    0 => keys - 1,
                    1 => i,
                    _ => {
                        if (7..43).contains(&i) {
                            42
                        } else if (47..63).contains(&i) {
                            62
                        } else {
                            i
                        }
                    }
                })
                .collect();
            let meta = words(
                &exec.device,
                &ends
                    .iter()
                    .flat_map(|&end| [0, end as u32])
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            exec.run(
                kernel,
                &[&qb, &kb, &vb, &meta, &meta, &output, &t],
                &[
                    heads as u32,
                    heads as u32,
                    0,
                    (1f32 / 128f32.sqrt()).to_bits(),
                ],
                [heads, rows.div_ceil(tile), 1],
                128,
            )
            .unwrap();
            let got = unsafe { output.read_f32(0, rows * heads * 128) };
            for row in 0..rows {
                for h in 0..heads {
                    let scores: Vec<_> = (0..=ends[row])
                        .map(|j| {
                            (0..128)
                                .map(|d| {
                                    q[(row * heads + h) * 128 + d].to_f32()
                                        * k[(j * heads + h) * 128 + d].to_f32()
                                })
                                .sum::<f32>()
                                / 128f32.sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let p: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum = p.iter().sum::<f32>();
                    for d in 0..128 {
                        let expected = p
                            .iter()
                            .enumerate()
                            .map(|(j, p)| p * v[(j * heads + h) * 128 + d].to_f32())
                            .sum::<f32>()
                            / sum;
                        assert!(
                            (got[(row * heads + h) * 128 + d] - expected).abs() < 2e-4,
                            "row {row}, head {h}, col {d}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Metal GPU; isolated long-attention benchmark"]
fn image_attention_deep_benchmark() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let (rows, keys, heads) = (4096usize, 4129usize, 32usize);
    let input = e
        .f16_to_device(
            &(0..keys * heads * 128)
                .map(|i| half::f16::from_f32((i as f32 * 0.03).sin()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = e.device.alloc(rows * heads * 128 * 4).unwrap();
    let meta = words(
        &e.device,
        &(0..rows)
            .flat_map(|_| [0, (keys - 1) as u32])
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (kernel, tile) in [("qi_attention", 32), ("qi_attention_deep", 16)] {
        let t = words(
            &e.device,
            &(0..rows)
                .step_by(32)
                .flat_map(|i| [i as u32, 32])
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for iteration in 0..6 {
            let start = std::time::Instant::now();
            e.run(
                kernel,
                &[&input, &input, &input, &meta, &meta, &out, &t],
                &[32, 32, 0, (1f32 / 128f32.sqrt()).to_bits()],
                [heads, rows / tile, 1],
                128,
            )
            .unwrap();
            eprintln!(
                "{kernel} iteration={iteration} ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.
            );
        }
    }
}

#[test]
#[ignore = "requires Metal device"]
fn normalization_and_guidance_match_scalar_contracts() {
    let exec = Ops {
        device: MetalDevice::new(Some(1 << 30)).unwrap(),
    };
    let n = 512;
    let x = (0..2 * n)
        .map(|i| (i as f32 * 0.3).sin() + 0.25)
        .collect::<Vec<_>>();
    let scale = (0..n).map(|i| (i % 7) as f32 * 0.05).collect::<Vec<_>>();
    let input = exec.to_device(&x).unwrap();
    let weight = exec.to_device(&scale).unwrap();
    let output = exec.alloc::<f32>(2 * n).unwrap();
    for mode in [0, 1, 2, 3] {
        exec.run(
            "qi_norm",
            &[&input, &weight, &output],
            &[n as u32, 0, 1e-6f32.to_bits(), mode, 0],
            [2, 1, 1],
            256,
        )
        .unwrap();
        let got = unsafe { output.read_f32(0, 2 * n) };
        for row in 0..2 {
            let a = &x[row * n..(row + 1) * n];
            let mean = if mode & 1 == 1 {
                a.iter().sum::<f32>() / n as f32
            } else {
                0.
            };
            let inv = (a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32 + 1e-6)
                .sqrt()
                .recip();
            for d in 0..n {
                let mut normalized = (a[d] - mean) * inv;
                if mode == 3 {
                    normalized = half::bf16::from_f32(normalized).to_f32();
                }
                let scale = if mode == 3 {
                    half::bf16::from_f32(1. + scale[d]).to_f32()
                } else {
                    1. + scale[d]
                };
                let mut expected = normalized * scale;
                if mode & 2 == 2 {
                    expected = half::bf16::from_f32(expected).to_f32();
                }
                assert!((got[row * n + d] - expected).abs() < 2e-6);
            }
        }
    }
    let unconditional = exec.to_device(&[1., 2., 3., 4.]).unwrap();
    let conditional = exec.to_device(&[2., 4., 6., 8.]).unwrap();
    exec.run(
        "qi_guidance",
        &[&unconditional, &conditional],
        &[4, 3f32.to_bits()],
        [1, 1, 1],
        256,
    )
    .unwrap();
    assert_eq!(unsafe { unconditional.read_f32(0, 4) }, [4., 8., 12., 16.]);
}

#[test]
#[ignore = "requires Qwen-Image VAE and Metal device"]
fn vae_bands_match_whole_plane() {
    let root = std::env::var("PADDOCK_QI_MODELS").expect("model root");
    let exec = Rc::new(Ops {
        device: MetalDevice::new(None).unwrap(),
    });
    let vae = vae::VaeDecoder::load(
        exec.clone(),
        &Path::new(&root).join("Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors"),
    )
    .unwrap();
    let latents = exec
        .to_device(
            &(0..8 * 12 * 64)
                .map(|i| ((i as f32 * 0.37).sin()) * 0.4)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let whole = vae.decode_with(&latents, 12, 8, None).unwrap();
    let banded = vae.decode_with(&latents, 12, 8, Some(8)).unwrap();
    let max_error = whole
        .iter()
        .zip(&banded)
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap();
    let different = whole.iter().zip(&banded).filter(|(a, b)| a != b).count();
    eprintln!(
        "VAE bands: {different}/{} channels differ, max byte difference {max_error}",
        whole.len()
    );
    assert_eq!(whole, banded);
}

#[test]
#[ignore = "requires Metal device"]
fn im2row_matches_cpu_for_rectangles_and_upsampling() {
    use half::f16;
    use objc2_metal::MTLBuffer;
    let exec = Ops {
        device: MetalDevice::new(Some(1 << 30)).unwrap(),
    };
    let (h, w, c) = (5, 7, 3);
    let host = (0..h * w * c)
        .map(|n| f16::from_f32(n as f32))
        .collect::<Vec<_>>();
    let x = exec.f16_to_device(&host).unwrap();
    for up in [false, true] {
        let scale = if up { 2 } else { 1 };
        let (wo, ho) = (w * scale, h * scale);
        for y0 in [0, ho - 2] {
            let mut result = exec.alloc_f16(2 * wo * 9 * c).unwrap();
            exec.vae_im2row3(&x, &mut result, h, w, c, y0, 2, up)
                .unwrap();
            let got = unsafe {
                std::slice::from_raw_parts(
                    result.raw.contents().as_ptr().cast::<f16>(),
                    2 * wo * 9 * c,
                )
            }
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>();
            let mut expected = Vec::new();
            for y in y0..y0 + 2 {
                for xx in 0..wo {
                    for ky in -1..=1 {
                        for kx in -1..=1 {
                            let yy = y as isize + ky;
                            let xxx = xx as isize + kx;
                            for cc in 0..c {
                                expected.push(
                                    if yy < 0 || xxx < 0 || yy >= ho as isize || xxx >= wo as isize
                                    {
                                        0.
                                    } else {
                                        host[((yy as usize / scale) * w + xxx as usize / scale) * c
                                            + cc]
                                            .to_f32()
                                    },
                                );
                            }
                        }
                    }
                }
            }
            assert_eq!(got, expected);
        }
    }
}

#[test]
#[ignore = "requires Metal device"]
fn shader_primitives_compile_and_noise_is_repeatable() {
    let e = Ops {
        device: MetalDevice::new(Some(1 << 30)).unwrap(),
    };
    let x = e.alloc::<f32>(64 * 5).unwrap();
    let y = e.alloc::<f32>(64 * 5).unwrap();
    for out in [&x, &y] {
        e.run("qi_noise", &[out], &[5, 42, 0, 0], [2, 1, 1], 256)
            .unwrap();
    }
    let x = unsafe { x.read_f32(0, 320) };
    let y = unsafe { y.read_f32(0, 320) };
    assert_eq!(x, y);
    assert!(x.iter().all(|v| v.is_finite()));
    assert!(x.iter().any(|v| v.abs() > 1.));
}

#[test]
#[ignore = "requires Qwen-Image Q4 and companion model files"]
fn full_model_smoke() {
    let root = std::env::var("PADDOCK_QI_MODELS").expect("model root");
    let root = Path::new(&root);
    let te = root.join("Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf");
    let map = paddock_models::mapped::MappedGguf::open(&te).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let sys = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    let ids=tokenizer.encode(&format!("{sys}<|im_start|>user\nA red apple on a white table.<|im_end|>\n<|im_start|>assistant\n")).unwrap();
    let drop = tokenizer.encode(sys).unwrap().len();
    std::mem::drop(map);
    let mut model = QwenImage::load(
        &root.join("Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf"),
        &te,
        &root.join("Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors"),
        8192,
        None,
    )
    .unwrap();
    let size = std::env::var("PADDOCK_QI_SIZE")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(256);
    let steps = std::env::var("PADDOCK_QI_STEPS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(2);
    let req = GenerateRequest {
        prompt_ids: &ids,
        drop,
        negative: None,
        width: size,
        height: size,
        steps,
        seed: 42,
        noise_offset: 0,
        guidance: 1.,
        references: &[],
        image_pad_id: 0,
    };
    let started = std::time::Instant::now();
    assert!(model.render(&req, 0, &mut |_, _| Ok(()), &|| true).is_err());
    let image = model
        .render(&req, 0, &mut |_, _| Ok(()), &|| false)
        .unwrap();
    eprintln!(
        "Metal Qwen-Image: {size}x{size}, {steps} steps in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(image.pixels.len(), size * size * 4);
    assert!(image.pixels.chunks_exact(4).any(|v| v[0] != v[1]));
    if let Some(path) = std::env::var_os("PADDOCK_QI_RGBA") {
        std::fs::write(path, image.pixels).unwrap();
    }
    if steps <= 4 && size <= 256 {
        let mut previews = 0;
        let rendered = model.render(
            &req,
            1,
            &mut |_, _| {
                previews += 1;
                Err("preview receiver closed".into())
            },
            &|| false,
        );
        assert!(
            rendered
                .err()
                .expect("render must stop")
                .to_string()
                .contains("preview receiver closed")
        );
        assert_eq!(previews, 1);
    }
}

#[test]
#[ignore = "requires pinned Qwen-Image MLX checkpoint and Metal"]
fn mlx_model_smoke_and_cancel() {
    let root = std::env::var("PADDOCK_QI_MLX").expect("MLX checkpoint");
    let root = Path::new(&root);
    let tok = paddock_tokenizer::GgufTokenizer::from_hf_dir(&root.join("processor")).unwrap();
    let sys = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    let user = std::env::var("PADDOCK_QI_PROMPT")
        .unwrap_or_else(|_| "A red apple on a white table.".into());
    let seed = std::env::var("PADDOCK_QI_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(42);
    let prompt = format!("{sys}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
    let ids = tok.encode(&prompt).unwrap();
    let drop = tok.encode(sys).unwrap().len();
    let mut model = QwenImage::load_mlx(root, 8192, None).unwrap();
    if let Some(directory) = std::env::var_os("PADDOCK_QI_FIXTURE") {
        let dir = Path::new(&directory);
        let save = |name: &str, buffer: &Buffer, count: usize| {
            let data = unsafe { buffer.read_f32(0, count) };
            assert!(
                data.iter().all(|v| v.is_finite()),
                "nonfinite fixture: {name}"
            );
            std::fs::write(
                dir.join(name),
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        };
        std::fs::write(dir.join("ids.json"), serde_json::to_vec(&ids).unwrap()).unwrap();
        let hidden = model
            .text
            .encode(&model.exec, &ids, drop, &|| false)
            .unwrap();
        save("hidden.f32", &hidden, (ids.len() - drop) * 4096);
        let (lw, lh) = (8, 12);
        let latent = model
            .exec
            .to_device(
                &(0..lw * lh * 64)
                    .map(|i| (i as f32 * 0.037).sin())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let prefix = model
            .dit
            .prefix(&model.exec, &hidden, ids.len() - drop, &|| false)
            .unwrap();
        let sc = dit::Scratch::new(&model.exec.device, lw * lh, prefix.len).unwrap();
        let velocity = model.exec.alloc::<f32>(lw * lh * 64).unwrap();
        model
            .dit
            .step(
                &model.exec,
                &prefix,
                &latent,
                &sc,
                0.75,
                lw,
                lh,
                &velocity,
                &|| false,
            )
            .unwrap();
        save("latent.f32", &latent, lw * lh * 64);
        save("velocity.f32", &velocity, lw * lh * 64);
        let pixels = model.vae.decode(&latent, lw, lh).unwrap();
        std::fs::write(dir.join("vae.rgba"), pixels).unwrap();
        let rgba = model
            .exec
            .to_device(
                &(0..128 * 192 * 4)
                    .map(|i| {
                        if i % 4 == 3 {
                            1.
                        } else {
                            (i as f32 * 0.003).sin()
                        }
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let encoder =
            vae_encode::Encoder::load(model.exec.clone(), &root.join("vae/model.safetensors"))
                .unwrap();
        let encoded = encoder.encode(&rgba, 128, 192, &|| false).unwrap();
        save("reference-image.f32", &rgba, 128 * 192 * 4);
        save("encoded.f32", &encoded, 8 * 12 * 64);
        assert!(encoder.encode(&rgba, 128, 192, &|| true).is_err());
        std::mem::drop(encoder);
        // Independent full-trajectory oracle: export the exact initial noise and
        // schedule, not merely a seed (MLX and CUDA use different PRNGs).
        let size = std::env::var("PADDOCK_QI_FIXTURE_SIZE")
            .map(|v| v.parse::<usize>().expect("fixture size must be an integer"))
            .unwrap_or(256);
        assert!((32..=2752).contains(&size) && size.is_multiple_of(32));
        std::fs::write(
            dir.join("generation-size.json"),
            serde_json::to_vec(&size).unwrap(),
        )
        .unwrap();
        let (lw, lh, steps) = (size / 16, size / 16, 40);
        let e = &model.exec;
        let mut x = e.alloc::<f32>(lw * lh * 64).unwrap();
        e.run(
            "qi_noise",
            &[&x],
            &[(lw * lh) as u32, seed as u32, (seed >> 32) as u32, 0],
            [(lw * lh * 64).div_ceil(256), 1, 1],
            256,
        )
        .unwrap();
        save("initial.f32", &x, lw * lh * 64);
        let schedule = Schedule::new(steps, mu_for_tokens(lw * lh));
        std::fs::write(
            dir.join("sigmas.json"),
            serde_json::to_vec(&schedule.sigmas).unwrap(),
        )
        .unwrap();
        let prefix = model
            .dit
            .prefix(e, &hidden, ids.len() - drop, &|| false)
            .unwrap();
        let sc = dit::Scratch::new(&e.device, lw * lh, prefix.len).unwrap();
        let velocity = e.alloc::<f32>(lw * lh * 64).unwrap();
        for i in 0..steps {
            model
                .dit
                .step(
                    e,
                    &prefix,
                    &x,
                    &sc,
                    schedule.sigmas[i],
                    lw,
                    lh,
                    &velocity,
                    &|| false,
                )
                .unwrap();
            e.scale_add(
                &mut x,
                &velocity,
                schedule.sigmas[i + 1] - schedule.sigmas[i],
                lw * lh * 64,
            )
            .unwrap();
            save(&format!("step-{i:02}.f32"), &x, lw * lh * 64);
        }
        std::fs::write(
            dir.join("generation.rgba"),
            model.vae.decode(&x, lw, lh).unwrap(),
        )
        .unwrap();
    }
    let size = std::env::var("PADDOCK_QI_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let steps = std::env::var("PADDOCK_QI_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let request = GenerateRequest {
        prompt_ids: &ids,
        drop,
        negative: None,
        width: size,
        height: size,
        steps,
        seed,
        noise_offset: 0,
        guidance: 1.,
        references: &[],
        image_pad_id: 0,
    };
    let now = std::time::Instant::now();
    let image = model
        .render(&request, 0, &mut |_, _| Ok(()), &|| false)
        .unwrap();
    eprintln!(
        "MLX {size}x{size} {steps} steps: {:.3}s; weights {}",
        now.elapsed().as_secs_f64(),
        model.weights
    );
    assert_eq!(image.pixels.len(), size * size * 4);
    // The canonical apple smoke has dark and light pixels. Arbitrary fixture
    // prompts (e.g. a pale glass sculpture after only two steps) do not carry
    // that colour-range contract; their quality is checked by the oracle.
    if std::env::var_os("PADDOCK_QI_PROMPT").is_none() {
        assert!(image.pixels.iter().any(|&x| x > 128) && image.pixels.iter().any(|&x| x < 64));
    }
    if let Some(path) = std::env::var_os("PADDOCK_QI_RGBA") {
        std::fs::write(path, &image.pixels).unwrap();
    }
    if size <= 512 && steps <= 4 {
        // Conditional/unconditional calls reuse target K/V, not their cached
        // prefix. Equal conditions must remain identical when CFG runs twice.
        let guided = GenerateRequest {
            guidance: 2.,
            negative: Some((&ids, drop)),
            ..request
        };
        let result = model
            .render(&guided, 0, &mut |_, _| Ok(()), &|| false)
            .unwrap();
        assert_eq!(image.pixels, result.pixels);
    }
    assert!(
        model
            .render(&request, 0, &mut |_, _| Ok(()), &|| true)
            .err()
            .unwrap()
            .to_string()
            .contains("client went away")
    );
}
