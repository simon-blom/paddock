use super::*;

#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; GPU prefill tile election"]
fn prefill_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for names in [
        vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ],
        vec!["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
        vec!["blk.0.ffn_down.weight"],
    ] {
        let weights: Vec<_> = names
            .iter()
            .map(|name| {
                let shape = &map.tensor_info(name).unwrap().dims;
                Weight::load(&d, &map, name, &[shape[0] as usize, shape[1] as usize]).unwrap()
            })
            .collect();
        if !weights.iter().all(|w| matches!(w.ty, 12 | 13 | 14 | 23)) {
            continue;
        }
        let k = weights[0].k;
        let workspace = d.alloc(384 * k * 2).unwrap();
        for m in [144usize, 192, 257, 279, 288] {
            let x = d
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let b: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let base: Vec<_> = weights.iter().zip(&a).collect();
            let narrow: Vec<_> = weights.iter().zip(&b).collect();
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for offset in 0..2 {
                    let at = if round % 2 == 0 { offset } else { 1 - offset };
                    let cmd = d.begin().unwrap();
                    for _ in 0..16 {
                        if at == 1 {
                            forward::prefill96_projection(&cmd, &narrow, &x, m, &workspace);
                        } else if weights.len() == 1 {
                            weights[0].linear(&cmd, &x, &a[0], m, 1., &workspace);
                        } else {
                            projections(&cmd, &base, &x, m, &workspace);
                        }
                    }
                    let ms = cmd.finish().unwrap() * 1000. / 16.;
                    if round > 1 {
                        times[at].push(ms);
                    }
                }
            }
            for (i, w) in weights.iter().enumerate() {
                assert_eq!(
                    unsafe { a[i].read_f32(0, m * w.n) },
                    unsafe { b[i].read_f32(0, m * w.n) },
                    "prefill m={m} {}",
                    names[i]
                );
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "prefill {:?} m={m}: old {:.4} ms tile96 {:.4} ms",
                names, times[0][3], times[1][3]
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; GPU fused projection election"]
fn grouped_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for names in [
        vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ],
        vec!["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
    ] {
        let weights: Vec<_> = names
            .iter()
            .map(|name| {
                let shape = &map.tensor_info(name).unwrap().dims;
                Weight::load(&d, &map, name, &[shape[0] as usize, shape[1] as usize]).unwrap()
            })
            .collect();
        println!(
            "fused {:?} types {:?}",
            names,
            weights.iter().map(|w| w.ty).collect::<Vec<_>>()
        );
        if !weights.iter().all(|w| matches!(w.ty, 12..=14)) {
            continue;
        }
        let k = weights[0].k;
        let workspace = d.alloc(128 * k * 2).unwrap();
        for m in [3usize, 4] {
            let x = d
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let b: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let base: Vec<_> = weights.iter().zip(&a).collect();
            let pair: Vec<_> = weights.iter().zip(&b).collect();
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for offset in 0..2 {
                    let at = if round % 2 == 0 { offset } else { 1 - offset };
                    let cmd = d.begin().unwrap();
                    for _ in 0..16 {
                        if at == 0 {
                            projections(&cmd, &base, &x, m, &workspace);
                        } else {
                            forward::pair_projection(&cmd, &pair, &x, m);
                        }
                    }
                    let ms = cmd.finish().unwrap() * 1000. / 16.;
                    if round > 1 {
                        times[at].push(ms);
                    }
                }
            }
            for (i, w) in weights.iter().enumerate() {
                assert_eq!(
                    unsafe { a[i].read_f32(0, m * w.n) },
                    unsafe { b[i].read_f32(0, m * w.n) },
                    "fused m={m} {}",
                    names[i]
                );
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "fused {:?} m={m}: old {:.4} ms pair {:.4} ms",
                names, times[0][3], times[1][3]
            );
        }
    }
}

#[test]
#[ignore = "GPU merge election, not serving throughput"]
fn merge_election() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    for hd in [256usize, 512] {
        for rows in [1usize, 4, 32] {
            let selected = d
                .upload(
                    &(0..rows as u32)
                        .rev()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = d.alloc(rows * HEADS * hd * 4).unwrap();
            let b = d.alloc(a.len()).unwrap();
            for splits in [4usize, 16, 32] {
                let parts = d
                    .upload(
                        &(0..rows * HEADS * splits * (hd + 2))
                            .flat_map(|i| {
                                let v = if i % (hd + 2) == hd + 1 {
                                    1. + (i % 7) as f32
                                } else {
                                    ((i * 13 + i / (hd + 2)) % 31) as f32 / 17. - 1.
                                };
                                v.to_le_bytes()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..9 {
                    for offset in 0..2 {
                        let at = if round % 2 == 0 { offset } else { 1 - offset };
                        let cmd = d.begin().unwrap();
                        // Amortize command submission over 60 dependent dispatches,
                        // just as a target decode pays one merge per layer.
                        for _ in 0..60 {
                            cmd.dispatch(
                                if at == 0 {
                                    "gemma_merge"
                                } else {
                                    "gemma_merge_shared"
                                },
                                &[&parts, if at == 0 { &a } else { &b }, &selected],
                                &[HEADS as u32, splits as u32, hd as u32],
                                [HEADS * rows, 1, 1],
                                32,
                            );
                        }
                        let ms = cmd.finish().unwrap() * 1000. / 60.;
                        if round > 1 {
                            times[at].push(ms);
                        }
                    }
                }
                let reference = unsafe { a.read_f32(0, rows * HEADS * hd) };
                let actual = unsafe { b.read_f32(0, reference.len()) };
                assert!(actual.iter().all(|x| x.is_finite()));
                assert_eq!(reference, actual, "hd={hd} rows={rows} splits={splits}");
                for t in &mut times {
                    t.sort_by(f64::total_cmp);
                }
                println!(
                    "merge hd={hd} rows={rows} splits={splits}: old {:.5} ms shared {:.5} ms",
                    times[0][3], times[1][3]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; GPU verification projection election"]
fn verification_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for (name, k, n) in [
        ("blk.0.attn_q.weight", 5376usize, 8192usize),
        ("blk.0.ffn_down.weight", 21504, 5376),
        ("token_embd.weight", 5376, 262144),
    ] {
        let w = Weight::load(&d, &map, name, &[k, n]).unwrap();
        for m in [2usize, 3, 4, 5, 6, 7, 8, 16, 32] {
            let x = d
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = d.alloc(m * n * 4).unwrap();
            let b = d.alloc(a.len()).unwrap();
            let p = [k as u32, n as u32, m as u32, w.ty, 1f32.to_bits()];
            let base = match m {
                1 => ("linear_kquant1", 4, 1),
                2 => ("linear_kquant2", 16, 2),
                3 => ("linear_kquant3", 16, 3),
                4 => ("linear_kquant4", 16, 4),
                5..=8 if w.ty != 14 => ("gemma_verify8", 16, 8),
                _ => ("gemma_f32_32", 16, 32),
            };
            let mut candidates = vec![base];
            if m <= 4 && matches!(w.ty, 12..=14) {
                candidates.push((
                    match m {
                        2 => "gemma_pair2",
                        3 => "gemma_pair3",
                        _ => "gemma_pair4",
                    },
                    32,
                    m,
                ));
            }
            if (5..=8).contains(&m) {
                candidates.push((
                    match m {
                        5 => "gemma_full5",
                        6 => "gemma_full6",
                        7 => "gemma_full7",
                        _ => "gemma_full8",
                    },
                    16,
                    m,
                ));
            }
            if m >= 4 {
                candidates.push(("gemma_staged_f32_32", 16, 32));
            }
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                base.0,
                &[&w.buffer, &x, &a],
                &p,
                [n.div_ceil(base.1), m.div_ceil(base.2), 1],
                128,
            );
            cmd.finish().unwrap();
            let reference = unsafe { a.read_f32(0, m * n) };
            let magnitude = reference.iter().map(|v| v.abs()).fold(1f32, f32::max);
            let mut times = vec![Vec::new(); candidates.len()];
            let mut errors = vec![0f32; candidates.len()];
            // A few microsecond dispatches are not enough to reach sustained
            // GPU clocks. Warm a real work train before alternating candidates.
            let cmd = d.begin().unwrap();
            for _ in 0..32 {
                cmd.dispatch(
                    base.0,
                    &[&w.buffer, &x, &a],
                    &p,
                    [n.div_ceil(base.1), m.div_ceil(base.2), 1],
                    128,
                );
            }
            cmd.finish().unwrap();
            for round in 0..9 {
                // Alternate order to avoid granting one candidate all the cold clocks.
                for offset in 0..candidates.len() {
                    let at = if round % 2 == 0 {
                        offset
                    } else {
                        candidates.len() - 1 - offset
                    };
                    let (kernel, cols, rows) = candidates[at];
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, &x, &b],
                        &p,
                        [n.div_ceil(cols), m.div_ceil(rows), 1],
                        128,
                    );
                    let ms = cmd.finish().unwrap() * 1000.;
                    if round > 1 {
                        times[at].push(ms);
                    }
                    if round == 0 {
                        let actual = unsafe { b.read_f32(0, m * n) };
                        assert!(actual.iter().all(|v| v.is_finite()));
                        errors[at] = reference
                            .iter()
                            .zip(&actual)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        assert!(
                            errors[at] / magnitude < 0.00001,
                            "{name} {kernel} m={m}: error={}",
                            errors[at]
                        );
                    }
                }
            }
            for (i, (kernel, _, _)) in candidates.iter().enumerate() {
                times[i].sort_by(f64::total_cmp);
                println!(
                    "{name} m={m} {kernel}: {:.4} ms error={}",
                    times[i][3], errors[i]
                );
            }
        }
    }
}

#[test]
#[ignore = "GPU attention cost diagnostic, not serving throughput"]
fn verification_attention_election() {
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let (hd, kh, ring) = (256usize, 16usize, 1536usize);
    for c in [1usize, 4] {
        for count in [4usize, 8] {
            let rows = c * count;
            let q = d
                .upload(
                    &(0..rows * HEADS * hd)
                        .flat_map(|i| {
                            (((i * 7 + i / hd) % 17) as f32 / 257. - 8. / 257.).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let halfs = |factor: usize| {
                (0..c * ring * kh * hd)
                    .flat_map(|i| {
                        half::f16::from_f32(((i * factor + i / hd) % 19) as f32 / 256. - 9. / 256.)
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>()
            };
            let k = d.upload(&halfs(3)).unwrap();
            let v = d.upload(&halfs(11)).unwrap();
            let selected = upload(&(0..rows as u32).collect::<Vec<_>>());
            let pages = upload(&[0]);
            let tiles = upload(
                &(0..c)
                    .flat_map(|i| [(i * count) as u32, count as u32])
                    .collect::<Vec<_>>(),
            );
            let parts = d.alloc(rows * HEADS * SPLITS * (hd + 2) * 4).unwrap();
            let a = d.alloc(rows * HEADS * hd * 4).unwrap();
            let b = d.alloc(a.len()).unwrap();
            for prefix in [128usize, 1024] {
                let meta = upload(
                    &(0..c)
                        .flat_map(|s| (0..count).flat_map(move |i| [s as u32, (prefix + i) as u32]))
                        .collect::<Vec<_>>(),
                );
                let p = [HEADS as u32, kh as u32, 1, 1024, ring as u32];
                let splits = (prefix + count)
                    .min(1024)
                    .div_ceil(128)
                    .max(16usize.div_ceil(rows))
                    .clamp(1, SPLITS);
                let mut params = p.to_vec();
                params.push(splits as u32);
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..9 {
                    for (mode, time) in times.iter_mut().enumerate() {
                        let cmd = d.begin().unwrap();
                        if mode == 0 {
                            cmd.dispatch(
                                "gemma_decode256",
                                &[&q, &k, &v, &meta, &pages, &selected, &parts],
                                &params,
                                [kh, rows, splits],
                                128,
                            );
                            cmd.dispatch(
                                "gemma_merge",
                                &[&parts, &a, &selected],
                                &[HEADS as u32, splits as u32, hd as u32],
                                [HEADS * rows, 1, 1],
                                32,
                            );
                        } else {
                            cmd.dispatch(
                                "gemma_verify_attn256",
                                &[&q, &k, &v, &meta, &pages, &b, &tiles],
                                &p,
                                [HEADS, c, 1],
                                128,
                            );
                        }
                        let ms = cmd.finish().unwrap() * 1000.;
                        if round > 1 {
                            time.push(ms);
                        }
                    }
                }
                let expected = unsafe { a.read_f32(0, rows * HEADS * hd) };
                let actual = unsafe { b.read_f32(0, expected.len()) };
                assert!(actual.iter().all(|x| x.is_finite()));
                let error = expected
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(error < 0.00001, "attention error {error}");
                for time in &mut times {
                    time.sort_by(f64::total_cmp);
                }
                println!(
                    "SWA c={c} block={count} ctx={prefix}: per-query {:.4} ms, F32 matrix {:.4} ms",
                    times[0][3], times[1][3]
                );
            }
        }
    }
}

// A GPU-to-GPU projection probe using the unchanged real checkpoint, never
// a host matrix reference. Timings are diagnostics, not a serving score.
#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; projection election diagnostic"]
fn f32_matrix_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let device = MetalDevice::new(Some(4 << 30)).unwrap();
    for (name, k, n) in [
        ("blk.0.attn_q.weight", 5376, 8192),
        ("blk.0.ffn_down.weight", 21504, 5376),
        ("token_embd.weight", 5376, 262144),
    ] {
        let w = Weight::load(&device, &map, name, &[k, n]).unwrap();
        for rows in [4usize, 8, 16, 32] {
            let x = device
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = device.alloc(rows * n * 4).unwrap();
            let b = device.alloc(rows * n * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, w.ty, 1f32.to_bits()];
            let baseline = if rows <= 4 {
                "linear_kquant4"
            } else if rows <= 8 {
                "gemma_verify8"
            } else {
                "gemma_verify16"
            };
            let tile = rows.min(16);
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                baseline,
                &[&w.buffer, &x, &a],
                &p,
                [n.div_ceil(16), rows.div_ceil(tile), 1],
                128,
            );
            cmd.finish().unwrap();
            let reference = unsafe { a.read_f32(0, rows * n) };
            for (kernel, bm) in [
                (baseline, tile),
                ("gemma_f32_8", 8),
                ("gemma_f32_16", 16),
                ("gemma_f32_32", 32),
            ] {
                let mut times = Vec::new();
                for _ in 0..5 {
                    let cmd = device.begin().unwrap();
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, &x, &b],
                        &p,
                        [n.div_ceil(16), rows.div_ceil(bm), 1],
                        128,
                    );
                    times.push(cmd.finish().unwrap() * 1000.);
                }
                let actual = unsafe { b.read_f32(0, rows * n) };
                let error = reference
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(actual.iter().all(|x| x.is_finite()));
                assert!(error < 0.05, "{name} {kernel} rows={rows} error={error}");
                times.sort_by(f64::total_cmp);
                println!(
                    "{name} rows={rows} {kernel}: {:.3} ms, max error {error}",
                    times[2]
                );
            }
        }
    }
}
