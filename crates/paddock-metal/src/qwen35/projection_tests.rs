use super::*;

// Rejected tile-64 candidate retained only in the GPU election harness.
fn prefill64(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    m: usize,
    workspace: &Buffer,
) {
    let k = planes[0].0.k;
    cmd.dispatch(
        "linear_input_padded",
        &[input, workspace],
        &[k as u32, 0, m as u32],
        [
            (k.div_ceil(128) * 128 * m.div_ceil(128) * 128).div_ceil(256),
            1,
            1,
        ],
        256,
    );
    prefill64_prepared(cmd, planes, workspace, m);
}

fn prefill64_prepared(cmd: &Commands<'_>, planes: &[(&Weight, &Buffer)], input: &Buffer, m: usize) {
    assert!((m > 0) && (1..=3).contains(&planes.len()));
    let k = planes[0].0.k;
    assert!(
        planes
            .iter()
            .all(|(w, _)| w.k == k && matches!(w.ty, 12..=14 | 23))
    );
    let columns = 32;
    let row_groups = m.div_ceil(64);
    if planes.len() == 1 {
        let (w, out) = planes[0];
        cmd.dispatch(
            "linear_ktile64",
            &[&w.buffer, input, out],
            &[k as u32, w.n as u32, m as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(columns), row_groups, 1],
            128,
        );
    } else {
        let third = planes.get(2).unwrap_or(&planes[1]);
        cmd.dispatch(
            "linear_multi_ktile64",
            &[
                &planes[0].0.buffer,
                &planes[1].0.buffer,
                &third.0.buffer,
                input,
                planes[0].1,
                planes[1].1,
                third.1,
            ],
            &[
                k as u32,
                planes[0].0.n as u32,
                planes[1].0.n as u32,
                if planes.len() == 3 {
                    third.0.n as u32
                } else {
                    0
                },
                m as u32,
                planes[0].0.ty,
                planes[1].0.ty,
                third.0.ty,
            ],
            [
                planes.iter().map(|(w, _)| w.n.div_ceil(columns)).sum(),
                row_groups,
                1,
            ],
            128,
        );
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_QWEN_MODEL; exact fixed-schedule GPU graph comparison"]
fn coarsened_real_graph_preserves_fixed_schedule_logits() {
    struct RestoreMode(bool);
    impl Drop for RestoreMode {
        fn drop(&mut self) {
            projection::BASELINE_FOR_TEST.with(|value| value.set(self.0));
        }
    }
    let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 1024, 4, None).unwrap();
    for m in 1usize..=4 {
        let mut reference = Vec::new();
        for baseline in [true, false] {
            let _mode = RestoreMode(projection::BASELINE_FOR_TEST.with(|v| v.replace(baseline)));
            model.reset();
            for c in &mut model.cache {
                c.table.clear(&mut model.pool);
                c.history.clear();
            }
            let prompts: Vec<Vec<u32>> = (0..m)
                .map(|s| (0..769).map(|i| 1000 + s as u32 * 700 + i).collect())
                .collect();
            for (s, prompt) in prompts.iter().enumerate() {
                assert_eq!(model.prepare(s, prompt).unwrap(), 0);
            }
            // Bypass wall-time admission: both GPU graphs see the same row
            // packing, positions and tokens. No CPU model math/reference.
            for first in (0..769).step_by(128) {
                let end = (first + 128).min(769);
                let rows: Vec<_> = prompts
                    .iter()
                    .enumerate()
                    .flat_map(|(s, p)| (first..end).map(move |i| (s, p[i], i as u32)))
                    .collect();
                model.execute(&rows, &[]).unwrap();
            }
            for step in 0..16 {
                let rows: Vec<_> = (0..m)
                    .map(|s| (s, 10000 + s as u32 * 31 + step, 769 + step))
                    .collect();
                let logits = model.execute(&rows, &(0..m).collect::<Vec<_>>()).unwrap();
                assert!(logits.iter().all(|v| v.is_finite()));
                if baseline {
                    reference.push(logits);
                } else {
                    let expected = &reference[step as usize];
                    assert_eq!(expected.len(), logits.len());
                    if let Some((at, (a, b))) = expected
                        .iter()
                        .zip(&logits)
                        .enumerate()
                        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                    {
                        panic!("m={m} step={step} logit={at}: baseline {a} paired {b}");
                    }
                }
            }
        }
    }
}

#[test]
fn coarsened_projection_preserves_f32_and_domain_tails() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    for k in [256usize, 768, 5120] {
        for shapes in [
            [(12u32, 1usize), (13, 35), (14, 67)],
            [(23, 35), (13, 48), (23, 3)],
        ] {
            let weights: Vec<_> = shapes
                .iter()
                .map(|&(ty, n)| {
                    let size = match ty {
                        12 => 144,
                        13 => 176,
                        14 => 210,
                        _ => 136,
                    };
                    // Bit-field fixtures only: all projection arithmetic runs on
                    // the GPU, compared with the independent elected baseline.
                    let mut bytes: Vec<u8> = (0..k * n / 256 * size)
                        .map(|i| ((i * 73 + i / 19 * 37 + 11) % 256) as u8)
                        .collect();
                    for (i, block) in bytes.chunks_exact_mut(size).enumerate() {
                        let at = if ty == 14 { 208 } else { 0 };
                        block[at..at + 2].copy_from_slice(
                            &half::f16::from_f32((i % 7 + 1) as f32 / 8192.).to_le_bytes(),
                        );
                        if matches!(ty, 12 | 13) {
                            block[2..4].copy_from_slice(
                                &half::f16::from_f32((i % 3 + 1) as f32 / 16384.).to_le_bytes(),
                            );
                        }
                    }
                    Weight {
                        buffer: d.upload(&bytes).unwrap(),
                        ty,
                        k,
                        n,
                    }
                })
                .collect();
            let workspace = d.alloc(128 * k * 2).unwrap();
            for m in 1..=4 {
                let x = d
                    .upload(
                        &(0..m * k)
                            .flat_map(|i| {
                                (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let allocate = || {
                    weights
                        .iter()
                        .map(|w| {
                            d.upload(
                                &vec![12345.25f32; m * w.n + 8]
                                    .iter()
                                    .flat_map(|v| v.to_le_bytes())
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>()
                };
                let a = allocate();
                let b = allocate();
                for domains in 1..=3 {
                    let base: Vec<_> = weights.iter().zip(&a).take(domains).collect();
                    let pair: Vec<_> = weights.iter().zip(&b).take(domains).collect();
                    let cmd = d.begin().unwrap();
                    if domains == 1 {
                        weights[0].linear(&cmd, &x, &a[0], m, 1., &workspace);
                    } else {
                        projections(&cmd, &base, &x, m, &workspace);
                    }
                    projection::pair(&cmd, &pair, &x, m);
                    cmd.finish().unwrap();
                    for i in 0..domains {
                        let count = m * weights[i].n;
                        let original = unsafe { a[i].read_f32(0, count + 8) };
                        let paired = unsafe { b[i].read_f32(0, count + 8) };
                        assert!(original.iter().chain(&paired).all(|v| v.is_finite()));
                        assert_eq!(
                            original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            paired.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "k={k} m={m} domains={domains} plane={i}"
                        );
                        assert!(
                            paired[count..].iter().all(|&v| v == 12345.25),
                            "output guard"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_QWEN38_GGUF; exact GPU projection election"]
fn decode_projection_election() {
    projection_election(false);
}

#[test]
#[ignore = "requires PADDOCK_QWEN38_GGUF; prefill tile election"]
fn prefill_projection_election() {
    projection_election(true);
}

fn projection_election(prefill: bool) {
    let path = std::env::var("PADDOCK_QWEN38_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for names in [
        vec!["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"],
        vec![
            "blk.3.attn_q.weight",
            "blk.3.attn_k.weight",
            "blk.3.attn_v.weight",
        ],
        vec!["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
        vec!["blk.0.ffn_down.weight"],
        vec!["blk.0.ssm_out.weight"],
        vec!["output.weight"],
    ] {
        if prefill && names[0] == "output.weight" {
            continue;
        }
        let weights: Vec<_> = names
            .iter()
            .map(|name| {
                let shape = &map.tensor_info(name).unwrap().dims;
                Weight::load(&d, &map, name, &[shape[0] as usize, shape[1] as usize]).unwrap()
            })
            .collect();
        let k = weights[0].k;
        let workspace = d.alloc((if prefill { 512 } else { 128 }) * k * 2).unwrap();
        let rows: &[usize] = if prefill {
            &[128, 256, 512]
        } else {
            &[1, 2, 3, 4]
        };
        for &m in rows {
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
            let candidate = |cmd: &Commands<'_>, planes: &[(&Weight, &Buffer)]| {
                if prefill {
                    prefill64(cmd, planes, &x, m, &workspace);
                } else {
                    projection::pair(cmd, planes, &x, m);
                }
            };
            // A few milliseconds of dispatches do not settle GPU clocks after
            // pipeline compilation. Warm both candidates for a sustained window.
            let warm = std::time::Instant::now();
            while warm.elapsed().as_millis() < 200 {
                let cmd = d.begin().unwrap();
                for _ in 0..16 {
                    candidate(&cmd, &base);
                    if weights.len() == 1 {
                        weights[0].linear(&cmd, &x, &b[0], m, 1., &workspace);
                    } else {
                        projections(&cmd, &pair, &x, m, &workspace);
                    }
                }
                cmd.finish().unwrap();
            }
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for offset in 0..2 {
                    let at = if round % 2 == 0 { offset } else { 1 - offset };
                    let cmd = d.begin().unwrap();
                    for _ in 0..16 {
                        if at == 1 {
                            candidate(&cmd, &pair);
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
                let a = unsafe { a[i].read_f32(0, m * w.n) };
                let b = unsafe { b[i].read_f32(0, m * w.n) };
                assert!(a.iter().chain(&b).all(|v| v.is_finite()));
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "m={m} {}",
                    names[i]
                );
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "projection {:?} types {:?} m={m} prefill={prefill}: previous {:.4} ms candidate {:.4} ms",
                names,
                weights.iter().map(|w| w.ty).collect::<Vec<_>>(),
                times[0][3],
                times[1][3]
            );
        }
    }
}
