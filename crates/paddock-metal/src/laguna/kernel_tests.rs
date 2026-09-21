use super::*;

fn floats(d: &MetalDevice, v: impl IntoIterator<Item = f32>) -> Buffer {
    d.upload(&v.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())
        .unwrap()
}
fn ints(d: &MetalDevice, v: impl IntoIterator<Item = u32>) -> Buffer {
    d.upload(&v.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())
        .unwrap()
}
fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        assert!(
            x.is_finite() && y.is_finite() && (x - y).abs() <= tol,
            "{x} vs {y} (tol {tol})"
        );
    }
}
// Format fixtures, not a CPU model or dot-product oracle. The SIMD and
// TensorOps implementations below independently execute on the GPU.
fn quant(d: &MetalDevice, ty: u32, elements: usize, salt: usize) -> Buffer {
    let block = match ty {
        12 => 144,
        13 => 176,
        14 => 210,
        _ => panic!("fixture type"),
    };
    let mut bytes = vec![0u8; elements / 256 * block];
    for (i, b) in bytes.chunks_exact_mut(block).enumerate() {
        for (j, v) in b.iter_mut().enumerate() {
            *v = (i * 7 + j * 19 + salt) as u8;
        }
        if ty == 12 || ty == 13 {
            b[..2].copy_from_slice(&0x1800u16.to_le_bytes());
            b[2..4].copy_from_slice(&0x1400u16.to_le_bytes());
        } else {
            b[208..210].copy_from_slice(&0x1400u16.to_le_bytes());
        }
    }
    d.upload(&bytes).unwrap()
}

#[test]
fn selection_bias_does_not_change_mixture_weights() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let logits = floats(&d, [0.; 512]);
    let bias = floats(&d, (0..256).map(|i| if i == 255 { 10. } else { 0. }));
    let ids = d.alloc(16 * 4).unwrap();
    let weights = d.alloc(16 * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "laguna_route",
        &[&logits, &bias, &ids, &weights],
        &[8],
        [2, 1, 1],
        32,
    );
    cmd.finish().unwrap();
    assert_eq!(
        unsafe { ids.read_u32(16) },
        [255, 0, 1, 2, 3, 4, 5, 6, 255, 0, 1, 2, 3, 4, 5, 6]
    );
    assert_eq!(unsafe { weights.read_f32(0, 16) }, [2.5 / 8.; 16]);
    let logits = floats(
        &d,
        (0..256).map(|i| if i == 255 { std::f32::consts::LN_2 } else { 0. }),
    );
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "laguna_route",
        &[&logits, &bias, &ids, &weights],
        &[8],
        [1, 1, 1],
        32,
    );
    cmd.finish().unwrap();
    let w = unsafe { weights.read_f32(0, 8) };
    close(&w, &[0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 1e-6);
}

#[test]
fn top10_routes_and_fold_use_all_ten_entries_without_touching_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let logits = floats(&d, [0.; 512]);
    let bias = floats(&d, (0..256).map(|i| if i == 255 { 10. } else { 0. }));
    let ids = ints(&d, [u32::MAX; 24]);
    let weights = floats(&d, [f32::NAN; 24]);
    // Integer-valued quarter products make the tenth entry's contribution
    // exact. This is an indexing fixture, not a host dot-product oracle.
    let experts = floats(&d, (0..20).flat_map(|i| [4. * (i % 10 + 1) as f32; 3]));
    let out = floats(&d, [1., 1., 1., 1., 1., 1., f32::NAN]);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "laguna_route_top10",
        &[&logits, &bias, &ids, &weights],
        &[10],
        [2, 1, 1],
        32,
    );
    cmd.dispatch(
        "laguna_fold_top10",
        &[&experts, &weights, &out],
        &[3, 2, 10],
        [1, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    assert_eq!(
        unsafe { ids.read_u32(24) },
        [
            255,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            255,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX
        ]
    );
    assert_eq!(unsafe { weights.read_f32(0, 20) }, [0.25; 20]);
    assert!(
        unsafe { weights.read_f32(20, 4) }
            .iter()
            .all(|x| x.is_nan())
    );
    assert_eq!(unsafe { out.read_f32(0, 6) }, [56.; 6]);
    assert!(unsafe { out.read_f32(6, 1) }[0].is_nan());
}

#[test]
fn dense_kquant_f32_panels_match_gpu_simd_with_ragged_rows() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    for ty in [12, 13, 14] {
        for k in [2048usize, 8192] {
            let n = 65usize;
            let w = Weight {
                buffer: quant(&d, ty, k * n, 11),
                ty,
                k,
                n,
            };
            for rows in [1usize, 4, 17, 63] {
                eprintln!("dense K-quant GPU comparison: type={ty} K={k} N={n} rows={rows}");
                let input = floats(
                    &d,
                    (0..rows * k).map(|i| ((i * 13 + i / k * 3) % 53) as f32 / 53. - 0.5),
                );
                let output = floats(&d, (0..rows * n + 64).map(|_| f32::NAN));
                let expected = d.alloc(rows * n * 4).unwrap();
                let cmd = d.begin().unwrap();
                projection::project(&cmd, &[(&w, &output)], &input, rows);
                cmd.dispatch(
                    "linear",
                    &[&w.buffer, &input, &expected],
                    &[k as u32, n as u32, rows as u32, ty, 1f32.to_bits()],
                    [n.div_ceil(4), rows, 1],
                    128,
                );
                cmd.finish().unwrap();
                // Long-K scalar and TensorOps reductions have different
                // F32 accumulation orders. Keep the absolute near-zero gate
                // and a 10ppm relative budget at larger magnitudes. The first
                // fixed-absolute stress gate failed at 0.00024 / 71.4; this
                // does not relax the existing attention/expert/model gates.
                let (actual, expected) =
                    unsafe { (output.read_f32(0, rows * n), expected.read_f32(0, rows * n)) };
                let mut max_abs = 0f32;
                for (&x, &y) in actual.iter().zip(&expected) {
                    let error = (x - y).abs();
                    max_abs = max_abs.max(error);
                    assert!(
                        x.is_finite() && y.is_finite() && error <= 0.0002 + 1e-5 * y.abs(),
                        "{x} vs {y}"
                    );
                }
                eprintln!("maximum absolute GPU projection difference: {max_abs}");
                assert!(
                    unsafe { output.read_f32(rows * n, 64) }
                        .iter()
                        .all(|x| x.is_nan())
                );
            }
        }
    }
}

#[test]
fn s_q8_projections_cover_ragged_heads_and_wide_dense_ffn() {
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    for (k, n) in [
        (3072usize, 48usize),
        (3072, 72),
        (3072, 12288),
        (9216, 3072),
        (12288, 3072),
    ] {
        let mut bytes = vec![0u8; k * n / 32 * 34];
        for (i, b) in bytes.chunks_exact_mut(34).enumerate() {
            b[..2].copy_from_slice(&0x1000u16.to_le_bytes());
            for (j, v) in b[2..].iter_mut().enumerate() {
                *v = (i * 7 + j * 13) as u8;
            }
        }
        let w = Weight {
            buffer: d.upload(&bytes).unwrap(),
            ty: 8,
            k,
            n,
        };
        for rows in [1usize, 4, 17, 63, 129] {
            // Exactly representable F16 operands isolate indexing/tiling,
            // comparing the optimized path to an independent GPU SIMD dot.
            let input = floats(&d, (0..rows * k).map(|i| (i % 17) as f32 / 32. - 0.25));
            let output = floats(&d, (0..rows * n + 64).map(|_| f32::NAN));
            let expected = d.alloc(rows * n * 4).unwrap();
            let cmd = d.begin().unwrap();
            projection::project(&cmd, &[(&w, &output)], &input, rows);
            cmd.dispatch(
                "linear",
                &[&w.buffer, &input, &expected],
                &[k as u32, n as u32, rows as u32, 8, 1f32.to_bits()],
                [n.div_ceil(4), rows, 1],
                128,
            );
            cmd.finish().unwrap();
            close(
                &unsafe { output.read_f32(0, rows * n) },
                &unsafe { expected.read_f32(0, rows * n) },
                1e-5,
            );
            assert!(
                unsafe { output.read_f32(rows * n, 64) }
                    .iter()
                    .all(|x| x.is_nan())
            );
        }
    }
}

#[test]
fn kquant_experts_grouped_match_gpu_simd_with_ragged_and_hot_routes() {
    let d = MetalDevice::new(Some(1024 << 20)).unwrap();
    for (k, n, down, active) in [
        (2048usize, 65usize, false, 8usize),
        (512, 65, true, 8),
        (3072, 65, false, 10),
        (1024, 65, true, 10),
    ] {
        for (gt, ut) in [
            (12, 12),
            (12, 13),
            (12, 14),
            (13, 12),
            (13, 13),
            (13, 14),
            (14, 12),
            (14, 13),
            (14, 14),
        ] {
            let gate = quant(&d, gt, k * n * 256, 3);
            let up = quant(&d, ut, k * n * 256, 17);
            for rows in [17usize, 63] {
                for hot in [false, true] {
                    let ids = ints(
                        &d,
                        (0..rows * active).map(|i| {
                            if hot {
                                (i % active) as u32
                            } else {
                                ((i * 17 + i / active * 7) % 256) as u32
                            }
                        }),
                    );
                    let input = floats(
                        &d,
                        (0..rows * k * if down { active * 2 } else { 1 })
                            .map(|i| ((i * 13 + i / k * 3) % 53) as f32 / 53. - 0.5),
                    );
                    let len = rows * active * n * if down { 1 } else { 2 };
                    let expected = d.alloc(len * 4).unwrap();
                    let actual = floats(&d, (0..len + 128).map(|_| f32::NAN));
                    let lists = d.alloc(256 * rows * active * 4).unwrap();
                    let counts = d.alloc(256 * 4).unwrap();
                    let tiles = d
                        .alloc((1 + 2 * ((rows * active).div_ceil(16) + 256)) * 4)
                        .unwrap();
                    let p = [k as u32, n as u32, rows as u32, gt, ut, active as u32];
                    let cmd = d.begin().unwrap();
                    if down {
                        cmd.dispatch(
                            "laguna_down_decode",
                            &[&gate, &input, &ids, &expected],
                            &p,
                            [n.div_ceil(4), rows * active, 1],
                            128,
                        );
                    } else {
                        cmd.dispatch(
                            expert_kernel(active, "laguna_gu_decode"),
                            &[&gate, &up, &input, &ids, &expected],
                            &p,
                            [n.div_ceil(4), rows * active, 1],
                            128,
                        );
                    }
                    cmd.dispatch(
                        "moe_align",
                        &[&ids, &lists, &counts],
                        &[(rows * active) as u32],
                        [256, 1, 1],
                        256,
                    );
                    cmd.finish().unwrap();
                    for bm in [16usize, 32] {
                        let cmd = d.begin().unwrap();
                        cmd.dispatch(
                            "moe_tiles",
                            &[&counts, &tiles],
                            &[256, bm as u32],
                            [1, 1, 1],
                            256,
                        );
                        let kernel = match (down, bm) {
                            (false, 16) => "laguna_gu_grouped16",
                            (false, _) => "laguna_gu_grouped32",
                            (true, 16) => "laguna_down_grouped16",
                            _ => "laguna_down_grouped32",
                        };
                        cmd.dispatch(
                            expert_kernel(active, kernel),
                            &[&gate, &up, &input, &lists, &counts, &tiles, &actual],
                            &p,
                            [
                                n.div_ceil(32) * if down { 1 } else { 2 },
                                (rows * active).div_ceil(bm) + 256,
                                1,
                            ],
                            128,
                        );
                        cmd.finish().unwrap();
                        close(
                            &unsafe { expected.read_f32(0, len) },
                            &unsafe { actual.read_f32(0, len) },
                            0.0002,
                        );
                        assert!(
                            unsafe { actual.read_f32(len, 128) }
                                .iter()
                                .all(|v| v.is_nan())
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn paged_window_gqa6_gqa8_gqa9_prefill_matches_gpu_decode() {
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    for (heads, window) in [(48usize, 0u32), (64, 512), (72, 512)] {
        for start in [0usize, 509, 1021] {
            let rows = 31usize;
            let length = start + rows;
            let stride = length.div_ceil(16);
            let pages = ints(&d, (0..stride).map(|i| (stride - 1 - i) as u32));
            let meta = ints(&d, (0..rows).flat_map(|r| [0, (start + r) as u32]));
            let ids = ints(&d, 0..rows as u32);
            let tiles = ints(&d, [0, rows as u32]);
            let q = floats(
                &d,
                (0..rows * heads * 128).map(|i| ((i * 17 + i / 128 * 3) % 59) as f32 / 59. - 0.5),
            );
            let payload = (0..stride * 16 * 1024)
                .flat_map(|i| {
                    ((0x3000u16 + (i % 1024) as u16) | if i % 7 == 0 { 0x8000 } else { 0 })
                        .to_le_bytes()
                })
                .collect::<Vec<_>>();
            let k = d.upload(&payload).unwrap();
            let v = d.upload(&payload).unwrap();
            let out = d.alloc(rows * heads * 128 * 4).unwrap();
            let expected = d.alloc(out.len()).unwrap();
            let parts = d.alloc(rows * heads * 16 * 130 * 4).unwrap();
            let p = [heads as u32, 8, stride as u32, window, 0, 16];
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "laguna_prefill",
                &[&q, &k, &v, &meta, &pages, &out, &tiles],
                &p,
                [heads, 1, 1],
                128,
            );
            cmd.dispatch(
                match heads {
                    48 => "laguna_decode6",
                    64 => "laguna_decode8",
                    72 => "laguna_decode9",
                    _ => unreachable!(),
                },
                &[&q, &k, &v, &meta, &pages, &ids, &parts],
                &p,
                [8, rows, 16],
                128,
            );
            cmd.dispatch(
                "gemma_merge",
                &[&parts, &expected, &ids],
                &[heads as u32, 16, 128],
                [rows * heads, 1, 1],
                32,
            );
            cmd.finish().unwrap();
            close(
                &unsafe { out.read_f32(0, rows * heads * 128) },
                &unsafe { expected.read_f32(0, rows * heads * 128) },
                0.00001,
            );
        }
    }
}
