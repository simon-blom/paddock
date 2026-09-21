use super::*;

#[test]
fn image_projection_expansion_matches_inline_q8_and_preserves_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let k = 768usize;
    for n in [256usize, 1025] {
        let mut bytes: Vec<_> = (0..k * n / 32 * 34)
            .map(|i| ((i * 73 + i / 19 * 37 + 11) % 256) as u8)
            .collect();
        for (i, b) in bytes.chunks_exact_mut(34).enumerate() {
            let scale = if i % 2 == 0 { 1. / 1024. } else { -1. / 512. };
            b[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        }
        let w = Weight {
            buffer: d.upload(&bytes).unwrap(),
            ty: 8,
            k,
            n,
        };
        for rows in [1024usize, 1025, 1422, 2048] {
            let x = d
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let prefix = k * rows.div_ceil(128) * 128;
            let work = d.alloc((prefix + k * n) * 2).unwrap();
            let expected = d.alloc(rows * n * 4).unwrap();
            let out = d
                .upload(
                    &(0..(rows + 128) * n)
                        .flat_map(|_| f32::NAN.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let cmd = d.begin().unwrap();
            w.linear(&cmd, &x, &expected, rows, 1., &work);
            image_projection(&cmd, &[(&w, &out)], &x, rows, &work);
            cmd.finish().unwrap();
            let expected = unsafe { expected.read_f32(0, rows * n) };
            let actual = unsafe { out.read_f32(0, rows * n) };
            assert!(
                actual.iter().all(|x| x.is_finite()) && actual == expected,
                "expanded projection changed n={n} rows={rows}"
            );
            assert!(
                unsafe { out.read_f32(rows * n, 128 * n) }
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
    }
}

#[test]
fn compact_dflash_staging_matches_independent_gpu_expansion_with_ragged_guards() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let (k, n) = (768usize, 71usize);
    for (ty, size) in [(12u32, 144usize), (13, 176), (14, 210), (23, 136)] {
        let mut bytes: Vec<_> = (0..k * n / 256 * size)
            .map(|i| ((i * 73 + i / 19 * 37 + 11) % 256) as u8)
            .collect();
        for b in bytes.chunks_exact_mut(size) {
            let at = if ty == 14 { 208 } else { 0 };
            b[at..at + 2].copy_from_slice(&half::f16::from_f32(1. / 1024.).to_le_bytes());
            if matches!(ty, 12 | 13) {
                b[2..4].copy_from_slice(&half::f16::from_f32(1. / 4096.).to_le_bytes());
            }
        }
        let w = d.upload(&bytes).unwrap();
        for rows in [33usize, 63, 64, 65, 96, 97, 128, 129, 255, 256, 511, 512] {
            let x = d
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let expanded = d.alloc(k * n * 2).unwrap();
            let xf = d.alloc(k * rows * 2).unwrap();
            let padded = d.alloc(k * rows.div_ceil(128) * 128 * 2).unwrap();
            let expected = d.alloc(rows * n * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, ty, 1f32.to_bits()];
            let cmd = d.begin().unwrap();
            // Independent scalar GPU dequantization + a different matrix
            // tile checks the compact staging, not a CPU tensor oracle.
            cmd.dispatch(
                "linear_prepare",
                &[&w, &x, &expanded, &xf],
                &p,
                [(k * rows.max(n)).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "linear_mpp",
                &[&expanded, &xf, &expected],
                &p,
                [n.div_ceil(64), rows.div_ceil(32), 1],
                128,
            );
            cmd.dispatch(
                "linear_input_padded",
                &[&x, &padded],
                &p,
                [(k * rows.div_ceil(128) * 128).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish().unwrap();
            let expected = unsafe { expected.read_f32(0, rows * n) };
            for (name, tile) in [
                ("muse_df_ktile64", 64),
                ("muse_df_ktile96", 96),
                ("muse_df_ktile128", 128),
            ] {
                let out = d
                    .upload(
                        &(0..(rows + 128) * n)
                            .flat_map(|_| f32::NAN.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    name,
                    &[&w, &padded, &out],
                    &p,
                    [n.div_ceil(32), rows.div_ceil(tile), 1],
                    128,
                );
                cmd.finish().unwrap();
                let actual = unsafe { out.read_f32(0, rows * n) };
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    actual.iter().all(|v| v.is_finite()) && error < 0.0001,
                    "{name} type={ty} rows={rows} error={error}"
                );
                assert!(
                    unsafe { out.read_f32(rows * n, 128 * n) }
                        .iter()
                        .all(|v| v.is_nan())
                );
            }
        }
    }
}

#[test]
fn q8_f32_matrix_preserves_operands_ragged_rows_and_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let (k, n) = (6656usize, 71usize);
    let mut bytes: Vec<_> = (0..k * n / 32 * 34)
        .map(|i| ((i * 73 + i / 19 * 37 + 11) % 256) as u8)
        .collect();
    for b in bytes.chunks_exact_mut(34) {
        b[..2].copy_from_slice(&half::f16::from_f32(1. / 1024.).to_le_bytes());
    }
    let w = d.upload(&bytes).unwrap();
    for rows in [1usize, 3, 8, 9, 16, 17, 31, 32, 33, 64] {
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let expected = d.alloc(rows * n * 4).unwrap();
        let p = [k as u32, n as u32, rows as u32, 8, 1f32.to_bits()];
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "linear_q8_r1",
            &[&w, &x, &expected],
            &p,
            [n.div_ceil(4), rows, 1],
            128,
        );
        cmd.finish().unwrap();
        let expected = unsafe { expected.read_f32(0, rows * n) };
        for (name, bm) in [
            ("muse_q8_f32_16", 16),
            ("muse_q8_f32_32", 32),
            ("muse_q8_f32_64", 64),
        ] {
            let out = d
                .upload(
                    &(0..(rows + 64) * n)
                        .flat_map(|_| f32::NAN.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                name,
                &[&w, &x, &out],
                &p,
                [n.div_ceil(16), rows.div_ceil(bm), 1],
                128,
            );
            cmd.finish().unwrap();
            let actual = unsafe { out.read_f32(0, rows * n) };
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                actual.iter().all(|v| v.is_finite()) && error < 0.0001,
                "{name} rows={rows} error={error}"
            );
            assert!(
                unsafe { out.read_f32(rows * n, 64 * n) }
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
    }
}

#[test]
#[ignore = "actual Muse Q8 projection microbenchmark; not a serving comparison"]
fn muse_q8_verification_projection_election() {
    let path = std::env::var("PADDOCK_MUSE_GGUF").unwrap();
    let map = MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for (name, k, n) in [
        ("blk.0.attn_q.weight", 6656usize, 4096usize),
        ("blk.0.ffn_down.weight", 19968, 6656),
        ("output.weight", 6656, 202048),
    ] {
        let w = Weight::load(&d, &map, name, &[k, n]).unwrap();
        assert_eq!(w.ty, 8);
        for rows in [8usize, 16, 32, 64] {
            let x = d
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let out = d.alloc(rows * n * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, 8, 1f32.to_bits()];
            let baseline = if rows <= 8 {
                ("linear_q8_r8", 4, 8)
            } else {
                ("linear_q8_r16", 4, 16)
            };
            let candidates = [
                baseline,
                ("muse_q8_f32_16", 16, 16),
                ("muse_q8_f32_32", 16, 32),
                ("muse_q8_f32_64", 16, 64),
            ];
            let mut times = vec![Vec::new(); candidates.len()];
            for round in 0..7 {
                for j in 0..candidates.len() {
                    let at = if round % 2 == 0 {
                        j
                    } else {
                        candidates.len() - j - 1
                    };
                    let (kernel, bn, bm) = candidates[at];
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, &x, &out],
                        &p,
                        [n.div_ceil(bn), rows.div_ceil(bm), 1],
                        128,
                    );
                    let ms = cmd.finish().unwrap() * 1000.;
                    if round > 0 {
                        times[at].push(ms);
                    }
                }
            }
            for (at, (kernel, _, _)) in candidates.iter().enumerate() {
                times[at].sort_by(f64::total_cmp);
                eprintln!(
                    "MUSE_Q8 {name} rows={rows} {kernel} median_ms={:.4}",
                    times[at][3]
                );
            }
        }
    }
}

#[test]
#[ignore = "actual Muse image-prefill projection diagnostic; not serving performance"]
fn muse_q8_image_prefill_projection_election() {
    let path = std::env::var("PADDOCK_MUSE_GGUF").unwrap();
    let map = MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for (name, k, n) in [
        ("blk.0.attn_q.weight", 6656usize, 4096usize),
        ("blk.0.ffn_down.weight", 19968, 6656),
        ("blk.0.ffn_gate.weight", 6656, 19968),
    ] {
        let w = Weight::load(&d, &map, name, &[k, n]).unwrap();
        assert_eq!(w.ty, 8);
        let expanded = d.alloc(k * n * 2).unwrap();
        for rows in [512usize, 1024, 1422, 2048] {
            let x = d
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let xf = d.alloc((rows.div_ceil(128) * 128 * k + k * n) * 2).unwrap();
            let a = d.alloc(rows * n * 4).unwrap();
            let b = d.alloc(rows * n * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, 8, 1f32.to_bits()];
            let c = d.alloc(rows * n * 4).unwrap();
            let mut times = [Vec::new(), Vec::new(), Vec::new()];
            for round in 0..7 {
                for offset in 0..3 {
                    let at = (round + offset) % 3;
                    let cmd = d.begin().unwrap();
                    if at == 0 {
                        w.linear(&cmd, &x, &a, rows, 1., &xf);
                    } else if at == 1 {
                        cmd.dispatch(
                            "linear_prepare",
                            &[&w.buffer, &x, &expanded, &xf],
                            &p,
                            [(k * rows.max(n)).div_ceil(256), 1, 1],
                            256,
                        );
                        cmd.dispatch(
                            "linear_mpp",
                            &[&expanded, &xf, &b],
                            &p,
                            [n.div_ceil(64), rows.div_ceil(32), 1],
                            128,
                        );
                    } else {
                        cmd.dispatch(
                            "linear_input_padded",
                            &[&x, &xf],
                            &p,
                            [(k * rows.div_ceil(128) * 128).div_ceil(256), 1, 1],
                            256,
                        );
                        cmd.dispatch(
                            "muse_q8_expand",
                            &[&w.buffer, &xf],
                            &p,
                            [(k * n).div_ceil(1024), 1, 1],
                            256,
                        );
                        cmd.dispatch(
                            "linear_kexpanded128",
                            &[&xf, &c],
                            &p,
                            [n.div_ceil(64), rows.div_ceil(128), 1],
                            128,
                        );
                    }
                    let ms = cmd.finish().unwrap() * 1000.;
                    if round > 0 {
                        times[at].push(ms);
                    }
                }
            }
            let a = unsafe { a.read_f32(0, rows * n) };
            let b = unsafe { b.read_f32(0, rows * n) };
            let c = unsafe { c.read_f32(0, rows * n) };
            assert!(a == c, "expanded128 changed {name} rows={rows}");
            let max = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                b.iter().all(|v| v.is_finite()) && max < 0.0001,
                "{name} rows={rows} max={max}"
            );
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "MUSE_IMAGE_PROJECTION {name} rows={rows} tile_ms={} expanded32_ms={} expanded128_ms={} max={max}",
                times[0][3], times[1][3], times[2][3]
            );
        }
    }
}
