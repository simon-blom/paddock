//! Same-weight GPU diagnostics for the Flash Next F32 projection election.
use super::{MetalDevice, WIDE, close, fixture, guard, guarded, read_f, residual, upload};

#[test]
#[ignore = "requires elected Flash Next GGUF and independent MPS state fixtures"]
fn all_hc_injection_planes_preserve_ordered_gpu_results() {
    let (dir, manifest, map) = fixture();
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let sample = manifest["hc"]
        .as_array()
        .unwrap()
        .iter()
        .max_by_key(|c| c["rows"].as_u64().unwrap())
        .unwrap();
    let input = read_f(dir.join(format!("{}.norm.f32", sample["id"].as_str().unwrap())));
    for layer in 0..48 {
        for family in ["attn", "ffn"] {
            let name = format!("blk.{layer}.hc_{family}_inject.weight");
            let w = residual::load_weight(&d, &map, &name, &[WIDE, 4], 0).unwrap();
            for rows in [1, 4, 9, 31, 128] {
                let x = upload(
                    &d,
                    &input
                        .iter()
                        .copied()
                        .cycle()
                        .take(rows * WIDE)
                        .collect::<Vec<_>>(),
                );
                let old = guarded(&d, &vec![0.; rows * 4]);
                let ordered = guarded(&d, &vec![0.; rows * 4]);
                let parallel = guarded(&d, &vec![0.; rows * 4]);
                let cmd = d.begin().unwrap();
                for (kernel, out, groups, threads) in [
                    ("linear", &old, [1, rows, 1], 128),
                    ("q4x_f32_mv", &ordered, [1, rows, 1], 128),
                    ("q4x_inject_parallel", &parallel, [4, rows, 1], 256),
                ] {
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, &x, out],
                        &[WIDE as u32, 4, rows as u32, 0, 1f32.to_bits()],
                        groups,
                        threads,
                    );
                }
                cmd.finish().unwrap();
                let before = unsafe { old.read_f32(0, rows * 4) };
                let after = unsafe { ordered.read_f32(0, rows * 4) };
                assert!(before.iter().all(|v| v.is_finite()));
                assert_eq!(
                    before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{name}, rows={rows}"
                );
                close(&unsafe { parallel.read_f32(0, rows * 4) }, &before, &name);
                for b in [&old, &ordered, &parallel] {
                    guard(b, rows * 4);
                }
            }
        }
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
#[ignore = "requires elected GGUF, MPS fixtures and otherwise idle GPU; timing only"]
fn time_hc_injection_candidates() {
    assert!(std::env::var_os("PADDOCK_METAL_PROFILE").is_none());
    let (_, _, map) = fixture();
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let w = residual::load_weight(&d, &map, "blk.0.hc_attn_inject.weight", &[WIDE, 4], 0).unwrap();
    for rows in [1, 4, 128] {
        let x = upload(&d, &vec![0.125; rows * WIDE]);
        let y = d.alloc(rows * 4 * 4).unwrap();
        let mut times = [vec![], vec![], vec![]];
        for round in 0..24 {
            // Rotate order each round so clock/cache drift does not always
            // favor the same candidate. All arms read the same raw weights.
            for j in 0..3 {
                let i = (j + round) % 3;
                let (kernel, groups, threads) = [
                    ("linear", [1, rows, 1], 128),
                    ("q4x_f32_mv", [1, rows, 1], 128),
                    ("q4x_inject_parallel", [4, rows, 1], 256),
                ][i];
                let cmd = d.begin().unwrap();
                for _ in 0..32 {
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, &x, &y],
                        &[WIDE as u32, 4, rows as u32, 0, 1f32.to_bits()],
                        groups,
                        threads,
                    );
                }
                let us = cmd.finish().unwrap() * 1e6 / 32.;
                if round >= 3 {
                    times[i].push(us);
                }
            }
        }
        for (name, t) in ["linear", "ordered", "parallel"]
            .into_iter()
            .zip(&mut times)
        {
            t.sort_by(f64::total_cmp);
            eprintln!(
                "Flash Next HC {name} rows={rows}: median_us={:.3}",
                t[t.len() / 2]
            );
        }
    }
}

#[test]
#[ignore = "requires elected Flash Next GGUF and MPS fixtures; GPU comparison only"]
fn q8_banked_staging_preserves_all_actual_planes() {
    let (dir, manifest, map) = fixture();
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    let id = manifest["hc"][0]["id"].as_str().unwrap();
    let input = read_f(dir.join(format!("{id}.norm.f32")));
    let mut count = 0;
    for info in map
        .tensor_infos()
        .filter(|i| i.raw_type == 8 && i.dims.len() == 2)
    {
        let k = info.dims[0] as usize;
        let n = info.dims[1] as usize;
        assert!(k.is_multiple_of(64));
        let w = residual::load_weight(&d, &map, &info.name, &[k, n], 8).unwrap();
        for rows in [9, 31, 33, 128] {
            let x = upload(
                &d,
                &input
                    .iter()
                    .copied()
                    .cycle()
                    .take(rows * k)
                    .collect::<Vec<_>>(),
            );
            let old = guarded(&d, &vec![0.; rows * n]);
            let new = guarded(&d, &vec![0.; rows * n]);
            let cmd = d.begin().unwrap();
            for (kernel, y) in [("q4x_q8_mm", &old), ("q4x_q8_mm64", &new)] {
                cmd.dispatch(
                    kernel,
                    &[&w.buffer, &x, y],
                    &[k as u32, n as u32, rows as u32, 8, 1f32.to_bits()],
                    [n.div_ceil(32), rows.div_ceil(32), 1],
                    128,
                );
            }
            cmd.finish().unwrap();
            let a = unsafe { old.read_f32(0, rows * n) };
            let b = unsafe { new.read_f32(0, rows * n) };
            assert!(a.iter().all(|v| v.is_finite()));
            for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{} rows={rows} at={i}", info.name);
            }
            guard(&old, rows * n);
            guard(&new, rows * n);
        }
        count += 1;
    }
    assert_eq!(count, 248);
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
#[ignore = "requires elected GGUF, MPS fixtures and an idle GPU; timing only"]
fn time_q8_banked_staging() {
    assert!(std::env::var_os("PADDOCK_METAL_PROFILE").is_none());
    let (_, _, map) = fixture();
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    for (name, k, n) in [
        ("blk.0.hc_attn_down.weight", 10240, 320),
        ("blk.0.hc_attn_up.weight", 320, 10240),
        ("blk.0.ffn_down_shexp.weight", 640, 2560),
    ] {
        let w = residual::load_weight(&d, &map, name, &[k, n], 8).unwrap();
        for rows in [9, 33, 128] {
            let x = upload(&d, &vec![0.125; rows * k]);
            let y = d.alloc(rows * n * 4).unwrap();
            let mut times = [vec![], vec![]];
            for round in 0..24 {
                for j in 0..2 {
                    let i = (j + round) % 2;
                    let kernel = ["q4x_q8_mm", "q4x_q8_mm64"][i];
                    let cmd = d.begin().unwrap();
                    for _ in 0..8 {
                        cmd.dispatch(
                            kernel,
                            &[&w.buffer, &x, &y],
                            &[k as u32, n as u32, rows as u32, 8, 1f32.to_bits()],
                            [n.div_ceil(32), rows.div_ceil(32), 1],
                            128,
                        );
                    }
                    let us = cmd.finish().unwrap() * 1e6 / 8.;
                    if round >= 3 {
                        times[i].push(us);
                    }
                }
            }
            for (label, t) in ["single", "banked"].into_iter().zip(&mut times) {
                t.sort_by(f64::total_cmp);
                eprintln!(
                    "Flash Next Q8 {name} rows={rows} {label}: median_us={:.3}",
                    t[t.len() / 2]
                );
            }
        }
    }
}

fn split_down(
    cmd: &crate::device::Commands<'_>,
    w: &crate::weights::Weight,
    x: &crate::device::Buffer,
    partial: &crate::device::Buffer,
    y: &crate::device::Buffer,
    rows: usize,
    splits: usize,
) {
    assert!(matches!(splits, 4 | 8 | 16 | 32));
    cmd.dispatch(
        "q4x_hc_down_split",
        &[&w.buffer, x, partial],
        &[rows as u32, splits as u32],
        [10, rows.div_ceil(32), splits],
        128,
    );
    cmd.dispatch(
        "q4x_hc_down_reduce",
        &[partial, y],
        &[rows as u32, splits as u32],
        [(rows * 320).div_ceil(256), 1, 1],
        256,
    );
}

#[test]
#[ignore = "requires elected GGUF, state fixtures and independent MPS HC-down fixtures"]
fn hc_down_split_covers_every_plane_and_split_count() {
    let (dir, manifest, map) = fixture();
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    let id = manifest["hc"][0]["id"].as_str().unwrap();
    let input = read_f(dir.join(format!("{id}.norm.f32")));
    let reference = std::path::PathBuf::from(
        std::env::var("PADDOCK_FLASH_NEXT_DOWN_REFERENCE")
            .expect("independent MPS HC-down fixtures"),
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("results.json")).unwrap()).unwrap();
    assert_eq!(metadata["device"], "mps");
    assert_eq!(metadata["complete"], true);
    assert_eq!(metadata["cases"].as_array().unwrap().len(), 291);
    let mut failures = [0usize; 5];
    let mut maxima = [0f32; 5];
    let mut compare = |arm: usize, actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite());
            let error = (a - b).abs();
            maxima[arm] = maxima[arm].max(error);
            if error > 3e-5 + 3e-5 * b.abs() {
                failures[arm] += 1;
            }
        }
    };
    let mut count = 0;
    for info in map
        .tensor_infos()
        .filter(|i| i.raw_type == 8 && i.dims == [10240, 320])
    {
        let w = residual::load_weight(&d, &map, &info.name, &[10240, 320], 8).unwrap();
        for rows in [9, 33, 128] {
            let x = upload(
                &d,
                &input
                    .iter()
                    .copied()
                    .cycle()
                    .take(rows * WIDE)
                    .collect::<Vec<_>>(),
            );
            let old = guarded(&d, &vec![0.; rows * 320]);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4x_q8_mm",
                &[&w.buffer, &x, &old],
                &[10240, 320, rows as u32, 8, 1f32.to_bits()],
                [10, rows.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            let expected = read_f(reference.join(format!("{}-r{rows}.f32", info.name)));
            // The old sequential MPP path is a measured diagnostic arm, not
            // the oracle for a changed reduction tree. Keep the independent
            // GPU allowance unchanged and report old-path failures too.
            compare(0, &unsafe { old.read_f32(0, rows * 320) }, &expected);
            for (arm, splits) in [4, 8, 16, 32].into_iter().enumerate() {
                let partial = guarded(&d, &vec![0.; rows * 320 * splits]);
                let y = guarded(&d, &vec![0.; rows * 320]);
                let cmd = d.begin().unwrap();
                split_down(&cmd, &w, &x, &partial, &y, rows, splits);
                cmd.finish().unwrap();
                compare(arm + 1, &unsafe { y.read_f32(0, rows * 320) }, &expected);
                guard(&partial, rows * 320 * splits);
                guard(&y, rows * 320);
            }
            guard(&old, rows * 320);
        }
        count += 1;
    }
    assert_eq!(count, 97);
    for (i, splits) in [0, 4, 8, 16, 32].into_iter().enumerate() {
        eprintln!(
            "HC down independent GPU splits={splits}: violations={} max_abs={:e}",
            failures[i], maxima[i]
        );
    }
    assert_eq!(
        &failures[1..],
        &[0; 4],
        "split variants must meet the unchanged independent GPU gate"
    );
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
#[ignore = "requires elected GGUF, MPS fixtures and idle GPU; timing only"]
fn time_hc_down_split_candidates() {
    assert!(std::env::var_os("PADDOCK_METAL_PROFILE").is_none());
    let (_, _, map) = fixture();
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    let w = residual::load_weight(&d, &map, "blk.0.hc_attn_down.weight", &[10240, 320], 8).unwrap();
    for rows in [9, 33, 128] {
        let x = upload(&d, &vec![0.125; rows * WIDE]);
        let y = d.alloc(rows * 320 * 4).unwrap();
        let partial = d.alloc(rows * WIDE * 4).unwrap();
        let mut times = [vec![], vec![], vec![], vec![], vec![]];
        for round in 0..24 {
            for j in 0..5 {
                let i = (round + j) % 5;
                let cmd = d.begin().unwrap();
                for _ in 0..8 {
                    if i == 0 {
                        cmd.dispatch(
                            "q4x_q8_mm",
                            &[&w.buffer, &x, &y],
                            &[10240, 320, rows as u32, 8, 1f32.to_bits()],
                            [10, rows.div_ceil(32), 1],
                            128,
                        );
                    } else {
                        split_down(&cmd, &w, &x, &partial, &y, rows, [0, 4, 8, 16, 32][i]);
                    }
                }
                let us = cmd.finish().unwrap() * 1e6 / 8.;
                if round >= 3 {
                    times[i].push(us);
                }
            }
        }
        for (splits, t) in [0, 4, 8, 16, 32].into_iter().zip(&mut times) {
            t.sort_by(f64::total_cmp);
            eprintln!(
                "Flash Next HC down splits={splits} rows={rows}: median_us={:.3}",
                t[t.len() / 2]
            );
        }
    }
}
