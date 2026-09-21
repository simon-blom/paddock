use super::affine;
use crate::device::MetalDevice;
use paddock_models::safetensors::{SafetensorsFile, ShardedSafetensors};
use std::path::Path;

fn poison_output(y: &crate::device::Buffer, count: usize) {
    // Previous command completed. An unwritten candidate element must not
    // inherit a passing value from the baseline it is compared against.
    unsafe { y.write_u32(&vec![f32::NAN.to_bits(); count + 32]) };
}

fn grouped_matrix_case(
    d: &MetalDevice,
    w: &crate::weights::Weight,
    x: &crate::device::Buffer,
    ids: &crate::device::Buffer,
    y: &crate::device::Buffer,
    f: &SafetensorsFile,
    base: &str,
    rows: usize,
    count: usize,
    per_entry: bool,
) {
    let k = w.k;
    let n = w.n / 512;
    let expected = f
        .bytes(if per_entry {
            "grouped_entry"
        } else {
            "grouped"
        })
        .unwrap()
        .1;
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mv",
        &[&w.buffer, x, ids, y],
        &[
            k as u32,
            n as u32,
            (rows * 10) as u32,
            u32::from(per_entry),
            512,
        ],
        [n.div_ceil(16), rows * 10, 1],
        128,
    );
    let baseline_seconds = cmd.finish().unwrap();
    let vector = unsafe { y.read_f32(0, count) };
    poison_output(y, count);
    let entries = rows * 10;
    let lists = d.alloc(512 * entries * 4).unwrap();
    let counts = d.alloc(512 * 4).unwrap();
    let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "moe_align",
        &[ids, &lists, &counts],
        &[entries as u32],
        [512, 1, 1],
        256,
    );
    cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
    let mut params = vec![
        k as u32,
        n as u32,
        entries as u32,
        u32::from(per_entry),
        512,
    ];
    params.extend([u32::MAX; 32]);
    cmd.dispatch(
        "q4a_expert_mm",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(32), entries.div_ceil(32) + 512, 1],
        128,
    );
    let seconds = cmd.finish().unwrap();
    let matrix = unsafe { y.read_f32(0, count + 32) };
    assert!(matrix[count..].iter().all(|v| v.is_nan()));
    poison_output(y, count);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mm_wide",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
        128,
    );
    let wide_seconds = cmd.finish().unwrap();
    let wide = unsafe { y.read_f32(0, count + 32) };
    assert!(wide[count..].iter().all(|v| v.is_nan()));
    assert!(
        matrix[..count] == wide[..count],
        "wide expert matrix changed arithmetic"
    );
    eprintln!(
        "AFFINE_GROUPED_WIDE {base} rows={rows} per_entry={per_entry} matrix_s={seconds} wide_s={wide_seconds}"
    );
    for row in (0..rows).step_by(7) {
        params[5 + row / 32] &= !(1 << (row % 32));
    }
    poison_output(y, count);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mm_wide",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
        128,
    );
    cmd.dispatch(
        "q4a_expert_vector_masked",
        &[&w.buffer, x, ids, y],
        &params,
        [n.div_ceil(16), entries, 1],
        128,
    );
    cmd.finish().unwrap();
    let mixed = unsafe { y.read_f32(0, count + 32) };
    assert!(mixed[count..].iter().all(|v| v.is_nan()));
    for row in 0..rows {
        let expected = if row % 7 == 0 { &vector } else { &matrix };
        let span = row * 10 * n..(row + 1) * 10 * n;
        assert!(
            mixed[span.clone()] == expected[span],
            "mixed expert contract changed row {row}"
        );
    }
    let mut max_error = 0f32;
    let mut unequal = 0;
    let mut peak = 0f32;
    for (&a, b) in matrix[..count].iter().zip(
        expected
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b)),
    ) {
        assert!(a.is_finite() && b.is_finite());
        max_error = max_error.max((a - b).abs());
        peak = peak.max(b.abs());
        unequal += usize::from(a != b);
    }
    eprintln!(
        "AFFINE_GROUPED {base} rows={rows} baseline_s={baseline_seconds} matrix_s={seconds} error={max_error} peak={peak} unequal={unequal}/{count}"
    );
    assert_eq!(
        unequal, 0,
        "same-checkpoint grouped expert operation mismatch"
    );
}

#[test]
#[ignore = "same-checkpoint MLX GPU fixtures: PADDOCK_FLASH_NEXT_MLX_AFFINE_REFERENCE"]
fn flash_next_affine_matches_mlx_gpu() {
    let dir = std::env::var("PADDOCK_FLASH_NEXT_MLX_AFFINE_REFERENCE").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let scratch = d.alloc(affine::WORKSPACE_BYTES).unwrap();
    let mut failures = Vec::new();
    for c in manifest["cases"].as_array().unwrap() {
        let k = c["k"].as_u64().unwrap() as usize;
        let n = c["n"].as_u64().unwrap() as usize;
        let rows = c["rows"].as_u64().unwrap() as usize;
        let experts = c["expert"].as_bool().unwrap();
        let shape = if experts { vec![512, n, k] } else { vec![n, k] };
        let ty = if c["bits"] == 8 {
            affine::A8G64
        } else {
            affine::A4G32
        };
        let base = c["base"].as_str().unwrap();
        eprintln!(
            "AFFINE_CASE file={} input={}",
            c["file"],
            c.get("input").and_then(|v| v.as_str()).unwrap_or("sine")
        );
        let w = affine::load(&d, &source, base, &shape, ty).unwrap();
        let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
        let x = d.upload(f.bytes("x").unwrap().1).unwrap();
        let count = rows * n * if experts { 10 } else { 1 };
        let y = d
            .upload(
                &(0..count + 32)
                    .flat_map(|_| f32::NAN.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let ids = if experts {
            Some(d.upload(f.bytes("ids").unwrap().1).unwrap())
        } else {
            None
        };
        if experts && rows > 128 {
            grouped_matrix_case(
                &d,
                &w,
                &x,
                ids.as_ref().unwrap(),
                &y,
                &f,
                base,
                rows,
                count,
                false,
            );
            if let Some((_, raw)) = f.bytes("x_entry") {
                let input = d.upload(raw).unwrap();
                grouped_matrix_case(
                    &d,
                    &w,
                    &input,
                    ids.as_ref().unwrap(),
                    &y,
                    &f,
                    base,
                    rows,
                    count,
                    true,
                );
            }
            continue;
        }
        let cmd = d.begin().unwrap();
        if let Some(ids) = &ids {
            cmd.dispatch(
                "q4a_expert_mv",
                &[&w.buffer, &x, ids, &y],
                &[k as u32, n as u32, (rows * 10) as u32, 0, 512],
                [n.div_ceil(16), rows * 10, 1],
                128,
            );
        } else if rows >= 13 && n > 48 {
            // Keep the original scalar-staged matrix as an independent
            // exactness seam for split-K and rejected tile candidates.
            let group = c["group"].as_u64().unwrap() as usize;
            let mut parts = (512 / (n.div_ceil(32) * rows.div_ceil(32)))
                .min(k / group.max(32))
                .max(1);
            while !k.is_multiple_of(parts * group.max(32)) {
                parts -= 1;
            }
            cmd.dispatch(
                "q4a_mm",
                &[&w.buffer, &x, &y],
                &[
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    group as u32,
                    parts as u32,
                    0,
                ],
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
        } else {
            affine::project(&cmd, &w, &x, &y, rows);
        }
        let baseline_seconds = cmd.finish().unwrap();
        let actual = unsafe { y.read_f32(0, count + 32) };
        eprintln!("AFFINE_TIME {base} rows={rows} experts={experts} gpu_s={baseline_seconds}");
        if let Some(ids) = &ids {
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            affine::experts(&cmd, &w, &x, ids, &y, rows * 10, false);
            let specialized_seconds = cmd.finish().unwrap();
            let specialized = unsafe { y.read_f32(0, count) };
            assert_eq!(
                &actual[..count],
                &specialized,
                "expert step specialization changed arithmetic {base} rows={rows}"
            );
            eprintln!(
                "AFFINE_EXPERT_ENTRY {base} rows={rows} baseline_s={baseline_seconds} specialized_s={specialized_seconds}"
            );
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_expert_order",
                &[ids, &scratch],
                &[(rows * 10) as u32],
                [1, 1, 1],
                256,
            );
            affine::experts_ordered(&cmd, &w, &x, ids, &y, rows * 10, false, Some(&scratch));
            let ordered_seconds = cmd.finish().unwrap();
            let ordered = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &ordered[..count],
                "expert ordering changed arithmetic {base} rows={rows}"
            );
            assert!(ordered[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_ORDER {base} rows={rows} baseline_s={baseline_seconds} ordered_s={ordered_seconds}"
            );
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if k.is_multiple_of(512) {
                    "q4a_expert4_fast_pair"
                } else {
                    "q4a_expert4_pair"
                },
                &[&w.buffer, &x, ids, &y, &scratch],
                &[k as u32, n as u32, (rows * 10) as u32, 0, 512],
                [n.div_ceil(16) * 4, (rows * 10).div_ceil(8), 1],
                128,
            );
            let pair_seconds = cmd.finish().unwrap();
            let paired = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &paired[..count],
                "expert pair changed arithmetic {base} rows={rows}"
            );
            assert!(paired[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_PAIR {base} rows={rows} baseline_s={baseline_seconds} ordered_s={ordered_seconds} pair_s={pair_seconds}"
            );
        }
        if !experts && rows == 4 {
            let cmd = d.begin().unwrap().with_independent_rows(true);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let batched = unsafe { y.read_f32(0, count) };
            for row in 0..rows {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[row * k * 4..(row + 1) * k * 4])
                    .unwrap();
                let out = d.alloc(n * 4).unwrap();
                let cmd = d.begin().unwrap();
                affine::project(&cmd, &w, &input, &out, 1);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, n) },
                    batched[row * n..(row + 1) * n],
                    "independent rows changed singleton contraction {base} row={row}"
                );
            }
        }
        if !experts && [4, 128].contains(&rows) {
            let spans = if rows == 4 {
                vec![(0, 1, 1), (1, 3, 3)]
            } else {
                vec![(0, 1, 1), (1, 33, 33), (34, 94, 94)]
            };
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(&spans);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let mixed = unsafe { y.read_f32(0, count + 32) };
            assert!(mixed[count..].iter().all(|v| v.is_nan()));
            for &(start, len, _) in &spans {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[start * k * 4..(start + len) * k * 4])
                    .unwrap();
                let out = d.alloc(len * n * 4).unwrap();
                let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
                affine::project(&cmd, &w, &input, &out, len);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, len * n) },
                    mixed[start * n..(start + len) * n],
                    "ragged projection changed per-sequence contraction {base} span={start}/{len}"
                );
            }
        }
        if !experts && rows >= 13 && n > 48 {
            poison_output(&y, count);
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            affine::project(&cmd, &w, &x, &y, rows);
            let split_seconds = cmd.finish().unwrap();
            let parallel = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &parallel[..count],
                "parallel split-K changed native BF16 arithmetic: {base} rows={rows}"
            );
            assert!(parallel[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_SPLIT {base} rows={rows} baseline_s={baseline_seconds} parallel_s={split_seconds}"
            );
        }
        if !experts && rows >= 812 && n > 48 {
            // All checkpoint planes here have a single K partition. Retain
            // the reordered packed kernel as a GPU cache-ordering ablation.
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if c["bits"] == 4 {
                    "q4a_mm4_reuse"
                } else {
                    "q4a_mm8_reuse"
                },
                &[&w.buffer, &x, &y],
                &[
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    c["group"].as_u64().unwrap() as u32,
                    1,
                    0,
                ],
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
            let packed_seconds = cmd.finish().unwrap();
            let packed = unsafe { y.read_f32(0, count + 32) };
            assert!(
                actual[..count] == packed[..count],
                "packed dense changed {base}"
            );
            assert!(packed[count..].iter().all(|v| v.is_nan()));
            eprintln!("AFFINE_REUSE {base} rows={rows} gpu_s={packed_seconds}");
        }
        if !experts && [4, 128].contains(&rows) {
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let whole = unsafe { y.read_f32(0, count) };
            let slices = if rows == 4 {
                vec![(0, 1), (1, 3)]
            } else {
                vec![(0, 1), (1, 7), (8, 23), (31, 33), (64, 63), (127, 1)]
            };
            for (start, len) in slices {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[start * k * 4..(start + len) * k * 4])
                    .unwrap();
                let out = d.alloc(len * n * 4).unwrap();
                let spans = [(0, len, rows)];
                let cmd = d
                    .begin()
                    .unwrap()
                    .with_projection_workspace(&scratch)
                    .with_projection_rows(&spans);
                affine::project(&cmd, &w, &input, &out, len);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, len * n) },
                    whole[start * n..(start + len) * n],
                    "logical projection changed when sliced: {base} span={start}/{len}/{rows}"
                );
            }
        }
        assert!(
            actual[count..].iter().all(|v| v.is_nan()),
            "output guard {base}"
        );
        let expected = f
            .bytes("y")
            .unwrap()
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b));
        let mut error = 0f32;
        let mut peak = 0f32;
        let mut unequal = 0;
        for (&a, b) in actual[..count].iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(a.to_bits() & 0xffff, 0);
            error = error.max((a - b).abs());
            peak = peak.max(b.abs());
            unequal += usize::from(a != b);
        }
        eprintln!("AFFINE {base} m={rows} error={error} peak={peak} unequal={unequal}/{count}");
        // An operation-format bound is not a generation-parity claim. Tiny
        // row-invariant projections must match exactly; all model outputs
        // still need their separate greedy/logit qualification.
        if ((n <= 48 || rows == 1 || experts) && unequal != 0) || error > peak * 0.008 + 0.0001 {
            failures.push(format!(
                "{base} rows={rows} error={error} unequal={unequal}"
            ));
        }
    }
    let c = &manifest["gather"];
    let base = c["base"].as_str().unwrap();
    let (info, _) = source.bytes(&format!("{base}.weight")).unwrap();
    let w = affine::load(&d, &source, base, &[info.shape[0], 160], affine::A4G32).unwrap();
    let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
    let ids = d.upload(f.bytes("ids").unwrap().1).unwrap();
    let y = d.alloc(7 * 160 * 4).unwrap();
    let cmd = d.begin().unwrap();
    affine::gather(&cmd, &w, &ids, &y, 7);
    cmd.finish().unwrap();
    let actual = unsafe { y.read_f32(0, 7 * 160) };
    let expected = f
        .bytes("y")
        .unwrap()
        .1
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "same-checkpoint PLE shard gather");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn flash_next_mlx_expert_order_is_stable_bounded_permutation() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    for entries in [1, 10, 40, 64, 90, 330, 1280] {
        for invalid in [false, true] {
            let ids = (0..entries)
                .map(|i| {
                    if invalid && i % 13 == 0 {
                        512
                    } else {
                        (i * 73 + 511) % 512
                    }
                })
                .collect::<Vec<u32>>();
            let input = d
                .upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let out = d.upload(&vec![0xA5; (entries as usize + 16) * 4]).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_expert_order",
                &[&input, &out],
                &[entries],
                [1, 1, 1],
                256,
            );
            cmd.finish().unwrap();
            let actual = unsafe { out.read_u32(entries as usize + 16) };
            // Control-index verification only; no host model arithmetic.
            let mut expected = (0..entries)
                .filter(|&i| ids[i as usize] < 512)
                .collect::<Vec<_>>();
            expected.sort_by_key(|&i| (ids[i as usize], i));
            expected.resize(entries as usize, u32::MAX);
            assert_eq!(&actual[..entries as usize], expected);
            assert!(actual[entries as usize..].iter().all(|&v| v == 0xA5A5A5A5));
        }
    }
}
