use super::*;

#[test]
#[ignore = "real Muse tower projection election; not a serving comparison"]
fn muse_tower_projection_tiles() {
    let path = std::env::var("PADDOCK_MUSE_MMPROJ").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(1 << 30)).unwrap();
    for (name, k, n) in [
        ("v.blk.0.attn_q.weight", 1536usize, 1536usize),
        ("v.blk.0.ffn_up.weight", 1536, 8960),
        ("v.blk.0.ffn_down.weight", 8960, 1536),
        ("mm.0.weight", 6144, 4096),
    ] {
        let w = Weight::load(&d, &map, name, &[k, n]).unwrap();
        assert_eq!(w.ty, 30);
        for rows in [64usize, 400, 1024, 3136, 5476] {
            let x = d
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let outputs = (0..2)
                .map(|_| d.alloc(rows * n * 4).unwrap())
                .collect::<Vec<_>>();
            let candidates = [("mv_bmm64", 64), ("mv_bmm128", 128)];
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for offset in 0..2 {
                    let i = (round + offset) % 2;
                    let (name, bm) = candidates[i];
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        name,
                        &[&w.buffer, &x, &outputs[i], &w.buffer],
                        &[k as u32, n as u32, rows as u32, 0],
                        [n.div_ceil(64), rows.div_ceil(bm), 1],
                        128,
                    );
                    let ms = cmd.finish().unwrap() * 1000.;
                    if round > 0 {
                        times[i].push(ms);
                    }
                }
            }
            let a = unsafe { outputs[0].read_f32(0, rows * n) };
            for at in 1..2 {
                let b = unsafe { outputs[at].read_f32(0, rows * n) };
                let max = a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                for t in &mut times {
                    t.sort_by(f64::total_cmp);
                }
                eprintln!(
                    "MUSE_TOWER_PROJECTION {name} rows={rows} bm64_ms={} candidate={} candidate_ms={} max={max}",
                    times[0][4], candidates[at].0, times[at][4]
                );
                assert!(
                    a == b && b.iter().all(|v| v.is_finite()),
                    "tower tile changed arithmetic"
                );
            }
        }
    }
}

#[test]
fn muse_fused_erf_projection_matches_separate_gpu_epilogue() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let (k, n) = (64usize, 97usize);
    for m in [17usize, 1025, 5476] {
        let w = d
            .upload(
                &(0..k * n)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 7) % 31) as f32 / 64. - 0.25).to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let x = d
            .upload(
                &(0..m * k)
                    .flat_map(|i| (((i * 11) % 73) as f32 / 16. - 2.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let bias = d
            .upload(
                &(0..n)
                    .flat_map(|i| (i as f32 / 97. - 0.5).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let a = d.alloc(m * n * 4).unwrap();
        let b = d
            .upload(
                &(0..(m + 128) * n)
                    .flat_map(|_| f32::NAN.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for has_bias in [false, true] {
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "vis_bmm_fast64",
                &[&w, &x, &a, &bias],
                &[k as u32, n as u32, m as u32, u32::from(has_bias)],
                [n.div_ceil(64), m.div_ceil(64), 1],
                128,
            );
            cmd.dispatch(
                "mv_gelu",
                &[&a],
                &[(m * n) as u32],
                [(m * n).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "mv_bmm128",
                &[&w, &x, &b, &bias],
                &[k as u32, n as u32, m as u32, if has_bias { 3 } else { 4 }],
                [n.div_ceil(64), m.div_ceil(128), 1],
                128,
            );
            cmd.finish().unwrap();
            let a = unsafe { a.read_f32(0, m * n) };
            let b = unsafe { b.read_f32(0, m * n) };
            assert!(b.iter().all(|v| v.is_finite()));
            assert!(a == b, "fused projection changed erf-GELU arithmetic");
        }
        assert!(
            unsafe { b.read_f32(m * n, 128 * n) }
                .iter()
                .all(|v| v.is_nan())
        );
    }
}

#[test]
#[ignore = "requires canonical Muse BF16 mmproj; invoked by greedy-parity.py"]
fn muse_vision_white_tensor_capture() {
    let path = std::env::var("PADDOCK_MUSE_MMPROJ").expect("mmproj");
    let d = MetalDevice::new(None).unwrap();
    let v = Vision::load(&d, Path::new(&path), 6656).unwrap();
    let pixels = vec![255; 112 * 112 * 3];
    let mut job = v.start(&d, &[(&pixels, 112, 112)]).unwrap();
    assert_eq!(job.rows, 64);
    let mut captures = serde_json::Map::new();
    let mut capture = |name: String, buffer: &Buffer| {
        // Edge samples use the external debug tool's order. Tensor arithmetic
        // remains GPU-only; this reads completed results for comparison.
        let values = unsafe { buffer.read_f32(0, 64 * E) };
        let mut samples = Vec::new();
        for r in [0, 1, 2, 61, 62, 63] {
            for c in [0, 1, 2, E - 3, E - 2, E - 1] {
                samples.push((values[r * E + c] * 10000.).round() / 10000.);
            }
        }
        captures.insert(
            name,
            serde_json::json!({"dims":[E,64,1,1],"samples":samples}),
        );
    };
    capture("pre_ln".into(), &job.x);
    // Replay only the first block's front into its disposable scratch. The
    // residual is untouched, and step() below recomputes these operations.
    // This localizes the first error without another tower implementation.
    let cmd = d.begin().unwrap();
    v.norm(&cmd, &job.x, &v.blocks[0].ln1, &job.stage, 64);
    cmd.finish().unwrap();
    capture("layer_inp_normed-0".into(), &job.stage);
    let cmd = d.begin().unwrap();
    for (w, b, out) in [
        (&v.blocks[0].q, &v.blocks[0].qb, &job.q),
        (&v.blocks[0].k, &v.blocks[0].kb, &job.k),
        (&v.blocks[0].v, &v.blocks[0].vb, &job.v),
    ] {
        Vision::mm(&cmd, w, &job.stage, out, Some(b), 64, 0);
    }
    cmd.dispatch(
        "mv_qkv",
        &[&job.q, &job.k, &job.v, &job.xy, &job.qh, &job.kh, &job.vh],
        &[64],
        [(64 * E).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "mv_attention",
        &[&job.qh, &job.kh, &job.vh, &job.attn, &job.local],
        &[0],
        [16, job.local_count, 1],
        64,
    );
    cmd.finish().unwrap();
    capture("kqv_out-0".into(), &job.attn);
    let mut projections = serde_json::Map::new();
    for (name, buffer) in [("Qcur-0", &job.q), ("Kcur-0", &job.k), ("Vcur-0", &job.v)] {
        let values = unsafe { buffer.read_f32(0, 64 * E) };
        let mut samples = Vec::new();
        for row in [0, 1, 2, 61, 62, 63] {
            for head in [0, 1, 2, 13, 14, 15] {
                for channel in [0, 1, 2, 93, 94, 95] {
                    samples.push((values[row * E + head * 96 + channel] * 10000.).round() / 10000.);
                }
            }
        }
        projections.insert(
            name.into(),
            serde_json::json!({"dims":[96,16,64,1],"samples":samples}),
        );
    }
    let outputs = loop {
        let out = v.step(&d, &mut job, Duration::ZERO).unwrap();
        capture(format!("layer_out-{}", job.layer - 1), &job.x);
        if let Some(out) = out {
            break out;
        }
    };
    captures.extend(projections);
    eprintln!("VISION_TENSORS {}", serde_json::Value::Object(captures));
    assert_eq!(outputs[0].tokens, 16);
    let values = unsafe { outputs[0].embd.read_f32(0, 16 * 6656) };
    assert!(values.iter().all(|v| v.is_finite()));
    if let Some(path) = paddock_models::dev_var_os!("PADDOCK_METAL_EMBEDDING_CAPTURE") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&16i32.to_le_bytes()).unwrap();
        file.write_all(&6656i32.to_le_bytes()).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
}

#[test]
fn muse_erf_gelu_and_ragged_window_attention_are_finite_and_isolated() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    let values = [-10f32, -3., -1., 0., 1., 3., 10.];
    let x = d
        .upload(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch("mv_gelu", &[&x], &[7], [1, 1, 1], 256);
    cmd.finish().unwrap();
    let y = unsafe { x.read_f32(0, 7) };
    for (got, want) in y
        .iter()
        .zip([0., -0.004049694, -0.15865526, 0., 0.8413447, 2.9959502, 10.])
    {
        assert!((got - want).abs() < 0.000002, "GELU {got} versus {want}");
    }
    let grid: Vec<_> = (0..8193)
        .flat_map(|i| (i as f32 / 256. - 16.).to_le_bytes())
        .collect();
    let fast = d.upload(&grid).unwrap();
    let series = d.upload(&grid).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "mv_gelu",
        &[&fast],
        &[8193],
        [8193usize.div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "mv_gelu_series_check",
        &[&series],
        &[8193],
        [8193usize.div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    let fast = unsafe { fast.read_f32(0, 8193) };
    let series = unsafe { series.read_f32(0, 8193) };
    let max = fast
        .iter()
        .zip(&series)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let worst = fast
        .iter()
        .zip(&series)
        .enumerate()
        .max_by(|(_, (a, b)), (_, (c, d))| (*a - *b).abs().total_cmp(&(*c - *d).abs()))
        .unwrap();
    eprintln!(
        "ERF_GRID x={} fast={} series={}",
        worst.0 as f32 / 256. - 16.,
        worst.1.0,
        worst.1.1
    );
    // This compares two *F32 implementations*, not an exact erf oracle.
    // The positive-term series loses up to ~4e-6 at saturation (x=5.4414),
    // whereas the constant-time approximation correctly rounds to x. Keep
    // that measured check error separate from A&S's real-arithmetic bound.
    assert!(
        fast.iter().all(|v| v.is_finite()) && max < 0.000005,
        "erf approximations differ: {max}"
    );
    for (i, value) in fast.iter().enumerate() {
        let x = i as f32 / 256. - 16.;
        if x.abs() >= 6. {
            assert_eq!(*value, if x > 0. { x } else { 0. });
        }
    }
    let domains = [17usize, 65, 3];
    let rows = domains.iter().sum::<usize>();
    let halfs = |mult| {
        (0..rows * E)
            .flat_map(|i| {
                half::f16::from_f32(((i * mult + i / 96) % 97) as f32 / 32. - 1.5).to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let q = d.upload(&halfs(3)).unwrap();
    let k = d.upload(&halfs(7)).unwrap();
    let v = d.upload(&halfs(11)).unwrap();
    let out = d.alloc(rows * E * 4).unwrap();
    let check = d.alloc(rows * E * 4).unwrap();
    let mut tiles = Vec::new();
    let mut bounds = Vec::new();
    let mut first = 0;
    for count in domains {
        for at in (0..count).step_by(32) {
            tiles.extend([
                (first + at) as u32,
                (count - at).min(32) as u32,
                first as u32,
                count as u32,
            ]);
        }
        for _ in 0..count {
            bounds.extend([first as u32, (first + count) as u32]);
        }
        first += count;
    }
    let count = tiles.len() / 4;
    let tiles = upload(&d, &tiles).unwrap();
    let bounds = upload(&d, &bounds).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "mv_attention",
        &[&q, &k, &v, &out, &tiles],
        &[0],
        [16, count, 1],
        64,
    );
    cmd.dispatch(
        "mv_attention_check",
        &[&q, &k, &v, &check, &bounds],
        &[0],
        [16, rows, 1],
        32,
    );
    cmd.finish().unwrap();
    let a = unsafe { out.read_f32(0, rows * E) };
    let b = unsafe { check.read_f32(0, rows * E) };
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        a.iter().all(|v| v.is_finite()) && max < 0.00002,
        "attention max={max}"
    );
}

#[test]
fn muse_resize_respects_full_checkpoint_budget_and_geometry() {
    // Equally exact square aspect ratios elect the larger grid, matching
    // the published processor (also checked by reference prompt counts).
    assert_eq!(resize(256, 256).unwrap(), (280, 280));
    assert_eq!(resize(768, 768).unwrap(), (784, 784));
    assert_eq!(resize(1792, 1792).unwrap(), (1792, 1792));
    for (w, h) in [
        (1, 1),
        (127, 317),
        (320, 200),
        (768, 768),
        (1920, 1080),
        (4096, 4096),
    ] {
        let (x, y) = resize(w, h).unwrap();
        assert_eq!((x % 28, y % 28), (0, 0));
        assert!(x * y <= 4096 * 784);
    }
    for (w, h) in [(0, 1), (1, 0), (65537, 1), (1, 65537)] {
        assert!(resize(w, h).is_err());
    }
}

#[test]
#[ignore = "requires canonical Muse Glimmer BF16 mmproj"]
fn muse_tower_finite_ragged_and_cooperative_yields() {
    let path = std::env::var("PADDOCK_MUSE_MMPROJ").expect("mmproj");
    let d = MetalDevice::new(None).unwrap();
    let map = MappedGguf::open(Path::new(&path)).unwrap();
    for (key, value) in &map.gguf().metadata {
        if key.starts_with("clip.") {
            eprintln!("{key}={value:?}");
        }
    }
    let v = Vision::load(&d, Path::new(&path), 6656).unwrap();
    let images: Vec<_> = [(1, 1), (112, 112), (127, 317), (476, 504)]
        .into_iter()
        .map(|(w, h)| {
            (
                (0..w * h * 3)
                    .map(|i| ((i * 17 + i / 31) % 251) as u8)
                    .collect::<Vec<_>>(),
                w,
                h,
            )
        })
        .collect();
    let encode = |inputs: &[(&[u8], usize, usize)], budget| {
        let mut job = v.start(&d, inputs).unwrap();
        let mut yields = 0;
        loop {
            if let Some(out) = v.step(&d, &mut job, budget).unwrap() {
                break (out, yields);
            }
            yields += 1;
            assert!(yields < 50);
        }
    };
    let (batch, yields) = encode(
        &images
            .iter()
            .map(|(rgb, w, h)| (&rgb[..], *w, *h))
            .collect::<Vec<_>>(),
        Duration::ZERO,
    );
    assert_eq!(yields, 49);
    for ((rgb, w, h), batch) in images.iter().zip(batch) {
        let (single, _) = encode(&[(rgb, *w, *h)], Duration::from_secs(60));
        assert_eq!(batch.tokens, single[0].tokens);
        let a = unsafe { batch.embd.read_f32(0, batch.tokens * 6656) };
        let b = unsafe { single[0].embd.read_f32(0, batch.tokens * 6656) };
        assert!(a.iter().all(|v| v.is_finite()));
        let max = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "MUSE_TOWER {w}x{h} tokens={} ragged_max={max}",
            batch.tokens
        );
        assert!(max < 0.002, "ragged encoder disagreement {max}");
    }
}

#[test]
#[ignore = "requires canonical Muse BF16 mmproj; full 4096-soft-token allocation/compute stress"]
fn muse_tower_full_checkpoint_image_budget() {
    let path = std::env::var("PADDOCK_MUSE_MMPROJ").expect("mmproj");
    let d = MetalDevice::new(None).unwrap();
    let v = Vision::load(&d, Path::new(&path), 6656).unwrap();
    let baseline = d.allocated_bytes();
    let white = vec![255; 1792 * 1792 * 3];
    assert!(
        v.start(&d, &[(&white, 1792, 1792), (&white, 1792, 1792)])
            .is_err(),
        "oversized encoder wave was not bounded"
    );
    let mut job = v.start(&d, &[(&white, 1792, 1792)]).unwrap();
    assert_eq!(job.rows, MAX_PATCHES);
    let output = loop {
        if let Some(out) = v.step(&d, &mut job, Duration::ZERO).unwrap() {
            break out;
        }
    };
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].tokens, 4096);
    assert!(
        unsafe { output[0].embd.read_f32(0, 4096 * 6656) }
            .iter()
            .all(|v| v.is_finite())
    );
    eprintln!(
        "MUSE_FULL_BUDGET gpu_seconds={} peak_live_bytes={}",
        job.gpu_seconds,
        d.allocated_bytes()
    );
    drop(output);
    drop(job);
    assert_eq!(
        d.allocated_bytes(),
        baseline,
        "completed maximum-budget job leaked device scratch"
    );
}
