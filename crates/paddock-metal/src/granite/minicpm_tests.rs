use super::*;
use paddock_models::safetensors::SafetensorsFile;

thread_local! {
    // Side-by-side regression/performance tests, never a serving switch.
    pub(super) static BASELINE_ATTENTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_PROJECTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_PREFILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn top(row: &[f32]) -> u32 {
    assert!(row.iter().all(|v| v.is_finite()));
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32
}

#[test]
fn minicpm_greedy_ties_match_reference_and_serving_sampler() {
    assert_eq!(top(&[1.0, 3.0, 3.0, 2.0]), 1);
    assert_eq!(top(&[0.0; 16]), 0);
}

#[test]
fn minicpm_direct_prefill_handles_fragmented_pages_and_ragged_queries() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let rows = 65;
    let stride = 520;
    let q = d
        .upload(
            &(0..rows * 2048)
                .flat_map(|i| (((i * 17 % 113) as f32 - 56.0) / 64.0).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let query = d.alloc((rows + 32) * 2048 * 2).unwrap();
    let kv = |mult| {
        d.upload(
            &(0..stride * 2 * 16 * 256)
                .flat_map(|i| {
                    half::f16::from_f32(((i * mult + i / 128) % 97) as f32 / 64.0 - 0.75)
                        .to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let k = kv(3);
    let v = kv(11);
    let tiles = upload(&[0, 32, 32, 1, 33, 32]);
    let out = d.alloc((rows * 2048 + 16) * 4).unwrap();
    let expected = d.alloc(out.len()).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "attention_query",
        &[&q, &query],
        &[2048, 0, rows as u32],
        [((rows + 32) * 2048).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    for past in [0, 17, 1024, 8192] {
        let meta = upload(
            &(0..rows)
                .flat_map(|r| {
                    if r < 33 {
                        [1, (past + r) as u32]
                    } else {
                        [0, (past + r - 33) as u32]
                    }
                })
                .collect::<Vec<_>>(),
        );
        for fragmented in [false, true] {
            let pages = upload(
                &(0..2)
                    .flat_map(|s| {
                        (0..stride).map(move |p| {
                            if fragmented {
                                ((stride - 1 - p) * 2 + s) as u32
                            } else {
                                (s * stride + p) as u32
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
            );
            unsafe {
                out.write_u32(&vec![12345f32.to_bits(); rows * 2048 + 16]);
            }
            let cmd = d.begin().unwrap();
            let p = [16, 2, stride as u32, (1.0 / 128f32.sqrt()).to_bits()];
            cmd.dispatch(
                "granite_attention_prefill128",
                &[&query, &k, &v, &meta, &pages, &expected, &tiles],
                &p,
                [16, 3, 1],
                128,
            );
            cmd.dispatch(
                "llama_prefill_direct",
                &[&query, &k, &v, &meta, &pages, &out, &tiles],
                &p,
                [16, 3, 1],
                128,
            );
            cmd.finish().unwrap();
            let got = unsafe { out.read_f32(0, rows * 2048 + 16) };
            let want = unsafe { expected.read_f32(0, rows * 2048) };
            let error = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("MINICPM_DIRECT past={past} fragmented={fragmented} max={error}");
            assert!(
                got[..rows * 2048]
                    .iter()
                    .zip(&want)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() < 0.0002)
            );
            assert!(got[rows * 2048..].iter().all(|v| *v == 12345.0));
        }
    }
}

/// Fixed work, independent of EOS/greedy divergence. Use dispatch counters only
/// to attribute stages; counter instrumentation changes command submission.
#[test]
#[ignore = "requires PADDOCK_METAL_TEST_MODEL MiniCPM checkpoint"]
fn minicpm_fixed_work_decode() {
    let path = std::env::var("PADDOCK_METAL_TEST_MODEL").unwrap();
    let number = |key, fallback| {
        std::env::var(key)
            .ok()
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(fallback)
    };
    let batch = number("PADDOCK_MINICPM_BATCH", 4);
    let prompt = number("PADDOCK_MINICPM_PROMPT", 128);
    let steps = number("PADDOCK_MINICPM_STEPS", 64);
    BASELINE_ATTENTION.set(std::env::var_os("PADDOCK_MINICPM_BASELINE_ATTENTION").is_some());
    BASELINE_PROJECTION.set(std::env::var_os("PADDOCK_MINICPM_BASELINE_PROJECTION").is_some());
    BASELINE_PREFILL.set(std::env::var_os("PADDOCK_MINICPM_BASELINE_PREFILL").is_some());
    assert!(batch > 0 && prompt > 0 && steps > 0);
    let mut model = Granite::load(Path::new(&path), prompt + steps + 8, batch, None).unwrap();
    let mut times = Vec::new();
    for round in 0..3 {
        model.reset();
        for slot in 0..batch {
            // Distinct prefixes; don't accidentally measure prefix reuse.
            model
                .prefill_begin(slot, vec![1000 + (round * batch + slot) as u32; prompt])
                .unwrap();
        }
        let start = std::time::Instant::now();
        while !model.pending.is_empty() {
            model.forward_mixed(&[], 512).unwrap();
        }
        eprintln!(
            "MINICPM_PREFILL round={round} c={batch} prompt={prompt} ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        for step in 0..steps {
            let rows: Vec<_> = (0..batch)
                .map(|slot| (slot, 2000 + step as u32, (prompt + step) as u32))
                .collect();
            let start = std::time::Instant::now();
            let (logits, done) = model.forward_mixed(&rows, 512).unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(logits.len(), batch * model.vocab);
            assert!(done.is_empty());
            assert!(logits.iter().all(|v| v.is_finite()));
            if round > 0 {
                times.push(ms);
            }
        }
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    times.sort_by(f64::total_cmp);
    eprintln!(
        "MINICPM_FIXED batch={batch} prompt={prompt} steps={steps} mean_ms={mean:.4} median_ms={:.4} p99_ms={:.4} aggregate_tps={:.2}",
        times[times.len() / 2],
        times[(times.len() * 99 / 100).min(times.len() - 1)],
        batch as f64 * 1000.0 / mean
    );
}

#[test]
fn minicpm_gqa8_paged_attention_matches_independent_gpu_scan() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let rows = 5;
    let heads = 16;
    let stride = 514;
    let q = d
        .upload(
            &(0..rows * heads * 128)
                .flat_map(|i| (((i * 17 + i / 128) % 113) as f32 / 57.0 - 1.0).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let halfs = |mult| {
        (0..stride * 3 * 16 * 256)
            .flat_map(|i| {
                half::f16::from_f32(((i * mult + i / 128) % 97) as f32 / 31.0 - 1.5).to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let k = d.upload(&halfs(3)).unwrap();
    let v = d.upload(&halfs(11)).unwrap();
    let meta = upload(&[0, 0, 1, 16, 2, 137, 0, 2048, 1, 8192]);
    // Interleaved, reversed physical pages, not identity-contiguous storage.
    let pages = upload(
        &(0..3)
            .flat_map(|s| (0..stride).rev().map(move |p| (p * 3 + s) as u32))
            .collect::<Vec<_>>(),
    );
    let selected = [4u32, 0, 2, 1, 3];
    let indices = upload(&selected);
    let out = d.alloc(rows * heads * 128 * 4).unwrap();
    let expected = d.alloc(out.len()).unwrap();
    let parts = d.alloc(rows * heads * 32 * 130 * 4).unwrap();
    let scale = (1.0 / 128f32.sqrt()).to_bits();
    let cmd = d.begin().unwrap();
    for row in 0..rows {
        cmd.dispatch(
            "attention",
            &[&q, &k, &v, &meta, &pages, &expected],
            &[16, 2, 128, row as u32, stride as u32, scale, 1],
            [16, 1, 1],
            32,
        );
    }
    cmd.finish().unwrap();
    let want = unsafe { expected.read_f32(0, rows * heads * 128) };
    for count in [1, 3, 5] {
        for splits in [1, 2, 8, 32] {
            unsafe {
                out.write_u32(&vec![12345.0f32.to_bits(); rows * heads * 128]);
            }
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "llama_decode",
                &[&q, &k, &v, &meta, &pages, &indices, &parts],
                &[16, 2, stride as u32, 0, 0, splits as u32],
                [2, count, splits],
                128,
            );
            cmd.dispatch(
                "muse_merge",
                &[&parts, &out, &indices],
                &[16, splits as u32, 128],
                [16 * count, 1, 1],
                32,
            );
            cmd.finish().unwrap();
            let got = unsafe { out.read_f32(0, rows * heads * 128) };
            for row in 0..rows {
                let actual = &got[row * heads * 128..(row + 1) * heads * 128];
                if selected[..count].contains(&(row as u32)) {
                    let reference = &want[row * heads * 128..(row + 1) * heads * 128];
                    assert!(
                        actual
                            .iter()
                            .zip(reference)
                            .all(|(a, b)| a.is_finite() && (a - b).abs() < 2e-5),
                        "rows={count} splits={splits} row={row}"
                    );
                } else {
                    assert!(actual.iter().all(|v| *v == 12345.0));
                }
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_TEST_MODEL MiniCPM Q4 GGUF"]
fn minicpm_paired_projections_preserve_every_value() {
    let path = std::env::var("PADDOCK_METAL_TEST_MODEL").unwrap();
    let model = Granite::load(Path::new(&path), 128, 4, None).unwrap();
    assert!(!model.mlx && model.width == 2048 && model.ff == 6144);
    let d = &model.device;
    for rows in [3, 4] {
        for index in [0, 21, 41] {
            let l = &model.layers[index];
            let input = d
                .upload(
                    &(0..rows * model.ff)
                        .flat_map(|i| {
                            (((i * 17 + i / 128) % 113) as f32 / 57.0 - 1.0).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            for weights in [
                vec![&l.q, &l.k, &l.v],
                vec![&l.gate, &l.up],
                vec![&l.o],
                vec![&l.down],
                vec![model.head.as_ref().unwrap()],
            ] {
                assert!(weights.iter().all(|w| matches!(w.ty, 12..=14)));
                let expected: Vec<_> = weights
                    .iter()
                    .map(|w| d.alloc(rows * w.n * 4).unwrap())
                    .collect();
                let actual: Vec<_> = weights
                    .iter()
                    .map(|w| d.alloc(rows * w.n * 4).unwrap())
                    .collect();
                let cmd = d.begin().unwrap();
                for (w, out) in weights.iter().zip(&expected) {
                    w.linear(&cmd, &input, out, rows, 1.0, &model.scratch.gemm_input);
                }
                let planes: Vec<_> = weights.iter().copied().zip(&actual).collect();
                model.project(&cmd, &planes, &input, rows, 1.0);
                cmd.finish().unwrap();
                for ((w, expected), actual) in weights.iter().zip(&expected).zip(&actual) {
                    let count = rows * w.n;
                    let want = unsafe { expected.read_f32(0, count) };
                    let got = unsafe { actual.read_f32(0, count) };
                    assert!(got.iter().all(|v| v.is_finite()));
                    assert_eq!(
                        got, want,
                        "layer={index} rows={rows} k={} n={} type={}",
                        w.k, w.n, w.ty
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_TEST_MODEL MiniCPM Q4 GGUF"]
fn minicpm_prefill_expansion_diagnostic() {
    let path = std::env::var("PADDOCK_METAL_TEST_MODEL").unwrap();
    let model = Granite::load(Path::new(&path), 1024, 4, None).unwrap();
    let d = &model.device;
    for rows in [128usize, 512] {
        for (name, w) in [
            ("q", &model.layers[0].q),
            ("k", &model.layers[0].k),
            ("v", &model.layers[0].v),
            ("o", &model.layers[0].o),
            ("gate", &model.layers[0].gate),
            ("up", &model.layers[0].up),
            ("down", &model.layers[0].down),
        ] {
            let input = d
                .upload(
                    &(0..rows * w.k)
                        .flat_map(|i| (((i * 37 % 203) as f32 - 101.0) / 128.0).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let out = d.alloc(rows * w.n * 4).unwrap();
            let expected = d.alloc(rows * w.n * 4).unwrap();
            let p = [w.k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()];
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "linear_input_padded",
                &[&input, &model.scratch.gemm_input],
                &p,
                [(rows * w.k).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "linear_input_padded",
                &[&input, &model.scratch.attn_parts],
                &p,
                [(rows * w.k).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish().unwrap();
            let mut timings = [Vec::new(), Vec::new()];
            for round in 0..7 {
                for variant in [round % 2, 1 - round % 2] {
                    let cmd = d.begin().unwrap();
                    for _ in 0..8 {
                        if variant == 0 {
                            w.linear_prepared(
                                &cmd,
                                &model.scratch.gemm_input,
                                &expected,
                                rows,
                                1.0,
                            );
                        } else {
                            cmd.dispatch(
                                "linear_kexpand",
                                &[&w.buffer, &model.scratch.attn_parts],
                                &p,
                                [(w.k * w.n).div_ceil(1024), 1, 1],
                                256,
                            );
                            cmd.dispatch(
                                "linear_kexpanded128",
                                &[&model.scratch.attn_parts, &out],
                                &p,
                                [w.n.div_ceil(64), rows.div_ceil(128), 1],
                                128,
                            );
                        }
                    }
                    let us = cmd.finish().unwrap() * 1e6 / 8.0;
                    if round > 0 {
                        timings[variant].push(us);
                    }
                }
            }
            let actual = unsafe { out.read_f32(0, rows * w.n) };
            let want = unsafe { expected.read_f32(0, rows * w.n) };
            let error = actual
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let neq = actual.iter().zip(&want).filter(|(a, b)| a != b).count();
            for t in &mut timings {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "MINICPM_TILE name={name} rows={rows} type={} old_us={:.2} new_us={:.2} neq={neq} max={error}",
                w.ty, timings[0][3], timings[1][3]
            );
            assert!(
                actual
                    .iter()
                    .zip(&want)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() < 0.002 + b.abs() * 0.0001)
            );
        }
    }
}

#[test]
#[ignore = "requires independent MiniCPM5 MLX operation fixtures"]
fn minicpm_mlx_decode_contracts() {
    let root = std::env::var("PADDOCK_MINICPM_REFERENCE").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let selected = upload(&[0]);
    let pages = upload(&(0..256).collect::<Vec<_>>());
    let parts = d.alloc(16 * 32 * 130 * 4).unwrap();
    let out = d.alloc(2048 * 4).unwrap();
    let mut failures = 0;
    for item in manifest["decode"]
        .as_array()
        .expect("--trace-decode fixtures")
    {
        let f = SafetensorsFile::open(&root.join(item["file"].as_str().unwrap())).unwrap();
        let q = d.upload(f.bytes("q").unwrap().1).unwrap();
        let k = d.upload(f.bytes("k").unwrap().1).unwrap();
        let v = d.upload(f.bytes("v").unwrap().1).unwrap();
        let meta = upload(&[0, item["length"].as_u64().unwrap() as u32 - 1]);
        let cmd = d.begin().unwrap();
        let splits = (item["length"].as_u64().unwrap() as usize)
            .div_ceil(128)
            .clamp(8, 32);
        cmd.dispatch(
            "llama_mlx_decode",
            &[&q, &k, &v, &meta, &pages, &selected, &parts],
            &[16, 2, 256, 0, 0, splits as u32],
            [2, 1, splits],
            128,
        );
        cmd.dispatch(
            "muse_merge",
            &[&parts, &out, &selected],
            &[16, splits as u32, 128],
            [16, 1, 1],
            32,
        );
        cmd.dispatch("gmlx_round", &[&out], &[2048], [8, 1, 1], 256);
        cmd.finish().unwrap();
        let got = unsafe { out.read_f32(0, 2048) };
        let want: Vec<_> = f
            .bytes("attention")
            .unwrap()
            .1
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let neq = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        let error = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(got.iter().all(|v| v.is_finite()));
        eprintln!(
            "MINICPM_DECODE layer={} different={neq}/2048 max={error}",
            item["layer"]
        );
        failures += usize::from(neq != 0);
    }
    assert_eq!(failures, 0, "independent decode attention differs");
}

#[test]
#[ignore = "requires independent MiniCPM5 MLX operation fixtures"]
fn minicpm_mlx_operation_contracts() {
    let root = std::env::var("PADDOCK_MINICPM_REFERENCE").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let model = Granite::load(
        Path::new(manifest["model"].as_str().unwrap()),
        1024,
        4,
        None,
    )
    .unwrap();
    let d = &model.device;
    let s = &model.scratch;
    let failures = std::cell::Cell::new(0);
    let mut fixtures: Vec<_> = [1usize, 4, 33, 128]
        .into_iter()
        .map(|rows| (format!("ops-{rows}.safetensors"), 0, rows, 19))
        .collect();
    if let Some(layers) = manifest["layers"].as_array() {
        fixtures.extend(layers.iter().map(|v| {
            (
                v["file"].as_str().unwrap().to_owned(),
                v["layer"].as_u64().unwrap() as usize,
                v["rows"].as_u64().unwrap() as usize,
                v["offset"].as_u64().unwrap() as usize,
            )
        }));
    }
    for (file, index, rows, offset) in fixtures {
        let layer = &model.layers[index];
        let f = SafetensorsFile::open(&root.join(file)).unwrap();
        let upload = |name| d.upload(f.bytes(name).unwrap().1).unwrap();
        let check = |name, out: &Buffer, count: usize| {
            let got = unsafe { out.read_f32(0, count) };
            let want: Vec<_> = f
                .bytes(name)
                .unwrap()
                .1
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let error = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let neq = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert!(got.iter().all(|v| v.is_finite()));
            let passed = if matches!(name, "embedding" | "norm" | "swiglu" | "qr") {
                got == want
            } else {
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| (a - b).abs() <= 2e-5 + b.abs() * 0.016)
                    && (name == "attention" || neq * 200 <= count)
            };
            if !passed {
                failures.set(failures.get() + 1);
            }
            eprintln!(
                "MINICPM_OP layer={index} rows={rows} op={name} max={error} different={neq}/{count} passed={passed}"
            );
        };
        let x = upload("x");
        unsafe {
            s.ids
                .write_u32(&(1000..1000 + rows as u32).collect::<Vec<_>>());
        }
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "mlx_embed",
            &[&model.embedding.buffer, &s.ids, &s.x],
            &[2048, rows as u32, 130560, 1f32.to_bits()],
            [(rows * 2048).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        check("embedding", &s.x, rows * 2048);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "mlx_rms",
            &[&x, &layer.norm.buffer, &s.norm],
            &[2048, 0, 1e-6f32.to_bits()],
            [rows, 1, 1],
            512,
        );
        cmd.finish().unwrap();
        check("norm", &s.norm, rows * 2048);
        let norm = upload("norm");
        let cmd = d.begin().unwrap();
        model.project(
            &cmd,
            &[(&layer.q, &s.q), (&layer.k, &s.k), (&layer.v, &s.v)],
            &norm,
            rows,
            1.0,
        );
        model.project(
            &cmd,
            &[(&layer.gate, &s.gate), (&layer.up, &s.up)],
            &norm,
            rows,
            1.0,
        );
        cmd.finish().unwrap();
        for (name, out, width) in [
            ("q", &s.q, 2048),
            ("k", &s.k, 256),
            ("v", &s.v, 256),
            ("gate", &s.gate, 6144),
            ("up", &s.up, 6144),
        ] {
            check(name, out, rows * width);
        }
        let gate = upload("gate");
        let up = upload("up");
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "mlx_swiglu",
            &[&gate, &up],
            &[(rows * 6144) as u32],
            [(rows * 6144).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        check("swiglu", &gate, rows * 6144);
        let q = upload("q");
        let k = upload("k");
        let v = upload("v");
        unsafe {
            s.meta.write_u32(
                &(0..rows)
                    .flat_map(|r| [0, (r + offset) as u32])
                    .collect::<Vec<_>>(),
            );
            s.pages.write_u32(&(0..64).collect::<Vec<_>>());
        }
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "llama_mlx_rope",
            &[&q, &k, &v, &layer.keys, &layer.values, &s.meta, &s.pages],
            &[2048, 256, 128, rows as u32, 64, 5000000f32.to_bits()],
            [(rows * 1152).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        check("qr", &q, rows * 2048);
        // Reuse reference Q/K/V and a zero-based physical cache to isolate
        // attention from affine/RoPE differences. No uninitialized prefix.
        let qr = upload("qr");
        let kr: Vec<_> = f
            .bytes("kr")
            .unwrap()
            .1
            .chunks_exact(4)
            .flat_map(|b| {
                half::bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap()))
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let vb: Vec<_> = f
            .bytes("v")
            .unwrap()
            .1
            .chunks_exact(4)
            .flat_map(|b| {
                half::bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap()))
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let keys = d.upload(&kr).unwrap();
        let values = d.upload(&vb).unwrap();
        unsafe {
            s.meta
                .write_u32(&(0..rows as u32).flat_map(|r| [0, r]).collect::<Vec<_>>());
            s.attention_tiles.write_u32(
                &(0..rows)
                    .step_by(32)
                    .flat_map(|r| [r as u32, (rows - r).min(32) as u32])
                    .collect::<Vec<_>>(),
            );
        }
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "llama_mlx_prefill",
            &[
                &qr,
                &keys,
                &values,
                &s.meta,
                &s.pages,
                &s.attn,
                &s.attention_tiles,
            ],
            &[16, 2, 64, 0, 0, 1],
            [16, rows.div_ceil(32), 1],
            128,
        );
        cmd.dispatch(
            "gmlx_round",
            &[&s.attn],
            &[(rows * 2048) as u32],
            [(rows * 2048).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        check("attention", &s.attn, rows * 2048);
        // Isolate downstream projections using the reference operands, not
        // cascading attention/SwiGLU errors into a misleading GEMM diagnosis.
        if f.bytes("o").is_some() {
            let attention = upload("attention");
            let activation = upload("swiglu");
            let cmd = d.begin().unwrap();
            model.project(&cmd, &[(&layer.o, &s.delta)], &attention, rows, 1.0);
            cmd.finish().unwrap();
            check("o", &s.delta, rows * 2048);
            let cmd = d.begin().unwrap();
            model.project(&cmd, &[(&layer.down, &s.delta)], &activation, rows, 1.0);
            cmd.finish().unwrap();
            check("down", &s.delta, rows * 2048);
        }
    }
    assert_eq!(failures.get(), 0, "independent operation contracts failed");
}

#[test]
#[ignore = "requires PADDOCK_METAL_TEST_MODEL MiniCPM GGUF or MLX"]
fn minicpm_cache_admission_and_memory_contract() {
    let path = std::env::var("PADDOCK_METAL_TEST_MODEL").unwrap();
    let path = Path::new(&path);
    assert!(Granite::load(path, 131073, 4, None).is_err());
    assert!(Granite::load(path, 1024, 513, None).is_err());
    assert!(Granite::load(path, 1024, 4, Some(16 << 20)).is_err());
    let mut m = Granite::load(path, 32768, 4, None).unwrap();
    eprintln!(
        "MINICPM_MEMORY weights={} kv={} total={} workspace={}",
        m.weight_bytes,
        m.kv_bytes,
        m.device.allocated_bytes(),
        m.device.allocated_bytes() - m.weight_bytes - m.kv_bytes
    );
    assert!(m.device.allocated_bytes() - m.weight_bytes - m.kv_bytes <= 201326592);
    let p: Vec<_> = (1000..1553).collect();
    m.prefill_begin(3, p.clone()).unwrap();
    let (_, done) = m.forward_mixed(&[], 512).unwrap();
    assert!(done.is_empty());
    let (_, done) = m.forward_mixed(&[], 512).unwrap();
    assert_eq!(done.len(), 1);
    assert!(done[0].1.iter().all(|v| v.is_finite()));
    m.reset();
    let cached = m.forward_prefill(3, &p).unwrap();
    assert_eq!(m.take_prefill_reused(3), 544);
    m.reset();
    assert_eq!(m.forward_prefill(3, &p).unwrap(), cached);
    m.prefill_begin(1, vec![700; 73]).unwrap();
    let (decode, done) = m.forward_mixed(&[(3, 800, 553)], 7).unwrap();
    assert_eq!(decode.len(), m.vocab);
    assert!(decode.iter().all(|v| v.is_finite()));
    assert!(done.is_empty());
    assert!(m.prefill_abort(1));
    assert!(m.forward_mixed(&[(4, 1, 0)], 1).is_err());
    assert!(m.forward_prefill(0, &[m.vocab as u32]).is_err());
    m.release_inactive_slots(&[false; 4]);
    assert!(m.forward_mixed(&[], 1).unwrap().0.is_empty());
}

#[test]
#[ignore = "requires independent MiniCPM5 MLX fixtures at PADDOCK_MINICPM_REFERENCE"]
fn minicpm_mlx_reference_generations_and_batch() {
    let root = std::env::var("PADDOCK_MINICPM_REFERENCE").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let path = Path::new(manifest["model"].as_str().unwrap());
    let mut passed = 0;
    let mut total = 0;
    for batch in [1, 4] {
        let mut model = Granite::load(path, 2048, batch, None).unwrap();
        for wave in manifest["cases"].as_array().unwrap().chunks(batch) {
            let mut expected = Vec::new();
            let mut positions = Vec::new();
            for (slot, case) in wave.iter().enumerate() {
                let ids: Vec<_> = case["prompt"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                positions.push(ids.len() as u32);
                model.prefill_begin(slot, ids).unwrap();
                expected.push(
                    case["tokens"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect::<Vec<_>>(),
                );
            }
            let mut next = vec![0; wave.len()];
            while !model.pending.is_empty() {
                let (_, done) = model.forward_mixed(&[], 512).unwrap();
                for (slot, logits, _) in done {
                    next[slot] = top(&logits);
                }
            }
            let mut generated = vec![Vec::new(); wave.len()];
            let mut active = vec![true; wave.len()];
            for _ in 0..128 {
                let mut rows = Vec::new();
                for slot in 0..wave.len() {
                    if !active[slot] {
                        continue;
                    }
                    generated[slot].push(next[slot]);
                    if matches!(next[slot], 1 | 130073) {
                        active[slot] = false;
                    } else {
                        rows.push((slot, next[slot], positions[slot]));
                        positions[slot] += 1;
                    }
                }
                if rows.is_empty() {
                    break;
                }
                let (logits, _) = model.forward_mixed(&rows, 512).unwrap();
                for (row, (slot, _, _)) in rows.iter().enumerate() {
                    next[*slot] = top(&logits[row * model.vocab..(row + 1) * model.vocab]);
                }
            }
            for i in 0..wave.len() {
                total += 1;
                if generated[i] == expected[i] {
                    passed += 1;
                } else {
                    eprintln!(
                        "MINICPM mismatch c={batch} case={total}: got {:?}, want {:?}",
                        generated[i], expected[i]
                    );
                }
            }
            model.reset();
        }
    }
    // Independent full-vocabulary first-step error, not merely fluent text.
    let mut model = Granite::load(path, 2048, 1, None).unwrap();
    for (i, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
        let ids: Vec<_> = case["prompt"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let out = model.forward_prefill(0, &ids).unwrap();
        let fixture = SafetensorsFile::open(&root.join(format!("{i}.safetensors"))).unwrap();
        let want: Vec<_> = fixture
            .bytes("logits")
            .unwrap()
            .1
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let error = out
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "MINICPM first case={i} max_error={error} top={}/{}",
            top(&out),
            top(&want)
        );
        assert_eq!(top(&out), top(&want));
    }
    eprintln!("MINICPM full generations {passed}/{total}");
    assert_eq!(passed, total);
}

#[test]
#[ignore = "requires MiniCPM --trace-decode fixtures"]
fn minicpm_mlx_teacher_forced_margins() {
    let root = std::env::var("PADDOCK_MINICPM_REFERENCE").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let fixture = SafetensorsFile::open(&root.join("decode-logits.safetensors")).unwrap();
    let want: Vec<_> = fixture
        .bytes("logits")
        .unwrap()
        .1
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let case = &manifest["cases"][6];
    let prompt: Vec<_> = case["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let tokens: Vec<_> = case["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut model = Granite::load(
        Path::new(manifest["model"].as_str().unwrap()),
        2048,
        1,
        None,
    )
    .unwrap();
    let mut got = model.forward_prefill(0, &prompt).unwrap();
    let mut failures = 0;
    for (step, &token) in tokens.iter().enumerate() {
        let expected = &want[step * model.vocab..(step + 1) * model.vocab];
        assert_eq!(top(expected), token);
        let predicted = top(&got);
        let error = got
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let margin = expected[token as usize] - expected[predicted as usize];
        eprintln!(
            "MINICPM_FORCED step={step} got={predicted} expected={token} reference_margin={margin} native_margin={} max={error}",
            got[token as usize] - got[predicted as usize]
        );
        failures += usize::from(predicted != token);
        if step + 1 < tokens.len() {
            got = model
                .forward_mixed(&[(0, token, (prompt.len() + step) as u32)], 512)
                .unwrap()
                .0;
        }
    }
    assert_eq!(failures, 0, "teacher-forced top tokens differ");
}
