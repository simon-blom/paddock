use super::{deltanet, mlx, moe, ple, qsa, residual};
use crate::device::{Buffer, MetalDevice};
use paddock_models::safetensors::SafetensorsFile;
use std::path::Path;

fn compare(b: &Buffer, f: &SafetensorsFile, name: &str) -> bool {
    let (info, raw) = f.bytes(name).unwrap();
    let n = info.shape.iter().product();
    let actual = unsafe { b.read_f32(0, n) };
    let expected = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b));
    let mut max = 0f32;
    let mut peak = 0f32;
    let mut unequal = 0;
    for (&a, b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite(), "{name} nonfinite");
        max = max.max((a - b).abs());
        peak = peak.max(b.abs());
        unequal += usize::from(a != b);
    }
    eprintln!(
        "STAGE {name} shape={:?} error={max} peak={peak} unequal={unequal}/{n}",
        info.shape
    );
    max <= peak * 0.008 + 0.0001
}

#[test]
#[ignore = "PADDOCK_FLASH_NEXT_MLX_STAGE_REFERENCE: independent same-weights MLX-VLM GPU fixtures"]
fn flash_next_mlx_stages_match_gpu_reference() {
    let dir = std::env::var("PADDOCK_FLASH_NEXT_MLX_STAGE_REFERENCE").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let s = mlx::Source::open(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    let hc = residual::HyperConnection::load_mlx(
        &d,
        &s,
        &format!("{}.layers.0.attn_hyper_connection", mlx::ROOT),
        true,
    )
    .unwrap();
    let dn = deltanet::Weights::load_mlx(&d, &s, 0).unwrap();
    let qsa = qsa::Weights::load_mlx(&d, &s, 3).unwrap();
    let moe = moe::Weights::load_mlx(&d, &s, 0).unwrap();
    let hw = residual::Workspace::new(&d, 128).unwrap();
    let mw = moe::Workspace::new(&d, 128).unwrap();
    let mut failed = vec![];
    for c in manifest["cases"].as_array().unwrap() {
        let rows = c["rows"].as_u64().unwrap() as usize;
        let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
        let x = d.upload(f.bytes("x").unwrap().1).unwrap();
        let h = d.upload(f.bytes("h").unwrap().1).unwrap();
        let y = d.alloc(rows * 2560 * 4).unwrap();
        let mut ds = deltanet::State::new(&d, &dn, 128, 1, 128).unwrap();
        let mut qs = qsa::State::new(&d, &qsa, 128, 1, 128).unwrap();
        let positions = (0..rows).map(|i| (0, i)).collect::<Vec<_>>();
        let cmd = d.begin().unwrap();
        hc.encode(&cmd, &h, &hw, rows);
        hc.combine(&cmd, &h, &x, &hw, rows);
        cmd.finish().unwrap();
        for (b, name) in [
            (&hw.norm, "hc_norm"),
            (&hw.low, "hc_low"),
            (&hw.gate, "hc_up"),
            (&hw.inject, "hc_gain"),
            (&hw.mixed, "hc_mix"),
            (&h, "hc_combined"),
        ] {
            if !compare(b, &f, name) {
                failed.push(format!("{name} rows={rows}"));
            }
        }
        ds.run(&d, &dn, &positions, &x, &y).unwrap();
        for (b, name) in [
            (&ds.scratch.qkv, "dn_qkv"),
            (&ds.scratch.convolved, "dn_qk"),
            (&ds.scratch.attn, "dn_gated"),
            (&y, "dn_output"),
            (&ds.cache.state, "dn_state"),
        ] {
            if !compare(b, &f, name) {
                failed.push(format!("{name} rows={rows}"));
            }
        }
        let a = unsafe { ds.scratch.convolved.read_f32(0, rows * 10240) };
        let b = f
            .bytes("dn_qk")
            .unwrap()
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        for (name, start, end) in [("q", 0, 2048), ("k", 2048, 4096), ("v", 4096, 10240)] {
            let unequal = (0..rows)
                .flat_map(|r| (start..end).map(move |i| r * 10240 + i))
                .filter(|&i| a[i] != b[i])
                .count();
            eprintln!("DN_PART {name} rows={rows} unequal={unequal}");
        }
        qs.run(&d, &qsa, &positions, &x, &y).unwrap();
        if f.bytes("qsa_gated").is_some() {
            compare(&qs.scratch.attn, &f, "qsa_gated");
        }
        if !compare(&y, &f, "qsa_output") {
            failed.push(format!("qsa rows={rows}"));
        }
        let cmd = d.begin().unwrap();
        moe.encode(&cmd, &x, &mw, rows).unwrap();
        cmd.finish().unwrap();
        for (b, name) in [
            (&mw.logits, "moe_logits"),
            (&mw.shared_output, "moe_shared"),
            (&mw.shared_scale, "moe_scale"),
        ] {
            compare(b, &f, name);
        }
        let native = unsafe { mw.ids.read_u32(rows * 10) };
        let reference = f
            .bytes("moe_ids")
            .unwrap()
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as u32)
            .collect::<Vec<_>>();
        let disagree = native
            .chunks(10)
            .zip(reference.chunks(10))
            .filter(|(a, b)| a.iter().any(|i| !b.contains(i)))
            .count();
        eprintln!("ROUTE rows={rows} set_mismatches={disagree}");
        compare(&mw.weights, &f, "moe_weights");
        let actual = unsafe { mw.down.read_f32(0, rows * 10 * 2560) };
        let expected = f
            .bytes("moe_routed")
            .unwrap()
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        let mut error = 0f32;
        let mut peak = 0f32;
        for row in 0..rows {
            for e in 0..10 {
                if let Some(re) = reference[row * 10..row * 10 + 10]
                    .iter()
                    .position(|&id| id == native[row * 10 + e])
                {
                    for d in 0..2560 {
                        let a = actual[(row * 10 + e) * 2560 + d];
                        let b = expected[(row * 10 + re) * 2560 + d];
                        error = error.max((a - b).abs());
                        peak = peak.max(b.abs());
                    }
                }
            }
        }
        eprintln!("EXPERT rows={rows} aligned_error={error} peak={peak}");
        if !compare(&mw.output, &f, "moe_output") {
            failed.push(format!("moe rows={rows}"));
        }
    }
    assert!(failed.is_empty(), "failed stages: {}", failed.join(", "));
}

#[test]
#[ignore = "PADDOCK_FLASH_NEXT_MLX_STAGE_REFERENCE; rounding diagnostic, not model qualification"]
fn flash_next_mlx_rounding_diagnostic() {
    let dir = std::env::var("PADDOCK_FLASH_NEXT_MLX_STAGE_REFERENCE").unwrap();
    let dir = Path::new(&dir);
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    for rows in [1, 4, 33, 128] {
        let f = SafetensorsFile::open(&dir.join(format!("rows-{rows}.safetensors"))).unwrap();
        let norm = d.upload(f.bytes("hc_norm").unwrap().1).unwrap();
        let gate = d.upload(f.bytes("hc_up").unwrap().1).unwrap();
        let sg = d.alloc(rows * 10240 * 4).unwrap();
        let product = d.alloc(rows * 10240 * 4).unwrap();
        let sum = d.alloc(rows * 2560 * 4).unwrap();
        for sig in 0..3 {
            for prod in 0..2 {
                for order in 0..3 {
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        "q4b_trace_pointwise",
                        &[&norm, &gate, &sg, &product, &sum],
                        &[rows as u32, sig, prod, order],
                        [(rows * 2560).div_ceil(256), 1, 1],
                        256,
                    );
                    cmd.finish().unwrap();
                    eprintln!("POINTWISE rows={rows} sig={sig} prod={prod} order={order}");
                    compare(&sg, &f, "hc_sigmoid");
                    compare(&product, &f, "hc_product");
                    compare(&sum, &f, "hc_sum");
                }
            }
        }
        let x = d.upload(f.bytes("dn_conv").unwrap().1).unwrap();
        let y = d.alloc(rows * 10240 * 4).unwrap();
        for sum in 0..2 {
            for scale in 0..2 {
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "q4b_trace_qk",
                    &[&x, &y],
                    &[rows as u32, sum, scale],
                    [32, rows, 1],
                    32,
                );
                cmd.finish().unwrap();
                eprintln!("QK_ROUND rows={rows} sum={sum} scale={scale}");
                compare(&y, &f, "dn_qk");
            }
        }
    }
}

#[test]
#[ignore = "PADDOCK_FLASH_NEXT_MLX_PLE_REFERENCE; 32 GB resident GPU PLE gate"]
fn flash_next_mlx_ple_matches_gpu_reference() {
    let dir = std::env::var("PADDOCK_FLASH_NEXT_MLX_PLE_REFERENCE").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let s = mlx::Source::open(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let d = MetalDevice::new(Some(36 << 30)).unwrap();
    let w = ple::Weights::load_mlx(&d, &s).unwrap();
    let table = s.table(&d).unwrap();
    let mut failed = vec![];
    for c in manifest["cases"].as_array().unwrap() {
        let rows = c["rows"].as_u64().unwrap() as usize;
        let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
        let ids = f
            .bytes("ids")
            .unwrap()
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        let h = d.upload(f.bytes("h").unwrap().1).unwrap();
        let mut state = ple::State::new(&d, 128, 1, 128).unwrap();
        state
            .run(
                &d,
                &w,
                &table,
                &ids.iter()
                    .enumerate()
                    .map(|(i, &t)| (0, i, t))
                    .collect::<Vec<_>>(),
                &h,
            )
            .unwrap();
        if !compare(&state.embedding, &f, "ple_embedding") {
            failed.push(format!("embedding {rows}"));
        }
        for (b, name) in [
            (&state.key, "ple_key"),
            (&state.query, "ple_query"),
            (&state.value, "ple_value"),
            (&state.gated, "ple_gated"),
            (&state.norm, "ple_norm"),
        ] {
            if f.bytes(name).is_some() {
                compare(b, &f, name);
            }
        }
        if !compare(&h, &f, "ple_output") {
            failed.push(format!("output {rows}"));
        }
    }
    assert!(failed.is_empty(), "PLE mismatches: {failed:?}");
}
