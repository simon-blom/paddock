use super::*;
use crate::{device::MetalDevice, weights::Weight};
use paddock_models::mapped::MappedGguf;
use serde_json::Value;
use std::path::{Path, PathBuf};

#[test]
fn storage_bounds_and_cross_row_superblocks() {
    for (ty, block, bytes) in [(20, 32, 18), (21, 256, 110), (22, 256, 82)] {
        for dims in [vec![640, 66], vec![2560, 640, 512], vec![160, 64]] {
            let size = dims.iter().product::<usize>() / block * bytes;
            validate(ty, &dims, size).unwrap();
            assert!(validate(ty, &dims, size - 1).is_err());
            assert!(validate(ty, &dims, size + 1).is_err());
        }
        for dims in [
            vec![],
            vec![32],
            vec![32, 0],
            vec![33, 256],
            vec![usize::MAX, 2],
            vec![32, 2, 3, 4],
        ] {
            assert!(validate(ty, &dims, 0).is_err(), "{ty} {dims:?}");
        }
    }
    assert!(validate(21, &[640, 65], 0).is_err());
    assert!(validate(22, &[160, 65], 0).is_err());
    assert!(validate(18, &[256, 1], 98).is_err());
}

fn hexes(text: &str) -> Vec<u64> {
    text.split(|c: char| c.is_whitespace() || c == ',')
        .filter_map(|word| word.strip_prefix("0x"))
        .map(|word| u64::from_str_radix(word.trim_end_matches('u'), 16).unwrap())
        .collect()
}

#[test]
fn codebooks_are_byte_identical_to_noticed_format_data() {
    let original = include_str!("../../../paddock-kernels/src/reference/iq_grids.rs");
    let metal = include_str!("../../../../packs/metal/iquant_tables.metal");
    for (name, width, count) in [("IQ2S_GRID", 64, 1024), ("IQ3S_GRID", 32, 512)] {
        let body = original
            .split(&format!("pub const {name}:"))
            .nth(1)
            .unwrap()
            .split("= [")
            .nth(1)
            .unwrap()
            .split("];")
            .next()
            .unwrap();
        let source = hexes(body);
        assert_eq!(source.len(), count);
        let expected = source
            .into_iter()
            .flat_map(|n| {
                if width == 64 {
                    vec![n & 0xffff_ffff, n >> 32]
                } else {
                    vec![n]
                }
            })
            .collect::<Vec<_>>();
        let body = metal
            .split(&format!("metal_{name}["))
            .nth(1)
            .unwrap()
            .split("= {")
            .nth(1)
            .unwrap()
            .split("};")
            .next()
            .unwrap();
        assert_eq!(hexes(body), expected);
    }
}

#[test]
fn smallest_half_scale_survives_gpu_widening() {
    // Fixed format vectors, not a CPU decoder: scale bits 0x0001, codebook
    // entry zero, signs/scales zero. IQ2_S gives 8*(1/8)*2^-24; IQ3_S gives
    // 1*2^-24; IQ4_NL gives -127*2^-24. Literal F32 bits catch two GPU paths
    // agreeing only because both accidentally flush an F16 subnormal.
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    for (ty, bytes, width, golden) in [
        (20, 18, 32, 0xb6fe0000),
        (21, 110, 256, 0x33800000),
        (22, 82, 256, 0x33800000),
    ] {
        let mut raw = vec![0u8; bytes];
        raw[0] = 1;
        let w = d.upload(&raw).unwrap();
        let ids = upload_u(&d, &[0]);
        let out = output(&d, width);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "iq_gather",
            &[&w, &ids, &out],
            &[ty, width as u32, 1, 1],
            [(width / 4).div_ceil(256), 1, 1],
            256,
        );
        cmd.submit().unwrap().wait().unwrap();
        assert!(
            unsafe { out.read_f32(0, width) }
                .iter()
                .all(|v| v.to_bits() == golden),
            "format {ty}: half subnormal lost"
        );
        guard(&out, width);
    }
    assert_eq!(d.allocated_bytes(), 0);
}

fn upload_u(d: &MetalDevice, v: &[u32]) -> Buffer {
    d.upload(&v.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn read_f(path: impl AsRef<Path>) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert!(b.len().is_multiple_of(4));
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn number(v: &Value, key: &str) -> usize {
    v[key].as_u64().unwrap() as usize
}
fn string<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap()
}

fn close(actual: &[f32], expected: &[f32], abs: f32, rel: f32) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0f32;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "nonfinite at {i}: {a} {b}");
        let error = (a - b).abs();
        max = max.max(error);
        assert!(
            error <= abs + rel * b.abs(),
            "at {i}: {a} vs {b}, delta {error}"
        );
    }
    max
}
fn output(d: &MetalDevice, elems: usize) -> Buffer {
    d.upload(
        &vec![-9876.5f32; elems + 17]
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn guard(b: &Buffer, elems: usize) {
    assert_eq!(unsafe { b.read_f32(elems, 17) }, vec![-9876.5f32; 17]);
}

// A minimal one-tensor container exercises the real raw Weight::load path,
// not a test-only decoder. All bytes came from the independent GPU fixture.
fn container(path: &Path, raw: &[u8], ty: u32, k: usize, n: usize) {
    let mut bytes = Vec::new();
    for v in [0x4655_4747u32, 3] {
        bytes.extend(v.to_le_bytes());
    }
    for v in [1u64, 0, 1] {
        bytes.extend(v.to_le_bytes());
    } // tensors, metadata, name length
    bytes.push(b'w');
    bytes.extend(2u32.to_le_bytes());
    for v in [k as u64, n as u64] {
        bytes.extend(v.to_le_bytes());
    }
    bytes.extend(ty.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(raw);
    std::fs::write(path, bytes).unwrap();
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("paddock-iq-{}-{now}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
#[ignore = "M5 + independent reference fixtures in PADDOCK_METAL_IQ_REFERENCE"]
fn independent_mps_format_and_projection_parity() {
    let root = PathBuf::from(std::env::var("PADDOCK_METAL_IQ_REFERENCE").unwrap());
    let report: Value =
        serde_json::from_slice(&std::fs::read(root.join("results.json")).unwrap()).unwrap();
    assert_eq!(report["complete"], true);
    assert_eq!(report["device"], "mps");
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        if report["suite"] == "checkpoint" {
            76
        } else {
            108
        }
    );
    let d = MetalDevice::new(Some(1 << 30)).unwrap();
    let temp = Temp::new();
    for case in cases {
        let (ty, k, n, rows) = (
            number(case, "ty") as u32,
            number(case, "k"),
            number(case, "n"),
            number(case, "rows"),
        );
        let (id, key) = (string(case, "id"), string(case, "weight"));
        let raw = std::fs::read(root.join(format!("{key}.raw"))).unwrap();
        container(&temp.0.join("fixture.gguf"), &raw, ty, k, n);
        let map = MappedGguf::open(&temp.0.join("fixture.gguf")).unwrap();
        // Default model ingestion must stay closed: other consumers use
        // embedding/fused kernels whose format support has not been extended.
        assert!(Weight::load(&d, &map, "w", &[k, n]).is_err());
        let w = Weight::load_iq(&d, &map, "w", &[k, n]).unwrap();
        assert_eq!(w.buffer.len(), raw.len());
        assert_eq!(w.ty, ty);
        assert!(Weight::load_iq(&d, &map, "w", &[k, n + 1]).is_err());
        let input = d
            .upload(&std::fs::read(root.join(format!("{id}.input.f32"))).unwrap())
            .unwrap();
        let out = output(&d, rows * n);
        let scratch = d
            .alloc(k.div_ceil(128) * 128 * rows.div_ceil(128) * 128 * 2)
            .unwrap();
        let cmd = d.begin().unwrap();
        w.linear(&cmd, &input, &out, rows, 0.375, &scratch);
        cmd.submit().unwrap().wait().unwrap();
        let error = close(
            &unsafe { out.read_f32(0, rows * n) },
            &read_f(root.join(format!("{id}.output.f32"))),
            2e-4,
            2e-5,
        );
        guard(&out, rows * n);
        if matches!(rows, 1 | 4 | 9) {
            let other = output(&d, rows * n);
            let cmd = d.begin().unwrap();
            w.linear(&cmd, &input, &out, rows, 1.0, &scratch);
            cmd.submit().unwrap().wait().unwrap();
            let separate = unsafe { out.read_f32(0, rows * n) };
            let cmd = d.begin().unwrap();
            crate::weights::projections(&cmd, &[(&w, &out), (&w, &other)], &input, rows, &scratch);
            cmd.submit().unwrap().wait().unwrap();
            close(&unsafe { out.read_f32(0, rows * n) }, &separate, 0., 0.);
            close(&unsafe { other.read_f32(0, rows * n) }, &separate, 0., 0.);
            guard(&out, rows * n);
            guard(&other, rows * n);
        }
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "linear_input_padded",
            &[&input, &scratch],
            &[k as u32, n as u32, rows as u32],
            [(scratch.len() / 2).div_ceil(256), 1, 1],
            256,
        );
        w.linear_prepared(&cmd, &scratch, &out, rows, 0.375);
        cmd.submit().unwrap().wait().unwrap();
        close(
            &unsafe { out.read_f32(0, rows * n) },
            &read_f(root.join(format!("{id}.prepared.f32"))),
            2e-4,
            2e-5,
        );
        guard(&out, rows * n);
        if rows == 1 {
            let ids = upload_u(&d, &(0..n as u32).collect::<Vec<_>>());
            let values = output(&d, k * n);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "iq_gather",
                &[&w.buffer, &ids, &values],
                &[ty, k as u32, n as u32, n as u32],
                [(k * n / 4).div_ceil(256), 1, 1],
                256,
            );
            cmd.submit().unwrap().wait().unwrap();
            close(
                &unsafe { values.read_f32(0, k * n) },
                &read_f(root.join(format!("{key}.weights.f32"))),
                0.,
                0.,
            );
            guard(&values, k * n);
            // Invalid gather ids propagate nonfinite values without touching
            // an invalid table address, including UINT_MAX.
            unsafe { ids.write_u32(&vec![u32::MAX; n]) };
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "iq_gather",
                &[&w.buffer, &ids, &values],
                &[ty, k as u32, n as u32, n as u32],
                [(k * n / 4).div_ceil(256), 1, 1],
                256,
            );
            cmd.submit().unwrap().wait().unwrap();
            assert!(
                unsafe { values.read_f32(0, k * n) }
                    .iter()
                    .all(|x| x.is_nan())
            );
            guard(&values, k * n);
        }
        eprintln!("{id}: MPS max_abs={error}");
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
#[ignore = "M5 + independent reference fixtures in PADDOCK_METAL_IQ_REFERENCE"]
fn independent_mps_512_expert_projections() {
    let root = PathBuf::from(std::env::var("PADDOCK_METAL_IQ_REFERENCE").unwrap());
    let report: Value =
        serde_json::from_slice(&std::fs::read(root.join("results.json")).unwrap()).unwrap();
    assert_eq!(report["complete"], true);
    assert_eq!(report["device"], "mps");
    let cases = report["experts"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    let d = MetalDevice::new(Some(1 << 30)).unwrap();
    for c in cases {
        let (k, n, e, active, count) = (
            number(c, "k"),
            number(c, "n"),
            number(c, "entries"),
            number(c, "active"),
            number(c, "experts"),
        );
        let ty = number(c, "ty") as u32;
        let id = string(c, "id");
        let raw = std::fs::read(root.join(format!("{}.raw", string(c, "weight")))).unwrap();
        validate(ty, &[k, n, count], raw.len()).unwrap();
        let w = d.upload(&raw).unwrap();
        let x = d
            .upload(&std::fs::read(root.join(format!("{id}.input.f32"))).unwrap())
            .unwrap();
        let ids = d
            .upload(&std::fs::read(root.join(format!("{id}.ids.u32"))).unwrap())
            .unwrap();
        let lists = d.alloc(count * e * 4).unwrap();
        let counts = d.alloc(count * 4).unwrap();
        let max_tiles = e.div_ceil(32) + count;
        let tiles = d.alloc((1 + max_tiles * 2) * 4).unwrap();
        let out = output(&d, e * n);
        let expected = read_f(root.join(format!("{id}.output.f32")));
        let params = [k as u32, n as u32, e as u32, active as u32, count as u32];
        let mv = ["iq_expert_mv20", "iq_expert_mv21", "iq_expert_mv22"][(ty - 20) as usize];
        let mm = ["iq_expert_mm20", "iq_expert_mm21", "iq_expert_mm22"][(ty - 20) as usize];
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            mv,
            &[&w, &x, &ids, &out],
            &params,
            [n.div_ceil(4), e, 1],
            128,
        );
        cmd.submit().unwrap().wait().unwrap();
        let mv_error = close(&unsafe { out.read_f32(0, e * n) }, &expected, 2e-4, 2e-5);
        guard(&out, e * n);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "moe_align",
            &[&ids, &lists, &counts],
            &[e as u32],
            [count, 1, 1],
            256,
        );
        cmd.dispatch(
            "iq_tiles512",
            &[&counts, &tiles],
            &[count as u32],
            [1, 1, 1],
            512,
        );
        cmd.dispatch(
            mm,
            &[&w, &x, &lists, &counts, &tiles, &out],
            &params,
            [n.div_ceil(32), max_tiles, 1],
            128,
        );
        cmd.submit().unwrap().wait().unwrap();
        let mm_error = close(&unsafe { out.read_f32(0, e * n) }, &expected, 2e-4, 2e-5);
        guard(&out, e * n);
        let (ids_host, counts_host, lists_host, tiles_host) = unsafe {
            (
                ids.read_u32(e),
                counts.read_u32(count),
                lists.read_u32(count * e),
                tiles.read_u32(1 + max_tiles * 2),
            )
        };
        let mut expected_tiles = Vec::new();
        for expert in 0..count {
            let entries = ids_host
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| (x == expert as u32).then_some(i as u32))
                .collect::<Vec<_>>();
            assert_eq!(counts_host[expert] as usize, entries.len());
            assert_eq!(&lists_host[expert * e..expert * e + entries.len()], entries);
            for first in (0..entries.len()).step_by(32) {
                expected_tiles.extend([expert as u32, first as u32]);
            }
        }
        assert_eq!(tiles_host[0] as usize * 2, expected_tiles.len());
        assert_eq!(&tiles_host[1..1 + expected_tiles.len()], expected_tiles);
        unsafe { ids.write_u32(&vec![u32::MAX; e]) };
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            mv,
            &[&w, &x, &ids, &out],
            &params,
            [n.div_ceil(4), e, 1],
            128,
        );
        cmd.submit().unwrap().wait().unwrap();
        assert!(unsafe { out.read_f32(0, e * n) }.iter().all(|x| x.is_nan()));
        guard(&out, e * n);
        eprintln!("{id}: MPS mv max_abs={mv_error}, grouped max_abs={mm_error}");
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
#[ignore = "M5 + synthetic MPS routing fixtures in PADDOCK_METAL_IQ_REFERENCE"]
fn independent_mps_top10_router_and_invalid_rows() {
    let root = PathBuf::from(std::env::var("PADDOCK_METAL_IQ_REFERENCE").unwrap());
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let input = d
        .upload(&std::fs::read(root.join("routing.input.f32")).unwrap())
        .unwrap();
    let ids = d.alloc(24 * 10 * 4).unwrap();
    let weights = output(&d, 24 * 10);
    let shared = output(&d, 24);
    let invalid = d.alloc(24 * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "iq_route512",
        &[&input, &ids, &weights, &shared, &invalid],
        &[24],
        [24, 1, 1],
        32,
    );
    cmd.submit().unwrap().wait().unwrap();
    let read_u = |name: &str| {
        std::fs::read(root.join(name))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let flags = unsafe { invalid.read_u32(24) };
    assert_eq!(unsafe { ids.read_u32(24 * 10) }, read_u("routing.ids.u32"));
    assert_eq!(flags, read_u("routing.invalid.u32"));
    let actual_weights = unsafe { weights.read_f32(0, 24 * 10) };
    let actual_shared = unsafe { shared.read_f32(0, 24) };
    let expected_weights = read_f(root.join("routing.weights.f32"));
    let expected_shared = read_f(root.join("routing.shared.f32"));
    for row in 0..24 {
        if flags[row] == 0 {
            close(
                &actual_weights[row * 10..(row + 1) * 10],
                &expected_weights[row * 10..(row + 1) * 10],
                1e-7,
                1e-6,
            );
            close(
                &actual_shared[row..row + 1],
                &expected_shared[row..row + 1],
                1e-7,
                1e-6,
            );
        } else {
            assert!(
                actual_weights[row * 10..(row + 1) * 10]
                    .iter()
                    .all(|x| x.is_nan())
            );
            assert!(actual_shared[row].is_nan());
        }
    }
    guard(&weights, 24 * 10);
    guard(&shared, 24);
}

#[test]
#[ignore = "128GiB M5, full PADDOCK_FLASH_NEXT_MODEL and checkpoint PADDOCK_METAL_IQ_REFERENCE; uploads 28.8GB PLE table"]
fn full_size_ple_gather_uses_64_bit_offsets() {
    let path = PathBuf::from(std::env::var("PADDOCK_FLASH_NEXT_MODEL").unwrap());
    let root = PathBuf::from(std::env::var("PADDOCK_METAL_IQ_REFERENCE").unwrap());
    let report: Value =
        serde_json::from_slice(&std::fs::read(root.join("results.json")).unwrap()).unwrap();
    assert_eq!(report["suite"], "checkpoint");
    assert_eq!(report["complete"], true);
    let case = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["weight"] == "ple-selected")
        .unwrap();
    let rows = case["source"]["original_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| u32::try_from(r.as_u64().unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 64);
    assert_eq!(rows[63], 320001535);
    let d = MetalDevice::new(Some(32 << 30)).unwrap();
    let map = MappedGguf::open(&path).unwrap();
    let start = std::time::Instant::now();
    let table =
        Weight::load_iq(&d, &map, "per_layer_token_embd.weight", &[160, 320001536]).unwrap();
    assert_eq!(table.buffer.len(), 28800138240);
    let ids = upload_u(&d, &rows);
    let out = output(&d, 64 * 160);
    let params = [20, 160, 64, 320001536];
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "iq_gather",
        &[&table.buffer, &ids, &out],
        &params,
        [(64usize * 160 / 4).div_ceil(256), 1, 1],
        256,
    );
    cmd.submit().unwrap().wait().unwrap();
    let expected = read_f(root.join("ple-selected.weights.f32"));
    close(&unsafe { out.read_f32(0, 64 * 160) }, &expected, 0., 0.);
    guard(&out, 64 * 160);
    let mut invalid = rows.clone();
    invalid[0] = u32::MAX;
    invalid[1] = 320001536;
    unsafe { ids.write_u32(&invalid) };
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "iq_gather",
        &[&table.buffer, &ids, &out],
        &params,
        [(64usize * 160 / 4).div_ceil(256), 1, 1],
        256,
    );
    cmd.submit().unwrap().wait().unwrap();
    let values = unsafe { out.read_f32(0, 64 * 160) };
    assert!(values[..320].iter().all(|v| v.is_nan()));
    close(&values[320..], &expected[320..], 0., 0.);
    guard(&out, 64 * 160);
    eprintln!(
        "full PLE allocation={} bytes, last row={}, exact 64-row gather, elapsed={:?} (load diagnostic, not TTFT)",
        table.buffer.len(),
        rows[63],
        start.elapsed()
    );
    drop((table, ids, out));
    assert_eq!(d.allocated_bytes(), 0);
}
