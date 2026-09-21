use super::{ple, residual};
use crate::device::{Buffer, MetalDevice};
use paddock_models::mapped::MappedGguf;
use residual::{WIDE, WIDTH};
use serde_json::Value;
use std::path::PathBuf;

mod projection;

fn fixture() -> (PathBuf, Value, MappedGguf) {
    let dir = PathBuf::from(
        std::env::var("PADDOCK_FLASH_NEXT_STATE_REFERENCE").expect("MPS fixture directory"),
    );
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("results.json")).unwrap()).unwrap();
    assert_eq!(manifest["complete"], true);
    assert_eq!(manifest["device"], "mps");
    let model = PathBuf::from(
        std::env::var("PADDOCK_FLASH_NEXT_MODEL").expect("elected three-shard artifact"),
    );
    let map = MappedGguf::open(&model).unwrap();
    assert_eq!(map.tensor_count(), 1224);
    (dir, manifest, map)
}
fn read_f(path: impl AsRef<std::path::Path>) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap();
    assert!(raw.len().is_multiple_of(4));
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn read_u(path: impl AsRef<std::path::Path>) -> Vec<u32> {
    let raw = std::fs::read(path).unwrap();
    assert!(raw.len().is_multiple_of(4));
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}
fn upload(d: &MetalDevice, x: &[f32]) -> Buffer {
    d.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn guarded(d: &MetalDevice, x: &[f32]) -> Buffer {
    let mut v = x.to_vec();
    v.extend([-9876.5; 17]);
    upload(d, &v)
}
fn guard(b: &Buffer, n: usize) {
    assert_eq!(unsafe { b.read_f32(n, 17) }, [-9876.5; 17]);
}

/// Isolate the generic projection shapes identified by whole-walk counters.
/// This is a warm-weight diagnostic, not end-to-end serving performance.
#[test]
#[ignore = "requires elected GGUF and an otherwise idle GPU; diagnostic timing only"]
fn time_flash_next_scalar_projection_shapes() {
    assert!(std::env::var_os("PADDOCK_METAL_PROFILE").is_none());
    let model = PathBuf::from(std::env::var("PADDOCK_FLASH_NEXT_MODEL").unwrap());
    let map = MappedGguf::open(&model).unwrap();
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    for (name, k, n, ty) in [
        ("blk.0.hc_attn_inject.weight", WIDE, 4, 0),
        ("blk.0.ssm_alpha.weight", WIDTH, 48, 0),
        ("blk.0.ffn_gate_inp.weight", WIDTH, 512, 0),
        ("blk.3.indexer.q_proj.weight", WIDTH, 512, 30),
        ("blk.3.indexer.k_proj.weight", WIDTH, 128, 30),
    ] {
        let w = residual::load_weight(&d, &map, name, &[k, n], ty).unwrap();
        for rows in [1, 4, 128] {
            // Fixed input data only; no host inference or reference math.
            let x = upload(&d, &vec![0.125; k * rows]);
            let y = d.alloc(n * rows * 4).unwrap();
            let mut times = vec![];
            for iteration in 0..24 {
                let cmd = d.begin().unwrap();
                for _ in 0..32 {
                    cmd.dispatch(
                        "linear",
                        &[&w.buffer, &x, &y],
                        &[k as u32, n as u32, rows as u32, ty, 1f32.to_bits()],
                        [n.div_ceil(4), rows, 1],
                        128,
                    );
                }
                let us = cmd.finish().unwrap() * 1e6 / 32.;
                if iteration >= 3 {
                    times.push(us);
                }
            }
            times.sort_by(f64::total_cmp);
            assert!(
                unsafe { y.read_f32(0, n * rows) }
                    .iter()
                    .all(|v| v.is_finite())
            );
            eprintln!(
                "Flash Next scalar projection {name} K={k} N={n} M={rows}: median_us={:.3}",
                times[times.len() / 2]
            );
        }
    }
    assert_eq!(d.allocated_bytes(), 0);
}

fn close(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}[{i}] nonfinite {a} {b}"
        );
        let error = (a - b).abs();
        max = max.max(error);
        // F32 reduction-order allowance, pinned before executing the gate.
        // This is a subgraph diagnostic, not a fuzzy full-model parity claim.
        assert!(
            error <= 3e-5 + 3e-5 * b.abs(),
            "{label}[{i}] {a} vs {b}: {error}"
        );
    }
    max
}

#[test]
#[ignore = "requires elected Flash Next GGUF and independent MPS state fixtures"]
fn actual_hyper_connections_match_independent_gpu() {
    let (dir, manifest, map) = fixture();
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    for case in manifest["hc"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let rows = case["rows"].as_u64().unwrap() as usize;
        let inject = case["inject"].as_bool().unwrap();
        let load = |suffix: &str| read_f(dir.join(format!("{id}.{suffix}.f32")));
        let w = residual::HyperConnection::load(&d, &map, case["prefix"].as_str().unwrap(), inject)
            .unwrap();
        let before = d.allocated_bytes();
        let s = residual::Workspace::new(&d, rows).unwrap();
        assert_eq!(
            d.allocated_bytes() - before,
            residual::Workspace::bytes(rows).unwrap() as u64
        );
        let h = guarded(&d, &load("input"));
        let delta = inject.then(|| upload(&d, &load("delta")));
        let cmd = d.begin().unwrap();
        w.encode(&cmd, &h, &s, rows);
        if let Some(delta) = &delta {
            w.combine(&cmd, &h, delta, &s, rows);
        }
        cmd.submit().unwrap().wait().unwrap();
        close(
            &unsafe { s.norm.read_f32(0, rows * WIDE) },
            &load("norm"),
            "hc norm",
        );
        let err = close(
            &unsafe { s.mixed.read_f32(0, rows * WIDTH) },
            &load("mixed"),
            id,
        );
        if inject {
            close(
                &unsafe { s.inject.read_f32(0, rows * 4) },
                &load("inject"),
                "hc inject",
            );
            close(
                &unsafe { h.read_f32(0, rows * WIDE) },
                &load("combined"),
                "hc combine",
            );
        } else {
            assert_eq!(unsafe { h.read_f32(0, rows * WIDE) }, load("input"));
        }
        guard(&h, rows * WIDE);
        eprintln!("Flash Next HC {id}: max mixed error {err:e}");
    }
    assert_eq!(d.allocated_bytes(), 0);
}

struct Stream {
    tokens: Vec<u32>,
    input: Vec<f32>,
    output: Vec<f32>,
    embedding: Vec<f32>,
    normalized: Vec<f32>,
    ring: Vec<f32>,
    ids: Vec<u32>,
}
fn streams(dir: &std::path::Path, manifest: &Value) -> Vec<Stream> {
    manifest["ple"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let id = case["id"].as_str().unwrap();
            let f = |s: &str| read_f(dir.join(format!("{id}.{s}.f32")));
            Stream {
                tokens: case["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect(),
                input: f("input"),
                output: f("output"),
                embedding: f("embedding"),
                normalized: f("normalized"),
                ring: f("ring"),
                ids: read_u(dir.join(format!("{id}.ids.u32"))),
            }
        })
        .collect()
}

#[test]
#[ignore = "requires full 28.8GB PLE table and independent MPS state fixtures"]
fn actual_ple_chunk_decode_slot_reset_and_ledger() {
    let (dir, manifest, map) = fixture();
    let reference = streams(&dir, &manifest);
    let d = MetalDevice::new(Some(32 << 30)).unwrap();
    {
        let weights = ple::Weights::load(&d, &map).unwrap();
        let table = ple::load_table(&d, &map).unwrap();
        // Every schedule includes c=4, but slots are reordered and ragged.
        // Whole prompts and tiny chunks exercise both sides of the Q8
        // projection election as well as repeated 9-row ring wraparound.
        for chunk in [128, 1, 4, 9, 16, 31, 33] {
            let before = d.allocated_bytes();
            let mut state = ple::State::new(&d, 128, 4, 128).unwrap();
            assert_eq!(
                d.allocated_bytes() - before,
                ple::State::bytes(128, 4, 128).unwrap() as u64
            );
            let mut offsets = [0; 4];
            let mut rounds = 0;
            let mut maximum = 0f32;
            while offsets
                .iter()
                .enumerate()
                .any(|(s, &p)| p < reference[s].tokens.len())
            {
                let mut rows = Vec::new();
                let mut input = Vec::new();
                let mut expected = Vec::new();
                let mut ids = Vec::new();
                let mut embedded = Vec::new();
                for slot in [3, 1, 0, 2] {
                    // Mixed decode rider beside prefill; lengths diverge.
                    let count = if chunk > 1 && slot == rounds % 4 {
                        1
                    } else {
                        chunk
                    };
                    let end = (offsets[slot] + count).min(reference[slot].tokens.len());
                    for pos in offsets[slot]..end {
                        let r = &reference[slot];
                        rows.push((slot, pos, r.tokens[pos]));
                        input.extend_from_slice(&r.input[pos * WIDE..(pos + 1) * WIDE]);
                        expected.extend_from_slice(&r.output[pos * WIDE..(pos + 1) * WIDE]);
                        ids.extend_from_slice(&r.ids[pos * 16..(pos + 1) * 16]);
                        embedded.extend_from_slice(&r.embedding[pos * WIDTH..(pos + 1) * WIDTH]);
                    }
                    offsets[slot] = end;
                }
                let hidden = guarded(&d, &input);
                state.run(&d, &weights, &table, &rows, &hidden).unwrap();
                assert_eq!(
                    unsafe { state.ids.read_u32(ids.len()) },
                    ids,
                    "integer hash chunk={chunk}"
                );
                assert_eq!(
                    unsafe { state.embedding.read_f32(0, embedded.len()) },
                    embedded,
                    "exact resident IQ4 gather"
                );
                maximum = maximum.max(close(
                    &unsafe { hidden.read_f32(0, expected.len()) },
                    &expected,
                    "PLE output",
                ));
                guard(&hidden, expected.len());
                rounds += 1;
            }
            let history = unsafe { state.history.read_u32(8) };
            for (slot, r) in reference.iter().enumerate() {
                close(
                    &unsafe { state.ring.read_f32(slot * 9 * WIDE, 9 * WIDE) },
                    &r.ring,
                    "PLE final ring",
                );
                assert_eq!(
                    history[slot * 2..slot * 2 + 2],
                    [r.tokens[r.tokens.len() - 1], r.tokens[r.tokens.len() - 2]]
                );
            }
            // Rejected late row and reset must not disturb live neighbors.
            let original_ring = unsafe { state.ring.read_f32(0, 4 * 9 * WIDE) };
            let original_history = unsafe { state.history.read_u32(8) };
            let h = guarded(&d, &reference[0].input[..2 * WIDE]);
            assert!(
                state
                    .run(
                        &d,
                        &weights,
                        &table,
                        &[(0, offsets[0], 1), (1, offsets[1], 248320)],
                        &h
                    )
                    .is_err()
            );
            assert!(state.reset(&d, 4).is_err());
            assert_eq!(
                unsafe { state.ring.read_f32(0, 4 * 9 * WIDE) },
                original_ring
            );
            assert_eq!(unsafe { state.history.read_u32(8) }, original_history);
            assert_eq!(
                unsafe { h.read_f32(0, 2 * WIDE) },
                reference[0].input[..2 * WIDE]
            );
            state.reset(&d, 1).unwrap();
            let cleared = unsafe { state.ring.read_f32(0, 4 * 9 * WIDE) };
            assert!(cleared[9 * WIDE..18 * WIDE].iter().all(|&v| v == 0.));
            for slot in [0, 2, 3] {
                assert_eq!(
                    cleared[slot * 9 * WIDE..(slot + 1) * 9 * WIDE],
                    original_ring[slot * 9 * WIDE..(slot + 1) * 9 * WIDE]
                );
            }
            // Reuse the canceled slot with a different stream, shorter than
            // the dilation span. Its result is identical to a fresh slot.
            let r = &reference[2];
            let h = guarded(&d, &r.input[..4 * WIDE]);
            let rows = (0..4).map(|p| (1, p, r.tokens[p])).collect::<Vec<_>>();
            state.run(&d, &weights, &table, &rows, &h).unwrap();
            close(
                &unsafe { h.read_f32(0, 4 * WIDE) },
                &r.output[..4 * WIDE],
                "reused slot",
            );
            let ring = unsafe { state.ring.read_f32(9 * WIDE, 9 * WIDE) };
            close(
                &ring[..4 * WIDE],
                &r.normalized[..4 * WIDE],
                "short reset ring",
            );
            assert!(ring[4 * WIDE..].iter().all(|&v| v == 0.));
            eprintln!(
                "Flash Next PLE chunk={chunk}, {rounds} mixed-slot batches: max output error {maximum:e}"
            );
        }
    }
    assert_eq!(
        d.allocated_bytes(),
        0,
        "full PLE table and all state released"
    );
}

#[test]
fn ple_signed_zero_and_nonfinite_gates() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    {
        // Four independent streams: exact zero, positive, negative, NaN.
        // Literal expected limits test sign(0) and guard nonfinite reductions.
        let mut k = vec![0.; WIDE];
        k[WIDTH..2 * WIDTH].fill(1.);
        k[2 * WIDTH..3 * WIDTH].fill(-1.);
        k[3 * WIDTH] = f32::NAN;
        let key = upload(&d, &k);
        let query = upload(&d, &vec![1.; WIDE]);
        let value = upload(&d, &vec![2.; WIDTH]);
        let out = guarded(&d, &vec![0.; WIDE]);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4x_ple_gate",
            &[&key, &query, &value, &out],
            &[WIDTH as u32],
            [4, 1, 1],
            256,
        );
        cmd.submit().unwrap().wait().unwrap();
        let y = unsafe { out.read_f32(0, WIDE) };
        assert!(y[..WIDTH].iter().all(|&v| v == 1.));
        assert!(y[WIDTH..2 * WIDTH].iter().all(|&v| v > 1.99 && v < 2.));
        assert!(y[2 * WIDTH..3 * WIDTH].iter().all(|&v| v > 0. && v < 0.01));
        assert!(y[3 * WIDTH..].iter().all(|v| v.is_nan()));
        guard(&out, WIDE);
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
fn literal_folded_norm_and_residual_identities() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    {
        let rows = 3;
        let x = upload(&d, &vec![2.; rows * WIDTH]);
        let h = guarded(&d, &vec![0.; rows * WIDE]);
        let gamma = upload(&d, &vec![0.; WIDE]);
        let norm = guarded(&d, &vec![99.; rows * WIDE]);
        let delta = upload(&d, &vec![3.; rows * WIDTH]);
        let inject = upload(&d, &vec![0.; rows * 4]);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4x_hc_init",
            &[&x, &h],
            &[WIDTH as u32, rows as u32],
            [(rows * WIDE).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "q4x_norm",
            &[&h, &gamma, &norm],
            &[WIDTH as u32, 4, residual::EPS.to_bits()],
            [4, rows, 1],
            256,
        );
        cmd.dispatch(
            "q4x_hc_combine",
            &[&h, &delta, &inject],
            &[WIDTH as u32, rows as u32],
            [(rows * WIDE).div_ceil(256), 1, 1],
            256,
        );
        cmd.submit().unwrap().wait().unwrap();
        // A stored zero gamma must stay zero, proving we did not add 1.
        assert!(
            unsafe { norm.read_f32(0, rows * WIDE) }
                .iter()
                .all(|&x| x == 0.)
        );
        // 2*sigmoid(0)=1: a zero-logit write is an ordinary residual add.
        assert!(
            unsafe { h.read_f32(0, rows * WIDE) }
                .iter()
                .all(|&x| x == 5.)
        );
        guard(&h, rows * WIDE);
        guard(&norm, rows * WIDE);
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
fn malformed_gpu_hash_rows_fail_closed_without_out_of_bounds_reads() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    {
        let u = |x: &[u32]| {
            d.upload(&x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let tokens = u(&[248320, 1, 1, 1]);
        let meta = u(&[0, 0, 0, 1, 4, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 9]);
        let history = u(&[248044, 248044]);
        let mut sentinel = vec![0; 64];
        sentinel.extend([0x13579bdf; 17]);
        let ids = u(&sentinel);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4x_ple_hash",
            &[&tokens, &meta, &history, &ids],
            &[4, 1],
            [4, 1, 1],
            32,
        );
        cmd.submit().unwrap().wait().unwrap();
        let result = unsafe { ids.read_u32(81) };
        assert_eq!(result[..64], [u32::MAX; 64]);
        assert_eq!(result[64..], [0x13579bdf; 17]);
    }
    assert_eq!(d.allocated_bytes(), 0);
}
