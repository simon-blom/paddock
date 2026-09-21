use super::deltanet::{self as dn, CELLS, CONV, VD, VH};
use super::residual::WIDTH;
use crate::device::{Buffer, MetalDevice};
use paddock_models::mapped::MappedGguf;
use serde_json::Value;
use std::path::PathBuf;

fn read(path: PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn upload(d: &MetalDevice, v: &[f32]) -> Buffer {
    d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn close(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let error = (a - b).abs();
        max = max.max(error);
        // Pinned before running. Diagnostic F32 reduction-order allowance,
        // not a substitute for exact greedy full-generation parity.
        assert!(
            a.is_finite() && b.is_finite() && error <= 3e-5 + 3e-5 * b.abs(),
            "{label}[{i}]: {a} vs {b}, error {error}"
        );
    }
    max
}
struct Stream {
    length: usize,
    input: Vec<f32>,
    qkv: Vec<f32>,
    convolved: Vec<f32>,
    gates: Vec<f32>,
    attn: Vec<f32>,
    output: Vec<f32>,
    state: Vec<f32>,
    history: Vec<f32>,
    resumed_state: Vec<f32>,
}

#[test]
#[ignore = "requires elected Flash Next GGUF and independent MPS DeltaNet fixtures"]
fn actual_deltanet_mixed_chunks_decode_reset_and_fork() {
    let dir = PathBuf::from(std::env::var("PADDOCK_FLASH_NEXT_DELTANET_REFERENCE").unwrap());
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("results.json")).unwrap()).unwrap();
    assert_eq!(manifest["complete"], true);
    assert_eq!(manifest["device"], "mps");
    let map = MappedGguf::open(&PathBuf::from(
        std::env::var("PADDOCK_FLASH_NEXT_MODEL").unwrap(),
    ))
    .unwrap();
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    for layer in [3, 47, 48, usize::MAX] {
        assert!(dn::Weights::load(&d, &map, layer).is_err());
        assert_eq!(d.allocated_bytes(), 0);
    }
    for layer in [0, 2, 46] {
        let w = dn::Weights::load(&d, &map, layer).unwrap();
        let weights_bytes = d.allocated_bytes();
        assert!(dn::State::new(&d, &w, 513, 4, 128).is_err());
        assert_eq!(d.allocated_bytes(), weights_bytes);
        let streams: Vec<_> = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["layer"] == layer)
            .map(|c| {
                let id = c["id"].as_str().unwrap();
                let f = |suffix| read(dir.join(format!("{id}.{suffix}.f32")));
                Stream {
                    length: c["length"].as_u64().unwrap() as usize,
                    input: f("input"),
                    qkv: f("qkv"),
                    convolved: f("convolved"),
                    gates: f("gates"),
                    attn: f("attn"),
                    output: f("output"),
                    state: f("state"),
                    history: f("history"),
                    resumed_state: f("resumed_state"),
                }
            })
            .collect();
        assert_eq!(streams.len(), 4);
        for chunk in [1, 4, 8, 15, 16, 31, 32, 33, 128] {
            let before = d.allocated_bytes();
            let mut s = dn::State::new(&d, &w, 128, 4, 128).unwrap();
            assert_eq!(
                d.allocated_bytes() - before,
                dn::State::bytes(128, 4, 128).unwrap() as u64
            );
            let mut positions = [0; 4];
            let mut batches = 0;
            let mut worst = 0f32;
            while positions.iter().zip(&streams).any(|(&p, t)| p < t.length) {
                let mut rows = vec![];
                let mut input = vec![];
                for slot in [3, 1, 0, 2] {
                    let n = (streams[slot].length - positions[slot]).min(if slot == batches % 4 {
                        1
                    } else {
                        chunk
                    });
                    for pos in positions[slot]..positions[slot] + n {
                        rows.push((slot, pos));
                        input.extend_from_slice(
                            &streams[slot].input[pos * WIDTH..(pos + 1) * WIDTH],
                        );
                    }
                    positions[slot] += n;
                }
                let x = upload(&d, &input);
                let output = upload(&d, &vec![-9876.5; rows.len() * WIDTH + 17]);
                s.run(&d, &w, &rows, &x, &output).unwrap();
                for (b, width, reference, label) in [
                    (
                        &s.scratch.qkv,
                        CONV,
                        streams.iter().map(|s| &s.qkv).collect::<Vec<_>>(),
                        "qkv",
                    ),
                    (
                        &s.scratch.convolved,
                        CONV,
                        streams.iter().map(|s| &s.convolved).collect(),
                        "convolved",
                    ),
                    (
                        &s.scratch.gates,
                        VH * 2,
                        streams.iter().map(|s| &s.gates).collect(),
                        "gates",
                    ),
                    (
                        &s.scratch.attn,
                        VD,
                        streams.iter().map(|s| &s.attn).collect(),
                        "attn",
                    ),
                    (
                        &output,
                        WIDTH,
                        streams.iter().map(|s| &s.output).collect(),
                        "output",
                    ),
                ] {
                    for (r, &(slot, pos)) in rows.iter().enumerate() {
                        let error = close(
                            &unsafe { b.read_f32(r * width, width) },
                            &reference[slot][pos * width..(pos + 1) * width],
                            label,
                        );
                        if label == "output" {
                            worst = worst.max(error);
                        }
                    }
                }
                assert_eq!(
                    unsafe { output.read_f32(rows.len() * WIDTH, 17) },
                    [-9876.5; 17]
                );
                assert_eq!(unsafe { x.read_f32(0, input.len()) }, input);
                batches += 1;
            }
            for (slot, t) in streams.iter().enumerate() {
                close(
                    &unsafe { s.cache.state.read_f32(slot * CELLS, CELLS) },
                    &t.state,
                    "final state",
                );
                close(
                    &unsafe { s.cache.history.read_f32(slot * 3 * CONV, 3 * CONV) },
                    &t.history,
                    "history",
                );
            }
            let state = unsafe { s.cache.state.read_f32(0, 4 * CELLS) };
            let history = unsafe { s.cache.history.read_f32(0, 4 * 3 * CONV) };
            let x = upload(&d, &streams[0].input[..2 * WIDTH]);
            let y = upload(&d, &vec![-9876.5; 2 * WIDTH]);
            // Valid first row followed by an invalid later row: no partial
            // admission, cache write, output write or logical publication.
            assert!(s.run(&d, &w, &[(0, 65), (1, usize::MAX)], &x, &y).is_err());
            let undersized = d.alloc(4).unwrap();
            assert!(s.run(&d, &w, &[(0, 65)], &undersized, &y).is_err());
            assert!(s.run(&d, &w, &[(0, 65)], &x, &undersized).is_err());
            assert!(s.reset(&d, 4).is_err());
            assert!(s.copy_slot(&d, 0, 4).is_err());
            assert!(s.copy_slot(&d, 0, 0).is_err());
            assert_eq!(s.lengths, positions);
            assert_eq!(unsafe { s.cache.state.read_f32(0, 4 * CELLS) }, state);
            assert_eq!(
                unsafe { s.cache.history.read_f32(0, 4 * 3 * CONV) },
                history
            );
            assert_eq!(
                unsafe { y.read_f32(0, 2 * WIDTH) },
                vec![-9876.5; 2 * WIDTH]
            );
            s.reset(&d, 1).unwrap();
            assert!(
                unsafe { s.cache.state.read_f32(CELLS, CELLS) }
                    .iter()
                    .all(|&v| v == 0.)
            );
            assert!(
                unsafe { s.cache.history.read_f32(3 * CONV, 3 * CONV) }
                    .iter()
                    .all(|&v| v == 0.)
            );
            for slot in [0, 2, 3] {
                assert_eq!(
                    unsafe { s.cache.state.read_f32(slot * CELLS, CELLS) },
                    state[slot * CELLS..(slot + 1) * CELLS]
                );
                assert_eq!(
                    unsafe { s.cache.history.read_f32(slot * 3 * CONV, 3 * CONV) },
                    history[slot * 3 * CONV..(slot + 1) * 3 * CONV]
                );
            }
            // Reuse a cancelled slot with an unrelated short prefix.
            let x = upload(&d, &streams[2].input[..5 * WIDTH]);
            let y = d.alloc(5 * WIDTH * 4).unwrap();
            s.run(&d, &w, &(0..5).map(|p| (1, p)).collect::<Vec<_>>(), &x, &y)
                .unwrap();
            close(
                &unsafe { y.read_f32(0, 5 * WIDTH) },
                &streams[2].output[..5 * WIDTH],
                "cancel/reuse",
            );
            // Restore over that unrelated slot, then advance both copies
            // independently. Copying state without conv would fail here.
            s.copy_slot(&d, 0, 1).unwrap();
            let t = &streams[0];
            let mut tail = t.input[t.length * WIDTH..].to_vec();
            tail.extend_from_slice(&t.input[t.length * WIDTH..]);
            let x = upload(&d, &tail);
            let y = d.alloc(6 * WIDTH * 4).unwrap();
            let rows: Vec<_> = [1, 0]
                .into_iter()
                .flat_map(|slot| (t.length..t.length + 3).map(move |p| (slot, p)))
                .collect();
            s.run(&d, &w, &rows, &x, &y).unwrap();
            for (row, slot) in [(0, 1), (3, 0)] {
                close(
                    &unsafe { y.read_f32(row * WIDTH, 3 * WIDTH) },
                    &t.output[t.length * WIDTH..],
                    "fork continuation",
                );
                close(
                    &unsafe { s.cache.state.read_f32(slot * CELLS, CELLS) },
                    &t.resumed_state,
                    "resumed state",
                );
            }
            eprintln!(
                "Flash Next DeltaNet layer={layer} chunk={chunk} batches={batches} max output error={worst:e}; reset/fork/ledger pass"
            );
        }
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
fn sigmoid_output_gate_is_literal_and_preserves_nan() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let x = upload(&d, &vec![1.; VD]);
    let z = upload(
        &d,
        &(0..VD)
            .map(|i| {
                if i % 4 == 0 {
                    0.
                } else if i % 4 == 1 {
                    f32::INFINITY
                } else if i % 4 == 2 {
                    f32::NEG_INFINITY
                } else {
                    f32::NAN
                }
            })
            .collect::<Vec<_>>(),
    );
    let w = upload(&d, &[2.; 128]);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4x_dn_gated_norm",
        &[&x, &z, &w],
        &[0f32.to_bits()],
        [VH, 1, 1],
        32,
    );
    cmd.submit().unwrap().wait().unwrap();
    for (i, x) in unsafe { x.read_f32(0, VD) }.into_iter().enumerate() {
        if i % 4 == 3 {
            assert!(x.is_nan());
        } else {
            assert_eq!(x, [1., 2., 0.][i % 4]);
        }
    }
}
