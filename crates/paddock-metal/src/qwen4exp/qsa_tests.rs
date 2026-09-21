use super::{qsa, residual::WIDTH};
use crate::device::{Buffer, MetalDevice};
use paddock_models::mapped::MappedGguf;
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> (PathBuf, Value) {
    let dir = PathBuf::from(std::env::var("PADDOCK_FLASH_NEXT_QSA_REFERENCE").unwrap());
    let m: Value =
        serde_json::from_slice(&std::fs::read(dir.join("results.json")).unwrap()).unwrap();
    assert_eq!(m["complete"], true);
    assert_eq!(m["device"], "mps");
    assert_eq!(m["index_cache"], "f32");
    assert_eq!(m["kv_cache"], "f16");
    (dir, m)
}
fn floats(p: PathBuf) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn uints(p: PathBuf) -> Vec<u32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}
fn upload(d: &MetalDevice, v: &[f32]) -> Buffer {
    d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn upload_u(d: &MetalDevice, v: &[u32]) -> Buffer {
    d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn close(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let e = (a - b).abs();
        max = max.max(e);
        // Predeclared allowance includes RoPE transcendental/reduction order
        // and downstream F16-cache rounding, not full-generation parity.
        assert!(
            a.is_finite() && b.is_finite() && e <= 2e-4 + 3e-5 * b.abs(),
            "{label}[{i}]: {a} vs {b}, error {e}"
        );
    }
    max
}

#[test]
#[ignore = "requires independent MPS QSA fixtures"]
fn radix_selection_matches_gpu_stable_topk_at_full_context() {
    let (dir, m) = fixture();
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    for c in m["selection"].as_array().unwrap() {
        let id = c["id"].as_u64().unwrap();
        let blocks = c["blocks"].as_u64().unwrap() as usize;
        let scores = upload(&d, &floats(dir.join(format!("select-{id}.scores.f32"))));
        // Zero blocks uses a one-token incomplete tail, still no complete block.
        let meta = upload_u(&d, &[0, (blocks * 4).saturating_sub(1) as u32, 0, 1]);
        let ids = upload_u(&d, &vec![0x12345678; 512 + 17]);
        let count = d.alloc(4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4s_select",
            &[&scores, &meta, &ids, &count],
            &[1, blocks.max(1) as u32, 1, 1, 4],
            [1, 1, 1],
            256,
        );
        cmd.submit().unwrap().wait().unwrap();
        let got = unsafe { ids.read_u32(529) };
        assert_eq!(
            &got[..512],
            uints(dir.join(format!("select-{id}.ids.u32"))),
            "blocks={blocks}, case={id}"
        );
        assert_eq!(&got[512..], &[0x12345678; 17]);
        assert_eq!(
            unsafe { count.read_u32(1) }[0],
            c["count"].as_u64().unwrap() as u32
        );
        eprintln!("QSA exact radix selection: case={id} blocks={blocks}");
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
fn selection_zero_ties_and_invalid_scores_fail_closed() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let meta = upload_u(&d, &[0, 2051, 0, 1]);
    let ids = d.alloc(512 * 4).unwrap();
    let count = d.alloc(4).unwrap();
    for bad in [None, Some(f32::NAN), Some(f32::INFINITY), Some(-1.)] {
        let mut values: Vec<_> = (0..513)
            .map(|i| if i % 2 == 0 { 0. } else { -0. })
            .collect();
        if let Some(bad) = bad {
            values[512] = bad;
        }
        let scores = upload(&d, &values);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4s_select",
            &[&scores, &meta, &ids, &count],
            &[1, 513, 1, 1, 4],
            [1, 1, 1],
            256,
        );
        cmd.submit().unwrap().wait().unwrap();
        if bad.is_some() {
            assert_eq!(unsafe { count.read_u32(1) }, [u32::MAX]);
            assert_eq!(unsafe { ids.read_u32(512) }, vec![u32::MAX; 512]);
        } else {
            assert_eq!(unsafe { count.read_u32(1) }, [512]);
            assert_eq!(unsafe { ids.read_u32(512) }, (0..512).collect::<Vec<_>>());
        }
    }
}

struct Stream {
    length: usize,
    probes: Vec<usize>,
    input: Vec<f32>,
    query: Vec<f32>,
    index_query: Vec<f32>,
    raw: Vec<f32>,
    pooled: Vec<f32>,
    output: Vec<f32>,
    selected: Vec<u32>,
    counts: Vec<u32>,
}
fn run(
    d: &MetalDevice,
    s: &mut qsa::State,
    w: &qsa::Weights,
    streams: &[Stream],
    rows: &[(usize, usize)],
) -> f32 {
    let x = upload(
        d,
        &rows
            .iter()
            .flat_map(|&(slot, pos)| {
                streams[slot].input[pos * WIDTH..(pos + 1) * WIDTH]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>(),
    );
    let y = upload(d, &vec![-9876.5; rows.len() * WIDTH + 17]);
    s.run(d, w, rows, &x, &y).unwrap();
    let mut worst = 0f32;
    let counts = unsafe { s.scratch.counts.read_u32(rows.len()) };
    let ids = unsafe { s.scratch.selected.read_u32(rows.len() * 512) };
    for (r, &(slot, pos)) in rows.iter().enumerate() {
        let t = &streams[slot];
        for (b, expected, width, label) in [
            (&s.scratch.query, &t.query, 6144, "query"),
            (&s.scratch.index_query, &t.index_query, 512, "index query"),
            (&s.scratch.raw, &t.raw, 128, "raw index keys"),
        ] {
            close(
                &unsafe { b.read_f32(r * width, width) },
                &expected[pos * width..(pos + 1) * width],
                label,
            );
        }
        if let Some(probe) = t.probes.iter().position(|&p| p == pos) {
            assert_eq!(counts[r], t.counts[probe]);
            assert_eq!(
                &ids[r * 512..(r + 1) * 512],
                &t.selected[probe * 512..(probe + 1) * 512],
                "selected blocks slot={slot} pos={pos}"
            );
            worst = worst.max(close(
                &unsafe { y.read_f32(r * WIDTH, WIDTH) },
                &t.output[probe * WIDTH..(probe + 1) * WIDTH],
                "attention output",
            ));
        }
    }
    assert_eq!(unsafe { y.read_f32(rows.len() * WIDTH, 17) }, [-9876.5; 17]);
    worst
}

#[test]
#[ignore = "requires elected Flash Next GGUF and independent MPS QSA fixtures"]
fn actual_qsa_sparse_boundary_mixed_prefill_decode_reset_and_fork() {
    let (dir, m) = fixture();
    let map = MappedGguf::open(&PathBuf::from(
        std::env::var("PADDOCK_FLASH_NEXT_MODEL").unwrap(),
    ))
    .unwrap();
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    for layer in [0, 2, 48, usize::MAX] {
        assert!(qsa::Weights::load(&d, &map, layer).is_err());
    }
    for layer in [3, 47] {
        let w = qsa::Weights::load(&d, &map, layer).unwrap();
        let streams: Vec<_> = m["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["layer"] == layer)
            .map(|c| {
                let id = c["id"].as_str().unwrap();
                let f = |suffix| floats(dir.join(format!("{id}.{suffix}.f32")));
                Stream {
                    length: c["length"].as_u64().unwrap() as usize,
                    probes: c["probes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect(),
                    input: f("input"),
                    query: f("query"),
                    index_query: f("index_query"),
                    raw: f("raw"),
                    pooled: f("pooled"),
                    output: f("output"),
                    selected: uints(dir.join(format!("{id}.selected.u32"))),
                    counts: uints(dir.join(format!("{id}.counts.u32"))),
                }
            })
            .collect();
        let before = d.allocated_bytes();
        let mut s = qsa::State::new(&d, &w, 128, 5, 2061).unwrap();
        assert_eq!(
            d.allocated_bytes() - before,
            qsa::State::bytes(128, 5, 2061).unwrap() as u64
        );
        let page_stride = 2061usize.div_ceil(16);
        let block_stride = page_stride * 4;
        let mut prefix_worst = 0f32;
        for first in (0..2040).step_by(128) {
            prefix_worst = prefix_worst.max(run(
                &d,
                &mut s,
                &w,
                &streams,
                &(first..(first + 128).min(2040))
                    .map(|pos| (0, pos))
                    .collect::<Vec<_>>(),
            ));
        }
        s.copy_slot(&d, 0, 4).unwrap();
        for chunk in [1, 4, 9, 16, 31, 128] {
            for slot in 0..4 {
                s.reset(&d, slot).unwrap();
            }
            s.copy_slot(&d, 4, 0).unwrap();
            let mut pos = [2040, 0, 0, 0];
            let mut round = 0;
            let mut worst = prefix_worst;
            while pos.iter().zip(&streams).any(|(&p, t)| p < t.length) {
                let mut rows = vec![];
                for slot in [3, 1, 0, 2] {
                    let count = (streams[slot].length - pos[slot]).min(if slot == round % 4 {
                        1
                    } else {
                        chunk
                    });
                    rows.extend((pos[slot]..pos[slot] + count).map(|p| (slot, p)));
                    pos[slot] += count;
                }
                worst = worst.max(run(&d, &mut s, &w, &streams, &rows));
                round += 1;
            }
            for (slot, t) in streams.iter().enumerate() {
                close(
                    &unsafe {
                        s.cache
                            .pooled
                            .read_f32(slot * block_stride * 128, t.pooled.len())
                    },
                    &t.pooled,
                    "cached pooled keys",
                );
            }
            let keys = unsafe { s.cache.keys.read_u32(s.cache.keys.len() / 4) };
            let values = unsafe { s.cache.values.read_u32(s.cache.values.len() / 4) };
            let pooled = unsafe { s.cache.pooled.read_u32(s.cache.pooled.len() / 4) };
            let ring = unsafe { s.cache.ring.read_u32(s.cache.ring.len() / 4) };
            let lengths = s.lengths.clone();
            let x = upload(&d, &streams[0].input[..WIDTH]);
            let y = upload(&d, &vec![-9876.5; WIDTH]);
            assert!(s.run(&d, &w, &[(0, 2061)], &x, &y).is_err());
            assert!(s.run(&d, &w, &[(1, 37), (2, usize::MAX)], &x, &y).is_err());
            assert!(s.reset(&d, 5).is_err());
            assert!(s.copy_slot(&d, 0, 5).is_err());
            assert!(s.copy_slot(&d, 0, 0).is_err());
            assert_eq!(s.lengths, lengths);
            assert_eq!(
                unsafe { s.cache.keys.read_u32(s.cache.keys.len() / 4) },
                keys
            );
            assert_eq!(
                unsafe { s.cache.values.read_u32(s.cache.values.len() / 4) },
                values
            );
            assert_eq!(
                unsafe { s.cache.pooled.read_u32(s.cache.pooled.len() / 4) },
                pooled
            );
            assert_eq!(
                unsafe { s.cache.ring.read_u32(s.cache.ring.len() / 4) },
                ring
            );
            assert_eq!(unsafe { y.read_f32(0, WIDTH) }, vec![-9876.5; WIDTH]);
            s.reset(&d, 1).unwrap();
            // Complete bytes of all other slots remain unchanged, including
            // the immutable saved prefix. Page permutation is reversed.
            let now = unsafe { s.cache.keys.read_u32(s.cache.keys.len() / 4) };
            for slot in [0, 2, 3, 4] {
                let start = (4 - slot) * page_stride * 16 * 512 / 2;
                let len = page_stride * 16 * 512 / 2;
                assert_eq!(&now[start..start + len], &keys[start..start + len]);
            }
            // Restore a prefix ending inside a block. Continuing the fork
            // needs the raw ring, not just normalized completed blocks/KV.
            s.reset(&d, 0).unwrap();
            run(&d, &mut s, &w, &streams, &[(0, 0), (0, 1), (0, 2)]);
            s.copy_slot(&d, 0, 1).unwrap();
            let t = &streams[0];
            let x = upload(&d, &t.input[3 * WIDTH..7 * WIDTH]);
            let y = d.alloc(4 * WIDTH * 4).unwrap();
            s.run(&d, &w, &(3..7).map(|p| (1, p)).collect::<Vec<_>>(), &x, &y)
                .unwrap();
            let probe = t.probes.iter().position(|&p| p == 3).unwrap();
            close(
                &unsafe { y.read_f32(0, WIDTH) },
                &t.output[probe * WIDTH..(probe + 1) * WIDTH],
                "incomplete-block fork",
            );
            eprintln!(
                "QSA layer={layer} chunk={chunk}: {round} mixed batches; exact selections; max output error={worst:e}; cache/reset/fork pass"
            );
        }
    }
    assert_eq!(d.allocated_bytes(), 0);
}
