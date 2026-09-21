use super::*;
use paddock_engine::generator::Generator;

fn upload(d: &MetalDevice, values: &[f32]) -> Buffer {
    d.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn close(a: &[f32], b: &[f32], limit: f32) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    let error = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("GPU max_abs={error}");
    assert!(error < limit, "{error} >= {limit}");
}
#[test]
fn routing_and_grouped_mxfp4() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    for experts in [32, 128] {
        let logits = upload(&d, &vec![0.; experts]);
        let bias = upload(&d, &vec![0.; experts]);
        let ids = d.alloc(16).unwrap();
        let probabilities = d.alloc(16).unwrap();
        let c = d.begin().unwrap();
        c.dispatch(
            "moe_route",
            &[&logits, &bias, &ids, &probabilities],
            &[experts as u32],
            [1, 1, 1],
            32,
        );
        c.finish().unwrap();
        assert_eq!(unsafe { ids.read_u32(4) }, [0, 1, 2, 3]);
        assert_eq!(unsafe { probabilities.read_f32(0, 4) }, [0.25; 4]);
        let invalid = upload(&d, &vec![f32::NAN; experts]);
        let c = d.begin().unwrap();
        c.dispatch(
            "moe_route",
            &[&invalid, &bias, &ids, &probabilities],
            &[experts as u32],
            [1, 1, 1],
            32,
        );
        c.finish().unwrap();
        assert!(
            unsafe { ids.read_u32(4) }
                .iter()
                .all(|&id| id < experts as u32)
        );
    }
    // A correct 32-expert route does not establish 120B coverage. Exercise
    // all 128 experts, unused experts, skewed lists, both BM variants and a
    // partial BK64 contraction. The oracle is a distinct GPU SIMD path.
    for experts in [32, 128] {
        for width in [64, 96] {
            for bm in [16, 32] {
                for concentrated in [false, true] {
                    grouped_mxfp4(&d, experts, width, bm, concentrated);
                }
            }
        }
    }
}

fn grouped_mxfp4(d: &MetalDevice, experts: usize, width: usize, bm: usize, concentrated: bool) {
    let (k, n) = (width, width);
    let w = |salt: usize| {
        let bytes = (0..experts * n * k / 32)
            .flat_map(|b| {
                std::iter::once(122u8)
                    .chain((0..16).map(move |i| ((b * 19 + i * 7 + salt) % 256) as u8))
            })
            .collect::<Vec<_>>();
        d.upload(&bytes).unwrap()
    };
    let gate = w(1);
    let up = w(3);
    let down = w(7);
    for rows in [1usize, 17, 33, 129, 512] {
        let x = upload(
            d,
            &(0..rows * k)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 32.)
                .collect::<Vec<_>>(),
        );
        let picks = (0..rows * 4)
            .map(|i| {
                // Four distinct experts per token in both cases. The sparse
                // case leaves most expert counts at zero, including expert 0.
                if concentrated {
                    (experts - 1 - i % 4) as u32
                } else {
                    ((i * 3 + i / 4) % experts) as u32
                }
            })
            .collect::<Vec<_>>();
        let ids = d
            .upload(
                &picks
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let lists = d.alloc(experts * rows * 4 * 4).unwrap();
        let counts = d.alloc(experts * 4).unwrap();
        let cap = (rows * 4).div_ceil(bm) + experts;
        let tiles = d.alloc((1 + 2 * cap) * 4).unwrap();
        let a = d.alloc(rows * 4 * n * 2 * 4).unwrap();
        let b = d.alloc(a.len()).unwrap();
        let da = d.alloc(rows * 4 * n * 4).unwrap();
        let db = d.alloc(da.len()).unwrap();
        let p = [k as u32, n as u32, rows as u32];
        let c = d.begin().unwrap();
        c.dispatch(
            "moe_align",
            &[&ids, &lists, &counts],
            &[(rows * 4) as u32],
            [experts, 1, 1],
            256,
        );
        c.dispatch(
            "moe_tiles",
            &[&counts, &tiles],
            &[experts as u32, bm as u32],
            [1, 1, 1],
            128,
        );
        c.dispatch(
            "moe_gu_decode",
            &[&gate, &up, &x, &ids, &a],
            &p,
            [n.div_ceil(4), rows * 4, 1],
            128,
        );
        c.dispatch(
            if bm == 32 {
                "moe_gu_grouped32"
            } else {
                "moe_gu_grouped"
            },
            &[&gate, &up, &x, &lists, &counts, &tiles, &b],
            &p,
            [n.div_ceil(MOE_NTILE) * 2, cap, 1],
            128,
        );
        c.dispatch(
            "moe_down_decode",
            &[&down, &a, &ids, &da],
            &p,
            [n.div_ceil(4), rows * 4, 1],
            128,
        );
        c.dispatch(
            if bm == 32 {
                "moe_down_grouped32"
            } else {
                "moe_down_grouped"
            },
            &[&down, &down, &a, &lists, &counts, &tiles, &db],
            &p,
            [n.div_ceil(MOE_NTILE), cap, 1],
            128,
        );
        c.finish().unwrap();
        eprintln!(
            "experts={experts} width={width} BM={bm} rows={rows} concentrated={concentrated}"
        );
        close(
            &unsafe { a.read_f32(0, rows * 4 * n * 2) },
            &unsafe { b.read_f32(0, rows * 4 * n * 2) },
            0.001,
        );
        close(
            &unsafe { da.read_f32(0, rows * 4 * n) },
            &unsafe { db.read_f32(0, rows * 4 * n) },
            0.001,
        );
        let counts = unsafe { counts.read_u32(experts) };
        let lists = unsafe { lists.read_u32(experts * rows * 4) };
        let tiles = unsafe { tiles.read_u32(1 + 2 * cap) };
        assert_eq!(counts.iter().sum::<u32>(), (rows * 4) as u32);
        let mut schedule = Vec::new();
        for (e, &count) in counts.iter().enumerate() {
            let wanted = picks
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == e as u32)
                .map(|(i, _)| i as u32)
                .collect::<Vec<_>>();
            assert_eq!(&lists[e * rows * 4..e * rows * 4 + count as usize], &wanted);
            for first in (0..count as usize).step_by(bm) {
                schedule.extend([e as u32, first as u32]);
            }
        }
        assert_eq!(tiles[0] as usize * 2, schedule.len());
        assert!(tiles[0] as usize <= cap);
        assert_eq!(&tiles[1..1 + schedule.len()], &schedule);
    }
}

#[test]
fn attention_sinks_windows_pages_and_empty_splits() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let m = 33usize;
    let q = upload(
        &d,
        &(0..m * QWIDTH)
            .map(|i| ((i * 13 % 63) as f32 - 31.) / 32.)
            .collect::<Vec<_>>(),
    );
    let kh = |salt: usize| {
        d.upload(
            &(0..512 * KVWIDTH)
                .flat_map(|i| {
                    half::f16::from_f32(((i * 17 + salt) % 127) as f32 / 64. - 1.).to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let k = kh(1);
    let v = kh(9);
    let sinks = upload(
        &d,
        &(0..64).map(|i| (i as f32 - 32.) / 8.).collect::<Vec<_>>(),
    );
    let meta = d.alloc(m * 8).unwrap();
    let pages = d.alloc(32 * 4).unwrap();
    let ids = d.alloc(m * 4).unwrap();
    let tiles = d.alloc(16).unwrap();
    let qh = d.alloc(m * QWIDTH * 2).unwrap();
    let a = d.alloc(m * QWIDTH * 4).unwrap();
    let b = d.alloc(a.len()).unwrap();
    let c = d.alloc(a.len()).unwrap();
    let parts = d.alloc(m * 64 * SPLITS * 66 * 4).unwrap();
    // SAFETY: fresh buffers, no submission in flight. A non-identity page
    // permutation catches accidental contiguous-address assumptions.
    unsafe {
        pages.write_u32(&(0..32).map(|i| (i * 7) % 32).collect::<Vec<_>>());
        ids.write_u32(&(0..m as u32).collect::<Vec<_>>());
        tiles.write_u32(&[0, 32, 32, 1]);
    }
    for first in [0u32, 127, 257] {
        for window in [0u32, 128] {
            unsafe {
                meta.write_u32(
                    &(0..m as u32)
                        .flat_map(|i| [0, first + i])
                        .collect::<Vec<_>>(),
                );
            }
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "linear_input",
                &[&q, &qh],
                &[QWIDTH as u32, 0, m as u32],
                [(m * QWIDTH).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "oss_attention_check",
                &[&q, &k, &v, &meta, &pages, &sinks, &a],
                &[32, window],
                [64, m, 1],
                32,
            );
            cmd.dispatch(
                "oss_attention_prefill",
                &[&qh, &k, &v, &meta, &pages, &b, &tiles, &sinks],
                &[64, 8, 32, 0.125f32.to_bits(), window],
                [64, 2, 1],
                128,
            );
            cmd.finish().unwrap();
            let reference = unsafe { a.read_f32(0, m * QWIDTH) };
            eprintln!("attention first={first} window={window}");
            close(&reference, &unsafe { b.read_f32(0, m * QWIDTH) }, 0.002);
            for splits in [1usize, 3, 16] {
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "oss_attention_decode",
                    &[
                        &q,
                        &k,
                        &v,
                        &meta,
                        &pages,
                        &ids,
                        &sinks,
                        if splits == 1 { &c } else { &parts },
                    ],
                    &[
                        64,
                        8,
                        32,
                        0.125f32.to_bits(),
                        m as u32,
                        splits as u32,
                        window,
                    ],
                    [8, m, splits],
                    128,
                );
                if splits > 1 {
                    cmd.dispatch(
                        "attention_gqa_merge64",
                        &[&parts, &c, &ids],
                        &[splits as u32, 64],
                        [m * 64, 1, 1],
                        32,
                    );
                }
                cmd.finish().unwrap();
                close(&reference, &unsafe { c.read_f32(0, m * QWIDTH) }, 0.00001);
            }
        }
    }
}

#[test]
#[ignore = "requires elected GPT-OSS GGUF in PADDOCK_GPT_OSS_TEST_MODEL and M5"]
fn real_model_lifecycle() {
    let path = std::env::var("PADDOCK_GPT_OSS_TEST_MODEL").unwrap();
    real_lifecycle(&path, 24, 32);
}

#[test]
#[ignore = "requires elected 120B GGUF in PADDOCK_GPT_OSS_120B_TEST_MODEL and sufficient M5 memory"]
fn real_120b_lifecycle() {
    let path = std::env::var("PADDOCK_GPT_OSS_120B_TEST_MODEL").unwrap();
    real_lifecycle(&path, 36, 128);
}

fn real_lifecycle(path: &str, layers: usize, experts: usize) {
    let map = paddock_models::mapped::MappedGguf::open(std::path::Path::new(&path)).unwrap();
    for (key, value) in &map.gguf().metadata {
        if key.starts_with("gpt-oss.") {
            eprintln!("{key}: {value:?}");
        }
    }
    for t in map
        .gguf()
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.0.") || !t.name.starts_with("blk."))
    {
        eprintln!("{} {:?} {:?}", t.name, t.dims, t.ggml_type);
    }
    drop(map);
    let path = std::path::Path::new(path);
    assert!(matches!(
        GptOss::load(path, 4096, 4, Some(1 << 20)),
        Err(MetalError::Memory(_))
    ));
    let mut m = GptOss::load(path, 4096, 4, None).unwrap();
    assert_eq!(
        m.layers.len(),
        layers,
        "the wrong checkpoint is not a coverage gate"
    );
    assert_eq!(m.experts, experts);
    let used = m.device.allocated_bytes();
    eprintln!(
        "weights={} KV={} scratch={} budget={}",
        m.weight_bytes,
        m.kv_bytes,
        used - m.weight_bytes - m.kv_bytes,
        m.device.budget_bytes()
    );
    // Cross the 512-row command boundary using real weights, then resume
    // from the retained full-page prefix (including all sliding layers).
    let boundary = (0..513).map(|i| 200 + (i % 41) as u32).collect::<Vec<_>>();
    let long = m.prefill(3, &boundary).unwrap();
    m.reset();
    let resumed = m.prefill(0, &boundary).unwrap();
    assert_eq!(m.take_prefill_reused(0), 512);
    close(&long, &resumed, 0.00001);
    m.reset();
    let tokens = (0..161).map(|i| 100 + (i % 31) as u32).collect::<Vec<_>>();
    let a = m.prefill(0, &tokens).unwrap();
    m.reset();
    let again = m.prefill(2, &tokens).unwrap();
    assert!(m.take_prefill_reused(2) >= 144);
    close(&a, &again, 0.3);
    let mixed = (0..4)
        .map(|i| (i, tokens.iter().map(|t| t + i as u32).collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    m.reset();
    for (slot, p) in &mixed {
        m.prefill_begin(*slot, p.clone()).unwrap();
    }
    let mut done = m.forward_mixed(&[], 31).unwrap().1;
    assert!(m.prefill_abort(1));
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 64).unwrap().1);
    }
    assert_eq!(done.len(), 3);
    assert!(
        done.iter()
            .all(|r| r.0 != 1 && r.1.iter().all(|v| v.is_finite()))
    );
    let decodes = done
        .iter()
        .map(|r| (r.0, 100, r.2 as u32))
        .collect::<Vec<_>>();
    assert_eq!(m.forward_mixed(&decodes, 0).unwrap().0.len(), 3 * VOCAB);
    m.release_inactive_slots(&[false; 4]);
    assert!(m.slots.iter().all(|s| s.history.is_empty()));
    for (slot, logits, _) in &done {
        let single = m.prefill(0, &mixed[*slot].1).unwrap();
        close(logits, &single, 0.1);
        let top = |xs: &[f32]| {
            xs.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(top(logits), top(&single), "mixed/single slot {slot}");
    }
    assert!(m.prefill(4, &tokens).is_err());
    assert!(m.prefill(0, &[]).is_err());
    assert!(m.prefill(0, &[VOCAB as u32]).is_err());
    assert!(m.prefill(0, &vec![100; 4097]).is_err());
    assert_eq!(used, m.device.allocated_bytes());
}
