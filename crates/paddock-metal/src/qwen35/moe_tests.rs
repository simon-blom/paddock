use super::*;

fn upload_u(d: &MetalDevice, values: &[u32]) -> Buffer {
    d.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn upload_f(d: &MetalDevice, values: &[f32]) -> Buffer {
    d.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
#[ignore = "GPU-only causal isolation on PADDOCK_METAL_QWEN_MOE_MODEL; not a parity gate"]
fn chunk_route_precision_diagnostic() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MOE_MODEL").expect("35B MoE Q8 GGUF");
    let mut model = Qwen35::load(Path::new(&path), 1024, 4, None).unwrap();
    assert_eq!(model.geometry, Geometry::MOE_35B);
    let prompt: Vec<u32> = (2000..2073).collect();
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    for precise in [false, true] {
        PRECISE_FOR_TEST.with(|v| v.set(precise));
        for grouped in [true, false] {
            GROUPED_FOR_TEST.with(|v| v.set(grouped));
            for serial_recurrence in [false, true] {
                model.diagnostic_serial_prefill = serial_recurrence;
                model.reset();
                while model.evict_checkpoint() {}
                let reference = model.forward_prefill(0, &prompt).unwrap();
                model.reset();
                while model.evict_checkpoint() {}
                model.prefill_begin(0, prompt.clone()).unwrap();
                let mut finished = Vec::new();
                for _ in 0..20 {
                    finished.extend(model.forward_mixed(&[], 7).unwrap().1);
                    if !finished.is_empty() {
                        break;
                    }
                }
                assert_eq!(finished.len(), 1);
                let actual = &finished[0].1;
                let error = reference
                    .iter()
                    .zip(actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "precise={precise} grouped={grouped} serial_recurrence={serial_recurrence}: max_abs={error} top={}/{}",
                    top(&reference),
                    top(actual)
                );
            }
        }
    }
    GROUPED_FOR_TEST.with(|v| v.set(true));
    PRECISE_FOR_TEST.with(|v| v.set(true));
}

#[test]
#[ignore = "requires the elected PADDOCK_METAL_QWEN_MOE_MODEL and an M5"]
fn elected_35b_loader_bounds_and_resident_ledger() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MOE_MODEL").expect("35B MoE Q8 GGUF");
    let path = Path::new(&path);
    assert!(matches!(
        Qwen35::load(path, 4096, 4, Some(1 << 20)),
        Err(MetalError::Memory(_))
    ));
    for (ctx, batch) in [(0, 4), (32769, 4), (4096, 0), (4096, 65)] {
        assert!(matches!(
            Qwen35::load(path, ctx, batch, None),
            Err(MetalError::Model(_))
        ));
    }
    let mut m = Qwen35::load(path, 4096, 4, None).unwrap();
    assert_eq!(m.geometry, Geometry::MOE_35B);
    assert_eq!(m.layers.len(), 40);
    assert!(m.layers.iter().all(|l| l.moe.is_some()));
    let bytes = m.device.allocated_bytes();
    let workspace = m.moe_scratch.as_ref().unwrap();
    assert_eq!(
        Workspace::bytes(CHUNK),
        [
            &workspace.logits,
            &workspace.shared_gate,
            &workspace.ids,
            &workspace.weights,
            &workspace.lists,
            &workspace.counts,
            &workspace.tiles,
            &workspace.gu,
            &workspace.out
        ]
        .iter()
        .map(|b| b.len() as u64)
        .sum::<u64>()
    );
    let p = [1000, 2000, 3000, 4000];
    assert!(
        m.forward_prefill(0, &p)
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
    m.forward(100).unwrap();
    m.release_inactive_slots(&[false; 4]);
    assert_eq!(m.device.allocated_bytes(), bytes);
    eprintln!(
        "35B weights={} KV/recurrent={} scratch={} total={} grant={}",
        m.weight_bytes,
        m.kv_bytes,
        bytes - m.weight_bytes - m.kv_bytes,
        bytes,
        m.device.budget_bytes()
    );
}

#[test]
fn routing_ties_extremes_compaction_and_shared_fold() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let rows = 4;
    let mut scores = vec![0.; rows * EXPERTS];
    scores[EXPERTS..2 * EXPERTS].fill(-10000.);
    scores[EXPERTS + 255] = 10000.;
    scores[2 * EXPERTS + 255] = std::f32::consts::LN_2;
    scores[3 * EXPERTS..].fill(f32::NAN);
    let logits = upload_f(&d, &scores);
    let ids = d.alloc(rows * ACTIVE * 4).unwrap();
    let weights = d.alloc(ids.len()).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "qmoe_route",
        &[&logits, &ids, &weights],
        &[],
        [rows, 1, 1],
        32,
    );
    cmd.finish().unwrap();
    let selected = unsafe { ids.read_u32(rows * ACTIVE) };
    let probabilities = unsafe { weights.read_f32(0, rows * ACTIVE) };
    assert_eq!(&selected[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(&probabilities[..8], &[0.125; 8]);
    assert_eq!(selected[8], 255);
    assert_eq!(probabilities[8], 1.);
    assert!(probabilities[9..16].iter().all(|&v| v == 0.));
    assert!(selected.iter().all(|&i| i < EXPERTS as u32));
    assert_eq!(&selected[16..24], &[255, 0, 1, 2, 3, 4, 5, 6]);
    for (actual, expected) in probabilities[16..24].iter().zip([
        2. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 9.,
    ]) {
        assert!((actual - expected).abs() < 1e-7);
    }
    assert!(probabilities[24..].iter().all(|v| !v.is_finite()));

    // A known constant expert output is also a GPU fold witness: normalized
    // equal routes preserve 2, and a zero shared gate halves the shared 6.
    let out = upload_f(&d, &[2.; 8 * 64]);
    let shared = upload_f(&d, &[0.]);
    let delta = upload_f(&d, &[6.; 64]);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "qmoe_fold",
        &[&out, &weights, &shared, &delta],
        &[64, 1],
        [1, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { delta.read_f32(0, 64) }, vec![5.; 64]);
}

#[test]
fn grouped_q8_matches_gpu_simd_with_sparse_concentrated_and_ragged_routes() {
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    for width in [64usize, 96] {
        let bytes = |salt: usize| {
            let mut b = Vec::new();
            for block in 0..EXPERTS * width * width / 32 {
                b.extend(half::f16::from_f32(1. / 64.).to_le_bytes());
                b.extend((0..32).map(|j| (((j * salt + block * 3) % 17) as i8 - 8) as u8));
            }
            b
        };
        let gate = d.upload(&bytes(3)).unwrap();
        let up = d.upload(&bytes(7)).unwrap();
        for rows in [1usize, 17, 33, 129, 512] {
            for concentrated in [false, true] {
                let selected = (0..rows * ACTIVE)
                    .map(|i| {
                        if concentrated {
                            (255 - i % 8) as u32
                        } else {
                            ((i * 17 + i / 8) % EXPERTS) as u32
                        }
                    })
                    .collect::<Vec<_>>();
                let ids = upload_u(&d, &selected);
                let x = upload_f(
                    &d,
                    &(0..rows * width)
                        .map(|i| ((i * 7 % 13) as f32 - 6.) / 128.)
                        .collect::<Vec<_>>(),
                );
                let lists = d.alloc(EXPERTS * rows * ACTIVE * 4).unwrap();
                let counts = d.alloc(EXPERTS * 4).unwrap();
                let tiles = d
                    .alloc((1 + 2 * ((rows * ACTIVE).div_ceil(16) + EXPERTS)) * 4)
                    .unwrap();
                let vector = d.alloc(rows * ACTIVE * width * 2 * 4).unwrap();
                let tensor = d.alloc(vector.len()).unwrap();
                let p = [width as u32, width as u32, rows as u32];
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "moe_align",
                    &[&ids, &lists, &counts],
                    &[(rows * ACTIVE) as u32],
                    [EXPERTS, 1, 1],
                    256,
                );
                cmd.dispatch(
                    "qmoe_gu_decode",
                    &[&gate, &up, &x, &ids, &vector],
                    &p,
                    [width.div_ceil(4), rows * ACTIVE, 1],
                    128,
                );
                cmd.finish().unwrap();
                let entries = unsafe { lists.read_u32(EXPERTS * rows * ACTIVE) };
                let sizes = unsafe { counts.read_u32(EXPERTS) };
                for e in 0..EXPERTS {
                    let expected = selected
                        .iter()
                        .enumerate()
                        .filter(|(_, id)| **id == e as u32)
                        .map(|(i, _)| i as u32)
                        .collect::<Vec<_>>();
                    assert_eq!(sizes[e] as usize, expected.len());
                    assert_eq!(
                        &entries[e * rows * ACTIVE..e * rows * ACTIVE + expected.len()],
                        expected
                    );
                }
                let expected = unsafe { vector.read_f32(0, rows * ACTIVE * width * 2) };
                for (tile, strict) in [(16usize, false), (32, false), (16, true), (32, true)] {
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        "moe_tiles",
                        &[&counts, &tiles],
                        &[EXPERTS as u32, tile as u32],
                        [1, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        if strict && tile == 16 {
                            "qmoe_gu_strict16"
                        } else if strict {
                            "qmoe_gu_strict32"
                        } else if tile == 16 {
                            "qmoe_gu_grouped16"
                        } else {
                            "qmoe_gu_grouped32"
                        },
                        &[&gate, &up, &x, &lists, &counts, &tiles, &tensor],
                        &p,
                        [
                            width.div_ceil(if strict { 32 } else { 64 }) * 2,
                            (rows * ACTIVE).div_ceil(tile) + EXPERTS,
                            1,
                        ],
                        128,
                    );
                    cmd.finish().unwrap();
                    assert_eq!(
                        unsafe { tensor.read_f32(0, expected.len()) },
                        expected,
                        "gate/up width={width} rows={rows} tile={tile} concentrated={concentrated}"
                    );
                    let schedule =
                        unsafe { tiles.read_u32(1 + 2 * ((rows * ACTIVE).div_ceil(16) + EXPERTS)) };
                    assert_eq!(
                        schedule[0] as usize,
                        sizes
                            .iter()
                            .map(|&n| (n as usize).div_ceil(tile))
                            .sum::<usize>()
                    );
                    let mut offset = 0;
                    for (e, &count) in sizes.iter().enumerate() {
                        for first in (0..count as usize).step_by(tile) {
                            assert_eq!(
                                &schedule[1 + offset * 2..3 + offset * 2],
                                &[e as u32, first as u32]
                            );
                            offset += 1;
                        }
                    }
                    // Independent GPU SIMD consumes exactly the same routed
                    // activation values. No host expert arithmetic oracle.
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        "qmoe_down_decode",
                        &[&gate, &vector, &ids, &tensor],
                        &p,
                        [width.div_ceil(4), rows * ACTIVE, 1],
                        128,
                    );
                    cmd.finish().unwrap();
                    let down = unsafe { tensor.read_f32(0, rows * ACTIVE * width) };
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        if strict && tile == 16 {
                            "qmoe_down_strict16"
                        } else if strict {
                            "qmoe_down_strict32"
                        } else if tile == 16 {
                            "qmoe_down_grouped16"
                        } else {
                            "qmoe_down_grouped32"
                        },
                        &[&gate, &up, &vector, &lists, &counts, &tiles, &tensor],
                        &p,
                        [
                            width.div_ceil(if strict { 32 } else { 64 }),
                            (rows * ACTIVE).div_ceil(tile) + EXPERTS,
                            1,
                        ],
                        128,
                    );
                    cmd.finish().unwrap();
                    let actual = unsafe { tensor.read_f32(0, down.len()) };
                    let error = down
                        .iter()
                        .zip(&actual)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        actual.iter().all(|v| v.is_finite()) && error < 0.00001,
                        "down error {error}"
                    );
                }
            }
        }
    }
}
