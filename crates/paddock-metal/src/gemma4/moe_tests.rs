use super::*;

fn floats(d: &MetalDevice, values: impl IntoIterator<Item = f32>) -> Buffer {
    d.upload(
        &values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn ints(d: &MetalDevice, values: impl IntoIterator<Item = u32>) -> Buffer {
    d.upload(
        &values
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn q8(d: &MetalDevice, k: usize, n: usize, salt: usize) -> Weight {
    let bytes = (0..k * n / 32)
        .flat_map(|b| {
            let mut block = half::f16::from_f32(1. / 1024.).to_le_bytes().to_vec();
            block.extend((0..32).map(|i| (((b * 7 + i * 11 + salt) % 127) as i8 - 63) as u8));
            block
        })
        .collect::<Vec<_>>();
    Weight {
        buffer: d.upload(&bytes).unwrap(),
        ty: 8,
        k,
        n,
    }
}
fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    let error = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(error <= tolerance, "max error {error} > {tolerance}");
}

#[test]
fn router_ties_scales_and_nonuniform_selected_softmax() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let logits = floats(
        &d,
        (0..4 * EXPERTS).map(|i| match i / EXPERTS {
            0 => 0.,
            1 => {
                if i % EXPERTS == 103 {
                    2f32.ln()
                } else {
                    0.
                }
            }
            2 => {
                if i % EXPERTS == 127 {
                    10000.
                } else {
                    -10000.
                }
            }
            _ => f32::NAN,
        }),
    );
    let scale = floats(&d, (0..EXPERTS).map(|e| 1. + e as f32 / 128.));
    let ids = d.alloc(4 * ACTIVE * 4).unwrap();
    let weights = d.alloc(ids.len()).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "gmoe_route",
        &[&logits, &scale, &ids, &weights],
        &[],
        [4, 1, 1],
        32,
    );
    cmd.finish().unwrap();
    let ids = unsafe { ids.read_u32(4 * ACTIVE) };
    let weights = unsafe { weights.read_f32(0, 4 * ACTIVE) };
    assert_eq!(&ids[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(&ids[8..16], &[103, 0, 1, 2, 3, 4, 5, 6]);
    for (j, &value) in weights.iter().take(8).enumerate() {
        assert!((value - (1. + j as f32 / 128.) / 8.).abs() < 1e-7);
    }
    assert!((weights[8] - 2. / 9. * (1. + 103. / 128.)).abs() < 1e-6);
    assert_eq!(ids[16], 127);
    assert_eq!(weights[16], 1. + 127. / 128.);
    assert!(weights[17..24].iter().all(|&v| v == 0.));
    assert!(ids[24..].iter().all(|&v| v < 128));
    assert!(weights[24..].iter().all(|v| !v.is_finite()));
}

#[test]
fn fused_q8_grouped_matches_independent_gpu_columns_and_tail_guards() {
    let d = MetalDevice::new(Some(1024 << 20)).unwrap();
    // Real contraction dimensions, odd output tiles, sparse and concentrated
    // routing. Synthetic weight construction is not a CPU inference oracle.
    for (k, n, down) in [(2816usize, 704usize, false), (704, 65, true)] {
        let w = q8(&d, k, n * 128 * if down { 1 } else { 2 }, 3);
        for rows in [17usize, 63, 129] {
            for concentrated in [false, true] {
                let assignment = (0..rows * 8)
                    .map(|i| {
                        if concentrated {
                            (i % 8) as u32
                        } else {
                            ((i * 17 + i / 8 * 7) % 128) as u32
                        }
                    })
                    .collect::<Vec<_>>();
                let ids = ints(&d, assignment);
                let lists = d.alloc(128 * rows * 8 * 4).unwrap();
                let counts = d.alloc(128 * 4).unwrap();
                let tiles = d.alloc((1 + 2 * (rows * 8 / 16 + 129)) * 4).unwrap();
                let input = floats(
                    &d,
                    (0..rows * k * if down { 16 } else { 1 })
                        .map(|i| ((i * 13 + i / k * 3) % 53) as f32 / 53. - 0.5),
                );
                let len = rows * 8 * n * if down { 1 } else { 2 };
                let expected = d.alloc(len * 4).unwrap();
                let actual = floats(&d, (0..len + 128).map(|_| f32::NAN));
                let p = [k as u32, n as u32, rows as u32];
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    if down {
                        "qmoe_down_decode"
                    } else {
                        "gmoe_gu_decode"
                    },
                    &[&w.buffer, &input, &ids, &expected],
                    &p,
                    [n.div_ceil(4), rows * 8, 1],
                    128,
                );
                cmd.dispatch(
                    "moe_align",
                    &[&ids, &lists, &counts],
                    &[(rows * 8) as u32],
                    [128, 1, 1],
                    256,
                );
                cmd.finish().unwrap();
                for bm in [16usize, 32] {
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        "moe_tiles",
                        &[&counts, &tiles],
                        &[128, bm as u32],
                        [1, 1, 1],
                        256,
                    );
                    let kernel = match (down, bm) {
                        (false, 16) => "gmoe_gu_strict16",
                        (false, _) => "gmoe_gu_strict32",
                        (true, 16) => "qmoe_down_strict16",
                        _ => "qmoe_down_strict32",
                    };
                    cmd.dispatch(
                        kernel,
                        &[
                            &w.buffer, &w.buffer, &input, &lists, &counts, &tiles, &actual,
                        ],
                        &p,
                        [
                            n.div_ceil(32) * if down { 1 } else { 2 },
                            (rows * 8).div_ceil(bm) + 128,
                            1,
                        ],
                        128,
                    );
                    cmd.finish().unwrap();
                    close(
                        &unsafe { expected.read_f32(0, len) },
                        &unsafe { actual.read_f32(0, len) },
                        0.0001,
                    );
                    assert!(
                        unsafe { actual.read_f32(len, 128) }
                            .iter()
                            .all(|v| v.is_nan())
                    );
                }
            }
        }
    }
}

#[test]
fn strict_dense_projection_handles_2112_contraction_tail() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let w = q8(&d, 2112, 67, 11);
    for rows in [2usize, 7, 17, 31, 65] {
        let input = floats(
            &d,
            (0..rows * w.k).map(|i| ((i * 7 + i / w.k * 13) % 67) as f32 / 67. - 0.5),
        );
        let expected = d.alloc(rows * w.n * 4).unwrap();
        let actual = floats(&d, (0..(rows + 64) * w.n).map(|_| f32::NAN));
        let workspace = d.alloc(4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "linear_q8_r1",
            &[&w.buffer, &input, &expected],
            &[w.k as u32, w.n as u32, rows as u32, 8, 1f32.to_bits()],
            [w.n.div_ceil(4), rows, 1],
            128,
        );
        project(&cmd, &[(&w, &actual)], &input, rows, &workspace);
        cmd.finish().unwrap();
        close(
            &unsafe { expected.read_f32(0, rows * w.n) },
            &unsafe { actual.read_f32(0, rows * w.n) },
            0.0001,
        );
        assert!(
            unsafe { actual.read_f32(rows * w.n, 64 * w.n) }
                .iter()
                .all(|v| v.is_nan())
        );
    }
}

#[test]
fn fused_branch_norms_match_separate_gpu_rms_and_add() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let (width, rows) = (2816usize, 7usize);
    let left_values = (0..width * rows)
        .map(|i| ((i * 17 + i / width * 3) % 97) as f32 / 97. - 0.5)
        .collect::<Vec<_>>();
    let left = floats(&d, left_values.iter().copied());
    let right = floats(
        &d,
        (0..width * rows).map(|i| ((i * 11 + i / width * 7) % 71) as f32 / 71. - 0.5),
    );
    let shared = floats(&d, (0..width).map(|i| 0.5 + (i % 13) as f32 / 13.));
    let routed = floats(&d, (0..width).map(|i| 0.25 + (i % 23) as f32 / 23.));
    let expected = d.alloc(width * rows * 4).unwrap();
    let tmp = d.alloc(expected.len()).unwrap();
    let cmd = d.begin().unwrap();
    for (x, gamma, out) in [(&left, &shared, &expected), (&right, &routed, &tmp)] {
        cmd.dispatch(
            "rms",
            &[x, gamma, out],
            &[width as u32, 0, 1e-6f32.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    cmd.dispatch(
        "residual",
        &[&expected, &tmp],
        &[(rows * width) as u32, 1f32.to_bits()],
        [(rows * width).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "gmoe_branches",
        &[&left, &right, &shared, &routed],
        &[width as u32, 1e-6f32.to_bits()],
        [rows, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    close(
        &unsafe { expected.read_f32(0, rows * width) },
        &unsafe { left.read_f32(0, rows * width) },
        1e-6,
    );
}

#[test]
#[ignore = "requires elected Gemma 26B-A4B Q8 file; PADDOCK_GEMMA4_GGUF"]
fn elected_moe_geometry_budget_and_four_slot_ragged_lifecycle() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let path = Path::new(&path);
    for (ctx, batch, budget) in [
        (0, 4, None),
        (32769, 4, None),
        (4096, 0, None),
        (4096, 65, None),
        (4096, 4, Some(1 << 20)),
    ] {
        assert!(Gemma4::load(path, ctx, batch, budget).is_err());
    }
    let mut m = Gemma4::load(path, 4096, 4, None).unwrap();
    assert_eq!(m.layers.len(), 30);
    assert_eq!(m.width, 2816);
    assert!(m.layers.iter().all(|l| l.heads == 16 && l.moe.is_some()));
    // Thirty scalar layer scales live in Rust metadata after load; unlike
    // floating-point inference, reading these constants needs no GPU kernel.
    assert_eq!(m.weight_bytes, 26844036216 - 30 * 4);
    assert!(m.device.allocated_bytes() <= m.device.budget_bytes());
    eprintln!(
        "weights={} kv={} total={} grant={}",
        m.weight_bytes,
        m.kv_bytes,
        m.device.allocated_bytes(),
        m.device.budget_bytes()
    );
    let baseline = m.device.allocated_bytes();
    let map = paddock_models::mapped::MappedGguf::open(path).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let prompts=(0..4).map(|i|tok.encode(&format!("<bos><|turn>user\n{} What is {} plus 2? Number only.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>","The quiet river flows past the village. ".repeat(11+i*7),i+3)).unwrap()).collect::<Vec<_>>();
    let mut expected = Vec::new();
    for p in &prompts {
        m.reset();
        expected.push(m.forward_prefill(0, p).unwrap());
    }
    m.reset();
    for (i, p) in prompts.iter().enumerate() {
        m.prefill_begin(i, p.clone()).unwrap();
    }
    let mut done = Vec::new();
    for _ in 0..100 {
        let (_, v) = m.forward_mixed(&[], 127).unwrap();
        done.extend(v);
        if done.len() == 4 {
            break;
        }
    }
    assert_eq!(done.len(), 4);
    for (slot, logits, n) in done {
        assert_eq!(n, prompts[slot].len());
        close(&expected[slot], &logits, 0.08);
        let best = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(best(&expected[slot]), best(&logits));
    }
    m.reset();
    assert_eq!(baseline, m.device.allocated_bytes());
}
