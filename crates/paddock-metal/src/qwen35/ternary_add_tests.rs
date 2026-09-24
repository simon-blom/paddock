use super::*;
use paddock_kernels::reference::ternary::encode_ptq1_0;

thread_local! {
    pub(super) static ADD_PROJECTIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_UNPACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_PTQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(super) fn election(ptq: bool, rows: usize) -> Option<(&'static str, usize, usize)> {
    if !ADD_PROJECTIONS.with(|v| v.get()) {
        return None;
    }
    let names = if ptq {
        [
            "ptq1_add_full_1",
            "ptq1_add_full_2",
            "ptq1_add_full_3",
            "ptq1_add_full_4",
        ]
    } else {
        [
            "bonsai_add_full_1",
            "bonsai_add_full_2",
            "bonsai_add_full_3",
            "bonsai_add_full_4",
        ]
    };
    let (name, tile) = if rows <= 4 {
        (names[rows - 1], rows)
    } else if rows.is_multiple_of(4) {
        (names[3], 4)
    } else {
        (add(ptq, 4), 4)
    };
    Some((name, if ptq { 4 } else { 16 }, tile))
}
struct AddGuard;
impl AddGuard {
    fn new() -> Self {
        ADD_PROJECTIONS.with(|v| assert!(!v.replace(true)));
        Self
    }
}
impl Drop for AddGuard {
    fn drop(&mut self) {
        ADD_PROJECTIONS.with(|v| v.set(false));
    }
}
#[test]
#[ignore = "requires Bonsai MLX and the independent complete-generation reference"]
fn ternary_add_bonsai_complete_generations() {
    let _guard = AddGuard::new();
    super::bonsai_tests::bonsai_complete_generation_reference();
}
#[test]
#[ignore = "requires Bonsai PTQ1 and the independent complete-generation reference"]
fn ternary_add_ptq1_complete_generations() {
    let _guard = AddGuard::new();
    super::ternary_tests::ptq1_complete_generations_batch_and_cache();
}

#[test]
#[ignore = "requires Bonsai PTQ1; paired full-model decode diagnostic, not HTTP throughput"]
fn ptq1_add_execution_cost() {
    execution_cost(false);
}
#[test]
#[ignore = "requires Bonsai PTQ1; bit-exact full-model packed-unpack comparison"]
fn ptq1_unpack_execution_cost() {
    execution_cost(true);
}
fn execution_cost(unpack_comparison: bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            BASELINE_PTQ.with(|v| v.set(false));
            BASELINE_UNPACK.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_METAL_PTQ1_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 1024, 4, None).unwrap();
    let mut signatures = std::collections::BTreeMap::new();
    for round in 0..4 {
        for live in if unpack_comparison {
            vec![1, 2, 3, 4]
        } else {
            vec![1, 4]
        } {
            for turn in 0..2 {
                let baseline = (round + turn) % 2 == 0;
                BASELINE_PTQ.with(|v| v.set(baseline && !unpack_comparison));
                BASELINE_UNPACK.with(|v| v.set(baseline && unpack_comparison));
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                for slot in 0..live {
                    model
                        .prefill_begin(
                            slot,
                            (0..128)
                                .map(|i| 1000 + ((i + slot * 17) % 500) as u32)
                                .collect(),
                        )
                        .unwrap();
                }
                let start = std::time::Instant::now();
                while !model.pending.is_empty() {
                    model.forward_mixed(&[], 512).unwrap();
                }
                let ttft = start.elapsed().as_secs_f64();
                let mut gpu = Vec::new();
                let mut wall = Vec::new();
                let mut hash = blake3::Hasher::new();
                for step in 0..32 {
                    let rows = (0..live)
                        .map(|slot| (slot, 7000 + step, model.slots[slot].history.len() as u32))
                        .collect::<Vec<_>>();
                    let start = std::time::Instant::now();
                    let (values, _) = model.forward_mixed(&rows, 0).unwrap();
                    wall.push(start.elapsed().as_secs_f64());
                    gpu.push(model.last_gpu_seconds);
                    for value in values {
                        assert!(value.is_finite());
                        hash.update(&value.to_bits().to_le_bytes());
                    }
                }
                let signature = hash.finalize().to_hex().to_string();
                let expected = signatures
                    .entry((live, baseline && !unpack_comparison))
                    .or_insert_with(|| signature.clone());
                eprintln!(
                    "PTQ1_ADD_EXECUTION {}",
                    serde_json::json!({"round":round,"live":live,"baseline":baseline,"unpack_comparison":unpack_comparison,"prompt_tokens":128,"ttft_s":ttft,"gpu_s":gpu,"wall_s":wall,"allocated_bytes":model.device.allocated_bytes(),"logits_blake3":signature,"exact":&signature==expected})
                );
                assert_eq!(&signature, expected, "baseline={baseline} live={live}");
            }
        }
    }
}

fn upload(device: &MetalDevice, values: &[f32]) -> Buffer {
    device
        .upload_parts(&[&values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()])
        .unwrap()
}
fn weights(k: usize, n: usize, ptq: bool) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    let mut dense = Vec::new();
    let mut seed = 41u32;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for _ in 0..k * n / 128 {
        let codes: Vec<i8> = (0..128).map(|_| (random() % 3) as i8 - 1).collect();
        let scale = half::f16::from_f32((random() % 100 + 1) as f32 / 8192.);
        dense.extend(codes.iter().map(|c| f32::from(*c) * scale.to_f32()));
        if ptq {
            bytes.extend(encode_ptq1_0(&codes, scale));
        } else {
            for part in codes.chunks_exact(16) {
                bytes.extend(
                    part.iter()
                        .enumerate()
                        .fold(0u32, |w, (j, c)| w | ((*c + 1) as u32) << (j * 2))
                        .to_le_bytes(),
                );
            }
            scales.extend(scale.to_le_bytes());
        }
    }
    bytes.extend(scales);
    (bytes, dense)
}
fn add(ptq: bool, r: usize) -> &'static str {
    match (ptq, r) {
        (false, 1) => "bonsai_add_1",
        (false, 2) => "bonsai_add_2",
        (false, 3) => "bonsai_add_3",
        (false, 4) => "bonsai_add_4",
        (false, 8) => "bonsai_add_8",
        (true, 1) => "ptq1_add_1",
        (true, 2) => "ptq1_add_2",
        (true, 3) => "ptq1_add_3",
        (true, 4) => "ptq1_add_4",
        (true, 8) => "ptq1_add_8",
        _ => unreachable!(),
    }
}

fn full(ptq: bool, rows: usize) -> &'static str {
    match (ptq, rows) {
        (false, 1) => "bonsai_add_full_1",
        (false, 2) => "bonsai_add_full_2",
        (false, 3) => "bonsai_add_full_3",
        (false, 4) => "bonsai_add_full_4",
        (true, 1) => "ptq1_add_full_1",
        (true, 2) => "ptq1_add_full_2",
        (true, 3) => "ptq1_add_full_3",
        (true, 4) => "ptq1_add_full_4",
        _ => unreachable!(),
    }
}
fn unpack(rows: usize) -> &'static str {
    match rows {
        1 => "ptq1_unroll_1",
        2 => "ptq1_unroll_2",
        3 => "ptq1_unroll_3",
        4 => "ptq1_unroll_4",
        _ => unreachable!(),
    }
}
#[test]
fn ptq1_triplet_exhausts_all_byte_combinations() {
    let device = MetalDevice::new(None).unwrap();
    let out = device.upload_parts(&[&0u32.to_le_bytes()]).unwrap();
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "ptq1_triplet_check",
        &[&out],
        &[],
        [(1 << 24) / 256, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { out.read_f32(0, 1) }[0].to_bits(), 0);
}

#[test]
fn ptq1_unpack_ragged_spans_preserve_exact_outputs_and_guards() {
    let device = MetalDevice::new(None).unwrap();
    let (k, n, m) = (512usize, 5121usize, 19usize);
    let (bytes, _) = weights(k, n, true);
    let w = device.upload_parts(&[&bytes]).unwrap();
    let x = upload(
        &device,
        &(0..k * m)
            .map(|i| ((i * 139 % 997) as f32 - 498.) / 317.31)
            .collect::<Vec<_>>(),
    );
    let expected = upload(&device, &vec![12345.; m * n + 16]);
    let actual = upload(&device, &vec![12345.; m * n + 16]);
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "ptq1_add_full_1",
        &[&w, &x, &expected],
        &[k as u32, n as u32, m as u32],
        [n.div_ceil(4), m, 1],
        128,
    );
    for (first, count) in [(0, 1), (1, 2), (3, 3), (6, 4), (10, 5), (15, 4)] {
        let (kernel, tile) = ternary::decode_kernel(k, n, count);
        cmd.dispatch_at(
            kernel,
            &[&w, &x, &actual],
            &[0, first * k * 4, first * n * 4],
            &[k as u32, n as u32, count as u32],
            [n.div_ceil(4), count / tile, 1],
            128,
        );
    }
    cmd.finish().unwrap();
    let expected = unsafe { expected.read_f32(0, m * n + 16) };
    let actual = unsafe { actual.read_f32(0, m * n + 16) };
    assert!(actual[m * n..].iter().all(|v| *v == 12345.));
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
#[test]
fn ternary_add_projections_are_accurate_bounded_and_batch_stable() {
    let device = MetalDevice::new(None).unwrap();
    for (k, n, m) in [
        (128usize, 1usize, 1usize),
        (384, 19, 12),
        (512, 35, 33),
        (1024, 48, 17),
    ] {
        let input = (0..k * m)
            .map(|i| ((i * 139 % 997) as f32 - 498.) / 317.31)
            .collect::<Vec<_>>();
        let x = upload(&device, &input);
        for ptq in [false, true] {
            let (bytes, dense) = weights(k, n, ptq);
            let w = device.upload_parts(&[&bytes]).unwrap();
            let mut exact = Vec::new();
            for (name, r) in [1, 2, 3, 4, 8]
                .into_iter()
                .map(|r| (add(ptq, r), r))
                .chain(
                    [1, 2, 3, 4]
                        .into_iter()
                        .filter(|r| m.is_multiple_of(*r))
                        .map(|r| (full(ptq, r), r)),
                )
                .chain(
                    [1, 2, 3, 4]
                        .into_iter()
                        .filter(|r| ptq && m.is_multiple_of(*r))
                        .map(|r| (unpack(r), r)),
                )
                .chain(ptq.then_some(("ptq1_swar_1", 1)))
            {
                let out = upload(&device, &vec![12345.; m * n + 16]);
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    name,
                    &[&w, &x, &out],
                    &[k as u32, n as u32, m as u32],
                    [n.div_ceil(if ptq { 4 } else { 16 }), m.div_ceil(r), 1],
                    128,
                );
                cmd.finish().unwrap();
                let actual = unsafe { out.read_f32(0, m * n + 16) };
                assert!(actual[m * n..].iter().all(|v| *v == 12345.));
                if exact.is_empty() {
                    for row in 0..m {
                        for col in 0..n {
                            let reference = (0..k)
                                .map(|j| input[row * k + j] as f64 * dense[col * k + j] as f64)
                                .sum::<f64>();
                            assert!(
                                (actual[row * n + col] as f64 - reference).abs() < 0.00001,
                                "{name} k={k} n={n} m={m} row={row} col={col}"
                            );
                        }
                    }
                    exact = actual;
                } else {
                    assert_eq!(exact, actual, "{name} k={k} n={n} m={m}");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Bonsai PTQ1; real checkpoint tensor unpacking/row-reuse diagnostic"]
fn ptq1_unpack_checkpoint_cost() {
    let device = MetalDevice::new(None).unwrap();
    let path = std::env::var("PADDOCK_METAL_PTQ1_MODEL").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    for tensor in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.3.attn_k.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_out.weight",
        "output.weight",
    ] {
        let (info, bytes) = map.tensor_bytes(tensor).unwrap();
        assert_eq!(info.raw_type, 143);
        let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
        let w = device.upload(bytes).unwrap();
        let x = upload(
            &device,
            &(0..k * 4)
                .map(|i| ((i * 139 % 997) as f32 - 498.) / 317.31)
                .collect::<Vec<_>>(),
        );
        for m in [1usize, 2, 3, 4] {
            let out = device.alloc(n * m * 4).unwrap();
            let variants = [("ptq1_add_full_1", 1), ("ptq1_swar_1", 1), (unpack(m), m)];
            let mut exact = Vec::new();
            for round in 0..6 {
                for step in 0..variants.len() {
                    let trial = (round + step) % variants.len();
                    let (name, r) = variants[trial];
                    let cmd = device.begin().unwrap();
                    for _ in 0..5 {
                        cmd.dispatch(
                            name,
                            &[&w, &x, &out],
                            &[k as u32, n as u32, m as u32],
                            [n.div_ceil(4), m / r, 1],
                            128,
                        );
                    }
                    let ms = cmd.finish().unwrap() * 1000. / 5.;
                    let values = unsafe { out.read_f32(0, m * n) };
                    if exact.is_empty() {
                        assert!(values.iter().all(|v| v.is_finite()));
                        exact = values;
                    } else {
                        assert_eq!(values, exact, "{name} {tensor} m={m}");
                    }
                    if round > 0 {
                        eprintln!(
                            "PTQ1_CHECKPOINT_COST {}",
                            serde_json::json!({"tensor":tensor,"k":k,"n":n,"m":m,"name":name,"round":round,"ms":ms})
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "real-shape paired GPU diagnostic, not serving performance"]
fn ternary_add_projection_cost() {
    let device = MetalDevice::new(None).unwrap();
    for (k, n) in [(5120usize, 17408usize), (17408, 5120), (5120, 1024)] {
        let input = (0..k * 32)
            .map(|i| ((i * 139 % 997) as f32 - 498.) / 317.31)
            .collect::<Vec<_>>();
        let x = upload(&device, &input);
        for ptq in [false, true] {
            let (bytes, _) = weights(k, n, ptq);
            let w = device.upload_parts(&[&bytes]).unwrap();
            for m in [1usize, 4, 32] {
                let out = device.alloc(n * m * 4).unwrap();
                let base = match (ptq, m) {
                    (false, 1) => ("bonsai_full1", 16, 1),
                    (false, 4) => ("bonsai_full4", 16, 4),
                    (false, _) => ("bonsai_prefill32", 32, 32),
                    (true, 1) => ("ptq1_vectors1", 4, 1),
                    (true, 4) => ("ptq1_vectors4", 4, 4),
                    (true, _) => ("ptq1_mm32", 32, 32),
                };
                let cols = if ptq { 4 } else { 16 };
                let variants = [
                    base,
                    (full(ptq, m.min(4)), cols, m.min(4)),
                    (full(ptq, 1), cols, 1),
                ];
                for round in 0..6 {
                    for step in 0..variants.len() {
                        let trial = (step + round) % variants.len();
                        let (name, c, r) = variants[trial];
                        let cmd = device.begin().unwrap();
                        for _ in 0..5 {
                            cmd.dispatch(
                                name,
                                &[&w, &x, &out],
                                &[k as u32, n as u32, m as u32, bonsai::AFFINE2],
                                [n.div_ceil(c), m.div_ceil(r), 1],
                                128,
                            );
                        }
                        let ms = cmd.finish().unwrap() * 1000. / 5.;
                        if round > 0 {
                            eprintln!(
                                "TERNARY_ADD_COST {}",
                                serde_json::json!({"ptq":ptq,"k":k,"n":n,"m":m,"round":round,"trial":trial,"name":name,"ms":ms})
                            );
                        }
                    }
                }
            }
        }
    }
}
