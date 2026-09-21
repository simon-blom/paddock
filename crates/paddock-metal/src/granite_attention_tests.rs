//! GPU-only witnesses for Granite 3B's distinct hd64/GQA5 geometry. Synthetic
//! buffers exercise addressing; only GPU results are compared on the host.
use crate::device::{Buffer, MetalDevice};

fn words(device: &MetalDevice, data: &[u32]) -> Buffer {
    device
        .upload(
            &data
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .expect("upload GPU fixture words")
}

#[test]
fn hd128_compact_prefill_matches_gpu_reference() {
    let device = MetalDevice::new(Some(16 << 20)).unwrap();
    let (heads, kv_heads, stride, rows) = (8usize, 2usize, 19usize, 43usize);
    let width = heads * 128;
    let q = device
        .upload(
            &(0..rows * width)
                .flat_map(|i| (((i * 17 % 31) as f32 - 15.0) / 16.0).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let fixture = |factor, modulus, divisor| {
        device
            .upload(
                &(0..stride * 16 * kv_heads * 128)
                    .flat_map(|i| {
                        half::f16::from_f32(
                            ((i * factor % modulus) as f32 - (modulus / 2) as f32) / divisor,
                        )
                        .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    };
    let keys = fixture(7, 43, 16.0);
    let values = fixture(11, 47, 32.0);
    let pages = words(
        &device,
        &(0..2 * stride)
            .map(|i| (i * 7 % stride) as u32)
            .collect::<Vec<_>>(),
    );
    let meta = words(
        &device,
        &(0..rows)
            .flat_map(|i| {
                if i < 20 {
                    [0, i as u32]
                } else {
                    [1, 250 + i as u32]
                }
            })
            .collect::<Vec<_>>(),
    );
    let reference = device.alloc(rows * width * 4).unwrap();
    let actual = words(&device, &vec![f32::NAN.to_bits(); (rows + 3) * width]);
    let qhalf = device.alloc((rows + 32) * width * 2).unwrap();
    let tiles = words(&device, &[20, 23, 3, 17]);
    let scale = (1.0f32 / 128.0).to_bits();
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "attention",
        &[&q, &keys, &values, &meta, &pages, &reference],
        &[
            heads as u32,
            kv_heads as u32,
            128,
            0,
            stride as u32,
            scale,
            1,
        ],
        [heads, rows, 1],
        32,
    );
    cmd.dispatch(
        "attention_query",
        &[&q, &qhalf],
        &[width as u32, 0, rows as u32],
        [((rows + 32) * width).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "granite_attention_prefill128",
        &[&qhalf, &keys, &values, &meta, &pages, &actual, &tiles],
        &[heads as u32, kv_heads as u32, stride as u32, scale],
        [heads, 2, 1],
        128,
    );
    cmd.finish().unwrap();
    let expected = unsafe { reference.read_f32(3 * width, (rows - 3) * width) };
    let got = unsafe { actual.read_f32(3 * width, expected.len()) };
    assert!(expected.iter().chain(&got).all(|x| x.is_finite()));
    let error = expected
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(error < 1e-3, "max attention error {error}");
    for offset in [0, rows * width] {
        assert!(
            unsafe { actual.read_f32(offset, 3 * width) }
                .iter()
                .all(|x| x.is_nan()),
            "output guard at {offset}"
        );
    }
}

#[test]
fn hd64_gqa5_ragged_prefill_split_decode_and_guards() {
    let device = MetalDevice::new(Some(16 << 20)).unwrap();
    let (heads, kv_heads, stride, rows) = (10usize, 2usize, 19usize, 43usize);
    let width = heads * 64;
    let q = device
        .upload(
            &(0..rows * width)
                .flat_map(|i| (((i * 17 % 31) as f32 - 15.0) / 16.0).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let fixture = |factor, modulus, divisor| {
        device
            .upload(
                &(0..stride * 16 * kv_heads * 64)
                    .flat_map(|i| {
                        half::f16::from_f32(
                            ((i * factor % modulus) as f32 - (modulus / 2) as f32) / divisor,
                        )
                        .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    };
    let keys = fixture(7, 43, 16.0);
    let values = fixture(11, 47, 32.0);
    let pages = words(
        &device,
        &(0..2 * stride)
            .map(|i| (i * 7 % stride) as u32)
            .collect::<Vec<_>>(),
    );
    // Fresh short rows, an internal nonzero start, a second logical sequence
    // and a ragged final tile. The two slots share permuted physical pages.
    let meta = words(
        &device,
        &(0..rows)
            .flat_map(|i| {
                if i < 20 {
                    [0, i as u32]
                } else {
                    [1, 250 + i as u32]
                }
            })
            .collect::<Vec<_>>(),
    );
    let reference = device.alloc(rows * width * 4).unwrap();
    let sentinel = vec![f32::NAN.to_bits(); (rows + 3) * width];
    let actual = words(&device, &sentinel);
    let parts = device.alloc(rows * heads * 32 * 66 * 4).unwrap();
    let selected: Vec<_> = (3..rows as u32).rev().collect();
    let row_map = words(&device, &selected);
    let p = [
        heads as u32,
        kv_heads as u32,
        stride as u32,
        (1.0f32 / 64.0).to_bits(),
    ];
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "attention_check64",
        &[&q, &keys, &values, &meta, &pages, &reference],
        &p,
        [heads, rows, 1],
        32,
    );
    cmd.finish().unwrap();
    let expected = unsafe { reference.read_f32(3 * width, (rows - 3) * width) };
    let compare = |tolerance| {
        let got = unsafe { actual.read_f32(3 * width, expected.len()) };
        assert!(expected.iter().chain(&got).all(|x| x.is_finite()));
        let error = expected
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < tolerance, "max attention error {error}");
        for offset in [0, rows * width] {
            assert!(
                unsafe { actual.read_f32(offset, 3 * width) }
                    .iter()
                    .all(|x| x.is_nan()),
                "output guard at {offset}"
            );
        }
    };
    for splits in [1, 2, 7, 32] {
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "attention_gqa5_64",
            &[
                &q,
                &keys,
                &values,
                &meta,
                &pages,
                &row_map,
                if splits == 1 { &actual } else { &parts },
            ],
            &[p[0], p[1], p[2], p[3], selected.len() as u32, splits as u32],
            [kv_heads, selected.len(), splits],
            128,
        );
        if splits > 1 {
            cmd.dispatch(
                "attention_gqa_merge64",
                &[&parts, &actual, &row_map],
                &[splits as u32, heads as u32],
                [selected.len() * heads, 1, 1],
                32,
            );
        }
        cmd.finish().unwrap();
        compare(1e-5);
    }
    // Reset sentinels so a missing prefill write cannot inherit decode output.
    unsafe { actual.write_u32(&sentinel) };
    let qhalf = device.alloc((rows + 32) * width * 2).unwrap();
    let tiles = words(&device, &[20, 23, 3, 17]);
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "attention_query",
        &[&q, &qhalf],
        &[width as u32, 0, rows as u32],
        [((rows + 32) * width).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "attention_prefill_batched64",
        &[&qhalf, &keys, &values, &meta, &pages, &actual, &tiles],
        &p,
        [heads, 2, 1],
        128,
    );
    cmd.finish().unwrap();
    compare(1e-3);
}
