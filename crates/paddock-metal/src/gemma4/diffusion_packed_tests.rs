use super::*;

fn floats(d: &MetalDevice, n: usize) -> Buffer {
    d.upload(
        &(0..n)
            .flat_map(|i| (((i * 31 % 257) as f32 - 128.) / 256.).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn packed(d: &MetalDevice, ty: u32, k: usize, n: usize) -> Weight {
    let total = k * n;
    let mut bytes = Vec::new();
    if matches!(ty, 0x100 | 0x108) {
        bytes.extend((0..total / if ty == 0x100 { 2 } else { 1 }).map(|i| (i * 37 + 11) as u8));
        for b in [0x3b80u16, 0xbe00] {
            bytes.extend((0..total / 64).flat_map(|_| b.to_le_bytes()));
        }
    } else {
        let (block, size) = match ty {
            6 => (32, 22),
            8 => (32, 34),
            12 => (256, 144),
            14 => (256, 210),
            _ => unreachable!(),
        };
        for b in 0..total / block {
            let mut data: Vec<u8> = (0..size).map(|i| (b * 53 + i * 17 + 7) as u8).collect();
            let offset = if ty == 14 { 208 } else { 0 };
            data[offset..offset + 2]
                .copy_from_slice(&half::f16::from_f32(1. / 1024.).to_le_bytes());
            if ty == 12 {
                data[2..4].copy_from_slice(&half::f16::from_f32(1. / 2048.).to_le_bytes());
            }
            bytes.extend(data);
        }
    }
    Weight {
        buffer: d.upload(&bytes).unwrap(),
        ty,
        k,
        n,
    }
}
fn equal(a: &Buffer, b: &Buffer, n: usize) {
    let a = unsafe { a.read_f32(0, n) };
    let b = unsafe { b.read_f32(0, n) };
    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
    assert_eq!(
        a, b,
        "packed specialization changed the arithmetic contract"
    );
}

#[test]
fn packed_projections_keep_exact_f32_contract_and_tails() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    for (ty, k) in [
        (6, 2112),
        (8, 704),
        (12, 2816),
        (14, 2816),
        (0x100, 704),
        (0x108, 2816),
    ] {
        let n = 35;
        let w = packed(&d, ty, k, n);
        for rows in [1usize, 17, 65] {
            let x = floats(&d, k * rows);
            let expected = d.alloc(rows * n * 4).unwrap();
            let actual = d.upload(&vec![0xa5; (rows * n + 32) * 4]).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "dg_project",
                &[&w.buffer, &x, &expected],
                &[k as u32, n as u32, rows as u32, ty, u32::from(ty >= 0x100)],
                [n.div_ceil(32), rows.div_ceil(16), 1],
                128,
            );
            // Compare matrix specializations, not the intentionally different
            // MLX narrow-vector operation used by structured reads.
            let name = match ty {
                6 => "dg_project_q5",
                8 => "dg_project_q8",
                12 => "dg_project_q4",
                14 => "dg_project_q6",
                0x100 => "dg_project_a4",
                _ => "dg_project_a8",
            };
            cmd.dispatch(
                name,
                &[&w.buffer, &x, &actual],
                &[k as u32, n as u32, rows as u32, ty, u32::from(ty >= 0x100)],
                [n.div_ceil(32), rows.div_ceil(16), 1],
                128,
            );
            cmd.finish().unwrap();
            equal(&expected, &actual, rows * n);
            assert!(
                unsafe { actual.read_f32(rows * n, 32) }
                    .iter()
                    .all(|v| v.to_bits() == 0xa5a5a5a5)
            );
        }
    }
}

#[test]
fn packed_soft_embedding_keeps_exact_partition_contract() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    for (ty, name) in [
        (14, "dg_soft_embed_q6"),
        (8, "dg_soft_embed_q8"),
        (0x108, "dg_soft_embed_a8"),
    ] {
        let (k, n, rows, splits) = (256usize, 512, 17usize, 8);
        let w = packed(&d, ty, k, n);
        let p = floats(&d, rows * n);
        let expected = d.alloc(splits * rows * k * 4).unwrap();
        let actual = d.alloc(expected.len()).unwrap();
        let cmd = d.begin().unwrap();
        for (kernel, out) in [("dg_soft_embed", &expected), (name, &actual)] {
            cmd.dispatch(
                kernel,
                &[&w.buffer, &p, out],
                &[k as u32, n as u32, rows as u32, ty, splits as u32],
                [k / 32, rows.div_ceil(16), splits],
                128,
            );
        }
        cmd.finish().unwrap();
        equal(&expected, &actual, splits * rows * k);
    }
}

#[test]
fn packed_experts_preserve_scatter_and_partial_tiles() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    for (ty, kernel) in [
        (6, "dg_experts_q5"),
        (8, "dg_experts_q8"),
        (12, "dg_experts_q4"),
        (0x100, "dg_experts_a4"),
        (0x108, "dg_experts_a8"),
    ] {
        for down in [false, true] {
            let (k, n, rows) = (256usize, 35usize, 17usize);
            let stride = n * if down { 1 } else { 2 };
            let w = packed(&d, ty, k, 128 * stride);
            let x = floats(&d, rows * k * if down { 16 } else { 1 });
            let ids = d
                .upload(
                    &(0..rows * 8)
                        .flat_map(|i| ((i % 8) as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let lists = d.alloc(128 * rows * 8 * 4).unwrap();
            let counts = d.alloc(128 * 4).unwrap();
            let tiles = d
                .alloc((1 + 2 * ((rows * 8).div_ceil(16) + 128)) * 4)
                .unwrap();
            let len = rows * 8 * stride;
            let expected = d.alloc(len * 4).unwrap();
            let actual = d.upload(&vec![0xa5; (len + 32) * 4]).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "moe_align",
                &[&ids, &lists, &counts],
                &[(rows * 8) as u32],
                [128, 1, 1],
                256,
            );
            cmd.dispatch("moe_tiles", &[&counts, &tiles], &[128, 16], [1, 1, 1], 256);
            for (name, out) in [("dg_experts", &expected), (kernel, &actual)] {
                cmd.dispatch(
                    name,
                    &[&w.buffer, &x, &lists, &counts, &tiles, out],
                    &[
                        k as u32,
                        n as u32,
                        rows as u32,
                        ty,
                        u32::from(down),
                        u32::from(ty >= 0x100),
                    ],
                    [
                        n.div_ceil(32) * if down { 1 } else { 2 },
                        (rows * 8).div_ceil(16) + 128,
                        1,
                    ],
                    128,
                );
            }
            cmd.finish().unwrap();
            equal(&expected, &actual, len);
            assert!(
                unsafe { actual.read_f32(len, 32) }
                    .iter()
                    .all(|v| v.to_bits() == 0xa5a5a5a5)
            );
        }
    }
}
