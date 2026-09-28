use super::affine;
use crate::device::MetalDevice;
use paddock_models::safetensors::{SafetensorsFile, ShardedSafetensors};
use std::path::Path;

fn poison_output(y: &crate::device::Buffer, count: usize) {
    // Previous command completed. An unwritten candidate element must not
    // inherit a passing value from the baseline it is compared against.
    unsafe { y.write_u32(&vec![f32::NAN.to_bits(); count + 32]) };
}

#[test]
fn group32_expert_loaders_preserve_mixed_rows_and_output_guards() {
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let (k, n) = (64usize, 64usize);
    let mut raw = (0..512 * k * n / 2)
        .map(|i| (i * 37 + 19) as u8)
        .collect::<Vec<_>>();
    for value in [0.003f32, -0.04] {
        raw.extend(
            (0..512 * k * n / 32).flat_map(|_| ((value.to_bits() >> 16) as u16).to_le_bytes()),
        );
    }
    let w = d.upload(&raw).unwrap();
    for rows in [1usize, 9, 33, 129, 2048] {
        let entries = rows * 10;
        let ids = (0..entries)
            .map(|i| {
                if i / 10 % 7 == 0 {
                    (i % 10) as u32
                } else {
                    ((i / 10 * 37 + i % 10 * 17) % 512) as u32
                }
            })
            .collect::<Vec<_>>();
        let ids = d
            .upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        let lists = d.alloc(512 * entries * 4).unwrap();
        let counts = d.alloc(512 * 4).unwrap();
        let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
        let y = d.alloc((entries * n + 32) * 4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "moe_align",
            &[&ids, &lists, &counts],
            &[entries as u32],
            [512, 1, 1],
            256,
        );
        cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
        cmd.finish().unwrap();
        for per_entry in [false, true] {
            let input_rows = if per_entry { entries } else { rows };
            let x = d
                .upload(
                    &(0..input_rows * k)
                        .flat_map(|i| (((i * 29 % 137) as f32 - 68.) / 97.).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let staged = d.alloc(input_rows.next_multiple_of(32) * k * 2).unwrap();
            let mut p = vec![
                k as u32,
                n as u32,
                entries as u32,
                u32::from(per_entry),
                512,
            ];
            p.extend([u32::MAX; affine::MAX_ROWS.div_ceil(32)]);
            for row in (0..rows).step_by(7) {
                p[5 + row / 32] &= !(1 << (row % 32));
            }
            let mut expected = None;
            for kernel in [
                "q4a_expert_mm_tail",
                "q4a_expert_mm_group32",
                "q4a_expert_mm_group32_pad",
                "q4a_expert_mm_group32_packed",
            ] {
                poison_output(&y, entries * n);
                let cmd = d.begin().unwrap();
                let input = if kernel.ends_with("_packed") {
                    cmd.dispatch(
                        "q4a_input",
                        &[&x, &staged],
                        &[k as u32, n as u32, input_rows as u32, 0, 0, 0, 0],
                        [(input_rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
                        256,
                    );
                    &staged
                } else {
                    &x
                };
                cmd.dispatch(
                    kernel,
                    &[&w, input, &lists, &counts, &tiles, &y],
                    &p,
                    [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                    128,
                );
                cmd.dispatch(
                    "q4a_expert_vector_masked",
                    &[&w, &x, &ids, &y],
                    &p,
                    [n.div_ceil(16), entries, 1],
                    128,
                );
                cmd.finish().unwrap();
                let got = unsafe { y.read_f32(0, entries * n + 32) };
                assert!(got[..entries * n].iter().all(|v| v.is_finite()));
                assert!(got[entries * n..].iter().all(|v| v.is_nan()));
                if let Some(expected) = &expected {
                    assert!(
                        &got[..entries * n] == expected,
                        "{kernel} rows={rows} per_entry={per_entry}"
                    );
                } else {
                    expected = Some(got[..entries * n].to_vec());
                }
            }
        }
    }
}

#[test]
fn fused_expert_gate_up_preserves_bf16_boundaries_masks_and_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let (k, n) = (128usize, 80usize);
    let weight = |seed: usize| {
        let mut raw = (0..512 * k * n / 2)
            .map(|i| (i.wrapping_mul(37 + seed) ^ (i >> 5) ^ seed) as u8)
            .collect::<Vec<_>>();
        for value in [0.013f32, -0.08] {
            raw.extend((0..512 * k * n / 32).flat_map(|i| {
                ((value
                    .mul_add(1. + (i % 7) as f32 / 8., seed as f32 / 1000.)
                    .to_bits()
                    >> 16) as u16)
                    .to_le_bytes()
            }));
        }
        d.upload(&raw).unwrap()
    };
    let gate = weight(1);
    let up = weight(3);
    for rows in [1usize, 9, 33, 129, 1024, 2048] {
        let entries = rows * 10;
        let ids = d
            .upload(
                &(0..entries)
                    .flat_map(|i| {
                        (if i / 10 % 7 == 0 {
                            i % 10
                        } else {
                            (i / 10 * 37 + i % 10 * 17) % 512
                        } as u32)
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| (((i * 29 % 137) as f32 - 68.) / 97.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let packed = d.alloc(rows.next_multiple_of(32) * k * 2).unwrap();
        let lists = d.alloc(512 * entries * 4).unwrap();
        let counts = d.alloc(512 * 4).unwrap();
        let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
        let y = d.alloc((entries * n + 32) * 4).unwrap();
        let u = d.alloc((entries * n + 32) * 4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "moe_align",
            &[&ids, &lists, &counts],
            &[entries as u32],
            [512, 1, 1],
            256,
        );
        cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
        cmd.dispatch(
            "q4a_input",
            &[&x, &packed],
            &[k as u32, n as u32, rows as u32, 0, 0, 0, 0],
            [(rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        for mixed in [false, true] {
            let mut p = vec![k as u32, n as u32, entries as u32, 0, 512];
            p.extend([u32::MAX; affine::MAX_ROWS.div_ceil(32)]);
            if mixed {
                for row in (0..rows).step_by(7) {
                    p[5 + row / 32] &= !(1 << (row % 32));
                }
            }
            let mut expected = None;
            for fused in [0, 1, 2] {
                poison_output(&y, entries * n);
                poison_output(&u, entries * n);
                let cmd = d.begin().unwrap();
                if fused == 1 {
                    cmd.dispatch(
                        "q4a_expert_gate_up_packed",
                        &[&gate, &up, &packed, &lists, &counts, &tiles, &y],
                        &p,
                        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                        128,
                    );
                } else if fused == 2 {
                    cmd.dispatch(
                        "q4a_expert_gate_up_dispatch",
                        &[&gate, &up, &packed, &lists, &counts, &tiles, &y, &u],
                        &p,
                        [n.div_ceil(64), entries.div_ceil(32) + 512, 2],
                        128,
                    );
                } else {
                    for (w, out) in [(&gate, &y), (&up, &u)] {
                        cmd.dispatch(
                            "q4a_expert_mm_group32_packed",
                            &[w, &packed, &lists, &counts, &tiles, out],
                            &p,
                            [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                            128,
                        );
                    }
                }
                if mixed {
                    for (w, out) in [(&gate, &y), (&up, &u)] {
                        cmd.dispatch(
                            "q4a_expert_vector_masked",
                            &[w, &x, &ids, out],
                            &p,
                            [n.div_ceil(16), entries, 1],
                            128,
                        );
                    }
                }
                if fused == 1 {
                    if mixed {
                        cmd.dispatch(
                            "q4a_expert_swiglu_masked",
                            &[&y, &u],
                            &p,
                            [(entries * n).div_ceil(256), 1, 1],
                            256,
                        );
                    }
                } else {
                    cmd.dispatch(
                        "mlx_swiglu",
                        &[&y, &u],
                        &[(entries * n) as u32],
                        [(entries * n).div_ceil(256), 1, 1],
                        256,
                    );
                }
                cmd.finish().unwrap();
                let got = unsafe { y.read_f32(0, entries * n + 32) };
                assert!(got[..entries * n].iter().all(|v| v.is_finite()));
                assert!(got[entries * n..].iter().all(|v| v.is_nan()));
                assert!(
                    unsafe { u.read_f32(entries * n, 32) }
                        .iter()
                        .all(|v| v.is_nan())
                );
                if let Some(expected) = &expected {
                    assert!(&got[..entries * n] == expected, "rows={rows} mixed={mixed}");
                } else {
                    expected = Some(got[..entries * n].to_vec());
                }
            }
        }
    }
}

#[test]
fn unequal_projection_contracts_preserve_all_rows_and_guards() {
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    let spans = [
        (0, 1, 1),
        (1, 17, 700),
        (18, 15, 750),
        (33, 31, 800),
        (64, 64, 1024),
        (128, 31, 900),
        (159, 1, 1),
    ];
    let rows = 160;
    let short_spans = [
        (0, 4, 4),
        (4, 8, 8),
        (12, 8, 12),
        (20, 64, 64),
        (84, 64, 64),
        (148, 12, 12),
    ];
    let decode_spans = [
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (4, 64, 64),
        (68, 64, 64),
        (132, 28, 32),
    ];
    let scratch = d.alloc(affine::WORKSPACE_BYTES).unwrap();
    for (k, n, ty) in [
        (10240, 320, affine::A4G32),
        (2560, 6144, affine::A4G32),
        (2560, 512, affine::A8G64),
        (10240, 4, affine::A4G32),
        (320, 10240, affine::A4G32),
    ] {
        let (bits, group) = affine::format(ty);
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for plane in 0..2 {
            raw.extend((0..k * n / group).flat_map(|i| {
                half::bf16::from_f32(if plane == 0 {
                    0.0003 + (i % 17) as f32 * 0.0001
                } else {
                    -0.004 + (i % 7) as f32 * 0.0002
                })
                .to_le_bytes()
            }));
        }
        let w = crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            ty,
            k,
            n,
        };
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let y = d.alloc((rows * n + 32) * 4).unwrap();
        for spans in [&spans[..], &short_spans, &decode_spans] {
            poison_output(&y, rows * n);
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            for &(start, count, logical) in spans {
                affine::project_span(&cmd, &w, &x, &y, count, start, logical);
            }
            cmd.finish().unwrap();
            let expected = unsafe { y.read_f32(0, rows * n) };
            poison_output(&y, rows * n);
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(spans);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let got = unsafe { y.read_f32(0, rows * n + 32) };
            assert!(
                got[..rows * n] == expected,
                "merged projection changed K={k} N={n} type={ty}"
            );
            assert!(got[rows * n..].iter().all(|v| v.is_nan()));
        }
    }
}

#[test]
fn wide_physical_projection_preserves_logical_contraction_and_guards() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    let rows = 2048;
    let spans = [
        (0, 512, 1024),
        (512, 512, 1024),
        (1024, 512, 1024),
        (1536, 512, 1024),
    ];
    let scratch = d.alloc(affine::workspace_bytes(rows)).unwrap();
    for (k, n, ty) in [
        (10240, 320, affine::A4G32),
        (2560, 6144, affine::A4G32),
        (2560, 512, affine::A8G64),
        (320, 10240, affine::A4G32),
    ] {
        let (bits, group) = affine::format(ty);
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for value in [0.003f32, -0.04] {
            raw.extend((0..k * n / group).flat_map(|_| half::bf16::from_f32(value).to_le_bytes()));
        }
        let w = crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            k,
            n,
            ty,
        };
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let y = d.alloc((rows * n + 32) * 4).unwrap();
        poison_output(&y, rows * n);
        let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
        for &(start, count, logical) in &spans {
            affine::project_span(&cmd, &w, &x, &y, count, start, logical);
        }
        cmd.finish().unwrap();
        let expected = unsafe { y.read_f32(0, rows * n) };
        poison_output(&y, rows * n);
        let cmd = d
            .begin()
            .unwrap()
            .with_projection_workspace(&scratch)
            .with_projection_rows(&spans);
        affine::project(&cmd, &w, &x, &y, rows);
        cmd.finish().unwrap();
        let got = unsafe { y.read_f32(0, rows * n + 32) };
        assert!(
            got[..rows * n] == expected,
            "wide physical pass changed K={k} N={n} type={ty}"
        );
        assert!(got[rows * n..].iter().all(|v| v.is_nan()));
    }
}

#[test]
fn device_input_staging_preserves_offset_tail_and_guards() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let n = 515usize;
    for (k, bits, group, kernel, baseline) in [
        (320usize, 4, 32, "q4a_mm4_device64", "q4a_mm4_packed"),
        (320, 8, 64, "q4a_mm8_device64", "q4a_mm8_packed"),
        (320, 4, 32, "q4a_mm4_device64_group32", "q4a_mm4_packed"),
        (256, 4, 32, "q4a_mm4_device128_group32", "q4a_mm4_packed"),
        (320, 4, 32, "q4a_mm4_device64_pad8", "q4a_mm4_packed"),
        (256, 4, 32, "q4a_mm4_device128_pad8", "q4a_mm4_packed"),
        (320, 4, 32, "q4a_mm4_device64_pad16", "q4a_mm4_packed"),
        (256, 4, 32, "q4a_mm4_device128_pad16", "q4a_mm4_packed"),
    ] {
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for value in [0.003f32, -0.04] {
            raw.extend(
                (0..k * n / group)
                    .flat_map(|_| half::bf16::from_f32(value).to_bits().to_le_bytes()),
            );
        }
        let w = d.upload(&raw).unwrap();
        for rows in [1usize, 31, 32, 33, 63, 64, 65, 129] {
            let start = 3usize;
            let values = (0..(start + rows) * k)
                .map(|i| (i % 137) as f32 / 97. - 0.7)
                .collect::<Vec<_>>();
            let x = d
                .upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let padded = rows.next_multiple_of(32);
            let stage_words = padded * k / 2;
            let stage = d.alloc((stage_words + 16) * 4).unwrap();
            unsafe {
                stage.write_u32(&vec![0xdeadbeef; stage_words + 16]);
            }
            let count = (start + rows) * n;
            let y = d.alloc((count + 32) * 4).unwrap();
            let p = [
                k as u32,
                n as u32,
                rows as u32,
                bits as u32,
                group as u32,
                1,
                start as u32,
            ];
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                baseline,
                &[&w, &x, &y],
                &p,
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            let expected = unsafe { y.read_f32(0, count + 32) };
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_input",
                &[&x, &stage],
                &p,
                [(padded * k).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                kernel,
                &[&w, &stage, &y],
                &p,
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            let got = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(&got[start * n..count], &expected[start * n..count]);
            assert!(
                got[..start * n]
                    .iter()
                    .chain(&got[count..])
                    .all(|v| v.is_nan())
            );
            let staged = unsafe { stage.read_u32(stage_words + 16) };
            assert!(staged[stage_words..].iter().all(|&v| v == 0xdeadbeef));
            assert!(staged[rows * k / 2..stage_words].iter().all(|&v| v == 0));
        }
    }
}

#[test]
#[ignore = "rotated dense projection layout costs; exact output required, not serving timing"]
fn dense_layout_execution_cost() {
    let d = MetalDevice::new(Some(320 << 20)).unwrap();
    assert!(d.tensor_accelerated());
    for (k, n) in [
        (2560usize, 10240usize),
        (6144, 2560),
        (320, 10240),
        (10240, 320),
        (2560, 512),
    ] {
        let mut raw = (0..k * n / 2)
            .map(|i| (i * 37 + i / 127 + 19) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..k * n / 32).flat_map(|i| {
                let v = if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                };
                half::bf16::from_f32(v).to_le_bytes()
            }));
        }
        let w = d.upload(&raw).unwrap();
        for rows in [64usize, 512, 1024, 2048] {
            let start = 3;
            let x = d
                .upload(
                    &(0..(start + rows) * k)
                        .flat_map(|i| (((i % 137) as f32 / 97.) - 0.7).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let stage = d.alloc(rows.next_multiple_of(32) * k * 2).unwrap();
            let count = (start + rows) * n;
            let y = d.alloc((count + 32) * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, 4, 32, 1, start as u32];
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_input",
                &[&x, &stage],
                &p,
                [(rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish().unwrap();
            let kernels = if k.is_multiple_of(128) {
                [
                    "q4a_mm4_device128_group32",
                    "q4a_mm4_device128_pad8",
                    "q4a_mm4_device128_pad16",
                ]
            } else {
                [
                    "q4a_mm4_device64_group32",
                    "q4a_mm4_device64_pad8",
                    "q4a_mm4_device64_pad16",
                ]
            };
            let mut times = kernels.map(|_| Vec::new());
            let mut expected = None;
            for round in 0..7 {
                for route in (0..kernels.len()).map(|i| (i + round) % kernels.len()) {
                    poison_output(&y, count);
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        kernels[route],
                        &[&w, &stage, &y],
                        &p,
                        [n.div_ceil(32), rows.div_ceil(32), 1],
                        128,
                    );
                    let gpu = cmd.finish().unwrap();
                    let got = unsafe { y.read_f32(0, count + 32) };
                    assert!(
                        got[..start * n]
                            .iter()
                            .chain(&got[count..])
                            .all(|x| x.is_nan())
                    );
                    let got = got[start * n..count]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>();
                    if let Some(ref want) = expected {
                        assert!(
                            got == *want,
                            "layout changed k={k} n={n} rows={rows} route={route}"
                        );
                    } else {
                        expected = Some(got);
                    }
                    if round > 0 {
                        times[route].push(gpu);
                    }
                }
            }
            eprintln!(
                "FLASH_DENSE_LAYOUT {}",
                serde_json::json!({"k":k,"n":n,"rows":rows,"kernels":kernels,"gpu_seconds":times,"exact":true})
            );
        }
    }
}

#[test]
fn split_staging_preserves_contracts_offsets_and_bounded_arena() {
    split_staging_cases(false);
}

#[test]
#[ignore = "rotated GPU split-projection cost; not a serving benchmark"]
fn split_staging_execution_cost() {
    split_staging_cases(true);
}

fn split_staging_cases(measure: bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::PLAIN_SPLIT_FOR_TEST.with(|v| v.set(false));
            affine::PADDED_TILES_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    let d = MetalDevice::new(Some(160 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    for (k, n, rows, logical) in [
        (10240usize, 320usize, 64usize, 128usize),
        (10240, 320, 129, 512),
        (10240, 320, 512, 512),
        (10240, 320, 1024, 512),
        (10240, 320, 2048, 512),
        (10240, 320, 812, 812),
        (10240, 320, 1024, 1024),
        (10240, 320, 2048, 1024),
        (2560, 515, 129, 256),
        (2560, 515, 511, 256),
        (320, 96, 64, 128),
        (640, 160, 129, 256),
        (640, 2560, 64, 128),
    ] {
        let parts = affine::contraction(k, n, affine::A4G32, logical).1;
        let mut raw = (0..k * n / 2)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for plane in 0..2 {
            raw.extend((0..k * n / 32).flat_map(|i| {
                let value = if plane == 0 {
                    (1 + i % 29) as f32 * 0.0003
                } else {
                    -0.04 + (i % 17) as f32 * 0.002
                };
                half::bf16::from_f32(value).to_bits().to_le_bytes()
            }));
        }
        let w = crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            ty: affine::A4G32,
            k,
            n,
        };
        let start = 3;
        let x = d
            .upload(
                &(0..(start + rows) * k)
                    .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let count = (start + rows) * n;
        let y = d.alloc((count + 32) * 4).unwrap();
        let arena = d.alloc(affine::workspace_bytes(2048)).unwrap();
        affine::PLAIN_SPLIT_FOR_TEST.with(|v| v.set(true));
        affine::PADDED_TILES_FOR_TEST.with(|v| v.set(false));
        poison_output(&y, count);
        let cmd = d.begin().unwrap().with_projection_workspace(&arena);
        affine::project_span(&cmd, &w, &x, &y, rows, start, logical);
        cmd.finish().unwrap();
        let expected = unsafe { y.read_f32(0, count + 32) };
        // Include an arena that forces multiple physical slices and a small
        // arena where input staging must fall back to the original split.
        let arenas = if measure {
            vec![arena.len()]
        } else {
            vec![
                arena.len(),
                rows * n * parts * 2 + 128,
                32 * (k + n * parts) * 2,
            ]
        };
        for bytes in arenas {
            // The fallback's partial-only arena must fit too.
            let bytes = bytes.max(rows * n * parts * 2);
            let scratch = d.alloc(bytes).unwrap();
            let mut times = [Vec::new(), Vec::new(), Vec::new()];
            for round in 0..if measure { 8 } else { 1 } {
                for index in 0..3 {
                    let route = (index + round) % 3;
                    affine::PLAIN_SPLIT_FOR_TEST.with(|v| v.set(route == 0));
                    affine::PADDED_TILES_FOR_TEST.with(|v| v.set(route == 2));
                    poison_output(&y, count);
                    let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
                    affine::project_span(&cmd, &w, &x, &y, rows, start, logical);
                    let seconds = cmd.finish().unwrap();
                    let got = unsafe { y.read_f32(0, count + 32) };
                    assert_eq!(
                        &got[start * n..count],
                        &expected[start * n..count],
                        "split staging K={k} N={n} rows={rows} logical={logical} arena={bytes} route={route}"
                    );
                    assert!(
                        got[..start * n]
                            .iter()
                            .chain(&got[count..])
                            .all(|v| v.is_nan())
                    );
                    if round > 0 {
                        times[route].push(seconds);
                    }
                }
            }
            if measure {
                eprintln!(
                    "FLASH_SPLIT_STAGING {}",
                    serde_json::json!({
                    "k":k,"n":n,"rows":rows,"logical":logical,"parts":parts,
                    "arena":bytes,"gpu_seconds":times,"exact":true})
                );
            }
        }
    }
}

fn grouped_matrix_case(
    d: &MetalDevice,
    w: &crate::weights::Weight,
    x: &crate::device::Buffer,
    ids: &crate::device::Buffer,
    y: &crate::device::Buffer,
    f: &SafetensorsFile,
    base: &str,
    rows: usize,
    count: usize,
    per_entry: bool,
) {
    let k = w.k;
    let n = w.n / 512;
    let expected = f
        .bytes(if per_entry {
            "grouped_entry"
        } else {
            "grouped"
        })
        .unwrap()
        .1;
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mv",
        &[&w.buffer, x, ids, y],
        &[
            k as u32,
            n as u32,
            (rows * 10) as u32,
            u32::from(per_entry),
            512,
        ],
        [n.div_ceil(16), rows * 10, 1],
        128,
    );
    let baseline_seconds = cmd.finish().unwrap();
    let vector = unsafe { y.read_f32(0, count) };
    poison_output(y, count);
    let entries = rows * 10;
    let lists = d.alloc(512 * entries * 4).unwrap();
    let counts = d.alloc(512 * 4).unwrap();
    let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "moe_align",
        &[ids, &lists, &counts],
        &[entries as u32],
        [512, 1, 1],
        256,
    );
    cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
    let mut params = vec![
        k as u32,
        n as u32,
        entries as u32,
        u32::from(per_entry),
        512,
    ];
    params.extend([u32::MAX; 32]);
    cmd.dispatch(
        "q4a_expert_mm",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(32), entries.div_ceil(32) + 512, 1],
        128,
    );
    let seconds = cmd.finish().unwrap();
    let matrix = unsafe { y.read_f32(0, count + 32) };
    assert!(matrix[count..].iter().all(|v| v.is_nan()));
    poison_output(y, count);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mm_wide",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
        128,
    );
    let wide_seconds = cmd.finish().unwrap();
    let wide = unsafe { y.read_f32(0, count + 32) };
    assert!(wide[count..].iter().all(|v| v.is_nan()));
    assert!(
        matrix[..count] == wide[..count],
        "wide expert matrix changed arithmetic"
    );
    eprintln!(
        "AFFINE_GROUPED_WIDE {base} rows={rows} per_entry={per_entry} matrix_s={seconds} wide_s={wide_seconds}"
    );
    let kernels = [
        "q4a_expert_mm_wide",
        "q4a_expert_mm_tail",
        "q4a_expert_mm_group32",
        "q4a_expert_mm_group32_pad",
        "q4a_expert_mm_group32_packed",
    ];
    let mut times = std::array::from_fn::<_, 5, _>(|_| Vec::new());
    let input_rows = if per_entry { entries } else { rows };
    let staged = d.alloc(input_rows.div_ceil(32) * 32 * k * 2).unwrap();
    for round in 0..7 {
        for i in 0..kernels.len() {
            let route = (i + round) % kernels.len();
            poison_output(y, count);
            let cmd = d.begin().unwrap();
            let input = if route == 4 {
                cmd.dispatch(
                    "q4a_input",
                    &[x, &staged],
                    &[k as u32, n as u32, input_rows as u32, 0, 0, 0, 0],
                    [(input_rows.div_ceil(32) * 32 * k).div_ceil(256), 1, 1],
                    256,
                );
                &staged
            } else {
                x
            };
            cmd.dispatch(
                kernels[route],
                &[&w.buffer, input, &lists, &counts, &tiles, y],
                &params,
                [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                128,
            );
            let elapsed = cmd.finish().unwrap();
            let candidate = unsafe { y.read_f32(0, count + 32) };
            assert!(candidate[count..].iter().all(|v| v.is_nan()));
            assert!(
                candidate[..count] == matrix[..count],
                "expert tile changed arithmetic: {} {base} rows={rows} per_entry={per_entry}",
                kernels[route]
            );
            if round > 0 {
                times[route].push(elapsed);
            }
        }
    }
    eprintln!(
        "AFFINE_EXPERT_TILES {}",
        serde_json::json!({"base":base,"rows":rows,"per_entry":per_entry,"kernels":kernels,"gpu_seconds":times})
    );
    for row in (0..rows).step_by(7) {
        params[5 + row / 32] &= !(1 << (row % 32));
    }
    poison_output(y, count);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mm_tail",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
        128,
    );
    cmd.dispatch(
        "q4a_expert_vector_masked",
        &[&w.buffer, x, ids, y],
        &params,
        [n.div_ceil(16), entries, 1],
        128,
    );
    cmd.finish().unwrap();
    let mixed = unsafe { y.read_f32(0, count + 32) };
    assert!(mixed[count..].iter().all(|v| v.is_nan()));
    for row in 0..rows {
        let expected = if row % 7 == 0 { &vector } else { &matrix };
        let span = row * 10 * n..(row + 1) * 10 * n;
        assert!(
            mixed[span.clone()] == expected[span],
            "mixed expert contract changed row {row}"
        );
    }
    let mut max_error = 0f32;
    let mut unequal = 0;
    let mut peak = 0f32;
    for (&a, b) in matrix[..count].iter().zip(
        expected
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b)),
    ) {
        assert!(a.is_finite() && b.is_finite());
        max_error = max_error.max((a - b).abs());
        peak = peak.max(b.abs());
        unequal += usize::from(a != b);
    }
    eprintln!(
        "AFFINE_GROUPED {base} rows={rows} baseline_s={baseline_seconds} matrix_s={seconds} error={max_error} peak={peak} unequal={unequal}/{count}"
    );
    assert_eq!(
        unequal, 0,
        "same-checkpoint grouped expert operation mismatch"
    );
}

#[test]
#[ignore = "same-checkpoint MLX GPU fixtures: PADDOCK_FLASH_NEXT_MLX_AFFINE_REFERENCE"]
fn flash_next_affine_matches_mlx_gpu() {
    let dir = std::env::var("PADDOCK_FLASH_NEXT_MLX_AFFINE_REFERENCE").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let scratch = d.alloc(affine::WORKSPACE_BYTES).unwrap();
    let mut failures = Vec::new();
    for c in manifest["cases"].as_array().unwrap() {
        let k = c["k"].as_u64().unwrap() as usize;
        let n = c["n"].as_u64().unwrap() as usize;
        let rows = c["rows"].as_u64().unwrap() as usize;
        let experts = c["expert"].as_bool().unwrap();
        let shape = if experts { vec![512, n, k] } else { vec![n, k] };
        let ty = if c["bits"] == 8 {
            affine::A8G64
        } else {
            affine::A4G32
        };
        let base = c["base"].as_str().unwrap();
        eprintln!(
            "AFFINE_CASE file={} input={}",
            c["file"],
            c.get("input").and_then(|v| v.as_str()).unwrap_or("sine")
        );
        let w = affine::load(&d, &source, base, &shape, ty).unwrap();
        let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
        let x = d.upload(f.bytes("x").unwrap().1).unwrap();
        let count = rows * n * if experts { 10 } else { 1 };
        let y = d
            .upload(
                &(0..count + 32)
                    .flat_map(|_| f32::NAN.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let ids = if experts {
            Some(d.upload(f.bytes("ids").unwrap().1).unwrap())
        } else {
            None
        };
        if experts && rows > 128 {
            grouped_matrix_case(
                &d,
                &w,
                &x,
                ids.as_ref().unwrap(),
                &y,
                &f,
                base,
                rows,
                count,
                false,
            );
            if let Some((_, raw)) = f.bytes("x_entry") {
                let input = d.upload(raw).unwrap();
                grouped_matrix_case(
                    &d,
                    &w,
                    &input,
                    ids.as_ref().unwrap(),
                    &y,
                    &f,
                    base,
                    rows,
                    count,
                    true,
                );
            }
            continue;
        }
        let cmd = d.begin().unwrap();
        if let Some(ids) = &ids {
            cmd.dispatch(
                "q4a_expert_mv",
                &[&w.buffer, &x, ids, &y],
                &[k as u32, n as u32, (rows * 10) as u32, 0, 512],
                [n.div_ceil(16), rows * 10, 1],
                128,
            );
        } else if rows >= 13 && n > 48 {
            // Keep the original scalar-staged matrix as an independent
            // exactness seam for split-K and rejected tile candidates.
            let group = c["group"].as_u64().unwrap() as usize;
            let mut parts = (512 / (n.div_ceil(32) * rows.div_ceil(32)))
                .min(k / group.max(32))
                .max(1);
            while !k.is_multiple_of(parts * group.max(32)) {
                parts -= 1;
            }
            cmd.dispatch(
                "q4a_mm",
                &[&w.buffer, &x, &y],
                &[
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    group as u32,
                    parts as u32,
                    0,
                ],
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
        } else {
            affine::project(&cmd, &w, &x, &y, rows);
        }
        let baseline_seconds = cmd.finish().unwrap();
        let actual = unsafe { y.read_f32(0, count + 32) };
        eprintln!("AFFINE_TIME {base} rows={rows} experts={experts} gpu_s={baseline_seconds}");
        if let Some(ids) = &ids {
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            affine::experts(&cmd, &w, &x, ids, &y, rows * 10, false);
            let specialized_seconds = cmd.finish().unwrap();
            let specialized = unsafe { y.read_f32(0, count) };
            assert_eq!(
                &actual[..count],
                &specialized,
                "expert step specialization changed arithmetic {base} rows={rows}"
            );
            eprintln!(
                "AFFINE_EXPERT_ENTRY {base} rows={rows} baseline_s={baseline_seconds} specialized_s={specialized_seconds}"
            );
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_expert_order",
                &[ids, &scratch],
                &[(rows * 10) as u32],
                [1, 1, 1],
                256,
            );
            affine::experts_ordered(&cmd, &w, &x, ids, &y, rows * 10, false, Some(&scratch));
            let ordered_seconds = cmd.finish().unwrap();
            let ordered = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &ordered[..count],
                "expert ordering changed arithmetic {base} rows={rows}"
            );
            assert!(ordered[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_ORDER {base} rows={rows} baseline_s={baseline_seconds} ordered_s={ordered_seconds}"
            );
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if k.is_multiple_of(512) {
                    "q4a_expert4_fast_pair"
                } else {
                    "q4a_expert4_pair"
                },
                &[&w.buffer, &x, ids, &y, &scratch],
                &[k as u32, n as u32, (rows * 10) as u32, 0, 512],
                [n.div_ceil(16) * 4, (rows * 10).div_ceil(8), 1],
                128,
            );
            let pair_seconds = cmd.finish().unwrap();
            let paired = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &paired[..count],
                "expert pair changed arithmetic {base} rows={rows}"
            );
            assert!(paired[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_PAIR {base} rows={rows} baseline_s={baseline_seconds} ordered_s={ordered_seconds} pair_s={pair_seconds}"
            );
        }
        if !experts && rows == 4 {
            let cmd = d.begin().unwrap().with_independent_rows(true);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let batched = unsafe { y.read_f32(0, count) };
            for row in 0..rows {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[row * k * 4..(row + 1) * k * 4])
                    .unwrap();
                let out = d.alloc(n * 4).unwrap();
                let cmd = d.begin().unwrap();
                affine::project(&cmd, &w, &input, &out, 1);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, n) },
                    batched[row * n..(row + 1) * n],
                    "independent rows changed singleton contraction {base} row={row}"
                );
            }
        }
        if !experts && rows > 1 {
            // Independent launches retain the pre-coalescing implementation
            // as the oracle, including unaligned starts and tiny physical
            // slices of a large logical contract.
            let mut spans = Vec::new();
            let mut at = 0;
            for len in [1, 7, 23, 33, rows / 4, rows] {
                let count = len.min(rows - at);
                if count > 0 {
                    spans.push((at, count, rows));
                    at += count;
                }
            }
            poison_output(&y, count);
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(&spans);
            affine::project(&cmd, &w, &x, &y, rows);
            let merged_seconds = cmd.finish().unwrap();
            let merged = unsafe { y.read_f32(0, count + 32) };
            assert!(merged[count..].iter().all(|v| v.is_nan()));
            poison_output(&y, count);
            let mut separate_seconds = 0.;
            for &span in &spans {
                let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
                affine::project_span(&cmd, &w, &x, &y, span.1, span.0, span.2);
                separate_seconds += cmd.finish().unwrap();
            }
            let separate = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &merged[..count],
                &separate[..count],
                "dense coalescing changed {base} rows={rows}"
            );
            assert!(separate[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_COALESCED {base} rows={rows} merged_s={merged_seconds} separate_s={separate_seconds}"
            );
        }
        if !experts && [4, 128].contains(&rows) {
            let spans = if rows == 4 {
                vec![(0, 1, 1), (1, 3, 3)]
            } else {
                vec![(0, 1, 1), (1, 33, 33), (34, 94, 94)]
            };
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(&spans);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let mixed = unsafe { y.read_f32(0, count + 32) };
            assert!(mixed[count..].iter().all(|v| v.is_nan()));
            for &(start, len, _) in &spans {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[start * k * 4..(start + len) * k * 4])
                    .unwrap();
                let out = d.alloc(len * n * 4).unwrap();
                let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
                affine::project(&cmd, &w, &input, &out, len);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, len * n) },
                    mixed[start * n..(start + len) * n],
                    "ragged projection changed per-sequence contraction {base} span={start}/{len}"
                );
            }
        }
        if !experts && rows >= 13 && n > 48 {
            if affine::contraction(k, n, ty, rows).1 == 1 && k.is_multiple_of(64) {
                let staged = d.alloc(rows.next_multiple_of(32) * k * 2).unwrap();
                let kernel = match (ty, k.is_multiple_of(128)) {
                    (affine::A4G32, true) => "q4a_mm4_device128",
                    (affine::A4G32, false) => "q4a_mm4_device64",
                    (_, true) => "q4a_mm8_device128",
                    (_, false) => "q4a_mm8_device64",
                };
                let group_kernel = match (ty, k.is_multiple_of(128)) {
                    (affine::A4G32, true) => "q4a_mm4_device128_group32",
                    (affine::A4G32, false) => "q4a_mm4_device64_group32",
                    _ => kernel,
                };
                let params = [
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    c["group"].as_u64().unwrap() as u32,
                    1,
                    0,
                ];
                let mut times = [Vec::new(), Vec::new(), Vec::new()];
                for round in 0..7 {
                    for index in 0..3 {
                        let route = (index + round) % 3;
                        poison_output(&y, count);
                        let cmd = d.begin().unwrap();
                        if route > 0 {
                            cmd.dispatch(
                                "q4a_input",
                                &[&x, &staged],
                                &params,
                                [(rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
                                256,
                            );
                            cmd.dispatch(
                                if route == 2 { group_kernel } else { kernel },
                                &[&w.buffer, &staged, &y],
                                &params,
                                [n.div_ceil(32), rows.div_ceil(32), 1],
                                128,
                            );
                        } else {
                            cmd.dispatch(
                                if ty == affine::A4G32 {
                                    "q4a_mm4_packed"
                                } else {
                                    "q4a_mm8_packed"
                                },
                                &[&w.buffer, &x, &y],
                                &params,
                                [n.div_ceil(32), rows.div_ceil(32), 1],
                                128,
                            );
                        }
                        let elapsed = cmd.finish().unwrap();
                        let got = unsafe { y.read_f32(0, count + 32) };
                        assert!(got[count..].iter().all(|v| v.is_nan()));
                        assert!(
                            got[..count] == actual[..count],
                            "device input changed {base} rows={rows} route={route}"
                        );
                        if round > 0 {
                            times[route].push(elapsed);
                        }
                    }
                }
                eprintln!(
                    "AFFINE_DEVICE_INPUT {}",
                    serde_json::json!({"base":base,"rows":rows,"gpu_seconds":times})
                );
            }
            poison_output(&y, count);
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            affine::project(&cmd, &w, &x, &y, rows);
            let split_seconds = cmd.finish().unwrap();
            let parallel = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &parallel[..count],
                "parallel split-K changed native BF16 arithmetic: {base} rows={rows}"
            );
            assert!(parallel[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_SPLIT {base} rows={rows} baseline_s={baseline_seconds} parallel_s={split_seconds}"
            );
        }
        if !experts && rows >= 812 && n > 48 {
            // All checkpoint planes here have a single K partition. Retain
            // the reordered packed kernel as a GPU cache-ordering ablation.
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if c["bits"] == 4 {
                    "q4a_mm4_reuse"
                } else {
                    "q4a_mm8_reuse"
                },
                &[&w.buffer, &x, &y],
                &[
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    c["group"].as_u64().unwrap() as u32,
                    1,
                    0,
                ],
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
            let packed_seconds = cmd.finish().unwrap();
            let packed = unsafe { y.read_f32(0, count + 32) };
            assert!(
                actual[..count] == packed[..count],
                "packed dense changed {base}"
            );
            assert!(packed[count..].iter().all(|v| v.is_nan()));
            eprintln!("AFFINE_REUSE {base} rows={rows} gpu_s={packed_seconds}");
        }
        if !experts && [4, 128].contains(&rows) {
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let whole = unsafe { y.read_f32(0, count) };
            let slices = if rows == 4 {
                vec![(0, 1), (1, 3)]
            } else {
                vec![(0, 1), (1, 7), (8, 23), (31, 33), (64, 63), (127, 1)]
            };
            for (start, len) in slices {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[start * k * 4..(start + len) * k * 4])
                    .unwrap();
                let out = d.alloc(len * n * 4).unwrap();
                let spans = [(0, len, rows)];
                let cmd = d
                    .begin()
                    .unwrap()
                    .with_projection_workspace(&scratch)
                    .with_projection_rows(&spans);
                affine::project(&cmd, &w, &input, &out, len);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, len * n) },
                    whole[start * n..(start + len) * n],
                    "logical projection changed when sliced: {base} span={start}/{len}/{rows}"
                );
            }
        }
        assert!(
            actual[count..].iter().all(|v| v.is_nan()),
            "output guard {base}"
        );
        let expected = f
            .bytes("y")
            .unwrap()
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b));
        let mut error = 0f32;
        let mut peak = 0f32;
        let mut unequal = 0;
        for (&a, b) in actual[..count].iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(a.to_bits() & 0xffff, 0);
            error = error.max((a - b).abs());
            peak = peak.max(b.abs());
            unequal += usize::from(a != b);
        }
        eprintln!("AFFINE {base} m={rows} error={error} peak={peak} unequal={unequal}/{count}");
        // An operation-format bound is not a generation-parity claim. Tiny
        // row-invariant projections must match exactly; all model outputs
        // still need their separate greedy/logit qualification.
        if ((n <= 48 || rows == 1 || experts) && unequal != 0) || error > peak * 0.008 + 0.0001 {
            failures.push(format!(
                "{base} rows={rows} error={error} unequal={unequal}"
            ));
        }
    }
    let c = &manifest["gather"];
    let base = c["base"].as_str().unwrap();
    let (info, _) = source.bytes(&format!("{base}.weight")).unwrap();
    let w = affine::load(&d, &source, base, &[info.shape[0], 160], affine::A4G32).unwrap();
    let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
    let ids = d.upload(f.bytes("ids").unwrap().1).unwrap();
    let y = d.alloc(7 * 160 * 4).unwrap();
    let cmd = d.begin().unwrap();
    affine::gather(&cmd, &w, &ids, &y, 7);
    cmd.finish().unwrap();
    let actual = unsafe { y.read_f32(0, 7 * 160) };
    let expected = f
        .bytes("y")
        .unwrap()
        .1
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "same-checkpoint PLE shard gather");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn flash_next_mlx_expert_order_is_stable_bounded_permutation() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    for entries in [1, 10, 40, 64, 90, 330, 1280] {
        for invalid in [false, true] {
            let ids = (0..entries)
                .map(|i| {
                    if invalid && i % 13 == 0 {
                        512
                    } else {
                        (i * 73 + 511) % 512
                    }
                })
                .collect::<Vec<u32>>();
            let input = d
                .upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let out = d.upload(&vec![0xA5; (entries as usize + 16) * 4]).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_expert_order",
                &[&input, &out],
                &[entries],
                [1, 1, 1],
                256,
            );
            cmd.finish().unwrap();
            let actual = unsafe { out.read_u32(entries as usize + 16) };
            // Control-index verification only; no host model arithmetic.
            let mut expected = (0..entries)
                .filter(|&i| ids[i as usize] < 512)
                .collect::<Vec<_>>();
            expected.sort_by_key(|&i| (ids[i as usize], i));
            expected.resize(entries as usize, u32::MAX);
            assert_eq!(&actual[..entries as usize], expected);
            assert!(actual[entries as usize..].iter().all(|&v| v == 0xA5A5A5A5));
        }
    }
}
