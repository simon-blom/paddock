//! GPU-versus-GPU guards against premature Q4_K half-scale division. These
//! dyadic fixtures describe tensors; no CPU projection is the reference.
use super::*;

#[test]
fn q4k_subnormal_scales_survive_all_consumers() {
    check_q4k_subnormal_scale(true);
}

#[test]
fn q4k_prefill_verify_and_gemv_match_gpu_dyadic_fixtures() {
    // Run this entry with both Metal validators. The separate strict-F32
    // Gemma guard above runs with API validation: its existing padded MPP
    // kernel exceeds the instrumented threadgroup limit (33,280 bytes).
    // Do not weaken that guard or change Gemma arithmetic to qualify Qwen.
    check_q4k_subnormal_scale(false);
}

fn check_q4k_subnormal_scale(check_gemma_f32: bool) {
    let device = MetalDevice::new(Some(32 << 20)).unwrap();
    let (k, n, rows) = (256usize, 32usize, 32usize);
    let mut packed = vec![0u8; n * 144];
    for (col, block) in packed.chunks_exact_mut(144).enumerate() {
        // Half subnormal scales 24 and 40 ulps: /16 lands on opposite
        // sides of the same ties-to-even result, two subnormal ulps.
        block[..2].copy_from_slice(&(if col % 2 == 0 { 24u16 } else { 40u16 }).to_le_bytes());
        block[4..8].fill(1); // first four scales = 1, all minima = 0
        block[12..16].fill(1); // remaining four scales = 1
        block[16..].fill(0x11); // both nibbles = 1
    }
    let quant = Weight {
        buffer: device.upload(&packed).unwrap(),
        ty: 12,
        k,
        n,
    };
    let fixture = |early_half_division: bool| Weight {
        buffer: device
            .upload(
                &(0..n * k)
                    .flat_map(|i| {
                        let units = if early_half_division && (i % k / 32) % 2 == 1 {
                            32.0f32
                        } else if (i / k) % 2 == 0 {
                            24.0
                        } else {
                            40.0
                        };
                        (units / 16_777_216.0).to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ty: 0,
        k,
        n,
    };
    let exact = fixture(false);
    let prematurely_rounded = fixture(true);
    let input = device
        .upload(
            &vec![1.0f32; rows * k]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let workspace = device.alloc((128 * k + n * k) * 2).unwrap();
    let a = device.alloc(rows * n * 4).unwrap();
    let b = device.alloc(a.len()).unwrap();
    let p = [k as u32, n as u32, rows as u32, 12, 1.0f32.to_bits()];

    let cmd = device.begin().unwrap();
    exact.linear(&cmd, &input, &a, rows, 1.0, &workspace);
    quant.linear(&cmd, &input, &b, rows, 1.0, &workspace);
    cmd.finish().unwrap();
    let expected_half = unsafe { a.read_f32(0, rows * n) };
    assert_eq!(
        expected_half,
        unsafe { b.read_f32(0, rows * n) },
        "bounded half tile"
    );

    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "linear_input_padded",
        &[&input, &workspace],
        &p,
        [(128 * k).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "linear_kexpand",
        &[&quant.buffer, &workspace],
        &p,
        [(k * n).div_ceil(1024), 1, 1],
        256,
    );
    cmd.dispatch("linear_kexpanded128", &[&workspace, &b], &p, [1, 1, 1], 128);
    cmd.finish().unwrap();
    assert_eq!(
        expected_half,
        unsafe { b.read_f32(0, rows * n) },
        "device-expanded half tile"
    );

    let cmd = device.begin().unwrap();
    prematurely_rounded.linear(&cmd, &input, &a, rows, 1.0, &workspace);
    if check_gemma_f32 {
        cmd.dispatch(
            "gemma_staged_f32_32",
            &[&quant.buffer, &input, &b],
            &p,
            [n.div_ceil(16), 1, 1],
            128,
        );
    }
    cmd.finish().unwrap();
    let rounded = unsafe { a.read_f32(0, rows * n) };
    assert_ne!(
        expected_half, rounded,
        "fixture must detect premature half-scale division"
    );
    if check_gemma_f32 {
        assert_eq!(
            expected_half,
            unsafe { b.read_f32(0, rows * n) },
            "strict F32 staging unchanged"
        );
    }

    let cmd = device.begin().unwrap();
    projection::project(&cmd, &[(&quant, &b)], &input, rows, &workspace);
    cmd.finish().unwrap();
    assert_eq!(
        expected_half,
        unsafe { b.read_f32(0, rows * n) },
        "Qwen projection dispatch preserves subnormal scales"
    );

    let cmd = device.begin().unwrap();
    quant.linear(&cmd, &input, &b, 1, 1.0, &workspace);
    cmd.finish().unwrap();
    assert_eq!(
        &expected_half[..n],
        unsafe { b.read_f32(0, n) },
        "F32 GEMV unchanged"
    );
}
