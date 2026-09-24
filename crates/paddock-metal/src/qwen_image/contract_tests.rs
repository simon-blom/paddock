//! Exact intermediate BF16 contracts, independent of a downloaded checkpoint.
use super::*;

fn bf(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

#[test]
#[ignore = "requires Metal GPU"]
fn timestep_input_and_silu_materialize_bf16_boundaries() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let values: Vec<_> = (0..512).map(|i| (i as f32 * 0.131).sin()).collect();
    let xb = e.to_device(&values).unwrap();
    e.run("qi_round_bf", &[&xb], &[512], [2, 1, 1], 256)
        .unwrap();
    let expected: Vec<_> = values.into_iter().map(bf).collect();
    assert_eq!(unsafe { xb.read_f32(0, 512) }, expected);
    e.run("qi_activation", &[&xb], &[512, 2], [2, 1, 1], 256)
        .unwrap();
    let expected: Vec<_> = expected
        .into_iter()
        .map(|x| {
            let tail = bf(1. / bf(1. + bf(x.abs().exp())));
            bf(x * if x < 0. { tail } else { bf(1. - tail) })
        })
        .collect();
    assert_eq!(unsafe { xb.read_f32(0, 512) }, expected);
}

#[test]
#[ignore = "requires Metal GPU"]
fn dit_modulated_norm_rounds_scale_before_multiply() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let width = 4096;
    // Exactly centered/unit-variance input and nonrepresentable 1+scale.
    // The old fused multiply used the unrounded scale and fails this gate.
    let x: Vec<_> = (0..width)
        .map(|i| if i % 2 == 0 { 1. } else { -1. })
        .collect();
    let scale: Vec<_> = (0..width)
        .map(|i| bf(0.00390625 + (i % 17) as f32 * 0.0078125))
        .collect();
    let xb = e.to_device(&x).unwrap();
    let sb = e.to_device(&scale).unwrap();
    let out = e.device.alloc(width * 4).unwrap();
    e.run(
        "qi_norm",
        &[&xb, &sb, &out],
        &[width as u32, 0, 1e-6f32.to_bits(), 3, 0],
        [1, 1, 1],
        256,
    )
    .unwrap();
    let expected: Vec<_> = x
        .iter()
        .zip(&scale)
        .map(|(x, s)| bf(bf(x / (1.0f32 + 1e-6).sqrt()) * bf(1. + s)))
        .collect();
    assert_eq!(unsafe { out.read_f32(0, width) }, expected);
}

#[test]
#[ignore = "requires Metal GPU"]
fn dit_residual_retains_tanh_and_product_rounding() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let n: usize = 4096;
    let x: Vec<_> = (0..n).map(|i| bf((i % 37) as f32 * 0.113 - 2.)).collect();
    let y: Vec<_> = (0..n).map(|i| bf((i % 101) as f32 * 0.177 - 7.)).collect();
    let gate: Vec<_> = (0..n).map(|i| bf((i % 29) as f32 * 0.137 - 1.3)).collect();
    let xb = e.to_device(&x).unwrap();
    let yb = e.to_device(&y).unwrap();
    let gb = e.to_device(&gate).unwrap();
    e.run(
        "qi_mlx_gated_add",
        &[&xb, &yb, &gb],
        &[n as u32, n as u32, 0],
        [n.div_ceil(256), 1, 1],
        256,
    )
    .unwrap();
    let expected: Vec<_> = (0..n)
        .map(|i| bf(x[i] + bf(y[i] * bf(gate[i].tanh()))))
        .collect();
    assert_eq!(unsafe { xb.read_f32(0, n) }, expected);
}

#[test]
#[ignore = "requires Metal GPU"]
fn dit_head_norm_rounds_before_weight_but_text_norm_does_not() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let x: Vec<_> = (0..128).map(|i| bf((i % 23) as f32 * 0.25 - 2.)).collect();
    let norm: Vec<_> = (0..128).map(|i| bf(1.01 + (i % 7) as f32 * 0.1)).collect();
    let xb = e.to_device(&x).unwrap();
    let wb = e.to_device(&norm).unwrap();
    let pos = words(&e.device, &[0, 0, 0]).unwrap();
    let out = e.device.alloc(128 * 2).unwrap();
    e.run(
        "qi_mlx_head_rope",
        &[&xb, &wb, &pos, &out],
        &[1, 0, 0],
        [1, 1, 1],
        32,
    )
    .unwrap();
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / 128. + 1e-6)
        .sqrt()
        .recip();
    let expected: Vec<_> = x
        .iter()
        .zip(&norm)
        .map(|(v, w)| bf(bf(v * inv) * w))
        .collect();
    let actual: Vec<_> = unsafe { out.read_u32(64) }
        .into_iter()
        .flat_map(|v| [v as u16, (v >> 16) as u16])
        .map(|v| half::f16::from_bits(v).to_f32())
        .collect();
    assert_eq!(actual, expected);
    assert_ne!(
        expected,
        x.iter()
            .zip(&norm)
            .map(|(v, w)| bf(v * inv * w))
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires Metal GPU"]
fn dit_fused_gate_up_matches_separate_bf16_swiglu() {
    let e = Ops {
        device: MetalDevice::new(None).unwrap(),
    };
    let (rows, width) = (17, 12288);
    let gate: Vec<_> = (0..rows * width)
        .map(|i| bf((i % 103) as f32 * 0.137 - 7.))
        .collect();
    let up: Vec<_> = (0..rows * width)
        .map(|i| bf((i % 97) as f32 * 0.331 - 13.))
        .collect();
    let joined: Vec<_> = gate
        .chunks(width)
        .zip(up.chunks(width))
        .flat_map(|(g, u)| g.iter().chain(u).copied())
        .collect();
    let input = e.to_device(&joined).unwrap();
    let expected = e.to_device(&gate).unwrap();
    let ub = e.to_device(&up).unwrap();
    let out = e.device.alloc(rows * width * 4).unwrap();
    let cmd = e.device.begin().unwrap();
    cmd.dispatch(
        "qi_mlx_swiglu",
        &[&expected, &ub],
        &[(rows * width) as u32],
        [(rows * width).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "qi_mlx_gate_up",
        &[&input, &out],
        &[rows as u32, width as u32],
        [(rows * width).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { out.read_f32(0, rows * width) }, unsafe {
        expected.read_f32(0, rows * width)
    });
}
