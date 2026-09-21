use super::*;
use objc2_metal::MTLBuffer;

fn halves(b: &Buffer, n: usize) -> Vec<f32> {
    // Test callers have completed the producing command buffer.
    unsafe {
        std::slice::from_raw_parts(b.raw.contents().as_ptr().cast::<half::f16>(), n)
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }
}

#[test]
fn native_bf16_projection_preserves_range_and_identity() {
    // An identity contraction, not a host matrix oracle. These values expose
    // accidental F16 reinterpretation, underflow, overflow and double rounding.
    let device = MetalDevice::new(Some(16 << 20)).unwrap();
    let values: [f32; 8] = [1.001, 1e-10, 131072.0, -131072.0, 0.0, -1e-10, 3.5, -7.0];
    let input = (0..32 * 8).map(|i| values[i % 8]).collect::<Vec<_>>();
    let x = device
        .upload(
            &input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let w = Weight {
        buffer: device
            .upload(
                &(0..64 * 32)
                    .flat_map(|i| {
                        half::bf16::from_f32(if i % 32 == i / 32 % 32 { 1.0 } else { 0.0 })
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ty: 30,
        k: 32,
        n: 64,
    };
    let b = Weight {
        buffer: device.upload(&vec![0; 64 * 4]).unwrap(),
        ty: 0,
        k: 64,
        n: 1,
    };
    let out = device.alloc(8 * 64 * 4).unwrap();
    let cmd = device.begin().unwrap();
    Vision::mm::<false, false>(&cmd, &w, &x, &out, &b, 8, 1);
    cmd.finish().unwrap();
    let actual = unsafe { out.read_f32(0, 8 * 64) };
    for (i, &v) in actual.iter().enumerate() {
        assert_eq!(v, input[i / 64 * 32 + i % 32], "element {i}");
    }
    // The accelerated vision route has a distinct, explicitly bounded
    // multiplication contract. It must still retain exponent range and
    // finite output. This does not relax the strict identity assertion above.
    let cmd = device.begin().unwrap();
    Vision::mm::<true, false>(&cmd, &w, &x, &out, &b, 8, 1);
    cmd.finish().unwrap();
    for (i, v) in unsafe { out.read_f32(0, 8 * 64) }.into_iter().enumerate() {
        let expected = input[i / 64 * 32 + i % 32];
        assert!(v.is_finite());
        assert!(
            (v - expected).abs() <= expected.abs() / 1024.0,
            "accelerated identity {i}: {v} / {expected}"
        );
    }
}

#[test]
fn fused_gelu_preserves_large_outliers_and_finite_output() {
    let d = MetalDevice::new(Some(16 << 20)).unwrap();
    let values = [-100.0, -34.0, -12.0, 0.0, 12.0, 16.0, 34.0, 100.0];
    let x = d
        .upload(
            &values
                .iter()
                .flat_map(|&v| {
                    (0..32).flat_map(move |i| {
                        half::f16::from_f32(if i == 0 { v } else { 0.0 }).to_le_bytes()
                    })
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let w = Weight {
        buffer: d
            .upload(
                &(0..64 * 32)
                    .flat_map(|i| {
                        half::f16::from_f32(if i % 32 == 0 { 1.0 } else { 0.0 }).to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ty: 1,
        k: 32,
        n: 64,
    };
    let b = Weight {
        buffer: d.upload(&vec![0; 64 * 4]).unwrap(),
        ty: 0,
        k: 64,
        n: 1,
    };
    let y = d.alloc(values.len() * 64 * 2).unwrap();
    let cmd = d.begin().unwrap();
    Vision::mm::<false, true>(&cmd, &w, &x, &y, &b, values.len(), 3);
    cmd.finish().unwrap();
    let actual = halves(&y, values.len() * 64);
    for (i, &v) in values.iter().enumerate() {
        assert!(
            actual[i * 64..(i + 1) * 64]
                .iter()
                .all(|&a| a == v.max(0.0)),
            "outlier {v}"
        );
    }
}

#[test]
fn half_weights_consume_float_activations_and_preserve_float_epilogues() {
    let device = MetalDevice::new(Some(16 << 20)).unwrap();
    // Native FP16 towers still consume F32 layernorm/attention buffers. Include
    // values outside FP16's range so a stray cast is caught, not merely a bad
    // byte reinterpretation. Exercise both matrix tile sizes and epilogues.
    for rows in [7usize, 129] {
        let input: Vec<f32> = (0..rows * 32)
            .map(|i| [1.001, 131072., -131072., 1e-10][i % 4])
            .collect();
        let x = device
            .upload_parts(&[&input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()])
            .unwrap();
        let w = Weight {
            buffer: device
                .upload_parts(&[&(0..64 * 32)
                    .flat_map(|i| {
                        half::f16::from_f32(if i % 32 == (i / 32) % 32 { 1. } else { 0. })
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>()])
                .unwrap(),
            ty: 1,
            k: 32,
            n: 64,
        };
        let bias = Weight {
            buffer: device.upload_parts(&[&vec![0; 64 * 4]]).unwrap(),
            ty: 0,
            k: 64,
            n: 1,
        };
        let output = device.alloc(rows * 64 * 4).unwrap();
        for epilogue in [1, 3] {
            let cmd = device.begin().unwrap();
            Vision::mm::<true, false>(&cmd, &w, &x, &output, &bias, rows, epilogue);
            cmd.finish().unwrap();
            // SAFETY: the completed command initialized every output element.
            for (i, actual) in unsafe { output.read_f32(0, rows * 64) }
                .into_iter()
                .enumerate()
            {
                let value = input[i / 64 * 32 + i % 32];
                let expected = if epilogue == 1 {
                    value
                } else {
                    value
                        * 0.5
                        * (1. + (0.7978846 * value * (1. + 0.044715 * value * value)).tanh())
                };
                assert!(
                    (actual - expected).abs() <= 1e-5 * expected.abs().max(1e-10),
                    "{rows}/{epilogue}/{i}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn noncausal_tensor_attention_matches_gpu_oracle_and_isolates_ragged_images() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    // Three KV tiles in the middle image exercise repeated online rescaling
    // and accumulation; both image boundaries remain deliberately ragged.
    let sizes = [37usize, 132, 13];
    let rows: usize = sizes.iter().sum();
    let plane = |seed: usize| {
        d.upload(
            &(0..(rows + 64) * 1280)
                .flat_map(|i| {
                    let v = if i % 80 < 72 && i / 1280 < rows {
                        ((i * seed % 31) as f32 - 15.0) / 64.0
                    } else {
                        0.0
                    };
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (q, k, v) = (plane(3), plane(7), plane(11));
    let mut tiles = Vec::new();
    let mut bounds = Vec::new();
    let mut first = 0;
    for n in sizes {
        for row in (0..n).step_by(32) {
            tiles.extend([
                (first + row) as u32,
                (n - row).min(32) as u32,
                first as u32,
                n as u32,
            ]);
        }
        for _ in 0..n {
            bounds.extend([first as u32, (first + n) as u32]);
        }
        first += n;
    }
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let t = upload(&tiles);
    let b = upload(&bounds);
    let out = d
        .upload(
            &(0..(rows + 32) * E)
                .flat_map(|_| half::f16::NAN.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let reference = d.alloc(rows * E * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "vis_attention",
        &[&q, &k, &v, &out, &t],
        &[1],
        [16, tiles.len() / 4, 1],
        64,
    );
    cmd.dispatch(
        "vis_attention_check",
        &[&q, &k, &v, &reference, &b],
        &[],
        [16, rows, 1],
        32,
    );
    cmd.finish().unwrap();
    let actual = halves(&out, rows * E);
    let expected = unsafe { reference.read_f32(0, rows * E) };
    assert!(actual.iter().all(|v| v.is_finite()));
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 0.0001, "GPU attention error {max}");
    let full = d.alloc(rows * E * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "vis_attention",
        &[&q, &k, &v, &full, &t],
        &[0],
        [16, tiles.len() / 4, 1],
        64,
    );
    cmd.finish().unwrap();
    let actual = unsafe { full.read_f32(0, rows * E) };
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max < 0.000002,
        "F32 probability/output GPU attention error {max}"
    );
    assert!(
        halves(&out, (rows + 32) * E)[rows * E..]
            .iter()
            .all(|v| v.is_nan()),
        "tail guard"
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MMPROJ and an M5"]
fn tower_finite_and_ragged_batch_isolation() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("mmproj path");
    let device = MetalDevice::new(None).unwrap();
    let map = MappedGguf::open(Path::new(&path)).unwrap();
    let width = map.tensor_info("mm.2.weight").unwrap().dims[1] as usize;
    assert!(matches!(width, 2048 | 4096 | 5120));
    let vision = Vision::load(&device, Path::new(&path), width).unwrap();
    let red: Vec<u8> = (0..256 * 256).flat_map(|_| [255, 0, 0]).collect();
    let pattern: Vec<u8> = (0..288 * 288 * 3).map(|i| (i * 13 % 251) as u8).collect();
    let encode = |images: &[(&[u8], usize, usize)]| {
        let mut job = vision.start(&device, images).unwrap();
        loop {
            let output = vision.step(&device, &mut job).unwrap();
            assert!(
                unsafe { job.x.read_f32(0, job.rows * E) }
                    .iter()
                    .all(|v| v.is_finite()),
                "block {}",
                job.layer
            );
            if let Some(output) = output {
                return output;
            }
        }
    };
    let a = encode(&[(&red, 256, 256)]);
    let b = encode(&[(&pattern, 288, 288)]);
    // Submission grouping must not change any vision embedding bit. Include
    // ragged row counts and reverse image order below as separate coverage.
    let mut grouped = vision.start(&device, &[(&pattern, 288, 288)]).unwrap();
    let grouped_out = loop {
        if let Some(out) = vision
            .step_budget(&device, &mut grouped, PREFILL_QUANTUM)
            .unwrap()
        {
            break out;
        }
    };
    assert!(
        grouped.submits < 29,
        "blocks should share command submissions"
    );
    let mut fused = vision.start(&device, &[(&pattern, 288, 288)]).unwrap();
    let fused_out = vision
        .step_blocks(&device, &mut fused, 27, true)
        .unwrap()
        .unwrap();
    assert_eq!(fused.layer, 27);
    assert_eq!(
        fused.submits, 2,
        "setup plus blocks/merger; no publication-only yield"
    );
    assert_eq!(
        unsafe { fused_out[0].embd.read_f32(0, 81 * width) },
        unsafe { b[0].embd.read_f32(0, 81 * width) },
        "fused and independently submitted tower/merger must be byte-identical"
    );
    assert_eq!(
        unsafe { grouped_out[0].embd.read_f32(0, 81 * width) },
        unsafe { b[0].embd.read_f32(0, 81 * width) }
    );
    for inputs in [
        vec![(&red[..], 256, 256), (&pattern[..], 288, 288)],
        vec![(&pattern[..], 288, 288), (&red[..], 256, 256)],
    ] {
        let batched = encode(&inputs);
        for (i, o) in batched.iter().enumerate() {
            let reference = if inputs[i].1 == 256 { &a[0] } else { &b[0] };
            let actual = unsafe { o.embd.read_f32(0, o.nx * o.ny * width) };
            let expected = unsafe { reference.embd.read_f32(0, actual.len()) };
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("ragged tower image {i}: max error {max}");
            assert!(max < 0.02, "ragged tower image contamination {max}");
        }
    }
}

/// Diagnostic transport only: all model arithmetic uses the real GPU path.
/// Invoked by the greedy-parity harness (--vision-tensor-debug), whose newest
/// prebuilt GPU reference prints the same first/last-three tensor samples.
#[test]
#[ignore = "requires canonical mmproj and an M5; tensor diagnostic via greedy-parity.py"]
fn vision_white_tensor_capture() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("mmproj path");
    let device = MetalDevice::new(None).unwrap();
    let vision = Vision::load(&device, Path::new(&path), 5120).unwrap();
    assert_eq!(vision.mean, [0.5; 3]);
    assert_eq!(vision.std, [0.5; 3]);
    let white = vec![255u8; 256 * 256 * 3];
    let mut job = vision.start(&device, &[(&white, 256, 256)]).unwrap();
    let mut captures = serde_json::Map::new();
    let edge = |n: usize| (0..n).filter(move |&i| i < 3 || i >= n.saturating_sub(3));
    let mut capture =
        |name: String, buffer: &Buffer, dims: [usize; 3], strides: [usize; 3], offset: usize| {
            let len = offset
                + (dims[2] - 1) * strides[2]
                + (dims[1] - 1) * strides[1]
                + (dims[0] - 1) * strides[0]
                + 1;
            let values = unsafe { buffer.read_f32(0, len) };
            assert!(values.iter().all(|v| v.is_finite()));
            let mut samples = Vec::new();
            for k in edge(dims[2]) {
                for j in edge(dims[1]) {
                    for i in edge(dims[0]) {
                        samples.push(
                            values[offset + i * strides[0] + j * strides[1] + k * strides[2]],
                        );
                    }
                }
            }
            captures.insert(
                name,
                serde_json::json!({"dims": [dims[0],dims[1],dims[2],1], "samples": samples}),
            );
        };
    capture("inp_pos_emb".into(), &job.x, [E, job.rows, 1], [1, E, 0], 0);
    for layer in 0..27 {
        let b = &vision.blocks[layer];
        // Observe the real normalization separately. step repeats it with the
        // same input; this observation cannot affect the residual stream.
        let cmd = device.begin().unwrap();
        vision.ln(&cmd, &job.x, &b.ln1, &b.ln1b, &job.stage, job.rows);
        cmd.finish().unwrap();
        capture(
            format!("ln1-{layer}"),
            &job.stage,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
        if layer == 0 {
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "vis_bmm64",
                &[&b.qkv.buffer, &job.stage, &job.qkv, &b.qkvb.buffer],
                &[E as u32, (E * 3) as u32, job.rows as u32, 1],
                [(E * 3).div_ceil(64), job.rows.div_ceil(64), 1],
                128,
            );
            cmd.finish().unwrap();
            capture(
                "Qcur-strict-0".into(),
                &job.qkv,
                [72, 16, job.rows],
                [1, 72, E * 3],
                0,
            );
        }
        // Observe the attention boundary before the production step reuses
        // its scratch for LN2. All arithmetic is the native GPU path; this
        // repeated observation leaves the residual stream untouched.
        let cmd = device.begin().unwrap();
        Vision::mm::<true, false>(&cmd, &b.qkv, &job.stage, &job.qkv, &b.qkvb, job.rows, 1);
        cmd.dispatch(
            "vis_qkv",
            &[&job.qkv, &job.xy, &job.q, &job.k, &job.v],
            &[job.rows as u32],
            [((job.rows + 64) * 1280).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "vis_attention",
            &[&job.q, &job.k, &job.v, &job.attn, &job.tiles],
            &[0],
            [16, job.tile_count, 1],
            64,
        );
        cmd.finish().unwrap();
        capture(
            format!("kqv_out-{layer}"),
            &job.attn,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
        assert!(vision.step(&device, &mut job).unwrap().is_none());
        capture(
            format!("ffn_inp_normed-{layer}"),
            &job.attn,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
        for (name, offset) in [("Qcur", 0), ("Kcur", E), ("Vcur", E * 2)] {
            capture(
                format!("{name}-{layer}"),
                &job.qkv,
                [72, 16, job.rows],
                [1, 72, E * 3],
                offset,
            );
        }
        capture(
            format!("layer_out-{layer}"),
            &job.x,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
    }
    let outputs = vision.step(&device, &mut job).unwrap().unwrap();
    if let Some(path) = std::env::var_os("PADDOCK_METAL_EMBEDDING_CAPTURE") {
        use std::io::Write;
        // Serialize completed GPU output, never a host inference result.
        // create_new prevents an accidental overwrite of an earlier capture.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&64i32.to_le_bytes()).unwrap();
        file.write_all(&5120i32.to_le_bytes()).unwrap();
        let values = unsafe { outputs[0].embd.read_f32(0, 64 * 5120) };
        file.write_all(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    capture(
        "merger_output".into(),
        &outputs[0].embd,
        [5120, 64, 1],
        [1, 5120, 0],
        0,
    );
    eprintln!("VISION_TENSORS {}", serde_json::Value::Object(captures));
}

/// Encoder-only cost curve; deliberately separate from HTTP TTFT and from
/// tensor-capture instrumentation. Every repetition starts a cold GPU job.
#[test]
#[ignore = "requires canonical mmproj and an M5; GPU cost diagnostic"]
fn vision_encoder_cost_curve() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("mmproj path");
    let device = MetalDevice::new(None).unwrap();
    let vision = Vision::load(&device, Path::new(&path), 5120).unwrap();
    for side in [256, 768, 1024] {
        let rgb: Vec<u8> = (0..side * side * 3).map(|i| (i * 13 % 251) as u8).collect();
        for repeat in 0..3 {
            let start = std::time::Instant::now();
            let mut job = vision.start(&device, &[(&rgb, side, side)]).unwrap();
            while vision
                .step_blocks(&device, &mut job, 27, true)
                .unwrap()
                .is_none()
            {}
            eprintln!(
                "VISION_COST {}",
                serde_json::json!({"side":side,"repeat":repeat,
                "gpu_ms":job.gpu_seconds*1000.0,"wall_ms":start.elapsed().as_secs_f64()*1000.0})
            );
        }
    }
}
