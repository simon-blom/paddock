use super::*;

#[test]
#[ignore = "requires DiffusionGemma MLX and PADDOCK_DG_VISION_DIR"]
fn diffusion_vision_reference_capture() {
    use std::io::Write;
    let path = std::env::var("PADDOCK_DG_MODEL").unwrap();
    let out = std::env::var("PADDOCK_DG_VISION_DIR").unwrap();
    let source = super::super::mlx::Source::open(Path::new(&path)).unwrap();
    let device = MetalDevice::new(None).unwrap();
    let vision = Vision::load_mlx_at(&device, &source, "model.encoder.", 2816).unwrap();
    for (case, (w, h)) in [(96, 96), (511, 317), (1920, 1080)].into_iter().enumerate() {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                rgb.extend([
                    (x * 17 + y * 3) as u8,
                    (x * 5 + y * 11) as u8,
                    (x ^ y) as u8,
                ]);
            }
        }
        let started = std::time::Instant::now();
        let mut job = vision.start(&device, &[(&rgb, w, h)]).unwrap();
        let capture = |stage: usize, job: &Job| {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(Path::new(&out).join(format!("vision-{case}-layer-{stage}.f32")))
                .unwrap();
            let bytes: Vec<_> = unsafe { job.x.read_f32(0, job.rows * E) }
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            file.write_all(&bytes).unwrap();
        };
        capture(0, &job);
        let output = loop {
            if let Some(outputs) = vision
                .step(&device, &mut job, Duration::from_millis(10))
                .unwrap()
            {
                capture(27, &job);
                break outputs;
            }
            if job.layer == 1 {
                capture(1, &job);
                for (name, buffer) in [("qh", &job.qh), ("kh", &job.kh), ("vh", &job.vh)] {
                    // SAFETY: the layer submission is complete. Two BF16
                    // values share each word in the padded head buffer.
                    let words = unsafe { buffer.read_u32(job.rows * 16 * 80 / 2) };
                    let bytes: Vec<_> = words
                        .iter()
                        .flat_map(|v| [f32::from_bits(v << 16), f32::from_bits(v & 0xffff0000)])
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(Path::new(&out).join(format!("vision-{case}-{name}.f32")))
                        .unwrap();
                    file.write_all(&bytes).unwrap();
                }
                for (name, buffer, count) in [
                    ("q", &job.q, job.rows * E),
                    ("attn", &job.attn, job.rows * E),
                    ("gate", &job.gate, job.rows * F),
                    ("down", &job.delta, job.rows * E),
                ] {
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(Path::new(&out).join(format!("vision-{case}-{name}.f32")))
                        .unwrap();
                    let bytes: Vec<_> = unsafe { buffer.read_f32(0, count) }
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    file.write_all(&bytes).unwrap();
                }
            }
        };
        let values = unsafe { output[0].embd.read_f32(0, output[0].tokens * 2816) };
        assert!(values.iter().all(|v| v.is_finite()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&out).join(format!("vision-{case}.f32")))
            .unwrap();
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        file.write_all(&bytes).unwrap();
        eprintln!(
            "DG_VISION case={case} rows={} gpu_ms={} wall_ms={}",
            output[0].tokens,
            job.gpu_seconds * 1000.,
            started.elapsed().as_secs_f64() * 1000.
        );
        let mut warm = Vec::new();
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let mut job = vision.start(&device, &[(&rgb, w, h)]).unwrap();
            while vision
                .step(&device, &mut job, Duration::from_millis(10))
                .unwrap()
                .is_none()
            {}
            warm.push(serde_json::json!({"gpu_ms":job.gpu_seconds * 1000., "wall_ms":started.elapsed().as_secs_f64() * 1000.}));
        }
        let report = serde_json::json!({"case":case,"warm":warm});
        eprintln!("DG_VISION_WARM {report}");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&out).join(format!("native-{case}.json")))
            .unwrap();
        serde_json::to_writer_pretty(&mut file, &report).unwrap();
    }
}

#[test]
fn gemma_vision_channel_metadata_is_fail_closed() {
    assert!(channels(Some(&Value::Array(vec![Value::F32(0.); 3])), 0.));
    assert!(channels(Some(&Value::Array(vec![Value::F32(1.); 3])), 1.));
    for v in [
        None,
        Some(Value::F32(0.)),
        Some(Value::Array(vec![])),
        Some(Value::Array(vec![Value::F32(0.); 2])),
        Some(Value::Array(vec![Value::F32(f32::NAN); 3])),
        Some(Value::Array(vec![Value::F32(0.5); 3])),
    ] {
        assert!(!channels(v.as_ref(), 0.));
    }
}

#[test]
fn gemma_image_resize_bounds() {
    assert_eq!(resize(768, 768).unwrap(), (768, 768));
    assert_eq!(resize(256, 256).unwrap(), (432, 432));
    assert_eq!(resize(1024, 1024).unwrap(), (768, 768));
    for (w, h) in [
        (1, 1),
        (73, 177),
        (1279, 720),
        (1920, 1080),
        (21, 379),
        (800, 600),
    ] {
        let (x, y) = resize(w, h).unwrap();
        assert_eq!(x % 48, 0);
        assert_eq!(y % 48, 0);
        assert!(x * y <= BUDGET.max_pixels as usize);
    }
    for (w, h) in [(0, 1), (1, 0), (usize::MAX, 1), (1, 65_536)] {
        assert!(resize(w, h).is_err());
    }
}

#[test]
#[ignore = "requires canonical Gemma 4 BF16 mmproj"]
fn gemma_vision_load_and_white() {
    let path = std::env::var("PADDOCK_GEMMA4_MMPROJ").expect("mmproj");
    let device = MetalDevice::new(None).unwrap();
    let vision = Vision::load(&device, Path::new(&path), 5376).unwrap();
    let image = vec![255u8; 256 * 256 * 3];
    let mut job = vision.start(&device, &[(&image, 256, 256)]).unwrap();
    let mut captures = serde_json::Map::new();
    let mut capture = |name: String, buffer: &Buffer| {
        let values = unsafe { buffer.read_f32(0, 729 * E) };
        let mut samples = Vec::new();
        for r in [0, 1, 2, 726, 727, 728] {
            for d in [0, 1, 2, E - 3, E - 2, E - 1] {
                samples.push((values[r * E + d] * 10000.).round() / 10000.);
            }
        }
        captures.insert(
            name,
            serde_json::json!({"dims":[E,729,1,1],"samples":samples}),
        );
    };
    capture("pos_embd".into(), &job.x);
    let cmd = device.begin().unwrap();
    vision.norm(
        &cmd,
        &job.x,
        &vision.blocks[0].norm,
        &job.stage,
        job.rows,
        true,
    );
    cmd.finish().unwrap();
    capture("layer_inp_normed-0".into(), &job.stage);
    let output = loop {
        if let Some(v) = vision.step(&device, &mut job, Duration::ZERO).unwrap() {
            capture(format!("layer_out-{}", job.layer - 1), &job.x);
            break v;
        }
        capture(format!("layer_out-{}", job.layer - 1), &job.x);
        if job.layer == 1 {
            for (name, buffer) in [
                ("projection_q-0", &job.q),
                ("projection_k-0", &job.k),
                ("projection_v-0", &job.v),
                ("kqv_out-0", &job.attn),
            ] {
                capture(name.into(), buffer);
            }
        }
    };
    eprintln!("VISION_TENSORS {}", serde_json::Value::Object(captures));
    assert_eq!(output[0].tokens, 81);
    // SAFETY: the encoder has completed its final submission.
    let values = unsafe { output[0].embd.read_f32(0, 81 * 5376) };
    assert!(values.iter().all(|v| v.is_finite()));
    eprintln!(
        "GEMMA_VISION gpu_ms={} first={:?}",
        job.gpu_seconds * 1000.,
        &values[..12]
    );
    if let Some(path) = paddock_models::dev_var_os!("PADDOCK_METAL_EMBEDDING_CAPTURE") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&81i32.to_le_bytes()).unwrap();
        file.write_all(&5376i32.to_le_bytes()).unwrap();
        for v in values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}
