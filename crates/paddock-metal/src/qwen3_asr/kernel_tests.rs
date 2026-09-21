use super::*;
#[test]
fn qasr_audio_geometry() {
    for (frames, n) in [
        (1, 1),
        (99, 13),
        (100, 13),
        (101, 14),
        (800, 104),
        (801, 105),
    ] {
        assert_eq!(paddock_engine::audio::audio_token_count(frames), n);
    }
    let m = paddock_engine::audio::MelFeatures {
        data: vec![0.; 128 * 100],
        n_frames: 101,
        n_samples: 16000,
        global_max: 0.,
    };
    assert!(audio::validate(&m).is_err());
}

#[test]
fn qasr_window_attention_matches_independent_gpu_scan() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let sizes = [13, 104, 103, 7];
    let rows = sizes.iter().sum::<usize>();
    let plane = |seed: usize| {
        d.upload(
            &(0..(rows + 64) * 1024)
                .flat_map(|i| {
                    half::f16::from_f32(if i / 1024 < rows {
                        ((i * seed % 31) as f32 - 15.) / 8.
                    } else {
                        0.
                    })
                    .to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (q, k, v) = (plane(3), plane(7), plane(11));
    let mut tiles = Vec::new();
    let mut bounds = Vec::new();
    let mut start = 0;
    for n in sizes {
        for at in (0..n).step_by(32) {
            tiles.extend([
                (start + at) as u32,
                (n - at).min(32) as u32,
                start as u32,
                n as u32,
            ]);
        }
        for _ in 0..n {
            bounds.extend([start as u32, (start + n) as u32]);
        }
        start += n;
    }
    let words = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let t = words(&tiles);
    let b = words(&bounds);
    let out = words(&vec![f32::NAN.to_bits(); (rows + 32) * 1024]);
    let scan = d.alloc(rows * 1024 * 4).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "qasr_attention",
        &[&q, &k, &v, &out, &t],
        &[0],
        [16, tiles.len() / 4, 1],
        64,
    );
    c.dispatch(
        "qasr_attention_check",
        &[&q, &k, &v, &scan, &b],
        &[],
        [16, rows, 1],
        32,
    );
    c.finish().unwrap();
    let a = unsafe { out.read_f32(0, rows * 1024) };
    let b = unsafe { scan.read_f32(0, rows * 1024) };
    let delta = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("Qwen3-ASR attention delta {delta}");
    assert!(a.iter().all(|v| v.is_finite()));
    assert!(delta < 0.0001);
    assert!(
        unsafe { out.read_f32(rows * 1024, 32 * 1024) }
            .iter()
            .all(|x| x.is_nan())
    );
}

#[test]
fn qasr_conv_gather_preserves_chunk_boundaries_and_padding() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let input = d
        .upload(
            &(0..2 * 100 * 128)
                .flat_map(|i| (i as f32 + 1.).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let output = d.alloc(50 * 64 * 9 * 4).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "qasr_conv_rows",
        &[&input, &output],
        &[128, 100, 1, 1, 0, 1],
        [(50usize * 64 * 9).div_ceil(256), 1, 1],
        256,
    );
    c.finish().unwrap();
    let first = unsafe { output.read_f32(0, 9) };
    assert_eq!(
        first,
        vec![0., 0., 0., 0., 12801., 12929., 0., 12802., 12930.]
    );
    // Last output cell reads input times 97..99 and frequencies 125..127;
    // no borrowed row from the preceding/following conv chunk.
    let last = unsafe { output.read_f32((50 * 64 - 1) * 9, 9) };
    assert_eq!(last[0], (12800 + 97 * 128 + 125 + 1) as f32);
    assert_eq!(last[8], 25600.);
}
