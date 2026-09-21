use super::*;
use paddock_engine::{
    generator::{Generator, MmAdmit},
    service::MmChunk,
};
const MODEL: &str = concat!(
    env!("HOME"),
    "/paddock/models/Unlimited-OCR-GGUF/Unlimited-OCR-Q8_0.gguf"
);
const TOWER: &str = concat!(
    env!("HOME"),
    "/paddock/models/Unlimited-OCR-GGUF/mmproj-Unlimited-OCR-F16.gguf"
);
fn model() -> UnlimitedOcr {
    UnlimitedOcr::load(Path::new(MODEL), 4096, 4, None).unwrap()
}
fn delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
fn image(w: usize, h: usize) -> Vec<MmChunk> {
    vec![
        MmChunk::Text(vec![0]),
        MmChunk::Image {
            rgb: (0..w * h * 3)
                .map(|i| ((i * 131 + 7) % 256) as u8)
                .collect(),
            w,
            h,
        },
        MmChunk::Text(vec![110, 111, 112]),
    ]
}
#[test]
#[ignore = "requires elected R2 Q8 checkpoint and Apple10 GPU"]
fn decoder_lifecycle_mixed_and_ring() {
    let mut m = model();
    assert_eq!(m.weight_bytes, 3120896000);
    assert!(UnlimitedOcr::load(Path::new(MODEL), 0, 4, None).is_err());
    assert!(UnlimitedOcr::load(Path::new(MODEL), 4096, 17, None).is_err());
    assert!(UnlimitedOcr::load(Path::new(MODEL), 4096, 4, Some(1024)).is_err());
    let tokens = (0..65).map(|i| 110 + i * 7 % 137).collect::<Vec<_>>();
    let full = m.forward_prefill(0, &tokens).unwrap();
    let baseline = m.device.allocated_bytes();
    m.reset();
    m.slots[0].prompt = tokens.len();
    let mut scan = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        scan = m.execute(&[(0, token, i as u32)], &[0]).unwrap();
    }
    eprintln!("Unlimited whole/scan max={}", delta(&full, &scan));
    assert!(delta(&full, &scan) < 0.05);
    m.reset();
    m.prefill_begin(0, tokens.clone()).unwrap();
    assert!(m.prefill_begin(0, tokens.clone()).is_err());
    assert!(m.forward_prefill(0, &tokens).is_err());
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 7).unwrap().1);
    }
    assert_eq!(done.len(), 1);
    assert!(delta(&full, &done[0].1) < 0.05);
    let seq = [127, 131, 137, 139, 149];
    m.forward_prefill(1, &seq).unwrap();
    let dense = m.execute(&[(1, 151, 5)], &[0]).unwrap();
    m.forward_prefill(1, &seq).unwrap();
    m.prefill_begin(2, (0..127).map(|i| 150 + i % 19).collect())
        .unwrap();
    let mixed = m.forward_mixed(&[(1, 151, 5)], 128).unwrap().0;
    eprintln!("Unlimited mixed max={}", delta(&dense, &mixed));
    assert!(delta(&dense, &mixed) < 0.05);
    assert!(m.prefill_abort(2));
    // Each slot wraps only generated KV. Live block counts plateau after
    // prompt+128, and multiple complete wraps preserve true rotary positions.
    let mut blocks = 0;
    for pos in 6..300 {
        let logits = m
            .execute(&[(1, 200 + (pos % 17) as u32, pos as u32)], &[0])
            .unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
        if pos == 133 {
            blocks = m.slots[1].table.blocks().len();
        }
        if pos > 133 {
            assert_eq!(blocks, m.slots[1].table.blocks().len());
        }
    }
    assert_eq!(blocks, 9);
    assert_eq!(m.slots[1].history.len(), 300);
    m.publish(1);
    assert!(
        m.radix.match_prefix(&m.slots[1].history).is_empty(),
        "generated ring KV must never enter the prompt radix"
    );
    m.release_inactive_slots(&[]);
    assert_eq!(m.device.allocated_bytes(), baseline);
    assert!(m.execute(&[(4, 1, 0)], &[0]).is_err());
    assert!(m.execute(&[(0, VOCAB as u32, 0)], &[0]).is_err());
    assert!(m.forward_batch(&[1], &[]).is_err());
}

#[test]
#[ignore = "requires R2 decoder/tower and Apple10 GPU"]
fn maximum_32_tile_image_and_admission_bounds() {
    let mut m = model();
    m.attach_vision(Path::new(TOWER)).unwrap();
    assert_eq!(m.weight_bytes, 3120896000 + 822808576);
    let baseline = m.device.allocated_bytes();
    for chunks in [
        image(0, 2),
        image(1, 513),
        image(8193, 1),
        vec![MmChunk::Text(vec![1])],
    ] {
        let result = m.prefill_begin_multimodal(vec![(0, chunks)]);
        assert!(matches!(result[0].1, MmAdmit::Failed(_)));
        assert!(!m.encoding_pending());
        assert_eq!(baseline, m.device.allocated_bytes());
    }
    let result = m.prefill_begin_multimodal(vec![(0, image(256, 8192))]);
    assert!(matches!(result[0].1, MmAdmit::Encoding));
    assert_eq!(m.slots[0].prompt, 3797);
    let mut peak = baseline;
    while m.encoding_pending() {
        let out = m.encode_step();
        assert!(out.iter().all(|(_, a)| matches!(a, MmAdmit::Queued)));
        peak = peak.max(m.device.allocated_bytes());
    }
    while !m.pending.is_empty() {
        let out = m.forward_mixed(&[], 512).unwrap().1;
        assert!(
            out.iter()
                .all(|(_, logits, _)| logits.iter().all(|x| x.is_finite()))
        );
    }
    eprintln!(
        "Unlimited max32 live additional bytes={} bound={}",
        peak - baseline,
        vision::Vision::workspace_bound()
    );
    assert!(peak - baseline < vision::Vision::workspace_bound());
    m.release_inactive_slots(&[]);
    assert_eq!(baseline, m.device.allocated_bytes());
}
#[test]
#[ignore = "requires R2 decoder/tower and Apple10 GPU"]
fn tower_base_and_tiled_lifecycle() {
    let mut m = model();
    m.attach_vision(Path::new(TOWER)).unwrap();
    assert!(m.attach_vision(Path::new(TOWER)).is_err());
    let baseline = m.device.allocated_bytes();
    let (full, n) = m.forward_prefill_multimodal(0, &image(336, 392)).unwrap();
    assert_eq!(n, 277);
    assert!(full.iter().all(|x| x.is_finite()));
    m.reset();
    assert_eq!(baseline, m.device.allocated_bytes());
    let admitted = m.prefill_begin_multimodal(vec![
        (0, image(700, 1000)),
        (1, image(336, 392)),
        (2, image(336, 392)),
    ]);
    assert!(admitted.iter().all(|(_, a)| matches!(a, MmAdmit::Encoding)));
    assert!(m.forward_prefill(0, &[1, 2]).is_err());
    m.encode_step();
    m.encode_step();
    assert!(m.prefill_abort(2));
    m.forward_prefill(3, &[101, 102, 103]).unwrap();
    let expected = m.execute(&[(3, 104, 3)], &[0]).unwrap();
    m.forward_prefill(3, &[101, 102, 103]).unwrap();
    m.encode_step();
    let actual = m.forward_mixed(&[(3, 104, 3)], 0).unwrap().0;
    assert_eq!(delta(&expected, &actual), 0.);
    let mut queued = Vec::new();
    while m.encoding_pending() {
        queued.extend(m.encode_step());
    }
    assert_eq!(queued.len(), 2);
    assert!(queued.iter().all(|(_, a)| matches!(a, MmAdmit::Queued)));
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 127).unwrap().1);
    }
    assert_eq!(done.len(), 2);
    for (slot, logits, _) in done {
        assert!(logits.iter().all(|x| x.is_finite()));
        if slot == 1 {
            eprintln!("Unlimited image chunk delta={}", delta(&full, &logits));
            assert!(delta(&full, &logits) < 0.05);
        }
    }
    assert!(m.radix.match_prefix(&vec![IMAGE; 512]).is_empty());
    m.release_inactive_slots(&[]);
    assert_eq!(baseline, m.device.allocated_bytes());
    m.prefill_begin_multimodal(vec![(0, image(1240, 1754))]);
    m.encode_step();
    m.encode_step();
    m.reset();
    assert_eq!(baseline, m.device.allocated_bytes());
}

#[test]
fn relative_attention_matches_independent_gpu_scan() {
    let d = MetalDevice::new(Some(192 << 20)).unwrap();
    // Window14 has a 4-key final tile; grid20 has 16. Two different images
    // detect cross-image reads. Bias swings force nontrivial online rescaling.
    for side in [14usize, 20] {
        let rows = 2 * side * side;
        let plane = |seed: usize| {
            d.upload(
                &(0..(rows + 64) * 768)
                    .flat_map(|i| {
                        let v = if i / 768 < rows {
                            ((i * seed % 61) as f32 - 30.0) / 32.0
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
        let bias = |seed: usize| {
            d.upload(
                &(0..rows * 12 * side)
                    .flat_map(|i| (((i * seed % 127) as f32 - 63.0) / 8.0).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let (rh, rw) = (bias(13), bias(17));
        let mut tiles = Vec::new();
        for image in 0..2 {
            for row in (0..side * side).step_by(32) {
                for x in [
                    image * side * side + row,
                    (side * side - row).min(32),
                    image * side * side,
                    side * side,
                ] {
                    tiles.extend_from_slice(&(x as u32).to_le_bytes());
                }
            }
        }
        let t = d.upload(&tiles).unwrap();
        let actual = d
            .upload(
                &(0..(rows + 32) * 768)
                    .flat_map(|_| f32::NAN.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let expected = d.alloc(rows * 768 * 4).unwrap();
        let c = d.begin().unwrap();
        c.dispatch(
            "uov_sam_attention",
            &[&q, &k, &v, &actual, &t, &rh, &rw],
            &[0, side as u32],
            [12, tiles.len() / 16, 1],
            64,
        );
        c.dispatch(
            "uov_sam_check",
            &[&q, &k, &v, &expected, &rh, &rw],
            &[side as u32],
            [12, rows, 1],
            32,
        );
        c.finish().unwrap();
        // SAFETY: command completed, allocated output spans checked above.
        let a = unsafe { actual.read_f32(0, (rows + 32) * 768) };
        let b = unsafe { expected.read_f32(0, rows * 768) };
        let max = delta(&a[..rows * 768], &b);
        eprintln!("SAM side={side} GPU scan delta={max}");
        assert!(
            max < 0.00001,
            "relative attention indexing/accumulation {max}"
        );
        assert!(a[rows * 768..].iter().all(|v| v.is_nan()));
    }
}

#[test]
fn pixel_patches_match_pillow_pad() {
    use objc2_metal::MTLBuffer;
    // Pillow ImageOps.pad RGB9x5 ->16x16, BICUBIC, gray127; pixels only,
    // never a CPU model oracle. Input byte i=(131*i+7)%256. Stored HWC.
    const GOLDEN: [u8; 768] = [
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 0, 154, 0, 41, 96, 47, 143, 4,
        149, 84, 73, 90, 9, 160, 15, 108, 71, 113, 157, 31, 166, 52, 145, 75, 56, 157, 80, 171, 66,
        180, 142, 109, 147, 65, 195, 71, 140, 131, 146, 200, 81, 206, 122, 170, 128, 73, 225, 79,
        64, 131, 67, 87, 111, 93, 130, 79, 136, 111, 108, 117, 85, 145, 91, 126, 114, 138, 148,
        103, 145, 111, 155, 50, 118, 133, 58, 168, 41, 164, 117, 88, 128, 41, 180, 47, 121, 111,
        127, 184, 57, 190, 101, 152, 107, 48, 210, 54, 210, 87, 216, 169, 134, 175, 104, 210, 110,
        156, 169, 162, 220, 116, 226, 156, 189, 180, 129, 229, 105, 215, 169, 14, 226, 90, 26, 158,
        7, 134, 74, 57, 98, 9, 150, 15, 90, 80, 96, 154, 25, 160, 69, 121, 75, 16, 181, 22, 224,
        43, 230, 168, 110, 174, 79, 219, 85, 146, 164, 152, 231, 89, 237, 143, 186, 161, 105, 236,
        92, 218, 144, 88, 228, 93, 97, 129, 95, 115, 98, 107, 115, 103, 124, 109, 118, 119, 124,
        131, 117, 137, 125, 134, 131, 120, 144, 125, 179, 33, 185, 130, 88, 136, 52, 177, 58, 112,
        127, 118, 187, 63, 193, 111, 149, 117, 78, 192, 84, 178, 102, 184, 183, 107, 189, 93, 207,
        99, 136, 174, 142, 222, 98, 228, 158, 173, 164, 108, 233, 114, 197, 155, 203, 252, 106,
        255, 154, 159, 160, 98, 166, 104, 9, 161, 15, 76, 82, 82, 161, 19, 167, 74, 117, 79, 35,
        165, 43, 149, 61, 166, 153, 70, 171, 50, 192, 57, 104, 158, 109, 210, 72, 216, 139, 155,
        145, 82, 221, 88, 180, 133, 186, 241, 79, 246, 125, 255, 131, 84, 225, 90, 19, 140, 25, 70,
        67, 76, 134, 30, 140, 71, 104, 77, 44, 141, 50, 129, 65, 135, 134, 71, 140, 59, 157, 65,
        97, 120, 103, 170, 45, 176, 104, 120, 110, 55, 180, 61, 144, 102, 150, 199, 53, 205, 81,
        234, 87, 107, 187, 113, 150, 104, 156, 131, 123, 137, 105, 165, 111, 146, 134, 152, 168,
        122, 174, 131, 169, 137, 136, 179, 142, 183, 147, 189, 181, 78, 187, 143, 15, 149, 64, 90,
        70, 25, 150, 31, 114, 72, 120, 169, 23, 175, 58, 213, 64, 122, 162, 128, 225, 84, 231, 166,
        156, 172, 90, 241, 96, 189, 152, 195, 239, 113, 245, 133, 228, 139, 138, 241, 144, 254,
        143, 255, 229, 57, 235, 130, 0, 136, 44, 75, 50, 10, 135, 16, 99, 57, 105, 154, 8, 160,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    ];
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let src = d
        .upload(
            &(0..9 * 5 * 3)
                .map(|i| ((131 * i + 7) % 256) as u8)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let cx = d.alloc(16 * 8 * 4).unwrap();
    let cy = d.alloc(9 * 8 * 4).unwrap();
    let temp = d.alloc(16 * 5 * 3).unwrap();
    let out = d.alloc(768 * 2).unwrap();
    let c = d.begin().unwrap();
    c.dispatch("gv_coeff", &[&cx], &[9, 16, 8], [1, 1, 1], 256);
    c.dispatch("gv_coeff", &[&cy], &[5, 9, 8], [1, 1, 1], 256);
    c.dispatch(
        "gv_resize_h",
        &[&src, &cx, &temp],
        &[9, 16, 5, 8],
        [1, 1, 1],
        256,
    );
    c.dispatch(
        "uov_patches",
        &[&temp, &cy, &out],
        &[16, 9, 16, 8, 0, 0, 0, 4, 1, 0],
        [3, 1, 1],
        256,
    );
    c.finish().unwrap();
    // SAFETY: the completed GPU command wrote the entire allocated span.
    let pixels =
        unsafe { std::slice::from_raw_parts(out.raw.contents().as_ptr().cast::<half::f16>(), 768) };
    for (i, p) in pixels.iter().enumerate() {
        let byte = ((p.to_f32() * 0.5 + 0.5) * 255.0).round() as u8;
        assert_eq!(
            byte,
            GOLDEN[(i % 256) * 3 + i / 256],
            "pixel channel index {i}"
        );
    }
}
