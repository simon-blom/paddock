use super::*;

#[test]
fn granite_qformer_attention_matches_independent_gpu() {
    let d = MetalDevice::new(None).unwrap();
    for kl in [16, 64] {
        let qr = 3 * 16;
        let kr = 3 * kl;
        let data = |rows: usize, salt: usize| {
            let f = (0..rows * 1152)
                .map(|i| (((i * 13 + salt * 17) % 199) as f32 - 99.) / 99.)
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let x = d.upload(&f).unwrap();
            let h = d.alloc(rows * 1152 * 2).unwrap();
            let c = d.begin().unwrap();
            c.dispatch(
                "grv_half",
                &[&x, &h],
                &[(rows * 1152) as u32],
                [(rows * 1152).div_ceil(256), 1, 1],
                256,
            );
            c.finish().unwrap();
            h
        };
        let q = data(qr, 1);
        let k = data(kr, 2);
        let v = data(kr, 3);
        let a = d.alloc(qr * 1152 * 4).unwrap();
        let b = d.alloc(a.len()).unwrap();
        let indices = (0..3)
            .flat_map(|i| [(i * 16) as u32, 16, (i * kl) as u32, kl as u32])
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let tiles = d.upload(&indices).unwrap();
        let bounds = (0..qr)
            .flat_map(|r| [((r / 16) * kl) as u32, ((r / 16 + 1) * kl) as u32])
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let bounds = d.upload(&bounds).unwrap();
        let c = d.begin().unwrap();
        c.dispatch(
            "grv_qattention",
            &[&q, &k, &v, &a, &tiles],
            &[0],
            [18, 3, 1],
            64,
        );
        c.dispatch(
            "grv_qattention_check",
            &[&q, &k, &v, &b, &bounds],
            &[0],
            [18, qr, 1],
            32,
        );
        c.finish().unwrap();
        let a = unsafe { a.read_f32(0, qr * 1152) };
        let b = unsafe { b.read_f32(0, qr * 1152) };
        assert!(a.iter().chain(&b).all(|v| v.is_finite()));
        let e = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("Q-Former keys={kl} MPP/witness max_abs={e}");
        assert!(e < 0.001);
    }
}

use paddock_engine::{generator::MmAdmit, service::MmChunk};
fn path(var: &str) -> std::path::PathBuf {
    std::env::var_os(var).expect(var).into()
}
#[test]
#[ignore = "requires Granite Vision 4.1 F16 mmproj and M5"]
fn granite_vision_tower_batch_and_abort() {
    let d = MetalDevice::new(None).unwrap();
    let v = vision::Vision::load(&d, &path("PADDOCK_GRANITE_MMPROJ"), 2560).unwrap();
    let images = [(384, 384), (641, 479)].map(|(w, h)| {
        (
            (0..w * h * 3)
                .map(|i| ((i * 71 + i / 19) % 256) as u8)
                .collect::<Vec<_>>(),
            w,
            h,
        )
    });
    let finish = |mut j: vision::Job| loop {
        if let Some(o) = v.step(&d, &mut j).unwrap() {
            eprintln!("Granite tower GPU {}ms", j.gpu_seconds * 1000.);
            break o;
        }
    };
    let single = images
        .iter()
        .map(|(rgb, w, h)| finish(v.start(&d, &[(rgb, *w, *h)]).unwrap()).remove(0))
        .collect::<Vec<_>>();
    let refs = images
        .iter()
        .map(|(rgb, w, h)| (&**rgb, *w, *h))
        .collect::<Vec<_>>();
    let batched = finish(v.start(&d, &refs).unwrap());
    for (a, b) in single.iter().zip(&batched) {
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.streams.len(), 8);
        for (i, (a, b)) in a.streams.iter().zip(&b.streams).enumerate() {
            let x = unsafe { a.read_f32(0, a.len() / 4) };
            let y = unsafe { b.read_f32(0, b.len() / 4) };
            assert!(x.iter().chain(&y).all(|v| v.is_finite()));
            let max = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("stream {i} grouped max_abs={max}");
            assert!(max < 0.0001);
        }
    }
    let before = d.allocated_bytes();
    let mut j = v.start(&d, &refs).unwrap();
    assert!(v.step(&d, &mut j).unwrap().is_none());
    drop(j);
    assert_eq!(before, d.allocated_bytes());
}
#[test]
#[ignore = "requires Granite Vision 4.1 Q8_0 target and F16 mmproj"]
fn granite_vision_prefill_chunks_cache_and_cancellation() {
    let mut m = Granite::load(&path("PADDOCK_METAL_TEST_MODEL"), 2048, 4, None).unwrap();
    m.attach_vision(&path("PADDOCK_GRANITE_MMPROJ")).unwrap();
    let prompt = |colour: u8| {
        vec![
            MmChunk::Text(vec![53; 19]),
            MmChunk::Image {
                rgb: vec![colour; 384 * 384 * 3],
                w: 384,
                h: 384,
            },
            MmChunk::Text(vec![22; 17]),
        ]
    };
    let p = prompt(255);
    let (baseline, n) = m.prefill_images(0, &p).unwrap();
    assert_eq!(n, 336);
    assert!(baseline.iter().all(|v| v.is_finite()));
    m.reset();
    let (again, _) = m.prefill_images(2, &p).unwrap();
    assert!(m.take_prefill_reused(2) >= 320);
    assert_eq!(m.image_cache_reuses(), 1);
    let err = baseline
        .iter()
        .zip(&again)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 0.08, "repeat {err}");
    m.reset();
    assert!(
        m.admit_images(vec![(1, prompt(0)), (3, p.clone())])
            .iter()
            .all(|(_, v)| matches!(v, MmAdmit::Encoding))
    );
    while m.encoding_pending() {
        assert!(
            m.step_images()
                .iter()
                .all(|(_, v)| matches!(v, MmAdmit::Queued))
        );
    }
    assert_eq!(
        m.take_prefill_reused(1),
        16,
        "different pixels cannot reuse image KV"
    );
    let mut completed = Vec::new();
    for _ in 0..30 {
        completed.extend(m.forward_mixed(&[], 31).unwrap().1);
        if completed.len() == 2 {
            break;
        }
    }
    assert_eq!(completed.len(), 2);
    assert!(m.pending.is_empty());
    let text = (100..149).collect::<Vec<_>>();
    m.prefill_begin(0, text).unwrap();
    m.admit_images(vec![(2, prompt(127))]);
    m.encode_step();
    assert!(m.prefill_abort(2));
    assert!(!m.image_slot_pending(2));
    assert!(m.slots[2].mm.is_none());
    while !m.pending.is_empty() {
        m.forward_mixed(&[], 17).unwrap();
    }
    m.release_inactive_slots(&[false; 4]);
    assert!(m.slots.iter().all(|s| s.mm.is_none()));
    assert!(matches!(
        m.admit_images(vec![(
            0,
            vec![MmChunk::Image {
                rgb: vec![],
                w: 384,
                h: 384
            }]
        )])
        .remove(0)
        .1,
        MmAdmit::Failed(_)
    ));
}
