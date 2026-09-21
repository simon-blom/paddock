use super::*;
use paddock_engine::generator::Generator;
const MODEL: &str = concat!(
    env!("HOME"),
    "/paddock/models/Qwen3-ASR-1.7B-GGUF/Qwen3-ASR-1.7B-Q8_0.gguf"
);
const TOWER: &str = concat!(
    env!("HOME"),
    "/paddock/models/Qwen3-ASR-1.7B-GGUF/mmproj-Qwen3-ASR-1.7B-bf16.gguf"
);
#[test]
#[ignore = "real elected Q8 weights and Apple10 GPU"]
fn qasr_decoder_lifecycle() {
    let mut m = Qwen3Asr::load(Path::new(MODEL), 2048, 4, None).unwrap();
    let ids = vec![1000; 65];
    let a = m.forward_prefill(0, &ids).unwrap();
    m.reset();
    let mut b = Vec::new();
    for &t in &ids {
        b = m.forward(t).unwrap();
    }
    let delta = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("Qwen3-ASR chunk delta {delta}");
    assert!(delta < 0.01);
    for slot in 0..4 {
        m.forward_prefill(slot, &ids).unwrap();
    }
    let out = m.forward_batch(&[2000; 4], &[65; 4]).unwrap();
    for x in out.chunks(VOCAB) {
        assert!(x.iter().all(|v| v.is_finite()));
        assert_eq!(x, &out[..VOCAB]);
    }
    m.prefill_begin(0, vec![2000; 80]).unwrap();
    assert!(m.prefill_begin(0, vec![2]).is_err());
    assert!(m.prefill_abort(0));
    m.release_inactive_slots(&[false; 4]);
    assert!(m.slots.iter().all(|s| s.history.is_empty()));
    m.forward_prefill(1, &[100, 101, 102]).unwrap();
    let expected = m.forward_mixed(&[(1, 103, 3)], 0).unwrap().0;
    m.forward_prefill(1, &[100, 101, 102]).unwrap();
    m.prefill_begin(0, vec![77; 127]).unwrap();
    assert!(m.forward_prefill(0, &[77]).is_err());
    let actual = m.forward_mixed(&[(1, 103, 3)], 128).unwrap().0;
    let delta = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("Qwen3-ASR mixed decode delta {delta}");
    assert!(delta < 0.01);
    assert!(m.forward_mixed(&[(1, 104, 4), (1, 105, 5)], 1).is_err());
    assert!(m.forward_prefill(4, &[1]).is_err());
    assert!(m.forward_prefill(0, &[VOCAB as u32]).is_err());
}
#[test]
#[ignore = "real elected decoder/tower and Apple10 GPU"]
fn qasr_audio_lifecycle() {
    let mut m = Qwen3Asr::load(Path::new(MODEL), 2048, 4, None).unwrap();
    assert!(m.attach_audio(Path::new(MODEL)).is_err());
    m.attach_audio(Path::new(TOWER)).unwrap();
    assert!(m.attach_audio(Path::new(TOWER)).is_err());
    let chunks = || {
        vec![
            paddock_engine::service::MmChunk::Text(vec![1000]),
            paddock_engine::service::MmChunk::Audio {
                samples: vec![0.; 16000],
                mel: None,
            },
            paddock_engine::service::MmChunk::Text(vec![2000]),
        ]
    };
    let (a, n) = m.forward_prefill_multimodal(0, &chunks()).unwrap();
    assert_eq!(n, 15);
    m.reset();
    let res = m.prefill_begin_multimodal((0..4).map(|i| (i, chunks())).collect());
    assert_eq!(res.len(), 4);
    while m.encoding_pending() {
        for (_, r) in m.encode_step() {
            assert!(!matches!(r, paddock_engine::generator::MmAdmit::Failed(_)));
        }
    }
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 512).unwrap().1);
    }
    assert_eq!(done.len(), 4);
    for (_, b, nb) in done {
        assert_eq!(nb, n);
        let delta = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("Qwen3-ASR packed audio delta {delta}");
        assert!(delta < 0.01);
    }
    m.reset();
    m.prefill_begin_multimodal(vec![(0, chunks())]);
    m.encode_step();
    assert!(m.prefill_abort(0));
    assert!(!m.encoding_pending());
    assert!(m.slots[0].mm.is_none());
}

#[test]
#[ignore = "real elected decoder/tower and Apple10 GPU"]
fn qasr_audio_capacity_and_cancelled_wave_isolation() {
    use paddock_engine::{audio::MelFeatures, generator::MmAdmit, service::MmChunk};
    let mut m = Qwen3Asr::load(Path::new(MODEL), 2048, 4, None).unwrap();
    m.attach_audio(Path::new(TOWER)).unwrap();
    let baseline = m.device.allocated_bytes();
    let chunks = |frames: usize| {
        vec![
            MmChunk::Text(vec![1000]),
            MmChunk::Audio {
                samples: vec![],
                mel: Some(MelFeatures {
                    data: vec![-1.; frames.div_ceil(100) * 100 * 128],
                    n_frames: frames,
                    n_samples: frames * 160,
                    global_max: -8.,
                }),
            },
            MmChunk::Text(vec![2000]),
        ]
    };
    let rejected = m.prefill_begin_multimodal(vec![(0, chunks(12001))]);
    assert!(matches!(rejected[0].1, MmAdmit::Failed(_)));
    m.forward_prefill(1, &[100, 101, 102]).unwrap();
    let expected = m.forward_mixed(&[(1, 103, 3)], 0).unwrap().0;
    m.forward_prefill(1, &[100, 101, 102]).unwrap();
    assert!(matches!(
        m.prefill_begin_multimodal(vec![(0, chunks(12000))])[0].1,
        MmAdmit::Encoding
    ));
    let mut peak = baseline;
    let mut ticks = 0;
    while m.encoding_pending() {
        for (_, a) in m.encode_step() {
            assert!(!matches!(a, MmAdmit::Failed(_)));
        }
        peak = peak.max(m.device.allocated_bytes());
        ticks += 1;
        if ticks == 5 {
            let actual = m.forward_mixed(&[(1, 103, 3)], 0).unwrap().0;
            assert_eq!(actual, expected);
        }
    }
    assert_eq!(m.pending[0].tokens.len(), 1562);
    assert!(ticks > 24);
    eprintln!(
        "Qwen3-ASR 120s wave sampled additional peak {} / {}",
        peak - baseline,
        audio::WORKSPACE
    );
    assert!(peak - baseline < audio::WORKSPACE);
    assert!(m.prefill_abort(0));
    m.reset();
    assert_eq!(m.device.allocated_bytes(), baseline);
    // Cancellation does not renumber another request's packed audio spans.
    let (expected, n) = m.forward_prefill_multimodal(0, &chunks(801)).unwrap();
    m.reset();
    m.prefill_begin_multimodal(vec![(0, chunks(100)), (2, chunks(801))]);
    m.encode_step();
    m.prefill_abort(0);
    while m.encoding_pending() {
        for (_, a) in m.encode_step() {
            assert!(!matches!(a, MmAdmit::Failed(_)));
        }
    }
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 512).unwrap().1);
    }
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].0, 2);
    assert_eq!(done[0].2, n);
    let delta = done[0]
        .1
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("Qwen3-ASR cancelled wave survivor delta {delta}");
    assert!(delta < 0.01);
    m.reset();
    assert_eq!(m.device.allocated_bytes(), baseline);
}
