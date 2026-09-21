use super::*;
// Where the elected checkpoints live on the machine that builds these tests.
const MODELS: &str = concat!(env!("HOME"), "/paddock/models");

#[test]
#[ignore = "requires official base/Plus tokenizer captures and elected GGUFs"]
fn gs_tokenizers_match_the_two_author_contracts() {
    use paddock_tokenizer::GgufTokenizer;
    for plus in [false, true] {
        let suffix = if plus { "-plus" } else { "" };
        let stem = if plus { "plus" } else { "base" };
        let path = format!(
            "{MODELS}/granite-speech-4.1-2b{suffix}-GGUF/granite-speech-4.1-2b{suffix}-Q8_0.gguf"
        );
        let map = MappedGguf::open(Path::new(&path)).unwrap();
        let native = GgufTokenizer::from_gguf(map.gguf()).unwrap();
        let author = GgufTokenizer::from_hf_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../target/metal/gs-{stem}-tokenizer-20260911")),
        )
        .unwrap();
        for text in [
            "USER: <|audio|>transcribe the speech with proper punctuation and capitalization.\n ASSISTANT:",
            "<|start_of_role|>user<|end_of_role|><|audio|>translate the speech to French.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
            "Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]",
            "Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.",
            "-hello -- you've 1234567\n ASSISTANT:",
            " français 日本語 Portuguese ä ö \r\n ",
        ] {
            assert_eq!(
                native.encode(text).unwrap(),
                author.encode(text).unwrap(),
                "plus={plus} {text:?}"
            );
        }
        eprintln!("gs plus={plus}: six tokenizer strings match author IDs exactly");
    }
}

#[test]
#[ignore = "requires M5 Metal"]
fn gs_shaw_matches_independent_gpu() {
    let d = MetalDevice::new(None).unwrap();
    for lengths in [[1, 3], [199, 201], [200, 231]] {
        let rows: usize = lengths.iter().sum();
        let data = |n: usize, salt: usize| {
            d.upload(
                &(0..n)
                    .map(|i| ((i * 13 + salt * 17) % 199) as f32 / 199. - 0.5)
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let q = data(rows * E, 1);
        let k = data(rows * E, 2);
        let v = data(rows * E, 3);
        let rel = Weight {
            buffer: data(401 * 128, 4),
            ty: 0,
            k: 128,
            n: 401,
        };
        let qr = d.alloc(rows * 8 * 401 * 4).unwrap();
        let a = d.alloc(rows * E * 4).unwrap();
        let b = d.alloc(a.len()).unwrap();
        let mut tiles = Vec::new();
        let mut bounds = Vec::new();
        let mut offset = 0;
        for len in lengths {
            for start in (0..len).step_by(200) {
                let n = (len - start).min(200);
                for _ in 0..n {
                    bounds.extend([(offset + start) as u32, (offset + start + n) as u32]);
                }
                for t in (0..n).step_by(32) {
                    tiles.extend([
                        (offset + start + t) as u32,
                        (n - t).min(32) as u32,
                        (offset + start) as u32,
                        n as u32,
                    ]);
                }
            }
            offset += len;
        }
        let nt = tiles.len() / 4;
        let tiles = upload(&d, &tiles).unwrap();
        let bounds = upload(&d, &bounds).unwrap();
        let c = d.begin().unwrap();
        Tower::plain(&c, &rel, &a, &q, &qr, rows * 8, 0);
        c.dispatch(
            "gs_attention",
            &[&q, &k, &v, &a, &tiles, &qr],
            &[0],
            [8, nt, 1],
            64,
        );
        c.dispatch(
            "gs_attention_check",
            &[&q, &k, &v, &rel.buffer, &bounds, &b],
            &[0],
            [8, rows, 1],
            32,
        );
        c.finish().unwrap();
        let a = unsafe { a.read_f32(0, rows * E) };
        let b = unsafe { b.read_f32(0, rows * E) };
        assert!(a.iter().chain(&b).all(|v| v.is_finite()));
        let max = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("gs Shaw {lengths:?}: max_abs={max}");
        assert!(max < 0.0001);
    }
}

#[test]
#[ignore = "requires both elected Granite Speech R2 pairs and M5"]
fn gs_real_load_and_ragged_tower() {
    for plus in [false, true] {
        let root = format!(
            "{MODELS}/granite-speech-4.1-2b{}-GGUF",
            if plus { "-plus" } else { "" }
        );
        let d = MetalDevice::new(None).unwrap();
        let tower = Tower::load(&d, &Path::new(&root).join("mmproj-model-f16.gguf"), plus).unwrap();
        let baseline = d.allocated_bytes();
        assert_eq!(baseline, if plus { 1162589552 } else { 1154200944 });
        eprintln!("gs plus={plus} tower_resident={baseline}");
        let inputs = [4800, 63840, 64320].map(|n| {
            mel::speech_features(
                &(0..n)
                    .map(|i| (i as f32 * 0.031).sin() * 0.05)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        });
        let finish = |mut j: Job| {
            let mut phases = 0;
            loop {
                phases += 1;
                if let Some(v) = tower.step(&d, &mut j).unwrap() {
                    assert_eq!(phases, 52); // initialization is separate
                    break v;
                }
            }
        };
        let serial = inputs
            .iter()
            .map(|m| finish(tower.start(&d, std::slice::from_ref(m)).unwrap()).remove(0))
            .collect::<Vec<_>>();
        let packed = finish(tower.start(&d, &inputs).unwrap());
        for (a, b) in serial.iter().zip(&packed) {
            let a = unsafe { a.read_f32(0, a.len() / 4) };
            let b = unsafe { b.read_f32(0, b.len() / 4) };
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let max = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("gs plus={plus} packed max_abs={max}");
            assert!(max < 0.001);
        }
        drop(serial);
        drop(packed);
        assert_eq!(baseline, d.allocated_bytes());
        let job = tower.start(&d, &inputs).unwrap();
        drop(job);
        assert_eq!(baseline, d.allocated_bytes());
        let mut bad = inputs[0].clone();
        bad.data[0] = f32::NAN;
        assert!(tower.start(&d, &[bad]).is_err());
        assert_eq!(baseline, d.allocated_bytes());
        assert!(Tower::load(&d, &Path::new(&root).join("mmproj-model-f16.gguf"), !plus).is_err());
        let target = Path::new(&root).join(format!(
            "granite-speech-4.1-2b{}-Q8_0.gguf",
            if plus { "-plus" } else { "" }
        ));
        let mut model = Granite::load(&target, 2048, 4, None).unwrap();
        model
            .attach_audio(&Path::new(&root).join("mmproj-model-f16.gguf"))
            .unwrap();
        assert_eq!(model.vocab(), 100353);
        assert!(model.supports_chunked_multimodal());
        let scratch = model.device.allocated_bytes() - model.weight_bytes - model.kv_bytes;
        eprintln!(
            "gs plus={plus} decoder/tower_weights={} kv={} scratch={scratch}",
            model.weight_bytes, model.kv_bytes
        );
    }
}

#[test]
#[ignore = "requires elected Granite Speech pairs and M5"]
fn gs_audio_lifecycle_capacity_and_cancellation() {
    use paddock_engine::{generator::MmAdmit, service::MmChunk};
    for plus in [false, true] {
        let suffix = if plus { "-plus" } else { "" };
        let root = format!("{MODELS}/granite-speech-4.1-2b{suffix}-GGUF");
        let path = Path::new(&root).join(format!("granite-speech-4.1-2b{suffix}-Q8_0.gguf"));
        let mut m = Granite::load(&path, 2048, 4, None).unwrap();
        assert!(m.attach_audio(&path).is_err());
        m.attach_audio(&Path::new(&root).join("mmproj-model-f16.gguf"))
            .unwrap();
        assert!(
            m.attach_audio(&Path::new(&root).join("mmproj-model-f16.gguf"))
                .is_err()
        );
        let baseline = m.device.allocated_bytes();
        let chunks = |frames: usize| {
            vec![
                MmChunk::Text(vec![1000]),
                MmChunk::Audio {
                    samples: vec![0.; frames * 320],
                    mel: Some(MelFeatures {
                        data: vec![-1.; frames * 160],
                        n_frames: frames,
                        n_samples: frames * 320,
                        global_max: -8.,
                    }),
                },
                MmChunk::Text(vec![2000]),
            ]
        };
        let (a, n) = m.forward_prefill_multimodal(0, &chunks(201)).unwrap();
        assert_eq!(n, 44);
        m.reset();
        assert_eq!(m.device.allocated_bytes(), baseline);
        let drain = |m: &mut Granite| {
            while m.encoding_pending() {
                for (_, a) in m.encode_step() {
                    assert!(!matches!(a, MmAdmit::Failed(_)));
                }
            }
            let mut done = Vec::new();
            while !m.pending.is_empty() {
                done.extend(m.forward_mixed(&[], 512).unwrap().1);
            }
            done
        };
        let admitted = m.prefill_begin_multimodal((0..4).map(|i| (i, chunks(201))).collect());
        assert!(admitted.iter().all(|(_, a)| matches!(a, MmAdmit::Encoding)));
        assert!(m.forward_prefill(0, &[1]).is_err());
        assert!(m.prefill_begin(0, vec![1]).is_err());
        let done = drain(&mut m);
        assert_eq!(done.len(), 4);
        for (_, b, nb) in done {
            assert_eq!(nb, n);
            let delta = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("gs plus={plus} packed decoder max_abs={delta}");
            assert!(delta < 0.01);
        }
        m.release_inactive_slots(&[false; 4]);
        assert!(m.slots.iter().all(|s| s.audio.is_empty()));
        assert_eq!(m.device.allocated_bytes(), baseline);
        assert_eq!(
            m.radix.evictable_blocks(&m.pool),
            0,
            "audio placeholder KV entered text radix"
        );
        m.prefill_begin_multimodal(vec![(0, chunks(15)), (2, chunks(201))]);
        m.encode_step();
        m.prefill_abort(0);
        let done = drain(&mut m);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].0, 2);
        assert_eq!(done[0].2, n);
        let delta = a
            .iter()
            .zip(&done[0].1)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(delta < 0.01);
        m.reset();
        assert_eq!(m.device.allocated_bytes(), baseline);
        assert!(matches!(
            m.prefill_begin_multimodal(vec![(0, chunks(6001))])[0].1,
            MmAdmit::Failed(_)
        ));
        let mut bad = chunks(15);
        if let MmChunk::Audio { samples, .. } = &mut bad[1] {
            samples.pop();
        }
        assert!(matches!(
            m.prefill_begin_multimodal(vec![(0, bad)])[0].1,
            MmAdmit::Failed(_)
        ));
        // The largest admitted wave still permits language work between tower
        // stages. The center convolution must not be truncated into chunks.
        m.forward_prefill(1, &[100, 101, 102]).unwrap();
        let expected = m.forward_mixed(&[(1, 103, 3)], 0).unwrap().0;
        m.forward_prefill(1, &[100, 101, 102]).unwrap();
        assert!(matches!(
            m.prefill_begin_multimodal(vec![(0, chunks(6000))])[0].1,
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
                assert_eq!(m.forward_mixed(&[(1, 103, 3)], 0).unwrap().0, expected);
            }
        }
        assert_eq!(m.pending[0].tokens.len(), 1202);
        // The time-budgeted scheduler groups cheap phases but still yields
        // on a large wave. The graph's exact 52 phases are checked above.
        assert!((6..=53).contains(&ticks));
        eprintln!(
            "gs plus={plus} max120s wave ticks={ticks} additional bytes={}",
            peak - baseline
        );
        assert!(peak - baseline < WORKSPACE);
        m.prefill_abort(0);
        m.reset();
        assert_eq!(m.device.allocated_bytes(), baseline);
        m.prefill_begin_multimodal(vec![(0, chunks(6000))]);
        m.encode_step();
        m.prefill_abort(0);
        assert!(!m.encoding_pending());
        assert_eq!(m.device.allocated_bytes(), baseline);
        assert!(m.forward_prefill(0, &[100352]).is_err());
    }
}
