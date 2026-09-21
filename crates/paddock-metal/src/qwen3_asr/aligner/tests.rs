use super::*;
use std::{cell::Cell, path::Path};
const DIR: &str = concat!(env!("HOME"), "/paddock/models/Qwen3-ForcedAligner-0.6B-hf");
#[test]
fn qalign_rope_preserves_each_bf16_boundary() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let bf = |x| half::bf16::from_f32(x).to_f32();
    let x = vec![1f32; 128]; // RMS rounds to exactly one before the learned weight.
    let w: Vec<_> = (0..128).map(|i| bf((i as f32 - 61.) / 19.)).collect();
    let upload = |v: &[f32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let q = upload(&x);
    let norm = upload(&w);
    let value = upload(&x);
    let meta = d
        .upload(
            &[0u32, 1]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = d.alloc(256).unwrap();
    let values = d.alloc(256).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "qalign_head_rope",
        &[&q, &norm, &meta, &out, &value, &values],
        &[1, 0, 0, 0, 1],
        [1, 1, 1],
        32,
    );
    c.finish().unwrap();
    let raw = unsafe { out.read_u32(64) };
    let actual: Vec<_> = raw
        .iter()
        .flat_map(|v| [(*v & 65535) as u16, (*v >> 16) as u16])
        .collect();
    for i in 0..128 {
        let angle = 1. / 1e6f32.powf((i % 64) as f32 / 64.);
        let other = if i < 64 { i + 64 } else { i - 64 };
        let expected = half::bf16::from_f32(
            bf(w[i] * bf(angle.cos()))
                + bf(w[other]
                    * if i < 64 {
                        -bf(angle.sin())
                    } else {
                        bf(angle.sin())
                    }),
        );
        assert_eq!(actual[i], expected.to_bits(), "rotary component {i}");
    }
    assert_eq!(unsafe { values.read_u32(64) }, vec![0x3f803f80; 64]);
}
fn request(samples: usize, words: usize) -> AlignReq {
    let mel = paddock_engine::audio::mel_features(
        &vec![0.; samples],
        paddock_engine::audio::MelPolicy::Qwen3Aligner,
    )
    .unwrap();
    let n_audio = paddock_engine::audio::audio_token_count(mel.n_frames);
    let mut ids = vec![151669];
    ids.extend(std::iter::repeat_n(super::super::AUDIO, n_audio));
    ids.push(151670);
    let mut ts_rows = Vec::new();
    for _ in 0..words {
        ids.push(9707);
        ts_rows.push(ids.len());
        ids.push(TIMESTAMP);
        ts_rows.push(ids.len());
        ids.push(TIMESTAMP);
    }
    AlignReq {
        ids,
        mel,
        splice_at: 1,
        n_audio,
        ts_rows,
    }
}
#[test]
fn qalign_head_ties_tail_and_nonfinite() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let mut x = vec![-2f32; 4 * LABELS];
    x[4999] = 1.;
    x[LABELS + 32] = 3.;
    x[LABELS + 4000] = 3.;
    x[2 * LABELS + 17] = f32::NAN;
    x[3 * LABELS + 4001] = f32::INFINITY;
    let x = d
        .upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .unwrap();
    let out = d.alloc(4 * 4).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "qalign_argmax",
        &[&x, &out],
        &[LABELS as u32],
        [4, 1, 1],
        256,
    );
    c.finish().unwrap();
    assert_eq!(
        unsafe { out.read_u32(4) },
        vec![4999, 32, u32::MAX, u32::MAX]
    );
}
#[test]
fn qalign_gpu_widen_is_byte_exact() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let bits = [0u16, 0x8000, 1, 0x3f80, 0xbf80, 0x7f7f];
    let x = d
        .upload(
            &bits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = d.alloc(bits.len() * 4).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "qalign_widen",
        &[&x, &out],
        &[bits.len() as u32],
        [1, 1, 1],
        256,
    );
    c.finish().unwrap();
    assert_eq!(
        unsafe { out.read_u32(bits.len()) },
        bits.iter().map(|&x| (x as u32) << 16).collect::<Vec<_>>()
    );
}
#[test]
#[ignore = "real official BF16 checkpoint and Apple10 GPU"]
fn qalign_packed_lifecycle_and_cancellation() {
    let mut m = Qwen3Aligner::load(Path::new(DIR), 2048, None).unwrap();
    let baseline = m.device.allocated_bytes();
    eprintln!(
        "aligner resident weights {} static allocation {}",
        m.weight_bytes, baseline
    );
    let reqs = [
        request(15840, 1),
        request(16000, 3),
        request(16160, 5),
        request(128160, 9),
    ];
    let serial = reqs
        .iter()
        .map(|r| m.run_batch(&[r], &|_| false).unwrap().remove(0).unwrap())
        .collect::<Vec<_>>();
    let refs = reqs.iter().collect::<Vec<_>>();
    let packed = m
        .run_batch(&refs, &|_| false)
        .unwrap()
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    eprintln!("aligner serial bins {serial:?}; packed {packed:?}");
    assert_eq!(serial, packed);
    assert_eq!(m.device.allocated_bytes(), baseline);
    let calls = Cell::new(0);
    let canceled = m
        .run_batch(&refs, &|i| {
            calls.set(calls.get() + 1);
            i == 1 && calls.get() > 8
        })
        .unwrap();
    assert!(canceled[1].is_err());
    for i in [0, 2, 3] {
        assert_eq!(canceled[i].as_ref().unwrap(), &serial[i]);
    }
    assert_eq!(m.device.allocated_bytes(), baseline);
    let calls = Cell::new(0);
    assert!(
        m.run_batch(&refs, &|_| {
            calls.set(calls.get() + 1);
            calls.get() > 10
        })
        .is_err()
    );
    assert_eq!(m.device.allocated_bytes(), baseline);
    assert_eq!(
        m.run_batch(&[&reqs[0]], &|_| false).unwrap()[0]
            .as_ref()
            .unwrap(),
        &serial[0]
    );
    let mut bad = request(16000, 1);
    bad.ts_rows[0] = usize::MAX;
    assert!(m.validate(&bad).is_err());
    bad = request(16000, 1);
    bad.ids[1] = 0;
    assert!(m.validate(&bad).is_err());
    bad = request(16000, 1);
    bad.mel.data[0] = f32::NAN;
    assert!(m.validate(&bad).is_err());
    bad = request(16000, 1);
    bad.n_audio += 1;
    assert!(m.validate(&bad).is_err());
    assert_eq!(m.device.allocated_bytes(), baseline);
}
#[test]
#[ignore = "real official BF16 checkpoint and Apple10 GPU"]
fn qalign_head_chunks_and_maximum_audio() {
    let mut m = Qwen3Aligner::load(Path::new(DIR), 8192, None).unwrap();
    let baseline = m.device.allocated_bytes();
    let req = request(120 * 16000, 70); // 140 selected rows crosses head chunk 128.
    let out = m.run_batch(&[&req], &|_| false).unwrap().remove(0).unwrap();
    assert_eq!(out.len(), 140);
    assert!(out.iter().all(|&i| i < LABELS as u32));
    assert_eq!(m.device.allocated_bytes(), baseline);
    assert_eq!(
        out,
        m.run_batch(&[&req], &|_| false).unwrap().remove(0).unwrap()
    );
    assert!(m.run_batch(&[&req, &req], &|_| false).is_err());
    assert_eq!(m.device.allocated_bytes(), baseline);
}

/// Input-aligned model diagnostic: author processor token IDs, native WAV/mel
/// frontend, same BF16 file. HTTP independently exercises native word packing.
/// This does not quietly replace the ordinary endpoint's inputs for parity.
#[test]
#[ignore = "official checkpoint, author fixtures, captured upstream MPS reference"]
fn qalign_author_fixture_raw_bins_and_packed_reproduction() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/metal");
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("qalign-fixtures-20260911/fixtures.json")).unwrap(),
    )
    .unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("qalign-reference-second-20260911.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(reference["complete"], true);
    let mut requests = Vec::new();
    for (case, refcase) in fixtures["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .zip(reference["cases"].as_array().unwrap())
    {
        assert_eq!(case["id"], refcase["id"]);
        let wav = paddock_engine::audio::decode::decode_audio(
            &std::fs::read(case["wav"].as_str().unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(wav.sample_rate, 16000);
        let mel = paddock_engine::audio::mel_features(
            &wav.samples,
            paddock_engine::audio::MelPolicy::Qwen3Aligner,
        )
        .unwrap();
        assert_eq!(mel.n_frames, refcase["frames"].as_u64().unwrap() as usize);
        let ids = refcase["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let ts_rows = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == TIMESTAMP).then_some(i))
            .collect();
        requests.push(AlignReq {
            n_audio: paddock_engine::audio::audio_token_count(mel.n_frames),
            ids,
            mel,
            ts_rows,
            splice_at: 1,
        });
    }
    let mut model = Qwen3Aligner::load(Path::new(DIR), 8192, None).unwrap();
    if std::env::var_os("PADDOCK_QALIGN_TRACE").is_some() {
        let actual = model.run_batch(&[&requests[8]], &|_| false).unwrap();
        eprintln!("traced bins {:?}", actual);
        return;
    }
    let baseline = model.device.allocated_bytes();
    let mut heads = Vec::new();
    let serial = requests
        .iter()
        .map(|r| {
            let bins = model
                .run_batch(&[r], &|_| false)
                .unwrap()
                .remove(0)
                .unwrap();
            // Test-only diagnostic: capture the unrounded native classifier
            // margins before investigating BF16 activation boundaries. Never
            // change the reference, the serving argmax or the equality gate.
            if r.ts_rows.len() <= HEAD_ROWS {
                let raw = unsafe { model.scratch.logits.read_u32(r.ts_rows.len() * LABELS) };
                heads.push(
                    raw.chunks(LABELS)
                        .map(|row| {
                            let mut ranked = row
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| (i, f32::from_bits(v)))
                                .collect::<Vec<_>>();
                            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                            ranked.truncate(8);
                            ranked
                        })
                        .collect::<Vec<_>>(),
                );
            } else {
                heads.push(Vec::new());
            }
            bins
        })
        .collect::<Vec<_>>();
    let mut packed = Vec::new();
    for chunk in requests.chunks(4) {
        packed.extend(
            model
                .run_batch(&chunk.iter().collect::<Vec<_>>(), &|_| false)
                .unwrap()
                .into_iter()
                .map(|r| r.unwrap()),
        );
    }
    let capture = serde_json::json!({"serial":serial,"packed":packed,"head_top8":heads,
        "ids": requests.iter().map(|r|&r.ids).collect::<Vec<_>>(),
        "reference": "qalign-reference-second-20260911.json"});
    if let Some(path) = std::env::var_os("PADDOCK_QALIGN_CAPTURE") {
        use std::io::Write;
        // Opt-in diagnostics never overwrite an earlier measurement.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&serde_json::to_vec_pretty(&capture).unwrap())
            .unwrap();
    }
    assert_eq!(serial, packed);
    assert_eq!(model.device.allocated_bytes(), baseline);
    let mut matched = 0;
    let mut total = 0;
    for (native, r) in serial.iter().zip(reference["cases"].as_array().unwrap()) {
        let expected = r["bins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let agree = native.iter().zip(&expected).filter(|(a, b)| a == b).count();
        matched += agree;
        total += native.len();
        eprintln!(
            "{}: {agree}/{} raw BF16 timestamp bins agree with upstream MPS",
            r["id"],
            native.len()
        );
    }
    eprintln!(
        "raw timestamp diagnostic {matched}/{total}; independent full agreement is not asserted by this native reproduction gate"
    );
}
