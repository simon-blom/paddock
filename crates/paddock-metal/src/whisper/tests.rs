//! Actual-device gates; independent external reference consumes the same GGUF.
use super::*;
fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn model_path() -> std::path::PathBuf {
    std::env::var_os("PADDOCK_WHISPER_MODEL")
        .map(Into::into)
        .expect("set PADDOCK_WHISPER_MODEL to an elected Nordic F16 GGUF")
}
fn mel(path: &Path) -> MelFeatures {
    let a = paddock_engine::audio::wav::decode_wav(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(a.sample_rate, 16000);
    paddock_engine::audio::whisper_features(&a.samples).unwrap()
}
fn f32_file(path: &Path, x: &[f32]) {
    std::fs::write(
        path,
        x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
}
#[test]
#[ignore = "requires Metal device"]
fn whisper_alignment_probs_match_scalar_softmax() {
    let d = MetalDevice::new(None).unwrap();
    for frames in [1usize, 7, 51, 203, 1500] {
        let rows = 5usize;
        let q: Vec<f32> = (0..rows * D).map(|i| (i % 37) as f32 / 37. - 0.5).collect();
        let mut k = vec![half::f16::NAN; 2 * T * D];
        for t in 0..frames {
            for j in 0..D {
                k[(T + t) * D + j] = half::f16::from_f32(((t * 7 + j) % 53) as f32 / 53. - 0.5);
            }
        }
        let qb = upload(&d, &q.iter().map(|v| v.to_bits()).collect::<Vec<_>>()).unwrap();
        let kb = d
            .upload(&k.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        let ids = upload(&d, &[3, 17]).unwrap();
        let out = d.alloc(2 * 3 * frames * 4).unwrap();
        let c = d.begin().unwrap();
        c.dispatch(
            "wh_align_probs",
            &[&qb, &kb, &ids, &out],
            &[1, frames as u32, 3, 1],
            [2, rows, 1],
            256,
        );
        c.finish().unwrap();
        let got = unsafe { out.read_f32(0, 2 * 3 * frames) };
        for (hi, head) in [3usize, 17].iter().enumerate() {
            for r in 2..rows {
                let logits: Vec<f64> = (0..frames)
                    .map(|t| {
                        (0..64)
                            .map(|j| {
                                q[r * D + head * 64 + j] as f64
                                    * k[(T + t) * D + head * 64 + j].to_f32() as f64
                            })
                            .sum::<f64>()
                            / 8.
                    })
                    .collect();
                let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denominator: f64 = logits.iter().map(|v| (v - maximum).exp()).sum();
                let row = &got[(hi * 3 + r - 2) * frames..(hi * 3 + r - 1) * frames];
                assert!((row.iter().sum::<f32>() - 1.).abs() < 2e-5);
                for (actual, logit) in row.iter().zip(logits) {
                    assert!((*actual as f64 - (logit - maximum).exp() / denominator).abs() < 2e-6);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires elected Whisper GGUF and probe JSON"]
fn whisper_alignment_chunk_parity_and_slot_isolation() {
    let probes: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("PADDOCK_WHISPER_PROBES").expect("probe JSON")).unwrap(),
    )
    .unwrap();
    let mut m = Whisper::load(&model_path(), 448, None).unwrap();
    m.prepare(2).unwrap();
    let map = MappedGguf::open(&model_path()).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let case = &probes[0];
    let a = mel(Path::new(case["wav"].as_str().unwrap()));
    let lang = m.lang_token(case["lang"].as_str().unwrap()).unwrap();
    let tokens = tokenizer
        .encode(" Testar funktionen med ljudinspelning. En längre mening för flera token.")
        .unwrap();
    m.encode_wave(&[0, 1], &[&a, &a]).unwrap();
    m.step(&[1], &[50258], &[0], None).unwrap();
    let baseline = m.device.allocated_bytes();
    assert!(m.align_tokens(2, lang, &tokens, a.n_samples).is_err());
    assert!(m.align_tokens(0, 1, &tokens, a.n_samples).is_err());
    assert!(m.align_tokens(0, lang, &[50257], a.n_samples).is_err());
    assert!(m.align_tokens(0, lang, &vec![1; 445], a.n_samples).is_err());
    assert!(m.align_tokens(0, lang, &tokens, 480001).is_err());
    assert!(m.align_tokens(0, lang, &tokens, 0).is_err());
    assert!(m.align_tokens(0, lang, &[], 0).unwrap().is_empty());
    let reference = m
        .capture_alignment(0, lang, &tokens, a.n_samples, 1)
        .unwrap()
        .0;
    for chunk in [8, 32, 64] {
        let start = std::time::Instant::now();
        let result = m
            .capture_alignment(0, lang, &tokens, a.n_samples, chunk)
            .unwrap()
            .0;
        let maximum = reference
            .iter()
            .zip(result)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "alignment chunk={chunk} capture_ms={} max_probability_delta={maximum}",
            start.elapsed().as_secs_f64() * 1000.
        );
        assert!(maximum < 0.002, "chunk {chunk}: {maximum}");
        assert_eq!(m.device.allocated_bytes(), baseline);
        assert_eq!(m.lengths[1], Some(1));
    }
    let boundaries = m.align_tokens(0, lang, &tokens, a.n_samples).unwrap();
    assert_eq!(boundaries.len(), tokens.len() + 1);
    assert!(boundaries.windows(2).all(|p| p[0] <= p[1]));
    assert!(
        boundaries
            .iter()
            .all(|b| *b >= 0. && *b <= a.n_samples as f32 / 16000.)
    );
    assert_eq!(m.device.allocated_bytes(), baseline);
    m.step(&[1], &[lang], &[1], None).unwrap();
    let after = m.logits_row(0).unwrap();
    m.encode_wave(&[1], &[&a]).unwrap();
    m.step(&[1], &[50258], &[0], None).unwrap();
    m.step(&[1], &[lang], &[1], None).unwrap();
    let clean = m.logits_row(0).unwrap();
    assert_eq!(after, clean, "alignment must not alter another slot");
}

#[test]
#[ignore = "requires elected GGUF, probes, retained transcripts and fresh capture directory"]
fn whisper_capture_word_alignments() {
    let probes: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("PADDOCK_WHISPER_PROBES").unwrap()).unwrap(),
    )
    .unwrap();
    let transcripts: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("PADDOCK_WHISPER_TRANSCRIPTS").unwrap()).unwrap(),
    )
    .unwrap();
    let out = std::path::PathBuf::from(std::env::var_os("PADDOCK_WHISPER_CAPTURE").unwrap());
    std::fs::create_dir(&out).unwrap();
    let mut m = Whisper::load(&model_path(), 448, None).unwrap();
    m.prepare(1).unwrap();
    let baseline = m.device.allocated_bytes();
    let mut results = Vec::new();
    for (i, case) in probes.as_array().unwrap().iter().enumerate() {
        let row = transcripts["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == case["id"])
            .unwrap();
        let tokens: Vec<_> = row["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .filter(|t| *t < 50257)
            .collect();
        if tokens.is_empty() {
            continue;
        }
        let a = mel(Path::new(case["wav"].as_str().unwrap()));
        let lang = m.lang_token(case["lang"].as_str().unwrap()).unwrap();
        m.encode_wave(&[0], &[&a]).unwrap();
        let (weights, heads, rows, frames) = m
            .capture_alignment(0, lang, &tokens, a.n_samples, 32)
            .unwrap();
        let file = format!("attention-{i}.f32");
        f32_file(&out.join(&file), &weights);
        let start = std::time::Instant::now();
        let boundaries = m.align_tokens(0, lang, &tokens, a.n_samples).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.;
        assert_eq!(m.device.allocated_bytes(), baseline);
        results.push(serde_json::json!({"id":case["id"],"wav":case["wav"],"lang":case["lang"],"tokens":tokens,
            "heads":heads,"rows":rows,"frames":frames,"attention":file,"boundaries":boundaries,"alignment_ms":ms}));
        eprintln!(
            "alignment {}: tokens={} frames={frames} ms={ms}",
            case["id"],
            tokens.len()
        );
    }
    std::fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .unwrap();
}
#[test]
#[ignore = "requires M5"]
fn whisper_projection_tiles_match_independent_gpu_matrix() {
    let d = MetalDevice::new(None).unwrap();
    let (k, n) = (1280usize, 65usize);
    let w = d
        .upload(
            &(0..k * n)
                .flat_map(|i| half::f16::from_f32((i % 19) as f32 / 32. - 0.25).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let x = upload(
        &d,
        &(0..19 * k)
            .map(|i| ((i % 31) as f32 / 16. - 0.75).to_bits())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let bias = upload(&d, &vec![0.125f32.to_bits(); n]).unwrap();
    for rows in [1usize, 2, 3, 4, 5, 8, 9, 15, 16] {
        for mode in [0u32, 1, 2, 3] {
            let a = upload(&d, &vec![0.5f32.to_bits(); rows * n]).unwrap();
            let b = upload(&d, &vec![0.5f32.to_bits(); rows * n]).unwrap();
            let c = d.begin().unwrap();
            let name = match rows {
                1 => "wh_mv1",
                2 => "wh_mv2",
                3..=4 => "wh_mv4",
                5..=8 => "wh_mv8",
                _ => "wh_mv16",
            };
            let p = [k as u32, n as u32, rows as u32, mode];
            c.dispatch(name, &[&w, &x, &a, &bias], &p, [n.div_ceil(4), 1, 1], 128);
            c.dispatch(
                "wh_mm",
                &[&w, &x, &b, &bias],
                &p,
                [n.div_ceil(64), rows.div_ceil(32), 1],
                128,
            );
            c.finish().unwrap();
            // SAFETY: both independent GPU contractions are complete.
            let (a, b) = unsafe { (a.read_f32(0, rows * n), b.read_f32(0, rows * n)) };
            let delta = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(delta < 1e-4, "rows={rows} mode={mode} delta={delta}");
        }
    }
}

#[test]
#[ignore = "requires M5 and elected Whisper GGUF"]
fn whisper_load_wave_and_decode_slots() {
    let mut m = Whisper::load(&model_path(), 448, None).unwrap();
    m.prepare(4).unwrap();
    let baseline = m.device.allocated_bytes();
    eprintln!(
        "whisper weights={} prepared={} scratch={} kv={}",
        m.weights_bytes,
        baseline,
        m.scratch
            .as_ref()
            .map(|_| Scratch::sizes(4).iter().sum::<usize>())
            .unwrap_or(0),
        L * 4 * (T + 448) * D * 4
    );
    let a = mel(&root().join("target/metal/qasr-audio-20260911/librispeech_mr_quilter.wav"));
    let mut b = a.clone();
    b.data[0] = f32::NAN;
    assert!(m.encode_wave(&[0, 0], &[&a, &a]).is_err());
    assert!(m.encode_wave(&[4], &[&a]).is_err());
    assert!(m.encode_wave(&[0], &[&b]).is_err());
    assert!(m.step(&[0], &[50258], &[0], None).is_err());
    let mut serial = Vec::new();
    for token in [50258, 50259, 50273, 50269] {
        m.encode_wave(&[0], &[&a]).unwrap();
        let out = m.step(&[0], &[token], &[0], None).unwrap();
        assert!(out.logprob[0].is_finite());
        serial.push(m.logits_row(0).unwrap());
    }
    m.encode_wave(&[3, 1, 0, 2], &[&a, &a, &a, &a]).unwrap();
    m.step(&[3, 1, 0, 2], &[50258, 50259, 50273, 50269], &[0; 4], None)
        .unwrap();
    for (i, want) in serial.iter().enumerate() {
        let got = m.logits_row(i).unwrap();
        let err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("packed/serial row={i} max_logit_delta={err}");
        assert!(err < 0.03);
    }
    let before = m.lengths.clone();
    assert!(m.step(&[0, 0], &[50258; 2], &[1; 2], None).is_err());
    assert!(m.step(&[0], &[V as u32], &[1], None).is_err());
    assert!(m.step(&[0], &[50258], &[0], None).is_err());
    assert_eq!(before, m.lengths);
    m.step(&[1, 3], &[50360; 2], &[1; 2], None).unwrap();
    m.encode_wave(&[0], &[&a]).unwrap();
    m.step(&[3, 0], &[50364, 50258], &[2, 0], None).unwrap();
    assert_eq!(m.device.allocated_bytes(), baseline);
    assert!(m.supports_word_times());
    assert!(m.prepare(17).is_err());
}
#[test]
#[ignore = "requires elected weights and explicit probe/capture paths"]
fn whisper_capture_generations() {
    let probes =
        std::path::PathBuf::from(std::env::var_os("PADDOCK_WHISPER_PROBES").expect("probe JSON"));
    let out = std::path::PathBuf::from(
        std::env::var_os("PADDOCK_WHISPER_CAPTURE").expect("fresh capture directory"),
    );
    std::fs::create_dir(&out).unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&probes).unwrap()).unwrap();
    let mut m = Whisper::load(&model_path(), 448, None).unwrap();
    m.prepare(4).unwrap();
    let map = MappedGguf::open(&model_path()).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let mut results = Vec::new();
    for (ci, case) in cases.as_array().unwrap().iter().enumerate() {
        let a = mel(Path::new(case["wav"].as_str().unwrap()));
        let start = std::time::Instant::now();
        m.encode_wave(&[0], &[&a]).unwrap();
        let enc_ms = start.elapsed().as_secs_f64() * 1000.;
        if ci == 0 {
            f32_file(&out.join("mel.f32"), &a.data);
            // SAFETY: encode_wave synchronized. x is the final encoder state.
            f32_file(&out.join("encoder.f32"), &unsafe {
                m.scratch.as_ref().unwrap().x.read_f32(0, T * D)
            });
        }
        let lang = m.lang_token(case["lang"].as_str().unwrap()).unwrap();
        let timestamps = case["timestamps"].as_bool().unwrap_or(false);
        let mut prompt = vec![50258, lang, 50360];
        if !timestamps {
            prompt.push(50364);
        }
        let mut tokens = Vec::new();
        let mut trace = Vec::new();
        let mut feed = prompt[0];
        let mut finish = "length";
        let cap = case["max_tokens"].as_u64().unwrap_or(256) as usize;
        let mut first_ms = 0.;
        for pos in 0..448 {
            let generating = pos + 1 >= prompt.len();
            let rules = paddock_engine::whisper::ts_state(
                &tokens,
                &m.time_scale(),
                timestamps && generating,
            );
            let r = m
                .step(
                    &[0],
                    &[feed],
                    &[pos as u32],
                    timestamps.then_some(rules.as_slice()),
                )
                .unwrap();
            if pos == 0 {
                f32_file(
                    &out.join(format!("sot-{ci}.f32")),
                    &m.logits_row(0).unwrap(),
                );
            }
            if !generating {
                feed = prompt[pos + 1];
                continue;
            }
            if trace.is_empty() {
                first_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            trace.push(serde_json::json!({"id":r.next[0],"logprob":r.logprob[0],"runner_up":r.runner_up[0]}));
            feed = r.next[0];
            if feed == 50257 {
                finish = "stop";
                break;
            }
            tokens.push(feed);
            if tokens.len() >= cap {
                break;
            }
        }
        let row = serde_json::json!({"id":case["id"],"tokens":tokens,"text":tokenizer.decode(&tokens,true).unwrap(),
            "finish_reason":finish,"trace":trace,"encoder_ms":enc_ms,"ttft_ms":first_ms,"wall_ms":start.elapsed().as_secs_f64()*1000.});
        eprintln!("{row}");
        results.push(row);
        std::fs::write(
            out.join("results.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"complete":false,"cases":results}))
                .unwrap(),
        )
        .unwrap();
    }
    std::fs::write(out.join("results.json"),serde_json::to_vec_pretty(&serde_json::json!({"complete":true,
        "model_sha256":sha(&model_path()),"probes_sha256":sha(&probes),
        "test_binary_sha256":sha(&std::env::current_exe().unwrap()),
        "shader_sha256":sha(&root().join("packs/metal/whisper.metal")),
        "weights_bytes":m.weights_bytes,"prepared_bytes":m.device.allocated_bytes(),"cases":results})).unwrap()).unwrap();
}

fn sha(path: &Path) -> String {
    let out = std::process::Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned()
}

#[test]
#[ignore = "requires M5 and elected Whisper GGUF; 16-slot capacity and 448-position cache"]
fn whisper_capacity_and_cache_tails() {
    let mut m = Whisper::load(&model_path(), 448, None).unwrap();
    assert!(Whisper::load(&model_path(), 7, None).is_err());
    assert!(m.prepare(0).is_err());
    m.prepare(16).unwrap();
    let baseline = m.device.allocated_bytes();
    let a = mel(&root().join("target/metal/qasr-audio-20260911/librispeech_mr_quilter.wav"));
    for first in (0..16).step_by(4) {
        m.encode_wave(&(first..first + 4).collect::<Vec<_>>(), &[&a; 4])
            .unwrap();
    }
    let ids = (0..16).collect::<Vec<u32>>();
    let r = m.step(&ids, &[50258; 16], &[0; 16], None).unwrap();
    assert!(r.next.iter().all(|&id| id == r.next[0]));
    // Exercise both sides of the split-K boundary, the final trained
    // position, then reuse a different slot without moving any old KV.
    for pos in 1..448 {
        m.step(&[15], &[50364], &[pos], None).unwrap();
    }
    assert!(m.step(&[15], &[50364], &[448], None).is_err());
    m.encode_wave(&[0], &[&a]).unwrap();
    m.step(&[14, 0], &[50360, 50258], &[1, 0], None).unwrap();
    assert_eq!(baseline, m.device.allocated_bytes());
    assert!(m.weights_bytes + (L * 16 * (T + 448) * D * 4) as u64 + 576716800 >= baseline);
    eprintln!("whisper capacity=16 ctx=448 prepared={baseline}; tail and reuse pass");
}
