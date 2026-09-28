use super::*;
use objc2_metal::MTLBuffer;
use serde_json::Value;

#[test]
fn packed_window_attention_matches_scalar_softmax() {
    let device = MetalDevice::new(Some(32 << 20)).unwrap();
    let sizes = [17usize, 65, 129];
    let rows: usize = sizes.iter().sum();
    let mut tiles = vec![];
    let mut base = 0;
    for n in sizes {
        for row in (0..n).step_by(32) {
            tiles.extend([
                (base + row) as u32,
                (n - row).min(32) as u32,
                base as u32,
                n as u32,
            ]);
        }
        base += n;
    }
    let tile_buffer = device
        .upload(
            &tiles
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for heads in [12, 16] {
        let d = heads * 64;
        let plane = |seed| {
            (0..rows * d)
                .map(|i| ((i * seed % 31) as f32 - 15.) / 64.)
                .collect::<Vec<_>>()
        };
        let (q, k, v) = (plane(3), plane(7), plane(11));
        let upload = |data: &[f32]| {
            device
                .upload(
                    &data
                        .iter()
                        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        let (qb, kb, vb) = (upload(&q), upload(&k), upload(&v));
        for window in [0usize, 1, 64] {
            let out = device.upload(&vec![0xff; (rows + 32) * d * 2]).unwrap();
            let c = device.begin().unwrap();
            c.dispatch(
                if heads == 12 {
                    "laya_attention12"
                } else {
                    "laya_attention16"
                },
                &[&qb, &kb, &vb, &out, &tile_buffer],
                &[1, window as u32],
                [heads, tiles.len() / 4, 1],
                64,
            );
            c.finish().unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    out.raw.contents().as_ptr().cast::<half::f16>(),
                    (rows + 32) * d,
                )
            };
            assert!(actual[..rows * d].iter().all(|v| v.is_finite()));
            assert!(
                actual[rows * d..].iter().all(|v| v.to_bits() == 0xffff),
                "output overrun"
            );
            let mut base = 0;
            for n in sizes {
                for row in 0..n {
                    let begin = if window == 0 {
                        0
                    } else {
                        row.saturating_sub(window)
                    };
                    let end = if window == 0 {
                        n
                    } else {
                        (row + window + 1).min(n)
                    };
                    for head in [0, heads - 1] {
                        let qi = (base + row) * d + head * 64;
                        let scores = (begin..end)
                            .map(|key| {
                                let ki = (base + key) * d + head * 64;
                                (0..64).map(|j| q[qi + j] * k[ki + j]).sum::<f32>() * 0.125
                            })
                            .collect::<Vec<_>>();
                        let p = probs(&scores);
                        for col in 0..64 {
                            let expected = (begin..end)
                                .zip(&p)
                                .map(|(key, prob)| prob * v[(base + key) * d + head * 64 + col])
                                .sum::<f32>();
                            assert!(
                                (actual[qi + col].to_f32() - expected).abs() < 0.00015,
                                "heads={heads} window={window} sequence={base} row={row} col={col}"
                            );
                        }
                    }
                }
                base += n;
            }
        }
    }
}

fn values(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn probs(v: &[f32]) -> Vec<f32> {
    let high = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p = v.iter().map(|x| (x - high).exp()).collect::<Vec<_>>();
    let den = p.iter().sum::<f32>();
    for x in &mut p {
        *x /= den;
    }
    p
}
fn top(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// Explicit opt-in: an absent checkpoint/reference is a failure, never a
/// passing skip. Run with LAYA_DIR and `--ignored --nocapture`.
#[test]
#[ignore = "requires the original checkpoints and the reference run's golden/reference.json"]
fn laya_external_golden_and_packed_invariance() {
    let dir = std::env::var_os("LAYA_DIR").expect("LAYA_DIR required");
    let dir = Path::new(&dir);
    let reference: Value =
        serde_json::from_slice(&std::fs::read(dir.join("golden/reference.json")).unwrap()).unwrap();
    let mut model = Laya::load(dir, None).unwrap();
    let info = model.info();
    eprintln!(
        "Laya resident weights={} workspace={}",
        info.weight_bytes, info.workspace_bytes
    );
    let mut all = vec![];
    let mut worst = [0.0f32; 3];
    for (ri, r) in reference["requests"].as_array().unwrap().iter().enumerate() {
        let checkpoint = Checkpoint::parse(r["route"]["model"].as_str().unwrap()).unwrap();
        let inputs = r["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|it| {
                (
                    it["ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect::<Vec<_>>(),
                    it["markers"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect::<Vec<_>>(),
                    it["qtype"].as_u64().unwrap() as u32,
                )
            })
            .collect::<Vec<_>>();
        let seqs = inputs
            .iter()
            .map(|(ids, markers, qtype)| LayaSeq {
                ids,
                markers,
                qtype: *qtype,
            })
            .collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let out = model.run(checkpoint, &seqs).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut diffs = [0.0f32; 3];
        for (i, _) in seqs.iter().enumerate() {
            let logits = &out.logits[out.offsets[i]..out.offsets[i + 1]];
            let expected = values(&r["precisions"]["fp32"]["logits"][i]);
            assert_eq!(
                top(logits),
                top(&expected),
                "request {ri} question {i} answer"
            );
            for (a, b) in logits.iter().zip(&expected) {
                diffs[0] = diffs[0].max((a - b).abs());
            }
            for (a, b) in probs(logits).iter().zip(probs(&expected)) {
                diffs[1] = diffs[1].max((a - b).abs());
            }
            for (a, b) in out.act[i * out.n_act..(i + 1) * out.n_act]
                .iter()
                .zip(values(&r["precisions"]["fp32"]["act"][i]))
            {
                diffs[2] = diffs[2].max((a - b).abs());
            }
        }
        eprintln!(
            "request {ri}: {} rows={} logit={:.6} probability={:.6} act={:.6} wall={ms:.3}ms",
            checkpoint.name(),
            out.rows,
            diffs[0],
            diffs[1],
            diffs[2]
        );
        for j in 0..3 {
            worst[j] = worst[j].max(diffs[j]);
        }
        all.push((checkpoint, inputs, out));
    }
    assert!(worst[0] < 0.05, "logit error {}", worst[0]);
    assert!(worst[1] < 0.005, "probability error {}", worst[1]);
    assert!(worst[2] < 0.01, "act error {}", worst[2]);
    for ck in Checkpoint::ALL {
        let selected = all
            .iter()
            .rev()
            .filter(|(c, _, _)| *c == ck)
            .collect::<Vec<_>>();
        let seqs = selected
            .iter()
            .flat_map(|(_, items, _)| {
                items.iter().map(|(ids, markers, qtype)| LayaSeq {
                    ids,
                    markers,
                    qtype: *qtype,
                })
            })
            .collect::<Vec<_>>();
        let packed = model.run(ck, &seqs).unwrap();
        let mut at = 0;
        for (_, items, alone) in selected {
            for i in 0..items.len() {
                assert_eq!(
                    &packed.logits[packed.offsets[at]..packed.offsets[at + 1]],
                    &alone.logits[alone.offsets[i]..alone.offsets[i + 1]],
                    "{} packed logit {i}",
                    ck.name()
                );
                assert_eq!(
                    &packed.act[at * packed.n_act..(at + 1) * packed.n_act],
                    &alone.act[i * alone.n_act..(i + 1) * alone.n_act],
                    "{} packed act {i}",
                    ck.name()
                );
                at += 1;
            }
        }
        eprintln!(
            "{} bit-exact packed: {} sequences / {} rows",
            ck.name(),
            seqs.len(),
            packed.rows
        );
        let max_len = model.info().config(ck).unwrap().max_len;
        for len in [
            1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 1023, 1024,
        ] {
            if len > max_len {
                continue;
            }
            let ids = (0..len).map(|i| 7 + (i % 5) as u32).collect::<Vec<_>>();
            let markers = [0, (len - 1) as u32];
            let seq = LayaSeq {
                ids: &ids,
                markers: &markers,
                qtype: 2,
            };
            let alone = model.run(ck, std::slice::from_ref(&seq)).unwrap();
            let prefix = [8u32; 15];
            let packed = model
                .run(
                    ck,
                    &[
                        LayaSeq {
                            ids: &prefix,
                            markers: &[2],
                            qtype: 0,
                        },
                        seq,
                    ],
                )
                .unwrap();
            assert_eq!(
                alone.logits,
                packed.logits[1..],
                "{} ragged {len} logits",
                ck.name()
            );
            assert_eq!(
                alone.act,
                packed.act[packed.n_act..],
                "{} ragged {len} action",
                ck.name()
            );
        }
        eprintln!("{} ragged tile/window boundaries bit-exact", ck.name());
    }
}
