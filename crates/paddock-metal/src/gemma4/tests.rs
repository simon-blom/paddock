use super::*;

#[test]
fn prefill_tile_never_adds_a_weight_sweep() {
    for rows in 0..=CHUNK + 1 {
        assert_eq!(
            forward::prefill96(rows),
            (129..=192).contains(&rows) || (257..=288).contains(&rows)
        );
        if forward::prefill96(rows) {
            assert_eq!(rows.div_ceil(96), rows.div_ceil(128));
            assert!(rows.div_ceil(96) * 96 < rows.div_ceil(128) * 128);
        }
    }
}

#[test]
fn gemma_kernels_compile_on_m5() {
    MetalDevice::new(Some(128 << 20)).unwrap();
}

#[test]
fn both_geometries_match_independent_gpu_attention_across_window_wrap() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    for strict in [false, true] {
        for (heads, hd, kh, window) in [
            (32, 256, 16, 64),
            (32, 512, 4, 0),
            (16, 256, 8, 64),
            (16, 512, 2, 0),
        ] {
            for prefix in [3usize, 97, 253] {
                let rows = 37;
                let ring = 128;
                let stride = (prefix + rows).div_ceil(16);
                let kv_rows = if window == 0 {
                    stride * 16 * 2
                } else {
                    ring * 2
                };
                let q = d
                    .upload(
                        &(0..rows * heads * hd)
                            .flat_map(|i| {
                                (((i * 7 + i / hd) % 17) as f32 / if strict { 257. } else { 256. }
                                    - 8. / 256.)
                                    .to_le_bytes()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let halfs = |mult: usize| {
                    (0..kv_rows * kh * hd)
                        .flat_map(|i| {
                            half::f16::from_f32(
                                ((i * mult + i / hd) % 19) as f32 / 256. - 9. / 256.,
                            )
                            .to_le_bytes()
                        })
                        .collect::<Vec<_>>()
                };
                let k = d.upload(&halfs(3)).unwrap();
                let v = d.upload(&halfs(11)).unwrap();
                let meta = upload(
                    &(0..rows)
                        .flat_map(|r| [1, (prefix + r) as u32])
                        .collect::<Vec<_>>(),
                );
                let pages = upload(
                    &(0..stride)
                        .map(|i| (i * 2) as u32)
                        .chain((0..stride).rev().map(|i| (i * 2 + 1) as u32))
                        .collect::<Vec<_>>(),
                );
                let tiles = upload(&[32, 5, 0, 32]);
                let selected = upload(&(0..rows as u32).rev().collect::<Vec<_>>());
                let qhalf = d.alloc((rows + 32) * heads * hd * 2).unwrap();
                let out = d.alloc(rows * heads * hd * 4).unwrap();
                let other = d.alloc(out.len()).unwrap();
                let shared = d.alloc(out.len()).unwrap();
                let parts = d.alloc(rows * heads * SPLITS * (hd + 2) * 4).unwrap();
                let p = [heads as u32, kh as u32, stride as u32, window, ring as u32];
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "attention_query",
                    &[&q, &qhalf],
                    &[(heads * hd) as u32, 0, rows as u32],
                    [((rows + 32) * heads * hd).div_ceil(256), 1, 1],
                    256,
                );
                cmd.dispatch(
                    if hd == 256 {
                        if strict {
                            "gmoe_prefill256"
                        } else {
                            "gemma_prefill256"
                        }
                    } else {
                        if strict {
                            "gmoe_prefill512"
                        } else {
                            "gemma_prefill512"
                        }
                    },
                    &[
                        if strict { &q } else { &qhalf },
                        &k,
                        &v,
                        &meta,
                        &pages,
                        &out,
                        &tiles,
                    ],
                    &p,
                    [heads, 2, 1],
                    128,
                );
                cmd.finish().unwrap();
                let expected = unsafe { out.read_f32(0, rows * heads * hd) };
                assert!(expected.iter().all(|x| x.is_finite()));
                // Two separate image domains with causal text between them.
                // An independent split-decode GPU contraction expresses each
                // interval with an upper position and its own visible width.
                // The actual image kernel must retain the real query's SWA lower
                // bound, not shift the whole window forward to the image's end.
                let uppers = (0..rows)
                    .map(|r| {
                        prefix
                            + if (2..18).contains(&r) {
                                17
                            } else if (24..35).contains(&r) {
                                34
                            } else {
                                r
                            }
                    })
                    .collect::<Vec<_>>();
                let limits = upload(&uppers.iter().map(|&n| n as u32).collect::<Vec<_>>());
                let bounded_meta = upload(
                    &uppers
                        .iter()
                        .flat_map(|&p| [1, p as u32])
                        .collect::<Vec<_>>(),
                );
                let selections = (0..rows).map(|r| upload(&[r as u32])).collect::<Vec<_>>();
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    if hd == 256 {
                        if strict {
                            "gmoe_image_prefill256"
                        } else {
                            "gemma_image_prefill256"
                        }
                    } else {
                        if strict {
                            "gmoe_image_prefill512"
                        } else {
                            "gemma_image_prefill512"
                        }
                    },
                    &[
                        if strict { &q } else { &qhalf },
                        &k,
                        &v,
                        &meta,
                        &pages,
                        &out,
                        &tiles,
                        &limits,
                    ],
                    &p,
                    [heads, 2, 1],
                    128,
                );
                for (r, selected) in selections.iter().enumerate() {
                    let mut bounded = p.to_vec();
                    if window != 0 {
                        let low = (prefix + r + 1).saturating_sub(window as usize);
                        bounded[3] = (uppers[r] - low + 1) as u32;
                    }
                    bounded.push(1);
                    cmd.dispatch(
                        if hd == 256 {
                            "gemma_decode256"
                        } else {
                            "gemma_decode512"
                        },
                        &[&q, &k, &v, &bounded_meta, &pages, selected, &parts],
                        &bounded,
                        [kh, 1, 1],
                        128,
                    );
                    cmd.dispatch(
                        "gemma_merge",
                        &[&parts, &other, selected],
                        &[heads as u32, 1, hd as u32],
                        [heads, 1, 1],
                        32,
                    );
                }
                cmd.finish().unwrap();
                let image = unsafe { out.read_f32(0, expected.len()) };
                let bounded = unsafe { other.read_f32(0, expected.len()) };
                assert!(image.iter().chain(&bounded).all(|v| v.is_finite()));
                let error = image
                    .iter()
                    .zip(&bounded)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    error < 0.00001,
                    "image hd={hd}, prefix={prefix}, error={error}"
                );
                assert!(
                    image
                        .iter()
                        .zip(&expected)
                        .any(|(a, b)| (a - b).abs() > 0.00001),
                    "image fixture did not distinguish bidirectional from causal attention"
                );
                if hd == 256 {
                    let verify_tiles = upload(
                        &(0..rows)
                            .step_by(8)
                            .flat_map(|at| [at as u32, (rows - at).min(8) as u32])
                            .collect::<Vec<_>>(),
                    );
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        "gemma_verify_attn256",
                        &[&q, &k, &v, &meta, &pages, &other, &verify_tiles],
                        &p,
                        [heads, rows.div_ceil(8), 1],
                        128,
                    );
                    cmd.finish().unwrap();
                    let verify = unsafe { other.read_f32(0, expected.len()) };
                    assert!(verify.iter().all(|x| x.is_finite()));
                    let error = expected
                        .iter()
                        .zip(&verify)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        error < 0.00001,
                        "F32 matrix attention hd={hd}, prefix={prefix}, error={error}"
                    );
                }
                for splits in [1, 3, 17, 32] {
                    let cmd = d.begin().unwrap();
                    let mut params = p.to_vec();
                    params.push(splits);
                    cmd.dispatch(
                        if hd == 256 {
                            "gemma_decode256"
                        } else {
                            "gemma_decode512"
                        },
                        &[&q, &k, &v, &meta, &pages, &selected, &parts],
                        &params,
                        [kh, rows, splits as usize],
                        128,
                    );
                    cmd.dispatch(
                        "gemma_merge",
                        &[&parts, &other, &selected],
                        &[heads as u32, splits, hd as u32],
                        [heads * rows, 1, 1],
                        32,
                    );
                    cmd.dispatch(
                        "gemma_merge_shared",
                        &[&parts, &shared, &selected],
                        &[heads as u32, splits, hd as u32],
                        [heads * rows, 1, 1],
                        32,
                    );
                    cmd.finish().unwrap();
                    let actual = unsafe { other.read_f32(0, expected.len()) };
                    let merged = unsafe { shared.read_f32(0, expected.len()) };
                    assert_eq!(
                        actual, merged,
                        "merge hd={hd} prefix={prefix} splits={splits}"
                    );
                    assert!(actual.iter().all(|x| x.is_finite()));
                    let error = expected
                        .iter()
                        .zip(actual)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        error < 0.00001,
                        "hd={hd} prefix={prefix} splits={splits} error={error}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Gemma 4 31B GGUF; set PADDOCK_GEMMA4_GGUF"]
fn gemma31b_greedy_smoke() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let tokens=tok.encode("<bos><|turn>user\nWhat is the capital of France? Answer with the city name only.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>").unwrap();
    let mut m = Gemma4::load(Path::new(&path), 4096, 4, None).unwrap();
    let mut logits = m.forward_prefill(0, &tokens).unwrap();
    let mut generated = Vec::new();
    for _ in 0..24 {
        let id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        print!("{}", tok.decode(&[id], false).unwrap());
        generated.push(id);
        if tok.id_to_token(id).as_deref() == Some("<turn|>") {
            break;
        }
        logits = m.forward(id).unwrap();
    }
    println!("\nlast_gpu_ms={}", m.last_gpu_seconds * 1000.);
    assert!(tok.decode(&generated, true).unwrap().contains("Paris"));
}

#[test]
#[ignore = "requires Gemma 4 31B GGUF; set PADDOCK_GEMMA4_GGUF"]
fn ring_snapshots_partial_page_cow_and_abort_preserve_continuations() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let mut m = Gemma4::load(Path::new(&path), 4096, 4, None).unwrap();
    let text = format!(
        "<bos><|turn>user\n{}\nWhat is 8 times 9? Number only.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        "The quiet village has a river and an old stone bridge. ".repeat(180)
    );
    let tokens = tok.encode(&text).unwrap();
    assert!(tokens.len() > m.ring && tokens.len() < m.context);
    let prefix = &tokens[..tokens.len() - 1];
    m.forward_prefill(0, prefix).unwrap();
    // Reference append and restored append use the same GPU arithmetic path;
    // byte identity is the state-management gate, not a widened tolerance.
    let a = m
        .execute(&[(0, *tokens.last().unwrap(), prefix.len() as u32)], &[0])
        .unwrap();
    let b = m.forward_prefill(1, &tokens).unwrap();
    assert_eq!(m.take_prefill_reused(1), prefix.len());
    assert_eq!(a, b, "snapshot restore or partial-page COW changed logits");
    let c = m.forward_prefill(2, &tokens).unwrap();
    assert_eq!(c, a);
    m.prefill_begin(3, tokens.clone()).unwrap();
    assert!(m.prefill_abort(3));
    assert!(m.slots[3].history.is_empty());
    let best = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let mut logits = a;
    for step in 0..16 {
        let id = best(&logits);
        let pos = tokens.len() + step;
        let before = m.execute(&[(0, id, pos as u32)], &[0]).unwrap();
        // Interleaved forwards share all temporary storage but no live state.
        let other = m.execute(&[(2, id, pos as u32)], &[0]).unwrap();
        let after = m.execute(&[(1, id, pos as u32)], &[0]).unwrap();
        assert_eq!(before, other);
        assert_eq!(before, after);
        logits = before;
    }
    assert!(logits.iter().all(|x| x.is_finite()));
    m.reset();
    for c in &mut m.cache {
        c.table.clear(&mut m.pool);
        c.history.clear();
    }
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
}
