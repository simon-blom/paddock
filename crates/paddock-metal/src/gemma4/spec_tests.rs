use super::*;

fn best(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn synthetic_weight_bytes(k: usize, n: usize, ty: u32, size: usize) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..k * n / 256 * size)
        .map(|i| ((i * 73 + i / 19 * 37 + 11) % 256) as u8)
        .collect();
    for b in bytes.chunks_exact_mut(size) {
        let at = if ty == 14 { 208 } else { 0 };
        b[at..at + 2].copy_from_slice(&half::f16::from_f32(1. / 1024.).to_le_bytes());
        if ty == 12 || ty == 13 {
            b[2..4].copy_from_slice(&half::f16::from_f32(1. / 4096.).to_le_bytes());
        }
    }
    bytes
}

#[test]
fn prefill96_preserves_ragged_domains_and_output_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let k = 256;
    let weights: Vec<_> = [(12, 37, 144), (13, 19, 176), (14, 71, 210), (23, 33, 136)]
        .into_iter()
        .map(|(ty, n, size)| Weight {
            buffer: d.upload(&synthetic_weight_bytes(k, n, ty, size)).unwrap(),
            ty,
            k,
            n,
        })
        .collect();
    let workspace = d.alloc(384 * k * 2).unwrap();
    for rows in [129usize, 191, 192, 257, 279, 287, 288] {
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let a: Vec<_> = weights
            .iter()
            .map(|w| d.alloc(rows * w.n * 4).unwrap())
            .collect();
        let b: Vec<_> = weights
            .iter()
            .map(|w| {
                d.upload(
                    &(0..(rows + 128) * w.n)
                        .flat_map(|_| f32::NAN.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            })
            .collect();
        let cmd = d.begin().unwrap();
        for (w, out) in weights.iter().zip(&a) {
            w.linear(&cmd, &x, out, rows, 1., &workspace);
        }
        cmd.finish().unwrap();
        for start in [0, 1] {
            for count in [1, 2, 3] {
                let planes: Vec<_> = weights.iter().zip(&b).skip(start).take(count).collect();
                let cmd = d.begin().unwrap();
                forward::prefill96_projection(&cmd, &planes, &x, rows, &workspace);
                cmd.finish().unwrap();
                for i in start..start + count {
                    let expected = unsafe { a[i].read_f32(0, rows * weights[i].n) };
                    let actual = unsafe { b[i].read_f32(0, expected.len()) };
                    assert!(actual.iter().all(|v| v.is_finite()));
                    assert_eq!(
                        expected, actual,
                        "rows={rows} start={start} count={count} plane={i}"
                    );
                    assert!(
                        unsafe { b[i].read_f32(rows * weights[i].n, 128 * weights[i].n) }
                            .iter()
                            .all(|x| x.is_nan())
                    );
                }
            }
        }
    }
}

#[test]
fn paired_projection_domains_preserve_odd_tails_and_mixed_formats() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let k = 5376;
    let weights: Vec<_> = [(12, 71, 144), (13, 19, 176), (14, 33, 210)]
        .into_iter()
        .map(|(ty, n, size)| Weight {
            buffer: d.upload(&synthetic_weight_bytes(k, n, ty, size)).unwrap(),
            ty,
            k,
            n,
        })
        .collect();
    let workspace = d.alloc(128 * k * 2).unwrap();
    for rows in [3usize, 4] {
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let a: Vec<_> = weights
            .iter()
            .map(|w| d.alloc(rows * w.n * 4).unwrap())
            .collect();
        let b: Vec<_> = weights
            .iter()
            .map(|w| d.alloc(rows * w.n * 4).unwrap())
            .collect();
        for count in [2, 3] {
            let cmd = d.begin().unwrap();
            for (w, out) in weights.iter().zip(&a).take(count) {
                w.linear(&cmd, &x, out, rows, 1., &workspace);
            }
            let planes: Vec<_> = weights.iter().zip(&b).take(count).collect();
            forward::pair_projection(&cmd, &planes, &x, rows);
            cmd.finish().unwrap();
            for i in 0..count {
                let expected = unsafe { a[i].read_f32(0, rows * weights[i].n) };
                let actual = unsafe { b[i].read_f32(0, expected.len()) };
                assert!(actual.iter().all(|v| v.is_finite()));
                assert_eq!(expected, actual, "rows={rows} domains={count} domain={i}");
            }
        }
    }
}

#[test]
fn wide_f32_verification_tiles_match_four_row_gpu_arithmetic_exactly() {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let (k, n) = (5376usize, 71usize);
    for (ty, size) in [(12u32, 144usize), (13, 176), (14, 210), (23, 136)] {
        let bytes = synthetic_weight_bytes(k, n, ty, size);
        let w = device.upload(&bytes).unwrap();
        for rows in [2usize, 3, 4, 5, 6, 7, 8, 9, 15, 16, 23, 32] {
            let x = device
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = device.alloc(rows * n * 4).unwrap();
            let b = device.alloc(rows * n * 4).unwrap();
            let cmd = device.begin().unwrap();
            let p = [k as u32, n as u32, rows as u32, ty, 1f32.to_bits()];
            cmd.dispatch(
                "linear_kquant_tail4",
                &[&w, &x, &a],
                &p,
                [n.div_ceil(16), rows.div_ceil(4), 1],
                128,
            );
            let tile = if rows <= 8 { 8 } else { 16 };
            cmd.dispatch(
                if tile == 8 {
                    "gemma_verify8"
                } else {
                    "gemma_verify16"
                },
                &[&w, &x, &b],
                &p,
                [n.div_ceil(16), rows.div_ceil(tile), 1],
                128,
            );
            cmd.finish().unwrap();
            let a = unsafe { a.read_f32(0, rows * n) };
            let b = unsafe { b.read_f32(0, rows * n) };
            assert!(a.iter().chain(&b).all(|x| x.is_finite()));
            assert_eq!(a, b, "format={ty}, rows={rows}");
            if rows <= 4 && matches!(ty, 12..=14) {
                let pair = device.alloc(rows * n * 4).unwrap();
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    match rows {
                        2 => "gemma_pair2",
                        3 => "gemma_pair3",
                        _ => "gemma_pair4",
                    },
                    &[&w, &x, &pair],
                    &p,
                    [n.div_ceil(32), 1, 1],
                    128,
                );
                cmd.finish().unwrap();
                assert_eq!(
                    a,
                    unsafe { pair.read_f32(0, rows * n) },
                    "pair rows={rows} format={ty}"
                );
            }
            if (5..=8).contains(&rows) {
                let full = device.alloc(rows * n * 4).unwrap();
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    match rows {
                        5 => "gemma_full5",
                        6 => "gemma_full6",
                        7 => "gemma_full7",
                        _ => "gemma_full8",
                    },
                    &[&w, &x, &full],
                    &p,
                    [n.div_ceil(16), 1, 1],
                    128,
                );
                cmd.finish().unwrap();
                assert_eq!(
                    a,
                    unsafe { full.read_f32(0, rows * n) },
                    "full {rows}, format={ty}"
                );
            }
            let matrix = device.alloc(rows * n * 4).unwrap();
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "gemma_f32_32",
                &[&w, &x, &matrix],
                &p,
                [n.div_ceil(16), rows.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            let matrix = unsafe { matrix.read_f32(0, rows * n) };
            assert!(matrix.iter().all(|x| x.is_finite()));
            let error = a
                .iter()
                .zip(&matrix)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // Matrix and SIMD reduction trees are different. Use a relative
            // infinity-norm arithmetic bound for these deliberately large
            // synthetic weights; the real continuation gate stays 0.1 and
            // the external same-weight generation gate remains exact bytes.
            let magnitude = a.iter().map(|x| x.abs()).fold(1f32, f32::max);
            assert!(
                error / magnitude < 0.00001,
                "F32 MPP format={ty}, rows={rows}, error={error}, magnitude={magnitude}"
            );
            let staged = device.alloc(rows * n * 4).unwrap();
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "gemma_staged_f32_32",
                &[&w, &x, &staged],
                &p,
                [n.div_ceil(16), rows.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            assert_eq!(
                matrix,
                unsafe { staged.read_f32(0, rows * n) },
                "F32 staging format={ty}, rows={rows}"
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF and PADDOCK_GEMMA4_MTP"]
fn assistant_chains_and_every_rejection_boundary_preserve_target_state() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let assistant = std::env::var("PADDOCK_GEMMA4_MTP").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap();
    let mut m = Gemma4::load(Path::new(&path), 4096, 4, None).unwrap();
    m.attach_mtp(Path::new(&assistant)).unwrap();
    let text = format!(
        "<bos><|turn>user\n{}\nContinue counting: one, two, three,<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        "The village has a river and a stone bridge. ".repeat(if m.moe_scratch.is_some() {
            220
        } else {
            160
        })
    );
    let tokens = tok.encode(&text).unwrap();
    assert!(tokens.len() > m.ring);
    let first = m.forward_prefill(0, &tokens).unwrap();
    let restored = m.forward_prefill(1, &tokens).unwrap();
    assert_eq!(best(&first), best(&restored));
    let pending = best(&restored);
    let before = m
        .slots
        .iter()
        .map(|s| s.history.clone())
        .collect::<Vec<_>>();
    let a = m.mtp_draft(&[(1, pending)], 7).unwrap().unwrap();
    let b = m.mtp_draft(&[(1, pending)], 7).unwrap().unwrap();
    let c = m
        .mtp_draft(&[(0, pending), (1, pending)], 7)
        .unwrap()
        .unwrap();
    assert_eq!(a, b, "draft chain changed the persistent target hidden");
    assert_eq!(a[0], c[1], "batched assistant changed draft IDs");
    assert_eq!(
        before,
        m.slots
            .iter()
            .map(|s| s.history.clone())
            .collect::<Vec<_>>()
    );
    println!("assistant draft: {:?}", tok.decode(&a[0], false).unwrap());

    // Cross a global page boundary and the sliding ring boundary. The
    // eight-token verify writes deliberately wrong future tokens; none may
    // survive a rejected suffix. Compare independent native GPU walks, not
    // a CPU model. Batch vs single-row rounding is checked by logit error + argmax;
    // the external same-weight harness is the exact generation gate.
    let mut worst_error = 0f32;
    let boundaries = if m.moe_scratch.is_some() {
        vec![m.ring + 15, 2045]
    } else {
        vec![m.ring + 15]
    };
    for prefix_len in boundaries {
        for n in 1..=spec::BLOCK {
            let prefix = &tokens[..prefix_len];
            m.forward_prefill(0, prefix).unwrap();
            m.forward_prefill(1, prefix).unwrap();
            let chunk: Vec<_> = tokens[prefix_len..prefix_len + spec::BLOCK].to_vec();
            let reqs = vec![(1, prefix_len, chunk.clone())];
            m.verify(&reqs, false).unwrap();
            assert_eq!(m.slots[1].history.len(), prefix_len);
            assert!(
                m.execute(&[(0, chunk[0], prefix_len as u32)], &[0])
                    .is_err()
            );
            assert!(m.commit_verify(&[0]).is_err());
            m.commit_verify(&[n as u32]).unwrap();
            for (i, &id) in chunk[..n].iter().enumerate() {
                m.execute(&[(0, id, (prefix_len + i) as u32)], &[0])
                    .unwrap();
            }
            assert_eq!(m.slots[0].history, m.slots[1].history);
            assert_eq!(m.mtp.as_ref().unwrap().cursor[1], Some(prefix_len + n));
            for step in 0..3 {
                let pos = prefix_len + n + step;
                let id = tokens[pos];
                let a = m.execute(&[(0, id, pos as u32)], &[0]).unwrap();
                let b = m.execute(&[(1, id, pos as u32)], &[0]).unwrap();
                let error = a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                worst_error = worst_error.max(error);
                assert!(b.iter().all(|x| x.is_finite()));
                assert_eq!(
                    best(&a),
                    best(&b),
                    "reject n={n}, step={step}, error={error}"
                );
                assert!(error < 0.1, "reject n={n}, step={step}, error={error}");
            }
        }
    }
    println!("all rejection boundaries: max continuation logit error {worst_error}");
    let pos = m.slots[1].history.len();
    m.verify(&[(1, pos, vec![pending; 3])], true).unwrap();
    assert!(m.prefill_abort(1));
    assert!(m.require_committed().is_ok());
    assert!(m.mtp.as_ref().unwrap().cursor[1].is_none());
    assert!(
        m.verify(&[(0, m.context - 1, vec![pending; 2])], true)
            .is_err()
    );
    m.reset();
    assert!(m.mtp.as_ref().unwrap().cursor.iter().all(Option::is_none));

    let mut logits = m.forward_prefill(0, &tokens).unwrap();
    let mut reference = vec![best(&logits)];
    for _ in 0..31 {
        logits = m.forward(*reference.last().unwrap()).unwrap();
        reference.push(best(&logits));
    }
    m.reset();
    let mut output = vec![best(&m.forward_prefill(1, &tokens).unwrap())];
    let mut accepted = 0;
    let mut rounds = 0;
    while output.len() < reference.len() {
        let pending = *output.last().unwrap();
        let k = if rounds % 2 == 0 { 3 } else { 7 };
        let draft = m.mtp_draft(&[(1, pending)], k).unwrap().unwrap().remove(0);
        let chunk: Vec<_> = std::iter::once(pending).chain(draft).collect();
        let picks = m
            .forward_spec_batch(&[(1, m.slots[1].history.len(), chunk.clone())])
            .unwrap()
            .unwrap();
        let count = 1 + chunk[1..]
            .iter()
            .zip(&picks)
            .take_while(|(a, b)| a == b)
            .count();
        output.extend_from_slice(&picks[..count]);
        accepted += count - 1;
        rounds += 1;
    }
    assert_eq!(&output[..reference.len()], &reference);
    assert!(accepted > 0);
    println!("32-token greedy MTP parity: {rounds} rounds, {accepted} accepted draft tokens");
}
