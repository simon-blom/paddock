use super::*;

fn pick(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0 as u32
}

#[test]
fn compact_recurrent_updates_commit_bit_exact_gpu_state() {
    recurrent_updates_commit_exact(false);
}

#[test]
fn mlx_compact_recurrent_updates_commit_bit_exact_gpu_state() {
    recurrent_updates_commit_exact(true);
}

fn recurrent_updates_commit_exact(mlx: bool) {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let kh = 2usize;
    let vh = 4usize;
    let cd = (kh * 2 + vh) * 128;
    let rows = 11usize;
    let slots = 2usize;
    let floats = |n: usize, salt: usize| -> Buffer {
        device
            .upload(
                &(0..n)
                    .flat_map(|i| (((i * salt) % 31) as f32 / 512. - 15. / 512.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    };
    let upload = |x: &[u32]| {
        device
            .upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let q = floats(rows * cd, 7);
    let gates = floats(rows * vh * 2, 13);
    let input = floats(rows * cd, 17);
    let state = floats(slots * vh * 128 * 128, 3);
    let original = unsafe { state.read_f32(0, slots * vh * 128 * 128) };
    let out = device.alloc(rows * vh * 128 * 4).unwrap();
    let reference_out = device.alloc(rows * vh * 128 * 4).unwrap();
    let updates = device.alloc(rows * vh * 257 * 4).unwrap();
    let meta = upload(
        &(0..rows)
            .flat_map(|r| {
                if r < 8 {
                    [1, 40 + r as u32]
                } else {
                    [0, 91 + (r - 8) as u32]
                }
            })
            .collect::<Vec<_>>(),
    );
    let spans = upload(&[0, 8, 1, 0, 8, 3, 0, 0]);
    let zero = upload(&[0; 11]);
    let checkpoints = upload(&[0; 8]);
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        if mlx { "mlx_dn_verify" } else { "dn_verify" },
        &[&q, &gates, &state, &spans, &meta, &out, &updates],
        &[
            kh as u32,
            vh as u32,
            cd as u32,
            rows as u32,
            slots as u32,
            0,
            rows as u32,
        ],
        [if mlx { 32 } else { 8 }, vh, 2],
        128,
    );
    cmd.finish().unwrap();
    assert_eq!(original, unsafe { state.read_f32(0, original.len()) });
    for n in 1..=8 {
        let commit = upload(&[0, n, 1, 0, 8, (n - 1) % 3 + 1, 0, 0]);
        let reference = floats(original.len(), 3);
        let actual = floats(original.len(), 3);
        let cv = floats(slots * cd * 3, 19);
        let cv_reference = floats(slots * cd * 3, 19);
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            if mlx {
                "mlx_dn_recurrent"
            } else {
                "dn_recurrent"
            },
            &[
                &q,
                &gates,
                &reference,
                &commit,
                &meta,
                &reference_out,
                &zero,
            ],
            &[
                kh as u32,
                vh as u32,
                cd as u32,
                rows as u32,
                slots as u32,
                0,
                0,
            ],
            [if mlx { 32 } else { 8 }, vh, 2],
            128,
        );
        cmd.dispatch(
            if mlx {
                "mlx_dn_verify_commit"
            } else {
                "dn_verify_commit"
            },
            &[&updates, &actual, &commit],
            &[vh as u32, slots as u32, rows as u32],
            [if mlx { 32 } else { 8 }, vh, 2],
            128,
        );
        cmd.dispatch(
            "dn_conv_commit",
            &[&input, &cv_reference, &commit, &meta, &checkpoints],
            &[
                kh as u32,
                vh as u32,
                cd as u32,
                rows as u32,
                slots as u32,
                0,
                0,
            ],
            [cd.div_ceil(256), 2, 1],
            256,
        );
        cmd.dispatch(
            "dn_verify_conv_commit",
            &[&input, &cv, &commit],
            &[cd as u32, slots as u32, rows as u32],
            [cd.div_ceil(256), 1, 2],
            256,
        );
        cmd.finish().unwrap();
        for (first, count) in [(0, n as usize), (8, ((n - 1) % 3 + 1) as usize)] {
            assert_eq!(
                unsafe { out.read_f32(first * vh * 128, count * vh * 128) },
                unsafe { reference_out.read_f32(first * vh * 128, count * vh * 128) },
                "verification output, accepted {n}, first {first}"
            );
        }
        assert_eq!(
            unsafe { actual.read_f32(0, original.len()) },
            unsafe { reference.read_f32(0, original.len()) },
            "state, accepted {n}"
        );
        assert_eq!(
            unsafe { cv.read_f32(0, slots * cd * 3) },
            unsafe { cv_reference.read_f32(0, slots * cd * 3) },
            "conv, accepted {n}"
        );
    }
}

#[test]
fn hierarchical_top16_and_argmax_preserve_ties_and_ragged_vocab() {
    let device = MetalDevice::new(Some(64 << 20)).unwrap();
    let n = 10003usize;
    let rows = 3usize;
    let values: Vec<_> = (0..rows * n)
        .map(|i| ((i * 17 + i / n) % 997) as f32 - 500.)
        .collect();
    let x = device
        .upload(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let parts = device.alloc(rows * n.div_ceil(4096) * 16 * 8).unwrap();
    let top = device.alloc(rows * 16 * 8).unwrap();
    let picks = device.alloc(rows * 4).unwrap();
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "df_top16",
        &[&x, &parts],
        &[n as u32],
        [n.div_ceil(4096), rows, 1],
        256,
    );
    cmd.dispatch(
        "df_top16_merge",
        &[&parts, &top],
        &[(n.div_ceil(4096) * 16) as u32],
        [rows, 1, 1],
        256,
    );
    cmd.dispatch("spec_argmax", &[&x, &picks], &[n as u32], [rows, 1, 1], 256);
    cmd.finish().unwrap();
    let top = unsafe { top.read_u32(rows * 16 * 2) };
    let picks = unsafe { picks.read_u32(rows) };
    for row in 0..rows {
        let mut ids: Vec<_> = (0..n).collect();
        ids.sort_by(|&a, &b| {
            values[row * n + b]
                .total_cmp(&values[row * n + a])
                .then(a.cmp(&b))
        });
        assert_eq!(picks[row], ids[0] as u32);
        for j in 0..16 {
            assert_eq!(top[(row * 16 + j) * 2], ids[j] as u32);
            assert_eq!(
                top[(row * 16 + j) * 2 + 1],
                values[row * n + ids[j]].to_bits()
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_QWEN_MODEL and an M5"]
fn compact_verify_all_rejection_boundaries_and_mtp_chain() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").unwrap();
    let path = Path::new(&path);
    let mut model = Qwen35::load(path, 512, 2, None).unwrap();
    model.attach_mtp(path).unwrap();
    let prompt: Vec<_> = (1000..1031).collect();
    let first = model.forward_prefill(0, &prompt).unwrap();
    let mut pending = pick(&first);
    let mut reference = vec![pending];
    for _ in 0..20 {
        pending = pick(&model.forward(pending).unwrap());
        reference.push(pending);
    }
    for accepted in 1..=8 {
        model.reset();
        model.forward_prefill(0, &prompt).unwrap();
        let chunk: Vec<_> = (0..8)
            .map(|i| {
                if i < accepted {
                    reference[i]
                } else {
                    1100 + i as u32
                }
            })
            .collect();
        let state = unsafe { model.state.read_f32(0, model.geometry.state()) };
        let conv = unsafe { model.conv.read_f32(0, model.geometry.conv() * 3) };
        let logits = model.verify(&[(0, prompt.len(), chunk)]).unwrap();
        assert_eq!(model.slots[0].history.len(), prompt.len());
        assert_eq!(
            state,
            unsafe { model.state.read_f32(0, model.geometry.state()) },
            "verify mutated recurrent state"
        );
        assert_eq!(
            conv,
            unsafe { model.conv.read_f32(0, model.geometry.conv() * 3) },
            "verify mutated convolution"
        );
        assert!(model.forward(reference[0]).is_err());
        assert!(model.commit_verify(&[0]).is_err());
        for i in 0..accepted {
            assert_eq!(
                pick(&logits[i * model.vocab..(i + 1) * model.vocab]),
                reference[i + 1],
                "boundary {accepted}, row {i}"
            );
        }
        model.commit_verify(&[accepted as u32]).unwrap();
        assert!(model.commit_verify(&[1]).is_err());
        assert_eq!(model.slots[0].history.len(), prompt.len() + accepted);
        assert_eq!(
            pick(&model.forward(reference[accepted]).unwrap()),
            reference[accepted + 1],
            "after rollback {accepted}"
        );
    }
    model.reset();
    model.forward_prefill(0, &prompt).unwrap();
    // A shorter synchronous MTP call must preserve the consumed prefix,
    // including when the longer call crosses another physical KV page.
    let shallow = model.mtp_draft(&[(0, reference[0])], 2).unwrap().unwrap();
    let deep = model.mtp_draft(&[(0, reference[0])], 7).unwrap().unwrap();
    assert_eq!(shallow[0], deep[0][..2]);
    let mut output = vec![reference[0]];
    let mut rounds = 0;
    let mut accepted_total = 0;
    while output.len() < 20 {
        let token = *output.last().unwrap();
        let draft = model
            .mtp_draft(&[(0, token)], 3)
            .unwrap()
            .unwrap()
            .remove(0);
        let chunk: Vec<_> = std::iter::once(token).chain(draft).collect();
        let req = vec![(0, model.slots[0].history.len(), chunk.clone())];
        let picks = model.forward_spec_batch(&req).unwrap().unwrap();
        let count = 1 + chunk[1..]
            .iter()
            .zip(&picks)
            .take_while(|(a, b)| a == b)
            .count();
        output.extend_from_slice(&picks[..count]);
        rounds += 1;
        accepted_total += count - 1;
    }
    assert_eq!(&output[..20], &reference[..20]);
    eprintln!("MTP {rounds} rounds, {accepted_total} accepted draft tokens");
    assert!(accepted_total > 0, "MTP never accepted a draft");
}

#[test]
#[ignore = "requires PADDOCK_METAL_QWEN_MODEL, PADDOCK_METAL_DFLASH_MODEL and an M5"]
fn dflash2_ragged_slots_prefix_resume_and_cancellation() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").unwrap();
    let draft = std::env::var("PADDOCK_METAL_DFLASH_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 2304, 4, None).unwrap();
    model.attach_mtp(Path::new(&path)).unwrap();
    model.attach_dflash(Path::new(&draft)).unwrap();
    let mapped = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_gguf(mapped.gguf()).unwrap();
    let short = tokenizer
        .encode("Write a Python function that implements merge sort. Explain its time complexity.")
        .unwrap();
    let long = tokenizer
        .encode(&format!(
            "{}\nWrite a detailed explanation of sorting algorithms.",
            "Sorting algorithms transform a sequence into ordered elements. ".repeat(240)
        ))
        .unwrap();
    assert!(long.len() > 2048 && long.len() < 2250);
    let prompts = [short.clone(), long, short];
    let mut reference = Vec::new();
    for p in &prompts {
        model.reset();
        let first = model.forward_prefill(0, p).unwrap();
        let mut ids = vec![pick(&first)];
        for _ in 0..31 {
            ids.push(pick(&model.forward(*ids.last().unwrap()).unwrap()));
        }
        reference.push(ids);
    }
    model.reset();
    let slots = [2, 0, 3];
    let mut output = vec![Vec::new(); 3];
    for (i, &slot) in slots.iter().enumerate() {
        output[i].push(pick(&model.forward_prefill(slot, &prompts[i]).unwrap()));
    }
    assert!(
        model.take_prefill_reused(0) > 0,
        "long conditioning prefix not reused"
    );
    let mut accepted = 0;
    while output.iter().any(|o| o.len() < 32) {
        let live: Vec<_> = slots
            .iter()
            .enumerate()
            .filter(|(i, _)| output[*i].len() < 32)
            .map(|(i, &s)| (i, s, *output[i].last().unwrap()))
            .collect();
        let drafts = model
            .spec_draft_batch(&live.iter().map(|r| (r.1, r.2)).collect::<Vec<_>>(), 7)
            .unwrap()
            .unwrap();
        let reqs: Vec<_> = live
            .iter()
            .zip(drafts)
            .map(|(&(i, s, t), ds)| {
                (
                    s,
                    model.slots[s].history.len(),
                    std::iter::once(t)
                        .chain(ds.into_iter().take(31 - output[i].len()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        // Exercise sampled verifier's explicit accept/commit transaction too.
        let logits = model.forward_spec_verify(&reqs).unwrap().unwrap();
        let picks: Vec<_> = logits.chunks(model.vocab).map(pick).collect();
        let mut counts = Vec::new();
        let mut base = 0;
        for ((i, _, _), (_, _, chunk)) in live.iter().zip(&reqs) {
            let count = 1 + chunk[1..]
                .iter()
                .zip(&picks[base..])
                .take_while(|(a, b)| a == b)
                .count();
            output[*i].extend_from_slice(&picks[base..base + count]);
            counts.push(count as u32);
            base += chunk.len();
            accepted += count - 1;
        }
        model.spec_commit(&counts).unwrap();
    }
    assert_eq!(output, reference);
    assert!(
        accepted > 12,
        "DFlash2 acceptance unexpectedly low: {accepted}"
    );
    model.prefill_begin(1, vec![1000; 99]).unwrap();
    model.forward_mixed(&[], 32).unwrap();
    model.prefill_abort(1);
    model.release_inactive_slots(&[true, false, true, false]);
    assert_eq!(
        pick(&model.forward_prefill(3, &prompts[0]).unwrap()),
        reference[0][0]
    );
    assert!(model.verify(&[(2, 1, vec![0]), (2, 1, vec![0])]).is_err());
    assert!(model.verify(&[(9, 1, vec![0])]).is_err());
}
