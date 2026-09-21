use crate::device::MetalDevice;

#[test]
fn grouped_draft_attention_matches_windowed_softmax() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let (heads, kv, dim, slots, ring, stride) =
        (32usize, 8usize, 128usize, 4usize, 2112usize, 256usize);
    let rows = slots * 8;
    let ints = |v: Vec<u32>| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let sample = |i: usize, salt: usize| {
        half::bf16::from_f32(((i * salt + i / 19) % 127) as f32 / 128. - 0.5).to_f32()
    };
    let query: Vec<f32> = (0..rows * heads * dim).map(|i| sample(i, 11)).collect();
    let x = d
        .upload(
            &query
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let scratch = d
        .alloc((rows + 32) * heads * dim * 2 + slots * kv * 8 * 32 * 130 * 4)
        .unwrap();
    let out = d.alloc(rows * heads * dim * 4).unwrap();
    let order = [3usize, 1, 0, 2];
    let tiles = ints((0..slots).flat_map(|i| [(i * 8) as u32, 8]).collect());
    let pages = ints(
        (0..slots)
            .flat_map(|s| {
                (0..stride).map(move |p| (s * ring / 16 + (ring / 16 - 1 - p % (ring / 16))) as u32)
            })
            .collect(),
    );
    for prefix in [5usize, 2047, 2159] {
        let mut keys: Vec<f32> = (0..slots * ring * kv * dim).map(|i| sample(i, 7)).collect();
        let mut values: Vec<f32> = (0..keys.len()).map(|i| sample(i, 17)).collect();
        let index = |slot: usize, t: usize, kh: usize| {
            ((slot * ring + (ring / 16 - 1 - (t / 16) % (ring / 16)) * 16 + t % 16) * kv + kh) * dim
        };
        for (cohort, &slot) in order.iter().enumerate() {
            let last = prefix + cohort * 17 + 7;
            let first = (prefix + cohort * 17 + 1).saturating_sub(2048);
            let mut live = vec![false; ring];
            for t in first..=last {
                live[(ring / 16 - 1 - (t / 16) % (ring / 16)) * 16 + t % 16] = true;
            }
            for (r, valid) in live.iter().enumerate() {
                if !valid {
                    let base = (slot * ring + r) * kv * dim;
                    keys[base..base + kv * dim].fill(f32::NAN);
                    values[base..base + kv * dim].fill(f32::NAN);
                }
            }
            if prefix >= 2047 {
                // Make the query-relative boundary observable: the first
                // query must see this strongly matching key, while the last
                // query must mask it. A nearly uniform fixture can hide a
                // seven-token mask error inside the BF16 tolerance.
                for kh in 0..kv {
                    let qbase = (cohort * 8 * heads + kh * 4) * dim;
                    let dst = index(slot, first, kh);
                    for j in 0..dim {
                        keys[dst + j] = half::bf16::from_f32(query[qbase + j] * 16.).to_f32();
                        values[dst + j] = 3.;
                    }
                }
            }
        }
        let bf = |v: &[f32]| {
            d.upload(
                &v.iter()
                    .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let k = bf(&keys);
        let v = bf(&values);
        let meta = ints(
            order
                .iter()
                .enumerate()
                .flat_map(|(c, &s)| {
                    (0..8).flat_map(move |r| [s as u32, (prefix + c * 17 + r) as u32])
                })
                .collect(),
        );
        let scale = 1f32 / (dim as f32).sqrt();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "splash_df_query",
            &[&x, &scratch],
            &[heads as u32, kv as u32, rows as u32],
            [(rows * heads * dim).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "splash_df_attention_grouped",
            &[&scratch, &k, &v, &meta, &pages, &out, &tiles],
            &[
                heads as u32,
                kv as u32,
                stride as u32,
                scale.to_bits(),
                rows as u32,
            ],
            [kv, slots, 8],
            128,
        );
        cmd.dispatch(
            "splash_df_attention_join",
            &[&scratch, &out],
            &[rows as u32],
            [heads * rows, 1, 1],
            32,
        );
        cmd.finish().unwrap();
        let got = unsafe { out.read_f32(0, rows * heads * dim) };
        assert!(
            got.iter().all(|x| x.is_finite()),
            "poisoned unused ring row leaked"
        );
        for (cohort, &slot) in order.iter().enumerate() {
            let last = prefix + cohort * 17 + 7;
            for row in [0usize, 7] {
                // Splash draft.metal, draft_attention_split_phase: each
                // proposal owns its left window boundary, while all noisy
                // query keys remain noncausally visible.
                let first = (prefix + cohort * 17 + row + 1).saturating_sub(2048);
                for head in [0usize, 3, 4, 31] {
                    let qb = ((cohort * 8 + row) * heads + head) * dim;
                    let scores: Vec<f64> = (first..=last)
                        .map(|t| {
                            (0..dim)
                                .map(|j| {
                                    f64::from(query[qb + j])
                                        * f64::from(keys[index(slot, t, head / 4) + j])
                                })
                                .sum::<f64>()
                                * f64::from(scale)
                        })
                        .collect();
                    let hi = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probabilities: Vec<f64> = scores.iter().map(|s| (s - hi).exp()).collect();
                    let sum: f64 = probabilities.iter().sum();
                    for channel in [0usize, 31, 64, 127] {
                        let expected = probabilities
                            .iter()
                            .enumerate()
                            .map(|(t, p)| {
                                p * f64::from(values[index(slot, first + t, head / 4) + channel])
                            })
                            .sum::<f64>()
                            / sum;
                        assert!(
                            (f64::from(got[qb + channel]) - expected).abs() < 0.003,
                            "prefix={prefix} cohort={cohort} row={row} head={head} channel={channel}: {} vs {expected}",
                            got[qb + channel]
                        );
                    }
                }
            }
        }
    }
}
