//! Packed affine projections: native 4-bit MPP operands, F32 quant-group
//! accumulation, BF16 activation boundaries. No per-dispatch dequantization.
use crate::{
    device::{Buffer, Commands},
    weights::Weight,
};
pub(crate) const PACKED4: u32 = 0x101;
pub(crate) const LONG_DECODE: usize = 2048;
pub(crate) const DECODE_PARTS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Decode,
    Prefix,
    Prefill,
}

impl Phase {
    pub(crate) fn for_row(prefill: bool, position: usize) -> Self {
        if !prefill {
            Self::Decode
        } else if position < 128 {
            // Keep narrow, long-K projections parallel for initial prompts.
            // Elect by absolute token position, not a batch/chunk's size, so
            // prefix replay and peer admission cannot change the contraction.
            Self::Prefix
        } else {
            Self::Prefill
        }
    }
}

#[cfg(test)]
#[path = "splash_draft_tests.rs"]
mod draft_tests;

pub(crate) fn attention_scratch_bytes(tiles: usize, heads: usize) -> usize {
    tiles * heads * 4 * 64 * 256 * 2
}

pub(crate) fn workspace_bytes(rows: usize, max_k: usize, max_n: usize) -> usize {
    rows.div_ceil(64) * 64 * (max_k * 2 + max_k / 64 * 4 + (max_n * 4).max(65536) * 4)
}

fn partitions(k: usize, n: usize) -> usize {
    let mut parts = (512 / n.div_ceil(128)).clamp(1, 16).min(k / 64);
    while !(k / 64).is_multiple_of(parts) {
        parts -= 1;
    }
    parts
}

fn phase_parts(k: usize, n: usize, phase: Phase) -> usize {
    // Narrow output planes still need K parallelism, including tiny cached
    // prompt suffixes. Wide prefill avoids a redundant split workspace.
    if phase != Phase::Decode
        && (n >= 8192 || (n >= 5120 && (k <= 6144 || phase == Phase::Prefill)))
    {
        1
    } else {
        partitions(k, n)
    }
}

fn tile_rows(rows: usize) -> usize {
    if rows <= 8 {
        8
    } else if rows <= 16 {
        16
    } else if rows <= 24 {
        24
    } else {
        32
    }
}

/// Prefill and decode have different parallelism needs. Use request roles,
/// never the number of co-scheduled rows, to choose the reduction tree.
fn spans<'a>(
    cmd: &'a Commands<'_>,
    rows: usize,
    phase_invariant: bool,
    fallback: &'a [(usize, usize, Phase); 1],
) -> &'a [(usize, usize, Phase)] {
    let Some(spans) = cmd.packed_spans() else {
        return fallback;
    };
    let mut end = 0;
    for &(start, count, _) in spans {
        assert!(start == end && count > 0);
        end += count;
    }
    if end == rows {
        spans
    } else {
        // The vocabulary head receives a compacted selection of plan rows.
        // Its full-K arithmetic is identical for both request roles.
        assert!(
            phase_invariant,
            "packed projection row roles do not match input"
        );
        fallback
    }
}

pub(crate) fn gate_up(
    cmd: &Commands<'_>,
    gate: &Weight,
    up: &Weight,
    input: &Buffer,
    out: &Buffer,
    rows: usize,
    scratch: &Buffer,
) {
    assert!(gate.ty == PACKED4 && up.ty == PACKED4 && gate.k == up.k && gate.n == up.n);
    let (k, n) = (gate.k, gate.n);
    let fallback = [(0, rows, Phase::Decode)];
    for &(offset, rows, prefill) in spans(cmd, rows, partitions(k, n) == 1, &fallback) {
        let tile = tile_rows(rows);
        let padded = rows.div_ceil(tile) * tile;
        let parts = phase_parts(k, n, prefill);
        // One full tile plus a small tail loses occupancy when dispatched
        // separately. Keep that two-tile wave together on the original path.
        let compact = (rows == 32 || rows >= 64) && parts == 1 && n.is_multiple_of(128);
        assert!(scratch.len() >= padded * (k * 2 + k / 64 * 4) + 2 * parts * rows * n * 4);
        cmd.dispatch(
            "splash_input",
            &[input, scratch],
            &[k as u32, rows as u32, padded as u32, offset as u32],
            [padded * (k / 64), 1, 1],
            32,
        );
        let p = [
            k as u32,
            n as u32,
            rows as u32,
            parts as u32,
            padded as u32,
            offset as u32,
        ];
        cmd.dispatch(
            if compact {
                "splash_pair32_bf16"
            } else if tile == 8 {
                "splash_pair8"
            } else if tile == 16 {
                "splash_pair16"
            } else if tile == 24 {
                "splash_pair24"
            } else if tile == 32 {
                "splash_pair32"
            } else {
                "splash_pair64"
            },
            &[&gate.buffer, &up.buffer, scratch, out],
            &p,
            [
                n.div_ceil(128),
                if compact {
                    rows / 32
                } else {
                    rows.div_ceil(tile)
                },
                parts * 2,
            ],
            if (16..=32).contains(&tile) { 128 } else { 256 },
        );
        if compact && !rows.is_multiple_of(32) {
            cmd.dispatch(
                "splash_pair32_bf16_tail",
                &[&gate.buffer, &up.buffer, scratch, out],
                &p,
                [n.div_ceil(128), 1, 2],
                128,
            );
        }
        cmd.dispatch(
            if compact {
                "splash_gateup_bf16"
            } else {
                "splash_gateup_reduce"
            },
            &[scratch, out],
            &p,
            [(rows * n).div_ceil(256), 1, 1],
            256,
        );
    }
}

pub(crate) fn project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    scratch: &Buffer,
) {
    assert!(
        !planes.is_empty(),
        "packed projection requires an output plane"
    );
    let k = planes[0].0.k;
    assert!(
        !planes.is_empty() && rows > 0 && planes.iter().all(|(w, _)| w.ty == PACKED4 && w.k == k)
    );
    let fallback = [(0, rows, Phase::Decode)];
    for &(offset, rows, prefill) in spans(
        cmd,
        rows,
        planes.iter().all(|(w, _)| partitions(k, w.n) == 1),
        &fallback,
    ) {
        let tile = tile_rows(rows);
        let padded_rows = rows.div_ceil(tile) * tile;
        assert!(scratch.len() >= padded_rows * k * 2 + padded_rows * (k / 64) * 4);
        cmd.dispatch(
            "splash_input",
            &[input, scratch],
            &[k as u32, rows as u32, padded_rows as u32, offset as u32],
            [padded_rows * (k / 64), 1, 1],
            32,
        );
        for &(w, out) in planes {
            // Fill narrow projections without multiplying the bandwidth of the
            // vocabulary head. Whole quant groups preserve affine epilogues.
            let parts = phase_parts(k, w.n, prefill);
            let tile = if w.n <= 256 && rows <= 32 { 8 } else { tile };
            let compact =
                tile == 32 && (rows == 32 || rows >= 64) && parts == 1 && w.n.is_multiple_of(128);
            assert!(
                scratch.len()
                    >= padded_rows * (k * 2 + k / 64 * 4)
                        + if parts > 1 {
                            parts * rows * w.n * 4
                        } else if compact {
                            rows * w.n * 2
                        } else {
                            0
                        }
            );
            let p = [
                k as u32,
                w.n as u32,
                rows as u32,
                parts as u32,
                padded_rows as u32,
                offset as u32,
            ];
            if w.n <= 256 && rows <= 8 {
                cmd.dispatch(
                    "splash_affine_vector",
                    &[&w.buffer, scratch, out],
                    &p,
                    [w.n.div_ceil(4), rows.div_ceil(2), parts],
                    256,
                );
            } else {
                cmd.dispatch(
                    if compact {
                        "splash_affine32_bf16"
                    } else if tile == 8 {
                        if k <= 6144 && w.n < 65536 {
                            "splash_affine8_compactpipe"
                        } else {
                            "splash_affine8_compact"
                        }
                    } else if tile == 16 {
                        "splash_affine16_compact4"
                    } else if tile == 24 {
                        "splash_affine24_compact4"
                    } else if tile == 32 {
                        "splash_affine32_compact4"
                    } else {
                        "splash_affine64_compact"
                    },
                    &[&w.buffer, scratch, out],
                    &p,
                    [
                        w.n.div_ceil(128),
                        if compact {
                            rows / 32
                        } else {
                            rows.div_ceil(tile)
                        },
                        parts,
                    ],
                    if (16..=32).contains(&tile) { 128 } else { 256 },
                );
            }
            if compact && !rows.is_multiple_of(32) {
                cmd.dispatch(
                    "splash_affine32_bf16_tail",
                    &[&w.buffer, scratch, out],
                    &p,
                    [w.n.div_ceil(128), 1, 1],
                    128,
                );
            }
            if parts > 1 || compact {
                cmd.dispatch(
                    if compact {
                        "splash_widen"
                    } else {
                        "splash_reduce"
                    },
                    &[scratch, out],
                    &p,
                    [(rows * w.n).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    #[test]
    fn prefix_projection_policy_is_position_stable() {
        assert_eq!(Phase::for_row(true, 127), Phase::Prefix);
        assert_eq!(Phase::for_row(true, 128), Phase::Prefill);
        for position in [0, 127, 128, 32767] {
            assert_eq!(Phase::for_row(false, position), Phase::Decode);
        }
        assert_eq!(phase_parts(17408, 5120, Phase::Prefix), 8);
        assert_eq!(phase_parts(17408, 5120, Phase::Prefill), 1);
        assert_eq!(phase_parts(17408, 5120, Phase::Decode), 8);
    }
    #[test]
    #[ignore = "real-weight isolated GPU sweep; not a serving benchmark"]
    fn packed_real_projection_shape_sweep() {
        let root = std::env::var("PADDOCK_SPLASH_MODEL").expect("verified packed package required");
        let source = paddock_models::splash::Target::open(std::path::Path::new(&root)).unwrap();
        let d = MetalDevice::new(Some(2 << 30)).unwrap();
        for name in [
            "blk.0.ssm_alpha.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "output.weight",
        ] {
            let tensor = source.tensor(name).unwrap();
            let (k, n) = (tensor.dims()[0], tensor.dims()[1]);
            let mut bytes = vec![0; tensor.tiled_bytes()];
            tensor.copy_tiled(&mut bytes).unwrap();
            let weight = d.upload(&bytes).unwrap();
            let max_rows = if n > 17408 { 32 } else { 2048 };
            let x = d
                .upload(
                    &(0..max_rows * k)
                        .flat_map(|i| {
                            half::bf16::from_f32(((i * 11 + i / 19) % 127) as f32 / 64. - 1.)
                                .to_f32()
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let scratch = d.alloc(workspace_bytes(max_rows, k, n.min(17408))).unwrap();
            let out = d.alloc(max_rows * n * 4).unwrap();
            for rows in [8usize, 16, 24, 32, 512, 2048]
                .into_iter()
                .filter(|r| *r <= max_rows)
            {
                for parts in [1usize, 2, 4, 8, 16].into_iter().filter(|p| {
                    (k / 64).is_multiple_of(*p)
                        && (*p == 1 || *p * rows * n * 4 + rows * k * 3 <= scratch.len())
                }) {
                    let mut baseline = Vec::new();
                    let kernels = [
                        (8, 128, 8, "splash_affine8_compact"),
                        (8, 128, 8, "splash_affine8_compactpipe"),
                        (16, 128, 4, "splash_affine16_compact4"),
                        (24, 128, 4, "splash_affine24_compact4"),
                        (32, 128, 4, "splash_affine32_compact4"),
                        (32, 128, 4, "splash_affine32_bf16"),
                        (32, 128, 8, "splash_affine32_compact"),
                        (64, 128, 8, "splash_affine64_compact"),
                    ];
                    for (tile, bn, sg, kernel) in kernels.into_iter().filter(|(t, _, _, kernel)| {
                        *t <= rows
                            && (*t != 24 || rows <= 24)
                            && (!kernel.starts_with("splash_affine32_bf16")
                                || (parts == 1 && n.is_multiple_of(128) && rows.is_multiple_of(32)))
                    }) {
                        let padded = rows.div_ceil(tile) * tile;
                        let cmd = d.begin().unwrap();
                        cmd.dispatch(
                            "splash_input",
                            &[&x, &scratch],
                            &[k as u32, rows as u32, padded as u32, 0],
                            [padded * k / 64, 1, 1],
                            32,
                        );
                        cmd.finish().unwrap();
                        let p = [
                            k as u32,
                            n as u32,
                            rows as u32,
                            parts as u32,
                            padded as u32,
                            0,
                        ];
                        let cmd = d.begin().unwrap();
                        for _ in 0..8 {
                            cmd.dispatch(
                                kernel,
                                &[&weight, &scratch, &out],
                                &p,
                                [n.div_ceil(bn), rows.div_ceil(tile), parts],
                                sg * 32,
                            );
                            if parts > 1 || kernel.starts_with("splash_affine32_bf16") {
                                cmd.dispatch(
                                    if parts > 1 {
                                        "splash_reduce"
                                    } else {
                                        "splash_widen"
                                    },
                                    &[&scratch, &out],
                                    &p,
                                    [(rows * n).div_ceil(256), 1, 1],
                                    256,
                                );
                            }
                        }
                        let seconds = cmd.finish().unwrap();
                        let got = unsafe { out.read_f32(0, rows * n) };
                        if baseline.is_empty() {
                            baseline = got;
                        } else {
                            assert!(
                                baseline == got,
                                "{name}/{rows}/{parts}/{kernel} changed arithmetic: {:?}",
                                baseline
                                    .iter()
                                    .zip(&got)
                                    .enumerate()
                                    .filter(|(_, (a, b))| a != b)
                                    .take(5)
                                    .collect::<Vec<_>>()
                            );
                        }
                        println!(
                            "real k={k} n={n} m={rows} parts={parts} kernel={kernel} us={:.1}",
                            seconds * 1e6 / 8.
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn splash_paged_attention_matches_softmax_reference() {
        attention_oracle(false, false);
    }
    #[test]
    #[ignore = "isolated long-context attention sweep; not a serving benchmark"]
    fn splash_attention_decode_shape_sweep() {
        attention_oracle(true, false);
    }
    #[test]
    #[ignore = "isolated prefill attention sweep; not a serving benchmark"]
    fn splash_attention_prefill_shape_sweep() {
        attention_oracle(true, true);
    }
    fn attention_oracle(sweep: bool, prefill: bool) {
        let d = MetalDevice::new(Some(512 << 20)).unwrap();
        let (heads, kv_heads, rows) = (
            24usize,
            4usize,
            if prefill {
                256usize
            } else if sweep {
                8usize
            } else {
                37usize
            },
        );
        let upload_u = |v: &[u32]| {
            d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let mut layouts = std::collections::HashMap::new();
        let prefixes = if sweep {
            vec![2049usize, 8193, 32769]
        } else {
            vec![3usize, 33, 997, 2049]
        };
        for (prefix, kernel, reverse) in prefixes
            .into_iter()
            .flat_map(|p| {
                [
                    (p, "splash_attention_prefill", false),
                    (p, "splash_attention_prefill", true),
                    (p, "splash_attention_prefill64", false),
                    (p, "splash_attention_prefill64", true),
                    (p, "splash_attention_prefill_grouped", false),
                    (p, "splash_attention_prefill_grouped", true),
                    (p, "splash_attention_decode_grouped", false),
                    (p, "splash_attention_decode_grouped", true),
                ]
            })
            .filter(|(_, kernel, reverse)| {
                !sweep
                    || (!reverse
                        && if prefill {
                            kernel.contains("prefill_grouped")
                        } else {
                            kernel.contains("decode")
                        })
            })
        {
            let stride = (prefix + rows).div_ceil(16);
            let mut page_ids: Vec<u32> = (0..stride as u32).collect();
            if reverse {
                page_ids.reverse();
            }
            let physical_row = |t: usize| page_ids[t / 16] as usize * 16 + t % 16;
            let mut values = vec![0.; stride * 16 * kv_heads * 256];
            for t in 0..stride * 16 {
                for j in 0..kv_heads * 256 {
                    let i = t * kv_heads * 256 + j;
                    values[physical_row(t) * kv_heads * 256 + j] =
                        half::bf16::from_f32(((i * 7 + i / 13) % 71) as f32 / 32. - 1.).to_f32();
                }
            }
            let mut keys: Vec<f32> = values.iter().map(|v| *v / 2.).collect();
            // Unused rows of the final physical page must never contaminate
            // attention, even if recycled storage contains a nonfinite value.
            for t in prefix + rows..stride * 16 {
                let start = physical_row(t) * kv_heads * 256;
                keys[start..start + kv_heads * 256].fill(f32::NAN);
                values[start..start + kv_heads * 256].fill(f32::NAN);
            }
            let q: Vec<f32> = (0..rows * heads * 256)
                .map(|i| {
                    half::bf16::from_f32(((i * 11 + i / 257) % 19) as f32 / 32. - 0.25).to_f32()
                })
                .collect();
            let bf = |v: &[f32]| {
                d.upload(
                    &v.iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let query = bf(&q);
            let decode = kernel.contains("decode");
            let grouped = kernel.contains("grouped");
            let splits = if decode { DECODE_PARTS } else { 4 };
            let tile_rows = if decode { 8 } else { 32 };
            let tile_count = rows.div_ceil(tile_rows);
            let query_f32 = d
                .upload(&q.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let packed = d.alloc(tile_count * tile_rows * heads * 256 * 2).unwrap();
            let k = bf(&keys);
            let v = bf(&values);
            let meta = upload_u(
                &(0..rows)
                    .flat_map(|r| [0, (prefix + r) as u32])
                    .collect::<Vec<_>>(),
            );
            let pages = upload_u(&page_ids);
            let limits = upload_u(&(0..rows).map(|r| (prefix + r) as u32).collect::<Vec<_>>());
            let mut tile_data: Vec<_> = (0..rows)
                .step_by(tile_rows)
                .map(|r| [r as u32, (rows - r).min(tile_rows) as u32])
                .collect();
            if decode {
                // Decode partials are compacted by tile, not by source row.
                // Deliberately decouple those indices to test the gather/join.
                tile_data.reverse();
            }
            let tiles = upload_u(&tile_data.into_iter().flatten().collect::<Vec<_>>());
            let parts = d
                .alloc(tile_count * tile_rows * heads * splits * 258 * 4)
                .unwrap();
            let staging = d
                .alloc(
                    (tile_count * heads * 4 * 64 * 256 * 2)
                        .max(tile_count * kv_heads * splits * 64 * 256 * 2),
                )
                .unwrap();
            let out = d.alloc(rows * heads * 256 * 4).unwrap();
            let cmd = d.begin().unwrap();
            if grouped {
                cmd.dispatch(
                    if decode {
                        "splash_attention_query_decode"
                    } else {
                        "splash_attention_query_grouped"
                    },
                    &[&query_f32, &packed, &tiles],
                    &[tile_count as u32],
                    [tile_count * tile_rows * heads, 1, 1],
                    256,
                );
            }
            let buffers = [
                if grouped { &packed } else { &query },
                &k,
                &v,
                &meta,
                &pages,
                &parts,
                &tiles,
                &limits,
                &staging,
            ];
            for _ in 0..if sweep { 8 } else { 1 } {
                cmd.dispatch(
                    kernel,
                    &buffers[..if kernel != "splash_attention_prefill" {
                        9
                    } else {
                        8
                    }],
                    &[
                        heads as u32,
                        kv_heads as u32,
                        stride as u32,
                        0.0625f32.to_bits(),
                    ],
                    if decode {
                        [kv_heads, tile_count, splits]
                    } else if grouped {
                        [kv_heads, tile_count * 4, 4]
                    } else {
                        [heads, tile_count, 4]
                    },
                    if kernel != "splash_attention_prefill" {
                        256
                    } else {
                        128
                    },
                );
                cmd.dispatch(
                    if decode {
                        "splash_attention_decode_join"
                    } else {
                        "qwen_attention_prefill_join"
                    },
                    &[&parts, &out, &tiles],
                    &[heads as u32],
                    [heads, tile_count, tile_rows],
                    32,
                );
            }
            let seconds = cmd.finish().unwrap();
            if sweep {
                println!(
                    "attention kernel={kernel} prefix={prefix} rows={rows} parts={splits} us={:.1}",
                    seconds * 1e6 / 8.
                );
            }
            let got = unsafe { out.read_f32(0, rows * heads * 256) };
            if reverse {
                assert_eq!(
                    &got,
                    layouts.get(&(prefix, kernel)).unwrap(),
                    "page topology changed attention"
                );
            } else {
                layouts.insert((prefix, kernel), got.clone());
            }
            // Independent scalar softmax oracle for the operator, including
            // ragged/causal rows and a deliberately reversed physical page map.
            for row in [0, 1, 31, 32, rows - 1].into_iter().filter(|r| *r < rows) {
                for head in [0, 5, 6, 23] {
                    let key_index = |t: usize| physical_row(t) * kv_heads * 256 + head / 6 * 256;
                    let scores: Vec<f64> = (0..=prefix + row)
                        .map(|t| {
                            (0..256)
                                .map(|j| {
                                    f64::from(q[(row * heads + head) * 256 + j])
                                        * f64::from(keys[key_index(t) + j])
                                })
                                .sum::<f64>()
                                / 16.
                        })
                        .collect();
                    let hi = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probs: Vec<f64> = scores.iter().map(|s| (s - hi).exp()).collect();
                    let denom: f64 = probs.iter().sum();
                    for channel in 0..256 {
                        let expected = probs
                            .iter()
                            .enumerate()
                            .map(|(t, p)| p * f64::from(values[key_index(t) + channel]))
                            .sum::<f64>()
                            / denom;
                        let value = f64::from(got[(row * heads + head) * 256 + channel]);
                        assert!(
                            (value - expected).abs() < 0.003,
                            "prefix {prefix} row {row} head {head} channel {channel}: {value} != {expected}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn packed_mpp_rows_ragged_columns_and_group_epilogues() {
        let d = MetalDevice::new(Some(256 << 20)).unwrap();
        for (k, n) in [
            (64usize, 48usize),
            (128, 48),
            (512, 256),
            (1024, 384),
            (128, 8192),
            (128, 8193),
        ] {
            let padded = n.div_ceil(256) * 256;
            let groups = k / 64;
            let mut bytes = vec![0u8; k * padded / 16 * 9];
            let code = |r: usize, g: usize, j: usize| ((r * 3 + g * 7 + j * 5) % 16) as u8;
            let scale =
                |r: usize, g: usize| half::bf16::from_f32((1 + (r + g) % 4) as f32 / 64.).to_f32();
            let bias =
                |r: usize, g: usize| half::bf16::from_f32(-(((r + g) % 5) as f32) / 16.).to_f32();
            for r in 0..n {
                for g in 0..groups {
                    let index = (r / 256 * groups + g) * 256 + r % 256;
                    for j in 0..32 {
                        bytes[index * 32 + j] = code(r, g, j * 2) | (code(r, g, j * 2 + 1) << 4);
                    }
                    for (base, value) in [
                        (k * padded / 2, scale(r, g)),
                        (k * padded / 2 + k * padded / 32, bias(r, g)),
                    ] {
                        bytes[base + index * 2..base + index * 2 + 2]
                            .copy_from_slice(&half::bf16::from_f32(value).to_le_bytes());
                    }
                }
            }
            let w = Weight {
                buffer: d.upload(&bytes).unwrap(),
                ty: PACKED4,
                k,
                n,
            };
            let input: Vec<f32> = (0..128 * k)
                .map(|i| half::bf16::from_f32((i % 19) as f32 / 19. - 0.4).to_f32())
                .collect();
            let x = d
                .upload(
                    &input
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let scratch = d.alloc(workspace_bytes(128, k, n)).unwrap();
            let out = d.alloc(128 * n * 4).unwrap();
            let up = d.alloc(128 * n * 4).unwrap();
            let fused = d.alloc(128 * n * 4).unwrap();
            let mut reference = Vec::new();
            for rows in [1, 2, 3, 7, 8, 9, 16, 24, 32, 33, 63, 64, 65, 127, 128] {
                let cmd = d.begin().unwrap();
                project(&cmd, &[(&w, &out)], &x, rows, &scratch);
                cmd.finish().unwrap();
                let got = unsafe { out.read_f32(0, rows * n) };
                if rows == 1 {
                    reference = got.clone();
                }
                assert_eq!(&reference, &got[..n], "row count changed first row");
                for r in 0..rows {
                    for col in 0..n {
                        let mut total = 0.;
                        for g in 0..groups {
                            let mut dot = 0.;
                            let mut sum = 0.;
                            for j in 0..64 {
                                let v = input[r * k + g * 64 + j];
                                dot += v * code(col, g, j) as f32;
                                sum += v;
                            }
                            total += dot * scale(col, g) + sum * bias(col, g);
                        }
                        assert!(
                            (got[r * n + col] - total).abs() <= 0.008 * total.abs().max(1.),
                            "{k}/{n}/{rows}: {} vs {total}",
                            got[r * n + col]
                        );
                    }
                }
                let cmd = d.begin().unwrap();
                project(&cmd, &[(&w, &up)], &x, rows, &scratch);
                cmd.dispatch(
                    "mlx_swiglu",
                    &[&out, &up],
                    &[(rows * n) as u32],
                    [(rows * n).div_ceil(256), 1, 1],
                    256,
                );
                gate_up(&cmd, &w, &w, &x, &fused, rows, &scratch);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, rows * n) },
                    unsafe { fused.read_f32(0, rows * n) },
                    "fused activation changed k={k} n={n} rows={rows}"
                );
            }
            let prefill = [(0, 128, Phase::Prefill)];
            let cmd = d.begin().unwrap().with_packed_spans(&prefill);
            project(&cmd, &[(&w, &out)], &x, 128, &scratch);
            gate_up(&cmd, &w, &w, &x, &fused, 128, &scratch);
            cmd.finish().unwrap();
            let pre = unsafe { out.read_f32(0, 128 * n) };
            let pre_gate = unsafe { fused.read_f32(0, 128 * n) };
            // Short cached suffixes and ragged chunks must use the same
            // full-K arithmetic as the corresponding uninterrupted prefill.
            for chunks in [
                &[
                    (0, 1, Phase::Prefill),
                    (1, 7, Phase::Prefill),
                    (8, 25, Phase::Prefill),
                    (33, 95, Phase::Prefill),
                ][..],
                &[(0, 33, Phase::Prefill), (33, 95, Phase::Prefill)][..],
            ] {
                let cmd = d.begin().unwrap().with_packed_spans(chunks);
                project(&cmd, &[(&w, &out)], &x, 128, &scratch);
                gate_up(&cmd, &w, &w, &x, &fused, 128, &scratch);
                cmd.finish().unwrap();
                assert_eq!(
                    pre,
                    unsafe { out.read_f32(0, 128 * n) },
                    "prefill chunk arithmetic changed"
                );
                assert_eq!(
                    pre_gate,
                    unsafe { fused.read_f32(0, 128 * n) },
                    "prefill gate/up chunk arithmetic changed"
                );
            }
            let cmd = d.begin().unwrap();
            project(&cmd, &[(&w, &out)], &x, 128, &scratch);
            gate_up(&cmd, &w, &w, &x, &fused, 128, &scratch);
            cmd.finish().unwrap();
            let decode = unsafe { out.read_f32(0, 128 * n) };
            let decode_gate = unsafe { fused.read_f32(0, 128 * n) };
            let mixed = [
                (0, 3, Phase::Decode),
                (3, 62, Phase::Prefill),
                (65, 7, Phase::Decode),
                (72, 56, Phase::Prefill),
            ];
            let cmd = d.begin().unwrap().with_packed_spans(&mixed);
            project(&cmd, &[(&w, &out)], &x, 128, &scratch);
            gate_up(&cmd, &w, &w, &x, &fused, 128, &scratch);
            cmd.finish().unwrap();
            let got = unsafe { out.read_f32(0, 128 * n) };
            let got_gate = unsafe { fused.read_f32(0, 128 * n) };
            for (start, count, prefill) in mixed {
                let range = start * n..(start + count) * n;
                assert_eq!(
                    got[range.clone()],
                    if prefill == Phase::Prefill {
                        &pre
                    } else {
                        &decode
                    }[range.clone()]
                );
                assert_eq!(
                    got_gate[range.clone()],
                    if prefill == Phase::Prefill {
                        &pre_gate
                    } else {
                        &decode_gate
                    }[range]
                );
            }
        }
    }
}
