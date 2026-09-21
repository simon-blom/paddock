//! Gates for the Bonsai lane: PrismML's ternary packings (PTQ1_0, PQ2_0) on
//! the i-quant streams, and the blockwise Walsh-Hadamard rotation their
//! checkpoints ask of a runtime.
//!
//! Every plane here is synthetic - built with the test reference's encoder -
//! so the gates need no model file. Both checks that can be exact are: a trit
//! times an f16 scale is exact in f32, and the rotation is one fixed tree of
//! additions per output, so the pack has to agree with the reference bit for
//! bit. Only the int8-activation lanes carry a tolerance, and that one is the
//! activation quantizer's, not the weights'.

mod common;

use paddock_engine::gpu::{GpuExecutor, HadamardGdnHeads};
use paddock_kernels::reference::hadamard::{GdnHeads, rotate_rows, unrotate_rows};
use paddock_kernels::reference::ternary::{
    PQ2_0, PTQ1_0, dequant_ternary, encode_pq2_0, encode_ptq1_0, ternary_block_bytes,
};
use paddock_models::ggml_type::GgmlType;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 33) as u32
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| (lcg(&mut s) as f32 / (1u32 << 31) as f32) - 0.5)
        .collect()
}

/// A raw `[in_dim, out_dim]` plane of one packing: random trits, random
/// positive f16 scales in the range the real file uses.
fn plane(raw: u32, in_dim: usize, out_dim: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut bytes = Vec::with_capacity(in_dim * out_dim / 128 * 34);
    for _ in 0..in_dim * out_dim / 128 {
        let w: Vec<i8> = (0..128).map(|_| (lcg(&mut s) % 3) as i8 - 1).collect();
        let d = half::f16::from_f32(0.004 + (lcg(&mut s) % 1000) as f32 * 3.0e-5);
        if raw == PTQ1_0 {
            bytes.extend_from_slice(&encode_ptq1_0(&w, d));
        } else {
            bytes.extend_from_slice(&encode_pq2_0(&w, d));
        }
    }
    bytes
}

fn ty_of(raw: u32) -> GgmlType {
    GgmlType::from_raw(raw)
}

fn ternary_exec() -> Option<GpuExecutor> {
    let exec = common::gpu()?;
    if !exec.has_kquant_ternary() {
        eprintln!("pack lacks the ternary lanes (slot 625) - skipping");
        return None;
    }
    Some(exec)
}

#[test]
fn ternary_repack_and_dequant_bitmatch_the_reference() {
    let Some(exec) = ternary_exec() else {
        return;
    };
    for raw in [PTQ1_0, PQ2_0] {
        let (in_dim, out_dim) = (5120usize, 96usize);
        let bytes = plane(raw, in_dim, out_dim, 0xB0_45A1 + raw as u64);
        assert_eq!(
            bytes.len(),
            in_dim * out_dim / 128 * ternary_block_bytes(raw).expect("ternary")
        );
        let mut cpu = vec![0f32; in_dim * out_dim];
        dequant_ternary(raw, &bytes, &mut cpu).expect("cpu dequant");

        let w = exec
            .repack_kquant_raw(&bytes, vec![in_dim, out_dim], ty_of(raw), "synthetic")
            .expect("repack");
        let mut out = exec.alloc(in_dim * out_dim).expect("alloc");
        exec.kquant_dequant_rp(&w, &mut out).expect("dequant");
        let gpu = exec.to_host(&out).expect("dtoh");
        let mism = gpu
            .iter()
            .zip(&cpu)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "{:?}: {} weights, {mism} bit mismatches",
            ty_of(raw),
            cpu.len()
        );
        assert_eq!(
            mism,
            0,
            "{:?}: GPU dequant differs from the reference",
            ty_of(raw)
        );
    }
}

#[test]
fn ternary_decode_lanes_match_the_reference_dot() {
    let Some(exec) = ternary_exec() else {
        return;
    };
    // the three input widths of Bonsai 2 27B
    for (in_dim, out_dim) in [(5120usize, 320usize), (6144, 192), (17408, 96)] {
        for raw in [PTQ1_0, PQ2_0] {
            let bytes = plane(raw, in_dim, out_dim, 0x7E_57 + in_dim as u64 + raw as u64);
            let mut wf = vec![0f32; in_dim * out_dim];
            dequant_ternary(raw, &bytes, &mut wf).expect("cpu dequant");
            let x = noise(in_dim, 0x51 + in_dim as u64);
            let want: Vec<f64> = (0..out_dim)
                .map(|o| {
                    wf[o * in_dim..(o + 1) * in_dim]
                        .iter()
                        .zip(&x)
                        .map(|(w, v)| *w as f64 * *v as f64)
                        .sum()
                })
                .collect();
            let norm = want.iter().map(|v| v * v).sum::<f64>().sqrt();

            let w = exec
                .repack_kquant_raw(&bytes, vec![in_dim, out_dim], ty_of(raw), "synthetic")
                .expect("repack");
            let dx = exec.to_device(&x).expect("htod");

            // exact class: f32 activations
            let mut y = exec.alloc(out_dim).expect("alloc");
            exec.kquant_gemv(&w, &dx, &mut y).expect("gemv f32");
            let got = exec.to_host(&y).expect("dtoh");
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (*a as f64 - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / norm;
            eprintln!(
                "{:?} {in_dim}x{out_dim} f32 lane: rel err {err:.2e}",
                ty_of(raw)
            );
            assert!(err < 1e-5, "{:?} f32 decode lane off by {err}", ty_of(raw));

            // serving class: int8 activations
            let mut xq = exec.alloc_i8(in_dim).expect("alloc");
            let mut xs = exec.alloc(in_dim / 32).expect("alloc");
            let mut sums = exec.alloc(in_dim / 16).expect("alloc");
            exec.quantize_q8_sums(&dx, &mut xq, &mut xs, &mut sums, in_dim)
                .expect("quantize");
            let mut y8 = exec.alloc(out_dim).expect("alloc");
            exec.kquant_gemv_w4a8(&w, &xq, &xs, None, &mut y8)
                .expect("gemv int8");
            let got8 = exec.to_host(&y8).expect("dtoh");
            let err8 = got8
                .iter()
                .zip(&want)
                .map(|(a, b)| (*a as f64 - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / norm;
            eprintln!(
                "{:?} {in_dim}x{out_dim} int8 lane: rel err {err8:.2e}",
                ty_of(raw)
            );
            assert!(
                err8 < 2e-2,
                "{:?} int8 decode lane off by {err8}",
                ty_of(raw)
            );
        }
    }
}

/// Every BATCH rung the engine can route a ternary plane to, against the
/// reference dot: the multi-column GEMV (2..5 rows), the dp4a walk, the
/// K-split mma (up to 64 rows) and the >64-row W4A8 tile - plus the embedding
/// gather, which has to be exact. The decode gate above covers one row only,
/// and a rung nobody ran is a rung nobody knows (the scratch-plane lesson).
#[test]
fn ternary_batch_rungs_match_the_reference_dot() {
    let Some(exec) = ternary_exec() else {
        return;
    };
    for raw in [PTQ1_0, PQ2_0] {
        for (in_dim, out_dim) in [(5120usize, 256usize), (17408, 128)] {
            let bytes = plane(raw, in_dim, out_dim, 0xBA7C + in_dim as u64 + raw as u64);
            let mut wf = vec![0f32; in_dim * out_dim];
            dequant_ternary(raw, &bytes, &mut wf).expect("cpu dequant");
            let w = exec
                .repack_kquant_raw(&bytes, vec![in_dim, out_dim], ty_of(raw), "synthetic")
                .expect("repack");

            for rows in [3usize, 7, 33, 130] {
                let x = noise(rows * in_dim, 0xC0DE + (rows * in_dim) as u64);
                let want: Vec<f64> = (0..rows * out_dim)
                    .map(|i| {
                        let (r, o) = (i / out_dim, i % out_dim);
                        wf[o * in_dim..(o + 1) * in_dim]
                            .iter()
                            .zip(&x[r * in_dim..(r + 1) * in_dim])
                            .map(|(w, v)| *w as f64 * *v as f64)
                            .sum()
                    })
                    .collect();
                let norm = want.iter().map(|v| v * v).sum::<f64>().sqrt();
                let rel = |got: &[f32]| {
                    got.iter()
                        .zip(&want)
                        .map(|(a, b)| (*a as f64 - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / norm
                };
                let dx = exec.to_device(&x).expect("htod");
                let mut xq = exec.alloc_i8(rows * in_dim).expect("alloc");
                let mut xs = exec.alloc(rows * in_dim / 32).expect("alloc");
                exec.quantize_q8(&dx, &mut xq, &mut xs, rows * in_dim)
                    .expect("quantize");
                let check = |lane: &str, got: Vec<f32>| {
                    let err = rel(&got);
                    eprintln!(
                        "{:?} {in_dim}x{out_dim} r={rows} {lane}: rel err {err:.2e}",
                        ty_of(raw)
                    );
                    assert!(err < 2e-2, "{:?} {lane} r={rows} off by {err}", ty_of(raw));
                };

                let mut y = exec.alloc(rows * out_dim).expect("alloc");
                exec.kquant_gemm_dp4a(&w, &xq, &xs, None, &mut y, rows)
                    .expect("dp4a");
                check("dp4a", exec.to_host(&y).expect("dtoh"));

                if exec.has_kquant_gemv_w4a8_nc() && GpuExecutor::kquant_gemv_w4a8_nc_fits(&w, rows)
                {
                    let mut y = exec.alloc(rows * out_dim).expect("alloc");
                    exec.kquant_gemv_w4a8_nc(&w, &xq, &xs, None, &mut y, rows)
                        .expect("nc gemv");
                    check("multi-column gemv", exec.to_host(&y).expect("dtoh"));
                }
                if rows <= 64 && exec.has_kquant_mma_ks() {
                    let mut part = exec.alloc(8 * 64 * out_dim).expect("alloc");
                    let mut y = exec.alloc(rows * out_dim).expect("alloc");
                    exec.kquant_gemm_mma_ks(&w, &xq, &xs, None, &mut part, &mut y, rows)
                        .expect("mma ks");
                    check("K-split mma", exec.to_host(&y).expect("dtoh"));
                }
                if rows > 64 && exec.has_kquant_iq_tile() {
                    let mut yq = exec
                        .alloc_u8(in_dim.div_ceil(128) * rows.next_multiple_of(128) * 144)
                        .expect("alloc");
                    exec.quantize_q8_mmq(&dx, &mut yq, in_dim, rows)
                        .expect("quantize mmq");
                    let mut y = exec.alloc(rows * out_dim).expect("alloc");
                    if exec.has_kquant_gemm_w4a8_pipe2() {
                        exec.kquant_gemm_w4a8_pipe2(&w, &yq, None, &mut y, rows)
                            .expect("tile pipe2");
                        check("W4A8 tile (pipe2)", exec.to_host(&y).expect("dtoh"));
                    }
                    if exec.has_kquant_gemm_w4a8_pipe() {
                        let mut y = exec.alloc(rows * out_dim).expect("alloc");
                        exec.kquant_gemm_w4a8_pipe(&w, &yq, None, &mut y, rows)
                            .expect("tile pipe");
                        check("W4A8 tile (pipe)", exec.to_host(&y).expect("dtoh"));
                    }
                    let mut y = exec.alloc(rows * out_dim).expect("alloc");
                    exec.kquant_gemm_w4a8(&w, &yq, None, &mut y, rows)
                        .expect("tile");
                    check("W4A8 tile", exec.to_host(&y).expect("dtoh"));
                }
            }

            // the embedding gather: rows of the table, exact
            let ids: Vec<u32> = vec![0, 1, (out_dim - 1) as u32, (out_dim / 2) as u32, 1];
            let d_ids = exec.to_device_u32(&ids).expect("htod");
            let mut rows_out = exec.alloc(ids.len() * in_dim).expect("alloc");
            exec.kquant_gather(&w, &d_ids, &mut rows_out, in_dim, ids.len())
                .expect("gather");
            let got = exec.to_host(&rows_out).expect("dtoh");
            for (i, &id) in ids.iter().enumerate() {
                let want = &wf[id as usize * in_dim..(id as usize + 1) * in_dim];
                let have = &got[i * in_dim..(i + 1) * in_dim];
                let mism = have
                    .iter()
                    .zip(want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    mism,
                    0,
                    "{:?} gather row {id}: {mism} mismatches",
                    ty_of(raw)
                );
            }
        }
    }
}

#[test]
fn hadamard_rotation_bitmatches_the_reference() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_hadamard() {
        eprintln!("pack lacks the Hadamard rotation (slot 626) - skipping");
        return;
    }
    let block = 1024usize;
    for (width, rows) in [(5120usize, 1usize), (6144, 7), (17408, 3), (5120, 300)] {
        let x = noise(rows * width, 0xAD + width as u64 + rows as u64);
        let signs: Vec<f32> = noise(width, 0x51_67 + width as u64)
            .iter()
            .map(|v| if *v < 0.0 { -1.0 } else { 1.0 })
            .collect();
        let dx = exec.to_device(&x).expect("htod");
        let ds = exec.to_device(&signs).expect("htod");

        let mut dy = exec.alloc(rows * width).expect("alloc");
        exec.hadamard_rotate(&dx, &mut dy, &ds, rows, width, block, None)
            .expect("rotate");
        let got = exec.to_host(&dy).expect("dtoh");
        let want = rotate_rows(&x, width, block, &signs, None);
        let mism = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!("rotate {rows}x{width}: {mism} bit mismatches");
        assert_eq!(
            mism, 0,
            "rotation {rows}x{width} differs from the reference"
        );

        // in place, and the lookup-side inverse on top of it: back to x
        let mut dz = exec.to_device(&x).expect("htod");
        exec.hadamard_rotate_inplace(&mut dz, &ds, rows, width, block, false)
            .expect("rotate in place");
        let inplace = exec.to_host(&dz).expect("dtoh");
        assert!(
            inplace
                .iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "in-place rotation {rows}x{width} differs from the out-of-place one"
        );
        exec.hadamard_rotate_inplace(&mut dz, &ds, rows, width, block, true)
            .expect("inverse");
        let back = exec.to_host(&dz).expect("dtoh");
        let want_back = unrotate_rows(&want, width, block, &signs);
        assert!(
            back.iter()
                .zip(&want_back)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "inverse {rows}x{width} differs from the reference"
        );
        let worst = back
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-5, "H(H(x)) drifted {worst} from x");
    }

    // the gated-delta-net output projection of Qwen3.8-27B: 48 value heads of
    // 128 in 16 key groups, tiled in the engine, grouped in the file
    let (width, rows) = (6144usize, 5usize);
    let heads = GdnHeads {
        head_dim: 128,
        n_k: 16,
        rep: 3,
    };
    let x = noise(rows * width, 0x6D);
    let signs: Vec<f32> = noise(width, 0x6E)
        .iter()
        .map(|v| if *v < 0.0 { -1.0 } else { 1.0 })
        .collect();
    let dx = exec.to_device(&x).expect("htod");
    let ds = exec.to_device(&signs).expect("htod");
    let mut dy = exec.alloc(rows * width).expect("alloc");
    let gdn = HadamardGdnHeads {
        head_dim: heads.head_dim,
        n_k: heads.n_k,
        rep: heads.rep,
    };
    exec.hadamard_rotate(&dx, &mut dy, &ds, rows, width, block, Some(gdn))
        .expect("rotate gdn");
    let got = exec.to_host(&dy).expect("dtoh");
    let want = rotate_rows(&x, width, block, &signs, Some(heads));
    let mism = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mism, 0, "grouped-head rotation differs from the reference");
}

/// The dedicated PTQ1_0 decode lane (slots 627 / 628). Two exact checks and
/// one honest one:
///   - the per-128 quantizer against the same arithmetic on the host, byte
///     for byte;
///   - the GEMV against the INTEGER dot of the GPU's own int8 activations
///     with the reference's trits - that takes the activation quantizer out
///     of the comparison, so what is left is f32 summation order;
///   - and the end-to-end error against the unquantized dot, printed next to
///     the per-32 lane's, because that is the class this lane trades in.
#[test]
fn ternary_b128_lane_matches_the_reference() {
    let Some(exec) = ternary_exec() else {
        return;
    };
    if !exec.has_ternary_gemv_b128() {
        eprintln!("pack lacks the ternary decode lane (slots 627/628) - skipping");
        return;
    }
    for (in_dim, out_dim) in [(5120usize, 328usize), (6144, 192), (17408, 101), (2304, 7)] {
        let bytes = plane(PTQ1_0, in_dim, out_dim, 0xB128 + in_dim as u64);
        let mut wf = vec![0f32; in_dim * out_dim];
        dequant_ternary(PTQ1_0, &bytes, &mut wf).expect("cpu dequant");
        let w = exec
            .repack_kquant_raw(&bytes, vec![in_dim, out_dim], GgmlType::Ptq1_0, "synthetic")
            .expect("repack");
        assert!(exec.ternary_gemv_b128_fits(&w));

        // a row with one loud strip, so the blocks do not all share a scale
        let mut x = noise(in_dim, 0xA5 + in_dim as u64);
        for v in &mut x[256..384] {
            *v *= 40.0;
        }
        let dx = exec.to_device(&x).expect("htod");
        let mut xq = exec.alloc_i8(in_dim).expect("alloc");
        let mut xs = exec.alloc(in_dim / 128).expect("alloc");
        exec.quantize_q8_b128(&dx, &mut xq, &mut xs, in_dim)
            .expect("quantize");
        let q = exec.to_host_i8(&xq).expect("dtoh");
        let s = exec.to_host(&xs).expect("dtoh");
        for (b, blk) in x.as_chunks::<128>().0.iter().enumerate() {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scl = amax * (1.0 / 127.0);
            assert_eq!(s[b].to_bits(), scl.to_bits(), "scale of block {b}");
            let inv = if scl > 0.0 { 1.0 / scl } else { 0.0 };
            for (i, v) in blk.iter().enumerate() {
                let want = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
                assert_eq!(q[b * 128 + i], want, "q[{b}][{i}]");
            }
        }

        let mut y = exec.alloc(out_dim).expect("alloc");
        exec.ternary_gemv_b128(&w, &xq, &xs, &mut y).expect("gemv");
        let got = exec.to_host(&y).expect("dtoh");

        // the integer dot the kernel is supposed to be: sum over blocks of
        // d * xs * sum(trit * q), with d * trit read off the dequantized plane
        let (mut e_int, mut n_int, mut e_f, mut n_f) = (0f64, 0f64, 0f64, 0f64);
        for o in 0..out_dim {
            let row = &wf[o * in_dim..(o + 1) * in_dim];
            let mut want_int = 0f64;
            let mut want_f = 0f64;
            for (b, scale) in s.iter().enumerate() {
                let mut acc = 0f64;
                for i in b * 128..(b + 1) * 128 {
                    acc += row[i] as f64 * q[i] as f64;
                    want_f += row[i] as f64 * x[i] as f64;
                }
                want_int += acc * *scale as f64;
            }
            e_int += (got[o] as f64 - want_int).powi(2);
            n_int += want_int * want_int;
            e_f += (got[o] as f64 - want_f).powi(2);
            n_f += want_f * want_f;
        }
        let (rel_int, rel_f) = ((e_int / n_int).sqrt(), (e_f / n_f).sqrt());

        // the per-32 lane on the same row, for the class comparison
        let mut xq32 = exec.alloc_i8(in_dim).expect("alloc");
        let mut xs32 = exec.alloc(in_dim / 32).expect("alloc");
        let mut sums = exec.alloc(in_dim / 16).expect("alloc");
        exec.quantize_q8_sums(&dx, &mut xq32, &mut xs32, &mut sums, in_dim)
            .expect("quantize");
        let mut y32 = exec.alloc(out_dim).expect("alloc");
        exec.kquant_gemv_w4a8(&w, &xq32, &xs32, None, &mut y32)
            .expect("gemv int8");
        let got32 = exec.to_host(&y32).expect("dtoh");
        let mut e32 = 0f64;
        for o in 0..out_dim {
            let want: f64 = wf[o * in_dim..(o + 1) * in_dim]
                .iter()
                .zip(&x)
                .map(|(w, v)| *w as f64 * *v as f64)
                .sum();
            e32 += (got32[o] as f64 - want).powi(2);
        }
        eprintln!(
            "PTQ1_0 {in_dim}x{out_dim}: vs integer dot {rel_int:.2e}, vs f32 dot {rel_f:.2e} \
             (per-32 lane {:.2e})",
            (e32 / n_f).sqrt()
        );
        assert!(
            rel_int < 2e-6,
            "kernel differs from its own integer dot: {rel_int}"
        );
        assert!(rel_f < 3e-2, "per-128 lane off by {rel_f}");
    }
}

/// Slot 629 against the pair it fuses: the f32 rotated rows, the int8 bytes
/// and the per-128 scales all have to be the standalone ops' to the bit, in
/// place and through the grouped-head permutation.
#[test]
fn fused_rotate_quantize_bitmatches_the_pair() {
    let Some(exec) = ternary_exec() else {
        return;
    };
    if !exec.has_hadamard_q8_b128() || !exec.has_ternary_gemv_b128() {
        eprintln!("pack lacks the fused rotate+quantize (slot 629) - skipping");
        return;
    }
    let gdn = HadamardGdnHeads {
        head_dim: 128,
        n_k: 16,
        rep: 3,
    };
    for (rows, width, heads) in [
        (1usize, 5120usize, None),
        (1, 17408, None),
        (3, 6144, None),
        (1, 6144, Some(gdn)),
        (5, 6144, Some(gdn)),
    ] {
        let mut host = noise(rows * width, 0xF05E + (rows * width) as u64);
        for v in &mut host[width / 2..width / 2 + 40] {
            *v *= 25.0;
        }
        let mut s = 0x51_9Eu64;
        let sg: Vec<f32> = (0..width)
            .map(|_| if lcg(&mut s) & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        let signs = exec.to_device(&sg).expect("htod");

        // the pair
        let x = exec.to_device(&host).expect("htod");
        let mut y = exec.alloc(rows * width).expect("alloc");
        exec.hadamard_rotate(&x, &mut y, &signs, rows, width, 1024, heads)
            .expect("rotate");
        let mut q = exec.alloc_i8(rows * width).expect("alloc");
        let mut sc = exec.alloc(rows * width / 128).expect("alloc");
        exec.quantize_q8_b128(&y, &mut q, &mut sc, rows * width)
            .expect("quantize");
        let (want_y, want_q, want_s) = (
            exec.to_host(&y).expect("dtoh"),
            exec.to_host_i8(&q).expect("dtoh"),
            exec.to_host(&sc).expect("dtoh"),
        );

        // the fused launch: in place without the permutation, into a second
        // plane with it
        let mut fx = exec.to_device(&host).expect("htod");
        let mut fy = exec.alloc(rows * width).expect("alloc");
        let mut fq = exec.alloc_i8(rows * width).expect("alloc");
        let mut fs = exec.alloc(rows * width / 128).expect("alloc");
        let out = heads.is_some().then_some(&mut fy);
        exec.hadamard_rotate_q8_b128(
            &mut fx, out, &signs, &mut fq, &mut fs, rows, width, 1024, heads,
        )
        .expect("fused");
        let got_y = exec
            .to_host(if heads.is_some() { &fy } else { &fx })
            .expect("dtoh");
        let (got_q, got_s) = (
            exec.to_host_i8(&fq).expect("dtoh"),
            exec.to_host(&fs).expect("dtoh"),
        );

        let my = got_y
            .iter()
            .zip(&want_y)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mq = got_q.iter().zip(&want_q).filter(|(a, b)| a != b).count();
        let ms = got_s
            .iter()
            .zip(&want_s)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "fused {rows}x{width}{}: {my} f32 / {mq} int8 / {ms} scale mismatches",
            if heads.is_some() {
                " (grouped heads)"
            } else {
                ""
            }
        );
        assert_eq!(
            (my, mq, ms),
            (0, 0, 0),
            "fused launch differs from the pair"
        );
    }
}

/// The column-major comparand of the table lane, and the multi-plane / GLU
/// launch, against the serving single-plane walk - which the gate above holds
/// to its own integer dot. The layouts only differ in where an operand sits
/// and the multi launch only in which plane a warp walks, so all of it has
/// to agree to the bit.
#[test]
fn ternary_layouts_and_multi_bitmatch_the_single_walk() {
    let Some(exec) = ternary_exec() else {
        return;
    };
    if !exec.has_ternary_gemv_b128() {
        eprintln!("pack lacks the ternary decode lane - skipping");
        return;
    }
    let in_dim = 5120usize;
    let outs = [1536usize, 256, 256];
    let planes: Vec<_> = outs
        .iter()
        .enumerate()
        .map(|(i, &o)| {
            let bytes = plane(PTQ1_0, in_dim, o, 0x3A11 + i as u64);
            exec.repack_kquant_raw(&bytes, vec![in_dim, o], GgmlType::Ptq1_0, "synthetic")
                .expect("repack")
        })
        .collect();
    let dx = exec.to_device(&noise(in_dim, 0x77)).expect("htod");
    let mut xq = exec.alloc_i8(in_dim).expect("alloc");
    let mut xs = exec.alloc(in_dim / 128).expect("alloc");
    exec.quantize_q8_b128(&dx, &mut xq, &mut xs, in_dim)
        .expect("quantize");
    let single = |w| {
        let w: &paddock_engine::gpu::RepackedKQ = w;
        let mut y = exec.alloc(w.dims[1]).expect("alloc");
        exec.ternary_gemv_b128(w, &xq, &xs, &mut y).expect("gemv");
        exec.to_host(&y).expect("dtoh")
    };
    let want: Vec<Vec<f32>> = planes.iter().map(single).collect();
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());

    {
        let mut y = exec.alloc(outs[0]).expect("alloc");
        exec.ternary_gemv_b128_layout(&planes[0], &xq, &xs, &mut y, true)
            .expect("column-major");
        assert!(
            same(&exec.to_host(&y).expect("dtoh"), &want[0]),
            "the column-major layout differs from the serving walk"
        );
    }

    if !exec.has_ternary_gemv_b128_multi() {
        eprintln!("pack lacks the multi-plane ternary launch (slot 630) - skipping that half");
        return;
    }
    let refs: Vec<&paddock_engine::gpu::RepackedKQ> = planes.iter().collect();
    assert!(exec.ternary_multi_fits(&refs));
    let mut ys: Vec<_> = outs
        .iter()
        .map(|&o| exec.alloc(o).expect("alloc"))
        .collect();
    {
        let mut it = ys.iter_mut();
        let (y0, y1, y2) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        exec.ternary_gemv_b128_multi(
            &mut [(&planes[0], y0), (&planes[1], y1), (&planes[2], y2)],
            &xq,
            &xs,
        )
        .expect("multi");
    }
    for (i, y) in ys.iter().enumerate() {
        assert!(
            same(&exec.to_host(y).expect("dtoh"), &want[i]),
            "plane {i} of the multi launch differs from its single launch"
        );
    }

    // gate | up + SwiGLU against the two launches and the engine's swiglu
    let mut g = exec.alloc(outs[1]).expect("alloc");
    exec.ternary_glu_b128(&planes[1], &planes[2], &xq, &xs, &mut g)
        .expect("glu");
    let mut gate = exec.to_device(&want[1]).expect("htod");
    let up = exec.to_device(&want[2]).expect("htod");
    exec.swiglu(&mut gate, &up, outs[1]).expect("swiglu");
    assert!(
        same(
            &exec.to_host(&g).expect("dtoh"),
            &exec.to_host(&gate).expect("dtoh")
        ),
        "the GLU launch differs from gemv + gemv + swiglu"
    );
}

/// Where the batch-1 decode lane and the rotation stand against the card:
/// weight bytes streamed per second over planes that cannot sit in L2 (clones
/// rotated past 4x its size), and the rotation's cost per
/// call at the three widths a Bonsai 27B decode step rotates. Not a gate.
#[test]
#[ignore]
fn ternary_decode_bandwidth() {
    let Some(exec) = ternary_exec() else {
        return;
    };
    for raw in [PTQ1_0, PQ2_0] {
        for (in_dim, out_dim, tag) in [
            (5120usize, 17408usize, "ffn gate"),
            (17408, 5120, "ffn down"),
        ] {
            let clones: Vec<_> = (0..28u64)
                .map(|c| {
                    let bytes = plane(raw, in_dim, out_dim, 0xC10 + c);
                    exec.repack_kquant_raw(&bytes, vec![in_dim, out_dim], ty_of(raw), "bench")
                        .expect("repack")
                })
                .collect();
            let bytes = (clones[0].data.len() + clones[0].scales.len()) as f64;
            let dx = exec.to_device(&noise(in_dim, 9)).expect("htod");
            let mut xq = exec.alloc_i8(in_dim).expect("alloc");
            let mut xs = exec.alloc(in_dim / 32).expect("alloc");
            let mut sums = exec.alloc(in_dim / 16).expect("alloc");
            exec.quantize_q8_sums(&dx, &mut xq, &mut xs, &mut sums, in_dim)
                .expect("quantize");
            let mut y = exec.alloc(out_dim).expect("alloc");
            // Variants run INTERLEAVED, round-robin, best round kept: this card
            // also drives a desktop, and numbers taken in separate runs differ
            // by more than the variants do.
            let table = clones.iter().all(|w| exec.ternary_gemv_b128_fits(w));
            let mut xq128 = exec.alloc_i8(in_dim).expect("alloc");
            let mut xs128 = exec.alloc(in_dim / 128).expect("alloc");
            if table {
                exec.quantize_q8_b128(&dx, &mut xq128, &mut xs128, in_dim)
                    .expect("quantize");
            }
            let variants: Vec<(String, u32)> =
                std::iter::once(("generic window lane".to_owned(), 99))
                    .chain(
                        [
                            ("table lane (serving: operands one load a word)", 0u32),
                            ("table lane, column-major operands", 1),
                        ]
                        .into_iter()
                        .filter(|_| table)
                        .map(|(n, pin)| (n.to_owned(), pin)),
                    )
                    .collect();
            let mut best = vec![f64::MAX; variants.len()];
            for round in 0..9 {
                for (vi, (_, rows)) in variants.iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    for w in &clones {
                        if *rows == 99 {
                            exec.kquant_gemv_w4a8(w, &xq, &xs, None, &mut y)
                                .expect("gemv");
                        } else {
                            exec.ternary_gemv_b128_layout(w, &xq128, &xs128, &mut y, *rows == 1)
                                .expect("gemv");
                        }
                    }
                    exec.synchronize().expect("sync");
                    let dt = t0.elapsed().as_secs_f64() / clones.len() as f64;
                    if round > 0 {
                        best[vi] = best[vi].min(dt);
                    }
                }
            }
            for ((name, _), dt) in variants.iter().zip(&best) {
                eprintln!(
                    "{:?} {tag} [{in_dim}x{out_dim}] {name}: {:.1} us, {:.1} GB/s of weights",
                    ty_of(raw),
                    dt * 1e6,
                    bytes / dt / 1e9
                );
            }
        }
    }
    if exec.has_hadamard() {
        for width in [5120usize, 6144, 17408] {
            let signs = exec.to_device(&vec![1.0f32; width]).expect("htod");
            let mut x = exec.to_device(&noise(width, 5)).expect("htod");
            exec.hadamard_rotate_inplace(&mut x, &signs, 1, width, 1024, false)
                .expect("warm");
            exec.synchronize().expect("sync");
            let reps = 2000;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                exec.hadamard_rotate_inplace(&mut x, &signs, 1, width, 1024, false)
                    .expect("rotate");
            }
            exec.synchronize().expect("sync");
            eprintln!(
                "rotation 1x{width}: {:.1} us a call",
                t0.elapsed().as_secs_f64() / reps as f64 * 1e6
            );
        }
    }
}
