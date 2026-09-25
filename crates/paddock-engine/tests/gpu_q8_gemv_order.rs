//! The Q8_0 decode GEMVs against their multi-token forms, bit for bit per
//! token - what a speculative verify walks these planes with, so a verify row
//! lands where a decode tick lands only while the two agree:
//!
//! - `pd_q8_0_gemv_repacked` (batch 1, whatever geometry its launcher elects -
//!   the warp-per-row form for short rows included) against the row-exact
//!   twin `pd_q8_0_gemv_repacked_rows` (slot 598) at 1..9 tokens, each token
//!   against a batch-1 call on that token alone. Shapes: the planes a
//!   qwen4_exp decode tick runs (the hc up [320 -> 10240], the shared-expert
//!   down [640 -> 2560]) and the block path's own (a full 2048-wide row, a
//!   4096-wide one), plus ragged row counts.
//! - `pd_q8_0_gemv_sk` (slot 588) at 2..9 tokens - the multi-token sibling,
//!   one weight read per span - against the same kernel at batch 1 per
//!   token, at the hc down's [10240 -> 320] and ragged shapes, splits 2 and 3.
//!
//! (A host emulation of the order is not the oracle here: the chunk's 16
//! products are one expression, and how nvcc contracts it into FMAs is the
//! compiler's, not the kernel's.)
//!
//! Gated on: CUDA device + built pack.

mod common;

use half::f16;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn plane(
    exec: &paddock_engine::gpu::GpuExecutor,
    in_dim: usize,
    out_dim: usize,
) -> paddock_engine::gpu::RepackedQ8 {
    let nb = in_dim / 32;
    let q: Vec<u8> = det(in_dim * out_dim, 3 + in_dim as u64)
        .iter()
        .map(|v| (v * 254.0) as i8 as u8)
        .collect();
    let sc: Vec<u8> = det(nb * out_dim, 5 + out_dim as u64)
        .iter()
        .flat_map(|v| f16::from_f32(v * 0.02 + 0.011).to_le_bytes())
        .collect();
    paddock_engine::gpu::RepackedQ8 {
        data: exec.to_device_u8(&q).expect("q"),
        scale: exec.to_device_u8(&sc).expect("scales"),
        dims: vec![in_dim, out_dim],
    }
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
fn q8_0_gemv_repacked_matches_its_row_exact_twin() {
    let Some(exec) = common::gpu() else {
        return;
    };
    for (in_dim, out_dim) in [
        (320usize, 10240usize),
        (640, 2560),
        (320, 17),
        (1024, 33),
        (2048, 5),
        (4096, 3),
    ] {
        let w = plane(&exec, in_dim, out_dim);
        let bias = exec.to_device(&det(out_dim, 13)).expect("bias");
        for with_bias in [false, true] {
            let b = with_bias.then_some(&bias);
            for batch in [1usize, 2, 3, 4, 5, 8, 9] {
                let xs = det(batch * in_dim, 9 + batch as u64);
                let d_x = exec.to_device(&xs).expect("x");
                let mut rows = exec.alloc(batch * out_dim).expect("y rows");
                assert!(
                    exec.q8_0_gemv_repacked_rows(&w, b, &d_x, &mut rows, batch)
                        .expect("rows"),
                    "pack predates slot 598"
                );
                let rows = exec.to_host(&rows).expect("dtoh");
                let mut diff = 0;
                for t in 0..batch {
                    let d_xt = exec
                        .to_device(&xs[t * in_dim..(t + 1) * in_dim])
                        .expect("x_t");
                    let mut one = exec.alloc(out_dim).expect("y");
                    exec.q8_0_gemv_repacked(&w, b, &d_xt, &mut one)
                        .expect("gemv");
                    let one = exec.to_host(&one).expect("dtoh");
                    diff += bit_diffs(&one, &rows[t * out_dim..(t + 1) * out_dim]);
                }
                eprintln!(
                    "q8_0 rows [{in_dim} -> {out_dim}] x {batch} bias {with_bias}: {diff} of {} differ from batch 1",
                    batch * out_dim
                );
                assert_eq!(
                    diff, 0,
                    "[{in_dim} -> {out_dim}] x {batch}: the twin parted from the batch-1 GEMV"
                );
            }
        }
    }
}

#[test]
fn q8_0_gemv_sk_tokens_match_batch_one() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_q8_0_gemv_sk() {
        return;
    }
    for (in_dim, out_dim) in [(10240usize, 320usize), (2048, 7), (4096, 33)] {
        let w = plane(&exec, in_dim, out_dim);
        let bias = exec.to_device(&det(out_dim, 17)).expect("bias");
        for split in [2usize, 3] {
            for (batch, with_bias) in [(2usize, false), (3, true), (5, false), (9, true)] {
                let b = with_bias.then_some(&bias);
                let xs = det(batch * in_dim, 21 + batch as u64);
                let d_x = exec.to_device(&xs).expect("x");
                let mut y = exec.alloc(batch * out_dim).expect("y");
                let mut part = exec.alloc(batch * out_dim * split).expect("partials");
                let mut cnt = exec.alloc_u32(batch * out_dim).expect("counters");
                exec.q8_0_gemv_sk(&w, b, &d_x, &mut y, &mut part, &mut cnt, batch, split)
                    .expect("sk batch");
                let y = exec.to_host(&y).expect("dtoh");
                let mut diff = 0;
                for t in 0..batch {
                    let d_xt = exec
                        .to_device(&xs[t * in_dim..(t + 1) * in_dim])
                        .expect("x_t");
                    let mut one = exec.alloc(out_dim).expect("y1");
                    let mut p1 = exec.alloc(out_dim * split).expect("p1");
                    let mut c1 = exec.alloc_u32(out_dim).expect("c1");
                    exec.q8_0_gemv_sk(&w, b, &d_xt, &mut one, &mut p1, &mut c1, 1, split)
                        .expect("sk 1");
                    let one = exec.to_host(&one).expect("dtoh");
                    diff += bit_diffs(&one, &y[t * out_dim..(t + 1) * out_dim]);
                }
                eprintln!(
                    "q8_0 sk [{in_dim} -> {out_dim}] split {split} x {batch} bias {with_bias}: {diff} of {} differ from batch 1",
                    batch * out_dim
                );
                assert_eq!(
                    diff, 0,
                    "[{in_dim} -> {out_dim}] split {split} x {batch}: a token parted from its batch-1 launch"
                );
            }
        }
    }
}
