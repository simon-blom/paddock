//! `pd_matvec_f32_batch` (the batch < 16 arm, and any batch whose output
//! count is not a multiple of 8): every output is its documented order - per
//! thread (256 of them) an ascending FMA chain over i = t, t+256, .., then a
//! warp shuffle-down tree, then a serial sum over the 8 warps - so each sum is
//! a function of that order alone, whatever the kernel does to schedule its
//! loads. The router feeds a top-k where a last-ulp change could flip a tie,
//! which is why the order is a contract. Checked bit for bit against a host
//! emulation of it over the shapes a qwen4_exp decode tick runs (the hc
//! inject 10240 -> 4, the GDN alpha/beta 2560 -> 96, the router 2560 -> 513)
//! and awkward ones (K tails, the BT=4 tile).
//!
//! Gated on: CUDA device + built pack.

mod common;

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

/// The kernel's order, on the host.
fn emulate(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize, batch: usize) -> Vec<f32> {
    const NT: usize = 256;
    let mut out = vec![0f32; batch * out_dim];
    for b in 0..batch {
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let xr = &x[b * in_dim..(b + 1) * in_dim];
            let acc: Vec<f32> = (0..NT)
                .map(|t| {
                    let mut a = 0f32;
                    let mut i = t;
                    while i < in_dim {
                        a = wr[i].mul_add(xr[i], a);
                        i += NT;
                    }
                    a
                })
                .collect();
            let mut total = 0f32;
            for warp in acc.chunks(32) {
                let mut v = warp.to_vec();
                for s in [16usize, 8, 4, 2, 1] {
                    // shfl_down: a lane past the warp's end reads its own value
                    let prev = v.clone();
                    for (lane, vl) in v.iter_mut().enumerate() {
                        *vl = prev[lane]
                            + if lane + s < 32 {
                                prev[lane + s]
                            } else {
                                prev[lane]
                            };
                    }
                }
                total += v[0];
            }
            out[b * out_dim + o] = total;
        }
    }
    out
}

#[test]
fn matvec_f32_batch_sums_in_its_documented_order() {
    let Some(exec) = common::gpu() else {
        return;
    };
    for (in_dim, out_dim, batch) in [
        (10240usize, 4usize, 1usize),
        (2560, 96, 1),
        (2560, 513, 1),
        (2560, 513, 3),
        (1000, 7, 5),
        (4099, 33, 2),
        (256, 5, 15),
        (2560, 13, 17),
    ] {
        let w = det(in_dim * out_dim, 7 + in_dim as u64);
        let x = det(in_dim * batch, 11 + out_dim as u64);
        let d_w = exec.to_device(&w).expect("w");
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(out_dim * batch).expect("y");
        exec.matvec_f32_raw(&d_w, in_dim, out_dim, &d_x, &mut d_y, batch)
            .expect("matvec");
        let got = exec.to_host(&d_y).expect("dtoh");
        let want = emulate(&w, &x, in_dim, out_dim, batch);
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        eprintln!(
            "matvec_f32 [{in_dim} -> {out_dim}] x {batch}: {diff} of {} differ",
            got.len()
        );
        assert_eq!(
            diff, 0,
            "[{in_dim} -> {out_dim}] x {batch} left its documented order"
        );
    }
}
