//! The dense-prediction kernels (slots 610-617), each against a plain
//! reference written out below - and the half activation interface (618-623),
//! each against its f32 twin, bit for bit: a half kernel is defined as the f32
//! one's result rounded to nearest, so there is no tolerance to choose.
//!
//! The model-level gate (`gpu_dinov3_golden`) says whether the whole tower
//! reproduces its golden vectors; this one says which op is wrong when it does
//! not, and it runs without the checkpoint. Every one of these is an indexing
//! kernel at heart - a re-lay, a gather, a strided reduction - so the shapes
//! are chosen to be small, NON-square where the op allows it, and to carry
//! more than one chip with a ragged row stride, which is where a transposed
//! axis or a dropped stride shows up as O(1) wrong rather than O(1e-6).
//!
//! Inputs come from an LCG, never `rand`: a failure has to reproduce.

mod common;

use half::f16;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn fill(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n).map(|_| lcg(seed)).collect()
}

fn gelu(v: f64) -> f64 {
    0.5 * v * (1.0 + libm_erf(v * std::f64::consts::FRAC_1_SQRT_2))
}

/// erf to ~1e-7 (Abramowitz-Stegun 7.1.26 is only 1e-7 too, but this series /
/// continued-fraction pair is good to 1e-12, well under the f16 store).
fn libm_erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 2.5 {
        // Maclaurin series
        let (mut term, mut sum, mut n) = (a, a, 0.0f64);
        while term.abs() > 1e-17 {
            n += 1.0;
            term *= -a * a / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // continued fraction for erfc
        let mut f = 0.0f64;
        for k in (1..60).rev() {
            f = (k as f64 / 2.0) / (a + f);
        }
        1.0 - (-a * a).exp() / ((a + f) * std::f64::consts::PI.sqrt())
    };
    if x < 0.0 { -r } else { r }
}

#[test]
fn patch_rows_normalize_and_lay_out() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // 2 chips, 12 px, patch 4 -> a 3x3 grid; 4 bands; 11 rows per chip (2 tail)
    let (chips, px, p, ch, chip_rows) = (2usize, 12usize, 4usize, 4usize, 11usize);
    let g = px / p;
    let mean = [0.27f32, 0.31, 0.33, 0.34];
    let std = [0.14f32, 0.13, 0.11, 0.15];
    let mut seed = 7u64;
    let pix: Vec<u8> = (0..chips * px * px * ch)
        .map(|_| (lcg(&mut seed) * 127.0 + 128.0) as u8)
        .collect();

    let d_pix = exec.to_device_u8(&pix).unwrap();
    let k = ch * p * p;
    let mut d_out = exec.alloc_f16(chips * chip_rows * k).unwrap();
    exec.dp_u8_patch_rows(&d_pix, &mut d_out, &mean, &std, chips, px, p, chip_rows)
        .unwrap();
    let got = exec.to_host_f16_len(&d_out, chips * chip_rows * k).unwrap();

    for b in 0..chips {
        for r in 0..chip_rows {
            for col in 0..k {
                let want = if r < g * g {
                    let (c, rem) = (col / (p * p), col % (p * p));
                    let (ky, kx) = (rem / p, rem % p);
                    let (gy, gx) = (r / g, r % g);
                    let src = ((b * px + gy * p + ky) * px + gx * p + kx) * ch + c;
                    // divide by 255 first: the statistics are in 0-1 units
                    f16::from_f32((pix[src] as f32 / 255.0 - mean[c]) / std[c])
                } else {
                    f16::from_f32(0.0)
                };
                let i = (b * chip_rows + r) * k + col;
                assert_eq!(
                    got[i].to_bits(),
                    want.to_bits(),
                    "chip {b} row {r} col {col}"
                );
            }
        }
    }
}

#[test]
fn qkv_split_folds_biases_and_ropes_only_the_patch_rows() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // 3 chips of 7 rows, the first 5 roped; 3 heads of 8
    let (chips, chip_rows, n_rope, heads, hd) = (3usize, 7usize, 5usize, 3usize, 8usize);
    let (d, half, rows) = (heads * hd, hd / 2, chips * chip_rows);
    let mut seed = 11u64;
    let qkv = fill(&mut seed, rows * 3 * d);
    let bq = fill(&mut seed, d);
    let bv = fill(&mut seed, d);
    let ang = fill(&mut seed, n_rope * half);
    let cs: Vec<f32> = ang.iter().map(|a| (a * 3.0).cos()).collect();
    let sn: Vec<f32> = ang.iter().map(|a| (a * 3.0).sin()).collect();

    let (mut dq, mut dk, mut dv) = (
        exec.alloc(rows * d).unwrap(),
        exec.alloc(rows * d).unwrap(),
        exec.alloc(rows * d).unwrap(),
    );
    exec.dp_qkv_split_rope(
        &exec.to_device(&qkv).unwrap(),
        &exec.to_device(&bq).unwrap(),
        &exec.to_device(&bv).unwrap(),
        &exec.to_device(&cs).unwrap(),
        &exec.to_device(&sn).unwrap(),
        &mut dq,
        &mut dk,
        &mut dv,
        d,
        hd,
        rows,
        chip_rows,
        n_rope,
    )
    .unwrap();
    let (q, k, v) = (
        exec.to_host(&dq).unwrap(),
        exec.to_host(&dk).unwrap(),
        exec.to_host(&dv).unwrap(),
    );

    let mut worst = 0.0f32;
    for r in 0..rows {
        let t = r % chip_rows;
        for h in 0..heads {
            for j in 0..half {
                let (e0, e1) = (h * hd + j, h * hd + j + half);
                let (q0, q1) = (qkv[r * 3 * d + e0] + bq[e0], qkv[r * 3 * d + e1] + bq[e1]);
                let (k0, k1) = (qkv[r * 3 * d + d + e0], qkv[r * 3 * d + d + e1]);
                // rotate_half: x*cos + cat(-x2, x1)*sin
                let (c, s) = if t < n_rope {
                    (cs[t * half + j], sn[t * half + j])
                } else {
                    (1.0, 0.0)
                };
                for (got, want) in [
                    (q[r * d + e0], q0 * c - q1 * s),
                    (q[r * d + e1], q1 * c + q0 * s),
                    (k[r * d + e0], k0 * c - k1 * s),
                    (k[r * d + e1], k1 * c + k0 * s),
                    (v[r * d + e0], qkv[r * 3 * d + 2 * d + e0] + bv[e0]),
                    (v[r * d + e1], qkv[r * 3 * d + 2 * d + e1] + bv[e1]),
                ] {
                    worst = worst.max((got - want).abs());
                }
            }
        }
    }
    // same f32 ops; only fma contraction can differ
    assert!(worst < 2e-6, "qkv split/rope max abs diff {worst}");
}

#[test]
fn layerscale_seam_updates_the_residual_and_norms_it() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // 1024 takes the register-staged arm, 2304 the re-reading one
    for n in [1024usize, 2304] {
        let rows = 5usize;
        let eps = 1e-5f32;
        let mut seed = 13u64 + n as u64;
        let x = fill(&mut seed, rows * n);
        let proj = fill(&mut seed, rows * n);
        let bias = fill(&mut seed, n);
        let ls: Vec<f32> = fill(&mut seed, n).iter().map(|v| v * 0.5).collect();
        let w = fill(&mut seed, n);
        let b = fill(&mut seed, n);

        let mut dx = exec.to_device(&x).unwrap();
        let mut dout = exec.alloc_f16(rows * n).unwrap();
        exec.dp_res_ls_ln_f16(
            &mut dx,
            &exec.to_device(&proj).unwrap(),
            &exec.to_device(&bias).unwrap(),
            &exec.to_device(&ls).unwrap(),
            &exec.to_device(&w).unwrap(),
            &exec.to_device(&b).unwrap(),
            &mut dout,
            rows,
            n,
            eps,
        )
        .unwrap();
        let gx = exec.to_host(&dx).unwrap();
        let gout = exec.to_host_f16_len(&dout, rows * n).unwrap();

        let (mut wx, mut wo) = (0.0f32, 0.0f32);
        for r in 0..rows {
            let xr: Vec<f64> = (0..n)
                .map(|i| {
                    x[r * n + i] as f64 + ls[i] as f64 * (proj[r * n + i] as f64 + bias[i] as f64)
                })
                .collect();
            let mean = xr.iter().sum::<f64>() / n as f64;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
            let inv = 1.0 / (var + eps as f64).sqrt();
            for i in 0..n {
                wx = wx.max((gx[r * n + i] - xr[i] as f32).abs());
                let want = (xr[i] - mean) * inv * w[i] as f64 + b[i] as f64;
                wo = wo.max((gout[r * n + i].to_f32() - want as f32).abs());
            }
        }
        assert!(wx < 2e-6, "n={n}: residual max abs diff {wx}");
        // outputs are O(1); an f16 store is good to ~1e-3 there
        assert!(wo < 2e-3, "n={n}: normed output max abs diff {wo}");
    }
}

#[test]
fn group_norm_reduces_over_pixels_and_group_channels() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // (pixels, channels, groups): one channel per group (the last decoder
    // stage), several, and a pixel count that is not a multiple of the 64-pixel
    // partial so the ragged last chunk is exercised
    for (px, c, g) in [(200usize, 32usize, 32usize), (129, 64, 32), (70, 256, 32)] {
        let chips = 2usize;
        let eps = 1e-5f32;
        let mut seed = 17u64 + c as u64;
        // give every channel its own offset so a group's mean is not ~0 -
        // that is what separates "centred squares" from E[x^2] - mean^2
        let x: Vec<f32> = (0..chips * px * c)
            .map(|i| lcg(&mut seed) + ((i % c) as f32) * 0.25 - 3.0)
            .collect();
        let xb = fill(&mut seed, c);
        let w = fill(&mut seed, c);
        let b = fill(&mut seed, c);

        let mut dout = exec.alloc_f16(chips * px * c).unwrap();
        let mut part = exec
            .alloc(paddock_engine::gpu::GpuExecutor::dp_group_norm_part_len(
                chips, px, c,
            ))
            .unwrap();
        let mut stat = exec.alloc(2 * chips * g).unwrap();
        exec.dp_group_norm_gelu_f16(
            &exec.to_device(&x).unwrap(),
            &exec.to_device(&xb).unwrap(),
            &exec.to_device(&w).unwrap(),
            &exec.to_device(&b).unwrap(),
            &mut dout,
            &mut part,
            &mut stat,
            chips,
            px,
            c,
            g,
            eps,
        )
        .unwrap();
        let got = exec.to_host_f16_len(&dout, chips * px * c).unwrap();

        let cg = c / g;
        let mut worst = 0.0f32;
        for chip in 0..chips {
            for gi in 0..g {
                let at = |p: usize, j: usize| {
                    let ch = gi * cg + j;
                    x[(chip * px + p) * c + ch] as f64 + xb[ch] as f64
                };
                let n = (px * cg) as f64;
                let mut mean = 0.0;
                for p in 0..px {
                    for j in 0..cg {
                        mean += at(p, j);
                    }
                }
                mean /= n;
                let mut var = 0.0;
                for p in 0..px {
                    for j in 0..cg {
                        var += (at(p, j) - mean).powi(2);
                    }
                }
                let inv = 1.0 / (var / n + eps as f64).sqrt();
                for p in 0..px {
                    for j in 0..cg {
                        let ch = gi * cg + j;
                        let y = (at(p, j) - mean) * inv * w[ch] as f64 + b[ch] as f64;
                        let want = gelu(y) as f32;
                        let have = got[(chip * px + p) * c + ch].to_f32();
                        worst = worst.max((have - want).abs() / (1.0 + want.abs()));
                    }
                }
            }
        }
        assert!(
            worst < 1.5e-3,
            "px={px} c={c} g={g}: group norm rel diff {worst}"
        );
    }
}

#[test]
fn im2row_gathers_nine_taps_with_zero_padding() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // non-square, 2 chips, and a source whose chips carry 3 extra rows each
    let (chips, h, w, c) = (2usize, 5usize, 7usize, 8usize);
    let src_rows = h * w + 3;
    let mut seed = 19u64;
    let src = fill(&mut seed, chips * src_rows * c);
    let src16: Vec<f16> = src.iter().map(|v| f16::from_f32(*v)).collect();
    let n = chips * h * w * 9 * c;

    let mut o32 = exec.alloc_f16(n).unwrap();
    exec.dp_im2row3_f32(
        &exec.to_device(&src).unwrap(),
        &mut o32,
        chips,
        h,
        w,
        c,
        src_rows,
    )
    .unwrap();
    let mut o16 = exec.alloc_f16(n).unwrap();
    exec.dp_im2row3_f16(
        &exec.f16_to_device(&src16).unwrap(),
        &mut o16,
        chips,
        h,
        w,
        c,
        src_rows,
    )
    .unwrap();
    let (g32, g16) = (
        exec.to_host_f16_len(&o32, n).unwrap(),
        exec.to_host_f16_len(&o16, n).unwrap(),
    );

    for chip in 0..chips {
        for y in 0..h {
            for x in 0..w {
                for t in 0..9 {
                    let (yy, xx) = (
                        y as isize + (t / 3) as isize - 1,
                        x as isize + (t % 3) as isize - 1,
                    );
                    let inside = yy >= 0 && yy < h as isize && xx >= 0 && xx < w as isize;
                    for ch in 0..c {
                        let want = if inside {
                            src16[(chip * src_rows + yy as usize * w + xx as usize) * c + ch]
                        } else {
                            f16::from_f32(0.0)
                        };
                        // TAP-outer: col = t*C + c
                        let i = ((chip * h + y) * w + x) * 9 * c + t * c + ch;
                        assert_eq!(
                            g32[i].to_bits(),
                            want.to_bits(),
                            "f32 src ({chip},{y},{x}) tap {t} ch {ch}"
                        );
                        assert_eq!(
                            g16[i].to_bits(),
                            want.to_bits(),
                            "f16 src ({chip},{y},{x}) tap {t} ch {ch}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn transposed_conv_seam_scatters_and_samples_the_skip() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // an hs x hs skip grid resized to 2h x 2w at factors 2 and 4, with the skip
    // chips strided by extra rows (the token plane's class + register rows)
    for (h, hs) in [(3usize, 3usize), (6, 3)] {
        let (chips, w, c) = (2usize, h, 8usize);
        let skip_rows = hs * hs + 5;
        let mut seed = 23u64 + h as u64;
        let g = fill(&mut seed, chips * h * w * 4 * c);
        let bias = fill(&mut seed, c);
        let skip = fill(&mut seed, chips * skip_rows * c);
        let (oh, ow) = (2 * h, 2 * w);

        let mut dout = exec.alloc(chips * oh * ow * c).unwrap();
        exec.dp_convt2_skip(
            &exec.to_device(&g).unwrap(),
            &exec.to_device(&bias).unwrap(),
            &exec.to_device(&skip).unwrap(),
            &mut dout,
            chips,
            h,
            w,
            c,
            hs,
            skip_rows,
        )
        .unwrap();
        let got = exec.to_host(&dout).unwrap();

        // torch upsample_bilinear2d, align_corners=False, given a size
        let src_index = |dst: usize, n_in: usize, n_out: usize| {
            let s = (n_in as f32 / n_out as f32) * (dst as f32 + 0.5) - 0.5;
            let s = s.max(0.0);
            let i0 = s as usize;
            (i0, (i0 + 1).min(n_in - 1), s - i0 as f32)
        };
        let mut worst = 0.0f32;
        for chip in 0..chips {
            for yy in 0..oh {
                for xx in 0..ow {
                    let tap = (yy & 1) * 2 + (xx & 1);
                    let (y0, y1, ly) = src_index(yy, hs, oh);
                    let (x0, x1, lx) = src_index(xx, hs, ow);
                    for ch in 0..c {
                        let sk =
                            |y: usize, x: usize| skip[(chip * skip_rows + y * hs + x) * c + ch];
                        let bil = (1.0 - ly) * ((1.0 - lx) * sk(y0, x0) + lx * sk(y0, x1))
                            + ly * ((1.0 - lx) * sk(y1, x0) + lx * sk(y1, x1));
                        let up =
                            g[((chip * h + yy / 2) * w + xx / 2) * 4 * c + tap * c + ch] + bias[ch];
                        let have = got[((chip * oh + yy) * ow + xx) * c + ch];
                        worst = worst.max((have - (up + bil)).abs());
                    }
                }
            }
        }
        assert!(worst < 3e-6, "h={h} hs={hs}: seam max abs diff {worst}");
    }
}

#[test]
fn heads_take_the_argmax_and_split_the_regression_column() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let (rows, ncls) = (1000usize, 10usize);
    let mut seed = 29u64;
    let o = fill(&mut seed, rows * (ncls + 1));
    let bias = fill(&mut seed, ncls + 1);

    let mut cls = exec.alloc_u8(rows).unwrap();
    let mut height = exec.alloc(rows).unwrap();
    let mut logits = exec.alloc_f16(rows * ncls).unwrap();
    let d_o = exec.to_device(&o).unwrap();
    let d_b = exec.to_device(&bias).unwrap();
    exec.dp_seg_heads(
        &d_o,
        &d_b,
        &mut cls,
        &mut height,
        Some(&mut logits),
        rows,
        ncls,
    )
    .unwrap();
    let (gc, gh, gl) = (
        exec.to_host_u8_len(&cls, rows).unwrap(),
        exec.to_host(&height).unwrap(),
        exec.to_host_f16_len(&logits, rows * ncls).unwrap(),
    );
    for r in 0..rows {
        let v: Vec<f32> = (0..ncls).map(|c| o[r * (ncls + 1) + c] + bias[c]).collect();
        // strict greater, ascending: the lowest index wins a tie
        let mut bi = 0;
        for c in 1..ncls {
            if v[c] > v[bi] {
                bi = c;
            }
        }
        assert_eq!(gc[r] as usize, bi, "row {r} argmax");
        assert_eq!(
            gh[r].to_bits(),
            (o[r * (ncls + 1) + ncls] + bias[ncls]).to_bits(),
            "row {r} height"
        );
        for c in 0..ncls {
            assert_eq!(
                gl[r * ncls + c].to_bits(),
                f16::from_f32(v[c]).to_bits(),
                "row {r} logit {c}"
            );
        }
    }
    // the logits plane is optional, and leaving it out must not move the rest
    let mut cls2 = exec.alloc_u8(rows).unwrap();
    let mut h2 = exec.alloc(rows).unwrap();
    exec.dp_seg_heads(&d_o, &d_b, &mut cls2, &mut h2, None, rows, ncls)
        .unwrap();
    assert_eq!(exec.to_host_u8_len(&cls2, rows).unwrap(), gc);
}

// ---------------------------------------------------------------- 618-623
// Inputs for the twin tests are drawn on the f16 grid (multiples of 1/256
// below 8, where f16 still resolves 1/256), so handing them to the f32 kernel
// as floats and to the half kernel as halves is handing both the same numbers.

fn fill_h(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| (lcg(seed) * scale * 256.0).round() / 256.0)
        .collect()
}

fn halves(v: &[f32]) -> Vec<f16> {
    let h: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
    // the premise of every test below
    assert!(h.iter().zip(v).all(|(a, b)| a.to_f32() == *b));
    h
}

fn same_bits(what: &str, got: &[f16], want: &[f16]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(bad, 0, "{what}: {bad} of {} halves differ", got.len());
}

#[test]
fn half_landing_gemm_is_the_f32_landing_rounded() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_dense_pred_h() {
        return;
    }
    // (in, out, rows): a ragged under-filled grid; two past one block per SM
    // on an 84-SM part, which is where the blocked raster is elected (ragged
    // in both axes, a weight-tile count that does not divide by the group of
    // four) - the small-K one cheap enough to check against a plain product;
    // and a narrow-M one, where the f32 entry may K-split.
    for (k, m, n) in [
        (64usize, 200usize, 300usize),
        (64, 600, 2200),
        (256, 1100, 12000),
        (1024, 32, 2100),
    ] {
        let mut seed = 31u64 + (k * m) as u64;
        let w: Vec<f32> = fill(&mut seed, k * m).iter().map(|v| v * 0.05).collect();
        let x = fill(&mut seed, k * n);
        let wt = paddock_engine::gpu::HalfTensor {
            buf: exec.to_device_f16(&w, "w").unwrap(),
            dims: vec![k, m],
        };
        let dx = exec.to_device_f16(&x, "x").unwrap();
        let mut y32 = exec.alloc(n * m).unwrap();
        let mut y16 = exec.alloc_f16(n * m).unwrap();
        exec.matvec_batch_f16(&wt, &dx, &mut y32, n).unwrap();
        exec.matvec_batch_f16_h(&wt, &dx, &mut y16, n).unwrap();
        let want: Vec<f16> = exec
            .to_host(&y32)
            .unwrap()
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let got = exec.to_host_f16_len(&y16, n * m).unwrap();
        if (k, m) == (1024, 32) {
            // the one shape class where the two entries may legitimately part:
            // the f32 entry K-splits an under-filled grid (partial sums
            // regroup), the landing never does. Same numbers to f32 noise.
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-3, "K={k} M={m} N={n}: max abs diff {worst}");
        } else {
            same_bits(&format!("gemm_h K={k} M={m} N={n}"), &got, &want);
        }

        // and the landing itself against a plain f64 product of the same
        // halves, on the small-K shapes - the twin test alone would pass if
        // both entries were wrong together, and the second of these is the
        // blocked raster's only plain reference
        if k == 64 {
            let (wh, xh) = (halves_of(&w), halves_of(&x));
            let mut worst = 0.0f64;
            for r in 0..n {
                for o in 0..m {
                    let mut acc = 0.0f64;
                    for i in 0..k {
                        acc += wh[o * k + i] as f64 * xh[r * k + i] as f64;
                    }
                    worst = worst.max((acc - got[r * m + o].to_f64()).abs() / (1.0 + acc.abs()));
                }
            }
            assert!(worst < 1e-3, "gemm_h vs f64 product: rel {worst}");
        }
    }
}

/// what the f16 upload holds, widened back
fn halves_of(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| f16::from_f32(*x).to_f32()).collect()
}

#[test]
fn half_attention_is_the_f32_attention_rounded() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_dense_pred_h() {
        return;
    }
    // hd 64 is the tower's; hd 72 pads to 80 inside the kernel, the arm where
    // a half interface could read past a head; 130 rows is a ragged last
    // query block and a ragged last key tile; two groups, three heads.
    for (hd, t) in [(64usize, 130usize), (72, 70)] {
        let (heads, batch) = (3usize, 2usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let n = batch * t * heads * hd;
        let mut seed = 77u64 + hd as u64;
        let q = fill_h(&mut seed, n, 4.0);
        let k = fill_h(&mut seed, n, 4.0);
        let v = fill_h(&mut seed, n, 2.0);

        let mut o32 = exec.alloc(n).unwrap();
        exec.vision_attn_x(
            &exec.to_device(&q).unwrap(),
            &exec.to_device(&k).unwrap(),
            &exec.to_device(&v).unwrap(),
            &mut o32,
            t,
            t,
            heads,
            hd,
            batch,
            scale,
        )
        .unwrap();
        let want: Vec<f16> = exec
            .to_host(&o32)
            .unwrap()
            .iter()
            .map(|x| f16::from_f32(*x))
            .collect();

        // the half kernel takes q PRE-SCALED: the same product, the same round
        let qs: Vec<f16> = q.iter().map(|x| f16::from_f32(x * scale)).collect();
        let mut o16 = exec.alloc_f16(n).unwrap();
        exec.vision_attn_h(
            &exec.f16_to_device(&qs).unwrap(),
            &exec.f16_to_device(&halves(&k)).unwrap(),
            &exec.f16_to_device(&halves(&v)).unwrap(),
            &mut o16,
            t,
            t,
            heads,
            hd,
            batch,
        )
        .unwrap();
        let got = exec.to_host_f16_len(&o16, n).unwrap();
        same_bits(&format!("vision_attn_h hd={hd} t={t}"), &got, &want);
    }
}

#[test]
fn half_qkv_split_is_the_f32_split_scaled_and_rounded() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_dense_pred_h() {
        return;
    }
    let (chips, chip_rows, n_rope, heads, hd) = (2usize, 7usize, 5usize, 3usize, 16usize);
    let (d, rows, half) = (heads * hd, chips * chip_rows, hd / 2);
    let qs = 0.25f32; // a power of two, so the scale itself rounds nothing
    let mut seed = 5u64;
    let qkv = fill_h(&mut seed, rows * 3 * d, 8.0);
    let bq = fill(&mut seed, d);
    let bv = fill(&mut seed, d);
    let ang: Vec<f32> = fill(&mut seed, n_rope * half)
        .iter()
        .map(|a| a * 3.0)
        .collect();
    let cos: Vec<f32> = ang.iter().map(|a| a.cos()).collect();
    let sin: Vec<f32> = ang.iter().map(|a| a.sin()).collect();
    let (dbq, dbv) = (exec.to_device(&bq).unwrap(), exec.to_device(&bv).unwrap());
    let (dc, ds) = (exec.to_device(&cos).unwrap(), exec.to_device(&sin).unwrap());

    let (mut q, mut k, mut v) = (
        exec.alloc(rows * d).unwrap(),
        exec.alloc(rows * d).unwrap(),
        exec.alloc(rows * d).unwrap(),
    );
    exec.dp_qkv_split_rope(
        &exec.to_device(&qkv).unwrap(),
        &dbq,
        &dbv,
        &dc,
        &ds,
        &mut q,
        &mut k,
        &mut v,
        d,
        hd,
        rows,
        chip_rows,
        n_rope,
    )
    .unwrap();
    let round = |b: &cudarc::driver::CudaSlice<f32>, mul: f32| -> Vec<f16> {
        exec.to_host(b)
            .unwrap()
            .iter()
            .map(|x| f16::from_f32(x * mul))
            .collect()
    };

    let (mut q16, mut k16, mut v16) = (
        exec.alloc_f16(rows * d).unwrap(),
        exec.alloc_f16(rows * d).unwrap(),
        exec.alloc_f16(rows * d).unwrap(),
    );
    exec.dp_qkv_split_rope_h(
        &exec.f16_to_device(&halves(&qkv)).unwrap(),
        &dbq,
        &dbv,
        &dc,
        &ds,
        &mut q16,
        &mut k16,
        &mut v16,
        d,
        hd,
        rows,
        chip_rows,
        n_rope,
        qs,
    )
    .unwrap();
    same_bits(
        "split_h q",
        &exec.to_host_f16_len(&q16, rows * d).unwrap(),
        &round(&q, qs),
    );
    same_bits(
        "split_h k",
        &exec.to_host_f16_len(&k16, rows * d).unwrap(),
        &round(&k, 1.0),
    );
    same_bits(
        "split_h v",
        &exec.to_host_f16_len(&v16, rows * d).unwrap(),
        &round(&v, 1.0),
    );
}

#[test]
fn half_seam_and_half_gelu_match_their_f32_twins() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_dense_pred_h() {
        return;
    }
    // ---- the LayerScale seam, both norm arms ----
    for n in [1024usize, 2304] {
        let rows = 5usize;
        let mut seed = 19u64 + n as u64;
        let x = fill(&mut seed, rows * n);
        let proj = fill_h(&mut seed, rows * n, 7.5);
        let bias = fill(&mut seed, n);
        let ls: Vec<f32> = fill(&mut seed, n).iter().map(|v| v * 0.5).collect();
        let (w, b) = (fill(&mut seed, n), fill(&mut seed, n));
        let (dbias, dls) = (exec.to_device(&bias).unwrap(), exec.to_device(&ls).unwrap());
        let (dw, db) = (exec.to_device(&w).unwrap(), exec.to_device(&b).unwrap());

        let mut xa = exec.to_device(&x).unwrap();
        let mut oa = exec.alloc_f16(rows * n).unwrap();
        exec.dp_res_ls_ln_f16(
            &mut xa,
            &exec.to_device(&proj).unwrap(),
            &dbias,
            &dls,
            &dw,
            &db,
            &mut oa,
            rows,
            n,
            1e-5,
        )
        .unwrap();
        let mut xb = exec.to_device(&x).unwrap();
        let mut ob = exec.alloc_f16(rows * n).unwrap();
        exec.dp_res_ls_ln_h(
            &mut xb,
            &exec.f16_to_device(&halves(&proj)).unwrap(),
            &dbias,
            &dls,
            &dw,
            &db,
            &mut ob,
            rows,
            n,
            1e-5,
        )
        .unwrap();
        let (ra, rb) = (exec.to_host(&xa).unwrap(), exec.to_host(&xb).unwrap());
        assert!(
            ra.iter().zip(&rb).all(|(p, q)| p.to_bits() == q.to_bits()),
            "n={n}: the residual differs between the f32 and the half seam"
        );
        same_bits(
            &format!("res_ls_ln_h n={n}"),
            &exec.to_host_f16_len(&ob, rows * n).unwrap(),
            &exec.to_host_f16_len(&oa, rows * n).unwrap(),
        );
    }

    // ---- bias + GELU in place: an even plane, an odd row width (the pair
    // straddles a row end and the bias index wraps), and an odd total (the
    // last element takes the scalar arm) ----
    for (rows, n) in [(6usize, 64usize), (4, 7), (3, 5)] {
        let mut seed = 3u64 + (rows * n) as u64;
        let x = fill_h(&mut seed, rows * n, 6.0);
        let bias = fill(&mut seed, n);
        let dbias = exec.to_device(&bias).unwrap();
        let mut want = exec.alloc_f16(rows * n).unwrap();
        exec.gelu_erf_bias_f16(&exec.to_device(&x).unwrap(), &dbias, &mut want, rows, n)
            .unwrap();
        let mut got = exec.f16_to_device(&halves(&x)).unwrap();
        exec.dp_gelu_bias_h(&mut got, &dbias, rows, n).unwrap();
        same_bits(
            &format!("gelu_bias_h {rows}x{n}"),
            &exec.to_host_f16_len(&got, rows * n).unwrap(),
            &exec.to_host_f16_len(&want, rows * n).unwrap(),
        );
        // and against the plain definition, so the twins are not wrong together
        let g = exec.to_host_f16_len(&got, rows * n).unwrap();
        for r in 0..rows {
            for i in 0..n {
                let v = gelu(x[r * n + i] as f64 + bias[i] as f64);
                assert!(
                    (g[r * n + i].to_f64() - v).abs() < 2e-3 * (1.0 + v.abs()),
                    "gelu_bias_h {rows}x{n} [{r}][{i}]: {} vs {v}",
                    g[r * n + i]
                );
            }
        }
    }
}
