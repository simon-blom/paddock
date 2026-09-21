//! CPU test reference for the blockwise Walsh-Hadamard rotation of the Bonsai
//! checkpoints (`prism.hadamard.*`, version 1).
//!
//! The weights of such a file are stored in a rotated basis, `W' = W * R^T`
//! with `R = H * diag(s)` per block of the input dimension, so a runtime has
//! to feed each rotated matmul `x' = H(s * x)`. `H` is the normalized
//! Sylvester matrix, `H[r][c] = (-1)^popcount(r & c) / sqrt(n)`, its own
//! inverse; `s` is a fixed +-1 vector over the whole input width. A table
//! consumed by row lookup (the token embedding) stores rotated rows and comes
//! back through the inverse, `h = s * H(z)`.
//!
//! The transform here is the in-place butterfly, stages h = 1, 2, 4, ..., low
//! element `a + b`, high element `a - b`, with the normalization multiplied
//! in on the way in. Every output is then one fixed tree of f32 additions, so
//! a GPU kernel that keeps the stage order agrees with this module bit for
//! bit however it spreads the work - the gate is identity, not a tolerance.

/// Head geometry of the gated-delta-net output projection's input, for files
/// that set `prism.hadamard.gdn_v_grouped`: the engine lays value heads out
/// tiled (`head = k + n_k * r`), the rotated weight expects them grouped
/// (`head = r + rep * k`). Whole heads move, `head_dim` elements at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnHeads {
    pub head_dim: usize,
    pub n_k: usize,
    pub rep: usize,
}

impl GdnHeads {
    /// Source element (tiled order) of grouped-order element `i`.
    pub fn tiled_index(&self, i: usize) -> usize {
        let (head, off) = (i / self.head_dim, i % self.head_dim);
        let (k, r) = (head / self.rep, head % self.rep);
        (k + self.n_k * r) * self.head_dim + off
    }
}

/// The butterfly over one block, normalization included.
fn transform_block(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two(),
        "Hadamard block {n} is not a power of two"
    );
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
    let mut h = 1;
    while h < n {
        for base in (0..n).step_by(2 * h) {
            for j in base..base + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// `x' = H(s * x)` over each `block`-wide strip of every `width`-wide row:
/// what a rotated matmul's input goes through. `signs` is `width` long; `gdn`
/// applies the tiled-to-grouped head permutation first.
pub fn rotate_rows(
    x: &[f32],
    width: usize,
    block: usize,
    signs: &[f32],
    gdn: Option<GdnHeads>,
) -> Vec<f32> {
    assert!(width.is_multiple_of(block) && x.len().is_multiple_of(width));
    assert_eq!(signs.len(), width);
    let mut out = vec![0f32; x.len()];
    for (row, dst) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        for (i, v) in dst.iter_mut().enumerate() {
            let src = gdn.map_or(i, |g| g.tiled_index(i));
            *v = row[src] * signs[i];
        }
        for strip in dst.chunks_exact_mut(block) {
            transform_block(strip);
        }
    }
    out
}

/// `h = s * H(z)`: a looked-up row of a rotated table, back in the model's
/// own basis.
pub fn unrotate_rows(z: &[f32], width: usize, block: usize, signs: &[f32]) -> Vec<f32> {
    assert!(width.is_multiple_of(block) && z.len().is_multiple_of(width));
    assert_eq!(signs.len(), width);
    let mut out = z.to_vec();
    for row in out.chunks_exact_mut(width) {
        for strip in row.chunks_exact_mut(block) {
            transform_block(strip);
        }
        for (v, s) in row.iter_mut().zip(signs) {
            *v *= s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn signs(n: usize, seed: u32) -> Vec<f32> {
        noise(n, seed)
            .iter()
            .map(|v| if *v < 0.0 { -1.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn the_butterfly_is_the_sylvester_matrix() {
        let n = 64;
        let x = noise(n, 7);
        let got = rotate_rows(&x, n, n, &vec![1.0; n], None);
        for (r, &have) in got.iter().enumerate() {
            let want: f64 = (0..n)
                .map(|c| {
                    let sign = if (r & c).count_ones() % 2 == 1 {
                        -1.0
                    } else {
                        1.0
                    };
                    sign * x[c] as f64 / (n as f64).sqrt()
                })
                .sum();
            assert!(
                (have as f64 - want).abs() < 1e-6,
                "row {r}: {have} vs {want}"
            );
        }
    }

    #[test]
    fn the_lookup_inverse_undoes_the_rotation() {
        let (width, block) = (5120, 1024);
        let s = signs(width, 3);
        let x = noise(2 * width, 11);
        let back = unrotate_rows(&rotate_rows(&x, width, block, &s, None), width, block, &s);
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn a_rotated_dot_product_is_the_plain_one() {
        // w' = H(s * w) is how a row of a folded weight is stored; feeding it
        // x' = H(s * x) has to give back w . x, because H is orthogonal
        let (width, block) = (2048, 1024);
        let s = signs(width, 5);
        let (w, x) = (noise(width, 1), noise(width, 2));
        let plain: f64 = w.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let (wr, xr) = (
            rotate_rows(&w, width, block, &s, None),
            rotate_rows(&x, width, block, &s, None),
        );
        let rotated: f64 = wr.iter().zip(&xr).map(|(a, b)| *a as f64 * *b as f64).sum();
        assert!((plain - rotated).abs() < 1e-4, "{plain} vs {rotated}");
    }

    #[test]
    fn gdn_heads_move_whole_from_tiled_to_grouped() {
        // Qwen3.8-27B: 48 value heads of 128 in 16 key groups, 3 to a group
        let g = GdnHeads {
            head_dim: 128,
            n_k: 16,
            rep: 3,
        };
        // grouped head 1 is group 0's second repeat, which tiled order keeps
        // at head 0 + 16 * 1
        assert_eq!(g.tiled_index(128), 16 * 128);
        assert_eq!(g.tiled_index(128 + 5), 16 * 128 + 5);
        // grouped head 3 is group 1's first repeat: tiled head 1
        assert_eq!(g.tiled_index(3 * 128), 128);
        let mut seen = vec![false; 48 * 128];
        for i in 0..48 * 128 {
            seen[g.tiled_index(i)] = true;
        }
        assert!(
            seen.iter().all(|&b| b),
            "the permutation covers every element once"
        );
    }
}
