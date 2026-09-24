//! Shared flow-matching schedule and timestep embeddings.

/// `scheduler_config.json`: dynamic exponential shift between these
/// image-token counts and shifts, the schedule stretched to end at
/// `shift_terminal`.
const BASE_SEQ_LEN: f32 = 256.0;
const MAX_SEQ_LEN: f32 = 8192.0;
const BASE_SHIFT: f32 = 0.5;
const MAX_SHIFT: f32 = 0.9;
const SHIFT_TERMINAL: f32 = 0.02;

/// `calculate_shift` for a target of `n_tokens` latent tokens - linear in the
/// token count, no clamp (2K extrapolates past `max_image_seq_len`).
pub fn mu_for_tokens(n_tokens: usize) -> f32 {
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN);
    let b = BASE_SHIFT - m * BASE_SEQ_LEN;
    n_tokens as f32 * m + b
}

/// The per-step sigmas, `steps + 1` long with the terminal 0.
pub struct Schedule {
    pub sigmas: Vec<f32>,
}

impl Schedule {
    pub fn new(steps: usize, mu: f32) -> Self {
        // np.linspace(1.0, 1/steps, steps) in f64, then astype(float32)
        let n = steps.max(1);
        let mut s: Vec<f32> = (0..n)
            .map(|i| {
                let start = 1.0f64;
                let stop = 1.0f64 / n as f64;
                let t = if n == 1 {
                    start
                } else {
                    start + (stop - start) * (i as f64 / (n - 1) as f64)
                };
                t as f32
            })
            .collect();
        // exponential time shift: e^mu / (e^mu + (1/t - 1))
        let emu = (mu as f64).exp() as f32;
        for t in &mut s {
            *t = emu / (emu + (1.0 / *t - 1.0));
        }
        // stretch so the last sigma lands on shift_terminal. One step is the
        // degenerate case: its only sigma is both the first (pinned at 1) and
        // the last, `1 - s[n-1]` is 0, and the reference formula divides 0 by
        // 0 - NaN, which then rides through the timestep embedding, the
        // modulation and every layernorm into an all-black image (the
        // one-step renders at every size came back that way before this
        // guard). One step means one Euler step from pure noise to 0, so the
        // sigma stays at 1.
        let last = 1.0 - s[n - 1];
        if last > 0.0 {
            let scale = last / (1.0 - SHIFT_TERMINAL);
            for t in &mut s {
                *t = 1.0 - (1.0 - *t) / scale;
            }
        }
        s.push(0.0);
        Self { sigmas: s }
    }
}

/// `QwenImage21TemporalTimesteps`: 256 dims, cos half then sin half,
/// max_period 10000, the [0, 1] sigma scaled by 1000 first. torch builds
/// `freqs = exp(-ln(10000) * arange(half) / half)` in f32.
pub fn timestep_embedding(sigma: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let t = sigma * 1000.0;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (-(10000.0f32).ln() * i as f32 / half as f32).exp();
        let a = t * freq;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 0/0 the reference's stretch produces at one step must not reach
    /// the model: one step is [1, 0], pure noise to the image in one Euler
    /// move.
    #[test]
    fn one_step_is_one_euler_step_from_pure_noise() {
        let s = Schedule::new(1, mu_for_tokens(4096));
        assert_eq!(s.sigmas, vec![1.0, 0.0]);
    }

    /// Every other length: starts at 1, the last sigma before the appended
    /// 0 is the terminal, strictly decreasing, all finite - at the 1024^2
    /// shift and at the 2K one the shift extrapolates to.
    #[test]
    fn the_schedule_starts_at_one_and_ends_at_the_terminal() {
        for tokens in [4096usize, 16384] {
            for steps in [2usize, 4, 20, 40] {
                let s = Schedule::new(steps, mu_for_tokens(tokens));
                assert_eq!(s.sigmas.len(), steps + 1);
                assert!(
                    s.sigmas.iter().all(|v| v.is_finite()),
                    "{steps}: {:?}",
                    s.sigmas
                );
                assert!((s.sigmas[0] - 1.0).abs() < 1e-6, "{steps}: {:?}", s.sigmas);
                assert!(
                    (s.sigmas[steps - 1] - SHIFT_TERMINAL).abs() < 1e-5,
                    "{steps}: {:?}",
                    s.sigmas
                );
                assert_eq!(s.sigmas[steps], 0.0);
                assert!(
                    s.sigmas.windows(2).all(|w| w[0] > w[1]),
                    "{steps}: {:?}",
                    s.sigmas
                );
            }
        }
    }
}
