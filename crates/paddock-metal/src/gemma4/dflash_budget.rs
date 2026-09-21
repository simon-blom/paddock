//! Acceptance-aware verification on the measured Muse/M5 row tiers.
//!
//! Like D-cut (arxiv.org/abs/2607.14647), optimize tokens advanced per unit
//! verification cost, not acceptance alone. This is a narrower election:
//! uniform 32/64-row c=4 batches, using *previous* commits instead of a trained
//! confidence head or current-token probabilities. Cross-request GPU utility
//! allocation remains the target. The noncausal drafter always runs 16 rows.

pub(super) struct Budget {
    full: u32,
    short: u32,
    observations: u32,
    short_rounds: u32,
}

impl Default for Budget {
    fn default() -> Self {
        // Fixed-point EMA of batch progress, including the certain anchor.
        Self {
            full: 64 * 256,
            short: 32 * 256,
            observations: 0,
            short_rounds: 0,
        }
    }
}

impl Budget {
    pub(super) fn depth(&self, live: usize) -> usize {
        // c=1 k7 falls onto slow <=8-row register kernels. c=2/3 and other
        // row tiers have not been elected. Full-block probes bound censoring:
        // a k7 success says nothing about whether positions 8..15 survive.
        // Measured full/short draft+verify cost is 1.67..1.68. Require <1.5x
        // progress before shortening, leaving headroom for switching/probes.
        if live == 4
            && self.observations >= 2
            && self.short_rounds < 8
            && self.full * 2 < self.short * 3
        {
            7
        } else {
            15
        }
    }

    pub(super) fn observe(&mut self, widths: impl Iterator<Item = usize>, counts: &[u32]) {
        if counts.len() != 4 {
            return;
        }
        let widths: Vec<_> = widths.collect();
        if widths == [16; 4] {
            let full: u32 = counts.iter().sum();
            let short: u32 = counts.iter().map(|&n| n.min(8)).sum();
            self.full = (self.full + full * 256) / 2;
            self.short = (self.short + short * 256) / 2;
            self.observations = self.observations.saturating_add(1);
            self.short_rounds = 0;
        } else if widths == [8; 4] {
            // Never learn an unobserved long suffix from a shortened round.
            // A fully accepted short batch merits an immediate full probe.
            self.short_rounds = if counts == [8; 4] {
                8
            } else {
                (self.short_rounds + 1).min(8)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_acceptance_keeps_full_depth() {
        let mut b = Budget::default();
        for _ in 0..100 {
            b.observe([16; 4].into_iter(), &[16; 4]);
            assert_eq!(b.depth(4), 15);
        }
    }

    #[test]
    fn low_acceptance_shortens_only_the_elected_batch() {
        let mut b = Budget::default();
        b.observe([16; 4].into_iter(), &[4; 4]);
        assert_eq!(b.depth(4), 15);
        b.observe([16; 4].into_iter(), &[4; 4]);
        assert_eq!(b.depth(4), 7);
        for live in [0, 1, 2, 3, 5, 8] {
            assert_eq!(b.depth(live), 15);
        }
        b = Budget::default();
        assert_eq!(b.depth(4), 15, "new admissions must start fresh");
    }

    #[test]
    fn censored_short_rounds_probe_and_recover() {
        let mut b = Budget::default();
        for _ in 0..2 {
            b.observe([16; 4].into_iter(), &[4; 4]);
        }
        for _ in 0..8 {
            assert_eq!(b.depth(4), 7);
            b.observe([8; 4].into_iter(), &[4; 4]);
        }
        assert_eq!(b.depth(4), 15);
        b.observe([16; 4].into_iter(), &[16; 4]);
        assert_eq!(b.depth(4), 15);
        for _ in 0..8 {
            b.observe([16; 4].into_iter(), &[1; 4]);
        }
        assert_eq!(b.depth(4), 7);
        b.observe([8; 4].into_iter(), &[8; 4]);
        assert_eq!(b.depth(4), 15, "fully accepted short batch probes now");
    }

    #[test]
    fn ragged_or_other_concurrency_does_not_train_full_depth_utility() {
        let mut b = Budget::default();
        for _ in 0..100 {
            b.observe([16; 3].into_iter(), &[1; 3]);
            b.observe([16, 16, 16, 8].into_iter(), &[1; 4]);
            b.observe([8; 4].into_iter(), &[1; 4]);
        }
        assert_eq!(b.observations, 0);
        assert_eq!(b.depth(4), 15);
    }
}
