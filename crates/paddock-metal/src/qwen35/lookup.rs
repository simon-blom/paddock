//! Cheap copy proposals compete with the neural drafter, not target decoding.
//! The ordinary transactional verifier still decides every committed token.
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

const GRAM: usize = 6;
pub(super) const MAX_DRAFT: usize = 5;

#[derive(Default)]
pub(super) struct Lookup {
    history: Vec<u32>,
    index: HashMap<[u32; GRAM], (usize, usize)>,
    rejected: VecDeque<([u32; GRAM], usize)>,
    candidate: Option<[u32; GRAM]>,
    neural_rate: Option<f64>,
    round: Option<(Instant, Option<[u32; GRAM]>)>,
}

impl Lookup {
    fn push(&mut self, token: u32) {
        self.history.push(token);
        let end = self.history.len();
        if let Some(&gram) = self.history.last_chunk::<GRAM>() {
            let entry = self.index.entry(gram).or_default();
            *entry = (end, entry.0);
        }
    }

    pub(super) fn propose(&mut self, committed: &[u32], pending: u32, budget: usize) -> Vec<u32> {
        self.round = None;
        self.candidate = None;
        // Restores, slot reuse and cancellation can replace the history. Check
        // the complete prefix, not just its length or final token. Extension
        // updates the index only for new tokens; the prefix comparison does
        // not rebuild or search the n-gram index for every generated token.
        if !committed.starts_with(&self.history) {
            self.history.clear();
            self.index.clear();
            self.rejected.clear();
            self.neural_rate = None;
        }
        for &token in &committed[self.history.len()..] {
            self.push(token);
        }
        self.push(pending);
        let end = self.history.len();
        if budget < 2 {
            return Vec::new();
        }
        let Some(&gram) = self.history.last_chunk::<GRAM>() else {
            return Vec::new();
        };
        self.rejected.retain(|(_, until)| *until > end);
        if self.rejected.iter().any(|(key, _)| *key == gram) {
            return Vec::new();
        }
        let (_, previous) = self.index[&gram];
        if previous < GRAM || previous >= end {
            return Vec::new();
        }
        let count = budget.min(MAX_DRAFT).min(end - previous);
        if count < 2 {
            return Vec::new();
        }
        self.candidate = Some(gram);
        self.history[previous..previous + count].to_vec()
    }

    pub(super) fn begin(&mut self, lookup: bool) {
        self.round = Some((Instant::now(), if lookup { self.candidate } else { None }));
    }

    pub(super) fn cancel(&mut self) {
        self.round = None;
    }

    pub(super) fn commit(&mut self, tokens: usize) {
        if let Some((start, key)) = self.round.take() {
            self.observe(key, tokens, start.elapsed().as_secs_f64());
        }
    }

    fn observe(&mut self, key: Option<[u32; GRAM]>, tokens: usize, seconds: f64) {
        if seconds <= 0. || !seconds.is_finite() {
            return;
        }
        let rate = tokens as f64 / seconds;
        if let Some(key) = key {
            // A failed continuation must not suppress unrelated copy matches.
            // Bound both the rejection table and its token lifetime; changes
            // of history/slot clear it. Target verification remains mandatory.
            if tokens < 2 || self.neural_rate.is_some_and(|neural| rate < neural * 1.05) {
                self.rejected.retain(|(k, _)| *k != key);
                if self.rejected.len() == 32 {
                    self.rejected.pop_front();
                }
                self.rejected
                    .push_back((key, self.history.len().saturating_add(64)));
            }
        } else {
            self.neural_rate = Some(self.neural_rate.map_or(rate, |old| old * 0.8 + rate * 0.2));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_histories_and_small_budgets_keep_the_index_consistent() {
        for length in 0..GRAM {
            let committed: Vec<_> = (0..length as u32).collect();
            for budget in 0..=MAX_DRAFT {
                let mut lookup = Lookup::default();
                assert!(lookup.propose(&committed, 99, budget).is_empty());
                let expected: Vec<_> = committed.iter().copied().chain([99]).collect();
                assert_eq!(lookup.history, expected);
                assert_eq!(lookup.index.len(), usize::from(length + 1 == GRAM));
                if length + 1 == GRAM {
                    assert_eq!(lookup.index[&[0, 1, 2, 3, 4, 99]], (GRAM, 0));
                }
                assert!(lookup.candidate.is_none());
            }
        }
    }

    #[test]
    fn copies_only_prior_continuations_and_respects_budget() {
        let prompt = [1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 99, 1, 2, 3, 4, 5];
        for budget in 0..12 {
            let mut lookup = Lookup::default();
            let actual = lookup.propose(&prompt, 6, budget);
            assert_eq!(
                actual,
                if budget < 2 {
                    vec![]
                } else {
                    vec![20, 21, 22, 23, 24][..budget.min(MAX_DRAFT)].to_vec()
                }
            );
        }
        let mut lookup = Lookup::default();
        assert!(lookup.propose(&[1, 2, 3, 4, 5], 6, 5).is_empty());
    }

    #[test]
    fn restores_same_length_histories_and_slot_reuse_do_not_keep_old_matches() {
        let prompt = [1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 99, 1, 2, 3, 4, 5];
        let mut lookup = Lookup::default();
        assert_eq!(lookup.propose(&prompt, 6, 5), [20, 21, 22, 23, 24]);
        let mut replacement = prompt.to_vec();
        replacement[6..11].copy_from_slice(&[30, 31, 32, 33, 34]);
        assert_eq!(lookup.propose(&replacement, 6, 5), [30, 31, 32, 33, 34]);
        assert!(lookup.propose(&[], 6, 5).is_empty());
        assert!(lookup.index.is_empty());
    }

    #[test]
    fn rejected_or_expensive_copies_cool_down_but_cheap_copies_continue() {
        let mut lookup = Lookup::default();
        let key = Some([1, 2, 3, 4, 5, 6]);
        lookup.observe(None, 2, 0.05);
        lookup.observe(key, 6, 0.08);
        assert!(lookup.rejected.is_empty());
        lookup.observe(key, 1, 0.08);
        assert_eq!(lookup.rejected.len(), 1);
        lookup.rejected.clear();
        lookup.observe(key, 3, 0.15);
        assert_eq!(lookup.rejected.len(), 1);
        lookup.candidate = key;
        lookup.begin(true);
        lookup.cancel();
        lookup.commit(6);
        assert_eq!(lookup.rejected.len(), 1);
        for i in 0..100 {
            lookup.observe(Some([i; GRAM]), 1, 0.08);
        }
        assert_eq!(lookup.rejected.len(), 32);
    }

    #[test]
    fn failed_copy_does_not_disable_unrelated_matches() {
        let mut lookup = Lookup::default();
        let mut history = vec![
            11, 12, 13, 14, 15, 16, 30, 31, 32, 33, 34, 99, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24,
            99, 1, 2, 3, 4, 5,
        ];
        assert_eq!(lookup.propose(&history, 6, 5), [20, 21, 22, 23, 24]);
        lookup.observe(lookup.candidate, 1, 0.08);
        history.push(6);
        history.extend([1, 2, 3, 4, 5]);
        assert_eq!(lookup.propose(&history, 6, 5), Vec::<u32>::new());
        history.push(6);
        history.extend([11, 12, 13, 14, 15]);
        assert_eq!(lookup.propose(&history, 16, 5), [30, 31, 32, 33, 34]);
        history.push(16);
        history.extend([0; 64]);
        history.extend([1, 2, 3, 4, 5]);
        assert!(!lookup.propose(&history, 6, 5).is_empty());
        assert!(lookup.rejected.is_empty());
    }
}
