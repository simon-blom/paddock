//! The chunked-prefill queue and the tick rules every mixed path shares
//! (split mixed tick, fused unified tick, spec-in-mixed round).
//!
//! A prompt that fits the tick runs WHOLE through the coalesced batch pass -
//! the whole-prompt ticks every gemma4 election was measured on. A prompt
//! LONGER than the scheduler's per-tick budget advances a bounded span per
//! tick instead (Sarathi-Serve's intra-prompt chunking, what vLLM's
//! max_num_batched_tokens and SGLang's chunked_prefill_size do): its global
//! KV and SWA window ring persist in the slot between ticks exactly as they
//! do between a pass's own PF_ROWS chunks, so a span boundary changes when
//! rows run, never what they compute. Before this every path took the head
//! prompt whole, and a 150K-token prompt was one tick: every live stream
//! waited for all of it, and no per-tick budget - the scheduler's tick pacer
//! (crate::pacing) included - could bound it.

use crate::gpu::GpuError;

use super::batch::mixed_tick_rows;

/// A queued chunked prefill.
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    pub tokens: Vec<u32>,
    /// Prefix-resume point, once the admission prologue ran (see
    /// `chunk_prologue`). The prologue runs
    /// when the prompt is first picked, NOT at prefill_begin: a burst that
    /// shares a prefix resumes off whatever its siblings have landed by
    /// then (gemma4 has no in-flight prefix dedupe in the scheduler).
    pub start: Option<usize>,
    /// Next row to compute (meaningful once `start` is set).
    pub cursor: usize,
}

/// One prompt's rows in a mixed tick: `tokens[from..to]` on `slot`.
pub(crate) struct PfShare {
    pub slot: usize,
    pub tokens: Vec<u32>,
    /// prefix-resume point - the checkpoint cut at finish is measured from it
    pub start: usize,
    pub from: usize,
    pub to: usize,
}

impl PfShare {
    /// This share completes its prompt (its last row gets the logits).
    pub(crate) fn finishes(&self) -> bool {
        self.to == self.tokens.len()
    }
}

impl super::GpuGemma4 {
    /// The admission prologue of queued prompt `qi` - fresh-sequence clear,
    /// prefix adopt, global-row backing for the whole prompt - once, the
    /// first time it is picked. Fallible (the pool can run dry); a failed
    /// prologue leaves `start` unset, so the retry runs it from scratch.
    fn chunk_prologue(&mut self, qi: usize) -> Result<(), GpuError> {
        if self.chunked[qi].start.is_some() {
            return Ok(());
        }
        let slot = self.chunked[qi].slot;
        assert!(
            slot < self.n_slots,
            "slot {slot} >= enabled {}",
            self.n_slots
        );
        // borrow the tokens out of the queue for the radix walk
        let toks = std::mem::take(&mut self.chunked[qi].tokens);
        self.gpool_clear_slot(slot);
        let res = self.prefix_resume(slot, &toks).and_then(|start| {
            self.ensure_global_rows(&[slot as u32], &[(toks.len() - 1) as u32])
                .map(|_| start)
        });
        let c = &mut self.chunked[qi];
        c.tokens = toks;
        let start = res?;
        c.start = Some(start);
        c.cursor = start;
        Ok(())
    }

    /// This tick's prefill shares, FIFO, PEEKED - nothing commits until the
    /// pass succeeds ([`Self::chunk_commit`]; a destructive take once lost
    /// queue entries on PoolExhausted and wedged the serve).
    ///
    /// - A prompt whose uncached rows fit `budget` (the scheduler's per-tick
    ///   token budget) runs whole, and prompts after the first join only
    ///   whole, within `cap` rows counted at their FULL length - the rules
    ///   every path already had. Full length, not uncached rows, also keeps
    ///   the pick independent of which prologues have run, which is what
    ///   makes [`Self::prefill_prepare_impl`] exact.
    /// - A head prompt whose uncached rows exceed `budget` runs `cap` rows
    ///   from its cursor, alone in its tick, until it finishes. The test is
    ///   on the prompt's uncached length, not on what is left of it, so a
    ///   prompt that started spanning keeps spanning: its last tick is no
    ///   fatter than the others.
    ///
    /// Runs the admission prologue of every prompt it picks.
    pub(crate) fn chunk_pick(
        &mut self,
        budget: usize,
        cap: usize,
    ) -> Result<Vec<PfShare>, GpuError> {
        let cap = cap.max(1);
        let mut shares = Vec::new();
        let mut used = 0usize;
        for qi in 0..self.chunked.len() {
            let next = self.chunked[qi].tokens.len();
            if qi > 0 && used + next > cap {
                break;
            }
            self.chunk_prologue(qi)?;
            let c = &self.chunked[qi];
            let start = c.start.expect("prologue ran");
            let (from, len) = (c.cursor, c.tokens.len());
            let spans = qi == 0 && len - start > budget;
            shares.push(PfShare {
                slot: c.slot,
                tokens: c.tokens.clone(),
                start,
                from,
                to: if spans { (from + cap).min(len) } else { len },
            });
            used += next;
            if spans {
                break;
            }
        }
        Ok(shares)
    }

    /// Commit a tick's shares once its pass has succeeded: finished prompts
    /// leave the queue, a spanning head keeps its advanced cursor. Shares are
    /// a queue prefix in order, and only the last can be unfinished.
    pub(crate) fn chunk_commit(&mut self, shares: &[PfShare]) {
        let n_fin = shares.iter().take_while(|s| s.finishes()).count();
        debug_assert!(n_fin + 1 >= shares.len(), "only the last share spans");
        if let Some(sh) = shares.get(n_fin) {
            debug_assert_eq!(self.chunked[n_fin].slot, sh.slot);
            self.chunked[n_fin].cursor = sh.to;
        }
        self.chunked.drain(..n_fin);
    }

    /// `Generator::prefill_abort`: drop slot `slot`'s queued prefill (the
    /// client hung up) - a spanning prompt would otherwise keep the tick busy
    /// until it finished. Refused (false) while a spec-in-mixed round is in
    /// flight, whose shares index the queue; the scheduler retries next tick.
    /// The slot's pool blocks return through `release_inactive_slots`, and
    /// its next admission runs a fresh prologue.
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        if self.mix_inflight.is_some() {
            return false;
        }
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    /// `Generator::prefill_prepare`: run the admission prologue of every
    /// prompt a tick under `budget` will pick (the default tick cap), so the
    /// queue view prices them at their real depth. A tick given a smaller
    /// budget picks a prefix of these (the rules are monotone in both
    /// numbers), so nothing a tick runs was left unresolved.
    pub(crate) fn prefill_prepare_impl(&mut self, budget: usize) -> Result<(), GpuError> {
        self.chunk_pick(budget, budget.clamp(1, mixed_tick_rows()))
            .map(|_| ())
    }

    /// `Generator::prefill_queue`: the resolved prompts, from their cursor.
    /// One whose prologue has not run is left out - its depth is unknown
    /// until its prefix match, and no tick runs it before
    /// [`Self::prefill_prepare_impl`] (or the tick itself) resolves it.
    pub(crate) fn prefill_queue_impl(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .filter(|c| c.start.is_some())
            .map(|c| (c.slot, c.cursor, c.tokens.len() - c.cursor))
            .collect()
    }
}
