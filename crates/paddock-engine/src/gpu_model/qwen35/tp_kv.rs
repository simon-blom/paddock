//! Isolated rank-0-authoritative logical paged KV lifecycle for GQA TP.
//!
//! Uses the production KvPool/BlockTable/PagedRadix bookkeeping. This module
//! carries no GPU payload and defines no production scheduler protocol. The
//! rank-0 side authorizes logical operations and hands the worker the ordered
//! operations plus the resulting end-of-tick state (one snapshot per tick -
//! see `authorize_all`/`mirror_tick`); the mirror replays the operations and
//! refuses any divergence before GPU work.
use crate::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use crate::paged_radix::PagedRadix;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Operation {
    Ensure { slot: usize, position: usize },
    Release { slot: usize },
    Publish { slot: usize, tokens: Vec<u32> },
    Reuse { slot: usize, tokens: Vec<u32> },
    Reset,
    Flush,
    /// Prefix-cache admission (TP resume): store `slot`'s full prompt tokens,
    /// release the slot's previous table, and on `resume > 0` adopt the cached
    /// prefix pages by refcount. Fails unless the radix holds a DeltaNet
    /// checkpoint at exactly `resume` under `tokens` - the coordinator's
    /// selection is validated independently on each rank, never trusted.
    /// `resume == 0` is a cold admission.
    Admit {
        slot: usize,
        tokens: Vec<u32>,
        resume: usize,
    },
    /// Claim a checkpoint pool index for the upcoming snapshot at cut
    /// `position` of `tokens` (which must be a cached node boundary). The
    /// index is reserved (not yet attached) so publication can happen after
    /// the GPU work succeeds; undo with `CheckpointRecycle`.
    CheckpointReserve { tokens: Vec<u32>, position: usize },
    /// Attach a previously reserved index at `position` of `tokens` (after
    /// the rank-local snapshot succeeded). Fails when nothing is reserved
    /// there.
    CheckpointAttach {
        tokens: Vec<u32>,
        position: usize,
        index: u32,
    },
    /// Return a reserved-but-unattached index to the free list.
    CheckpointRecycle { index: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    pub tables: Vec<Vec<u32>>,
    pub refcounts: Vec<u32>,
    pub free: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Event {
    pub sequence: u64,
    pub operation: Operation,
    pub state: Snapshot,
}

#[derive(Clone)]
pub struct MirroredKv {
    pool: KvPool,
    tables: Vec<BlockTable>,
    radix: PagedRadix,
    sequence: u64,
    max_ctx: usize,
    /// Each slot's full prompt tokens as admitted (`Admit`), so checkpoint
    /// reservations/publishes can name cuts by position without re-sending
    /// the prompt on every op. Release/reset clears the entry.
    tokens: Vec<Vec<u32>>,
    /// Checkpoint-pool capacity in indices (`set_state_capacity`); 0 = the
    /// cache runs pages-only (never resumes). Part of identity so a mirror
    /// with a different pool size fails closed at setup, not mid-tick.
    state_capacity: u32,
    /// The reuse position of the MOST RECENT apply of an `Admit`
    /// operation (0 = cold). Deliberately NOT mirrored state: it is the
    /// coordinator's accounting seam for the tick in flight, read
    /// immediately after `authorize`, exactly like the non-TP backend's
    /// `last_reused[slot]`. The worker never reads it, so no divergence is
    /// possible.
    admitted_reused: Option<usize>,
}

impl MirroredKv {
    pub fn new(blocks: u32, slots: usize, max_ctx: usize) -> Result<Self, &'static str> {
        if blocks == 0
            || slots == 0
            || slots > u32::MAX as usize
            || max_ctx == 0
            || max_ctx > u32::MAX as usize
            || max_ctx.div_ceil(BLOCK_TOKENS).checked_mul(slots).is_none()
        {
            return Err("invalid KV geometry");
        }
        Ok(Self {
            pool: KvPool::with_blocks(blocks),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            radix: PagedRadix::new(),
            sequence: 0,
            max_ctx,
            tokens: vec![Vec::new(); slots],
            state_capacity: 0,
            admitted_reused: None,
        })
    }

    /// Enable DeltaNet checkpointing: `n` pool indices. Call on both ranks
    /// with the SAME coordinator-resolved value (`TpInit` carries it) before
    /// any `CheckpointReserve`. Idempotent-ish: arms the radix free-list;
    /// a zero value keeps the cache pages-only.
    pub fn set_state_capacity(&mut self, n: u32) {
        self.state_capacity = n;
        self.radix.set_state_capacity(n);
    }

    /// The tokens `slot` was admitted with (empty = no live admission).
    pub fn slot_admitted_tokens(&self, slot: usize) -> &[u32] {
        self.tokens
            .get(slot)
            .map(|t| t.as_slice())
            .unwrap_or(&[])
    }

    /// The index reserved for `slot`'s checkpoint cut at `pos`, if any.
    pub fn slot_reserved_ckpt(&self, slot: usize, pos: usize) -> Option<u32> {
        self.radix.reserved_index(slot, pos)
    }

    /// The reuse position of the last applied `Admit`, consumed once
    /// (the accounting seam for the tick in flight).
    pub fn take_admitted_reused(&mut self) -> usize {
        self.admitted_reused.take().unwrap_or(0)
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            tables: self.tables.iter().map(|t| t.blocks().to_vec()).collect(),
            refcounts: (0..self.pool.capacity())
                .map(|b| self.pool.refcount(b))
                .collect(),
            free: self.pool.free_blocks(),
        }
    }

    pub fn device_table(&self) -> Vec<u32> {
        let bps = self.max_ctx.div_ceil(BLOCK_TOKENS);
        let mut host = vec![0; bps * self.tables.len()];
        for (i, t) in self.tables.iter().enumerate() {
            host[i * bps..i * bps + t.blocks().len()].copy_from_slice(t.blocks());
        }
        host
    }

    /// One slot's physical block ids in logical order (the lane->decode
    /// promotion walks exactly these pool offsets on both ranks' slabs).
    pub fn slot_blocks(&self, slot: usize) -> Option<&[u32]> {
        self.tables.get(slot).map(|t| t.blocks())
    }

    /// Serialize only a live slot whose entire read prefix is backed by
    /// allocated pages, in precisely the geometry of the GPU payload.
    pub fn checked_device_table(
        &self,
        slot: usize,
        position: usize,
        pool_blocks: u32,
        slots: usize,
        max_ctx: usize,
    ) -> Result<Vec<u32>, &'static str> {
        if pool_blocks != self.pool.capacity()
            || slots != self.tables.len()
            || max_ctx != self.max_ctx
            || position >= max_ctx
        {
            return Err("KV GPU geometry mismatch");
        }
        let t = self.tables.get(slot).ok_or("KV slot out of range")?;
        let needed = position / BLOCK_TOKENS + 1;
        if t.blocks().len() < needed
            || t.blocks()[..needed]
                .iter()
                .any(|&b| b >= pool_blocks || self.pool.refcount(b) == 0)
        {
            return Err("KV position has no live pages");
        }
        Ok(self.device_table())
    }

    /// Rank 0 emits an event only after the operation has succeeded.
    pub fn authorize(&mut self, operation: Operation) -> Result<Event, &'static str> {
        self.apply(&operation)?;
        self.sequence += 1;
        Ok(Event {
            sequence: self.sequence,
            operation,
            state: self.snapshot(),
        })
    }

    /// Rank 0 applies an ordered tick of operations and snapshots ONCE at the
    /// end (upstream-readiness B2): the per-row snapshots the old design
    /// authorized grew the wire O(rows x context); the resulting end state is
    /// the only thing the mirror needs to validate, because every apply is a
    /// deterministic function of the (already-agreed) prior state and the
    /// ordered operations. Stage the full tick and commit only after every
    /// operation succeeds, so an invalid later operation cannot half-apply.
    pub fn authorize_all(&mut self, operations: &[Operation]) -> Result<Snapshot, &'static str> {
        let mut staged = self.clone();
        for op in operations {
            staged.apply(op)?;
        }
        staged.sequence += operations.len() as u64;
        let snapshot = staged.snapshot();
        *self = staged;
        Ok(snapshot)
    }

    /// Rank 1 replays a rank-0 event; mismatched sequence or state fails closed.
    pub fn mirror(&mut self, event: &Event) -> Result<(), &'static str> {
        if event.sequence != self.sequence + 1 {
            return Err("KV event out of order");
        }
        self.apply(&event.operation)?;
        self.sequence += 1;
        if self.snapshot() != event.state {
            return Err("KV logical state diverged");
        }
        Ok(())
    }

    /// Rank 1 replays an ordered tick of rank-0 operations and requires the
    /// resulting state to equal `end_state` (upstream-readiness B2). The
    /// sequence advances by exactly `operations.len()`, matching the
    /// coordinator's `authorize_all`. Deterministic applies make the
    /// end-state equality a complete divergence check: if the prior states
    /// matched (validated by the previous tick's comparison) and the
    /// operations match, any mid-tick divergence necessarily shows in the
    /// end state. Replay/stale ordering is enforced one layer up - the
    /// worker's control loop requires each message's sequence to be exactly
    /// its counter + 1 before the mirror ever sees the tick. Fails closed on
    /// any mismatch, before GPU work.
    pub fn mirror_tick(
        &mut self,
        operations: &[Operation],
        end_state: &Snapshot,
    ) -> Result<(), &'static str> {
        if operations.is_empty() {
            // An empty tick is never a legitimate message; refusing it keeps
            // a malformed frame from reading as silent progress.
            return Err("KV tick carries no operations");
        }
        let mut staged = self.clone();
        for op in operations {
            staged.apply(op)?;
        }
        staged.sequence += operations.len() as u64;
        if staged.snapshot() != *end_state {
            return Err("KV logical state diverged");
        }
        *self = staged;
        Ok(())
    }

    fn apply(&mut self, op: &Operation) -> Result<(), &'static str> {
        match op {
            Operation::Ensure { slot, position } => {
                if *position >= self.max_ctx {
                    return Err("KV position out of range");
                }
                let t = self.tables.get_mut(*slot).ok_or("KV slot out of range")?;
                let needed = *position / BLOCK_TOKENS + 1;
                if needed.saturating_sub(t.blocks().len()) > self.pool.free_blocks() {
                    return Err("KV pool exhausted");
                }
                t.ensure(*position, &mut self.pool)
                    .map_err(|_| "KV pool exhausted")?;
            }
            Operation::Release { slot } => {
                self.tables
                    .get_mut(*slot)
                    .ok_or("KV slot out of range")?
                    .clear(&mut self.pool);
                // Reserved-but-unattached checkpoint indices die with the
                // admission that owned them; attached ones stay in the radix.
                self.radix.drop_slot_reservations(*slot);
                if let Some(toks) = self.tokens.get_mut(*slot) {
                    toks.clear();
                }
            }
            Operation::Publish { slot, tokens } => {
                if tokens.len() > self.max_ctx {
                    return Err("KV tokens out of range");
                }
                let t = self.tables.get(*slot).ok_or("KV slot out of range")?;
                if tokens.len() / BLOCK_TOKENS > t.blocks().len() {
                    return Err("KV prefix not backed");
                }
                self.radix.insert(tokens, t.blocks(), &mut self.pool);
            }
            Operation::Reuse { slot, tokens } => {
                let t = self.tables.get_mut(*slot).ok_or("KV slot out of range")?;
                if !t.blocks().is_empty() || tokens.len() > self.max_ctx {
                    return Err("KV reuse needs empty slot");
                }
                let blocks = self.radix.match_prefix(tokens);
                t.share_prefix(&blocks, &mut self.pool);
            }
            Operation::Reset => {
                for t in &mut self.tables {
                    t.clear(&mut self.pool);
                }
                for t in &mut self.tokens {
                    t.clear();
                }
                self.radix.recycle_all_reserved();
                // Cache retains published blocks; reset releases slots, not the cache.
            }
            Operation::Flush => {
                for t in &mut self.tables {
                    t.clear(&mut self.pool);
                }
                for t in &mut self.tokens {
                    t.clear();
                }
                self.radix.recycle_all_reserved();
                while self.radix.evict_lru(&mut self.pool).is_some() {}
            }
            Operation::Admit {
                slot,
                tokens,
                resume,
            } => {
                if *slot >= self.tables.len() || tokens.len() > self.max_ctx || tokens.is_empty() {
                    return Err("TP admit slot or prompt out of range");
                }
                let reused = if *resume > 0 {
                    // One authoritative adoption walk: both ranks require the
                    // cached chain AND its checkpoint at exactly the
                    // coordinator-chosen position. A rank that cannot satisfy
                    // it fails closed instead of resuming alone.
                    let depth = *resume / BLOCK_TOKENS;
                    let (blocks, ckpt) = self
                        .radix
                        .match_full_upto(tokens, depth)
                        .ok_or("TP admit resume chain not cached")?;
                    if ckpt.is_none() || resume % BLOCK_TOKENS != 0 {
                        return Err("TP admit resume checkpoint missing");
                    }
                    {
                        let t = self
                            .tables
                            .get_mut(*slot)
                            .ok_or("KV slot out of range")?;
                        t.clear(&mut self.pool);
                        t.share_prefix(&blocks, &mut self.pool);
                    }
                    *resume
                } else {
                    self.tables
                        .get_mut(*slot)
                        .ok_or("KV slot out of range")?
                        .clear(&mut self.pool);
                    0
                };
                self.tokens[*slot] = tokens.clone();
                self.admitted_reused = Some(reused);
            }
            Operation::CheckpointReserve { tokens, position } => {
                if tokens.len() > self.max_ctx {
                    return Err("TP checkpoint reserve tokens out of range");
                }
                let slot = self
                    .tokens
                    .iter()
                    .position(|t| t.as_slice() == tokens.as_slice())
                    .ok_or("TP checkpoint reserve for unadmitted prompt")?;
                self.radix
                    .reserve_state_for(slot, *position)
                    .ok_or("TP checkpoint reserve exhausted")?;
            }
            Operation::CheckpointAttach {
                tokens,
                position,
                index,
            } => {
                let slot = self
                    .tokens
                    .iter()
                    .position(|t| t.as_slice() == tokens.as_slice())
                    .ok_or("TP checkpoint attach for unadmitted prompt")?;
                if !self
                    .radix
                    .attach_reserved_at(slot, tokens, *position, *index)
                {
                    return Err("TP checkpoint attach has no reservation");
                }
            }
            Operation::CheckpointRecycle { index } => {
                // Reserved-only index: putting it straight back on the free
                // list is the exact inverse of alloc_state's pop.
                self.radix.recycle_state(*index);
            }
        }
        Ok(())
    }
}

/// The pure resume decision (host-testable): given the deepest checkpoint
/// probe under the prompt and the prompt length, pick the resume position.
/// Mirrors the non-TP gate: block-aligned (the probe position always is),
/// at least two blocks deep, strictly inside the prompt, and either deep
/// enough to be worth the restore (`min_cache_prefix`) or on a narrow serve
/// (`slots <= resume_live_max`) where short resumes are still net-positive.
/// Rank 0 runs this ONCE; both ranks validate the exact selection in `Admit`.
pub fn tp_resume_decision(ckpt: Option<(usize, u32)>, t_len: usize, slots: usize) -> usize {
    let floor = if slots <= resume_live_max_tp() {
        2 * BLOCK_TOKENS
    } else {
        min_cache_prefix_tp().max(2 * BLOCK_TOKENS)
    };
    match ckpt {
        Some((pos, _)) if pos >= floor && pos >= 2 * BLOCK_TOKENS && pos < t_len => pos,
        _ => 0,
    }
}

/// Narrow-serve threshold for the TP resume gate (same seam as the non-TP
/// `resume_live_max`): below this configured slot count a short-prefix
/// resume still pays for itself. TP=2 serves are always narrow, so the
/// default (12) admits every resume the cache can serve.
fn resume_live_max_tp() -> usize {
    12
}

/// Minimum prefix worth a TP restore (same default as the non-TP
/// `min_cache_prefix`; a TP restore is two rank-local copies instead of page
/// adoption only, but the per-request state restore is the same order of
/// work). Env-overridable for sweeps.
fn min_cache_prefix_tp() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_MIN_CACHE_PREFIX")
            .and_then(|v| v.to_str().and_then(|s| s.parse().ok()))
            .unwrap_or(512)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_zero_lifecycle_and_prefix_mirror() {
        let mut a = MirroredKv::new(5, 2, 48).unwrap();
        let mut b = MirroredKv::new(5, 2, 48).unwrap();
        let tokens: Vec<u32> = (0..33).collect();
        for op in [
            Operation::Ensure {
                slot: 0,
                position: 31,
            },
            Operation::Publish {
                slot: 0,
                tokens: tokens[..32].to_vec(),
            },
            Operation::Release { slot: 0 },
            Operation::Reuse {
                slot: 1,
                tokens: tokens.clone(),
            },
            Operation::Ensure {
                slot: 1,
                position: 32,
            },
            Operation::Release { slot: 1 },
            Operation::Reset,
            Operation::Reuse { slot: 0, tokens },
        ] {
            let event = a.authorize(op).unwrap();
            b.mirror(&event).unwrap();
            assert_eq!(a.device_table(), b.device_table());
            assert_eq!(a.snapshot(), b.snapshot());
        }
        assert!(b.mirror(&a.authorize(Operation::Reset).unwrap()).is_ok());
        assert_eq!(a.snapshot().free, 3); // two radix references survive slot reset
        let flushed = a.authorize(Operation::Flush).unwrap();
        b.mirror(&flushed).unwrap();
        assert_eq!(a.snapshot().free, 5);
        assert!(a.snapshot().refcounts.iter().all(|&rc| rc == 0));
        assert!(
            b.mirror(&Event {
                sequence: 99,
                operation: Operation::Reset,
                state: a.snapshot(),
            })
            .is_err()
        );
    }

    #[test]
    fn exhausted_grow_is_atomic_and_bad_mirror_fails_closed() {
        let mut a = MirroredKv::new(1, 2, 48).unwrap();
        let first = a
            .authorize(Operation::Ensure {
                slot: 0,
                position: 0,
            })
            .unwrap();
        let before = a.snapshot();
        assert!(a.checked_device_table(0, 0, 1, 2, 48).is_ok());
        assert!(a.checked_device_table(0, 16, 1, 2, 48).is_err());
        assert!(a.checked_device_table(0, 0, 2, 2, 48).is_err());
        assert!(
            a.authorize(Operation::Ensure {
                slot: 0,
                position: 32
            })
            .is_err()
        );
        assert_eq!(a.snapshot(), before);
        let mut b = MirroredKv::new(1, 2, 48).unwrap();
        let mut wrong = first.clone();
        wrong.state.free += 1;
        assert!(b.mirror(&wrong).is_err());
        let mut fresh = MirroredKv::new(1, 2, 48).unwrap();
        fresh.mirror(&first).unwrap();
        assert_eq!(fresh.snapshot(), before);
    }

    #[test]
    fn tick_authorize_and_mirror_round_trip_including_prefix_reuse() {
        let mut a = MirroredKv::new(5, 2, 48).unwrap();
        let mut b = MirroredKv::new(5, 2, 48).unwrap();
        // A span-shaped tick: rows across two slots, page-crossing positions.
        let tick = vec![
            Operation::Ensure {
                slot: 0,
                position: 14,
            },
            Operation::Ensure {
                slot: 0,
                position: 15,
            },
            Operation::Ensure {
                slot: 0,
                position: 16,
            },
            Operation::Ensure {
                slot: 1,
                position: 0,
            },
            Operation::Ensure {
                slot: 1,
                position: 1,
            },
        ];
        let end = a.authorize_all(&tick).unwrap();
        b.mirror_tick(&tick, &end).unwrap();
        assert_eq!(a.snapshot(), b.snapshot());
        assert_eq!(a.sequence, 5);
        assert_eq!(b.sequence, 5);
        // A second tick continues from the agreed state.
        let end2 = a
            .authorize_all(&[Operation::Ensure {
                slot: 0,
                position: 17,
            }])
            .unwrap();
        b.mirror_tick(
            &[Operation::Ensure {
                slot: 0,
                position: 17,
            }],
            &end2,
        )
        .unwrap();
        assert_eq!(a.snapshot(), b.snapshot());
    }

    #[test]
    fn mirror_tick_fails_closed_on_divergent_state_sequence_and_bad_ops() {
        let mut a = MirroredKv::new(4, 2, 48).unwrap();
        let ops = vec![
            Operation::Ensure {
                slot: 0,
                position: 0,
            },
            Operation::Ensure {
                slot: 0,
                position: 1,
            },
        ];
        let end = a.authorize_all(&ops).unwrap();

        // Divergent end state: same ops replayed on a smaller pool produce a
        // different layout -> the comparison must refuse.
        let mut b = MirroredKv::new(2, 2, 48).unwrap();
        assert!(b.mirror_tick(&ops, &end).is_err());

        // Correct replay, then a stale tick: sequence must advance by exactly
        // the operation count. NOTE: the replayed tick's Ensures are idempotent
        // (the pages already exist), so the mirror's state comparison alone
        // cannot reject a duplicated tick - replay/stale ordering is enforced
        // one layer up (the worker loop's message-sequence guard). Here we
        // verify the advance: the second tick still succeeds on the mirror's
        // own terms and lands two ticks ahead.
        let mut c = MirroredKv::new(4, 2, 48).unwrap();
        c.mirror_tick(&ops, &end).unwrap();
        assert_eq!(c.sequence, 2);
        c.mirror_tick(&ops, &end).unwrap();
        assert_eq!(c.sequence, 4);
        // Repeating the SAME tick content with a fresh end state on a fresh
        // mirror is fine (ops are authorized, not replay-guarded by content).
        let mut d = MirroredKv::new(4, 2, 48).unwrap();
        d.mirror_tick(&ops, &end).unwrap();
        assert_eq!(d.snapshot(), end);

        // An invalid operation inside the tick fails and leaves no partial
        // sequence advance on the coordinator either.
        let mut e = MirroredKv::new(1, 2, 48).unwrap();
        let before = e.snapshot();
        let before_sequence = e.sequence;
        let bad = vec![
            Operation::Ensure {
                slot: 0,
                position: 0,
            },
            Operation::Ensure {
                slot: 0,
                position: 64,
            },
        ];
        assert!(e.authorize_all(&bad).is_err());
        assert_eq!(e.snapshot(), before);
        assert_eq!(e.sequence, before_sequence);
        // An empty tick is malformed, never silent progress.
        assert!(e.mirror_tick(&[], &a.snapshot()).is_err());

        // A wrong END STATE with valid ops fails closed (the failure mode the
        // per-row design caught mid-tick: end-state equality still catches it).
        let mut f = MirroredKv::new(4, 2, 48).unwrap();
        let before = f.snapshot();
        let before_sequence = f.sequence;
        let mut wrong_end = end.clone();
        wrong_end.free += 1;
        assert!(f.mirror_tick(&ops, &wrong_end).is_err());
        assert_eq!(f.snapshot(), before);
        assert_eq!(f.sequence, before_sequence);
        // The failed tick did not consume the valid operations; replaying the
        // correct end state must still succeed from the original state.
        assert!(f.mirror_tick(&ops, &end).is_ok());
    }

    #[test]
    fn mixed_tick_release_and_flush_mirror_through_the_batch_api() {
        let mut a = MirroredKv::new(5, 2, 48).unwrap();
        let mut b = MirroredKv::new(5, 2, 48).unwrap();
        let tokens: Vec<u32> = (0..33).collect();
        // Prefill a slot, publish its prefix, then release it - the release
        // tick's end state must capture the returned blocks exactly.
        let prefill = vec![Operation::Ensure {
            slot: 0,
            position: 31,
        }];
        let end = a.authorize_all(&prefill).unwrap();
        b.mirror_tick(&prefill, &end).unwrap();
        let publish = a
            .authorize(Operation::Publish {
                slot: 0,
                tokens: tokens[..32].to_vec(),
            })
            .unwrap();
        b.mirror(&publish).unwrap();
        let release = vec![Operation::Release { slot: 0 }];
        let end = a.authorize_all(&release).unwrap();
        b.mirror_tick(&release, &end).unwrap();
        assert_eq!(a.snapshot(), b.snapshot());
        // A flush tick (the TpReset path's operation) mirrors identically.
        let flush = vec![Operation::Flush];
        let end = a.authorize_all(&flush).unwrap();
        b.mirror_tick(&flush, &end).unwrap();
        assert_eq!(a.snapshot(), b.snapshot());
        assert!(a.snapshot().refcounts.iter().all(|&rc| rc == 0));
    }
}
