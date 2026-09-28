//! Generic coordinator-authoritative logical paged-KV lifecycle for TP.
//!
//! Uses the production KvPool/BlockTable/PagedRadix bookkeeping. This module
//! carries no GPU payload and defines no model or transport protocol. The
//! coordinator authorizes ordered logical operations; every mirror replays
//! them and refuses divergence before GPU work.
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
    /// `position` of the admitted slot tokens (a future cached node boundary). The
    /// index is reserved (not yet attached) so publication can happen after
    /// the GPU work succeeds; undo with `CheckpointRecycle`.
    CheckpointReserve { slot: usize, position: usize },
    /// Attach a previously reserved index at `position` of `tokens` (after
    /// the rank-local snapshot succeeded). Fails when nothing is reserved
    /// there.
    CheckpointAttach {
        slot: usize,
        tokens: Vec<u32>,
        position: usize,
        index: u32,
    },
    /// Return a reserved-but-unattached index to the free list.
    CheckpointRecycle { slot: usize, position: usize, index: u32 },
}

/// The prefix-probe result for a resume decision (see
/// [`MirroredKv::match_prefix_probe`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixProbe {
    pub ckpt: Option<(usize, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    pub tables: Vec<Vec<u32>>,
    pub refcounts: Vec<u32>,
    pub free: usize,
    pub checkpoint_free: Vec<u32>,
    pub checkpoint_reserved: Vec<(usize, usize, u32)>,
    pub token_digests: Vec<(usize, [u8; 32])>,
    pub radix_digest: [u8; 32],
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
    /// Monotonic per-slot BlockTable generation. This is deliberately local
    /// derived state rather than wire identity: mirrored operations update it
    /// deterministically on all mirrors, and decode uses it only to avoid
    /// redundant validation/uploads.
    table_versions: Vec<u64>,
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
            table_versions: vec![0; slots],
            radix: PagedRadix::new(),
            sequence: 0,
            max_ctx,
            tokens: vec![Vec::new(); slots],
            state_capacity: 0,
            admitted_reused: None,
        })
    }

    /// Enable DeltaNet checkpointing: `n` pool indices. Call on all mirrors
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

    /// The checkpoint index attached at `position` under `slot`'s admitted
    /// tokens (the resume-time lookup; `None` when no checkpoint sits there).
    pub fn slot_checkpoint_index(&self, slot: usize, position: usize) -> Option<u32> {
        let tokens = self.tokens.get(slot)?;
        if tokens.len() < position {
            return None;
        }
        self.radix.ckpt_index_at(tokens, position)
    }

    /// Recycle every still-reserved checkpoint index owned by `slot`
    /// (mirror-side straggler cleanup after a publish tick).
    pub fn drop_slot_reservations(&mut self, slot: usize) {
        self.radix.drop_slot_reservations(slot);
    }

    /// Every reservation owned by `slot` as `(position, index)` (the publish
    /// tick's recycle composition reads this on BOTH ranks; the lists are
    /// mirror-deterministic so the composition cannot diverge).
    pub fn slot_reserved_indices(&self, slot: usize) -> Vec<(usize, u32)> {
        self.radix.slot_reservations(slot)
    }

    /// Read-only probe of the mirrored radix (the coordinator's decision
    /// consult): the deepest checkpoint `(position, index)` under the
    /// longest cached block prefix of `tokens`. Touches nothing - LRU and
    /// the recurrence flag advance only inside the authoritative `Admit`,
    /// so a probe can never diverge the two trees.
    pub fn match_prefix_probe(&self, tokens: &[u32]) -> PrefixProbe {
        PrefixProbe {
            ckpt: self.radix.probe_ckpt(tokens),
        }
    }

    /// Rank-0 coordinator helper: reserve checkpoint indices for `slot`'s
    /// cuts (mirrored op `CheckpointReserve`), returning `(cut, index)`
    /// pairs in cut order. Cuts already holding a reservation are skipped.
    pub fn reserve_cuts_for_slot(
        &mut self,
        slot: usize,
        cuts: &[usize],
    ) -> Result<Vec<(usize, u32)>, &'static str> {
        let mut out = Vec::with_capacity(cuts.len());
        let len = self.tokens.get(slot).ok_or("TP reserve: slot not admitted")?.len();
        for &cut in cuts {
            if cut == 0 || cut >= len || !cut.is_multiple_of(BLOCK_TOKENS) {
                return Err("TP checkpoint cut invalid");
            }
            if self.slot_reserved_ckpt(slot, cut).is_some() {
                continue;
            }
            // Exhaustion is an optional-cache miss, not a request failure.
            if self.radix.free_state_slots() == 0 {
                break;
            }
            let op = Operation::CheckpointReserve { slot, position: cut };
            if self.authorize_all(&[op]).is_err() {
                break; // e.g. protected checkpoints may not be stolen
            }
            if let Some(idx) = self.slot_reserved_ckpt(slot, cut) {
                out.push((cut, idx));
            }
        }
        Ok(out)
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
            checkpoint_free: self.radix.free_state_indices().to_vec(),
            checkpoint_reserved: self.radix.reserved_state().to_vec(),
            radix_digest: self.radix.mirror_digest(),
            token_digests: self.tokens.iter().map(|t| {
                let mut hasher = blake3::Hasher::new();
                for token in t { hasher.update(&token.to_le_bytes()); }
                (t.len(), *hasher.finalize().as_bytes())
            }).collect(),
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
    /// promotion walks exactly these pool offsets on all mirrors' slabs).
    pub fn slot_blocks(&self, slot: usize) -> Option<&[u32]> {
        self.tables.get(slot).map(|t| t.blocks())
    }

    /// Hot decode validation. Geometry/coverage are checked every call, while
    /// the O(context) block-id/refcount scan is repeated only when this slot's
    /// physical BlockTable generation changed.
    pub fn validate_device_table_versioned(
        &self,
        slot: usize,
        position: usize,
        pool_blocks: u32,
        slots: usize,
        max_ctx: usize,
        known_version: Option<u64>,
    ) -> Result<u64, &'static str> {
        if pool_blocks != self.pool.capacity()
            || slots != self.tables.len()
            || max_ctx != self.max_ctx
            || position >= max_ctx
        {
            return Err("KV GPU geometry mismatch");
        }
        let t = self.tables.get(slot).ok_or("KV slot out of range")?;
        let version = *self.table_versions.get(slot).ok_or("KV slot out of range")?;
        let needed = position / BLOCK_TOKENS + 1;
        if t.blocks().len() < needed {
            return Err("KV position has no live pages");
        }
        if known_version != Some(version)
            && t.blocks()[..needed]
                .iter()
                .any(|&b| b >= pool_blocks || self.pool.refcount(b) == 0)
        {
            return Err("KV position has no live pages");
        }
        Ok(version)
    }

    /// Validate that a live slot's entire read prefix is backed by allocated
    /// pages in precisely the geometry of the GPU payload. This is the hot
    /// decode check: it deliberately does not materialize the full all-slot
    /// host table.
    pub fn validate_device_table(
        &self,
        slot: usize,
        position: usize,
        pool_blocks: u32,
        slots: usize,
        max_ctx: usize,
    ) -> Result<(), &'static str> {
        self.validate_device_table_versioned(
            slot, position, pool_blocks, slots, max_ctx, None,
        )?;
        Ok(())
    }

    /// Serialize a validated full all-slot table for callers that actually
    /// need host bytes to upload. Decode should use validate_device_table
    /// and its persistent per-slot device-table cache instead.
    pub fn checked_device_table(
        &self,
        slot: usize,
        position: usize,
        pool_blocks: u32,
        slots: usize,
        max_ctx: usize,
    ) -> Result<Vec<u32>, &'static str> {
        self.validate_device_table(slot, position, pool_blocks, slots, max_ctx)?;
        Ok(self.device_table())
    }

    /// The coordinator emits an event only after the operation has succeeded.
    pub fn authorize(&mut self, operation: Operation) -> Result<Event, &'static str> {
        self.apply(&operation)?;
        self.sequence += 1;
        Ok(Event {
            sequence: self.sequence,
            operation,
            state: self.snapshot(),
        })
    }

    /// The coordinator applies an ordered tick of operations and snapshots ONCE at the
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

    /// Rank 1 replays a coordinator event; mismatched sequence or state fails closed.
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

    /// Rank 1 replays an ordered tick of coordinator operations and requires the
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
                let before = t.blocks().len();
                let needed = *position / BLOCK_TOKENS + 1;
                while needed.saturating_sub(t.blocks().len()) > self.pool.free_blocks() {
                    // A shared page may lose its radix ref without becoming
                    // free. Keep evicting until enough actual capacity exists.
                    if self.radix.evict_lru(&mut self.pool).is_none() {
                        return Err("KV pool exhausted");
                    }
                }
                t.ensure(*position, &mut self.pool)
                    .map_err(|_| "KV pool exhausted")?;
                if t.blocks().len() != before {
                    self.table_versions[*slot] = self.table_versions[*slot].wrapping_add(1);
                }
            }
            Operation::Release { slot } => {
                let t = self.tables.get_mut(*slot).ok_or("KV slot out of range")?;
                let changed = !t.blocks().is_empty();
                t.clear(&mut self.pool);
                if changed {
                    self.table_versions[*slot] = self.table_versions[*slot].wrapping_add(1);
                }
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
                if !blocks.is_empty() {
                    self.table_versions[*slot] = self.table_versions[*slot].wrapping_add(1);
                }
            }
            Operation::Reset => {
                for (i, t) in self.tables.iter_mut().enumerate() {
                    let changed = !t.blocks().is_empty();
                    t.clear(&mut self.pool);
                    if changed {
                        self.table_versions[i] = self.table_versions[i].wrapping_add(1);
                    }
                }
                for t in &mut self.tokens {
                    t.clear();
                }
                self.radix.recycle_all_reserved();
                // Cache retains published blocks; reset releases slots, not the cache.
            }
            Operation::Flush => {
                for (i, t) in self.tables.iter_mut().enumerate() {
                    let changed = !t.blocks().is_empty();
                    t.clear(&mut self.pool);
                    if changed {
                        self.table_versions[i] = self.table_versions[i].wrapping_add(1);
                    }
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
                if !self.tokens[*slot].is_empty() || !self.tables[*slot].blocks().is_empty()
                    || !self.radix.slot_reservations(*slot).is_empty()
                {
                    return Err("TP admit requires a released slot");
                }
                let reused = if *resume > 0 {
                    // One authoritative adoption walk: all mirrors require the
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
                    if !blocks.is_empty() {
                        self.table_versions[*slot] = self.table_versions[*slot].wrapping_add(1);
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
            Operation::CheckpointReserve { slot, position } => {
                let tokens = self.tokens.get(*slot).ok_or("TP checkpoint reserve slot invalid")?;
                if tokens.is_empty() || *position == 0 || *position >= tokens.len()
                    || !position.is_multiple_of(BLOCK_TOKENS)
                    || self.radix.reserved_index(*slot, *position).is_some()
                {
                    return Err("TP checkpoint reservation invalid");
                }
                self.radix
                    .reserve_state_for(*slot, *position)
                    .ok_or("TP checkpoint reserve exhausted")?;
            }
            Operation::CheckpointAttach {
                slot,
                tokens,
                position,
                index,
            } => {
                if self.tokens.get(*slot).map(Vec::as_slice) != Some(tokens.as_slice())
                    || !self.radix.attach_reserved_at(*slot, tokens, *position, *index)
                {
                    return Err("TP checkpoint attach has no matching reservation");
                }
            }
            Operation::CheckpointRecycle { slot, position, index } => {
                if !self.radix.recycle_reserved(*slot, *position, *index) {
                    return Err("TP checkpoint reservation already consumed or mismatched");
                }
            }
        }
        Ok(())
    }
}

/// The pure publish composition (host-testable): given a slot's checkpoint
/// reservations `(position, index)` and the set that were actually
/// snapshotted (`snapshotted` = positions whose GPU state landed), build the
/// ordered `Checkpoint*` operation list for the publish tick: attach every
/// snapshotted reservation (ascending position), recycle the rest. An
/// attached index is consumed by the attach; an unsnapshotted one returns to
/// the free list. All mirrors run the SAME composition over their
/// mirror-identical reservation lists, so the publication cannot diverge.
pub fn tp_publish_ops(
    slot: usize,
    reservations: &[(usize, u32)],
    tokens: Vec<u32>,
    snapshotted: &[usize],
) -> Vec<Operation> {
    let mut ops = Vec::with_capacity(reservations.len() + 1);
    // The new radix nodes must exist before an attached cut can be found.
    // authorize_all stages this entire sequence atomically.
    ops.push(Operation::Publish { slot, tokens: tokens.clone() });
    for &(pos, idx) in reservations {
        if snapshotted.contains(&pos) {
            ops.push(Operation::CheckpointAttach {
                slot,
                tokens: tokens.clone(),
                position: pos,
                index: idx,
            });
        } else {
            ops.push(Operation::CheckpointRecycle { slot, position: pos, index: idx });
        }
    }
    ops
}

/// Model/workload policy for deciding whether restoring a cached state is
/// worthwhile. Cache mechanics do not own these performance thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResumePolicy {
    pub min_cache_prefix: usize,
    pub narrow_slots_max: usize,
}

/// Given the deepest state checkpoint under a prompt, choose the resume point.
///
/// The cache enforces structural safety (at least two complete blocks and
/// strictly inside the prompt); the caller supplies profitability thresholds.
pub fn resume_decision(
    ckpt: Option<(usize, u32)>,
    t_len: usize,
    slots: usize,
    policy: ResumePolicy,
) -> usize {
    let floor = if slots <= policy.narrow_slots_max {
        2 * BLOCK_TOKENS
    } else {
        policy.min_cache_prefix.max(2 * BLOCK_TOKENS)
    };
    match ckpt {
        Some((pos, _)) if pos >= floor && pos < t_len => pos,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_RESUME_POLICY: ResumePolicy = ResumePolicy {
        min_cache_prefix: 512,
        narrow_slots_max: 12,
    };
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

    // ── prefix-cache resume (milestone 1) ────────────────────────────────

    fn mk(max_ctx: usize, slots: usize) -> MirroredKv {
        let blocks = (max_ctx.div_ceil(BLOCK_TOKENS) * slots) as u32;
        MirroredKv::new(blocks, slots, max_ctx).unwrap()
    }

    #[test]
    fn table_version_changes_only_when_physical_mapping_changes() {
        let mut kv = mk(64, 1);
        assert_eq!(kv.table_versions[0], 0);

        kv.authorize(Operation::Ensure { slot: 0, position: 0 }).unwrap();
        let first = kv.table_versions[0];
        assert_ne!(first, 0);

        // Same physical page: position advances, mapping generation does not.
        kv.authorize(Operation::Ensure { slot: 0, position: 15 }).unwrap();
        assert_eq!(kv.table_versions[0], first);

        // Crossing the 16-token page boundary grows the BlockTable.
        kv.authorize(Operation::Ensure { slot: 0, position: 16 }).unwrap();
        let grown = kv.table_versions[0];
        assert_ne!(grown, first);

        kv.authorize(Operation::Release { slot: 0 }).unwrap();
        let released = kv.table_versions[0];
        assert_ne!(released, grown);

        // Reusing an empty cached prefix changes the physical mapping again.
        kv.authorize(Operation::Ensure { slot: 0, position: 0 }).unwrap();
        assert_ne!(kv.table_versions[0], released);
    }

    /// Admit a cold prompt, publish its pages, checkpoint at the given cut.
    ///
    /// Mirrors the op sequence on coordinator and mirror at every step.
    fn seed_cache(
        head: &mut MirroredKv,
        worker: &mut MirroredKv,
        tokens: &[u32],
        slot: usize,
    ) {
        head.set_state_capacity(4);
        worker.set_state_capacity(4);
        let ops = vec![Operation::Admit {
            slot,
            tokens: tokens.to_vec(),
            resume: 0,
        }];
        let end = head.authorize_all(&ops).unwrap();
        worker.mirror_tick(&ops, &end).unwrap();
        // fill the slot's table with live pages (the physical backing a real
        // prefill wrote) and publish its full blocks
        let position = tokens.len() - 1;
        let ops2 = vec![Operation::Ensure { slot, position }];
        let end2 = head.authorize_all(&ops2).unwrap();
        worker.mirror_tick(&ops2, &end2).unwrap();
        let ops3 = vec![Operation::Publish {
            slot,
            tokens: tokens.to_vec(),
        }];
        let end3 = head.authorize_all(&ops3).unwrap();
        worker.mirror_tick(&ops3, &end3).unwrap();
    }

    fn attach_ckpt(
        head: &mut MirroredKv,
        worker: &mut MirroredKv,
        tokens: &[u32],
        slot: usize,
        cut: usize,
    ) -> (u32, u32) {
        let ops = vec![Operation::CheckpointReserve {
            slot,
            position: cut,
        }];
        let end = head.authorize_all(&ops).unwrap();
        worker.mirror_tick(&ops, &end).unwrap();
        let idx_head = head.slot_reserved_ckpt(slot, cut).unwrap();
        let idx_worker = worker.slot_reserved_ckpt(slot, cut).unwrap();
        let ops2 = vec![Operation::CheckpointAttach {
            slot,
            tokens: tokens.to_vec(),
            position: cut,
            index: idx_head,
        }];
        let end2 = head.authorize_all(&ops2).unwrap();
        worker.mirror_tick(&ops2, &end2).unwrap();
        (idx_head, idx_worker)
    }

    /// 1 + 2. Cold prompt probes nothing; an identical cached prompt probes
    /// a nonzero block-aligned resume at the attached checkpoint.
    #[test]
    fn cold_probe_is_none_and_identical_prompt_probes_the_checkpoint() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        let mut tokens: Vec<u32> = (0..96).collect(); // 6 blocks
        tokens.push(999);
        assert!(head.match_prefix_probe(&tokens).ckpt.is_none());
        seed_cache(&mut head, &mut worker, &tokens, 0);
        // attach at the 4-block boundary (64) - the deepest prefill-reachable
        // cut for a 97-token prompt per tp_checkpoint_cuts
        let cut = 64;
        let (idx, _) = attach_ckpt(&mut head, &mut worker, &tokens, 0, cut);
        // read-only probe leaves both trees untouched (equal snapshots)
        let before = (head.snapshot(), worker.snapshot());
        assert_eq!(
            head.match_prefix_probe(&tokens).ckpt,
            Some((cut, idx))
        );
        assert_eq!(head.snapshot(), before.0);
        assert_eq!(worker.snapshot(), before.1);
        // decision honors it
        assert_eq!(resume_decision(Some((cut, idx)), tokens.len(), 2, TEST_RESUME_POLICY), cut);
        // and the worker's tree sees the same checkpoint (mirror identity)
        assert_eq!(
            worker.match_prefix_probe(&tokens).ckpt,
            Some((cut, idx))
        );
    }

    /// 3. Long shared prefix + divergent tail: the tail prompt probes the
    ///
    /// checkpoint under the shared prefix only.
    #[test]
    fn shared_prefix_divergent_tail_probes_the_shared_checkpoint() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        let mut base: Vec<u32> = (0..80).collect(); // 5 blocks
        base.push(500);
        seed_cache(&mut head, &mut worker, &base, 0);
        let (idx, _) = attach_ckpt(&mut head, &mut worker, &base, 0, 64);
        // divergent tail: same first 64 tokens, different after
        let mut tail = base[..64].to_vec();
        tail.extend((900..1000).collect::<Vec<u32>>());
        tail.push(501);
        let probe = head.match_prefix_probe(&tail);
        assert_eq!(probe.ckpt, Some((64, idx)));
        assert_eq!(resume_decision(probe.ckpt, tail.len(), 2, TEST_RESUME_POLICY), 64);
        // worker agrees (rank symmetry)
        assert_eq!(worker.match_prefix_probe(&tail).ckpt, Some((64, idx)));
    }

    /// 4. A partial trailing page is never treated as reusable.
    ///
    /// Matching the exact cached sequence keeps the last block unmatched.
    #[test]
    fn partial_trailing_page_is_not_reusable() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        let mut tokens: Vec<u32> = (0..32).collect(); // exactly 2 blocks
        tokens.push(7);
        seed_cache(&mut head, &mut worker, &tokens, 0);
        let (idx, _) = attach_ckpt(&mut head, &mut worker, &tokens, 0, 16);
        // the exact 33-token prompt: match_full caps at len-1, so only block 1
        let probe = head.match_prefix_probe(&tokens);
        assert_eq!(probe.ckpt, Some((16, idx)));
        // ...but the decision gate refuses a 1-block resume (the non-TP
        // `pos >= 32` rule): the page is only partially useful and the tail
        // prefill dominates. The exact-repeat case resumes only when a DEEP
        // checkpoint exists (see the 6-block test).
        assert_eq!(resume_decision(probe.ckpt, tokens.len(), 2, TEST_RESUME_POLICY), 0);
        // a resumed Admit validates the exact checkpoint and rejects others
        let ops = vec![Operation::Admit {
            slot: 1,
            tokens: tokens.clone(),
            resume: 32, // = the full prompt: no checkpoint attached there
        }];
        assert!(head.authorize_all(&ops).is_err());
    }

    /// 5. The worker validates the coordinator-selected checkpoint exactly.
    ///
    /// A resume position the worker's tree cannot satisfy fails the mirror.
    #[test]
    fn worker_rejects_a_resume_it_cannot_satisfy() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        let mut tokens: Vec<u32> = (0..96).collect();
        tokens.push(999);
        seed_cache(&mut head, &mut worker, &tokens, 0);
        attach_ckpt(&mut head, &mut worker, &tokens, 0, 64);
        // coordinator selects 64: all mirrors accept
        let ops = vec![Operation::Admit {
            slot: 1,
            tokens: tokens.clone(),
            resume: 64,
        }];
        let end = head.authorize_all(&ops).unwrap();
        assert!(worker.mirror_tick(&ops, &end).is_ok());
        // a corrupted/mismatched selection (checkpoint at 80 only): the
        // worker's Admit re-walk must refuse it, so divergence is impossible
        let ops2 = vec![Operation::Admit {
            slot: 1,
            tokens: tokens.clone(),
            resume: 48, // no checkpoint attached at 48
        }];
        assert!(head.authorize_all(&ops2).is_err());
        assert!(worker.mirror_tick(&ops2, &end).is_err());
    }

    /// 6. Symmetric recompute on unavailability.
    ///
    /// A cold Admit succeeds identically on all mirrors even when a partial
    /// cache exists elsewhere.
    #[test]
    fn cold_admit_is_rank_symmetric_when_cache_unavailable() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        let tokens: Vec<u32> = (0..65).collect();
        // no cache at all: cold admission works on both
        let ops = vec![Operation::Admit {
            slot: 0,
            tokens: tokens.clone(),
            resume: 0,
        }];
        let end = head.authorize_all(&ops).unwrap();
        worker.mirror_tick(&ops, &end).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        assert!(head.slot_admitted_tokens(0) == tokens.as_slice());
    }

    /// 7 + 8. Slot release drops slot refs but not radix-owned cache pages.
    ///
    /// Eviction returns the tree's refs; reservations die with the slot.
    #[test]
    fn release_keeps_radix_pages_and_recycles_reservations() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        let mut tokens: Vec<u32> = (0..96).collect();
        tokens.push(999);
        seed_cache(&mut head, &mut worker, &tokens, 0);
        let (idx, _) = attach_ckpt(&mut head, &mut worker, &tokens, 0, 64);
        let free_after_seed = head.snapshot().free;
        // release the live slot: its table refs drop, the tree's stay
        let ops = vec![Operation::Release { slot: 0 }];
        let end = head.authorize_all(&ops).unwrap();
        worker.mirror_tick(&ops, &end).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        // the tree still pins the 6 PUBLISHED pages; only the un-published
        // partial-tail block (position 96's page) returned to the pool
        assert_eq!(
            head.snapshot().free,
            free_after_seed + 1,
            "tree pins published pages; the partial-tail page frees"
        );
        assert!(head.slot_admitted_tokens(0).is_empty());
        // a resumed re-adoption retakes refs from the tree's pages
        let ops2 = vec![Operation::Admit {
            slot: 0,
            tokens: tokens.clone(),
            resume: 64,
        }];
        let end2 = head.authorize_all(&ops2).unwrap();
        worker.mirror_tick(&ops2, &end2).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        let _ = idx;
        // flush returns everything (tree refs + slot refs)
        let ops3 = vec![Operation::Flush];
        let end3 = head.authorize_all(&ops3).unwrap();
        worker.mirror_tick(&ops3, &end3).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        assert!(head.snapshot().refcounts.iter().all(|&rc| rc == 0));
    }

    /// 9. A resume validates the checkpoint exactly at the resume position.
    ///
    /// GQA page adoption depth and checkpoint position cannot disagree.
    #[test]
    fn resumed_checkpoint_position_matches_adopted_prefix_depth() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        let mut tokens: Vec<u32> = (0..96).collect();
        tokens.push(999);
        seed_cache(&mut head, &mut worker, &tokens, 0);
        attach_ckpt(&mut head, &mut worker, &tokens, 0, 64);
        let ops = vec![Operation::Admit {
            slot: 1,
            tokens: tokens.clone(),
            resume: 64,
        }];
        let end = head.authorize_all(&ops).unwrap();
        worker.mirror_tick(&ops, &end).unwrap();
        // the slot's table now backs exactly 4 blocks (64 tokens) and the
        // checkpoint lookup at that boundary succeeds on all mirrors
        for kv in [&head, &worker] {
            assert!(kv.slot_checkpoint_index(1, 64).is_some());
            assert_eq!(kv.tables[1].blocks().len(), 4);
        }
        // and the device-table validation accepts every ADOPTED position
        // (0..64; a prefill's own Ensures grow the table past 64 afterwards)
        let bps = 128 / 16;
        for pos in [0usize, 16, 48, 63] {
            assert!(
                head.checked_device_table(1, pos, bps * 2, 2, 128).is_ok(),
                "adopted position {pos} must be live"
            );
        }
    }

    /// The pure decision gate: floors, block alignment, strictly-inside.
    #[test]
    fn tp_resume_decision_gate() {
        // below the 2-block floor: cold
        assert_eq!(resume_decision(Some((16, 0)), 100, 2, TEST_RESUME_POLICY), 0);
        // deep enough, inside: resume
        assert_eq!(resume_decision(Some((64, 0)), 100, 2, TEST_RESUME_POLICY), 64);
        // equal to the prompt length: cold (one token must remain to prefill)
        assert_eq!(resume_decision(Some((64, 0)), 64, 2, TEST_RESUME_POLICY), 0);
        // narrow serve admits short resumes; a wide one would gate on
        // the model-specific minimum prefix policy
        assert_eq!(resume_decision(Some((32, 0)), 100, 2, TEST_RESUME_POLICY), 32);
        // no checkpoint: cold
        assert_eq!(resume_decision(None, 100, 2, TEST_RESUME_POLICY), 0);
    }

    /// The pure publish composition: snapshotted cuts attach, the rest
    /// recycle, publish last, in cut order.
    #[test]
    fn tp_publish_composition() {
        let tokens = vec![1u32; 96];
        let ops = tp_publish_ops(
            1,
            &[(64, 3), (80, 5)],
            tokens.clone(),
            &[80],
        );
        assert_eq!(
            ops,
            vec![
                Operation::Publish { slot: 1, tokens: tokens.clone() },
                Operation::CheckpointRecycle { slot: 1, position: 64, index: 3 },
                Operation::CheckpointAttach {
                    slot: 1,
                    tokens: tokens.clone(),
                    position: 80,
                    index: 5,
                },
            ]
        );
        // everything snapshotted: no recycles
        let ops2 = tp_publish_ops(0, &[(64, 3)], vec![2u32; 32], &[64]);
        assert!(matches!(ops2[0], Operation::Publish { slot: 0, .. }));
        assert!(matches!(ops2[1], Operation::CheckpointAttach { position: 64, index: 3, .. }));
    }

    #[test]
    fn abort_before_first_chunk_releases_admission_reservations() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        head.set_state_capacity(1);
        worker.set_state_capacity(1);
        let tokens = vec![7; 97];
        let admit = [Operation::Admit { slot: 0, tokens: tokens.clone(), resume: 0 },
            Operation::CheckpointReserve { slot: 0, position: 80 }];
        let end = head.authorize_all(&admit).unwrap();
        worker.mirror_tick(&admit, &end).unwrap();
        assert!(head.authorize_all(&[Operation::Admit { slot: 0, tokens: tokens.clone(), resume: 0 }]).is_err());
        let release = [Operation::Release { slot: 0 }];
        let end = head.authorize_all(&release).unwrap();
        worker.mirror_tick(&release, &end).unwrap();
        assert!(head.snapshot().checkpoint_reserved.is_empty());
        assert_eq!(head.snapshot().checkpoint_free, vec![0]);
        let end = head.authorize_all(&[Operation::Admit { slot: 0, tokens, resume: 0 }]).unwrap();
        worker.mirror_tick(&[Operation::Admit { slot: 0, tokens: vec![7; 97], resume: 0 }], &end).unwrap();
    }

    #[test]
    fn cached_pages_evict_to_satisfy_new_ensure() {
        let mut kv = MirroredKv::new(3, 1, 64).unwrap();
        let tokens: Vec<u32> = (0..33).collect();
        kv.authorize_all(&[Operation::Admit { slot: 0, tokens: tokens.clone(), resume: 0 },
            Operation::Ensure { slot: 0, position: 31 },
            Operation::Publish { slot: 0, tokens }]).unwrap();
        kv.authorize_all(&[Operation::Release { slot: 0 }]).unwrap();
        assert_eq!(kv.snapshot().free, 1);
        kv.authorize_all(&[Operation::Admit { slot: 0, tokens: vec![99; 33], resume: 0 },
            Operation::Ensure { slot: 0, position: 31 }]).unwrap();
        assert_eq!(kv.snapshot().free, 0);
        assert_eq!(kv.tables[0].blocks().len(), 2);
    }

    #[test]
    fn overlapping_identical_slots_reserve_distinct_indices() {
        let mut kv = mk(128, 2);
        kv.set_state_capacity(2);
        let tokens = vec![5; 97];
        kv.authorize_all(&[Operation::Admit { slot: 0, tokens: tokens.clone(), resume: 0 },
            Operation::Admit { slot: 1, tokens, resume: 0 }]).unwrap();
        assert_eq!(kv.reserve_cuts_for_slot(0, &[80]).unwrap(), vec![(80, 0)]);
        assert_eq!(kv.reserve_cuts_for_slot(1, &[80]).unwrap(), vec![(80, 1)]);
        assert_eq!(kv.slot_reserved_ckpt(0, 80), Some(0));
        assert_eq!(kv.slot_reserved_ckpt(1, 80), Some(1));
    }

    #[test]
    fn cold_publish_release_and_identical_or_divergent_resume_mirror() {
        let mut head = mk(256, 2);
        let mut worker = mk(256, 2);
        head.set_state_capacity(4);
        worker.set_state_capacity(4);
        let tokens: Vec<u32> = (0..97).collect();
        let admit = vec![Operation::Admit { slot: 0, tokens: tokens.clone(), resume: 0 },
            Operation::CheckpointReserve { slot: 0, position: 80 },
            Operation::CheckpointReserve { slot: 0, position: 96 }];
        let end = head.authorize_all(&admit).unwrap();
        worker.mirror_tick(&admit, &end).unwrap();
        let reservations = head.slot_reserved_indices(0);
        assert_eq!(reservations, worker.slot_reserved_indices(0));
        let ensure = vec![Operation::Ensure { slot: 0, position: 96 }];
        let end = head.authorize_all(&ensure).unwrap();
        worker.mirror_tick(&ensure, &end).unwrap();
        // Production ordering: new nodes and both checkpoint attachments in
        // ONE staged publish, not the earlier seed-then-attach test sequence.
        let publish = tp_publish_ops(0, &reservations, tokens.clone(), &[80, 96]);
        let end = head.authorize_all(&publish).unwrap();
        worker.mirror_tick(&publish, &end).unwrap();
        assert_eq!(head.match_prefix_probe(&tokens).ckpt.map(|(p, _)| p), Some(96));
        let release = vec![Operation::Release { slot: 0 }];
        let end = head.authorize_all(&release).unwrap();
        worker.mirror_tick(&release, &end).unwrap();
        assert_eq!(head.snapshot().checkpoint_reserved, Vec::new());
        for (prompt, expected) in [(tokens.clone(), 96), {
            let mut divergent = tokens[..80].to_vec();
            divergent.extend(1000..1033);
            (divergent, 80)
        }] {
            let probe = head.match_prefix_probe(&prompt);
            assert_eq!(probe.ckpt.map(|(p, _)| p), Some(expected));
            let resume = resume_decision(probe.ckpt, prompt.len(), 2, TEST_RESUME_POLICY);
            let op = vec![Operation::Admit { slot: 1, tokens: prompt, resume }];
            let end = head.authorize_all(&op).unwrap();
            worker.mirror_tick(&op, &end).unwrap();
            assert_eq!(head.tables[1].blocks().len(), expected / BLOCK_TOKENS);
            let done = vec![Operation::Release { slot: 1 }];
            let end = head.authorize_all(&done).unwrap();
            worker.mirror_tick(&done, &end).unwrap();
        }
    }

    #[test]
    fn reservation_recycles_once_and_zero_capacity_serves_pages_only() {
        let mut kv = mk(128, 2);
        kv.set_state_capacity(0);
        let tokens: Vec<u32> = (0..65).collect();
        kv.authorize_all(&[Operation::Admit { slot: 0, tokens: tokens.clone(), resume: 0 }]).unwrap();
        assert!(kv.reserve_cuts_for_slot(0, &[48, 64]).unwrap().is_empty());
        kv.authorize_all(&[Operation::Ensure { slot: 0, position: 64 }]).unwrap();
        kv.authorize_all(&tp_publish_ops(0, &[], tokens.clone(), &[])).unwrap();
        kv.authorize_all(&[Operation::Release { slot: 0 }]).unwrap();
        assert!(kv.match_prefix_probe(&tokens).ckpt.is_none());
        kv.set_state_capacity(1);
        kv.authorize_all(&[Operation::Admit { slot: 0, tokens: tokens.clone(), resume: 0 }]).unwrap();
        let reserved = kv.reserve_cuts_for_slot(0, &[48, 64]).unwrap();
        assert_eq!(reserved.len(), 1);
        let (position, index) = reserved[0];
        let recycle = Operation::CheckpointRecycle { slot: 0, position, index };
        kv.authorize_all(std::slice::from_ref(&recycle)).unwrap();
        assert!(kv.authorize_all(&[recycle]).is_err());
        assert_eq!(kv.snapshot().checkpoint_free, vec![index]);
    }

    #[test]
    fn mirror_refuses_unmirrored_reservation_even_when_pages_match() {
        let mut head = mk(128, 2);
        let mut worker = mk(128, 2);
        head.set_state_capacity(1);
        worker.set_state_capacity(1);
        let tokens: Vec<u32> = (0..65).collect();
        let admit = Operation::Admit { slot: 0, tokens, resume: 0 };
        let end = head.authorize_all(&[admit.clone(), Operation::CheckpointReserve { slot: 0, position: 48 }]).unwrap();
        assert!(worker.mirror_tick(&[admit], &end).is_err());
        assert!(worker.slot_admitted_tokens(0).is_empty());
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
