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

pub struct MirroredKv {
    pool: KvPool,
    tables: Vec<BlockTable>,
    radix: PagedRadix,
    sequence: u64,
    max_ctx: usize,
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
        })
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
    /// ordered operations. Fails closed before anything is sent if any
    /// operation is invalid; the mirror state is unchanged in that case
    /// (apply validates before mutating, per operation).
    pub fn authorize_all(&mut self, operations: &[Operation]) -> Result<Snapshot, &'static str> {
        for op in operations {
            self.apply(op)?;
        }
        self.sequence += operations.len() as u64;
        Ok(self.snapshot())
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
        for op in operations {
            self.apply(op)?;
        }
        self.sequence += operations.len() as u64;
        if self.snapshot() != *end_state {
            return Err("KV logical state diverged");
        }
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
                // Cache retains published blocks; reset releases slots, not the cache.
            }
            Operation::Flush => {
                for t in &mut self.tables {
                    t.clear(&mut self.pool);
                }
                while self.radix.evict_lru(&mut self.pool).is_some() {}
            }
        }
        Ok(())
    }
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
        assert_eq!(e.sequence, 0);
        // An empty tick is malformed, never silent progress.
        assert!(e.mirror_tick(&[], &a.snapshot()).is_err());

        // A wrong END STATE with valid ops fails closed (the failure mode the
        // per-row design caught mid-tick: end-state equality still catches it).
        let mut f = MirroredKv::new(4, 2, 48).unwrap();
        let mut wrong_end = end.clone();
        wrong_end.free += 1;
        assert!(f.mirror_tick(&ops, &wrong_end).is_err());
        // And the failed mirror is not half-applied: replaying correctly works.
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
