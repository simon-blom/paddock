//! Phase 11 Stage A: rank-local CUDA-graph capture for TP=2 decode rows.
//!
//! Each TP decode row's per-layer compute splits at the NCCL all-reduces
//! into collective-free runs. Stage A captures exactly those runs with the
//! existing `CapturedGraph` infrastructure and leaves every collective
//! outside graph capture, replaying between the communicator's event
//! fences:
//!
//! ```text
//! replay PreAttn graph  (rmsnorm + attention/DeltaNet run -> partial)
//! eager: after_compute / all_reduce / before_compute  (NCCL stream)
//! eager: residual add
//! replay PreFfn graph   (rmsnorm + gate/up/swiglu/down -> partial)
//! eager: after_compute / all_reduce / before_compute  (NCCL stream)
//! eager: residual add
//! ```
//!
//! Embedding, the final norm and the lm-head GEMV stay eager single-kernel
//! launches: a one-kernel graph replay costs what the launch costs, so
//! there is nothing to win.
//!
//! Why the varying inputs are capture-safe: the decode kernels read
//! position, slot and the paged block table as buffer CONTENT from fixed
//! device buffers the caller re-stages with plain memcpys between replays
//! (outside capture - an unpinned host source inside a capture is
//! illegal), and every grid/blocked dimension is fixed at capture time.
//! The DeltaNet recurrent/conv payloads the run bakes are whatever pair
//! the caller swapped in, so a DeltaNet layer keys its graph per slot;
//! GQA and FFN runs are slot-agnostic and key per layer only.
//!
//! One capture's pre-sync (stream.synchronize) serializes the first token
//! that elects each graph - including waiting the preceding all-reduce -
//! so the first token per slot is measurably slower and later tokens are
//! faster. That trade is what the two-Spark Stage A validation measures.
//!
//! Off by default: `PADDOCK_TP_GRAPH=1` opts the serving ranks in after
//! model load. Parity examples and the accepted Phase 10 eager probes are
//! untouched (they never call `enable_tp_graphs`), so the deferred Phase 10
//! regression remains bit-for-bit eager.
use std::collections::HashMap;

use super::SendGraph;

/// Which collective-free run a cached graph replays. DeltaNet layers bake
/// the slot-swapped recurrent/conv buffer addresses, so they key per slot;
/// GQA and FFN runs are slot-agnostic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum TpRunKey {
    /// rmsnorm + the layer's GQA attention run (paged mode, slot-agnostic).
    Attn(usize),
    /// rmsnorm + DeltaNet one-row run + partial staging for `slot`.
    AttnDelta(usize, usize),
    /// rmsnorm + the FFN rank-local run.
    Ffn(usize),
}

/// The rank-local graph cache. Lives on the decode-lane `Qwen35TpRank`
/// only; the prefill lane never captures (its slots are owned by in-flight
/// spans and its runs are row-span shaped).
#[derive(Default)]
pub(crate) struct TpGraphs {
    map: HashMap<TpRunKey, SendGraph>,
}

impl TpGraphs {
    pub(crate) fn get(&self, key: TpRunKey) -> Option<&SendGraph> {
        self.map.get(&key)
    }
    pub(crate) fn insert(&mut self, key: TpRunKey, graph: SendGraph) {
        self.map.insert(key, graph);
    }
    /// Captured graph count (probe/log surface for the two-Spark gates).
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}
