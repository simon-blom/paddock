# Phase 11 Stage A — rank-local graph capture for TP=2 decode rows

Status: implemented and host-verified; two-Spark validation deferred (both
Sparks serving vLLM). Off by default: `PADDOCK_TP_GRAPH=1` opts the serving
pair in.

## Design

Each TP decode row's per-layer compute splits at the NCCL all-reduces into
collective-free runs. Stage A captures exactly those runs with the existing
`CapturedGraph` infrastructure (`gpu/graph.rs`) and leaves every collective
outside capture, replaying between the communicator's event fences:

```text
replay PreAttn graph  (rmsnorm + attention/DeltaNet run -> partial)
eager: after_compute / all_reduce / before_compute  (NCCL stream)
eager: residual add
replay PreFfn graph   (rmsnorm + gate/up/swiglu/down -> partial)
eager: after_compute / all_reduce / before_compute  (NCCL stream)
eager: residual add
```

Embedding, the final norm and the lm-head GEMV stay eager single-kernel
launches (a one-kernel graph replay costs what the launch costs).

Why the varying inputs are capture-safe: the decode kernels read position,
slot and the paged block table as buffer CONTENT from fixed device buffers
the caller re-stages with plain memcpys between replays (outside capture —
an unpinned host source inside a capture is illegal), and every grid/block
dimension is fixed at capture time. The DeltaNet recurrent/conv payloads a
run bakes are whatever pair the caller swapped in, so DeltaNet layers key
their graphs per slot (`PreAttnDelta(layer, slot)`); GQA and FFN runs are
slot-agnostic and key per layer (`PreAttn(layer)`, `PreFfn(layer)`).

Non-paged GQA is rejected at enable time: the non-paged KV append advances
a host-side counter a graph cannot replay. Isolated parity paths without
the mirrored KV pool never enable graphs anyway.

The prefill lane is never captured: its runs are row-span shaped and its
slots are owned by in-flight spans. `prefill_lane_step` passes `graphed:
false` explicitly, so the accepted Phase 10 overlap behavior is unchanged.

One capture's pre-sync (stream.synchronize) serializes the first token that
elects each graph — including waiting the preceding all-reduce — so the
first token per (layer, slot) is slower and later tokens are faster. That
trade is what the deferred two-Spark Stage A validation measures.

## Implementation

- `tp_graph.rs` (new): `TpRunKey` keys + `TpGraphs` per-rank graph cache;
  module docs carry the full design rationale.
- `ffn_tp.rs`: `forward` split into `run` (collective-free:
  gate/up GEMVs, SwiGLU, down GEMV) + `finish` (fences + all-reduce);
  `forward` composes both, semantics unchanged.
- `gqa_tp.rs`: `forward_at`'s collective-free middle extracted as
  `attention_run`; `finish` runs the fences + all-reduce;
  `stage_decode_inputs` re-stages position/axes/slot/block-table into the
  fixed buffers with the same validation as `forward_paged` (it reads the
  logical table through the same `checked_device_table` contract).
- `delta_tp.rs`: `forward` split into `decode_run` (collective-free run
  through `core`), `finish_partial` (memset + out-projection GEMVs into the
  capacity-sized `partial`), and `finish` (fences + all-reduce);
  `swap_state_in`/`swap_state_out` expose the eager `decode_slot` pointer
  swap to the graphed caller. All eager entry points compose the pieces in
  the same order as before — the eager path is semantics-preserving by
  construction.
- `tp_model.rs`: `capture_mixer_run`/`capture_ffn_run` follow the serial
  model's capture discipline (quiesce stream, thread-local capture, record,
  end + instantiate with no flags, surface record errors after clean end).
  `forward_token_body` takes a `graphed` flag; decode-lane call sites pass
  `Self::tp_graph_enabled()` (per-process `OnceLock` over
  `PADDOCK_TP_GRAPH`), the prefill lane passes `false`.
  `enable_tp_graphs` validates the paged-GQA requirement and captures
  DeltaNet slot 0's runs eagerly (its state pair is live in place);
  other slots capture lazily on their first decode row. `tp_graph_count`
  is the probe surface for the two-Spark gates.
- `tp_serve.rs`: both `TpCoordinator::load` and the rank-1 worker call
  `enable_tp_graphs` after `enable_prefill_lane` when
  `tp_graph_enabled_for_serve()` — one shared env read so the ranks cannot
  diverge (an eager all_reduce meeting a graphed one would mispair the
  collectives; the runner exports its environment to the worker rank, so
  setting it on the coordinator covers the pair).

## Host evidence (this tree, post-merge 56deb80)

- `cargo check -p paddock-engine` and `cargo check -p paddock-runner
  --all-targets`: clean, zero warnings.
- `cargo test -p paddock-engine --lib`: 452 passed / 0 failed (same count
  as the pre-Stage-A audit baseline — the run/finish splits are
  behavior-neutral).
- `cargo test -p paddock-runner --lib`: 550 passed / 0 failed.
- `cargo test -p paddock-dist`: 18 passed / 0 failed.
- `cargo test -p paddock-kernels`: 53 passed / 0 failed (ABI tripwires
  included).
- `git diff --check`: clean.

## Deferred gates (require both Sparks free of vLLM)

1. Two-Spark Phase 10 regression (bit-for-bit eager, `PADDOCK_TP_GRAPH`
   unset): release runner + CUDA pack rebuild, sync to the worker stage
   dir, then `phase10_live_scheduler.py` pipe + `PADDOCK_UNIFIED=1
   PADDOCK_NO_MIXED_SPEC=1` overlap suites — establishes the decode pipe
   and unified prefill/decode overlap remain functional on the merged
   tree.
2. Stage A GPU validation with `PADDOCK_TP_GRAPH=1` on both ranks: graphed
   vs eager token/ID parity on the accepted probes, `tp_graph_count`
   expectations (3 graphs per GQA+DeltaNet layer pair... per layer:
   GQA 2 + FFN 1, DeltaNet 2 per slot + FFN 1), first-token latency vs
   steady-state decode throughput, and the same mirrored-KV/DeltaNet-state
   probe assertions under graphed decode.
