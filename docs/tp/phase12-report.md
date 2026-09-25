# Phase 12 — KV/offload compatibility under rank-local TP geometry

Status: IMPLEMENTATION COMPLETE, HOST VERIFICATION COMPLETE, two-Spark
validation DEFERRED (this session's Sparks serve the local GLM 5.3 Flash
workload; per session rules no GPU work was attempted and no serving process
was touched).

Baseline: Phase 10 (TP=2 decode pipe + unified prefill/decode overlap,
accepted), Phase 11 Stage A (rank-local CUDA graphs, accepted), tree
`a7ebdce` on top of upstream integration base `394ef14`.

## Scope taken from the canonical plan

Phase 12 covers, per `../PADDOCK-PLAN.md`:

- KV tier payloads;
- per-layer KV types;
- FP8 KV where supported;
- host/NVMe offload;
- rank-local memory accounting.

Key rule (verbatim from the plan): a logical block ID may be mirrored; its
actual tier payload is rank-local. Each rank's offload policy operates on its
local footprint. No cross-rank KV migration.

## What the survey found before any code changed

The accepted Phase 9-11 TP serve path had no memory-management surface at
all, by design:

- `TpGenerator` inherited the `Generator` trait's no-op `tier_*` defaults;
  the TP=1 tier recipe (pool planes + DeltaNet checkpoint aux + cache
  namespace identity) lives only in the TP=1 batch enable path
  (`batch.rs:681-753`).
- The coordinator never issues `MirroredKv` `Publish`/`Reuse` mirroring (no
  radix chains exist), so a tier armed on the TP path would have had nothing
  to demote.
- The startup gate refused `kv_offload` and any KV dtype other than f16.
- `KvDtype::Fp16` was hardcoded at three TP load sites: coordinator rank load
  (`tp_serve.rs`), worker rank load (`tp_serve.rs`), and the prefill lane
  (`tp_model.rs` `enable_prefill_lane`). The lane's hardcoded dtype was a
  latent lane-vs-decode divergence: any future change to the decode arm alone
  would have forked lane and decode slab widths.

## Implementation

### 1. Per-layer KV types + FP8 KV (items: per-layer KV types; FP8 KV)

The KV dtype is now carried end-to-end on the TP wire and resolved exactly
once, rank-0-authoritative like everything else in the TP path:

- `paddock-dist` `TpInit` gained `kv_dtype: String`. Rank 0 resolves the
  dtype against ITS device (`kv_dtype_serve(cc)` reading
  `PADDOCK_KV_CACHE_DTYPE`, gated through
  `paddock_models::gpu_support::fp8_kv_blocked`) BEFORE composing `TpInit`,
  so rank 1 never reads the runner's env and the two ranks can never load
  different widths. Unknown values fail closed (`kv_dtype_parse` errors; rank
  1 rejects `TpInit` rather than guessing).
- The resolved dtype flows into both ranks' `Qwen35TpRank::load_slots` and —
  via the new `kv_dtype` field on `Qwen35TpRank` — into the prefill lane
  construction, which now reuses `self.kv_dtype` instead of hardcoding Fp16.
  Lane and decode slabs can no longer diverge.
- The startup gate (`startup.rs`) accepts `auto | f16 | fp8_e4m3`; anything
  else still refuses to start a TP pair. The refusal message names the
  accepted values.
- Semantics match TP=1's `apply_kv_dtype` exactly: today
  `gpu_support::fp8_kv` answers true on every die this build serves (fp8
  storage is software-emulated and byte-exact), so the device gate is a kept
  seam; if it ever answers blocked again, rank 0 demotes LOUDLY to f16 (error
  log names the double-sized pool) and the demoted value rides the wire.
- Kernel reality checked, not assumed: the TP GQA rank passes `self.dtype`
  into `kv_append_batch_paged` and `attn_decode_batch_paged`
  (`gqa_tp.rs:472/483/502/513/523/541`), and the shipped pack dispatches
  `__nv_fp8_e4m3` template arms on `kv_dtype == PD_KV_FP8_E4M3` in
  `pd_kv_append_batch_paged_kernel` (`packs/cuda/src/elementwise.cuh:2022`),
  `pd_attn_decode_batch_paged` (`packs/cuda/src/gemm/f32_qkv.cuh:2987`), and
  the fused qkv/rope/append chain (`elementwise.cuh:2297/2352`). The fp8
  plumbing is kernel-real for every op the TP rank actually launches.

Host parity evidence: fp8 vs f16 token parity, T=0 (two Spark pairs) —
deferred (see below).

### 2. Rank-local memory accounting (item: rank-local accounting)

- `Qwen35TpRank::context_mem_bytes()` sums exact per-rank context geometry:
  `local_kv_bytes()` per full-attention layer, `local_state_bytes()` per
  DeltaNet layer (both already rank-local by construction from the accepted
  Phase 4/7 shard geometry).
- `Qwen35TpRank::process_mem_used_bytes()` reports the executor's
  process-pool bytes (weights + context planes + scratch), the same
  measurement family TP=1 reports.
- The rank-0 coordinator thread measures both at load and reports them
  through the ready handshake (tuple widened: vocab, max_ctx,
  device_sampling, context_bytes, process_bytes).
- `TpGenerator` implements the three `Generator` memory methods for real:
  `kv_mem_bytes()` = exact rank-local context geometry; `device_mem_used()` =
  measured process bytes; `weights_mem_bytes()` = measured minus context
  (documented as an upper bound on resident weights + scratch, matching the
  TP=1 family's reporting semantics). TP=1 behavior is untouched — the TP=1
  generator's implementations are unchanged.

### 3. KV tier payloads + host/NVMe offload: BLOCKED, and the block is architectural

This is the load-bearing finding of the phase. The plan's rule ("a logical
block ID may be mirrored; its actual tier payload is rank-local") maps onto
the existing `MirroredKv` event stream cleanly, and the per-rank `PoolTier`
plumbing (per-rank `PlaneDesc`s over that rank's own GQA slab planes) is
straightforward. BUT arming any tier or prefix-cache reuse on the TP path
without additional machinery would serve wrong tokens silently for this
pinned model:

Qwen3.8-27B is a hybrid DeltaNet/full-attention model. The TP=1 tier and
prefix cache are ONLY valid because resume restores the DeltaNet recurrent
state + causal-conv window from checkpoint blobs (aux components) at the
resume boundary (`prefix.rs` consult/`RestoreFlow`, then
`restore_paged_state` at `batch.rs:1224` replaying `d_state_pool`
checkpoints into the slot). Under TP there is no DeltaNet checkpoint pool at
all: `DeltaTpRank` keeps per-slot LIVE state only (`slot_state`), there is no
state pool, no snapshot machinery (the probe's `slot_snapshot` is a
probe-only readback), and no aux-restore path. The coordinator also never
mirrors `Publish`/`Reuse` (no radix chains), so there are no chains to
restore from.

Therefore: a KV-only tier under TP would restore attention KV planes while
recomputing DeltaNet layers from zero recurrent state — wrong output for the
pinned hybrid checkpoint, not a graceful degradation. This blocker is
architectural (missing TP DeltaNet aux-checkpoint machinery + missing
Publish/Reuse mirroring on the coordinator), not a GPU-availability issue,
and it stands regardless of Spark availability.

What was deliberately NOT built in this phase (each would be dead code
behind a config flag that can never be safely enabled):

- coordinator-side `Publish`/`Reuse` mirroring;
- per-rank `PoolTier` over GQA slab planes;
- TP DeltaNet checkpoint pool + aux restore flow;
- offload accounting on the tier report.

The correct sequence, when tier work is funded, is: (a) mirror
Publish/Reuse at command boundaries; (b) build the TP DeltaNet
checkpoint-pool + aux restore (the analog of TP=1's `d_state_pool` +
`AuxPlan`/`begin_restore_aux`); (c) arm per-rank tiers with the plan's
mirrored-ID/rank-local-payload rule; (d) then run the two-Spark gates. Steps
(a)+(b) are prerequisite engineering, not validation, and are not started
here to avoid an unsafe partial tier.

`kv_offload.enabled=1` under TP continues to be refused at the startup gate
(exact existing behavior; the refusal is honest: the tier cannot be correct
for the hybrid checkpoint until the machinery above exists).

## Host verification (complete)

- `cargo check -p paddock-dist -p paddock-engine -p paddock-runner` — clean.
- `cargo check ... --tests` — clean.
- `cargo test -p paddock-engine --lib` — 458 passed, 0 failed (includes the
  10 tp_serve tests: 8 pre-existing + 2 new).
- `cargo test -p paddock-dist --lib -p paddock-runner --lib` — 566 passed,
  0 failed.
- `cargo clippy -p paddock-engine --all-targets` — no NEW findings: the 3
  `clippy::unwrap_used` errors and 5 warnings are byte-identical to the clean
  baseline (verified by `git stash` run on the untouched tree).
- New host tests in `tp_serve.rs`:
  - `kv_dtype_from_env_defaults_to_f16_and_demotes_below_sm89_loudly` —
    unset/auto/f16/"" serve f16 regardless of device; fp8 honored on
    sm_121/8_9 (today's allowlist covers every served die); unknown values
    never pick a dtype silently. (The initial test draft asserted an sm_89
    floor that TP=1 does not have; corrected to the actual
    `gpu_support::fp8_kv` semantics before landing — see the doc comment on
    `kv_dtype_from_env`.)
  - `kv_dtype_wire_roundtrip_covers_both_dtypes` — wire string round-trips
    both dtypes; junk fails closed.

## Deferred two-Spark validation (exact gates, not started)

1. FP8 KV TP serve pair (both Sparks, sm_121):
   - env `PADDOCK_KV_CACHE_DTYPE=fp8_e4m3`, both ranks log
     `kv cache: fp8-e4m3` (rank 0 only should log it; rank 1 learns via wire).
   - Acceptance: identical probe discipline as Phase 10/11 —
     `TP rank {0,1} probe` paths complete with the mirrored KV table,
     owned payload, and DeltaNet state bit-identical between eager baseline
     and fp8 serve (fp8 vs fp8), AND fp8-vs-f16 T=0 token parity on a fixed
     prompt set (the serve is allowed to differ from f16 numerically — fp8
     KV is lossy by design — but must produce coherent text with
     `finish_reason: stop`).
   - Measured: per-rank KV pool bytes must be reported by the new
     accounting at HALF the f16 value for the same max_ctx/slots (this is
     the accounting's first real cross-check, do it in the same run).
2. f16 regression pair (default env): Phase 10/11 probe outputs unchanged
   (bit-identical eager baseline discipline), confirming the plumbing change
   is a no-op at the default dtype.
3. Accounting spot-check (same pair as gate 2):
   - `kv_mem_bytes()` returns the exact per-rank context geometry (compare
     against `local_kv_bytes` + `local_state_bytes` sums logged at load).
   - `device_mem_used()` bounded by nvidia-smi for that process;
   - `weights_mem_bytes()` ≥ 0 and (weights + kv) ≥ device_mem_used - slack.
4. Wire-compat sanity: a Phase 11-era runner against a Phase 12 worker must
   FAIL CLOSED with a clear TpError (unknown kv_dtype), not interoperate
   half-wired. One negative test run on the pair is enough.

Required performance measurements (deferred with the gates): fp8 vs f16
decode tokens/s and prefill tokens/s on the pinned two-Spark config; expected
effect is halved per-rank KV bytes doubling usable max_ctx at fixed VRAM, no
throughput regression beyond noise on the attention kernel arm.

## Remaining limitations

- Tier/offload under TP remains refused by design (see blocker above); the
  plan's mirrored-ID/rank-local-payload rule is documented here and the
  implementation path is laid out, not coded.
- FP8 KV under TP has never run on any device in this repo's history; the
  kernel arms exist and the plumbing is dtype-complete, but "kernel-real" is
  static evidence only until gate 1 runs.
- The accounting handshake widened the rank-0 ready tuple; the probe paths
  in tp_serve tests cover the shape, but the tuple is not wire-shared with
  rank 1 (rank 1's process/weights bytes are its own business — the service
  reports rank 0's, matching how the scheduler already treats rank 0 as
  authoritative).
- No cross-rank KV migration was introduced (per plan).
