# Phase 15 — Qwen3.8-27B TP=2 speculative decoding

Status: accepted for the pinned Qwen3.8-27B UD-Q4_K_M checkpoint, TP=2 target, scheduler-owned n-gram speculation, and F16 KV. FP8 KV remains a separate follow-up lane.

## Baseline topology

The target remains the accepted Qwen3.8-27B TP=2 pair. The draft is scheduler-owned n-gram state on rank 0; no draft model is loaded on either TP rank. Rank 0 sends the proposed token rows through the existing TP command channel, and rank 1 replays the same ordered rows. This is the minimum topology change and avoids duplicating the distributed scheduler or introducing a TP=2 draft model.

Greedy verification uses the existing `forward_spec_batch` hook. Sampled verification uses `forward_spec_batch_plans` and the existing device `DevicePlan` representation. Sampling decisions and returned accepted-token IDs remain rank-0 authoritative; rank 1 participates in the same target forward/NCCL sequence but does not sample independently.

A rejection stops issuing further target rows for that request. Returned picks are padded only to preserve the generic scheduler's existing ragged-result contract; no post-rejection target row is executed. This keeps both ranks at the accepted target prefix without introducing cache rollback.

## Implementation

- Added TP serving hooks for greedy and device-plan speculative verification.
- Added a `SpecPlans` TP command and rank-0 coordinator dispatch.
- Reused existing row execution, KV authorization, sequence/ready handshakes, target sampling, and worker replay paths.
- Unsupported request shape, position, plan count, or slot ordering fails closed.
- The implementation does not change generic scheduler/speculation policy or draft placement.

## Verification so far

- `qwen35_tp_spec` direct two-rank probe passed on the pinned checkpoint with F16 KV: greedy zero/partial/full acceptance, page-crossing positions, and rank-symmetric target execution.
- The same probe passed sampled fixed-plan verification after the TP position cursor fix; the sampled oracle's token IDs and plan progression matched, and rank 1 never owned a sampler.
- The direct probe passed with CUDA graphs enabled and with `PADDOCK_UNIFIED=1`; both ranks exited cleanly.
- `qwen35_two_slot_oracle` passed the existing two-slot page/cancellation/release/reuse/reset non-spec lifecycle oracle with `PADDOCK_NO_SPEC=1`.
- Normal HTTP serving passed with speculation enabled (`--spec on`), CUDA graphs, `PADDOCK_UNIFIED=1`, two concurrent requests, cancellation, and clean rank shutdown. The runner TP gate now accepts an explicit spec policy while retaining fail-closed default behavior.
- The standalone `qwen35_tp_overlap` probe passed with and without `PADDOCK_UNIFIED`; the earlier rank-0 exit 139 reproduced only with the stale CUDA pack and disappeared after rebuilding the pack for the current engine.
- Release serving benchmark (F16, TP=2, max batch 2, two concurrent requests): speculative mode drafted 18 and accepted 15 tokens (83.3% acceptance), with sequential request times 5.14–5.16 s and concurrent times 9.91–9.99 s. The matched non-spec run took 5.04–5.06 s sequential and 9.68–9.81 s concurrent; this run shows a small speculative overhead, not a speedup. Metrics did not indicate a draft-placement bottleneck.
- Host `cargo check --workspace --all-targets`, `cargo test -p paddock-engine --lib` (460 passed, 0 failed), targeted example checks, and `git diff --check` passed after the final implementation and rebuilt CUDA pack.

## Required acceptance evidence

The direct correctness and serving gates are covered above. The rebuilt CUDA pack was used for the final serving smoke and benchmark. FP8 KV was intentionally deferred until the accepted F16 speculative lane has a separate need.

## Future work

Only if the baseline measurements show a material draft bottleneck should a replicated or TP-partitioned draft model be considered. That optimization is intentionally deferred.
