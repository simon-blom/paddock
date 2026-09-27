# Qwen3.8 TP=2 wider span-cap sweep

Date: 2026-09-28
Tested HEAD: `a882ab99f5903d83ca742bd66cd4bd173325e3b2` (`tp: batch DeltaNet alpha beta preparation on prefill spans`)

## Scope and setup

This was intended as an unprofiled cold-chat sweep of `PADDOCK_TP_SPAN_CAP` at 384, 512, 1024, and 2048. The production default was not changed and `PADDOCK_TP_PREFILL_PROFILE` was unset for every attempt.

The launcher used the existing two-node TP=2 configuration with:

- Qwen3.8-27B-UD-Q4_K_M
- F16 KV, max context 65,536, max batch 2
- unified mode (`PADDOCK_UNIFIED=1`), speculation off, TP graph off
- unchanged fixture `/home/sime/.hermes/cache/scratch/qwen38-hermes-20k-prompt.txt`
- `enable_thinking=false`, temperature 0, seed 1, max output 64
- expected prompt size: 22,130 tokens

The existing launcher performed its normal rank-0/rank-1 bootstrap. Before the second attempt, the exact current runner and CUDA pack were staged to the configured worker paths because the pre-existing remote runner hash was stale. The resulting runner hash was `08a8f015b2b9e4236000ba3687fee3a65e66d8baaa20e51dfe7fd0bf311cd89b` on both nodes; the pack hash was `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e` on both nodes.

## Results

The requested wider values are rejected during engine startup by the existing explicit span-cap gate. No HTTP server became ready, so there are no cold-chat wall times, prompt/completion token counts, prompt throughput values, generated outputs, or output hashes for these four caps.

| Span cap | Cold wall | Prompt tok/s | vs 192 | Output | Startup / errors / notes |
|---:|---:|---:|---:|---|---|
| 64 | 102.973 s historical | 214.9 effective historical | 0.339x | historical | Existing optimized result; production default remains 64 |
| 192 | 34.823 s historical | 635.5 effective historical | baseline | historical | Existing optimized result; fastest previously measured lane |
| 384 | N/A | N/A | N/A | N/A | Startup failed: `PADDOCK_TP_SPAN_CAP=384 unsupported: use one of [64, 128, 192] (default 64)`; coordinator 1, worker 1 |
| 512 | N/A | N/A | N/A | N/A | Startup failed: `PADDOCK_TP_SPAN_CAP=512 unsupported: use one of [64, 128, 192] (default 64)`; coordinator 1, worker 1 |
| 1024 | N/A | N/A | N/A | N/A | Startup failed: `PADDOCK_TP_SPAN_CAP=1024 unsupported: use one of [64, 128, 192] (default 64)`; coordinator 1, worker 1 |
| 2048 | N/A | N/A | N/A | N/A | Startup failed: `PADDOCK_TP_SPAN_CAP=2048 unsupported: use one of [64, 128, 192] (default 64)`; coordinator 1, worker 1 |

The historical cap-192 result is 2.957x faster than the historical cap-64 result, or 65.9% lower wall time. Effective prompt throughput from the 22,130-token workload is approximately 214.9 tok/s at cap 64 and 635.5 tok/s at cap 192. These calculations are historical context, not new measurements from this sweep.

## Failure classification and logs

Each cap was attempted with a fresh launcher/server lifecycle. Rank 0 reached TP bootstrap, connected rank 1, and began model loading. Engine startup then rejected the value before HTTP readiness. This is an explicit configured-size allowlist restriction, not an allocation/OOM, CUDA launch, NCCL, protocol, checkpoint/cache, or generated-output failure.

Observed per-cap process status:

- 384: coordinator exit 1, worker exit 1
- 512: coordinator exit 1, worker exit 1
- 1024: coordinator exit 1, worker exit 1
- 2048: coordinator exit 1, worker exit 1

No CUDA errors, NCCL errors, cache/checkpoint errors, protocol errors, OOM/allocation failures, or unusual runtime warnings appeared in the captured launcher logs. No span/checkpoint cuts were reached because model engine startup rejected each value before serving a request. Peak memory was not available because no benchmark request ran.

The source of the restriction is `crates/paddock-engine/src/gpu_model/qwen35/tp_span_cap.rs`: `SWEEP_CAPS` is `[64, 128, 192]`, `MAX_TP_SPAN_CAP` is 192, and both environment and wire resolution fail closed outside that set. No production code was changed to bypass this restriction; doing so would be outside this mechanical measurement pass and would require revalidating the larger-cap geometry and allocations.

Per-cap raw evidence is under:

`/home/sime/.hermes/cache/scratch/span-sweep-20260928-012910/`

The earlier pre-staging attempts are preserved under:

`/home/sime/.hermes/cache/scratch/span-sweep-20260928-012835/`

## Output comparison

No new 64-token outputs were generated because none of the requested caps reached HTTP serving. Therefore byte-identical/tail-different/materially-different comparison is not applicable. The optimization report records that the historical cap-64 and cap-192 outputs differed across caps on both binaries; generated-text equality is not treated as a strict numerical correctness oracle.

## Multi-request smoke test

Not run. There was no successful cap in the requested 384/512/1024/2048 sweep at which to run the requested two-simultaneous-request smoke test. No claim is made about multi-request behavior at the rejected values. The prior report's separate cap-192 two-slot smoke result remains historical evidence only.

## Measurement-supported conclusion

- The requested wider values cannot currently be benchmarked on this branch: all four fail the explicit `[64, 128, 192]` span-cap gate during startup.
- No conclusion about performance improving, flattening, or regressing above 192 is supported by this run.
- No requested size produced a measured failure from allocation, CUDA, NCCL, protocol, or runtime execution; all four failed earlier as explicit unsupported configuration values.
- The fastest successful cap with existing measured evidence remains historical cap 192 at 34.823 s, not a new result from this sweep.
- The production default remains 64. This report does not recommend changing it.
