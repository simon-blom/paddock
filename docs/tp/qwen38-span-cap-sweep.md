# Qwen3.8 TP=2 wider span-cap sweep

Date: 2026-09-28
Starting HEAD for this follow-up: `74e385c94d18d9a788e73141c9ce2fb6ae396eec`
Prior allowlist commit: `28938243ab3744e141867939f8e1fc70441f7b4f`
New allowlist commit: `065db8b029ed316e78ac6aa37840c9bdebe8c2d0`

## Scope and setup

The preceding sweep already measured 384, 512, 1024, and 2048. This follow-up therefore measured only 4096 and 8192; no previously successful span width was retested.

The benchmark used the same fresh-server, unprofiled TP=2 cold-chat methodology and configuration:

- Qwen3.8-27B-UD-Q4_K_M
- F16 KV, max context 65,536, max batch 2
- unified mode (`PADDOCK_UNIFIED=1`), speculation off, TP graph off
- unchanged fixture `/home/sime/.hermes/cache/scratch/qwen38-hermes-20k-prompt.txt`
- `enable_thinking=false`, temperature 0, seed 1, max output 64
- expected prompt size: 22,130 tokens
- `PADDOCK_TP_PREFILL_PROFILE` unset for headline timings

The exact release runner and CUDA pack were staged to the configured worker path. Runner SHA-256 was `373bf08f3462660be7d4aa95dbdf67b106e7be725978aa82ef91a8a760276849` on both nodes. CUDA pack SHA-256 remained `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e` on both nodes.

## Guard inspection and minimal change

The existing span-sized allocations remain runtime-cap based. The relevant paths allocate proportional to `cap * width`, including rows × 5120 activation/projection planes, rows × local head count alpha/beta and DeltaNet planes, convolution staging, quantized activation scratch, and chunked-scan scratch sized from `cap.div_ceil(64) + 32`. No separate 2048-row static buffer or kernel guard was found.

At the largest requested value, the basic arithmetic is still well within `usize`: 8,192 × 5,120 is 41,943,040 f32 elements / 160 MiB per f32 plane, and the chunked-scan allocation uses 160 chunk slots including padding. The state-sized scratch is large but uses the same checked allocator path. The expected risk is device memory capacity, not integer overflow or an obvious fixed-shape assumption.

The new code change is limited to `tp_span_cap.rs`:

- `DEFAULT_TP_SPAN_CAP` remains 64.
- `SWEEP_CAPS` now exactly contains `[64, 128, 192, 384, 512, 1024, 2048, 4096, 8192]`.
- `MAX_TP_SPAN_CAP` is 8192.
- `SpanCapError.allowed` and its parser test cover all nine values.
- Arbitrary values remain rejected through the same environment and wire allowlist.

No kernel, scheduler, protocol, or allocation implementation was redesigned. Host validation passed:

- `cargo test -p paddock-engine tp_span_cap --lib`: 3 passed
- `cargo test -p paddock-engine span_cap_geometry_boundaries_at_every_sweep_width --lib`: 1 passed
- `cargo build --release -p paddock-runner --bin paddock-runner`: passed
- `git diff --check`: passed

## Complete results

Historical 64 and 192 values are from the optimized-TP report. The 384–2048 rows are measurements from the preceding sweep. Only 4096 and 8192 are new measurements in this follow-up. Effective throughput is `22,130 / cold wall`.

| Span cap | Cold wall | Effective prompt tok/s | Incremental vs previous measured | Speedup vs 192 | Speedup vs 64 | Output | Notes |
|---:|---:|---:|---:|---:|---:|---|---|
| 64 | 102.973 s historical | 214.9 historical | — | 0.338x | baseline | historical | Production default |
| 192 | 34.823 s historical | 635.5 historical | 68.150 s lower vs 64 | baseline | 2.957x | historical | Historical optimized result |
| 384 | 32.767 s preceding sweep | 675.4 | 2.056 s lower vs 192 | 1.063x | 3.143x | `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84` | HTTP 200; prompt 22,130; completion 64 |
| 512 | 33.264 s preceding sweep | 665.3 | 0.497 s higher vs 384 | 1.047x | 3.096x | same | HTTP 200; prompt 22,130; completion 64 |
| 1024 | 36.291 s preceding sweep | 609.8 | 3.027 s higher vs 512 | 0.960x | 2.837x | same | HTTP 200; prompt 22,130; completion 64 |
| 2048 | 41.760 s preceding sweep | 529.9 | 5.469 s higher vs 1024 | 0.834x | 2.466x | same | HTTP 200; prompt 22,130; completion 64 |
| 4096 | 53.663 s new | 412.4 | 11.903 s higher vs 2048 (+28.5%) | 0.649x | 1.919x | `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84` | HTTP 200; prompt 22,130; completion 64; major regression |
| 8192 | N/A | N/A | Failed at startup after 4096 | N/A | N/A | N/A | CUDA OOM during TP worker startup; coordinator exit 137, worker exit 1 |

The 4096 request returned HTTP 200 with 22,130 prompt tokens, 64 completion tokens, `finish_reason=length`, and the same output hash as the preceding successful caps. The wrapper launcher status was 143 because the benchmark harness stopped the server after collecting the response; server logs showed normal drain and device-memory release. The forced-stop path did not preserve a separate per-cap worker status line.

Raw follow-up evidence:

`/home/sime/.hermes/cache/scratch/span-sweep-large-20260928-015758/`

## 2048 → 4096 → 8192 scaling

- 2048 → 4096 regressed from 41.760 s to 53.663 s: 11.903 s slower, approximately 28.5% slower, and effective throughput fell from 529.9 to 412.4 tok/s (approximately 22.2% lower).
- 4096 → 8192 did not produce a request timing: 8192 failed during startup before HTTP readiness.
- 8192 reached the expected memory cliff during model/scratch initialization. The worker log reports `CUDA out of memory (CUDA_ERROR_OUT_OF_MEMORY)`; the coordinator was killed with status 137 and the worker exited 1. No CUDA kernel execution, request, NCCL protocol, checkpoint, or output comparison was reached at 8192.

The only warnings in the successful 4096 run were the existing long mixed/prefill tick-stall warnings. No NCCL, protocol, state, checkpoint/cache, or kernel-launch errors were observed. No peak-memory metric was available from existing instrumentation, so no peak-memory number is claimed.

No targeted CUDA-event profiling was run. The 4096 regression and 8192 startup OOM already establish the relevant mechanical scaling boundary, and profiling would change headline timing.

## Serving smoke

The fastest measured width remains the preceding-sweep cap 384 at 32.767 s. Neither newly tested width became the fastest, so no additional serving smoke was run in this follow-up. The preceding cap-384 smoke remains documented above in the earlier version of this report, with both simultaneous requests HTTP 200 and clean coordinator/worker exits.

## Conclusions supported by the complete sweep

- Performance improved through cap 384, with a small flattening/regression at 512 and increasingly worse timings from 1024 through 4096.
- The 2048 → 4096 step is a clear regression, not a continued scaling benefit.
- 8192 is not a successful serving measurement: it hits a CUDA out-of-memory failure during TP worker startup.
- The fastest measured width remains 384 at 32.767 s.
- Multi-thousand-row spans are mechanically possible, but this TP path's measured optimum is far below the upstream 8192 main-chunk value, and the upstream value does not predict TP optimum here.
- The production default remains 64. No recommendation to change it is made; resumed-prefix/state correctness qualification remains outstanding.
