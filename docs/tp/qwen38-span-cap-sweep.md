# Qwen3.8 TP=2 wider span-cap sweep

Date: 2026-09-28
Starting HEAD: `a9172c1dde9efb5d5aa8ca280a0e174dba488f11`
Allowlist commit: `28938243ab3744e141867939f8e1fc70441f7b4f` (`tp: permit wider experimental span caps`)

## Scope and setup

This was an unprofiled fresh-server cold-chat sweep of `PADDOCK_TP_SPAN_CAP` at 384, 512, 1024, and 2048. `PADDOCK_TP_PREFILL_PROFILE` was unset for every headline run. The production default remains 64.

The launcher used the existing two-node TP=2 configuration with:

- Qwen3.8-27B-UD-Q4_K_M
- F16 KV, max context 65,536, max batch 2
- unified mode (`PADDOCK_UNIFIED=1`), speculation off, TP graph off
- unchanged fixture `/home/sime/.hermes/cache/scratch/qwen38-hermes-20k-prompt.txt`
- `enable_thinking=false`, temperature 0, seed 1, max output 64
- expected prompt size: 22,130 tokens

The exact release runner and CUDA pack were staged to the configured worker path before the sweep. Runner SHA-256 was `54864930e28cca4b859efe35b898f395d5d8b0f6cc1bb960e26ba5576f215207` on both nodes. CUDA pack SHA-256 was `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e` on both nodes.

## Phase 1: guard inspection and minimal change

The previous failure was an explicit validation gate, not a discovered CUDA or kernel limit. `tp_span_cap.rs` resolved the coordinator environment value and the rank-1 wire value through `SWEEP_CAPS`; the old array was `[64, 128, 192]`. Span-sized GPU allocations already derived their dimensions from the resolved runtime cap, including the TP span planes, DeltaNet prefill scratch, quantization staging, and chunked-scan scratch. Relevant allocations use expressions such as `cap * width` and `(cap.div_ceil(64) + 32) * ...`; no separate 192-row static buffer or kernel guard was found.

The smallest experimental change was therefore limited to `tp_span_cap.rs`:

- `DEFAULT_TP_SPAN_CAP` remains 64.
- `SWEEP_CAPS` is now exactly `[64, 128, 192, 384, 512, 1024, 2048]`.
- `MAX_TP_SPAN_CAP` is updated from 192 to 2048 as the documented largest experimental geometry.
- `SpanCapError.allowed` and the parser unit expectation were widened to seven entries.
- Arbitrary values remain rejected by the same allowlist and wire validation.

No kernel, scheduler, protocol, or allocation implementation was redesigned. Host validation passed:

- `cargo test -p paddock-engine tp_span_cap --lib`: 3 passed
- `cargo test -p paddock-engine span_cap_geometry_boundaries_at_every_sweep_width --lib`: 1 passed
- `cargo test -p paddock-engine span_cap_wire_install_fails_closed_and_keeps_last_good --lib`: 1 passed
- `cargo build --release -p paddock-runner --bin paddock-runner`: passed
- `git diff --check`: passed

## Benchmark results

Historical 64 and 192 values are from the optimized-TP report and are not new runs in this sweep. New timings are client wall time for the complete cold chat request, including 64 generated tokens.

| Cap | Cold wall | Effective prompt tok/s | Speedup vs 192 | Speedup vs 64 | Output | Notes |
|---:|---:|---:|---:|---:|---|---|
| 64 | 102.973 s historical | 214.9 historical | 0.338x | baseline | historical | Production default |
| 192 | 34.823 s historical | 635.5 historical | baseline | 2.957x | historical | Historical optimized result |
| 384 | 32.767 s new | 675.4 | 1.063x | 3.143x | `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84` | HTTP 200; prompt 22,130; completion 64 |
| 512 | 33.264 s new | 665.3 | 1.047x | 3.096x | `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84` | HTTP 200; prompt 22,130; completion 64 |
| 1024 | 36.291 s new | 609.8 | 0.960x | 2.837x | `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84` | HTTP 200; prompt 22,130; completion 64; regression vs 192 |
| 2048 | 41.760 s new | 529.9 | 0.834x | 2.466x | `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84` | HTTP 200; prompt 22,130; completion 64; regression vs 192 |

Effective throughput is `22,130 / cold wall`. Relative to historical cap 192, the new caps changed wall time by:

- 384: 2.056 s lower, 5.9% reduction
- 512: 1.559 s lower, 4.5% reduction
- 1024: 1.468 s higher, 4.2% regression
- 2048: 6.937 s higher, 19.9% regression

All four new caps started successfully, returned HTTP 200, produced 22,130 prompt tokens and 64 completion tokens with `finish_reason=length`, and generated the same output hash. The output text was byte-identical across the four new caps. This is output evidence only, not a strict numerical-correctness oracle.

The launcher was stopped after each completed request by signaling the exact runner child; its final process status was coordinator/worker `143` for the four headline runs because the benchmark harness terminated the server after collecting the response. The server logs show normal drain and device-memory release for each run. No request failed. Raw evidence:

`/home/sime/.hermes/cache/scratch/span-sweep-20260928-014349/`

Visible production route behavior included 8,192-row scheduler chunks for the long prompt, followed by a 5,746-row finishing chunk, checkpoint cuts at 22,112 and 22,128, and normal decode-pipe drain. No span-specific runtime error occurred.

No peak memory metric was available from existing instrumentation, so no peak-memory number is claimed. The allocations are cap-sized by construction; the 2048 run reached HTTP 200 and completed without OOM/allocation failure.

## Errors and warnings

No CUDA errors, NCCL errors, protocol errors, state/poison errors, checkpoint/cache errors, OOMs, allocation failures, or kernel launch failures were observed in the four new benchmark logs. The only warnings were the existing tick-stall warnings during long mixed/prefill work; they occurred for all caps and are not failure indicators. Coordinator and worker bootstrap completed for every run.

## Multi-request smoke at fastest new cap

The fastest new cap was 384 at 32.767 s. A two-simultaneous-request smoke test was run at `PADDOCK_TP_SPAN_CAP=384` using the established decode-rider plus long asynchronous prompt style:

- decode-rider: HTTP 200, prompt 71 tokens, completion 128 tokens, `finish=length`
- async-long-2200: HTTP 200, prompt 5,552 tokens, completion 8 tokens, `finish=length`
- both slots were exercised: logs show slot 0 Mixed and slot 1 Async ownership
- both requests completed decode after prefill; slot 1 logged prefill-span launch, finish, promotion to decode, and subsequent decode-pipe activity
- coordinator exit 0, worker exit 0
- no CUDA, NCCL, protocol, state, checkpoint, or cache errors

Smoke evidence:

`/home/sime/.hermes/cache/scratch/span-smoke-cap384-20260928-014952/`

## Conclusions supported by this sweep

- The allowlist was the only blocker to measuring the wider values; all four larger caps run successfully without changing kernels.
- Performance improves modestly from 192 to 384, with 512 slightly slower than 384 but still faster than 192.
- Improvement flattens at 384–512 and reverses at 1024.
- 1024 and 2048 are successful runtime measurements, not memory/kernel cliffs, but both regress against cap 192; 2048 is 19.9% slower than 192.
- The fastest newly measured cap is 384 at 32.767 s.
- The fastest successful cap in the combined historical/new table is 384, but this does not justify changing the production default. Resumed-prefix/state correctness qualification remains outstanding.

No recommendation to change the production default is made.
