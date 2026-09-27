Qwen3.8 TP1 profiling comparison against validated TP2@192

Date: 2026-09-27
Branch: review/qwen38-tp2-final

Scope

This is a testing-only comparison. The default span cap was not changed. TP1 was run as a fresh single-rank server with the same Qwen3.8-27B-UD-Q4_K_M model, CUDA pack, F16 KV, MAX_CTX=65536, max batch 2, SPEC=off, PADDOCK_UNIFIED=1, and the exact 22,130-token Hermes acceptance fixture used for TP2 validation.

The TP1 profiler is opt-in under PADDOCK_TP_PREFILL_PROFILE=1. It uses CUDA events and reports unified-span work before final norm/lm-head/sampling, matching the TP2 whole-advance boundary. TP1 has no NCCL intervals; NCCL is therefore zero by definition. The TP1 categories are local model-stage buckets. The small `other` value is derived as whole advance minus the three stage buckets; it includes unfenced stage transitions and unified-span work not assigned to GQA, DeltaNet, or FFN. It is not a claim that every instruction in that remainder is separately timed.

Implementation

- `crates/paddock-engine/src/gpu_model/qwen35/tp_prefill_profile.rs`: added the opt-in single-rank CUDA-event profile accumulator.
- `crates/paddock-engine/src/gpu_model/qwen35/batch.rs`: instrumented the TP1 unified-span lane, with whole timing ending before final norm/head/sampling and stage boundaries for GQA, DeltaNet, and FFN.
- `crates/paddock-engine/src/gpu_model/qwen35/prefix.rs`: retained the same opt-in stage machinery for the serial prefill path; the tested 22,130-token TP1 request used the unified-span lane.

TP1 profiled cold request

The first two profile records were the server's 300-row warmup (4 + 296 rows). The values below subtract those records from the cumulative profile, leaving exactly 22,130 request rows.

| Component | TP1 |
|---|---:|
| GQA local | 5.121 s |
| DeltaNet local | 7.702 s |
| FFN local | 13.957 s |
| Other (derived remainder) | 0.012 s |
| Whole prefill/advance | 26.792 s |
| Request wall | 32.086 s |
| Prompt throughput (22,130 / wall) | 689.7 tok/s |

TP1 profiler evidence: `/home/sime/.hermes/cache/scratch/tp1-prefill-profiled-server.log`.
TP1 profiled request JSON: `/home/sime/.hermes/cache/scratch/tp1-prefill-profiled-result.json`.

Direct component comparison

TP2 values are the validated cap-192 profiled values from the handoff. TP1 NCCL is zero by definition. TP2 total is local + NCCL.

| Component | TP1 | TP2@192 local | TP2 NCCL | TP2 total | TP2 total / TP1 |
|---|---:|---:|---:|---:|---:|
| GQA | 5.121 s | 3.410 s | 0.671 s | 4.081 s | 0.797x |
| DeltaNet | 7.702 s | 17.999 s | 2.377 s | 20.376 s | 2.646x |
| FFN | 13.957 s | 9.290 s | 2.488 s | 11.778 s | 0.844x |
| Whole prefill/advance | 26.792 s | 37.155 s | — | 37.155 s | 1.387x |

The TP2 category sum is 36.235 s, leaving 0.920 s of TP2 whole-advance work outside the three reported component buckets. This is a category remainder, not an additional NCCL measurement.

TP1 unprofiled cold request

- HTTP status: 200
- Prompt tokens: 22,130
- Completion: 64 tokens
- First generated token/output: matched the TP2 cap-192 baseline; output was coherent and identical to the acceptance fixture result.
- Wall: 32.175 s
- Prompt throughput: 687.8 tok/s
- Result JSON: `/home/sime/.hermes/cache/scratch/tp1-prefill-unprofiled-result.json`
- Server log: `/home/sime/.hermes/cache/scratch/tp1-prefill-unprofiled-server.log`

TP2@192 unprofiled reference

- Wall: 48.441 s
- Prompt throughput: 456.8 tok/s

TP2/TP1 wall ratio: 48.441 / 32.175 = 1.5055x.

Interpretation

- The largest avoidable TP2 overhead is DeltaNet: TP2 total is 20.376 s versus 7.702 s on TP1, a +12.674 s difference. It consists of +10.297 s TP2 local-compute overhead and +2.377 s NCCL.
- FFN does not show an intrinsic TP2 penalty in this measurement: TP2 total is 11.778 s versus 13.957 s TP1. Its TP2 NCCL cost is offset by lower TP2 local compute.
- GQA likewise has lower TP2 total than TP1 (4.081 s versus 5.121 s), despite 0.671 s NCCL.
- Whole TP2 advance is 37.155 s versus 26.792 s TP1. The difference is dominated by DeltaNet and the TP2 remainder/category boundary, not by NCCL alone.
- This comparison does not establish that TP2 is universally slower: it is specific to this model, fixture, two-node placement, and validated cap-192 lane. The next optimization target should be DeltaNet TP2 local execution first, with its NCCL interval treated as a separately measurable secondary target.

Validation and hashes

- `cargo check -p paddock-engine`: PASS
- `cargo build --release -p paddock-runner --bin paddock-runner`: PASS
- HTTP 200 and coherent deterministic output: PASS for profiled and unprofiled TP1 runs.
- No CUDA, KV, cache, NCCL, protocol, or checkpoint errors observed in the TP1 server logs.
- Runner SHA-256: `3cc229c26dc72c64f6a71668fed14271ecc807b3e54cf0a4079ac4eb6a1c65d5`
- CUDA pack SHA-256: `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e`
- Model SHA-256: `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`
