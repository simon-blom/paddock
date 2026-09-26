# TP=2 batched-prefill production review

Reviewed branch `review/qwen38-tp2-final` from starting HEAD `71837643c37153f365a10cdbb4b7465049cbbffe`; history inspected from `16687343baf51abb2f05c123d1b3dec0349e0f69` through that HEAD, particularly `18b2901` and `7183764`. Final commit and remote identity: see the final review handoff (this report is committed before target smoke).

## Findings, ordered by severity

- Blocker: no static defect confirmed in the inspected serving traversal. Target-device production behavior remains an acceptance gate, not a proven pass.
- High/operational: the configured remote runner was stale: SHA-256 `50fe4a17…` versus the rebuilt local `558c17ac…`. The launcher previously started it without an identity preflight. A fail-closed remote runner/pack hash check and remote path preflight now run before starting either rank. For smoke, an exact-hash copy was staged under the worker's cache and selected with `REMOTE_RUNNER`; no existing worker binary was overwritten.
- High/runtime: no confidently fixable model/protocol defect confirmed. Rank-specific GPU failure after Prepared can still leave the peer in a collective; the TCP timeout is not a NCCL timeout. A two-rank target run is required.
- Medium/test quality: the span-cap test parsed source text for a private constant and could fail merely on a formatting change. Replaced it with a direct compile-time comparison of actual constants. Other tests in `tp_serve.rs` reconstruct finisher arithmetic and compare a pure chunker to itself; these establish geometry but do not exercise a running scheduler or GPU event/promotion path. Do not mistake them for a two-rank test.
- Low: `tp_span.rs` still described production-owned scratch as a future prototype; `protocol.rs` described a serial/bootstrap-only channel and no pipelining; the lane-promotion comment asserted F16 even though TpInit carries the chosen dtype. Comments corrected. Opt-in `tp_trace` still has real ABC-probe consumers; removing it in this review would sacrifice a diagnostic rather than eliminate dead code.

## Static path trace

`TpCoordinator::forward_mixed` validates decode rows, the disjoint prompt run and cumulative positions, sends v3 `TpMixed { chunk_rows }` with one end-of-tick KV snapshot, and enters decode traversals then bounded prompt spans. The worker validates run geometry and mirrors the full ordered Ensure list before Prepared, then enters the same decode/span order. `span_begin` and the worker's `TpSpanLaunch` handler split contiguous runs with the shared `span_chunk_points`; only rank 0 runs final norm/head/sampling. `span_finish` joins the collective and lane streams on both ranks before copying finished slots' live KV blocks and recurrent/conv state lane-to-decode. The decode stream waits the lane event; the next lane enqueue waits the promotion-drained event. Both sides consume the launch tail Ready before a subsequent command. See `tp_serve.rs:1071-1254,1360-1583,2502-2666`, `tp_model.rs:1010-1275,1522-1607,1675-1839`, `gqa_tp.rs:640-934`, `delta_tp.rs:622-660`, and `protocol.rs:70-203` at the starting HEAD (line positions shift slightly after this review).

Intermediate spans skip the head and readback. No prompt-row whole-model serial loop remains in the mixed or lane span traversal; decode/speculation still use one-token forwards. GQA uses the span-owned paged table for append and attention and stages text positions on all four M-RoPE axes. DeltaNet prefill uses the capacity-backed input mode; decode retains its exact-size mode. The host tests cover 1/63/64/65 and repeated geometry, mirror/page crossing, occupancy/release selection, protocol version, missing `chunk_rows`, frame cap and bootstrap. The current host tests do not prove runtime collective pairing, numerical equality, live cancellation or throughput.

## Validation

- `cargo test -p paddock-engine --lib`: pass (481 tests after replacing the brittle cap test with a compile-time assertion).
- `cargo test -p paddock-engine --all-targets`: pass; some model-dependent/heavy tests self-skip.
- `cargo test -p paddock-dist`: pass (4 unit + 22 bootstrap).
- `cargo clippy -p paddock-engine -p paddock-dist --all-targets`: pass with existing unrelated warnings (`cuda.rs`, `qwen35_tp_spec.rs`, `qwen35_two_slot_oracle.rs`).
- `git diff --check`: pass.
- `cargo build --release -p paddock-runner --bin paddock-runner`: pass on the post-edit source.

## Production GPU smoke (two DGX Spark ranks, SPEC=off, TP_GRAPH=0)

The dry run passed. Local and staged worker runner hashes matched (`558c17ac6305e25a0f655131d15eed0a4ee20a329df0f60868cff0915baef7f2`); local and worker pack SHA-256 matched (`c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e`). The configured older worker runner was left untouched. Rank 0 joined the worker and `/v1/models` returned HTTP 200. Prompt-token counts below are from response usage, not word estimates:

- 4 prompt / 8 completion tokens: 0.557 s, output returned.
- 71 prompt / 1 completion tokens: 0.392 s; repeated `alpha` prompted immediate empty-text termination, so this is a structural boundary check, not greedy prose sanity.
- 250 prompt / 24 completion tokens: 2.608 s, coherent Rayleigh-scattering answer across several spans/pages.
- 1051 prompt / 1 completion tokens: 4.802 s; same repeated-word early termination caveat.
- Concurrent 15398-prompt-token request plus 9-prompt-token request: 78.523 s and 78.136 s wall respectively; both returned 32 completion tokens and coherent responses. Aggregate long-request prompt tokens / request wall = 196.1/s, NOT an isolated prefill throughput or TTFT metric. Coordinator logged two mixed-phase stalls (37795 and 34554 ms), a decode pipe begin/drain (sequence 108/136), and no span-launch event. This exercise used `mixed` and decode pipe, not a demonstrated async prefill-lane overlap. No NCCL/KV error appeared in the coordinator log.

The coordinator terminated gracefully and freed device memory when the launcher was stopped; the worker pidfile was removed and the API stopped responding. The worker was terminated by launcher cleanup; its log does not contain a recorded numeric exit status. Neither a live cancellation/reuse trace nor a direct TP=1 token/logit oracle was run. Rank-1 KV mirror and first decode are indirectly exercised by the completed requests, not independently traced. No ~20k Hermes request was sent.

A second, explicitly gated run set `PADDOCK_UNIFIED=1`: with a 12-token prompt decoding 128 tokens while a 1370-token prompt arrived, both returned coherent responses (22.166 s and 9.552 s respectively). Coordinator logs confirm `TP prefill span launched sequence=38 rows=1370 finishers=1`, `TP slot-mapped pipe began sequence=39 rows=1`, pipe drain sequence 40 and span finish sequence 41. This exercises repeated 64-row sub-spans, nonzero-slot/concurrent decode, lane completion/promotion, and the first decode after finisher without a visible NCCL or page failure; the logs demonstrate overlapping protocol flights, not measured simultaneous GPU execution. Separate tokenizer-counted requests of 1, 63, 64 and 65 prompt tokens returned one completion token each, without errors. Both runs stopped cleanly on the coordinator side; the remote pidfile disappeared. Numeric rank-1 exit status remains unrecorded.

## Open target gates / readiness

Streaming cancellation while a span is active followed by release/reuse, direct TP=1 oracle parity, both-rank numeric exit codes, protocol frame byte measurement, and independently timed TTFT remain open. Production `mixed` and opt-in unified-span smoke passed without observed hangs, but the branch is **not ready for the ~20k Hermes acceptance test** until these ownership and parity gates are established. Known serial GEMV-vs-batched GEMM numerical-class differences remain unchanged and tolerances were not altered.
