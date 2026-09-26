# TP=2 batched-prefill production review

Reviewed branch `review/qwen38-tp2-final` from starting HEAD `71837643c37153f365a10cdbb4b7465049cbbffe`; history inspected from `16687343baf51abb2f05c123d1b3dec0349e0f69` through that HEAD, particularly `18b2901` and `7183764`. Final commit and remote identity: see the final review handoff (this report is committed before target smoke).

## Findings, ordered by severity

- Blocker: no static defect confirmed in the inspected serving traversal. Target-device production behavior remains an acceptance gate, not a proven pass.
- High: no confidently fixable defect confirmed. Rank-specific GPU failure after Prepared can still leave the peer in a collective; the TCP timeout is not a NCCL timeout. A two-rank target run is required.
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

## Open target gates / readiness

Production smoke must establish both-rank startup/exits, ACK progression, actual mixed/overlap routes, first decode, 64-boundary/repeated spans, page crossing, nonzero slot/concurrent decode, cancellation/reuse, greedy sanity and measured TTFT. Direct TP=1 token/logit parity is distinct from HTTP text sanity; known serial GEMV-vs-batched GEMM numerical-class differences remain unchanged and tolerances were not altered. The ~20k Hermes test is **not ready** until the production smoke passes. No 20k test belongs to this review session.
