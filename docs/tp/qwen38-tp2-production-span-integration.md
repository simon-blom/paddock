# TP=2 Qwen3.5 production batched-prefill integration

Source: `review/qwen38-tp2-final`, based on `d1c16952ce6d4e992e2d119d054d1c20e5baab83`. This is a static/host handoff; no production two-node GPU run or numerical acceptance is claimed here.

## Execution before and after

Previously `TpCoordinator::forward_mixed` called a complete `forward_token_*`/`forward_token_enqueue` model traversal for every prompt row in the scheduler's chunk budget. `TpCoordinator::span_begin` called `prefill_lane_step` once per row; rank 1 mirrored both loops. The one-slot serial `prefill()` similarly forwarded every token separately. A long ~20k-token prompt was observed spending ~641 s in `mixed`. Asynchronous enqueue alone did not batch model weight reads, GQA/DeltaNet/FFN calls, or per-layer collectives.

The coordinator now takes the scheduler's existing `chunk_take(budget)` rows, which form a contiguous run for one slot. Mixed ticks execute decode rows unchanged and then partition their trailing prompt run into windows of at most `tp_span::TP_SPAN_CAP` (64). The `TpMixed` v3 command carries `chunk_rows` so the worker splits the same ordered wire rows and derives identical windows; a serial prefill uses the same message with zero decode rows. `TpSpanLaunch` retains its wire shape and partitions contiguous slot runs using the same chunker on both ranks. Decode/speculative verification, graph capture and TP=1/non-Qwen paths remain on their previous APIs. This change does not graph batched prefill.

`Qwen35TpRank::forward_span_advance` embeds the complete window, walks every layer once, appends each GQA KV row through the paged table, advances each DeltaNet row's recurrent/conv state, and uses batched FFN. It does not execute final norm, LM head, or logits readback. Only the last window of a *finishing prompt chunk* calls `forward_span_head_enqueue` (final norm and last-row LM head); rank 0 samples its resident logits or calls the blocking `forward_span_head` for host logits. A non-finishing scheduler tick advances its rows without a head. The overlapping lane uses `prefill_lane_span_advance` and `prefill_lane_span_finish` with an event-based result. The original `forward_prefill_span` probe wrapper retains advance + blocking head for the isolated oracle.

## Control and state

The coordinator validates each slot's cumulative position across the entire command before `authorize_all`, authorizes one ordered list of `Ensure` operations, and sends ONE end-of-tick mirrored KV snapshot (not one snapshot per row). Rank 1 validates/mirrors that same list before `Prepared`. Both ranks enqueue decode traversals first and then the same prompt-window sizes in wire order, preserving collective order; only rank 0 runs a final head/sampler (neither has a collective). The worker validates the trailing mixed run's one-slot/disjoint geometry before `Prepared`. The protocol version is 3, so an older worker cannot interpret `chunk_rows` as decode rows. The existing pipe Ready/drain sequence is unchanged.

The prefill lane owns its own GQA KV slabs and DeltaNet state until `span_finish` joins both streams and promotes finished slots to the decode lane on *both* ranks. Rank 0 marks any slot with enqueued span rows occupied so cancellation before the first decode can release and reset its KV and recurrent state. Each launch parks a fresh lane completion event, including non-finishing launches; an earlier finisher event cannot declare the new span complete. `SpanFinisher.fin_rows` is the final prompt position plus one (full prompt row count), which is what the scheduler's `finish_prefill` uses for its decode cursor. The lane has one logits plane, so launches with more than one finisher or a non-tail finisher are rejected on both ranks before enqueue; the natural `chunk_take` currently emits at most one.

## Host evidence and limitations

- `cargo test -p paddock-engine --lib`: 482 passed, 0 failed (including TP chunk geometry, occupancy/release bookkeeping, mixed positions/slots and KV mirror/page-crossing tests).
- `cargo test -p paddock-dist`: 4 unit + 22 bootstrap tests passed; v3 `TpMixed.chunk_rows` serde and missing-field failure tested.
- `cargo test -p paddock-engine --all-targets`: passed on the final source tree (library 482, relevant example tests and integration tests; some existing ignored tests).
- `cargo clippy -p paddock-dist -p paddock-engine --all-targets`: passed with existing warnings in `cuda.rs`, `qwen35_tp_spec.rs`, and `qwen35_two_slot_oracle.rs`. Strict `-D warnings` fails on those unrelated warnings.
- `cargo build --release -p paddock-manager --bin paddock`, `cargo build --release -p paddock-runner --bin paddock-runner`, and `cargo build --release -p paddock-engine --example qwen35_tp_span_probe`: succeeded. The runner and example were rebuilt on the final source tree; the manager binary was built earlier and must not be used for exact-head acceptance.
- `git diff --check`: passed.
- Workspace-wide `cargo fmt --all -- --check` reports existing formatting changes in unrelated files; no blanket formatting was applied.

No GPU measurement or numerical parity from this production path has been established. The earlier isolated ≤64-row prototype reported ~9.1x at 64 rows and ~11.6x / ~180 tok/s for 64x16 repeated spans versus ~16 tok/s serial; those numbers are **not** production results. Strict serial-versus-batched numerical-oracle violations remain known and unchanged. Device overlap, cancellation during a live span, page crossing, slot reuse, transition to decode, and long-context frame/throughput behavior still require live validation. A successful host suite cannot establish these gates.

## Hosted GPU validation — exact next commands

Run only when both GPUs are available, from the head repo. First ensure a clean exact-head build and inspect the ignored, host-specific `docs/tp/qwen38-tp2-two-node.env` (copy from `.env.example` if it does not exist). The launcher stages the exact runner on rank 1 over SSH, checks paths, and uses `--tp-worker`. Do not run the isolated ABC/span probe as a substitute for production integration.

```sh
cd /home/sime/repos/paddock
git status --short --branch
git rev-parse HEAD
cargo test -p paddock-engine --lib
cargo test -p paddock-dist
cargo build --release -p paddock-runner --bin paddock-runner
SPEC=off TP_GRAPH=0 TP_DRY_RUN=1 ./docs/tp/qwen38-tp2-two-node.sh
SPEC=off TP_GRAPH=0 ./docs/tp/qwen38-tp2-two-node.sh
```

In a second shell while the coordinator serves, run these deterministic `/v1/completions` requests against the configured `HTTP_PORT` (default 11540); the responses give the *actual* prompt-token counts, so adjust repetition until the required token boundaries are crossed. The reported `model` ID is fetched from the server, not guessed. Capture output and check HTTP status instead of trusting only the response text:

```sh
cd /home/sime/repos/paddock
export PORT=11540
export MODEL_ID="$(curl -fsS "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
for words in 1 63 64 65 1024 20000; do
  WORDS="$words" python3 -c 'import json,os; print(json.dumps({"model":os.environ["MODEL_ID"],"prompt":"alpha "*int(os.environ["WORDS"]),"max_tokens":8,"temperature":0}))' \
    | curl -fsS -w ' HTTP %{http_code} total %{time_total}s\n' -H 'Content-Type: application/json' \
        --data-binary @- "http://127.0.0.1:$PORT/v1/completions"
done
```

These are word repetition counts, NOT claims about exact token counts. Compare identical-input greedy IDs/logits to the same-checkpoint TP=1 eager oracle. Include tokenized counts 1, 63, 64, 65, repeated 64-row chunks, a ~20k-token prompt, nonzero slot with concurrent decode, prefix/page crossing, cancellation followed by slot reuse, and the first decode token after each finisher. Record rank-0 and rank-1 exit codes, binary/model/pack hashes, protocol/ACK trace, KV snapshot agreement, TTFT/prefill throughput, and whether the scheduler used mixed or overlap. Do not treat a successful isolated probe or HTTP text match as production logit/ID parity. Restore normal speculation/graph settings only after eager validation and run their established decode regressions without changing those paths.
