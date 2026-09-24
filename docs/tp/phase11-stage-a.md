# Phase 11 Stage A — rank-local graph capture for TP=2 decode rows

Status: ACCEPTED for the validated two-Spark workload at commit
`394ef1403f92aee6a5fe2fcbf4f6d8888ff31a34` (merge commit 394ef14).
Stage A remains opt-in with `PADDOCK_TP_GRAPH=1`. Stage B NCCL graph capture was
not attempted.

## Scope and design

Each TP decode row splits at the NCCL all-reduces. Stage A captures only the
collective-free rank-local runs and leaves NCCL outside CUDA graph capture:

```text
replay PreAttn graph  (rmsnorm + attention/DeltaNet run -> partial)
eager: compute->NCCL fence / all_reduce / NCCL->compute fence
eager: residual add
replay PreFfn graph   (rmsnorm + gate/up/swiglu/down -> partial)
eager: compute->NCCL fence / all_reduce / NCCL->compute fence
eager: residual add
```

Embedding, final norm, and lm-head GEMV stay eager. Prefill stays eager: the
prefill lane passes `graphed=false`, preserving Phase 10 overlap semantics.
GQA keys are per layer (`PreAttn(layer)`), FFN keys are per layer
(`PreFfn(layer)`), and DeltaNet keys are per layer and slot
(`PreAttnDelta(layer, slot)`) because the captured graph bakes the slot's
recurrent/conv state addresses. The graph path re-stages position, slot, and
paged block-table contents outside capture.

The implementation is in `tp_graph.rs`, `ffn_tp.rs`, `gqa_tp.rs`,
`delta_tp.rs`, `tp_model.rs`, and `tp_serve.rs`. `capture_mixer_run` and
`capture_ffn_run` quiesce the compute stream, capture only the corresponding
collective-free `run`, and instantiate with no flags. The `finish` functions
perform the fences and all-reduce after replay. This source tree has no
persistent validation trace instrumentation; the temporary trace used for the
runs below was removed before the final checks.

## Exact target configuration and artifacts

Two Spark processes ran on the 100 Gb/s fabric:

- rank 0/head: `192.168.100.10`
- rank 1/worker: `192.168.100.11`
- NCCL: `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_IB_HCA=rocep1s0f0`,
  `NCCL_IB_DISABLE=0`, `NCCL_NET=IB`
- TP: size 2, ranks 0/1, CUDA, KV dtype F16
- model: `/home/sime/models/Qwen3.8-27B-UD-Q4_K_M.gguf`
- `max_ctx=256`, `max_batch=2`, `--no-spec`
- pack: `packs/cuda/build/pd-cuda-sm120.so`
- checkpoint SHA-256:
  `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`
- CUDA pack SHA-256:
  `059bb62d2e6d863b32ac47208d29da7492618b2fa33015a3ee0ad6fe2a9c4d54`
- runner SHA-256:
  `e3ad9c754fa8839d3f98d5b0d426929435b4bf75d882a44efdcc7db5c0ab2396`
- the worker read back identical checkpoint, pack, and runner hashes before
  launch.

The ordinary pipe runs used graph variables unset, with `PADDOCK_NO_SPEC=1`.
The unified overlap runs used `PADDOCK_UNIFIED=1`,
`PADDOCK_NO_MIXED_SPEC=1`, and `PADDOCK_NO_SPEC=1`. Stage A runs added
`PADDOCK_TP_GRAPH=1`; diagnostic runs also set
`PADDOCK_TP_GRAPH_TRACE=1`. The graph trace is not part of the final source.

## Phase 10 merged-tree regression

The minimum regression was run again after the merge, without graph mode:

- `phase11-final-baseline-pipe`: ordinary TP=2 decode-pipe path, both ranks
  exited `0`.
- `phase11-final-baseline-overlap`: unified prefill/decode-overlap path, both
  ranks exited `0`.
- Each harness exercised concurrent requests, a client cancellation followed
  by survivor continuation, categorical sampling and slot reuse, an arrival
  during categorical decode, a 55-token continuation crossing the 16-token
  page boundary, and six randomized cancellation/arrival cases.
- The result arrays contain six regression records. The graph-enabled result
  arrays are byte-for-byte equal to their eager counterparts for both pipe and
  overlap:
  `phase10-phase11-final-baseline-pipe-results.json` equals
  `phase10-phase11-final-stagea-pipe-results.json`, and the corresponding
  overlap files are also equal.
- The live outputs included `phase11-final-baseline-pipe exits 0 0` and
  `phase11-final-baseline-overlap exits 0 0`. Logs for both ranks show the
  normal shutdown/drain path.

The standalone historical `phase10_verify_cancel.py` checker was not used as a
pass/fail gate: it expects old `[TP trace committed]` diagnostics that are not
emitted by the current merged runner. The live regression itself performed the
cancellation/reuse/page-crossing actions and compared the complete result
arrays.

## Stage A capture/replay evidence

The final graph pipe and unified-overlap logs were captured separately for both
ranks:

- `phase10-phase11-final-stagea-pipe-{head,worker}.log`
- `phase10-phase11-final-stagea-overlap-{head,worker}.log`

Per rank, both graph runs created 176 distinct cache entries, with cache count
reaching 176 and no eager fallback:

- 16 GQA `PreAttn` keys (layers 3, 7, ..., 63)
- 48 DeltaNet `PreAttnDelta(layer, slot=0)` keys
- 48 DeltaNet `PreAttnDelta(layer, slot=1)` keys
- 64 `PreFfn` keys

Thus the expected 16 + (48 x 2) + 64 = 176 key inventory was observed on
rank 0 and rank 1. Every key was created once in the diagnostic trace, then
reused. The pipe run recorded 80 selected replay traces per rank and the
unified-overlap run recorded 60 per rank. Selected replay positions included
0, 1, 15, 16, and 20; positions 15 and 16 straddle the 16-token KV page
boundary.

For each selected replay on both ranks, the observed event order was:

```text
pre-attn compute->NCCL
pre-attn NCCL->compute
pre-ffn replay compute->NCCL
pre-ffn NCCL->compute
```

The traces therefore show compute/NCCL boundaries on both ranks. The source
path confirms the collectives are not recorded: capture records `attention_run`
/`decode_run` and `finish_partial`, while `finish` performs the event fences and
NCCL after graph launch. TCP worker ACKs were not treated as GPU synchronization;
the GPU event fences and final stream/device drains remained authoritative.

## Graph/eager correctness

The two-slot oracle was run in eager and graph mode with `PADDOCK_NO_SPEC=1`,
`MAX_CTX=48`, two slots, rank-0-authorized operations, mirrored KV, and the
same 21-row slot-0 interleaving. Its script includes cancellation/release of
slot 1, slot reuse, page crossing, and exact reset replay.

- Eager and graph logs each contain 70 rank-parity rows: 35 in pass 0 and 35
  in exact replay pass 1.
- All 70 rows have `exact_rank=true`; all 35 replay rows have
  `exact_replay=true`.
- The graph and eager pass-row tuples (slot, position, greedy token, replay
  status, and checksum) are identical for all 70 rows.
- Interleaved slot 0 versus isolated slot 0 was exact across 21 rows in both
  modes.
- The independent TP=1 oracle checked 35 rows. The largest observed
  `max_abs` logit delta was `0.00803137`, below the oracle tolerance
  `ABS=1e-2` and `REL=1e-3`; no tolerance violation or greedy-token mismatch
  occurred.
- The graph trace covered slot 0 and slot 1 DeltaNet keys, and the exact
  reset/replay plus interleaving comparison exercised slot-specific recurrent
  and convolution state handling. The oracle uses mirrored paged KV and the
  15/16 boundary; it is a practical state/KV check rather than a separate
  byte dump of every recurrent state word.
- Both ranks exited cleanly. The graph oracle's rank-0 and rank-1 logs contain
  matching rank-parity checks and final synchronized completion.

## Decode pipe and unified overlap with graphs

`phase11-final-stagea-pipe` passed the accepted decode-pipe harness with graph
mode enabled. `phase11-final-stagea-overlap` passed the same six-record harness
with unified prefill/decode overlap enabled. Their complete eager/graph result
arrays are equal, not merely equal final text. The graph-overlap logs show
prefill spans launched and finished, decode-pipe begin/drain events, and normal
engine shutdown on both ranks. The prefill lane remained eager while the decode
lane replayed rank-local graphs; no stream/event/state race appeared in this
run.

## Benchmark

The benchmark used the same two-Spark runner configuration above, two sequential
streaming requests per process, prompt `Hi`, `max_tokens=96`,
`temperature=0`, `top_k=0`, `top_p=1.0`, `min_p=0.0`, `seed=42`, and no
speculation. Eager and graph were each run twice. Every request observed 96
tokens and the generated text was identical between modes.

Measured client-side values (milliseconds):

```text
                         eager mean       graph mean       graph-eager
startup/readiness*       41047.63          40547.06         -500.57
request 1 first token       74.41             74.53            +0.12
request 2 first token      171.15            183.44           +12.29
steady median gap           66.44             65.94            -0.51
request total              6464.71          6409.99           -54.72
```

`startup/readiness` includes model load and graph creation and is noisy across
these two process starts; it is not treated as an isolated capture-time
measurement. Request 2 is the first post-warm replay measurement: graph mode
paid about 12.3 ms (7.2%) in first-replay latency in this sample. Steady-state
streaming improved by about 0.51 ms/token gap (0.76%), roughly 15.05 tokens/s
eager versus 15.17 tokens/s graph. Capture amortization was established by the
176-entry one-time cache creation per rank and subsequent replay-only rows; the
steady-state sample was 96 decode tokens per request.

The raw benchmark records are:

- `phase11-bench-eager.json`, `phase11-bench-eager2.json`
- `phase11-bench-graph.json`, `phase11-bench-graph2.json`
- `phase11-bench-graph-{head,worker}.log`

All four benchmark process pairs exited `0 0`.

## Host verification and decision

After removing the temporary trace from `tp_model.rs`, the final tree was
checked with:

- `cargo check -p paddock-engine --offline --quiet`
- `cargo test -p paddock-engine --lib --offline --quiet`
- `cargo test -p paddock-runner --lib --offline --quiet`
- `cargo test -p paddock-dist --offline --quiet`
- `cargo test -p paddock-kernels --offline --quiet`
- `cargo check --all-targets --offline --quiet`
- `cargo build --release -p paddock-runner --offline --quiet`
- `git diff --check`

All completed successfully. The final accepted source has no diagnostic trace
changes beyond this evidence document.

Stage A is accepted for this TP=2 paged-KV decode workload. Remaining caveats:
NCCL itself is deliberately outside graphs, the prefill lane remains eager, and
this evidence is for the pinned Qwen3.8-27B F16-KV two-Spark configuration. Stage
B is worth investigating only after a separate design/prototype proves
collective capture and communicator/event semantics; this report does not claim
any Stage B readiness.
