# TP=2 upstream-readiness review

Reviewer: Hermes (source-review session, no builds/tests run).
Scope: local TP=2 implementation at `6f1259a653305aa7a4ff1c825153acdab4ad4dd3`
diffed against upstream merge-base `d4b4548` (truespar/paddock). Upstream sync
merges `56deb80` and `394ef14` were treated as synchronization points; upstream
owned changes were not attributed to the TP work.

Feature diff: 41 files, +11,908/-20.

## 1. Verdict

Not ready to submit today, but close. The architecture is genuinely
upstream-shaped: a dependency-free control-plane crate, a small NCCL
communicator, rank-local loaders with fail-closed validation, a
rank-0-authoritative serve loop with strict sequence/ACK discipline, and a
TP=1 path that is structurally untouched. Three blockers are concrete and
fixable: the two-node worker start has no supported operator path (B1), the
KV-mirror wire design overflows the protocol frame cap at real context sizes
(B2), and new library code carries `unwrap()`s the workspace denies (B3).
After those plus the cleanup below, this is a credible staged PR series.

## 2. BLOCKER findings

### B1 — No supported way to start the rank-1 worker on a second node

- Where: `crates/paddock-runner/src/startup.rs:894-951` (worker branch
  requires `ParallelConfig::is_worker_child()`),
  `crates/paddock-dist/src/config.rs:141-165` + `151-153`
  (`WorkerMustNotServe`), `crates/paddock-dist/src/config.rs:200-205`
  ("the marker is set by `worker_env`, never by users").
- Issue: a hand-started worker (`--tp-rank 1` or `PADDOCK_TP_RANK=1`) hits
  `resolved(true)` and is refused with exit 2. The only working manual path is
  hand-setting the internal marker `PADDOCK_TP_WORKER_CHILD=1` plus
  `PADDOCK_TP_MODEL`/`PADDOCK_TP_PACK` — exactly what the Phase 9 two-Spark
  validation did ("its manually started worker-child path",
  docs/tp/phase9-report.md:13). The documented runbook
  (docs/tp/phase2-report.md:219-221) omits the marker and the path vars, so
  following it verbatim exits 2.
- Why it matters upstream: two-node TP is the feature's headline; the
  documented flow fails and the working flow relies on a variable the code
  itself says users must never set.
- Remediation: an explicit worker-start form (e.g. `--tp-worker`), or promote
  the marker to a documented surface. Update the runbook.

### B2 — KV-mirror events scale O(rows x context) on the wire and overflow MAX_FRAME

- Where: `crates/paddock-engine/src/gpu_model/qwen35/tp_kv.rs:117-125`
  (`Event.state` = full `Snapshot`: all slot tables + refcounts), attached per
  row in `span_begin` (tp_serve.rs:1458-1466) and `forward_mixed`; frame cap
  `MAX_FRAME = 1 MiB` (paddock-dist/src/protocol.rs:213-218).
- Issue: a span/mixed tick carries one Ensure event per prompt row, each with
  a complete KV snapshot (~`ceil(max_ctx*slots/16)` block ids x 2 tables +
  refcounts, JSON). Frame size grows roughly rows x max_ctx x ~8 bytes;
  anything beyond roughly a few hundred prompt tokens at multi-k context
  exceeds 1 MiB -> `FrameTooLarge` -> pair poisoned -> every request fails.
  Secondary: decode ticks serialize/parse/compare the same snapshot per
  token, so per-token host cost grows linearly with context. All validation
  ran at max_ctx <= 256, which is why this never fired.
- Remediation: carry one end-of-tick `Snapshot` per message (positions already
  authorize row order), or chunk span launches so each frame stays bounded.
  Protocol change; bump `PROTOCOL_VERSION`.

### B3 — `unwrap()` in new library code vs the workspace deny

- Where: `crates/paddock-engine/src/gpu_model/qwen35/tp_serve.rs:1396`,
  `:1401` (`chunk_take`), `:2612` (`prefill_lane_finisher(rows.last().unwrap())`).
- Why it matters: `Cargo.toml` denies `clippy::unwrap_used` for library code;
  CONTRIBUTING requires `cargo clippy --workspace --all-targets` clean; the PR
  template has that checkbox. (Phase 12's report claims the 3 findings are
  byte-identical to the pre-TP baseline — they are in a new file, so that
  claim does not hold.)
- Remediation: mechanical — `expect` with the invariant named.

## 3. IMPORTANT findings

- I1 — Spec + FP8 KV accepted but never validated. `startup.rs:1014-1017`
  admits `--spec <policy>` together with `kv_cache_dtype fp8_e4m3`;
  phase15-report states FP8+spec is a deferred follow-up and was never run.
- I2 — `TpAcceptanceProbe` is validation machinery living in product code
  (`tp_serve.rs:255-432`, ~180 lines, plus call sites in both ranks), gated on
  `PADDOCK_TP_STATE_PROBE`, an env var not in `ENV_SURFACE` and set by no
  in-repo test or example; its magic rows belong to probe scripts outside the
  repo.
- I3 — History shape vs CONTRIBUTING: two upstream merge commits inside the
  series; CONTRIBUTING says rebase on main. 11.9k lines against "one change
  per pull request".
- I4 — Graph-mode agreement is not handshaken. `PADDOCK_TP_GRAPH` is read
  independently per process (`tp_model.rs:786-788`); a hand-started worker can
  diverge -> graphed/eager collective mispairing.
- I5 — Attribution gaps: commit `2546d17` (FFN primitive) has an empty message
  although `ffn_tp.rs:5-7` carries the tp-06 credit; `THIRD-PARTY-NOTICES` has
  no ErikBPF entry although materially adapted code ships.
- I6 — Superseded Phase 2 public API retained: `worker::work()`
  (worker.rs:192-214) and `spawn_worker_local()` (worker.rs:46-48) have no
  production callers; only dist tests call them.

## 4. NICE-TO-HAVE findings

- N1 — Phase numbers in user-facing strings ("Phase 9 requires...", tp_serve.rs
  120/513/2259, serving.rs 1542/2084).
- N2 — Worker GPU ordinal hardcoded to 0 (startup.rs:933); coordinator `--gpu`
  never reaches rank 1.
- N3 — `resolved_tp2` re-checks `rank == 1 && serving_mode` (config.rs:173-175)
  duplicating the caller's check; parameter named `_serving_mode` while used.
- N4 — `spec_batch_plans` allocates a dense `Vec<RowSample>` per verify row
  (tp_serve.rs:743-747).
- N5 — `PADDOCK_TP_GRAPH` read via raw `std::env::var` (tp_model.rs:787)
  instead of `dev_var!`.
- N6 — docs/tp/phase14-15-stop-point.md says Phase 15 is blocked;
  phase15-report.md says complete. Superseded doc.
- N7 — `engine_finisher_plan` note in phase10-progress.md references a symbol
  that does not exist.
- N8 — Example set: 11 new binaries (~2.8k lines); keep the oracle, spec probe
  and nccl_bench; fold or drop the phase-micro-probes.

## 5. Attribution / licensing assessment

Registered and present:

- `tensor_slice.rs:8-10` — tp-05 (895f8db) + tp-12 (d856695), commit-message
  attribution in `29e3342`. Correct.
- `gqa_tp.rs:3` — tp-09 (afedfdd), commit `5b95cbf` carries the plan-6 pattern
  verbatim. Correct.
- `delta_tp.rs:3-4` — tp-10/11 (8495d2d, 02ba232), commit `44708a4` attributes.
  Correct.
- `ffn_tp.rs:5-7` — tp-06 (6142c62) in the file header, but commit `2546d17`
  has no body.
- No CUDA-pack changes exist in the diff, so tp-01/tp-02/tp-14 "port" verdicts
  never became code — nothing to attribute there.

Gaps:

- `THIRD-PARTY-NOTICES` lacks any ErikBPF entry although the adaptations are
  substantial re-implementations derived from fork code (new transport, same
  sharding semantics). One line for
  `ErikBPF/paddock contrib/tp-05/06/09/10/11/12 (MIT OR Apache-2.0)` is the
  clean resolution.
- Zack Barra (@pagaldilz, layer-split `ded32a8`): audit verdict was "port
  selectively ... informs rank-local memory planning". No file or commit
  credits him and no recognizably derived code was found — this reads as
  inspiration, not adaptation. The PR description should say so explicitly.

## 6. Contribution-guideline compliance

Compliant: host-only test suites are substantial (18 dist tests incl. live
two-role handshakes, 566 runner/dist tests per phase reports, new tp_serve/
tp_model host tests); fail-closed gates everywhere (unknown `[parallel]` keys
hard-error via `deny_unknown_fields`; unsupported TP sizes/ranks refused, never
downgraded; `ENV_SURFACE` registration with a source-reading drift guard at
config.rs:860-878); no-silent-failures honored in the mirror/sequence design;
issue-first satisfied (truespar/paddock#22); commit subjects mostly follow
`area: ...`.

Violations / open items: B3 (clippy unwrap), I3 (merge-not-rebase; PR size vs
"one change per pull request"), I5 (THIRD-PARTY-NOTICES), and the PR template's
`Verified` section must carry the two-Spark measurements.

## 7. Configuration/API cleanup required before PR

- Decide the worker-start surface (B1) and document it in
  `paddock.example.toml` `[parallel]` + README; today `[parallel]` is in
  neither.
- Remove `TpAcceptanceProbe` + `PADDOCK_TP_STATE_PROBE` (I2).
- Keep `PADDOCK_TP_NO_SPAWN` (real two-node ops knob, sealed-env registered).
- Keep `PADDOCK_TP_GRAPH` as an explicitly experimental env, but move to
  `dev_var!` and one doc line (N5).
- `worker::work()` / `spawn_worker_local()` — test-only or delete (I6).
- Reword phase-number error strings (N1).
- `resolved_tp2` duplicate check / `_serving_mode` naming (N3).

## 8. Code / probe / docs cleanup required before PR

- Remove before PR: probe machinery (I2); the seven phase-micro-probe examples
  per N8 (keep `qwen35_two_slot_oracle`, `qwen35_tp_spec`, `nccl_bench`).
- Document only: `PADDOCK_TP_GRAPH`, `PADDOCK_TP_NO_SPAWN`, the pinned-SHA256
  checkpoint gate (tp_serve.rs:35,120).
- docs/tp/: do not ship 18 phase reports (~170KB with internal hostnames,
  binary hashes, scratch paths) upstream. Condense into one
  `docs/tensor-parallel.md` plus the fork-audit's attribution register;
  archive the phase reports out of the PR.
- Test-only protocol hooks: none found beyond the probe.

## 9. Commit-history reviewability

25 local commits in phase order; early ones (7049d80, 29e3342, 5b95cbf,
44708a4, e1d5d82) have reviewable bodies; later ones are one-liners with no
body (bfb0c78, c013c82, f8e649c, 4f0ef91, 6f1259a), which fails CONTRIBUTING's
"the body says why, and what you verified". Two upstream merge commits sit
inside the series. All fixable by the regroup below; no further archaeology
needed.

## 10. Recommended logical PR structure

Staged series (or one PR with these as commits if maintainers prefer after
#22 discussion), rebased on main with merges dropped:

1. `dist: rank/world config, worker bootstrap, control protocol` — paddock-dist
   + runner config/startup worker branch + ENV_SURFACE (host-tested).
2. `engine: NCCL process group and event-fenced collectives` — gpu/distributed.rs
   + nccl_bench.
3. `models: checked rank-local GGUF matrix shards` — tensor_slice.rs,
   `load_quantw_shard`, kquant parity test [tp-05/12 attribution].
4. `qwen35: rank-local TP primitives and mirrored paged KV` —
   gqa_tp/ffn_tp/delta_tp/tp_kv/tp_model/tp_graph [tp-06/09/10/11 attribution].
5. `qwen35: TP=2 serve coordinator, worker, and generator` — tp_serve.rs (probe
   removed), service.rs serial election + pipe context limit + spec hooks,
   generator.rs additions.
6. `runner: two-node TP=2 serving lane` — serving/startup/lib/config gates,
   worker-start surface (B1), docs.

## 11. Suggested PR title

`qwen3.8: two-node tensor-parallel serving (TP=2 over NCCL)`

## 12. Architecture summary (for the PR description)

Two OS processes per pair: rank 0 (coordinator) runs the normal
runner/API/scheduler and owns all scheduling, KV lifecycle, sampling and RNG;
rank 1 (worker) loads its rank-local shard and mirrors authorized commands
with no HTTP, tokenizer or sampler. A dependency-free `paddock-dist` crate
owns rank/world config (CLI > env > toml, strict tp_size 1|2, no silent
downgrade), the worker-child bootstrap, and a length-prefixed JSON control
protocol (1 MiB frame cap, versioned handshake, sequenced commands with
Prepared/Ready ACKs that are enqueue ACKs, never GPU fences). NCCL collectives
(all-reduce/all-gather/reduce-scatter/broadcast) enqueue on a dedicated stream
behind a small `Communicator` trait with `after_compute`/`before_compute`
event fences; the NCCL ID rides the bootstrap connection. Weights load
rank-locally via checked host-side block selection (`tensor_slice`) and the
existing repackers; GQA splits by complete KV-head group, FFN column/row,
DeltaNet by recurrent/conv state ownership. Rank 0 authorizes every logical KV
operation as a versioned event the worker mirrors and validates against its
own replay before any GPU work; CUDA graphs capture only the collective-free
runs, with NCCL replayed eagerly between fences. Speculation is
scheduler-owned n-gram with rank-0-authoritative greedy/sampled verification.
Any mismatch poisons the pair; nothing falls back to TP=1 silently.

## 13. Supported scope (for the PR description)

Validated on 2x DGX Spark over the RoCE fabric, pinned
`unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M (SHA-256 enforced at load) with an
explicit CUDA pack: TP=2 only (one GPU per rank process, rank-local weights);
GQA + DeltaNet hybrid backbone; paged KV with mirrored lifecycle; continuous
batching with max_batch 1-2; device-side sampling (greedy/temperature-only
categorical; rank 0 only); decode pipeline with fixed-context drain bound;
unified prefill/decode overlap via the prefill lane; CUDA graphs behind
opt-in `PADDOCK_TP_GRAPH=1` with NCCL outside capture; F16 KV, and FP8
(fp8_e4m3) KV for the validated non-spec and speculative paths; n-gram
speculative decoding on F16 and FP8 KV; cancellation, reset, release and slot
reuse; normal OpenAI/Anthropic-compatible serving on rank 0. Everything else
(MoE, other TP sizes, other model families or checkpoints, NCCL graph capture,
separate draft models) is refused by name at startup, not silently unsupported.

## 14. Known limitations (for the PR description)

- Control plane is plain TCP/JSON on a private fabric, no auth/TLS — same
  trust level as the existing runner/manager channel.
- Per-step control ACKs are host-synchronous by design; the pair is
  serial-eager at the scheduler boundary (overlap hides prefill, not the ACK).
- The pinned-checkpoint hash gate means the TP lane serves exactly the
  validated file; the loaders' broader quant coverage is exercised by the
  parity examples, not by serving.
- `--gpu` selects the coordinator's device; each rank process uses GPU ordinal
  0 on its own node.
- CUDA-graph capture is opt-in Stage A (decode runs only), with the first
  token per slot paying the capture pre-sync.
- FP8 KV + speculation is accepted for the target-validated TP=2 lane; its
  numerical outputs are not required to match F16 exactly, and correctness is
  established by within-FP8 oracle replay, rank agreement and state/position
  accounting.
- The manager/Studio has no TP start surface; `[parallel]` is config/CLI only.

## 15. Pre-PR checklist

1. Fix B1: explicit worker-start mode + update runbook + `[parallel]` example;
   add graph flag to `TpInit` (I4) in the same change.
2. Fix B2: end-of-tick KV snapshot per message (or bounded span frames); bump
   `PROTOCOL_VERSION`; re-run the two-Spark serving smoke at >= 4k context
   with a several-hundred-token prompt.
3. Fix B3: replace the three `unwrap()`s with named `expect()`s; run
   `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
   `cargo test --workspace`.
4. Resolve spec x fp8_e4m3 (I1).
5. Remove `TpAcceptanceProbe` + `PADDOCK_TP_STATE_PROBE` (I2); move its
   invariants into the oracle example/tests.
6. Delete or test-gate `worker::work()` / `spawn_worker_local()` (I6).
7. Prune examples per N8; condense docs/tp into one upstream doc + attribution
   register; delete the stale stop-point/progress contradictions (N6/N7).
8. Add ErikBPF line to THIRD-PARTY-NOTICES; put the tp-06 credit in the FFN
   commit message; one-line Zack Barra prior-art note in the PR description.
9. Reword phase-number error strings (N1); thread or document the worker GPU
   ordinal (N2); `dev_var!` for `PADDOCK_TP_GRAPH` (N5).
10. Rebase onto upstream/main (drop the two merge commits) and regroup per
    section 10; write PR bodies with the two-Spark measurements per the
    template's Verified section; open against truespar/paddock#22.

---

## 16. Remediation status

Updated after the bounded pre-PR cleanup pass (and again after the N1-N8
cleanup commit). See section 17 for evidence.

- B1 — fixed: explicit `--tp-worker` operator mode; the worker branch no
  longer requires the internal `PADDOCK_TP_WORKER_CHILD` marker; runbook and
  `paddock.example.toml` updated.
- B2 — independently verified (frame-size reproducer, see section 17), then
  fixed: wire carries ordered rows plus ONE end-of-tick mirror snapshot;
  `PROTOCOL_VERSION` bumped 1 -> 2; host tests cover cap-bounded frames, the
  would-have-overflowed case, mismatched mirror state failing closed, and
  protocol-version rejection.
- B3 — fixed: the three `unwrap()`s replaced with named `expect()`s; clippy
  `unwrap_used` verified clean for the new code.
- I1 — fixed in the opposite direction from the original suggestion, per the
  cleanup-pass instruction: the combined TP speculation + FP8 KV path was
  audited (no F16-specific assumptions found in the spec path — it runs the
  same resolved-dtype KV ops as the non-spec path) and is now an accepted,
  documented configuration. Host and target-device validation both pass; see
  section 17.
- I2 — removed: `TpAcceptanceProbe` and `PADDOCK_TP_STATE_PROBE` deleted from
  product code; the production drain-before-release invariant (already
  enforced independently of the probe via `prefill_lane_done()` checks) is
  retained.
- I4 — fixed at the protocol level (see section 20 for the follow-up that
  completed it): `TpInit` now carries the resolved graph mode; rank 0 is
  authoritative. NOTE: the first fix stopped capture-time divergence only;
  token EXECUTION still consulted a process-global env-backed flag until the
  section 20 correction.
- I6 — fixed: `spawn_worker_local()` (zero production callers) deleted;
  `work()` made `#[cfg(test)]` with its two bootstrap-loop tests moved from
  `tests/bootstrap.rs` into `worker.rs`'s unit-test module.
- I3 — intentionally deferred to the PR-preparation pass (history regrouping
  is forbidden in this session).
- I5 — partially fixed: `THIRD-PARTY-NOTICES` gains the ErikBPF entry; the
  FFN tp-06 commit-message repair is documented in section 18 for the later
  history pass (rewriting history is forbidden here).
- N1-N8 — resolved by the bounded cleanup commit on `qwen38-tp2-prepr`
  (details below and in section 19).

## 17. Validation evidence

The bounded target-validation pass was run after `05bf158` on the actual two-Spark
pair. No source code changes were required during target validation.

### Exact target environment and artifacts

- rank 0/head: `192.168.100.10`; rank 1/worker: `192.168.100.11`;
- GPU/driver: NVIDIA GB10, driver `580.173.02`, one GPU per process;
- CUDA toolkit: `13.0.88`;
- NCCL/RoCE: `NCCL_SOCKET_IFNAME=enp1s0f0np0`,
  `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_DISABLE=0`, `NCCL_NET=IB`;
- checkpoint: `Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256
  `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`;
- freshly rebuilt CUDA pack: `pd-cuda-sm120.so`, SHA-256
  `5908806748f0cfa62926ff1455e8d4d54fb7b7554360a0456d1eeb54280f77b0`;
- current release runner on both ranks, SHA-256
  `68f70d3c1590def934fbf96eee8145906bb7d7157691b7df3658d0cc6171685c`;
- current direct-oracle binaries were rebuilt from the same checkout and copied
  to `/home/sime/ffn-tp/paddock-tp-acceptance/`; their hashes matched the head
  copies before launch.

The worker used the accepted RoCE interface/HCA and the explicit `--tp-worker`
CLI path. No target launch used `PADDOCK_TP_WORKER_CHILD`.

### Host regression

The host gates were rerun against the same checkout before documenting target
acceptance: `cargo test -q -p paddock-dist` passed 18 tests total (2 + 16
across its test targets), `cargo test -q -p paddock-runner --lib` passed 572,
`cargo test -q -p paddock-engine --lib` passed 463, and
`cargo test -q -p paddock-engine --test tp_wire_frame` passed 4. Targeted
clippy with `-D clippy::unwrap_used` exited 0; only the existing non-error
example/style warnings remain. `git diff --check` is clean.

### B1 — explicit two-node worker startup: PASS

`/home/sime/.hermes/cache/scratch/accept-b1-{head,worker}.log` records the
first target startup using the supported operator path. Rank 0 listened,
rank 1 joined as `worker`, `tensor-parallel bootstrap complete` was logged,
rank 0 alone bound the HTTP API, and a normal completion returned HTTP 200
with text `", I"`. The worker-side socket check found no API listener.
The coordinator and worker both exited `0` after SIGINT; the coordinator log
ends with `shutdown: device memory freed - exiting`.

The bounded fail-early checks also passed: an explicit worker with a missing
model and one with a missing CUDA pack each exited `2` immediately with an
actionable `--tp-worker model not found` / `--tp-worker kernel pack not found`
message, without dialing a coordinator.

### B2 — long-context mirror-frame regression: PASS

`/home/sime/.hermes/cache/scratch/accept-b2-{head,worker}.log` records the
strong run with `max_ctx=8192`, F16 KV, and a 4,000-token prompt (17,999
characters; 400 repeated prompt sentences). The request returned HTTP 200,
one completion token, and `prompt_tokens=4000`; both ranks exited `0`.
The logs contain no `FrameTooLarge`, mirror mismatch, pair-poisoning, NCCL
mismatch, panic, or signal-139 symptom. The old design's crossing case remains
mechanically recorded by the host wire test (512 rows at 16k context: 2.18 MB);
the new design's host measurement was 60 KB for a 4,096-row span. The runtime
serving path does not expose encoded control-frame sizes, so no target frame
size is claimed beyond the passing request and the host wire measurements.

### F16 TP=2 non-spec regression: PASS

The live graph/unified HTTP regression (`accept-f16-live-*`) used max batch 2,
two concurrent requests, categorical arrival during decode, cancellation,
survivor continuation, release/reuse, and a 55-token page-boundary request;
both ranks exited `0`. The focused direct probes also passed:

- `accept-f16-pipe-head.log`: production greedy IDs through position 20/page
  crossing, drain, release/reuse and reset/replay;
- `accept-f16-overlap-head.log`: categorical decode/finisher IDs, full
  finisher/next-row logits, page crossing, drain and release/reuse.

### F16 speculative regression: PASS

`accept-f16-spec-head.log` records the direct TP oracle passing zero, partial and
full greedy acceptance, fixed-plan sampled replay, padded picks, committed
position advancement and both-rank replay; it exited `0 0` with graph mode
enabled. The live `accept-f16-spec-live-*` run then passed normal HTTP serving
with two concurrent requests, cancellation, survivor continuation,
release/reuse, page-boundary continuation, `PADDOCK_UNIFIED=1`, and
`PADDOCK_TP_GRAPH=1`; it exited `0 0` and logged prefill-span, slot-mapped-pipe
and decode-pipe activity.

### FP8 non-spec regression: PASS

The rank-0 log for the live run (`accept-fp8-live-head.log`) explicitly records
`kv cache: fp8-e4m3`; the same graph/unified HTTP workload as F16 completed,
including cancellation and reuse, with both ranks exiting `0`. The focused
`accept-fp8-{pipe,overlap,sampled}-head.log` probes passed page crossing,
rank-local device sampling, categorical replay, drain, release/reuse and reset
replay. FP8 output differences from F16 were treated as expected lossy-dtype
numerics, not as failures.

### FP8 + speculation: PASS — accepted configuration

The direct `accept-fp8-spec-head.log` oracle passed greedy zero/partial/full
acceptance, sampled deterministic `DevicePlan` replay, rank-0-only sampling,
rank-symmetric target execution, rejected-prefix position advancement and
within-FP8 deterministic replay; both ranks exited `0`. The companion
`accept-fp8-sampled-head.log` probe passed FP8 categorical replay, sparse-hole
handling, page crossing and release/reuse.

The live `accept-fp8-spec-live-*` run used TP=2, FP8 KV, `--spec on`,
`PADDOCK_UNIFIED=1`, and coordinator `PADDOCK_TP_GRAPH=1`; the remote worker's
local graph variable was explicitly unset/different. Two concurrent HTTP
requests, cancellation, survivor continuation, categorical reuse and the
55-token page-boundary request completed successfully. The rank-0 log records
FP8 resolution plus prefill-span, slot-mapped-pipe and decode-pipe activity;
both ranks exited `0`. No FP8 output was required to match F16 exactly.

### I4 graph-mode handshake: PASS (capture authority only — see section 20)

The coordinator runs set `PADDOCK_TP_GRAPH=1`; the explicit remote worker ran
with `PADDOCK_TP_GRAPH` unset in B1/B2 and unset or `0` in the live/direct
regressions. All graph-enabled pairs completed the same collectives and exited
cleanly. This validates the rank-0-resolved `TpInit.use_graphs` propagation in
practice for graph CAPTURE; rank 1 did not independently elect capture mode.
What this evidence did NOT cover: token execution still routed through a
process-global env-backed flag, so the runs could not have detected an
execution-mode divergence under the pre-section-20 code (a worker whose local
value disagreed would have captured per TpInit but executed per its own env).
The section 20 fix removes that residual path; revalidating the corrected
graph-authoritative invariant on the two-Spark pair remains open.

### Clean exit and performance observations

All required successful target runs ended with rank-0 and rank-1 exit `0`; no
exit 139, stale-pack warning, collective mismatch, mirror mismatch or poisoned
pair appeared. The rebuilt pack was used for every target run. The one
auxiliary two-slot example was not used as an acceptance gate: its first
attempt correctly refused without `PADDOCK_NO_SPEC=1`, and its legacy direct
control sequence did not complete after the precondition was supplied; the
required cancellation/reuse evidence is provided by the production HTTP
regressions and the passing pipe/overlap oracles.

The live logs do not emit drafted/accepted token counters, so no fabricated
acceptance rate is reported. Rough observed request latencies from the accepted
F16-spec live run were 55-token completion ~4.41 s, 28-token survivor ~2.81 s,
and 5-token concurrent request ~0.63 s; FP8-spec was ~4.37 s, ~2.78 s and
~0.62 s respectively. These are sanity observations, not benchmark claims.

### Outstanding target checklist

- B1 explicit `--tp-worker` startup: PASS — `accept-b1-*`, HTTP 200, no worker API,
  fail-early negatives, exits `0 0`.
- B2 long-context one-snapshot mirror protocol: PASS — `accept-b2-*`, 4,000
  prompt tokens at `max_ctx=8192`, HTTP 200, exits `0 0`, no frame/mirror errors.
- B3 host fail-closed/clippy gate: PASS — host evidence above, no target source
  changes needed.
- Non-spec F16 TP=2: PASS — live HTTP plus pipe/overlap oracles.
- F16 speculation: PASS — direct oracle plus graph/unified live HTTP.
- FP8 non-spec: PASS — direct pipe/overlap/sampling plus live HTTP.
- FP8 + speculation direct oracle: PASS — greedy and sampled fixed-plan oracle.
- FP8 + speculation lifecycle/optimized serving: PASS — cancellation,
  concurrent requests, reuse, page crossing, unified/graphs, clean HTTP exit.
- I4 graph handshake: PASS — coordinator-authoritative graph mode with worker
  environment unset/different.
- Clean two-rank shutdown: PASS — all required target runs exited `0 0`.

B1/B2/B3/I4 are fully accepted. FP8+spec is now an accepted TP=2
configuration. The branch is ready for PR-preparation/history-cleanup review;
that history work is intentionally not performed in this session.

## 18. Deferred-to-PR-preparation notes

- History regroup per section 10 (rebase, drop merges, staged logical
  commits).
- Commit-message attribution repair: `2546d17` needs its tp-06 credit added
  when the history is regrouped (content fix, not a rewrite of meaning).
- docs/tp condensation into one upstream doc (N6/N7 already resolved; the
  full condensation still belongs to this pass).

---

## 19. N1-N8 cleanup resolution

Bounded cleanup commit on `qwen38-tp2-prepr` (branched from `333512a`),
host-only; no GPU revalidation was required and none was run. Every change
below preserves the accepted runtime semantics: strings/naming/docs only,
one no-op control-flow simplification (N3), one allocation hoist that
provably produces the identical per-row plan vector (N4), and one config
read routed through the repo's dev-switch mechanism with an identical
resolved value in dev builds (N5).

- N1 — FIXED. All six product-facing phase-number strings replaced with
  capability-oriented wording (tp_serve.rs checkpoint-hash and init-shape
  errors plus the worker's TpInit-expectation error, service.rs scheduler
  selection log, serving.rs architecture and CUDA-pack errors). Comments,
  headers and historical phase reports retain their phase terminology.
- N2 — DOCUMENTED + one fail-closed guard. `--gpu` help now states it
  selects the rank-0 coordinator's device only; `run_worker` documents its
  always-0 worker ordinal; paddock.example.toml spells out the same. The
  explicit `--tp-worker` path now refuses `--gpu` with exit 2 and an
  actionable message instead of silently ignoring it. Device placement
  semantics unchanged (the worker still uses local GPU ordinal 0).
- N3 — FIXED, no behavioral change. The `WorkerMustNotServe` check moved
  from `resolved`'s explicit-rank arm into `resolved_tp2` (single check
  site; the `(Some(2), None)` default-rank arm passes rank 0 and is
  unaffected). The used-but-underscored `_serving_mode` parameter is now
  named `serving_mode`. The `rank1_may_not_serve_but_a_child_may_exist`
  dist test passes unchanged.
- N4 — FIXED, code changed. `spec_batch_plans` allocated a fresh dense
  `Vec<RowSample>` per verify row; it now allocates one scratch vector
  before the loop and refills it (`fill(Hole)` + one slot overwrite) per
  step. Since exactly one row is live per verify step, the plan vector
  passed to `run_rows_impl` is element-for-element identical; deterministic
  sampled semantics unchanged.
- N5 — FIXED. `PADDOCK_TP_GRAPH` is read via `paddock_models::dev_var!`
  (the repo's dev-switch mechanism, compile-time-dead in hardened builds)
  instead of raw `std::env::var`. Rank 0 still resolves graph mode and the
  worker still receives it via `TpInit.use_graphs`; no worker-local
  election. Dev-build resolved value identical (`"1"` ⇒ graphs on).
- N6 — RESOLVED as superseded-with-pointer. `docs/tp/phase14-15-stop-point.md`
  opens with a SUPERSEDED banner pointing at `phase15-report.md` and this
  review; the historical body is untouched. Final reports unaltered.
- N7 — FIXED. The `engine_finisher_plan` sentence in phase10-progress.md now
  states the symbol no longer exists and names the current one
  (`wire_finisher_plan`). No other Phase 10 content touched.
- N8 — PRUNED: seven examples removed
  (`qwen35_ffn_tp`, `qwen35_gqa_tp`, `qwen35_delta_tp`, `qwen35_model_tp`,
  `qwen35_tp_pipe`, `qwen35_tp_feedback`, `qwen35_tp_overlap`, ~2.2k lines).
  Classification: the five component/whole-model parity probes are REMOVE —
  their coverage was phase-gate evidence duplicated by the retained oracles
  and by the tp_serve/tp_model/gqa_tp/ffn_tp/delta_tp/tp_kv unit tests; the
  three pipe/feedback/overlap probes are REMOVE — each validated a stage
  that the later production gates (phase15-report, section 17 here)
  superseded. `qwen35_tp_sampled` is RETAINED (bonus, beyond the review's
  original trio): it is the only direct probe of the production-generator
  device-sampling path (device+host rows in one batch, hole handling,
  categorical replay) that no unit test or other example covers. Final
  retained set: `qwen35_two_slot_oracle`, `qwen35_tp_spec`,
  `qwen35_tp_sampled`, `nccl_bench`. The oracle was kept deliberately: it
  remains the only interleaved two-slot cancellation/reuse/reset-replay
  oracle. It compiles clean
  against the current architecture; the one auxiliary stall in section 17
  (the legacy direct control sequence not completing even with
  `PADDOCK_NO_SPEC=1` supplied) was not reproduced or diagnosed in this
  host-only pass - the required cancellation/reuse evidence remains the
  production HTTP regressions, and a fresh target run of the oracle is
  advisable in the later PR-preparation pass before relying on it again.
  `qwen35_tp_spec` compiles
  clean against the current architecture. No docs/scripts referenced the
  removed examples outside historical phase reports (untouched by design).

Validation for this cleanup (host-only, branch `qwen38-tp2-prepr`):
`cargo test -q -p paddock-dist` 18/18; `cargo test -q -p paddock-engine
--lib` 463; `cargo test -q -p paddock-engine --test tp_wire_frame` 4;
`cargo test -q -p paddock-runner --lib` 572; `cargo check --workspace
--all-targets` clean; `cargo clippy --workspace --all-targets` error-free
with only the pre-existing warning set (same six engine-lib sites as at
`333512a` plus the two retained examples' own pre-existing warnings;
clippy `-D clippy::unwrap_used` clean); `git diff --check` clean. Formatting:
repo-wide `cargo fmt --check` reports the same 56 hunks as at `333512a`
(pre-existing drift in untouched files); per-file rustfmt comparison before
vs after shows byte-identical drift content in every touched file, so no
formatting was applied to avoid mass-formatting unrelated code.

---

## 20. Follow-up fixes after the independent source review of `8b165a1`

Host-only bounded follow-up on `qwen38-tp2-prepr`. No rebases, no history
rewrites, no further pruning.

### I4 completed: graph-execution mode is runtime model state

An independent source review found that the first I4 fix was incomplete. It
made graph CAPTURE rank-0-authoritative (`TpInit.use_graphs` decides whether
each rank calls `enable_tp_graphs()`), but token EXECUTION still routed
through `Qwen35TpRank::tp_graph_enabled()` - a process-global `OnceLock`
backed by `PADDOCK_TP_GRAPH`. `forward_token_gpu`/`forward_token_enqueue`
(and the device-feedback path) consulted that global, so a worker process
whose local env disagreed with rank 0 would CAPTURE per `TpInit` but EXECUTE
per its own environment: the mispairing I4 set out to prevent remained
reachable at execution time.

Fix (structural, host-tested):

- `Qwen35TpRank` gains a `graphs_enabled: bool` runtime field, default
  false (eager) on every construction path including the prefill lane;
- `enable_tp_graphs()` performs capture and flips the field true only after
  successful setup; eager paths never touch it;
- every token forward (`forward_token_enqueue`, `forward_token_gpu`,
  `forward_device_feedback`) passes `self.graphs_enabled` to
  `forward_token_body`; the prefill lane passes literal `false` (never
  captured by design);
- the global/OnceLock resolver is deleted. Graph mode is resolved exactly
  once at serving setup via `Qwen35TpRank::resolve_graph_mode_for_serve()`
  (`dev_var!`, the repo-standard dev-switch mechanism), rank 0 composes it
  into `TpInit`, and both ranks convert the decision into model state via
  `enable_tp_graphs()`. No production code reads `PADDOCK_TP_GRAPH` after
  initialization; direct/probe callers configure graph mode explicitly by
  calling `enable_tp_graphs()`.

Host tests (tp_model.rs unit module): the resolver is a pure env read (no
global memoization); `graphs_enabled` is initialized false at exactly the
two construction sites and flipped true in exactly one place inside
`enable_tp_graphs`; every `forward_token_body` invocation consults the
stored field; exactly one executable `PADDOCK_TP_GRAPH` read exists in
production code (the setup-time resolver). The section 17 I4 evidence above
is reclassified as capture-authority-only; the corrected graph-authoritative
execution invariant has NOT been revalidated on the two-Spark pair and
requires target revalidation before any new GPU acceptance claim.

### B2 context bound: measured, no gate required

The v2 mirror wire sends one full `Snapshot` per mutating command, so frame
size grows with CONTEXT (the refcount array dominates: one u32 per physical
block, blocks = ctx/16 x 2 slots). Measured through the real `to_frame()`
encoder with the supported slot maximum (2):

- 16,384 ctx: 4,206 B (0.4% of the 1 MiB MAX_FRAME)
- 65,536 ctx: 16,494 B (1.6%)
- 131,072 ctx: 32,879 B (3.1%)
- 262,144 ctx: 65,647 B (6.3%)
- 1,048,576 ctx: 262,256 B (25.0%) - still under the cap

Growth is ~0.25 B per context token; extrapolation puts the frame-cap
crossing at ~4.19M tokens of context. No context Paddock currently claims or
supports for this TP lane approaches that, so NO fail-closed startup gate
was added (adding one would gate nothing reachable). `MAX_FRAME` is
unchanged and the wire format is untouched. The measured sizes are pinned by
`b2_snapshot_frame_growth_and_context_bound` in `tests/tp_wire_frame.rs`, so
a future wire change that breaks this bound fails in host CI first.

### Explicit `--tp-worker` dial-target env fallback fixed

The `--tp-worker` help documented a `PADDOCK_TP_MASTER_ADDR` /
`PADDOCK_TP_MASTER_PORT` fallback, but the branch constructed
`ParallelConfig` from the CLI fields only, so the documented env fallback
never applied. Fixed with a host-pure `paddock_dist::config::
resolve_worker_dial(cli_addr, cli_port, env_addr, env_port)`: CLI wins per
field, env fills only the gaps, an invalid env port fails early with the
actionable `BadInt` error naming `PADDOCK_TP_MASTER_PORT`, an empty env
string is treated as unset (the same rule `merge_env` applies), and a
missing address still refuses via the existing `EmptyMasterAddr` check. Both
env names are ENV_SURFACE-registered, so hardened seals keep them. Covered
by six new tests in `crates/paddock-dist/tests/bootstrap.rs` (CLI only, env
only, per-field CLI-over-env, invalid port, missing address, empty strings).

Validation for this follow-up (host-only): see the commit message. GPU
revalidation required: the two-Spark graph-mode pair (coordinator
`PADDOCK_TP_GRAPH=1` with a worker whose local value is unset/`0`) rerun
against the corrected execution-mode state before any further graph-path
acceptance claim.

---

## 21. Target revalidation of corrected I4 graph execution

Follow-up target revalidation was run from `c24966208249746d46084b42c078323a60e5fb73` on the same two-DGX-Spark topology. The corrected design is runtime state: `TpInit.use_graphs` is resolved by rank 0, both rank-local `Qwen35TpRank` instances call `enable_tp_graphs()`, and successful capture sets the stored `graphs_enabled=true`; token execution then uses that stored state rather than rereading the worker environment.

### Identity and transport

- rank 0/head: `gx10-d28a` / `192.168.100.10`; rank 1/worker: `gx10-d28b` / `192.168.100.11`;
- checkpoint SHA-256 (both nodes): `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`;
- freshly rebuilt CUDA pack `pd-cuda-sm120.so` SHA-256 (both nodes): `70b069faba0a1ba078af30bcbbc2ee0461cb0124726a7553708443bd919a00f3`;
- release runner SHA-256 (both nodes): `68f70d3c1590def934fbf96eee8145906bb7d7157691b7df3658d0cc6171685c`;
- `qwen35_two_slot_oracle` SHA-256 (both nodes): `c7a842b406bc9a19547b555d0512d19150e973c0d676221240f5076791fbd742`;
- CUDA toolkit `13.0`; NVIDIA driver `580.173.02`; NCCL `2.30.4+cuda13.2`;
- NCCL/RoCE pins on both ranks: `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_DISABLE=0`, `NCCL_NET=IB`.

### I4 corrected execution gate: PASS

A normal two-node TP=2 serve used the supported explicit `--tp-worker` path, `--max-batch 2`, F16 KV, and no speculation. Rank 0 explicitly set `PADDOCK_TP_GRAPH=1`; rank 1 explicitly set the deliberately conflicting local value `PADDOCK_TP_GRAPH=0` and did not force graph execution. The service completed three HTTP completions: two concurrent requests returned HTTP 200 with 12 generated tokens each, and a separate 361-token prompt returned HTTP 200 with 20 generated tokens, crossing the 16-token logical KV page boundary. The coordinator log recorded TP decode-pipe begin/drain events for the concurrent and boundary requests. Both ranks participated in bootstrap, model load, graph setup, multi-token decode and the same NCCL sequence; no collective mismatch, timeout/hang, graph capture/replay failure, mirror mismatch, panic, or exit 139 was observed. Rank 0 and rank 1 exited 0 after coordinated shutdown; the worker exposed no API.

This is behavioral evidence that rank 1 executed the coordinator's `TpInit`-selected graph mode despite its conflicting local environment. The serving logs do not expose a graph-count counter, so no graph-count claim is made beyond successful setup and multi-token execution.

### Two-slot oracle: PASS

The freshly matching `qwen35_two_slot_oracle` completed normally with `PADDOCK_NO_SPEC=1` and exited 0 on both ranks. Its scripted interleaved two-slot lifecycle covered admission and decode, cancellation, synchronized slot release, reuse/readmission, flush/reset replay, and the final rank-1 shutdown handshake; the output included all three scripted replay segments and both rank processes exited cleanly. The earlier auxiliary stall was not reproduced.

The corrected I4 target gate and the retained two-slot oracle now close the target revalidation required by section 20. No technical blocker remains before history regroup/rebase; no source architecture or product code was changed for this validation.

---

## 22. Post-review TP KV atomicity and Clippy follow-up

An independent source review identified that batched `authorize_all()` and `mirror_tick()` could mutate logical KV state before a later operation or final end-state comparison failed. Both methods now stage a cloned `MirroredKv` state and commit it only after the complete tick succeeds. Failed coordinator partial-operation ticks and failed worker end-state comparisons therefore preserve the exact snapshot and sequence; the latter is then replayed successfully with the correct end state. Existing successful lifecycle, page-boundary, release, flush, and prefix-reuse tests remain in place.

The five TP-specific Clippy findings were fixed narrowly: graph run-key variants were renamed to remove the shared `Pre` prefix; `TpMixer` variants now box the large rank-local mixer values; the redundant `GpuError` conversion was removed; the speculative loop derives position from its enumerated index; and finisher-plan assignment uses a single conditional pattern. Strict Clippy now leaves only the known upstream CUDA unnecessary-cast warning in `crates/paddock-engine/src/cuda.rs`. This is host-only cleanup evidence; no GPU revalidation claim is added here.

---

## 23. Final post-rebase TP=2 target acceptance

Validated from exact source commit `f98e15f61dff2b025367c885b4efeabfbb22181e` on the accepted two-DGX-Spark topology: rank 0/head `192.168.100.10`, rank 1/worker `192.168.100.11`. The pinned Qwen3.8-27B UD-Q4_K_M checkpoint matched on both nodes (`sha256 322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`). Fresh release artifacts built from that exact source were `paddock-runner` (`sha256 50fe4a1783e050a334381be0b7120ed8d89dee704ad1ecb2cb065c0275b17fd9`), `pd-cuda-sm120.so` (`sha256 c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e`), and `qwen35_two_slot_oracle` (`sha256 6742252e1b0677e67d94b317ed8ce93aef4e0b263fd2088bf9beddfdc3d2fa1f`). The same hashes were read back after transfer to rank 1. The worker host did not have Cargo/nvcc installed, so the fresh matching binaries/pack were built on rank 0 from the exact commit and hash-verified on rank 1; no stale artifact was used.

Target software and transport identity: CUDA toolkit `13.0.88` (CUDA 13.0), NVIDIA driver `580.173.02`, NCCL `2.30.4+cuda13.2`; both ranks used `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_DISABLE=0`, and `NCCL_NET=IB`. All serving runs used the supported explicit `--tp-worker` path. Rank 0 used `PADDOCK_TP_GRAPH=1`; rank 1 deliberately used `PADDOCK_TP_GRAPH=0`. `PADDOCK_UNIFIED=1` was used for the speculative production runs.

Gate results:

- Graph-authoritative F16 non-spec: PASS. TP=2, F16 KV, `--max-batch 2`, no speculation; normal HTTP completion, two concurrent HTTP 200 completions, multi-token decode, a 361-token request crossing the 16-token logical KV page boundary, and coordinated shutdown all completed. No collective mismatch, graph failure, mirror mismatch, panic, hang, or abnormal exit was observed. The coordinator log recorded decode-pipe begin/drain events and both ranks exited cleanly.
- F16 speculative decoding: PASS. Normal HTTP serving with n-gram speculation (`--spec on`, `PADDOCK_UNIFIED=1`) completed single and concurrent requests with both ranks synchronized and clean shutdown. The metrics endpoint reported `14` drafted and `14` accepted tokens for the repeated n-gram request. The initial non-repeating prompts correctly produced no draft candidates; this is not counted as a drafting failure.
- FP8 KV non-spec: PASS. `--kv-cache-dtype fp8_e4m3` completed two concurrent multi-token requests and a 361-token page-boundary request with HTTP 200 and clean shutdown.
- FP8 KV plus speculation: PASS. FP8 KV with n-gram speculation completed two concurrent requests; metrics reported `12` drafted and `10` accepted tokens. Both ranks remained synchronized and the run shut down cleanly.
- Lifecycle/concurrency: PASS. The production HTTP path covered simultaneous requests, completion/release, slot reuse/readmission, continued generation after reuse, and page-boundary continuation. The existing serving path was used; no new acceptance tooling was added.
- Two-slot oracle: PASS. `qwen35_two_slot_oracle` exited `0` on both ranks. Its interleaved two-slot lifecycle covered cancellation, synchronized release, reuse/readmission, flush/reset replay, exact rank replay, and final worker shutdown.

All observed target runs had clean coordinated shutdown; no technical blocker remains before history cleanup and PR preparation. The transactional `MirroredKv::authorize_all()`/`mirror_tick()` change introduced no observed successful-path regression in normal page allocation, page-boundary growth, release, reuse, prefix lifecycle, mirrored state, F16/FP8 serving, or the two-slot oracle. Host rollback coverage also passed: five focused TP KV tests, `5 passed, 0 failed`.
