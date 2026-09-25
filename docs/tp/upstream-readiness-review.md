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
(fp8_e4m3) KV for the validated non-spec path; n-gram speculative decoding on
F16 KV; cancellation, reset, release and slot reuse; normal OpenAI/Anthropic
-compatible serving on rank 0. Everything else (MoE, other TP sizes, other
model families or checkpoints, NCCL graph capture, spec with FP8 KV, separate
draft models) is refused by name at startup, not silently unsupported.

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
- FP8 KV + speculation is refused pending validation; FP8 KV without
  speculation is the validated lossy lane.
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

Updated after the bounded pre-PR cleanup pass. See section 17 for evidence.

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
  documented configuration. Host gates updated; target-device validation of
  the combined path is explicitly outstanding (see section 17).
- I2 — removed: `TpAcceptanceProbe` and `PADDOCK_TP_STATE_PROBE` deleted from
  product code; the production drain-before-release invariant (already
  enforced independently of the probe via `prefill_lane_done()` checks) is
  retained.
- I4 — fixed: `TpInit` now carries the resolved graph mode; rank 0 is
  authoritative; the worker no longer reads `PADDOCK_TP_GRAPH` locally.
- I6 — fixed: `spawn_worker_local()` (zero production callers) deleted;
  `work()` made `#[cfg(test)]` with its two bootstrap-loop tests moved from
  `tests/bootstrap.rs` into `worker.rs`'s unit-test module.
- I3 — intentionally deferred to the PR-preparation pass (history regrouping
  is forbidden in this session).
- I5 — partially fixed: `THIRD-PARTY-NOTICES` gains the ErikBPF entry; the
  FFN tp-06 commit-message repair is documented in section 18 for the later
  history pass (rewriting history is forbidden here).
- N1-N8 — intentionally deferred to the PR-preparation pass except where a fix
  fell out of the above (the probe removal removes its phase-number string
  sites; graph-mode doc note added).

## 17. Validation evidence

Host (this session, GLM vLLM stack occupies both Sparks, so no CUDA binary was
built and no GPU test ran):

- `cargo check --workspace --all-targets` — exit 0.
- `cargo test -p paddock-dist` — 18 passed (16 bootstrap incl. the
  strengthened previous-version rejection; 2 moved `work()` loop tests).
- `cargo test -p paddock-runner --lib` — 572 passed, incl. the new
  `tp2_gate` host tests (6): baseline accepted, spec+F16, spec+fp8_e4m3
  accepted (I1), fp8 non-spec accepted, offload refused, missing-model
  refused.
- `cargo test -p paddock-engine --lib` — 463 passed, incl. the new
  `authorize_all`/`mirror_tick` round-trip and fail-closed tests (divergent
  end state, empty tick, bad op leaving no partial advance).
- `cargo test -p paddock-engine --test tp_wire_frame` — 4 passed: old
  per-row-snapshot design measured crossing MAX_FRAME at 256 prompt rows
  (16k ctx, 2 slots; 512 rows = 2.18 MB), new end-of-tick design bounded
  (decode tick 4.2 KB, 512-row span 10.2 KB, 4096-row span 60 KB = 17x
  headroom), and a real-TCP round-trip mirror test incl. tampered-state
  rejection.
- `cargo clippy -p paddock-dist -p paddock-runner -p paddock-engine
  --all-targets` — 0 errors; B3's `unwrap_used` findings gone, plus the 7
  same-diff example `unwrap()`s clippy surfaced once the lib was fixed
  (also converted to `expect`). Remaining lib warnings verified identical
  to the pre-cleanup baseline (same 6, none introduced).
- `cargo fmt --all -- --check` — pre-existing drift across 25+ files at
  HEAD (incl. files this pass never touched, e.g. `kquant.rs`,
  `service.rs`, `delta_tp.rs`); left untouched per session scope. The four
  files this pass made fmt-dirty (`tp_kv.rs`, `tp_wire_frame.rs`,
  `bootstrap.rs`, `startup.rs`) were rustfmt-ed individually and are
  clean; the full-repo reformat belongs to the PR-preparation pass.
- `git diff --check` — clean.

Target-device validation — EXPLICITLY OUTSTANDING, not run:

The two Sparks are occupied by the GLM vLLM serving stack that hosts this
session itself (`VLLM::Worker_TP0`, `--nnodes 2 --node-rank 0` on
192.168.100.10 with the worker on .11), so stopping serving or running CUDA
tests would kill the session's own inference and requires user approval. The
following must run before any upstream submission:

1. Real two-node start via `--tp-worker` (no internal marker), handshake,
   serving, clean shutdown.
2. Long-context frame regression for B2: several-hundred-token prompt at
   multi-k context; the old design's largest frame would have exceeded 1 MiB;
   record the new design's actual largest encoded control frame.
3. Normal TP=2 non-spec serving smoke: unified overlap, graphs in the accepted
   mode, concurrent requests, clean rank exit.
4. Speculative F16 serving smoke (Phase 15 still works after the protocol
   change).
5. TP FP8 non-spec smoke.
6. The new combined spec + FP8 configuration validated per the cleanup-pass
   instruction (direct FP8 spec oracle, page/lifecycle coverage, optimized
   execution, normal serving, FP8-non-spec and F16-spec regressions).

## 18. Deferred-to-PR-preparation notes

- History regroup per section 10 (rebase, drop merges, staged logical
  commits).
- Commit-message attribution repair: `2546d17` needs its tp-06 credit added
  when the history is regrouped (content fix, not a rewrite of meaning).
- Example pruning (N8) and docs/tp condensation.
- Phase-number error strings (N1), worker GPU ordinal (N2), `dev_var!` for
  `PADDOCK_TP_GRAPH` (N5), `resolved_tp2` naming (N3).
