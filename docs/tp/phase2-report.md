# Phase 2 — Report: Distributed configuration and process bootstrap

Implementation commit: `7049d80` ("dist: add rank/world configuration and
process bootstrap"). Branch `main`, on top of upstream `249cf25` (v0.1.8).
Companion artifacts: `phase0-fork-audit.md`, `phase1-code-map.md`.

## 1. Scope and the one architectural decision

Phase 2 delivers the control plane for two rank processes: rank/world
configuration, worker-process bootstrap, the bootstrap control protocol, and
clean two-sided shutdown. Per the plan it is deliberately narrow: **no engine
changes, no GPU code, no NCCL** — those arrive in Phase 3+.

One decision shapes everything else: the control plane is a **new workspace
crate, `paddock-dist`, with zero dependency on `paddock-engine`**.

- It can be compiled, tested, and reviewed without touching the inference
  stack or a CUDA toolchain.
- It cannot import engine internals, so it cannot entangle control-plane
  concerns with execution concerns.
- Phase 3's Communicator will also live here (behind a `cuda`-gated module)
  unless review pushes it elsewhere — see §8, open questions.

## 2. New crate: `crates/paddock-dist/` (4 files + tests)

### 2.1 `src/config.rs` — `ParallelConfig`

The rank/world configuration, following the runner's existing layering
convention (CLI > `PADDOCK_*` env > toml > defaults).

```rust
pub struct ParallelConfig {
    pub tp_size: Option<usize>,      // world size, 1 or 2
    pub rank: Option<usize>,         // 0 = coordinator, 1 = worker
    pub master_addr: Option<String>, // rank-0 control listener
    pub master_port: Option<u16>,    // default 11560 (DEFAULT_MASTER_PORT)
}
```

Key properties, for review:

- **All fields `Option`.** Unset is distinguishable from set, and a config
  that never mentions TP resolves to `None` — the historical single-process
  path, byte-identical behavior. This is the regression gate.
- **`resolved(serving_mode: bool) -> Result<Option<Resolved>>`** validates
  into a concrete role. Validation rules, in match order:
  - `(None, None)` → historical path.
  - rank without tp_size → `RankOutOfRange` (a rank alone is an error, never
    a silent TP=1 downgrade).
  - `tp_size=1` with rank 0/absent → historical path; with any other rank →
    error.
  - `tp_size=2, rank 0` → coordinator; rank 1 → worker. **rank 1 +
    `serving_mode=true` → `WorkerMustNotServe`**: a user-facing serving
    start of rank 1 is refused. `serving_mode=false` is only legitimate for
    the coordinator-spawned child (see §4).
  - `tp_size=2` without a rank → **defaults to rank 0** ("start the pair
    from this box"). Both nodes defaulting to 0 is visible (both wait for a
    worker) and can never half-serve. This default was added because the
    first live smoke test caught the earlier hard error as an operator-hostile
    dead end.
  - rank ≥ 2 → `RankOutOfRange`; other world sizes → `UnsupportedTpSize`
    (no silent downgrade to TP=1; the plan requires explicit refusal).
- **`Resolved::worker_env()`** produces the exact env for the spawned child:
  `PADDOCK_TP_SIZE=2, PADDOCK_TP_RANK=1, PADDOCK_TP_WORKER_CHILD=1,
  PADDOCK_TP_MASTER_PORT=..., PADDOCK_TP_MASTER_ADDR=...`. Bind wildcards
  ("0.0.0.0"/"::") are translated to loopback dial targets for the
  local-spawn case; a two-node start sets a real address explicitly.
- **`ParallelConfig::is_worker_child()`** reads the `PADDOCK_TP_WORKER_CHILD`
  marker — set only by `worker_env()`, never by a user.
- toml parsing goes through `from_toml_value(&toml::Value)` (the runner hands
  it the parsed document; the struct also carries `#[serde(deny_unknown_
  fields)]` for direct serde use). Unknown `[parallel]` keys are hard errors,
  matching the runner's config philosophy.

### 2.2 `src/protocol.rs` — the control wire

Length-prefixed JSON over TCP: u32 LE length, then JSON, one request/one
response per exchange.

- `MAX_FRAME = 1 MiB` enforced on both send and receive — a confused peer
  cannot exhaust memory. Receive reads with `read_exact` only after the
  length check.
- `ControlMessage` enum: `Hello { version, tp_size, who }`,
  `Welcome { tp_size, session }`, `Reject { reason }`,
  `Shutdown { graceful }`. `#[serde(tag = "type", rename_all =
  "snake_case")]` — the future execution vocabulary (Prefill/Decode/
  ResetSlot/...) extends this enum as new variants; the frame format does
  not change.
- `PROTOCOL_VERSION = 1` carried in Hello and enforced by both sides.
- `handshake(stream, tp_size, who)` = worker-side exchange; returns the
  coordinator-assigned session id or a `Rejected` error with the reason.

### 2.3 `src/worker.rs` — bootstrap, both roles

- `coordinate(resolved, spawn_worker)`: rank 0 binds the control plane
  (`master_addr:master_port`), optionally spawns the worker child, then
  accepts **exactly one** worker. Wrong-version / wrong-world-size dials are
  greeted with `Reject` and the loop keeps waiting for the real worker — the
  mismatched-world-size gate from the plan's cheap-test list.
- `spawn_worker_local(resolved)`: re-executes `current_exe()` (same binary)
  with `worker_env()` layered over the parent environment, `stdin/stdout`
  null-vs-inherit choices documented in the code (stdout/stderr inherit so
  the worker's tracing lines reach the operator's stream — the first smoke
  test showed nulled stdout hides them).
- `work(resolved)`: rank 1 dials with a 30 s retry budget (supervisor races
  and spawn-vs-bind ordering are normal, not errors), handshakes, then
  serves the Phase 2 control loop — `Shutdown { graceful: true }` → `Ok`,
  `graceful: false` → `BootstrapError::Aborted` (nonzero exit), any other
  frame → error. **The worker never binds an HTTP port and never touches the
  engine.**
- `coordinate_and_store` / `broadcast_shutdown`: the coordinator's accepted
  connection is held in a process-global (`OnceLock<Mutex<TcpStream>>`) so
  the runner's shutdown path can release the worker without threading the
  stream through every layer. `broadcast_shutdown(true)` sends
  `Shutdown { graceful: true }` and is best-effort (a dead worker is not an
  error at coordinator exit).

### 2.4 `tests/bootstrap.rs` — 17 tests

Covers the plan's Phase 2 cheap-test list: config validation (unset, TP=1
collapse, rank-without-size, TP=3 refusal, rank out of range, worker-must-
not-serve vs child-allowed, missing master_addr, wildcard default, worker_env
contents, toml parse + unknown-key rejection); framing (round-trip,
oversized-frame refusal without reading the payload); and live two-role
handshakes over localhost (graceful shutdown, non-graceful → Aborted,
impostor rejected then real worker accepted, version mismatch rejected,
wrong message shape rejected). All host-only, no CUDA, no engine.

## 3. `paddock-runner` changes (4 files)

### 3.1 `config.rs` (+31/−2)

- New `Config` field: `pub parallel: ParallelConfig` with
  `#[serde(default)]`. The runner's `Config` carries
  `deny_unknown_fields`, so the key must exist in the struct before any
  manager/endpoint file can legally contain `[parallel]` — the field and
  this change are the same release. An absent table parses to the default
  (unconfigured); existing endpoint files are unaffected.
- `merge_env` gains the `PADDOCK_TP_*` overlay via
  `ParallelConfig::merge_env()`, mapped into the existing
  `ConfigError::BadEnv` shape (no new error enum at the runner boundary).
- **`ENV_SURFACE` gains five names** (`PADDOCK_TP_SIZE`, `PADDOCK_TP_RANK`,
  `PADDOCK_TP_MASTER_ADDR`, `PADDOCK_TP_MASTER_PORT`,
  `PADDOCK_TP_WORKER_CHILD`) plus the dev knob `PADDOCK_TP_NO_SPAWN`
  (§4). This is load-bearing, not bookkeeping: hardened builds
  (`seal_environment`) delete any `PADDOCK_*` variable not on this list, so
  an unregistered name would make TP work on dev builds and silently fail in
  shipped ones. The comment chain at `merge_env` documents the same-file
  rule.
- `Config::default()` initializes the field.

### 3.2 `startup.rs` (+110/−1)

Three insertions, in execution order:

1. **Worker-child branch** (in `run()`, after the service-verb dispatch,
   BEFORE config resolution): if `is_worker_child()`, the process initializes
   logging, builds `ParallelConfig::from_worker_env(...)` from the four env
   vars, calls `resolved(false)` (serving_mode=false — the one legitimate
   rank-1 path), and enters `paddock_dist::worker::work(&r)`. It never
   reaches config resolution, the toml reader, the model scan, the HTTP
   listener, or the banner. Exit codes: 0 on graceful shutdown, 1 on
   bootstrap/control-loop error (logged), 2 if marked as child but the env
   is not a valid rank-1 config — **it refuses to fall back to serving**,
   which is the "rank-1 cannot accidentally serve requests independently"
   gate from the plan.
2. **`resolve()` overlay**: four `if cli.<tp_flag>.is_some()` assignments,
   identical in shape to the neighboring `gpu`/`kernel_pack` handling. CLI
   wins over toml/env like every other flag.
3. **Coordinator bootstrap** (after logging init, before the tokio runtime):
   `cfg.parallel.resolved(true)` — a validation error here is a clean
   `ExitCode::from(2)` with a plain message (config errors at this point are
   stderr messages by upstream design, since the log subscriber may not
   exist yet... note: logging IS initialized before this point, so the
   message goes to both). If the resolution is TP=2, rank 0:
   `coordinate_and_store(&resolved, spawn_worker)` runs to completion before
   the serving stack starts — the API never answers into a half-formed TP
   pair. Spawn is skipped when `PADDOCK_TP_NO_SPAWN` is set (two-node starts
   wait for an SSH-started worker instead of spawning a local child).

Also: the vLLM-compat rejection message for `--tensor-parallel-size` now
points at paddock's own `--tp-size 2` instead of declaring the feature
absent.

### 3.3 `lib.rs` (+5)

One insertion in the graceful-shutdown path of `run()`, immediately before
the engine drain: `paddock_dist::worker::broadcast_shutdown(true)`. The
worker is released first so both ranks free device memory in parallel
rather than a rank-1 CUDA context dying late and stalling the next start on
its card (the same concern the existing engine-free comment documents).

### 3.4 `Cargo.toml` (+1)

`paddock-dist = { path = "../paddock-dist" }` dependency. Workspace
`Cargo.toml`/`Cargo.lock` gain the new member.

## 4. Env knobs (complete list)

| Name | Set by | Meaning |
|---|---|---|
| `PADDOCK_TP_SIZE` | user/CLI equivalent | world size 1 or 2 |
| `PADDOCK_TP_RANK` | user or coordinator | 0 coordinator, 1 worker |
| `PADDOCK_TP_MASTER_ADDR` | user or coordinator | bind (rank 0) / dial (rank 1) |
| `PADDOCK_TP_MASTER_PORT` | user or coordinator | default 11560 |
| `PADDOCK_TP_WORKER_CHILD` | coordinator spawn only | worker-child marker; a user-set value with bad rank env exits 2, never falls back to serving |
| `PADDOCK_TP_NO_SPAWN` | operator (dev builds) | rank 0 skips local spawn, waits for SSH-started worker |

All six are in `ENV_SURFACE`; none are read anywhere outside
`paddock-dist`/the two runner branches documented above.

## 5. How to run it

> Historical note (upstream-readiness remediation): the two-node worker line
> below is the ORIGINAL Phase 2 recipe and did not work as written - a
> manually started worker refused with `WorkerMustNotServe` unless the
> operator also set the internal `PADDOCK_TP_WORKER_CHILD` marker plus
> `PADDOCK_TP_MODEL`/`PADDOCK_TP_PACK`. The supported operator path is now
> `--tp-worker` (see the current runbook in `docs/tp/phase9-report.md` or
> `paddock.example.toml`); the lines below are kept for phase-report
> provenance and are superseded.

```sh
# Single box, two processes (what the smoke test ran):
PADDOCK_PORT=11981 PADDOCK_TP_SIZE=2 PADDOCK_TP_MASTER_PORT=11982 ./paddock-runner
# → coordinator serves the API on 11981, spawns a rank-1 child, control plane on 11982.

# Two nodes (worker pre-started on the other box, e.g. via SSH):
# worker box:  PADDOCK_TP_SIZE=2 PADDOCK_TP_RANK=1 PADDOCK_TP_MASTER_ADDR=192.168.100.10 ./paddock-runner
# head box:    PADDOCK_TP_SIZE=2 PADDOCK_TP_MASTER_ADDR=0.0.0.0 PADDOCK_TP_NO_SPAWN=1 ./paddock-runner
```

## 6. Validation evidence

- `cargo test -p paddock-dist`: 17/17.
- `cargo test -p paddock-runner --lib`: 506/506 (single-process path
  unregressed).
- `cargo check --workspace`: clean; no new warnings.
- **Live two-process smoke run on the head Spark** (debug build): coordinator
  bound control plane → spawned worker child (pid verified) → worker dialed
  127.0.0.1 → handshake accepted (`worker 'worker' joined as rank 1
  (session 1)`) → coordinator proceeded to serve its API → SIGTERM → worker
  logged `shutdown from coordinator (graceful=true) - worker exiting
  cleanly` → zero surviving processes.
- First smoke run (pre-fix) intentionally failed with exit 2 on
  `--tp-size` without rank — that failure produced the default-to-rank-0
  behavior, which was then re-smoke-tested green.

## 7. Known limitations (deliberate, phase-scoped)

- The worker idles after handshake: the execution vocabulary
  (Prefill/Decode/...) is Phase 8 territory; shard loading is Phase 4.
- Control protocol is plain TCP/JSON on the fabric — no auth/TLS. It is a
  private-RoCE assumption, same trust level as the existing runner/manager
  channel; revisit if the plan's networking gate demands it.
- The control plane is synchronous std::net with blocking reads; fine for
  bootstrap and a shutdown line, will be revisited (or moved behind the
  Communicator's event model) when the execution vocabulary arrives.
- No manager (`paddock`) surface yet: starting a TP endpoint from the
  Studio needs the manager to know about `[parallel]` — deferred until the
  execution path justifies the UI.
- One cosmetic tradeoff: the spawned child inherits stdout/stderr, so its
  log lines interleave with the coordinator's in the same stream. Reviewers
  who want them separated can flip the Stdio choice in
  `spawn_worker_local` (documented there).

## 8. Open questions carried into Phase 3

- Communicator home: `paddock-dist` (with a cuda-gated module) vs
  `paddock-engine::gpu`. Default leaning: keep it in `paddock-dist` so the
  control/execution split stays physical, with `paddock-engine` depending on
  it — but Phase 3's stream/event plumbing may argue for engine-side.
  Decision at Phase 3 start, recorded in its report.
- cudarc `nccl` feature enablement + libnccl install path on both Sparks
  (phase1-code-map §10 item 1).
