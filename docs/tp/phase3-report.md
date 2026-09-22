# Phase 3 — Report: Engine-side NCCL communicator and two-Spark collectives

Implementation under review: uncommitted changes on `main` above `d5c78a9`
(the Phase 2 implementation is `7049d80`). Companion: `phase2-report.md`;
acceptance criteria: `../../../PADDOCK-PLAN.md` (repository sibling,
`../PADDOCK-PLAN.md` from the repository root). Paths and line references below
refer to this working tree; they may move after review.

## 1. Scope and architectural decision

Phase 3 adds a GPU communicator and a standalone two-rank collective probe.
It does **not** shard weights, run a TP model, or send scheduler/execution
commands. The Phase 2 report's open question about communicator ownership is
resolved: NCCL lives in `paddock-engine::gpu::distributed`; `paddock-dist`
remains GPU-free and supplies only rank configuration, TCP handshake, the
opaque ID frame, and shutdown. Dependency direction is engine → dist.

The normal runner's Phase 2 worker still waits only for `Shutdown`: it does not
consume `NcclId` or construct a communicator. The focused benchmark owns both
sides of that extended bootstrap. This is a component-level Phase 3 result,
not end-to-end TP serving.

## 2. Code-change map for review

| Path | Change and review focus |
|---|---|
| `Cargo.toml` | Enable cudarc 0.19.9 `nccl` (`nccl-02030`) with existing `cuda-13000` and `fallback-dynamic-loading`; add `paddock-dist` as a workspace dependency. Review the runtime-loading invariant, not merely feature compilation. |
| `Cargo.lock` | Engine dependency graph gains `paddock-dist`; no new third-party package entry for NCCL because cudarc already contains the bindings. |
| `crates/paddock-engine/Cargo.toml` | Add dependency on the GPU-free control crate. |
| `crates/paddock-engine/src/gpu/mod.rs` | Export the new `distributed` GPU module. No model call site changed. |
| `crates/paddock-dist/src/protocol.rs` | Add `NCCL_ID_BYTES = 128`, `NcclId { id: Vec<u8> }`, fixed-size send/receive helpers and `BadNcclId` for malformed length. The existing length-prefixed JSON framing and `MAX_FRAME` cap still apply. This is an opaque byte transport, with no cudarc/CUDA/NCCL imports. |
| `crates/paddock-dist/src/worker.rs` | Extract `connect_worker` so the benchmark can retain the post-handshake TCP stream to receive the ID. `work` retains the Phase 2 shutdown-only behavior; it rejects unexpected messages. |
| `crates/paddock-dist/tests/bootstrap.rs` | Add a host-only TCP round-trip of the 128-byte ID and malformed three-byte-length rejection (18 integration tests total). |
| `crates/paddock-engine/src/gpu/distributed.rs` | Add `CollectiveError`, rank-0 ID creation, `Communicator` interface, NCCL initialization, dedicated communication stream, event fences, and checked all-reduce/all-gather/reduce-scatter/broadcast enqueues. This is the main review target. |
| `crates/paddock-engine/examples/nccl_bench.rs` | Standalone two-process bootstrap, deterministic f32 parity, GPU-event timings, transport audit and TP=1 no-NCCL-load probe. Does not use the runner, model weights, or scheduler. |

### 2.1 Bootstrap/control flow

1. Rank 0 calls `coordinate` (existing Hello/Welcome handshake), then
   `create_unique_id` (`gpu/distributed.rs:72`) and `send_nccl_id`.
2. Rank 1 calls `connect_worker` (`worker.rs:196`), then
   `receive_nccl_id` (`protocol.rs:46`), which requires exactly 128 bytes.
3. Each rank creates a CUDA context and calls
   `NcclCommunicator::from_resolved` (`gpu/distributed.rs:118`), which
   validates TP=2/rank and initializes its own NCCL process group from the
   common ID on a dedicated stream.
4. Rank 0 sends the existing `Shutdown` after the benchmark; rank 1 checks
   that its graceful flag agrees with its local result. Both processes exit
   successfully on the tested path.

The ID rides TCP; tensor payloads go through NCCL. No per-batch control
messages or new wire format were introduced. `PROTOCOL_VERSION` remains 1
because the existing tagged message vocabulary was extended, not reframed.

### 2.2 GPU execution and lifetime contract

`Communicator` exposes `rank`, `world_size`, four collectives, and two fence
methods. `after_compute` records an event on the producer compute stream and
makes the NCCL stream wait; `before_compute` records completion on the NCCL
stream and makes the compute stream wait. These calls enqueue dependencies and
do not synchronize the host in the normal path. The caller must keep buffers
alive until the dependent stream passes completion. `all_reduce` checks equal
input/output lengths; `all_gather` and `reduce_scatter` check world-size
multiples; `broadcast` checks the root rank. NCCL errors/status and CUDA
errors are surfaced through `CollectiveError`.

The benchmark disables cudarc automatic cross-stream event tracking to
exercise the explicit fences. It synchronizes only at parity readback,
measurement boundaries, and teardown (`examples/nccl_bench.rs:68-89,
105-205`). GPU event elapsed time divides by the iteration count after five
warmup enqueues. This is a microbenchmark, not a model hot path or a proof of
concurrent scheduler/graph correctness.

### 2.3 TP=1 boundary

`from_resolved(None, ...)` returns `None` before an NCCL call. Cudarc's
fallback-dynamic-loading arrangement loads `libnccl.so` on first NCCL use,
not when the binary starts. The standalone `tp1` mode checks `/proc/self/maps`
before and after a real CUDA allocation/copy on each Spark. That proves this
binary can use CUDA while leaving NCCL unmapped; it does **not** establish a
TP=1 performance baseline or exercise the full production runner.

## 3. Validation evidence

- `cargo check -p paddock-dist -p paddock-engine --lib`: passed.
- `cargo check -p paddock-engine --example nccl_bench` and focused
  `cargo build -p paddock-engine --example nccl_bench`: passed; no CUDA kernel
  pack rebuild was requested.
- `cargo test -p paddock-dist`: 18/18 integration tests passed (17 existing
  Phase 2 tests plus the NCCL-ID transport test).
- `cargo clippy -p paddock-dist -p paddock-engine --lib --example nccl_bench`:
  passed with an existing warning in `crates/paddock-engine/src/cuda.rs:83`
  (`unnecessary_cast`, outside Phase 3). A prior `-D warnings` run was not
  green; do not report warning-free Clippy.
- `git diff --check`: passed. The new, untracked Rust files are not included
  in `git diff` until staged; build/check and live execution cover them.
- NCCL runtime: user-local `nvidia-nccl-cu13==2.30.7` installed on both
  aarch64 Sparks; `LD_LIBRARY_PATH` points to
  `$HOME/.local/nccl/nvidia/nccl/lib` for the benchmark processes. NCCL
  reported `2.30.7+cuda13.3` at runtime against CUDA driver 13.0.
- The same built binary was copied to rank 1 and SHA-256 matched on both
  nodes before the final run. The **final run with explicit event tracking
  disabled** exited 0 on both ranks. Deterministic f32 checks passed for
  all-reduce (`1 + 2 = 3` on both ranks), all-gather (rank order 1 then 2),
  reduce-scatter (3), and broadcast from rank 0 (7). Both logged clean
  communicator/bootstrap shutdown. TP=1 CUDA copy/no-`libnccl.so` probe
  passed separately on both ranks.

### 3.1 Two-Spark measurements

Final run, rank 0; input bytes for each operation; f32 buffers, five warmups,
30 timed iterations up to 1 MiB and 10 at 8/32 MiB. Timing is GPU events on
the communication stream. `alg_GBps = input_bytes / elapsed_seconds / 1e9`;
for two ranks this benchmark prints all-reduce `bus_GBps = alg_GBps`,
all-gather `bus_GBps = alg_GBps / 2`. The all-gather output has twice the
listed input bytes. Rank 1 also passed all sizes; small rank-to-rank timing
variation is expected.

| Input | All-reduce µs | All-reduce alg/bus GB/s | All-gather µs | All-gather alg/bus GB/s |
|---:|---:|---:|---:|---:|
| 1 KiB | 14.57 | 0.070 / 0.070 | 11.14 | 0.092 / 0.046 |
| 64 KiB | 45.76 | 1.432 / 1.432 | 89.20 | 0.735 / 0.367 |
| 1 MiB | 873.75 | 1.200 / 1.200 | 249.24 | 4.207 / 2.104 |
| 8 MiB | 812.36 | 10.326 / 10.326 | 815.79 | 10.283 / 5.141 |
| 32 MiB | 3010.35 | 11.146 / 11.146 | 3551.42 | 9.448 / 4.724 |

The 1 MiB all-reduce result is non-monotonic; these are one-run component
measurements, not a reproducible throughput comparison or a TP=2 speedup
claim. No TP=1 model performance baseline is implied.

### 3.2 Actual transport and limitation

`NCCL_DEBUG=INFO` showed `NET/IB`, `rocep1s0f0:1/RoCE` and OOB/bootstrap on
`enp1s0f0np0:192.168.100.10` on rank 0, and `NET/IB` transfers on the same
RoCE HCA. Rank 1 was reached at `192.168.100.11`. The explicit settings were
`NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_IB_HCA=rocep1s0f0`,
`NCCL_IB_DISABLE=0`, `NCCL_NET=IB` on both processes.

**Important qualification:** NCCL also printed `GPU Direct RDMA Disabled`
and `GDR 0` on both nodes. This proves the IB/RoCE network plugin and selected
HCA, but does **not** prove direct GPU-to-NIC DMA or the absence of host
staging. It leaves the plan's stronger "no unexpected host bounce" networking
invariant open; investigate GDR eligibility before making direct-path or
peak-link-bandwidth claims. Missing optional NCCL plugins were logged but
were not fatal. No network configuration was changed by this phase.

## 4. Reproduction and review boundaries

Build and run the benchmark binary with the same architecture/runtime on each
Spark. Start rank 1 on the worker (`192.168.100.11`), then rank 0 on the head;
use an unused control port (the run above used 11562). The binary's usage is
`nccl_bench 1 192.168.100.10 [port]` and
`nccl_bench 0 192.168.100.10 [port]`; use `nccl_bench tp1` for the independent
no-NCCL probe. Both ranks require the installed NCCL library directory on
`LD_LIBRARY_PATH` for TP=2; TP=1 was verified without it.

Review priorities:

1. `gpu/distributed.rs`: stream/event ordering, buffer lifetime, NCCL status
   handling, shape checks and TP=1 short-circuit.
2. `protocol.rs`/`worker.rs`: fixed-size ID validation and preservation of
   the existing worker's shutdown-only behavior.
3. `examples/nccl_bench.rs`: parity assertions, GPU-event timing definition,
   per-rank shutdown/error behavior and transport evidence.
4. Manifest changes: `nccl-02030` ABI and lazy runtime-loading assumption.

Deferred: runner/model integration, rank-local loaders, scheduler messages,
CUDA graph capture, full workspace build, model parity, and reproducible
performance comparison. The current work remains **uncommitted** pending
review; Phase 2's report describes its own historical state, and its
speculative §8 suggestion to put the communicator in dist is superseded here.
