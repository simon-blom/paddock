# Phase 0 — Audit of existing two-GPU work (fork/PR reuse map)

Audit target: `truespar/paddock#22`, the `ErikBPF/paddock` 14-PR draft stack, and the
layer-split prerequisite it builds on. Recorded against current `main` at `249cf25`
(v0.1.8). Companion to `phase0-code-map.md` (now the Phase 1 artifact).

## 1. What the prior work is

- Base: `ded32a8` "feat: add multi-GPU layer split for qwen35 serving" by Zack Barra
  (@pagaldilz) — a **layer-split (pipeline-style) prerequisite**, NOT in upstream
  history (merge-base with upstream main is `7d0c62a`, i.e. v0.1.7-era "DGX Spark
  support and arm64 builds").
- Stack: 14 chained draft PRs on `ErikBPF/paddock` (`contrib/tp-01..14`), each
  targeting the previous branch; total 56 files, +7,710/−866 lines.
- Target: **qwen35 family tensor splitting**, validated with the pinned
  `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M checkpoint (model sha256
  `322e194f...23482`) on 2× RTX 5060 Ti 16 GB, FP8 KV, ctx 16,384, batch 1,
  speculation off. Note: Qwen3.8-27B dense is served through the qwen35 family.
- Evidence (from issue #22 + `docs/tensor-split-validation.md` in the fork):
  24 strict full-model logit comparisons with ZERO absolute error; 880 all-layer
  mixed-type comparisons (max abs err 3.8e-6); 15,998-token capacity gate +
  exact reset replay; CPU-only workspace suite passes; bench receipts with
  ABBA ordering. Honest scoping: no batching/MTP/multimodal claims, FFN mode
  vs full mode effectively tied on their workload, TTFT win (−55.9%) came from
  the tp-14 copy-batching fix over the naive gather loops.

## 2. Architecture classification (the part NOT to inherit)

The fork is a **single-process, multi-`CudaContext` design**: `load_split(execs)`
takes a `Vec` of executors in one process; cross-device movement is checked
copy helpers in `gpu/transfer.rs` (`copy_packed_rows`, `copy_pitched_rows`,
slice/region copies) over the driver-mediated path; reductions are staged
copies + pack-side kernels. The tested pair had **no P2P**.

Our design (plan-mandated): two OS processes (rank 0/1), NCCL collectives over
the ConnectX-7 RoCE lane, Communicator abstraction owning streams/events.
Therefore: every `Link`/transfer/reduction mechanism in tp-03/tp-04 is
**rewritten around NCCL**, not ported. What ports cleanly is the
*sharding semantics* above the transport: which tensors split on which axis,
quant-alignment constraints, KV/head ownership, parity methodology.

## 3. Component-by-component mapping

| Component (PR) | Content | Verdict |
|---|---|---|
| base `ded32a8` (pagaldilz) | layer-split placement, VRAM gates, executor links | **Port selectively**: placement/VRAM-gate logic informs rank-local memory planning; the layer-split mechanism itself is superseded by TP rank processes. Credit Zack Barra. |
| tp-01 cuda: per-device kernel attrs | per-context kernel attribute init (multi-context correctness) | **Port** (re-audit the 16-file set; part is inherited runner test-signature repair). |
| tp-02 qwen35: quantized prefill fallback | correctness fallback for quantized prefill under split | **Port** after re-diff against current ops.rs. |
| tp-03 cuda: cross-device projection scratch | f32 scratch retention across devices | **Reference**: the scratch-ownership idea adapts to NCCL buffer strategy; mechanism replaced. |
| tp-04 cuda: checked transfer + reduction | the transport layer | **Rewrite around NCCL** (this is precisely what the Communicator replaces). Keep its *checked/validated* API discipline and any transfer tests as oracle. |
| tp-05 gguf: quantized matrix shard loaders | rank-local GGUF shard loading with quant alignment | **Port** — direct input to Phase 4 `TensorSliceRequest` design. |
| tp-06 qwen35: two-GPU FFN block | gate/up column-split, down row-split semantics | **Port/adapt** to Phase 5 column/row-parallel layer semantics via Communicator. |
| tp-07 qwen35: FFN serving parity | parity harness, runner split config, Studio round-trip tests | **Reuse tests**; port config-surface ideas (runner split settings) when TP config lands in Phase 2. |
| tp-08 qwen35: bound + batch FFN prefill | prefill chunk bounding on split path | **Port** — informs chunked-prefill preservation under TP (Phase 9). |
| tp-09 qwen35: sharded GQA attention | head-sharding incl. GQA kv-head grouping | **Port** — core Phase 6 input; attribution example in the plan names exactly this PR. |
| tp-10 qwen35: sharded DeltaNet decode | recurrent-state sharding for decode | **Port** for Phase 12 (hybrid models); reference-only during dense bring-up. |
| tp-11 qwen35: sharded DeltaNet prefill | DeltaNet prefill splitting | **Port** for Phase 12, same posture as tp-10. |
| tp-12 qwen35: mixed tensor quantization | per-tensor quant-type variance across shards | **Port** — feeds Phase 4 packed/quant shard-alignment rules. |
| tp-13 qwen35: full tensor splitting | end-to-end integration of the above | **Reference/test oracle**: proves composition + holds the 24-comparison harness; the in-process architecture is not ported. |
| tp-14 qwen35: batch DeltaNet band copies | pitched/batched copy optimization (864 copies vs 1.7M) | **Reference** — technique reusable later if NCCL staging needs batching; do not port the copy loops themselves. |

Test assets (`crates/paddock-engine/tests/gpu_tensor_*.rs`, 20 binaries):
reuse as **test oracle patterns**. Geometry-pinned to the UD-Q4_K_M checkpoint;
GPU-gated (`--ignored`) by design. The in-process ones (two contexts in one
test) must be reworked for two-process ranks; the host-side tensor-split
row/column tests port more directly. Their parity methodology (exact logit
comparison, reset replay, capacity gate, cold-start regression) is the model
for our Phase 7 gates.

Also inherited in the stack (low priority): runner split-settings +
Studio round-trip test (`server-form-roundtrip.test.ts`), bench replay
(`scripts/bench_tensor_split.py` + fixture), `smem-attribute-check.py`.

## 4. Port-vs-main drift (integration base decision)

- Fork base `ded32a8` is NOT upstream ancestry; upstream main has 18 commits
  since merge-base `7d0c62a`, adding 11.5k lines, concentrated in the CUDA pack
  (NVFP4/MoE/ternary/hadamard kernels, qwen4exp) — the fork does not touch those.
- qwen35 Rust module drift since base: 2 upstream commits, 10 files,
  +1,260/−101 (`batch.rs`, `prefix.rs`, `load.rs` largest) — moderate and
  concentrated, not scattered.
- **Decision: port onto current main (`249cf25`), not onto the `ded32a8` base.**
  Building on `ded32a8` would inherit a dead prerequisite layer and re-create
  the rebase problem the plan warns about. Per-PR porting order: tp-01, tp-05,
  tp-12 (loader/alignment), tp-06/tp-09 (linear/attention semantics), tp-02/tp-08,
  tp-10/tp-11 (Phase 12), with tp-04/tp-13 consciously NOT ported (replaced by
  Communicator/NCCL). Each port commit carries attribution per plan §6.

## 5. Attribution register (captured now, per plan requirement)

- Zack Barra (@pagaldilz) — layer-split prerequisite `ded32a814cc0a25c1d8d141e523363e0356ca180`
  (placement/VRAM-gate logic if adapted).
- ErikBPF — draft PR stack #1..#14 on `ErikBPF/paddock`, tracking issue
  `truespar/paddock#22`, validation doc at fork rev `c54a419`.
- License: repo is MIT OR Apache-2.0 dual-licensed (both LICENSE files present in
  fork and main); fork work carries the same notices. Commit-message pattern per
  plan §6: "Adapted from ErikBPF/paddock PR #N (title), ported to current
  Paddock and replaced local peer transfers with the distributed
  communicator/NCCL path."

## 6. Consequences for the plan sequence

1. Phase 6 (dense bring-up model): the fork's proven sharding semantics are for
   the **qwen35 family with Qwen3.8-27B UD-Q4_K_M** — the exact checkpoint,
   quant, and parity receipts exist. This is a strong argument to make
   Qwen3.8-27B (UD-Q4_K_M GGUF, pinned rev `4ca7207`) the dense bring-up model
   instead of Qwen 3.5 9B, reusing tp-05/06/09 semantics directly. Open choice
   for the user.
2. Phase 4 shard loaders: tp-05 gives GGUF shard loading with quant alignment
   already thought through for Q4_K — reduces Phase 4 risk materially.
3. Phase 12 (hybrid/DeltaNet): tp-10/11/14 constitute a worked DeltaNet TP
   design (decode + prefill + copy batching) — the hardest Phase 12 problem is
   partially pre-solved as reference.
4. Validation controls the fork needed (`PADDOCK_NO_PREFILL_GRAPH=1`,
   `PADDOCK_KQ_EXACT_GEMV=1`) are Phase 10 (CUDA graphs) measurement caveats to
   record, not defaults to adopt.
5. Fork targets 2 GPUs in one box over PCIe with no P2P; our lane is
   2 nodes over RoCE. All transport performance expectations must be re-measured
   (Phase 3 microbenchmark); none of the fork's timing numbers transfer.
