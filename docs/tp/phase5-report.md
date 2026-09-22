# Phase 5 — Qwen3.8 dense FFN tensor-parallel primitive

Status: implementation for review, following accepted Phase 4 commit `29e3342`.
This is an isolated one-layer, one-token GPU FFN proof across the two Sparks;
it does not install TP execution in the model runner.

## Files and execution boundary

| File | Change |
|---|---|
| `crates/paddock-engine/src/gpu_model/qwen35/ffn_tp.rs` | New `FfnTpRank`: load three projection shards through Phase 4; execute local gate/up GEMVs, local SwiGLU, local down GEMV, and Phase 3 NCCL all-reduce. Checks rank/world, matrix geometry, input length and CUDA context. Includes a host-only small-matrix split/reconstruction test. |
| `crates/paddock-engine/src/gpu_model/qwen35/mod.rs` | Expose the isolated FFN TP primitive; existing qwen35 forward and TP=1 loaders remain untouched. |
| `crates/paddock-engine/examples/qwen35_ffn_tp.rs` | Standalone two-process bootstrap and real-checkpoint GPU parity oracle, with chosen layer and multiple deterministic input vectors. |
| `docs/tp/phase5-report.md` | This review handoff. |

Rank-local loading requests `OutputRows` for `ffn_gate` and `ffn_up`
(column-parallel outputs), and `InputColumns` for `ffn_down` (row-parallel
input). The two rank-local gate/up activations and SwiGLU output stay local;
only the hidden-width down result enters an all-reduce. The input is already
post-attention-normalized on each rank; residual addition and norm are outside
this primitive, so they cannot accidentally be applied twice. The TP loading
path uploads only rank-local weight bytes. The probe's independent serial
oracle uploads whole weights on rank 0 **only for verification**.
`paddock-dist` remains GPU-free; the control TCP connection carries bootstrap
ID and shutdown only.
No kernel pack was rebuilt, nor were Phase 3 NCCL semantics changed.

## Prior work / attribution

Erik Bogado's `ErikBPF/paddock` branch `contrib/tp-06-ffn-block`
(`6142c62`, `crates/paddock-engine/src/gpu_model/qwen35/tensor_ffn.rs`)
provides the gate/up output-row split, local SwiGLU and down input-column
split design, and its `tests/gpu_tensor_ffn.rs` compares against a serial
same-weight GPU oracle. This implementation adapts those concepts to **one
rank per process**. The fork's same-process `Link` broadcast, peer copy and
staged-copy reduction are not used: the Phase 3 `Communicator` fences the
engine compute stream and sums with NCCL. The in-repo `tensor_slice` module
(from Phase 4, attributed to `tp-05`/`tp-12`) enforces per-tensor quant-block
alignment and pre-upload selection. Source attribution is also present in
the new primitive's module header. Paddock and the fork are MIT OR Apache-2.0.

## Validation

- `cargo test -p paddock-engine --lib dense_ffn_channel_split_reconstructs_serial`:
  1 passed (small synthetic gate/up/down matrices, local activation and
  partial-output reconstruction).
- `cargo test -p paddock-models tensor_slice --lib`: 4 passed; the optional
  pinned-file test is ignored by default and was exercised in Phase 4.
- `cargo check -p paddock-engine --example qwen35_ffn_tp`,
  `cargo build -p paddock-engine --example qwen35_ffn_tp`, and
  `cargo clippy -p paddock-engine --lib --example qwen35_ffn_tp`: passed.
  Clippy still prints the pre-existing `src/cuda.rs:83` unnecessary-cast
  warning; there is no warning-free claim.
- `nccl_bench tp1`: passed a fresh-process CUDA allocation/copy and confirmed
  `libnccl.so` was absent from `/proc/self/maps`. The ordinary TP=1 qwen35
  forward code was not edited.
- Pinned checkpoint: `Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256
  `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
  A copy on the worker at `/home/sime/ffn-tp/` was hash-matched to the pinned
  local file. The test executable and existing CUDA pack were copied to the
  worker; executable and pack hashes matched the head before the first run.
- Real GPU test on head `192.168.100.10` and worker `192.168.100.11`, each
  one process/GPU, using the Phase 3 NCCL communicator. Layer 0 gate type
  `IQ4_XS` and layer 24 mixed gate/up/down `Q5_K/Q4_K/Q5_K` passed all three
  deterministic inputs apiece against the rank-0 serial same-weight GPU
  oracle, with criterion `abs(tp-serial) <= 1e-3 + 1e-3*abs(serial)` for
  **every** hidden output element. The final rebuilt binary's layer-24
  max absolute deviations were `0.00000012` for each of seeds 3, 17, 53;
  max reported relative deviations were `0.00018626`, `0.00012778`,
  `0.00044703`. Both ranks' output checksums agreed for each seed and both
  shut down cleanly. Earlier layer-0 max absolute deviations were
  `0.00000012`, `0.00000381`, `0.00000012` for the same seeds.

These are FFN-only GPU parity tests, not whole-layer token/logit parity or
throughput measurements. The copied checkpoint consumes approximately 16.5 GB
of worker storage and remains under `/home/sime/ffn-tp/` for follow-on probes.

## Limits and next gate

- This primitive currently accepts TP=2, one already-normalized token on each
  rank, plain dense qwen35 FFN weights in GGUF supported by the resident
  `QuantW` GEMV paths. It rejects geometry/quant-block cases that Phase 4
  cannot split evenly; mixed per-projection types are handled individually.
  Multi-row prefill, MoE FFN, rotation/fused planes and alternate Safetensors
  layouts have no TP integration here.
- The rank-local input must be mirrored by a future mixer path; its production
  ownership/synchronization is not implemented. There is no scheduler,
  end-to-end model walk, GQA, DeltaNet, graph capture, speculation or offload
  change. Preserve this isolation until the remaining model paths have
  explicit rank-local ownership and numerical tests.
