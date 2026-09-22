# Phase 4 — Rank-local tensor slicing and pre-upload GGUF loading

Status: implementation for review on `main` after Phase 3 (`e1d5d82`).
This is a host-side shard-loader milestone, not distributed model execution.
`paddock-dist`, the Phase 3 communicator, runner scheduling, and model forward
paths were not changed.

## 1. Files changed

| File | Purpose |
|---|---|
| `crates/paddock-models/src/tensor_slice.rs` (new) | Shared matrix shard request, checked byte/shape slicing, GGUF and Safetensors adapters, synthetic and pinned-checkpoint tests. |
| `crates/paddock-models/src/lib.rs` | Export the common host-only module. |
| `crates/paddock-engine/src/gpu/kquant.rs` | `GpuExecutor::load_quantw_shard`: use the host-selected GGUF bytes and local dimensions for existing Q8/k-quant repackers. No change to `load_quantw` or existing TP=1 callers. |
| `docs/tp/phase4-report.md` (this file) | Scope, attribution, validation and handoff. |

No pack/CUDA kernel source changed and no kernel pack was rebuilt.

## 2. Loader/shard abstraction

`TensorSliceRequest { kind, rank, world_size }` selects `Replicated`,
`OutputRows`, or `InputColumns`. The result `TensorShard { bytes, dims }`
has logical `[input, output]` dimensions; bytes are a borrowed mmap region
for whole/row slices, or a rank-local gathered host buffer for columns.
Both file formats share checked `slice_matrix` geometry/byte logic:

- GGUF supplies `[input, output]` and its **per-tensor**
  `GgmlType::block_layout()` (elements and bytes per block). Output rows
  occupy one contiguous region; input columns select whole blocks in *each*
  output row, preserving row stride. Before selection, the loader checks
  nonzero dimensions, whole blocks per row, checked row/total byte lengths,
  valid rank/world, equal rank division and block-aligned column endpoints.
  It returns the source GGML type with local dimensions.
- Safetensors supplies `[output, input]` (transposed logically), and plain
  BF16/F16/F32 bytes. The same row/column semantics run with one element per
  block and dtype-specific element width. Its adapter returns the source
  dtype and `[input, output]` local dimensions. This covers the native
  Safetensors matrix slicing seam; actual family-specific Safetensors TP
  weight upload policy is deferred until those model paths need TP.
- `load_quantw_shard` calls `gguf_shard` **before** invoking the established
  `repack_q8_blocks` or `repack_kquant_raw` GPU upload/repack seams. It keeps
  the existing per-tensor Q8, Q4_0 fallback, i-quant and k-quant pack
  capability checks and resident `QuantW` variants. No whole GGUF tensor is
  uploaded and sliced afterward. The old `load_quantw` path remains the
  untouched TP=1 path. Projection-specific shard-kind selection belongs to
  Phase 5; there is no unvalidated automatic split based on tensor name.

A memory-mapped source file is of course still accessible to each rank; the
claim is **rank-local staged/uploaded bytes**, not that the filesystem or
virtual address mapping excludes other ranks' weights.

## 3. Fork provenance and attribution

Read local refs `erikbpf/contrib/tp-05-shard-loaders` (`895f8db`) and
`erikbpf/contrib/tp-12-mixed-types` (`d856695`), by Erik Bogado / ErikBPF.
The former's `gpu/kquant.rs` host row selector (original lines 2072–2102),
per-row whole-block column gather (2145–2183), and reconstruction test
pattern (2104–2143, 2185–2210) informed this adaptation. The latter's
per-tensor type/layout selection in qwen35 attention/DeltaNet loaders
informed the common layout lookup: do not presume every tensor in a
UD-Q4_K_M file is Q4_K. The project and fork are dual-licensed MIT OR
Apache-2.0. Attribution is retained in the new module's header. This is
an independent API adapted from those *host slicing ideas*, rather than a
copy of their same-process link, peer-copy, reduction or model execution.

## 4. Supported and unsupported cases

- Host-side GGUF byte slicing uses the **verified** GGML block layouts in
  `paddock-models::ggml_type` (including Q4_K, Q5_K, Q6_K, Q3_K, IQ4_XS,
  Q8_0, IQ4_NL and plain F16/BF16/F32). The engine shard upload method
  accepts only the types its existing `load_quantw` quantized-resident lanes
  can serve; a type's known byte layout alone does not promise a kernel.
  In particular, Q4_0 follows the existing exact Q8_0 transcode when the
  pack cannot serve it directly. Different projection tensors may select
  different layouts in one model.
- The pinned target is `unsloth/Qwen3.8-27B-GGUF`, revision
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`, file
  `Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256
  `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
  The local file's checksum was reverified before the optional real-file
  test. It contains mixed block-0 quant types, including Q5_K, Q3_K and
  IQ4_XS; the test checks both shard axes against actual mmap bytes.
- Column splits that cut a GGML block (e.g., half of a 256-wide Q4_K row)
  are rejected, never rounded or silently dequantized. Non-2-D tensors,
  zero-sized axes, odd/uneven TP=2 axes, invalid rank/world, unknown GGML
  layouts, mismatched byte spans and overflow are rejected. No special
  ragged-shard/padding policy exists in this phase.
- Safetensors plain BF16/F16/F32 matrix slices work. F8 with separate
  scales, MXFP4, NVFP4 packed U8 with scale/metadata companions, U32
  packed formats and other composite planes are **not** treated as ordinary
  bytes by the Safetensors adapter. Their coupled payload/scale slicing
  needs explicit format-specific rules; they fail closed. Embeddings,
  fused projections, expert 3-D tensors and non-matrix side weights need
  a caller-specific shard/replication policy in later phases.

## 5. Validation performed

- `sha256sum` of the local pinned GGUF matched the exact digest above.
- `cargo test -p paddock-models tensor_slice --lib`: 4 passed, one
  real-checkpoint test ignored by default. Synthetic host tests reconstruct
  GGUF Q4_K/Q5_K/Q6_K/Q3_K/IQ4_XS/Q8_0/IQ4_NL output-row and per-row
  input-column blocks; verify borrowed rows/owned columns, TP=1 whole-byte
  equivalence, plain matrix reconstruction, invalid shape/byte length,
  non-block-aligned columns, odd output counts and invalid rank.
- A generated single-file Safetensors fixture opened through the normal
  loader checks BF16 row/column byte selection and rejects packed U8/missing
  names.
- `PADDOCK_TP_TEST_GGUF=... cargo test -p paddock-models
  tensor_slice::tests::pinned_checkpoint_mixed_quant_shards --lib --
  --ignored`: passed on the checksum-verified checkpoint. This checks
  output-row reconstruction and both column shards against each source row
  for representative mixed block-0 matrix types.
- `cargo test -p paddock-models`: 102 passed, 3 ignored (two unrelated
  optional artifact tests and the real-checkpoint test, run separately).
- `cargo check -p paddock-engine --lib`: passed.
- `cargo clippy -p paddock-models -p paddock-engine --lib`: passed with the
  existing `crates/paddock-engine/src/cuda.rs:83` unnecessary-cast warning; no warning-free claim.
- `git diff --check`: passed. No full CUDA build or GPU/model execution run.

These tests prove host byte selection and compilation of the GPU upload seam;
they do **not** prove model numerical parity, GPU repack parity of every type,
or TP serving. There are no new scheduler messages.

## 6. Phase 5 handoff

Phase 5 may rely on checked TP=2 output-row/input-column GGUF byte selection
and a `QuantW` upload path that stages only those selected bytes. It must
assign explicit shard kinds for qwen35 gate/up/down projections, account for
fused/side-weight policies, and validate GPU repack + FFN numerics against
TP=1 before declaring a model path supported. The same logical request and
shape convention applies to plain Safetensors matrices; packed Safetensors
requires paired-format adapters before model-specific use. Preserve the
Phase 3 engine-owned NCCL boundary and keep `paddock-dist` GPU-free.
