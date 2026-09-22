# Phase 6 — Isolated Qwen3.8 GQA TP and rank-local KV proof

Status: isolated primitive and two-Spark GPU parity proven on `main` after accepted Phase 5 (`2546d17`); **not installed in the production qwen35 forward path**.

## Implementation boundary

- `crates/paddock-engine/src/gpu_model/qwen35/gqa_tp.rs`: checked TP=2 geometry splits four KV heads into two per rank and their 24 query heads into 12 per rank (six Q heads per KV head). `attn_q` (interleaved query/gate) and `attn_k/v` select output rows; `attn_output` selects input columns using Phase 4's pre-upload quant-block-aware loader. Q/K norm, M-RoPE, local KV append, scalar attention, output gate and projection run locally. Only the hidden-width output is summed via Phase 3 NCCL stream/event fences. Caller supplies the same post-attention-norm input to both ranks; residual, prenorm, rotation and FFN stay outside. Unsupported ragged/odd KV grouping rejects; no implicit KV replication. `paddock-dist` remains GPU-free.
- `crates/paddock-engine/src/gpu_model/qwen35/mod.rs`: export only this isolated primitive; no change to the existing TP=1 loader, forward, KV planner, or service path.
- `crates/paddock-engine/examples/qwen35_gqa_tp.rs`: two-process NCCL bootstrap and independent rank-0 full-weight/full-KV GPU oracle. Two rank processes each own a local KV cache; rank 0's full copy is solely a parity oracle. Four deterministic one-token decode steps and exact reset replay (twice) use f16 KV, max context 16, no graph or speculation.

Each rank's 16-position K+V allocation is 32,768 bytes (two local KV heads × 256 elements × two bytes × 16 positions × two planes); a full four-head payload would be twice that. The small probe uses slot 0 and contiguous KV, not the production paged-KV allocator. Logical position and slot are mirrored by the deterministic probe inputs; production block-table/prefix mirroring, free/cancellation and per-layer `Demand`/`Plan` byte contributions are **not yet validated or changed**. These remain integration gates; this proof must not be mistaken for serving support.

## Attribution

The full-GQA-group split, interleaved Q/gate row ranges, KV ownership, and whole-weight GPU oracle pattern are adapted from Erik Bogado's `ErikBPF/paddock` `contrib/tp-09-attention-block` (`afedfdd`), `tensor_attention.rs` and `gpu_tensor_attention.rs`, dual-licensed MIT OR Apache-2.0. This implementation uses rank-local processes, Phase 4's common GGUF slicing, and Phase 3 NCCL rather than the fork's two local `CudaContext`s, `Link` input broadcast or staged-copy reduction. Attribution also appears next to the primitive; carry it into a future commit message/PR note if publishing.

## Validation

- Pinned `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M checkpoint at revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`, SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` verified on both Sparks. Existing CUDA pack SHA-256 `fb6e7797ba0189036c69f0929038a045a4f8f848363efc7fd81c1051ed727c2e` matched on both. Final example binary hash `e22cc42b9805cce8d4773c15c9db8f5de254417c5067c85e6c603031d48f9905` matched after transfer. No kernel pack rebuild.
- `cargo test -p paddock-engine --lib complete_groups_and_kv_payload`: 1 passed, testing Q/KV geometry, rank ranges, KV byte count, odd/ragged/rejected ranks and overflow.
- `cargo clippy -p paddock-engine --lib --example qwen35_gqa_tp`, `cargo build -p paddock-engine --example qwen35_gqa_tp`, `git diff --check`: passed. Clippy still reports the pre-existing `src/cuda.rs:83` unnecessary cast.
- Real two-node NCCL probe on head `192.168.100.10` and worker `192.168.100.11`, layer 3 (initial build) and layer 7 (final rebuilt binary): both ranks exited 0, all eight layer-7 output vectors passed every-element `abs(tp - serial) <= 1e-3 + 1e-3*abs(serial)` and matched each rank's own first-pass output exactly after reset. Layer-7 maximum absolute deviation per step was 0.00000024, 0.00000048, 0.00000024, 0.00000024; the same sequence repeated after reset. Rank-0 and rank-1 checksums agreed each step (70.730698, 46.368462, 57.496704, 49.317394). Earlier layer-3 run also passed all eight comparisons; its maximum absolute deviation across steps was 0.00000191. No full model/token parity or speedup claim follows from this component proof.

## Next boundary

Keep this standalone while designing the rank-0 authoritative logical block-table/slot lifecycle and rank-local per-layer KV accounting. Test mirrored allocation, free, reset, prefix reuse and paged payload numerics across ranks before wiring it into the qwen35 model forward. DeltaNet, scheduler protocol, graphs, offload and speculation remain outside Phase 6.
