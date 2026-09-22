# Phase 1 — Code map and baseline (TP=2 plan)

Plan numbering after the v2 revision: this is the **Phase 1** artifact
("Baseline and code map"); the Phase 0 artifact is `phase0-fork-audit.md`.
Recorded against `main` at `249cf25` ("Clarify macOS preview status and download
options"), version 0.1.8, on the head DGX Spark (GB10, aarch64, CUDA 13.0,
rustc 1.98.1 per rust-toolchain.toml).

Status: code map complete. TP=1 baseline numbers are recorded as pending —
the GPU was occupied by the GLM-5.3-Flash vLLM lane (VLLM::Worker_TP0,
~108 GiB) at measurement time and per user decision we wait for it to free
rather than stop it. The command to run is in "Baseline procedure".

**Bring-up model decision (user, 2026-09-22): Qwen3.8-27B** in the
`unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M checkpoint, pinned upstream revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, sha256
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` —
deliberately the exact checkpoint the ErikBPF stack's parity receipts pin
(24 zero-error full-model comparisons), so fork tp-05/06/09 semantics have a
known-good oracle from day one. This is the qwen35 family (dense 27B served
through it); the fork's tensor_attention/ffn modules are its direct input.
Baseline artifact will be `baseline-tp1-qwen38-27b.md` in this directory.

## 1. Crate map (workspace: 20 crates)

Serving path, outer to inner:

- `paddock-runner` — the serving binary. HTTP routes (OpenAI/Anthropic/Ollama
  compat) in `routes.rs`/`serving.rs`; talks to the engine through
  `paddock_engine::service::Engine` only (51 `paddock_engine::` call sites in
  `serving.rs`). One model, one port. Headless data plane.
- `paddock` (`paddock-manager`) — supervisor + Studio; starts/stops runners.
  Not on the inference hot path.
- `paddock-engine` — everything below the API. Key modules:
  - `service.rs` — `Engine::spawn(max_batch, build)` + `submit(GenRequest)`;
    owns the worker thread/tick loop, `EngineError`, `ShutdownCtl`.
  - `generator.rs` (1800 lines) — the per-tick execution: multimodal admission
    (`MmAdmit`), speculative acceptance (`SpecAccepted`, `SpecRsDraw`), row
    sampling (`RowSample`, `SampledStep`, `FinishSample`), `GenError`.
  - `gpu/` — the CUDA device layer (see §2).
  - `gpu_model/` — per-family model implementations (see §4).
  - `kv_plan.rs` / `kv_pool.rs` / `paged_radix.rs` — KV memory plan
    (`Demand`/`Plan`/`grant`, graph-scratch reserve), pool allocator, radix
    tree for prefix caching.
  - `kv_tier/` — host/NVME KV offload (accounting, catalog, cost, restore
    flow, ram transport, nvme store).
  - `spec.rs` / `spec_policy.rs` — speculative decode machinery.
  - `backend.rs` — the `Backend` seam (identity/health only in P0; CUDA +
    Metal are the backends).
- `paddock-kernels` — C-ABI loader for the kernel pack (`abi.rs`, 11k lines,
  `loader.rs` dlopen). The pack calls the CUDA runtime itself
  (`cudaLaunchKernel` etc.); Rust sees exported entry points only.
- `packs/cuda` — the CUDA source pack (`pack.cu`, `src/`, built by
  `build.sh` into `pd-cuda-smXXX.so` or `libpd-cuda.a` with `--static`).
- `paddock-bench` — the harness for baseline numbers: in-process mode
  (synthetic prompt, warmup discarded) and HTTP mode (streamed, TTFT real,
  prefill not observable).
- `paddock-metal` + `apps/macos` — the Apple-Silicon backend. Fully disjoint
  from the CUDA TP lane; no changes needed there.
- Others (`paddock-models`, `paddock-tokenizer`, `paddock-tls`, `paddock-api`,
  `paddock-mcp`, `paddock-estimator`, `paddock-filemeta`, `paddock-forensics`,
  `paddock-geo`, `paddock-heif`, `paddock-pdfium`, `paddock-admin`,
  `paddock-websearch`, `paddock-desktop`) — support surfaces, not TP-relevant.

## 2. CUDA runtime layer — where Phase 2 plugs in

The engine does NOT use raw `cudaStream*` calls. It uses **cudarc 0.19.9**
(workspace dep, `default-features = false`, behind the `cuda` feature of
`paddock-engine`). Consequences for the plan:

- `CudaContext` / `CudaStream` (safe wrappers) instead of raw handles.
  `gpu/mod.rs` owns: main compute `stream` (created non-blocking, given the
  GREATEST priority), `copy_stream` (event-gated device->host reads), and
  `side_stream` (second compute stream for decode-tick sub-DAG overlap, e.g.
  gemma4 A4B; `stream_ptr()` can flip main<->side).
- CUDA graphs via `gpu/graph.rs`: capture with
  `result::stream::begin_capture` on a real (non-legacy-default) stream,
  instantiate to `PaddockGraph`, replay with `result::graph::launch`.
  Replay is stream-agnostic (`launch_on`); capture is stream-bound. Cross-
  stream event tracking in cudarc is explicitly disabled at context creation
  (pure overhead + capture-illegal).
- **NCCL is already available as a cudarc feature**: `nccl = ["nccl-02030"]`
  in cudarc's own Cargo.toml. Phase 2 should add `features = ["nccl"]` to the
  workspace cudarc dep rather than introducing a new binding. NOTE: no
  libnccl is installed on the head node yet (checked ldconfig) — NCCL also
  needs to be present on both nodes, and cudarc's `nccl-02030` pins
  NCCL 2.3.0-era ABI? (verify exact version at Phase 2; ABI suffix 02030 maps
  to NCCL 2.30). GB10 aarch64 + ConnectX-7 RoCE is a supported NCCL transport
  family but must be measured (Phase 2 microbenchmark decides RoCE vs SOC
  paths).
- Kernel launches go through `paddock-kernels`' C ABI, not cudarc. The pack
  itself calls the CUDA runtime. TP collectives must therefore live in the
  Rust layer (cudarc NCCL) or a new pack export — do not assume the existing
  pack entry table can host collectives; the Communicator owns them.

## 3. Loaders

- GGUF: `paddock-models` (gguf crate) reads weights; engine maps per family.
- Safetensors: `gpu_model/st_load.rs` (194 lines) is the ST entry; per-family
  code consumes mapped tensors. Upstream formats: native FP8, NVFP4 (modelopt
  group-16), MXFP4, Q8_0, Q4_K_XL/L/M.
- The existing sharding-relevant facts to extend in Phase 3: tensors arrive
  name->bytes from either loader; per-family load code uploads them to device
  via `gpu/upload.rs` / `transfer.rs`. Rank-local slicing must happen BEFORE
  upload (plan's "load only the local weight shard" rule).
- Packed-format alignment constraint confirmed live: the Qwen3.8-Flash-Next
  NVFP4 checkpoint on disk stores routed experts as separate
  `layer-NNNNN-experts-AAAA-BBBB.safetensors` planes (512 experts in 128-block
  planes) plus bf16/ple-fp8 planes — MoE expert files are already physically
  chunked, which helps expert-shard loading but means expert-plane file
  boundaries are an additional alignment unit to respect.

## 4. Model families (gpu_model/) and TP posture

Family directories: `qwen3`, `qwen35` (hybrid DeltaNet + full-attn interval),
`qwen4exp` (Qwen3.8-Flash-Next: hybrid linear/full attention, MTP drafter,
NVFP4, hyper-connections, PLE), `gemma4` (31B dense + 26B-A4B MoE, fp8 KV
when pooled), `nemotron`, `gpt_oss`, `granite` (+ speech), `laguna`,
`whisper`, plus vision/OCR towers (`dinov3`, `deepseek_ocr`, `paddleocr_vl`,
`qwen3_asr`).

Phase 6 dense bring-up candidates in upstream's own catalog: Qwen 3.5 9B,
Qwen 3.8 27B, Gemma 4 31B, Nemotron. Excluded from bring-up per plan:
qwen35/qwen4exp (hybrid state), gemma4-26B-A4B (MoE), gpt_oss (MoE).
Recommendation recorded: **Qwen 3.5 9B** (smallest dense, standard attention
+ MLP, GGUF Q8_0 readily available) or **Qwen 3.8 27B** if a bigger model is
wanted for a meaningful TP=2 memory win. Decision deferred to user.

GPU-historical note: no GLM/Hy/MiniMax families exist in the engine. The
GLM-5.3-Flash / Hy3 / MiniMax-H3 artifacts on the KINGSTON drive are EXL3 /
vLLM-format and out of scope for Paddock TP work.

## 5. Async machinery inventory (to preserve, Phase 8)

- Decode-tick sub-DAG overlap on `side_stream` (gemma4 A4B) with
  greatest-priority main stream.
- Event-gated D2H metric reads on `copy_stream` (pipelined, never blocking
  the compute stream).
- Stream-agnostic graph replay (`launch_on`) used by the whisper admission
  graph to overlap decode.
- Pipelined decode + prefill/decode overlap live in the generator tick;
  speculative paths hook the same tick (`spec.rs`, ladder/adaptive policy in
  `spec_policy.rs`).
- CUDA graph capture is stream-bound; TP collectives issued between captured
  segments must respect capture boundaries (Phase 9 Stage A: capture local
  compute, leave NCCL uncaptured).

## 6. KV geometry, offload and expert-streaming map (plan requirement)

KV planning (`kv_plan.rs`) works in a two-level structure:

- `Demand` describes what a family needs: `block_bytes` (one pool block costs
  across EVERY layer that draws from the pool — one block id addresses all of
  them, so it is the whole-model cost), `per_slot_bytes` (the unshareable
  per-slot set: SWA rings, recurrent/DeltaNet state, conv windows, the slot's
  logits row, its block table), `reserves` (fixed charges incl. the graph/
  prefill scratch), floors for admission progress, and `retention_blocks`
  for the prefix radix tree.
- `Plan::plan(grant)` fits that demand into the device grant the KV allocator
  receives; families either enumerate reserves completely
  (`hedge_fraction: None` — qwen35 does since 2026-09-06, with a ledger audit
  after allocation) or take a hedge fraction (gpt_oss still does). The
  hedge comment records a measured 27B-Q4 incident: honestly-enumerated
  reserves budgeted an 11.8 GB pool, lazily-allocated spec state pushed past
  free, concurrency collapsed 74 → 31 t/s.
- `kv_pool.rs` is a refcounted block pool (`alloc`/`retain`/`release`/`cow`
  + `cow_at`), per-slot block tables via `blocks_for(window)`/`locate(pos)`,
  plus SWA ring windows. `paged_radix.rs` + `gpu_model/prefix_cache.rs`
  implement prefix caching on top of block reuse.

Per-family KV dtype posture (from `paddock.example.toml` + config parsing):
`kv_cache_dtype = "auto"` defaults per family — gemma4 uses fp8-e4m3 when
pooled, others f16; `"fp8_e4m3"` halves KV bytes (lossy). Qwen3.8-27B
bring-up runs f16 KV by default (the fork's validation used FP8 E4M3 KV —
note for parity comparisons: KV dtype changes numerics, so baseline and TP=2
parity runs must pin the same setting).

Offload interactions with device-memory ownership (Phase 11 surface):

- `kv_tier/` (host/NVME KV tiering): accounting, catalog, cost model,
  restore flow, RAM transport, NVME store; content-keyed by model; restores
  must win vs recomputation or the cache says so. Tiering moves whole
  tier-payloads of block data to/from the pool — under TP the payload for a
  block is only this rank's head-shard slice, so tier payloads must be
  rank-local too (payload.rs is the file to extend).
- `gpu/moe_cache.rs` (MoE expert streaming): experts live in page-locked
  host RAM (`HostMappedKq` gate/up/down planes), a device-side `ExpertCache`
  keeps hot experts resident (LRU) inside the decode graphs; slot bytes
  computed from gate/up/down plane sizes; `vram_bytes()` reports cache
  footprint. Under TP the rank-local rule applies per plane: each rank
  streams/maps only its expert-shard slice (Phase 13).
- `gpu/unified_mem.rs` exists as unified-memory helper surface (GB10
  coherent memory) — relevant to whether host-plane expert reads can avoid
  copies on Spark; investigate at Phase 13, not before.

## 7. Synchronization points / single-process assumptions found

- `Engine::spawn` builds one engine per process; the tick loop owns the CUDA
  context (cudarc context is thread-bound via `bind_to_thread`). A rank-1
  worker process duplicates this structure without the HTTP surface.
- KV plan computes a single grant from (total) device memory — a rank must
  plan against its LOCAL shard footprint (weights/KV heads), the exact
  Phase 6/11 requirement. Note the plan-side notion of "whole-model cost"
  (`block_bytes` spans all layers) is fine to keep if the rank's per-layer
  contributions are computed from local head/FFN shards.
- `graph_scratch_reserve_bytes()` is a global 3 GiB class reserve — same
  rank-local caveat.
- Slot/prefix-cache state (`paged_radix.rs`, `prefix_cache.rs`) is
  process-local today; under TP the logical block tables must mirror on both
  ranks while the KV payloads stay head-sharded.
- Kernel pack elections key on SM count (`sm_count_for_defaults` in the
  fork's split path; upstream elects kernels per device) — with two ranks on
  identical GB10s this is symmetric, but the election must not silently
  diverge if clocks/SM counts are read at different moments.
- Linear/attention calls are typed GEMM methods on `GpuExecutor`
  (`kq_gemm`, `q8_0_gemm_repacked[_x2]`, `f8t_gemm`, `f8r/f8d_gemm_mma_ks`,
  head_f8_gemm, warp-level GEMV chains) invoked at family forward sites —
  shard-local linears wrap HERE, not in the pack ABI (the pack ABI is one
  entry-point loader + typed tables in `paddock-kernels/src/abi.rs`).

## 8. Environment facts (head Spark)

- OS kernel 6.17.0-1029-nvidia, aarch64, CUDA toolkit 13.0 (V13.0.88),
  nvcc 13.0.88; release binaries were built with nvcc 13.3.73 (fatbins fine).
- GPU: NVIDIA GB10 (sm_121/sm_121a; pack build.sh treats 121 as an 'a'-feature
  target). Release binaries ship kernels linked in for sm_121/sm_121a —
  TP=1 baseline runs WITHOUT any source build.
- Kernel pack source build for local iteration: `packs/cuda/build.sh 121`
  (single-arch, fast); full validated release set is `--static 86,100,120`.
- `cargo check --workspace` passes clean at `249cf25` (25.9 s warm,
  30 s cold + pdfium fetch). `paddock-pdfium` needs
  `bash packs/pdfium/fetch.sh` once per checkout (staged static lib, arm64).
- cudarc 0.19.9 with optional `nccl` feature (02030 ABI). libnccl NOT
  installed on head yet.
- Fabric: head 192.168.100.10 / worker gx10-d28b 192.168.100.11 on the
  ConnectX-7 RoCE lane (enp1s0f0np0). Bulk/NCCL traffic MUST use .100.x.
- Model storage: /media/sime/KINGSTON/models (exfat, symlinked ~/models,
  1.3 TB free). The bring-up checkpoint
  `Qwen3.8-27B-UD-Q4_K_M.gguf` (15.3 GiB) is being downloaded there at pinned
  rev `4ca7207` (verify sha256 `322e194f...23482` before first serve).

## 9. Baseline procedure (to execute when the GPU frees up)

Use the source-built pack (building now for sm_121a) with the downloaded
checkpoint; pin KV dtype and speculation so TP=2 parity later compares
like-for-like:

    ./target/release/paddock-runner \
      --model /media/sime/KINGSTON/models/Qwen3.8-27B-UD-Q4_K_M.gguf \
      --device cuda --kernel-pack packs/cuda/build/pd-cuda-sm120.so \
      --max-ctx 16384 --kv-cache-dtype f16 --spec off

Then measure with the repo harness (in-process mode is the reference; it
discards warmup):

    cargo run -p paddock-bench --release -- \
      /media/sime/KINGSTON/models/Qwen3.8-27B-UD-Q4_K_M.gguf --device cuda \
      --pack packs/cuda/build/pd-cuda-sm121.so \
      --prompt-tokens 128 --decode-tokens 64
    # and concurrency variants via paddock-bench http mode against the runner

Record per the plan's Phase 1 list: single-request decode tok/s, multi-request
throughput, prefill tok/s, short and long context, GPU memory. Store results
in this directory as `baseline-tp1-qwen38-27b.md`.

Pending: GPU was in use by the GLM vLLM lane (108 GiB resident) — per user
decision we wait for it to free rather than stop it.

## 10. Open questions carried into Phase 2

- NCCL availability on both Sparks (install path, version matching cudarc's
  nccl-02030 ABI) — decide before Phase 3 (communicator).
- Whether the Communicator lives in `paddock-kernels` (C exports, capture-
  friendly) or in `paddock-engine` via cudarc NCCL (pure Rust). Default per
  this map: cudarc NCCL in `paddock-engine/gpu`, revisit if graph capture of
  collectives (Phase 10 Stage B) demands pack-level exports.
- ~~Dense bring-up model choice~~ — resolved: Qwen3.8-27B UD-Q4_K_M (see
  header).
