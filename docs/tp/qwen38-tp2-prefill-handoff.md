# TP=2 cold-prefill performance handoff

Branch: `review/qwen38-tp2-final`. Session start: `579b8fda88e41cdabbae91536742520d81346102` (clean). Prior session code HEAD: `bddbe88e1cb70f111f8c467398ad8edf67bb8356` (commits `b9628e5` cold checkpoint ownership/policy, `bddbe88` opt-in device-event profiler). This session's code HEAD: `069356c6b0b40efbead153552425a5f999af4923` plus the pending documentation commit; full commit list below. Pushed: NO.

## Completed changes

Phase 0 (`b9628e5`): cold `TpPrefixAdmit` now admits tokens/pages without DeltaNet checkpoint reservations. The first `TpMixed` tick that takes Mixed ownership reserves requested cuts exactly once before KV Ensures; protocol v5 carries `TpMixed.reserve_cuts`, which rank 1 validates and mirrors before Prepared. Resumed prompts remain Mixed-pinned and reserve at admission. Cold Async continues publishing KV pages only and does not consume/steal a checkpoint slot. `MirroredKv::snapshot()` replaces `authorize_all(&[])` as the snapshot getter. TP resume decisions now call the same `min_cache_prefix()` / `resume_live_max()` policy as non-TP, including environment semantics. Host tests cover existing-checkpoint survival through cold Async admission/publication, once-only Mixed reservation and mirror, and zero capacity.

Phase 1 (`bddbe88`): `PADDOCK_TP_PREFILL_PROFILE=1` records timing-enabled CUDA events on compute and NCCL streams for each span, with per-rank cumulative spans/rows, actual capacity-plane f32 bytes and collective counts. It is off by default and adds no events/sync on the default path. Profiling synchronizes once per span when enabled and therefore is a diagnostic breakdown, not comparable uninstrumented request throughput. The launcher forwards the opt-in gate to rank 1. Whole-advance excludes final head/sampling; `*-wait` measures the compute→NCCL→compute fence envelope and overlaps NCCL time (do not add both). GQA metadata records device upload time, not host allocation/table construction.

## Pre-optimization baseline (actual two-node run)

Qwen3.8-27B Q4_K_M, 2× DGX Spark, `MAX_CTX=65536 SPEC=off TP_GRAPH=0 PADDOCK_UNIFIED=1 KV_DTYPE=f16 MAX_BATCH=2`, cap 64, 22,130-token Hermes-style fixture, fresh cold prompt after model warm-up. HTTP 200; `cached_tokens=0`; 64 completion tokens, `finish=length`; profiled request wall 115.673 s. Rank-0 GPU event sum for span advances: 105.252 s. Rank 1 matched 348 spans and 22,130 rows (105.269 s whole advance). Per-rank measured rank-0 components:

| category | seconds | share of whole advance |
|---|---:|---:|
| GQA local compute | 9.625 | 9.14% |
| GQA NCCL | 1.241 | 1.18% |
| DeltaNet local compute | 30.210 | 28.70% |
| DeltaNet NCCL | 4.016 | 3.82% |
| FFN local compute | 53.919 | 51.23% |
| FFN NCCL | 5.024 | 4.77% |
| GQA metadata GPU upload | 0.022 | 0.02% |

NCCL-stream all-reduce time totals 10.281 s (9.77%). There were 348 spans, 22,130 rows (63.59 rows/span), and 44,544 all-reduces (GQA 5,568; DeltaNet 16,704; FFN 22,272); actual capacity-plane NCCL input is 58,384,711,680 bytes (54.375 GiB) per rank. Other time includes embedding, layer norms/residuals, fences and measurement overhead. Rank 1 measured 10.420 s NCCL. Prior accepted unprofiled baseline was 106.089 s derived prefill, ~208.6 prompt tok/s and 115.481 s wall. TP=1 wall was 32.280 s; neither TP=1 nor an unprofiled post-change production comparison was rerun.

An identical request on the same server returned HTTP 200 with `cached_tokens=22128`, `cache_write_tokens=2`, 10.323 s wall, `finish=length`, and byte-identical generated text. Both rank exits were 0 after normal signal shutdown; the rank logs had no NCCL, KV/page, CUDA, poison, protocol or panic errors. Evidence: `/home/sime/.hermes/cache/scratch/tp-prefill-profile64-{launcher,worker}.log` and `/home/sime/.hermes/cache/scratch/tp-prefill-profile64-{cold,hit}-result.json`. Runner hash on both nodes: `0b0b2b76a2f68dae51db74a68d1e892485789d74cf94046bc581c70fbf9b21af`.

## Bottleneck and circuit breakers

The exact 64-row cap and 348 whole-model traversals are confirmed. The claim that NCCL dominates is weakened by the measured 9.77% share. The proposed speedup from crossing the 65-row big-prefill dispatch threshold remains unresolved: the optimized rung and TP-sharded matrix compatibility have not yet been traced on GPU, and wider DeltaNet recurrence has not been validated. Do not change a constant alone, choose 192 as a default without a sweep, or tune NCCL environment before local compute/kernel dispatch. Preserve exact checkpoint-cut splitting, token-causal recurrent state and convolution, rank-symmetric collective order, four-axis text M-RoPE, and mirrored cache lifecycle.

## Code map for the next worker

- Span cap and staging: `gpu_model/qwen35/tp_span.rs:24,104-160` (`TP_SPAN_CAP`, `TpSpanPlanes::new`, `SpanGemmStaging`); `tp_model.rs:1078-1145,1215-1377` (`forward_span_advance`, `span_layer_walk`, `ensure_span_planes`); `tp_serve.rs:199-230,1400-1455,3110-3190` (`span_chunk_points`, `span_checkpoint_points`, Mixed/worker traversal). Update host geometry tests in `tp_serve.rs` around 64/65, 127/128/129, 191/192/193, page/cut boundaries and nonzero resume.
- DeltaNet capacity, scratch, recurrent ordering: `delta_tp.rs:26,226-300,700-850` (`SPAN_CAP`, scratch allocation, `prefill_slot`, `forward_profiled`, `prefill_run`, `finish_partial_prefill`); inspect pack kernel row limits before widening. This is a correctness gate, not a shape-only patch.
- Big-batch quant/GEMM staging: `ops.rs:2025,2342-2414,2558-2632` (`prefill_quant`, `prefill_mm_pre_any`, `prefill_mm_any`); `tp_span.rs:104-160` (dummy `yq`, `xsums`, `skfix` buffers under cap 64). Trace actual dispatch for sharded weights at rows >64.
- Q/K/V repeated activation quantization: `gqa_tp.rs:640-711,730-785` (`forward_paged_span`, `span_run`, three `prefill_mm_any` calls on `xn`); use existing split quant + pre-quant projection after validation. FFN gate/up: `ffn_tp.rs:170-220` (`forward_rows_capacity`, two projections on `xn`). Check DeltaNet projections for analogous reuse only after parity.
- GQA metadata: `gqa_tp.rs:665-711` builds and uploads the same block table/positions/slots/four-axis M-RoPE payload per GQA layer; stage once per whole-model span in `tp_model.rs:1078-1145`, keep checked `MirroredKv::checked_device_table` semantics and every consumer on the span-owned table.
- Live-row collectives: `gqa_tp.rs:928-943`, `ffn_tp.rs:233-247`, `delta_tp.rs:780-850` zero/submit capacity-sized f32 planes; `gpu/distributed.rs:81-105,158-182` (`Communicator` shape checks, fences and NCCL all-reduce). Ensure only live prefixes are reduced with compatible non-copying views and no stale suffix. Profile actual bytes again.

## Audit findings (this session, from code; supersedes the pending-audit note)

Traced every span-dependent allocation, kernel row limit and collective shape
at rows 64/128/192. Findings, each verified against the cited code:

1. **DeltaNet has no kernel row limit at 64.** `gated_delta_recurrent_v2`
   (packs/cuda/src/gemm/int8_mma.cuh:2332) loops `t` sequentially inside one
   (head, batch) block for any `n_tokens`; the non-TP path already feeds it
   multi-hundred-row chunks. `causal_conv1d_silu`, `deltanet_split_gqa_norm`,
   `deltanet_alpha_beta_gate` (per-row loop), `gated_rmsnorm` are all
   n_tokens-parameterized. The TP DeltaNet `SPAN_CAP = 64`
   (delta_tp.rs:26) is Rust-side allocation + validation only. Widening is
   allocation + cap threading; recurrence, conv window and gate math are
   span-length independent (the batched conv builds `[window ++ rows]`, takes
   the last k-1 rows as the new window — identical for any rows).
2. **Span staging placeholders are the real >64 blocker.** `SpanGemmStaging`
   and DeltaNet `PrefillGemm` allocate `yq`/`xsums`/`skfix` as one-element
   stubs (tp_span.rs:148-151, delta_tp.rs:228-231). Above 64 rows the k-quant
   W4A8 tile arm requires the flat mmq layout (`quantize_q8_mmq`:
   `ceil(in/128) * pad128(batch) * 144` bytes) and `mmq_sums` sums
   (`ceil(in/128) * pad128(batch) * 4` f32). Required sizing mirrors
   forward.rs:2031-2048. DeltaNet's `skfix` stub is SAFE to keep: every
   DeltaNet projection is forced k-quant at load (delta_tp.rs:405), and
   `kq_mm_pre` never touches skfix (Q8_0-only stream-K scratch); the shared
   GQA/FFN staging must size skfix for the Q8_0 mmq ladder
   (`256*128*128 + 256` f32) in case any projection is Q8_0.
3. **Kernel dispatch at 64/128/192 for TP-sharded Q4_K_M (static):** all TP
   projections are `QuantW::Kq` (Q4_K). `prefill_mm_any` → `prefill_quant`:
   batch <= 64 → strided `quantize_q8` (xq/xs); batch > 64 → flat mmq
   `quantize_q8_mmq` (yq only; no dense i-quant in Q4_K_M so no xq re-quant).
   `kq_mm_pre`: batch <= 64 → `q8_sums_strided` + `kquant_gemm_dp4a`; batch
   > 64 → `mmq_sums` + `kquant_gemm_w4a8_pipe2`/`_pipe`/`_w4a8` tile
   (pack-gated). The mcol mma rung (`prefill_mm_pre_p`, 65..=192) is NOT
   wired into the TP path — `prefill_mm_any` calls `prefill_mm_pre`, so TP
   takes the base mmq/W4A8-tile rungs. Widening engages those; mcol stays a
   non-TP bonus. Attention: `prefill_attn` r>24 paged f16 WMMA already
   engaged at 64 rows and is r-parameterized (non-TP runs 8192-row chunks
   through it). M-RoPE, norms, split_qg, kv_append_batch_paged: batch-
   parameterized, no row limit.
4. **Checkpoint-cut semantics are span-cap independent.**
   `span_checkpoint_points` (tp_serve.rs:217) splits each run at BOTH the row
   cap and every reserved cut, so a wider cap still lands each span exactly
   on a cut before snapshotting; snapshot/restore copy the live (recurrent,
   conv) pair whose advance is per-token. Multiple cuts inside one wide span
   force splits at each cut. Only host tests needed extending.
5. **Collectives are capacity-sized with zeroed suffixes**
   (gqa_tp.rs:933, ffn_tp.rs:236, delta_tp.rs:808/824). cudarc 0.19
   `CudaView`/`CudaViewMut` implement `DevicePtr`/`DevicePtrMut`, so
   all-reduces can ride contiguous live-row views without copies; both ranks
   derive identical `rows` from the same wire rows, so NCCL counts pair.
   Consumers (`exec.add`, head GEMV staging, traces) read only the live
   prefix, so a stale `reduced` suffix is never consumed.
6. **Repeated activation quantization:** GQA quantizes the same normalized
   `xn` three times (Q/K/V), FFN gate/up twice, DeltaNet in_qkv/gate twice.
   One `prefill_quant` + N `prefill_mm_pre` consumers is exact for the same
   input/in_dim; the k-quant min-term sums can likewise be computed once per
   staged input (identical yq → identical sums).
7. **GQA span metadata** (block table, positions, slots, 4-axis M-RoPE
   plane) is identical for every GQA layer of one span and is currently
   rebuilt + uploaded per layer (gqa_tp.rs:666-711). It can be staged once
   per whole-model span in `forward_span_advance` with identical
   `checked_device_table` validation.
8. **Rank symmetry of the cap:** the chunkers are pure functions of
   (len, cap); today both ranks share the compiled constant. A runtime cap
   must ride `TpInit` (the established rank-0-authoritative pattern:
   kv_dtype/use_graphs/ckpt_slots) so a hand-started worker cannot disagree;
   protocol bumps to v6.
9. **Graph capture is unaffected:** `TP_GRAPH` captures decode-lane
   rank-local runs only; the span path and the prefill lane are never
   captured, so widening cannot bake stale geometry into a graph.
10. **Memory cost of widening:** shared staging set grows by ~25 MB per
    rank per lane; DeltaNet per-layer staging grows ~2.3 MB per layer
    (~165 MB per rank across both lanes at cap 192, ~36 DeltaNet layers).
    Small against the KV pool and model weights; still worth logging.

Unresolved without GPU execution (documented, not guessed): actual
correctness/throughput of the >64 W4A8 tile + mmq rungs on TP-sharded weight
geometry (the kernels see shard-local, block-aligned dims — statically sound,
GPU-unproven), and whether any of 128/192 is faster than 64. No cap is
promoted by this session; production default stays 64.

## Session implementation record (HEAD `069356c` and the doc commit)

Starting code HEAD `bddbe88` (clean except this doc). Commits made, in order:

1. `2deef1b` — `tp: parameterize wider prefill span geometry`. One
   authoritative cap (`tp_span_cap::span_cap()`), resolved rank-0-side from
   `PADDOCK_TP_SPAN_CAP` over {64,128,192} and shipped in `TpInit`
   (protocol v5 -> v6, mandatory field, missing-field refusal tested); rank 1
   installs the wire value and never reads the env. Unsupported values fail
   closed on both ranks; 64 stays the default. Real flat-mmq staging for the
   >64 band in both `SpanGemmStaging` and DeltaNet `PrefillGemm`
   (`mmq_layout` = `ceil(in/128) * pad128(rows)`; yq 144 B/row-block,
   xsums 4 f32/row-block), plus the Q8_0 stream-K `skfix` scratch the shared
   set lacked (DeltaNet keeps a 1-elem skfix: all its projections are forced
   k-quant). DeltaNet span scratch and validation read the same resolution.
   The serve chunkers split at pure `span_chunk_points_at` /
   `span_checkpoint_points_at` so the geometry is host-testable at every
   sweep width; production wrappers read the resolved cap. No hot-path
   allocation: planes size once at first span.
2. `8ac7731` — `tp: reuse prefill activation quantization`. GQA Q/K/V (3x),
   FFN gate/up (2x) and DeltaNet in_qkv/gate (2x) each quantized identical
   bytes per span per layer; each site now runs one `prefill_quant` +
   per-weight `prefill_mm_pre_any`. Bit-identical staging bytes and stream
   order; the GEMM half reads xq/xs/yq by shared ref and only xsums/ssums/
   skfix mutably, so no scratch is consumed early.
3. `23e8086` — `tp: stage span metadata once per whole-model span`. The
   validated mirrored block table, per-row positions, slot ids and the
   [4, rows] M-RoPE plane upload once per span (`SpanGqa.staged` key
   (slot, position, rows)); `checked_device_table` semantics unchanged and
   still executed identically on both ranks.
4. `076792b` — `tp: reduce live-row collective views`. GQA/FFN/DeltaNet
   prefill all-reduce the contiguous live `rows*hidden` view (cudarc slice
   views, no copies) instead of the zeroed capacity plane; the per-layer
   suffix memsets are gone. Decode keeps the capacity collective the captured
   graphs bake. Both ranks derive identical `rows`, counts pair, order
   unchanged, suffix never consumed.
5. `069356c` — `tp: opt-in prefill kernel dispatch witness`.
   `PADDOCK_TP_KERNEL_TRACE=1` prints one deduped line per
   (rows, weight dims, rung: q8 | kq-tile | kq-dp4a) through
   `prefill_mm_pre_any`, proving in a serving log which prefill kernel family
   a span fired at 64/128/192.

## Host/static validation (this session)

- `cargo test -p paddock-engine --lib`: 518 passed (3 runs; new tests: span
  cap parser, wire install fail-closed, chunker/checkpoint geometry at
  63/64/65, 127/128/129, 191/192/193 for every sweep cap, cuts at cap
  boundaries and multiple cuts inside one wide span, TpInit span_cap
  roundtrip + missing-field refusal, fp16/fp8 dtype tests unchanged).
- `cargo test -p paddock-dist`: 6 unit + 22 bootstrap passed (protocol v6).
- `cargo test -p paddock-engine --test tp_wire_frame`: 6 passed.
- `cargo clippy -p paddock-engine -p paddock-dist --all-targets`: only the
  pre-existing `cuda.rs` u8 cast and two example warnings; no new warnings.
- `git diff --check`: clean. `cargo build --release -p paddock-runner`:
  passed; runner sha256
  `a236a4058580c4147e0bf684a1baa8247210cc53d56471e8d140006a3ecfcf09`.
- NOT run (GPU-touching, GLM serving is active on this host): any
  `--all-targets` engine test binary, GPU parity/oracle examples, and of
  course the two-node sweep. Nothing here measured performance.

## GPU validation package for the next session

Env (both ranks via the launcher; the cap rides TpInit so only the
coordinator's env matters for it, but keep NCCL vars as before):

- `MAX_CTX=65536 SPEC=off TP_GRAPH=0 PADDOCK_UNIFIED=1 KV_DTYPE=f16 MAX_BATCH=2`
- `PADDOCK_TP_CKPT_SLOTS=4`
- `PADDOCK_TP_SPAN_CAP=64` then `128` then `192` (only these values are
  accepted; anything else fails both ranks at init with a clear message)
- optional `PADDOCK_TP_PREFILL_PROFILE=1` (diagnostic only), optional
  `PADDOCK_TP_KERNEL_TRACE=1` (dispatch proof)

Sweep procedure (isolated, idle cluster, per cap value; restart the server
between caps — the cap is fixed at init):

1. Stage the runner built at `069356c` + doc commit on BOTH nodes (launcher
   refuses a hash mismatch): `sha256sum target/release/paddock-runner` on
   rank 0 must equal the staged remote runner's hash. Record both plus the
   CUDA-pack and model hashes (launcher prints/checks them).
2. Isolated repeated-prefill probe (parity + rows/span proof, one server per
   cap): send the 22,130-token Hermes fixture
   (`/home/sime/.hermes/cache/scratch/qwen38-hermes-20k-prompt.txt`)
   3x cold (no cache, `PADDOCK_TP_CKPT_SLOTS=0` for the pure-cold leg, then
   capacity 4 for the hit leg). Per request record: HTTP status, prompt
   tokens, cached, wall, `[TP-KERNEL-DISPATCH]` lines (both ranks), the
   `[TP-PREFILL-PROFILE]` rank0/rank1 summaries (spans, rows,
   reductions/bytes per kind, stage_ms), and rank exits.
3. Expected geometry: 22130 rows -> cap 64: 346 spans (63.59 rows/span
   measured before; ~348 reported with head spans), cap 128: 173 spans,
   cap 192: 116 spans (last span short: 22130 = 115*192 + 50 -> 115 full +
   1 x 50 rows). GQA/DeltaNet/FFN reduction counts scale 1/2 and 1/3 vs the
   44,544 all-reduces at cap 64.
4. Expected dispatch markers (cap 128/192, Q4_K_M, sm_120 Spark):
   `rung=kq-tile` for every projection family (the W4A8 >64 rung; no
   `kq-dp4a` lines for the 128/192-row spans), plus the attention WMMA class
   already active at 64. Any `kq-dp4a` line at rows>64 means a pack capability
   gate refused the tile - STOP and investigate before comparing timings.
5. Correctness gates per cap (before ANY timing comparison): HTTP 200,
   coherent generation, first generated token IDENTICAL to the accepted
   cap-64 baseline text (documented in the prefix-cache validation record),
   byte-identical short completion at temperature 0 seed 1, clean rank0/rank1
   shutdown, no CUDA/NCCL/KV/page/refcount/poison/cursor/checkpoint errors in
   either rank log, exact checkpoint cuts (cap 192: cuts 22112/22128 inside
   one wide span MUST split at both), cache-hit regression (identical
   request byte-identical output), divergent-tail resume, release/reuse/
   cancel, zero-checkpoint-capacity cold fallback.
6. Only after all gates pass at a width: fresh-server cold 22,130-token run
   WITHOUT the profiler for the headline number. Compare against the
   accepted baseline (derived prefill 106.089 s, wall 115.481 s, ~208.6
   prompt tok/s; TP1 wall 32.280 s). Do NOT promote a default cap from the
   sweep alone; report all three and let Simon choose.
7. Stop conditions: any first-token divergence, any NCCL/shape mismatch, any
   checkpoint-cut mismatch, any `kq-dp4a` fallback at rows>64, or gibberish
   output (the 2026-09-06 stale-yq signature) - stop that cap, record the
   logs, do not proceed to the next cap until explained.

Log evidence to keep: both ranks' full logs per cap, per-request JSON
results, the profile summaries, and the hash quadruple (runner, pack, model,
per-rank).
