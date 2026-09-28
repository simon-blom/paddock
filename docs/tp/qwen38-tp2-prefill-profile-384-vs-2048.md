# Qwen3.8 TP=2 cold prefill profile: span caps 384 and 2048

Starting HEAD for this task: `cc2fb0d95e33e4593792b96d44c1ae643c65c188`. Companion graph correctness/performance report: `qwen38-tp2-graph-decode-live-row.md`. Historical sweep and its decode-artifact correction remain in `qwen38-span-cap-sweep.md` and `qwen38-tp2-eager-decode-live-row.md`. This is diagnosis only: no prefill algorithm, threshold, kernel, or production cap was changed. The production `PADDOCK_TP_SPAN_CAP` default remains 64; no 4096/8192 run was made.

## Matched workload and accounting boundary

Two DGX Sparks, TP=2 Qwen3.8-27B-UD-Q4_K_M, F16 KV, context 65,536, max batch 2, unified mode, speculation off, `TP_GRAPH=0`; identical `/home/sime/.hermes/cache/scratch/qwen38-hermes-20k-prompt.txt`, `enable_thinking=false`, temperature 0, seed 1, max output 64. Fresh server per cap. The release runner SHA-256 `10258e62dee7f732f2307d49770fd0e9325d270e520b1285adf25e9b26b5a4b7` and CUDA pack SHA-256 `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e` matched on both ranks. `PADDOCK_TP_PREFILL_PROFILE=1` on both ranks; `PADDOCK_TP_KERNEL_TRACE=1` used in separate dispatch-witness runs. Six existing mixed/prefill tick-stall warnings per headline request; no CUDA, NCCL, or protocol errors. Each request returned HTTP 200, 22,130 prompt/64 completion tokens, and the established output SHA-256 `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84`.

The profiler creates timing-enabled events on compute and NCCL streams and synchronizes at every span end. For each rank independently, this report subtracts the cumulative `total_stage_ms`, `total_spans`, `total_rows`, reductions and bytes after the four startup/warmup spans (224, 16, 16, 40) from the last record of the cold request. The request begins at `TP prefix admit ... tokens=22130` and ends before the first following `TP decode pipe began`; only the 22,130 prompt rows are counted. Do not sum successive cumulative records, and do not treat `[TP-PREFILL-PROFILE-LAYERS]` (first warmup span only) as full-prompt data. All tables below are rank 0 unless stated otherwise. Rank 1 reproduced exact rows, windows, reduction counts and bytes; its `whole-advance` was 21.441 s / 21.241 s versus rank 0's 21.435 s / 21.238 s.

## Geometry, counts and traffic

| cap | Mixed scheduler ticks and chunk rows | Model span windows | Window row distribution | Full-width | Final partials | Model rows |
|---:|---|---:|---|---:|---|---:|
| 384 | 3: 8192 + 8192 + 5746 | 61 | 56×384, 2×128, 1×352, 1×16, 1×2 | 56 | 352, 16, 2 after the second 128 cut | 22,130 |
| 2048 | 3: 8192 + 8192 + 5746 | 13 | 10×2048, 1×1632, 1×16, 1×2 | 10 | 1632, 16, 2 | 22,130 |

Checkpoint cuts and the finishing tail can split chunks independently of cap. Per-rank reductions over the request: GQA 976 vs 208, DeltaNet 2,928 vs 624, FFN 3,904 vs 832 (16/48/64 per span). Exact per-rank f32 collective input bytes were unchanged across caps: GQA 7,251,558,400; DeltaNet 21,754,675,200; FFN 29,006,233,600. Thus 61/13 = 4.69× fewer model spans and collective calls, but the same 22,130 rows and exactly the same summed payload bytes. The scheduler still emitted three mixed ticks at either cap; 8,192 is a scheduler chunk size, not the GPU model span width.

## Full-prompt device-event accounting

Times are cumulative seconds. `ms/token` divides cumulative time by 22,130; `ms/span` divides by 61 or 13. Ratios are cap 2048 / cap 384. NCCL duration is the communication-stream event interval; wait is the compute-stream fence envelope containing it. Do **not** add NCCL and wait when reconciling `whole-advance`.

| Stage | 384 s | 2048 s | delta s | ratio | ms/token 384→2048 | ms/span 384→2048 |
|---|---:|---:|---:|---:|---:|---:|
| GQA local + metadata | 2.929 | 2.871 | -0.059 | 0.98 | .1324→.1297 | 48.0→220.8 |
| DeltaNet local (all internal substages) | 5.064 | 4.863 | -0.201 | 0.96 | .2288→.2198 | 83.0→374.1 |
| FFN local | 7.464 | 7.119 | -0.345 | 0.95 | .3373→.3217 | 122.4→547.6 |
| NCCL on comm stream, GQA | .629 | .663 | +.034 | 1.05 | .0284→.0300 | 10.3→51.0 |
| NCCL on comm stream, DeltaNet | 1.815 | 1.920 | +.104 | 1.06 | .0820→.0867 | 29.8→147.7 |
| NCCL on comm stream, FFN | 2.598 | 2.566 | -.031 | .99 | .1174→.1160 | 42.6→197.4 |
| NCCL subtotal (overlaps wait) | 5.042 | 5.149 | +.107 | 1.02 | .2278→.2327 | 82.7→396.1 |
| Compute-stream wait envelope (includes NCCL) | 5.095 | 5.162 | +.067 | 1.01 | .2302→.2332 | 83.5→397.1 |
| Unclassified inside `whole-advance` | .882 | 1.224 | +.341 | 1.39 | .0399→.0553 | 14.5→94.1 |
| **Whole model span-advance** | **21.435** | **21.238** | **-.197** | **.99** | **.9686→.9597** | **351.4→1633.7** |

The reconciled non-overlapping subtotal is GQA local + DeltaNet local + FFN local + compute-stream wait + unclassified: 21.435/21.238 s. The residual includes inter-layer glue/norms/adds and event-boundary effects that are not individually labeled; it is **not** a measured scheduler or NCCL kernel bucket. The admission→decode coordinator timestamp was 22.208/21.974 s in these profiled runs, leaving .773/.735 s outside the summed model span-advance events for scheduler, finishers, and other CPU/GPU work; do not conflate that with the within-span residual. Both ranks' stage sums agree closely, although individual stage intervals vary. The unprofiled accepted admission→decode baselines were 21.959/21.921 s; profiler/event overhead prevents using these different timing classes as a speedup comparison.

## DeltaNet internal full-prompt breakdown

These labels describe the **current** batched alpha/beta implementation. All times are seconds, cumulative across 48 DeltaNet layers and all cold prompt spans. `ms/token` and average `ms/span` use the same denominators as above; NCCL is separate from local and overlaps the wait envelope.

| Substage | 384 s | 2048 s | delta s | ratio | ms/token 384→2048 | ms/span 384→2048 |
|---|---:|---:|---:|---:|---:|---:|
| input activation quantization | .066 | .125 | +.059 | 1.90 | .0030→.0056 | 1.1→9.6 |
| input QKV projection | 1.093 | .966 | -.127 | .88 | .0494→.0437 | 17.9→74.3 |
| gate projection | .735 | .574 | -.161 | .78 | .0332→.0259 | 12.0→44.2 |
| alpha/beta batched + gate preparation | .654 | .595 | -.059 | .91 | .0296→.0269 | 10.7→45.8 |
| conv/split/gate preparation prelude | .003 | .001 | -.002 | .23 | .0001→.0000 | ~0→~0 |
| convolution/window staging + split/norm | .648 | .835 | +.188 | 1.29 | .0293→.0377 | 10.6→64.3 |
| recurrent kernel | .965 | .900 | -.066 | .93 | .0436→.0406 | 15.8→69.2 |
| norm/gate elementwise | .128 | .172 | +.045 | 1.35 | .0058→.0078 | 2.1→13.2 |
| output projection | .770 | .695 | -.075 | .90 | .0348→.0314 | 12.6→53.5 |
| other/prelude | .003 | .001 | -.002 | .22 | .0001→.0000 | ~0→~0 |
| **DeltaNet local total** | **5.064** | **4.863** | **-.201** | **.96** | **.2288→.2198** | **83.0→374.1** |
| DeltaNet NCCL (comm stream, outside local) | 1.815 | 1.920 | +.104 | 1.06 | .0820→.0867 | 29.8→147.7 |

The largest adverse DeltaNet substage is conv/split preparation (+.188 s), not alpha/beta. Even eliminating its entire cap-2048 regression would not materially change the ~22-second floor. `delta-conv-split-prep` currently includes state-window copies, causal convolution, copies back, and QKV split/norm; it is not a pure convolution-kernel timer. No inference that the convolution kernel alone regressed is warranted.

## Dispatch and explanation

Live `PADDOCK_TP_KERNEL_TRACE=1` witnesses on rank 0 showed both 384 and 2048 full-width spans taking `kq-tile` for the 5120×5120, 5120×3072, 5120×8704, 5120×6144 and 5120×512 projection geometries, and `q8` for another 5120×512 geometry. It records the weight family/tile rung, not the exact inner CUDA specialization. Sub-65-row tails use the narrow Q8/k-quant arms; cap 384 also had 128-row partials, and 2048 a 1632-row partial. Q8's `prefill_mm_pre_sk` threshold defaults to `batch > 1024` for the hi/pipe path unless `PADDOCK_MMQ_HI_MIN` or a family election overrides it (`mod.rs:2599-2635`, `ops.rs:2198-2222`); the pack exports pipe and hi entry points. Thus Q8 full-width 384 and 2048 are eligible for different Q8 GEMM rungs, while both use the `batch > 64` flat activation layout. The trace does not record the selected inner Q8 rung or runtime capability/override, so its exact selection is not claimed as empirically proven. K-quant >64 tile with optional pipe2/pipe has no 384→2048 threshold in the shown dispatch (`ops.rs:2298-2319`); the same broad family was observed. The FFN down path uses >64 SwiGLU+quantization fusion (`ops.rs:2426-2441`); both full widths qualify.

DeltaNet TP multi-row prefill explicitly uses the batched alpha/beta pair projection and `gated_delta_recurrent_v2` for `rows` tokens, without a 384→2048 dispatch branch (`delta_tp.rs:1025-1054,1145-1162`). Its CUDA v2 kernel iterates `for t < n_tokens` inside each state-column block (`packs/cuda/src/gemm/int8_mma.cuh:2332-2408`): fewer host spans do not reduce its token steps. The local CUDA API also exposes `gated_delta_chunked` with 64-row chunks and a different accumulation order (`gpu/deltanet.rs:1279-1347`), but this experiment did not call it; changing numerical class/state handling would require independent parity and is not this task's optimization. Activation quantization (>64 flat mmq) and conv/split local work become moderately less efficient at 2048 in the measured aggregate, partially offsetting better QKV/gate/output projections and lower FFN local time. No large adverse dispatch crossover explains the flat total.

An attempted read-only inspection of the external `truespar/paddock` upstream source was denied, and it was not retried through another route. Consequently no claim is made about its exact 8192-row kernel mechanism or whether its large-row path is reusable here. The upstream *principle supplied by the user*—prefill capacity separate from live decode width—is already reflected in the graph/eager decode fix, but an upstream algorithm comparison remains an evidence gap.

The ~22-second floor is accounted for predominantly by work proportional to **the same 22,130 rows**: FFN local 7.1–7.5 s, DeltaNet local 4.9–5.1 s, GQA local ~2.9 s, and compute-stream collective/wait ~5.1 s. Although the number of model spans falls by 48, the collected bytes do not fall and the local row work does not vanish. The measured local savings (~.605 s) are largely offset by a ~.067 s larger collective/wait envelope and ~.341 s larger unclassified span time; `whole-advance` improves only .197 s. Neither scheduler chunk count nor total payload shrinks with span cap. The data support a mostly token-linear throughput floor already reached by 384, with some wider-row inefficiency in subcomponents, rather than a hidden prefill regression like the old decode artifact.

## Next optimization target (not implemented)

First isolate **FFN local** (7.464/7.119 s, the largest single exclusive bucket): gate/up projection, SwiGLU+quantize, and down GEMM need separate CUDA-event times at the existing stage boundary before choosing a kernel change. If one K-quant W4A8 tile dominates, improve its arithmetic/memory throughput or staging at the measured row geometries, not merely increase span cap. Current code already has >64 tile/pipe2 and fused SwiGLU quantization, so an upstream implementation cannot be recommended without an authorized comparison. Correctness risk: quantized activation layout/rounding and shard reduction parity; scope: bounded FFN substage profile, representative-shape kernel benchmark, then two-rank output/state validation.

Second candidate: collective work (~5.0–5.2 s, 54.0 GiB per rank over three mixer/FFN families); changing traffic would require communication/fusion or algorithmic redesign, with high rank-order and graph/prefill correctness risk. Third candidate: DeltaNet local (~4.9–5.1 s); chunked recurrence exists locally but changes arithmetic and requires strict state and resume parity, especially with prefix-cache restoration still open. Neither was changed here. Prefix-cache restoration parity remains unproved.

Evidence: `/home/sime/.hermes/cache/scratch/decode-live-row-20260928-080728/` (paired full profiler logs on both ranks, request JSON, status); `/home/sime/.hermes/cache/scratch/decode-live-row-20260928-081234/` (separate dispatch-witness runs). All profiled and trace cold-run launchers deliberately returned 143 after client completion, not a clean exit claim; the separate graph two-request smoke in the companion report did confirm both ranks exited 0.
