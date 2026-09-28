# Qwen3.8 TP=2 eager DeltaNet decode: live-row reduction

Starting HEAD: `fe40477487712926f764185235852c9b357c68a0` (`review/qwen38-tp2-final`). This report supplements, rather than replaces, `qwen38-span-cap-sweep.md`. Production span cap remains 64.

## Contract and root cause

`DeltaTpRank` owns one persistent `Span` per rank and layer. Its row-major f32 `partial` and `reduced` planes each hold `span_cap * WIDTH` elements (`WIDTH = 5120`); the persistent recurrent and convolution state belongs to each rank and is swapped per slot. Prefill projects exactly `rows * WIDTH` elements and reduces that live prefix, leaving the capacity suffix untouched. Eager decode (`tp_model::forward_token_body` -> `decode_slot` -> `forward_profiled` -> `decode_run` -> `finish_partial`) passes `rows = 1`, producing row 0 with the out-projection GEMV. Previously `finish_partial` zeroed the full partial plane, and the subsequent NCCL sum transferred the full `partial`/`reduced` planes. Only row 0 of the returned reduced buffer is consumed by the residual add (`hidden = WIDTH`). The suffix is not used for decode. Each new token overwrites row 0 and reuses the same rank-local backing allocation; no token-time allocation or host synchronization is needed.

The change zeroes only `rows * WIDTH` partial elements for eager decode and passes contiguous prefix views of both planes to `all_reduce` (also to the optional profiled collective). The NCCL wrapper checks equal view lengths and enqueues on its own stream. The existing compute->NCCL and NCCL->compute CUDA event fences still bracket the collective. The full capacity allocation remains stable for prefill and for graph capture. The graph path calls `finish_partial(..., graph_capture=true)` while recording the mixer run and calls `finish` after replay: its full-plane memset, captured pointer, host-side collective count and semantics are **unchanged**. Graph decode is not optimized by this change and requires separate validation/optimization if desired. Eager and graph still share the per-layer span backing buffers, but model dispatch selects one path; no simultaneous use is introduced. Other TP mixers: GQA and decode FFN allocate one-width partial/reduced pairs already; the batched prefill FFN/GQA span adapters reduce only live rows. No analogous capacity-sized eager one-row collective was found there.

| cap | Before: f32 elements / per-rank all-reduce payload | After eager: elements / payload |
|---:|---:|---:|
| 384 | 1,966,080 / 7.5 MiB | 5,120 / 20 KiB |
| 2048 | 10,485,760 / 40 MiB | 5,120 / 20 KiB |

The old cap difference was 32.5 MiB per DeltaNet reduction, or 97.5 GiB per rank over 48 layers and 64 decode steps, plus full-plane memset. This is a byte-count estimate, not a measured NCCL-time attribution. The implementation uses `rows * WIDTH`, not a hard-coded one-row collective; the current eager `decode_slot`/`decode` callers structurally supply exactly one row. This follows the live-decode-width principle without changing the decode dispatcher.

## Validation

- Host: `cargo check -p paddock-engine --lib`, `cargo test -p paddock-engine --lib` (518 passed), `cargo test -p paddock-dist` (6 unit + 22 bootstrap passed), `cargo test -p paddock-engine --test gpu_delta_net_parity` (6 passed; optional heavy speed test skipped), release runner build and release NCCL example build all passed. `cargo clippy --release -p paddock-engine -p paddock-dist --lib` passed with one pre-existing `unnecessary_cast` warning in `src/cuda.rs:83`; with `-D warnings`, that warning fails the command. `git diff --check` passed.
- Two-rank NCCL old/full-versus-new/live-prefix probe in `examples/nccl_bench.rs`: identical deterministic rank-local f32 row with zero tail; at caps 384 and 2048 both ranks' row-0 outputs were bitwise equal (0 differing f32 bits). Both rank processes exited 0. Logs: `/home/sime/.hermes/cache/scratch/decode-nccl-rank{0,1}.log`. This tests the exact collective/view operation, not the whole model's internal tensors.
- Full-model cold chats below: HTTP 200, 22,130 prompt / 64 completion, length finish, identical SHA-256 output `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84` at all four caps, matching the historical successful outputs. This is output-level evidence, not a strict activation/state parity oracle.
- Two-request cap-384 serving smoke: both HTTP 200 (short request 71/128 tokens, long request 5,552/8 tokens), both slots admitted and progressed, decode pipe began/drained and async slot 1 promoted to decode; coordinator and worker exited 0. No CUDA, NCCL, state or protocol errors. Log: `/home/sime/.hermes/cache/scratch/span-smoke-cap384-20260928-074525/`. The 128-token request repeatedly reused eager decode scratch.

Runner SHA-256 `e94624204d2057376ba92d75c74c0fdb760d97efabf630e1131bd63a3bb1bb53` matched on both nodes; pack SHA-256 `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e` matched. The old full-span NCCL example and new runner are distinct artifacts. Both Spark GPUs were idle at preflight, and operator permission for GPU tests was obtained.

## Fresh-server cold-chat timing (seconds)

Same model Qwen3.8-27B-UD-Q4_K_M, F16 KV, context 65,536, max batch 2, unified, speculation off, TP graph off; same 22,130-token fixture, `enable_thinking=false`, temperature 0, seed 1, max output 64. Fresh server per row, no profiling. Phase boundary is coordinator cold `TP prefix admit (cache cold) ... tokens=22130` -> first subsequent `TP decode pipe began` -> `shutdown: draining engine`; the second interval includes client completion and shutdown signal latency, not solely GPU decode.

| cap | Historical client wall | Historical admission->decode | Historical decode->shutdown | New client wall | New admission->decode | New decode->shutdown |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 32.767 | 22.086 | 10.690 | 30.851 | 21.959 | 8.888 |
| 512 | 33.264 | unavailable | unavailable | 30.601 | 21.693 | 8.892 |
| 1024 | 36.291 | unavailable | unavailable | 30.673 | 21.716 | 8.941 |
| 2048 | 41.760 | 22.604 | 19.151 | 30.827 | 21.921 | 8.899 |

New cap-2048 minus cap-384 decode/shutdown is +0.011 s, versus historical +8.461 s. The admission->decode difference changed from historical +0.518 s to -0.039 s. These single runs show a near-flat pre-decode curve for caps 384–2048; 512/1024 are nominally fastest on pre-decode (21.693/21.716 s), but differences of tenths of a second in single runs do not establish a distinct optimum. No claim of a prefill improvement from the decode change: the pre-decode work was not modified. Do not select a new production cap from these data, and do not proceed to 4096/8192 without a separate decision. Historical 8192 startup OOM remains relevant regardless of the decode fix.

The old sweep's *total wall* ranked 384 ahead of 2048 partly because 2048 paid for a capacity-sized collective on every decode row. It remains a valid historical observation under that implementation, but is not evidence that prefill at 2048 was ~9 s slower. New cold runs all emitted six existing mixed/prefill tick-stall warnings per request, with no errors. The benchmark harness deliberately signaled the coordinator after collecting the full response; headline launchers returned 143 while requests succeeded. Raw results and full coordinator/worker-tail logs: `/home/sime/.hermes/cache/scratch/decode-live-row-20260928-074015/` (384, 2048), `/home/sime/.hermes/cache/scratch/decode-live-row-20260928-074258/` (512, 1024).

Remaining correctness boundary: matching sampled output and isolated NCCL parity do not establish numerical identity of every DeltaNet state element or prefix-cache restoration parity. The latter remains an independent outstanding qualification before changing the production default. Graph-enabled decode retains its original capacity-dependent cost and was not exercised in these TP_GRAPH=0 measurements.
