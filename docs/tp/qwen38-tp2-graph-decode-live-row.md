# TP2 graph DeltaNet decode: live-row collective

Starting HEAD `cc2fb0d95e33e4593792b96d44c1ae643c65c188` on `review/qwen38-tp2-final`. Production `PADDOCK_TP_SPAN_CAP` default remains 64.

## Contract

`Qwen35TpRank::enable_tp_graphs` captures the per-layer slot-0 mixer run; other DeltaNet slots capture lazily on first use. `capture_mixer_run` synchronizes the compute stream, begins thread-local capture, records `record_mixer_run`, ends and instantiates the graph keyed by `(layer, slot)`. The DeltaNet recorded run includes norm, one-row recurrence, the partial-buffer memset and out-projection GEMV. Replays use stable rank-local span partial and state pointers; slot-specific state is swapped in before launch. The NCCL sum and its compute/communication stream event fences run outside capture, after the graph replay. Residual add reads only the first WIDTH=5120 elements of the reduced plane. Buffer capacity remains `span_cap * WIDTH`, shared with prefill; prefill's independent live-prefix reduction is unchanged.

The graph-enabled `forward_token_body_graphed` processes exactly one `(token, position, slot)` at a time. There is no graph for a wider row batch; a scheduler tick with two live slots executes separate one-row bodies (and per-slot DeltaNet captures), not a two-row mixer graph. Thus `live_decode_rows=1` for each recorded graph today; the `finish(rows)` API uses `rows * WIDTH` so a future fixed-width graph can pass its own row count without rewriting the collective. Prefix views preserve the base pointer, and neither graph replay nor subsequent residual add consumes the capacity suffix. NCCL ordering and event fences are unchanged.

The implementation unifies eager and captured `finish_partial(e, rows)` to zero only the live prefix and makes post-replay `finish(e, group, rows)` reduce prefix views of persistent `partial` and `reduced`. At caps 384 and 2048 the one-row NCCL count changes from 1,966,080/10,485,760 f32 elements (7.5/40 MiB) to 5,120 (20 KiB). The graph-visible base allocation and recorded GEMV pointer remain unchanged. This reflects upstream Qwen3.5's separation between prefill chunk capacity and live decode width; no upstream code was copied.

## Two-node validation

Identical Qwen3.8-27B-UD-Q4_K_M, pack, F16 KV, context 65,536, batch 2, unified, speculation off, graph on. Fresh server per cold chat, 22,130 prompt tokens, 64 output tokens, seed 1, temperature 0, `enable_thinking=false`. Old-graph runs used the prior release executable; new-graph runs used the rebuilt executable staged identically on the worker. Each returned HTTP 200, length finish, 22,130/64 usage and SHA-256 text `4156a7f2716cbaf9d1734bbebfe8757841d200f0af4d9f6b5ef91d562f715f84`, matching validated eager output. Each cold-run harness deliberately stops the coordinator after response, so launcher status 143 is not a clean process-exit claim; client success and no errors are separately checked. Matched binary SHA-256 on both nodes for new graph `10258e62dee7f732f2307d49770fd0e9325d270e520b1285adf25e9b26b5a4b7`.

| Graph TP cap | Old client wall | Old admit->decode | Old decode->shutdown | New client wall | New admit->decode | New decode->shutdown |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 32.658 | 22.047 | 10.599 | 30.793 | 21.944 | 8.832 |
| 2048 | 41.636 | 22.514 | 19.107 | 30.620 | 21.770 | 8.844 |

Decode/shutdown cap gap changed from +8.508 s to +0.012 s. Neither phase interval isolates GPU-only decode; the latter includes response/stop latency. No prefill improvement is claimed. The two-rank isolated NCCL old/full-versus-live-prefix probe in `examples/nccl_bench.rs` from the preceding eager change established bitwise-identical row-0 outputs on both ranks at both caps. Full-model matching hash is output-level parity, not whole-state/activation bitwise proof.

A graph-enabled two-request smoke at cap 384 exercised slot 0 and slot 1, decode after prefill, lazy per-slot graph capture and replay, and 128 output tokens on the long-running request. Both requests returned HTTP 200 (71/128 and 5,552/8 prompt/completion usage); coordinator and worker exited 0. No CUDA, NCCL, graph, protocol or state errors appeared. There is no native B>1 graph mixer to test: two simultaneous live requests are dispatched as separate one-row graph runs. Strict prefix-cache restoration parity remains open.

Evidence: `/home/sime/.hermes/cache/scratch/decode-live-row-20260928-075705/` (old graph), `/home/sime/.hermes/cache/scratch/decode-live-row-20260928-080230/` (new graph), `/home/sime/.hermes/cache/scratch/span-smoke-cap384-20260928-080451/` (two requests). Host verification: `cargo check -p paddock-engine --lib`, 518 paddock-engine library tests, 28 paddock-dist tests, six DeltaNet GPU parity tests, release runner build, `cargo clippy --release -p paddock-engine -p paddock-dist --lib` and `git diff --check` passed. Clippy still emits the pre-existing unnecessary cast warning in `src/cuda.rs:83`; the optional heavy DeltaNet speed gate was skipped.
