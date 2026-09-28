# Stock Paddock v0.1.10 on one DGX Spark: Qwen3.8 baseline

## Result

The current official Linux ARM64 release is Paddock `v0.1.10`, runner build `0.1.10 (g610cd8f5)`, release commit `610cd8f5ea72e7421a99ca418d51de804d312730`. Its medium-prompt decode median was **17.706 client-observed output tok/s** (range 17.653–17.712), versus **14.945 tok/s** for the TP2 eager/no-spec reference: +2.761 tok/s / +18.5%. This is the stock binary's normal behavior, including observed default MTP/speculation; it is not a like-for-like no-spec comparison.

The 22,130-token long prompt was first rejected under untouched release defaults (`max_ctx=4096`). At the user's follow-up direction, it was rerun with 65,536 context. The release's default 32 slots could not fit that KV plan on one GB10, so `--max-batch 2` was also required; this matches the TP2 fixed base batch width. Results and the explicit deviations from release defaults are recorded below.

## Identity and setup

- Official latest release at test time: [`truespar/paddock` v0.1.10](https://github.com/truespar/paddock/releases/tag/v0.1.10), published 2026-09-25; release build commit `610cd8f5ea72e7421a99ca418d51de804d312730`.
- Official Linux aarch64 archive: `paddock-0.1.10-aarch64-linux.tar.gz`; SHA-256 `9f50baf2c14b22df39cd970792f02218eb92679f4c1dea98815d2168c3a2ab99` (verified against its published `.sha256`).
- Executed `paddock-runner` SHA-256: `bab2cd32743893fcc41082de161a3406b26d5b853a4194823a1ba5344a4e0f25`; `--version` returned `paddock-runner 0.1.10 (g610cd8f5)`.
- Host: one local DGX Spark GB10, Linux aarch64, NVIDIA driver 580.173.02. No remote rank or second Spark was used. The runner and its built-in kernels were extracted from the published archive and not modified. No repository source or published-binary bytes were changed.
- Model: `/home/sime/models/Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` (same model hash as TP2 report).
- Prompts copied byte-for-byte from the prior benchmark evidence: long 94,391 bytes, SHA-256 `eefe6ac5d9b7d0a01437d28f7c173b7160c7e60dc4cd3bd353d858c74fe3d518`; medium 14,012 bytes, SHA-256 `77933738c0851b091905bd01b22169f8a8b58ddc33df000e0c26eef867df952e`.
- Server used built-in defaults. CLI was only `paddock-runner --model <same GGUF> --host 127.0.0.1 --port <fresh port>`; loopback host and port isolate the test. No max-context, batch, KV dtype, spec, or graph options were set. Startup confirmed built-in defaults, 32 slots, `max_ctx=4096`, and `spec/MTP on`. The model log and nonzero draft/accept metrics confirm the stock default actually used speculative decoding.
- Each successful sample had a fresh runner process. The existing production benchmark client was reused unchanged: `max_tokens=512`, temperature 0, seed 1, `enable_thinking=false`, concurrency 1. Completion ended naturally at 194 tokens (`stop`), not at the requested limit.

## Medium workload: 3,446 prompt tokens

All three requests returned HTTP 200, generated 194 tokens, finished with `stop`, and produced the same output hash: `bb07cdd102e02fee7aed4163ae55a0099b7ab0eb9d72296a747f8d835fddec8a`.

| Metric | Median | Range | Timing source |
|---|---:|---:|---|
| Prompt / completion tokens | 3,446 / 194 | identical | Server usage in streamed response |
| TTFT | 4.145 s | 4.124–4.149 s | Client-observed, first content chunk |
| Prefill rate | 831.4 tok/s | 830.5–835.6 | Approximation: prompt tokens / client TTFT; not a device timer |
| Output/decode rate | 17.706 tok/s | 17.653–17.712 | Client-observed, completion tokens / stream time after first content chunk |
| Total client wall | 15.106 s | 15.077–15.134 s | Client-observed request start to stream completion |

The default runtime's Prometheus `/metrics` snapshots independently reported end-to-end request duration 15.106 s median (15.076–15.133), prompt/generation token counts 3,446/194, and `gen_ai_server_time_per_output_token_seconds` 0.05647 s/token median (inverse 17.707 tok/s), consistent with client decode throughput. Spec counters increased by 181 drafted and 111 accepted tokens in each run (61.3% accepted/drafted).

The server's exported `gen_ai_server_time_to_first_token_seconds` was 0.00704 s median (0.00670–0.00713), despite client-observed TTFT around 4.145 s; the metric help labels this client-visible TTFT, but the values conflict sharply. I report the counter as emitted and use client TTFT for comparison; its semantics/measurement are unresolved here. Paddock does not export a separate prefill tok/s metric.

### Comparison to TP2 eager/no-spec

| Medium workload | Stock v0.1.10 single Spark | TP2 branch A eager/no-spec | Difference |
|---|---:|---:|---:|
| Client TTFT | 4.145 s | 3.177 s | stock +30.4% |
| Approx. prefill rate | 831.4 tok/s | 1,084.6 tok/s | stock lower; proxy only |
| Output/decode tok/s | 17.706 | 14.945 | stock +18.5% (+2.761 tok/s) |
| Client total wall | 15.106 s | 16.363 s | stock -7.7% |
| Generated tokens | 194 | 197 | stock stopped 3 tokens earlier |

These are user-visible measurements under each requested configuration, not a controlled kernel comparison: the published runner used its default MTP/spec path, while TP2 reference A disabled speculation. The outputs also differ (TP2 hash `722eb0754f07d3abfb46f6aedcf2e34359522d54f0eeb03a7437d6c2aa35a15e`). The stock result answers how fast the release serves this prompt by default; it does not isolate a single-GPU-vs-TP2 hardware effect.

## Long workload: 22,130 prompt tokens, 65,536 context

All three fresh-server requests returned HTTP 200, stopped naturally at 108 completion tokens, and produced the same output hash: `e3a07a4353b2ba6e92c9be853a7fe0815db64daa2486db3576fce19d1e5bfdd9`.

The original default configuration had `max_ctx=4096` and rejected this prompt. Raising only `max_ctx` to 65,536 while leaving the default batch width (32 slots) failed startup: the runner reported 136 GiB of KV required versus 83.14 GiB available on this Spark, and estimated a maximum context near 40,048 at that width. The successful 64K run therefore used `--max-ctx 65536 --max-batch 2`; two slots match the TP2 base configuration and the KV plan then started successfully. All other serving options remained at published defaults, including built-in MTP/spec policy. The server's resolved KV plan confirmed `slots=2 max_ctx=65536`, an 8.5 GiB pool with 131,072 token capacity.

| Metric | Median | Range | Timing source |
|---|---:|---:|---|
| Prompt / completion tokens | 22,130 / 108 | identical | Server usage in streamed response |
| TTFT | 36.931 s | 36.767–37.099 s | Client-observed, first content chunk |
| Prefill rate | 599.2 tok/s | 596.5–601.9 | Approximation: prompt tokens / client TTFT; not a device timer |
| Output/decode rate | 12.054 tok/s | 12.047–12.095 | Client-observed, completion tokens / stream time after first content chunk |
| Total client wall | 45.890 s | 45.697–46.064 s | Client-observed request start to stream completion |

### Comparison to TP2 eager/no-spec

| Long workload | Stock v0.1.10 single Spark, 64K / batch 2 | TP2 branch A eager/no-spec | Difference |
|---|---:|---:|---:|
| Client TTFT | 36.931 s | 21.097 s | stock +75.1% (slower) |
| Approx. prefill rate | 599.2 tok/s | 1,049.0 tok/s | stock -42.9%; proxy only |
| Output/decode tok/s | 12.054 | 6.664 | stock +80.9% (+5.390 tok/s) |
| Client total wall | 45.890 s | 37.317 s | stock +23.0% (slower) |
| Generated tokens | 108 | 108 | same length |

So the single Spark release sustained substantially faster decode, but its much slower prefill erased that gain in end-to-end latency for this long prompt. This is an observed configuration comparison, not a controlled hardware-only comparison: one Spark vs two, and published stock defaults vs TP2 eager/no-spec.

The server's `/metrics` output-token timing agreed with client decode (median 0.08296 s/token, inverse 12.05 tok/s); request-duration metrics were 45.696–46.064 s. Its `gen_ai_server_time_to_first_token_seconds` remained inconsistent with client observation: 0.0149–0.0254 s reported versus 36.767–37.099 s client-observed. Use client TTFT for the comparison and treat the server TTFT metric as unresolved. The metrics reported zero drafted/accepted speculative tokens for all three long requests, despite the model load log describing MTP support as enabled; do not attribute this long-run decode result to observed speculative acceptance. Paddock does not export a separate prefill-throughput metric.

## Logs and warnings

- The original default-context attempt is preserved at `raw/long/r1/`; it ended in the expected HTTP 400. The initial 64K/batch-32 startup failure is preserved at `raw-long64k/failed-maxbatch32-r1/`.
- Three successful 64K-context runs, each with a fresh process, are in `raw-long64k/r{1,2,3}/`: exact command/config JSON, complete server logs, client result/SSE data, runner exit, and before/after `/metrics` snapshots. Each runner exited 0 after graceful shutdown.
- Each successful long run emitted repeated `tick-stall` WARNs during mixed/chunked prefill (mixed phase roughly 1.08–2.50 s per logged stall); no other WARN/ERROR class, CUDA/NCCL error, HTTP error, or runner failure was observed.
- The three `/metrics` snapshots agree on decode rate, but the server TTFT metric conflicts with the client timing as noted above.


## Evidence and reproducibility

- Raw timestamped evidence directory: `/home/sime/.hermes/cache/scratch/paddock-stock-baseline-20260928/`.
- `raw/medium/r{1,2,3}/`: exact CLI/config JSON, full server log, client stdout, raw SSE/response text/hash, runner exit, `/metrics` before/after.
- `raw/long/r1/`: original default-context config, startup/server log, 400 error, post-request metrics, runner exit.
- `raw-long64k/r{1,2,3}/`: 64K-context run configs, startup/server logs, client results/SSE, runner exits, and metrics snapshots; `raw-long64k/failed-maxbatch32-r1/` preserves the insufficient-memory startup attempt.
- Published archive/checksum, extracted version and capability output, and hashes: root-level `paddock-0.1.10-aarch64-linux.tar.gz*`, `published/`, `runner-version.txt`, `runner-capabilities.json`, and `provenance.sha256`.
- Scratch drivers: `stock_baseline.py` reproduces release-default samples; `long64k_extension.py` runs fresh 64K-context servers with the recorded batch-width adjustment. Both use the unchanged TP2 benchmark client.

This follow-up changes only benchmark documentation and scratch evidence. No source, released binary, or model was changed.
