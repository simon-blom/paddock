# Stock Paddock v0.1.10 on one DGX Spark: Qwen3.8 baseline

## Result

The current official Linux ARM64 release is Paddock `v0.1.10`, runner build `0.1.10 (g610cd8f5)`, release commit `610cd8f5ea72e7421a99ca418d51de804d312730`. Its medium-prompt decode median was **17.706 client-observed output tok/s** (range 17.653–17.712), versus **14.945 tok/s** for the TP2 eager/no-spec reference: +2.761 tok/s / +18.5%. This is the stock binary's normal behavior, including observed default MTP/speculation; it is not a like-for-like no-spec comparison.

The long prompt could not be served with the published binary's built-in defaults: the startup KV plan reports `max_ctx=4096`; the 22,130-token request received HTTP 400. No long-context performance number is available without overriding the default context, which this task did not authorize.

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

## Long workload: 22,130 prompt tokens

One fresh default-config server started successfully and reported `max_ctx=4096`; the request was rejected with HTTP 400 before generation. The client reports `<HTTPError 400: 'Bad Request'>`, and the server's post-request metrics recorded one 4xx. Prompt/completion usage, TTFT, prefill/decode rates, and request wall time are unavailable for this rejected request. There is no valid long-context median or range. I did not repeat an identical startup-and-rejection three times or raise context to 65,536 because the request was specifically for normal published defaults.

## Logs and warnings

- Successful medium requests: no CUDA, NCCL, HTTP, protocol, or worker errors; runner exit code 0 after each request.
- Expected `tick-stall` WARN records occurred during the release's startup warm wave (about 2.2 s with 32 warmup prompts) and mixed prefill during medium requests (roughly 1.08–1.30 s). They are preserved in the per-run server logs; no GPU failure accompanied them.
- The long request is the single expected 4xx described above, not an inference crash. Model loaded and freed cleanly.

## Evidence and reproducibility

- Raw timestamped evidence directory: `/home/sime/.hermes/cache/scratch/paddock-stock-baseline-20260928/`.
- `raw/medium/r{1,2,3}/`: exact CLI/config JSON, full server log, client stdout, raw SSE/response text/hash, runner exit, `/metrics` before/after.
- `raw/long/r1/`: config, startup/server log, 400 error, post-request metrics, runner exit.
- Published archive/checksum, extracted version and capability output, and hashes: root-level `paddock-0.1.10-aarch64-linux.tar.gz*`, `published/`, `runner-version.txt`, `runner-capabilities.json`, and `provenance.sha256`.
- Scratch benchmark driver: `/home/sime/.hermes/cache/scratch/paddock-stock-baseline-20260928/stock_baseline.py`; it starts and directly supervises a fresh published runner, then calls the unchanged TP2 benchmark client. Re-run a sample with `python3 <driver> --only medium-r1` after changing the selected label; it writes artifacts under `raw/`.

The repo was clean before this report was added. No source, released binary, or model was changed, and nothing was pushed.
