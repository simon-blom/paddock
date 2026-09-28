# Qwen3.8 TP2 production serving benchmark

## Status and provenance

- Source HEAD: `8644ce15f439f89b463887501d2b88240cfd2d6c` (`review/qwen38-tp2-final`). No source or kernel changes were made for this benchmark.
- Runner SHA-256: `408f2d2309830ee0e16ae97f48c5fb30a2799b51989fe97de430f1526bcf7b28` (`target/release/paddock-runner`). The remote worker runner had the same SHA-256.
- CUDA pack SHA-256 (both nodes): `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e`.
- Model SHA-256 (both nodes): `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
- Setup: two DGX Spark GB10 nodes, TP=2; Qwen3.8-27B-UD-Q4_K_M; F16 KV; unified; context 65,536; max batch 2; span cap 512; prefix cache disabled to avoid cross-run cache effects; temperature 0, seed 1, `enable_thinking=false`.
- Span cap default is unchanged at 64. No span-cap or speculative-K sweep was performed.
- Raw evidence root: `/home/sime/.hermes/cache/scratch/qwen38-tp2-production-20260928-083340/` (per-run config, coordinator and worker logs, client stdout, JSONL, SSE and output text/hash). The source benchmark client/matrix are `docs/tp/qwen38-tp2-production-benchmark.py` and `docs/tp/qwen38-tp2-production-matrix.py`; launcher is `docs/tp/qwen38-tp2-two-node.sh`.

## Speculative and graph configuration

Qwen's in-file MTP `nextn` block is the draft mechanism; target-model verification runs the proposed tokens through the backbone. The service defaults are `PADDOCK_SPEC_MAX_K=7` (maximum draft depth), `PADDOCK_SPEC_MAX_ROWS=32` (round row budget), and `PADDOCK_SPEC_K_MISS_FLOOR=1`; no overrides were set. Effective K is adaptive and also bounded by row budget/live slots, so 7 is a ceiling, not a claim that every round drafted seven. `SPEC=on` enabled speculation. Config D additionally set `SPEC_GRAPH=on`; A/B/C set it off. B/D set coordinator `PADDOCK_TP_GRAPH=1`; worker graph mode is propagated through TP initialization.

The TP graph implementation captures collective-free decode runs (pre-attention and pre-FFN compute); collectives remain outside graph capture and execute between replays. Separately, Qwen3.8's MTP code has graph caches and capture/launch paths for draft, target verification, and commit when `PADDOCK_SPEC_NOGRAPH` is unset; C explicitly set this variable, D left it unset. Thus D requested both TP decode graphs and speculative phase graphs, while C used eager speculative phases. The benchmark's settings and launcher environment are recorded per run. However, runtime logs contain no explicit graph-capture/replay counters or graph-mode startup confirmation. They show no graph/capture/replay errors, but this run cannot independently prove sustained replay, quantify eager fallback share, or establish the actual phase/shape coverage of replay. Treat B/D as graph-*requested* results, not conclusive graph-activity evidence. This is an instrumentation gap, not a failed request.

The implementation allows at most 7 draft tokens per slot by default, with `PADDOCK_SPEC_MAX_ROWS=32` bounding aggregate verify rows; TP=2 and batch 2 both served successfully. The observed mean accepted drafts per verification step was very low. The exposed metrics are global drafted/accepted token counters; per-position acceptance, target verify step count, CUDA stage timings, and draft-vs-target device time are not exposed here. Reported verify-step counts and mean acceptance per step below are estimates from generated-token count, not a direct server metric. Draft acceptance rate is accepted/drafted, distinct from useful accepted output tokens/s.

## Method

Every headline sample used a fresh server. One chat-completion request (or two clients synchronized on a start barrier) streamed output and saved raw SSE and content. Runs asked for `max_tokens=512`, but these greedy completions stopped naturally at 108 tokens (long) and 197 (medium), so the results are not 512-token decode runs. All completed with HTTP 200 and finish reason `stop`. Client wall and TTFT are monotonic client observations; prefill tok/s is prompt tokens divided by client TTFT (an end-to-end proxy, not a server/CUDA prefill timer); output tok/s is completion tokens divided by client time from first content chunk to stream completion. No coordinator/device timing is mixed into those values.

A/B/D have three measured runs for each workload; C concurrency-2 was interrupted after two valid runs, then its third run was completed in a fresh server. Initial non-successful setup attempts are preserved under `preliminary/` and are not included. The two-request test used a threading barrier; recorded overlap was about 29–34 seconds, and both requests completed during the same server lifecycle. This verifies concurrent client execution, not simultaneous GPU kernel occupancy. The client’s concurrency summary aggregates output tokens over the interval spanning both requests. Its per-request records report latency/TTFT and individual output rate.

The medium prompt was deterministically cut from the long fixture and is preserved as `medium-prompt.txt`; prompt token count was 3,446. The long fixture count was 22,130.

## Concurrency 1 — long context (22,130 prompt tokens; 108 generated)

Medians with observed ranges in parentheses. All values are client-observed; prefill is approximate prompt tokens / TTFT.

| Config | TTFT s | Prefill tok/s | Output tok/s | Total wall s | Completion/hash |
|---|---:|---:|---:|---:|---|
| A eager / no spec | 21.097 (20.981–21.146) | 1049.0 (1046.5–1054.8) | 6.664 (6.658–6.665) | 37.317 (37.185–37.353) | 108 / `e3a07a4353b2ba6e92c9be853a7fe0815db64daa2486db3576fce19d1e5bfdd9` |
| B graph requested / no spec | 21.096 (21.037–21.160) | 1049.0 (1045.8–1052.0) | 6.708 (6.701–6.712) | 37.195 (37.155–37.251) | same |
| C eager / spec | 21.123 (21.090–21.145) | 1047.7 (1046.6–1049.3) | 6.480 (6.461–6.480) | 37.806 (37.789–37.812) | same |
| D graph+spec requested | 21.141 (21.086–21.167) | 1046.8 (1045.5–1049.5) | 6.516 (6.504–6.541) | 37.742 (37.597–37.747) | same |

## Concurrency 1 — medium context (3,446 prompt tokens; 197 generated)

| Config | TTFT s | Prefill tok/s | Output tok/s | Total wall s | Completion/hash |
|---|---:|---:|---:|---:|---|
| A eager / no spec | 3.177 (3.176–3.181) | 1084.6 (1083.2–1085.1) | 14.945 (14.938–14.962) | 16.363 (16.342–16.365) | 197 / `722eb0754f07d3abfb46f6aedcf2e34359522d54f0eeb03a7437d6c2aa35a15e` |
| B graph requested / no spec | 3.173 (3.173–3.183) | 1085.9 (1082.8–1086.1) | 15.173 (15.169–15.204) | 16.157 (16.130–16.170) | same |
| C eager / spec | 3.182 (3.181–3.187) | 1082.9 (1081.2–1083.4) | 14.032 (14.026–14.037) | 17.220 (17.220–17.227) | same |
| D graph+spec requested | 3.185 (3.167–3.222) | 1081.8 (1069.6–1088.3) | 14.240 (14.120–14.278) | 17.020 (16.964–17.174) | same |

## Concurrency 2 — medium context (2 x 3,446 prompt tokens; each 197 generated)

Aggregate output tok/s is 394 tokens divided by aggregate client wall; ranges are the three server runs. Per-request output rates are medians across requests and runs. Request TTFT varies because the shared prefill/scheduler admits the requests at different times; both slots progressed and no request was starved.

| Config | Aggregate tok/s (range) | Aggregate wall s (range) | Request output tok/s (median; range) | Request TTFT median (range) | fairness |
|---|---:|---:|---:|---:|---|
| A | 12.042 (12.037–12.042) | 32.719 (32.718–32.732) | 7.074 (6.696–7.452) | 4.730 (3.177–6.283) | both complete; one request starts decoding later |
| B | 12.192 (12.168–13.406) | 32.317 (29.389–32.380) | 7.561 (6.810–7.599) | 3.272 (3.179–6.402) | both complete; one faster aggregate outlier |
| C | 12.687 (12.652–12.723) | 31.055 (30.968–31.143) | 7.097 (7.061–7.133) | 3.260 (3.168–3.350) | both complete; closely matched per-request rates |
| D | 11.595 (11.555–12.847) | 33.980 (30.669–34.098) | 7.136 (6.410–7.211) | 3.282 (3.177–6.461) | both complete; runs 2–3 show delayed second admission and lower aggregate |

All concurrency-2 runs had client overlap >29 seconds (median overlap: A 32.585 s, B 32.051 s, C 30.979 s, D 33.833 s). Aggregate output throughput improved with C over A, but D was noisy and slower than A's median. Per-request aggregate rates are not a sum of the two independently computed decode rates because they use overlapping intervals.

## Speculation details

| Workload/config | Drafted | Accepted | Accepted / drafted | Mean accepted drafts / estimated target step | Accepted-token client decode rate |
|---|---:|---:|---:|---:|---:|
| Long C | 33 | 9 | 27.3% | 0.091 | 0.540 tok/s |
| Long D | 33 | 9 | 27.3% | 0.091 | 0.543 tok/s |
| Medium C | 49 | 16 | 32.7% | 0.088 | 1.140 tok/s |
| Medium D | 49 | 16 | 32.7% | 0.088 | 1.157 tok/s |
| Medium C, concurrency 2 | 98 | 32 | 32.7% | not reliably exposed | aggregate accepted rate 1.03 tok/s |
| Medium D, concurrency 2 | 98 | 32 | 32.7% | not reliably exposed | aggregate accepted rate 0.94 tok/s median |

Counters were stable across all three C/D single-request repetitions and C/D concurrency-2 runs. The client estimates target verification steps as completion tokens minus a small initial/final adjustment; it is not a server-counted number. No position-wise acceptance statistics, direct target verification throughput, or isolated draft overhead are available. The observed useful accepted-token rate is far below total output rate: speculation is not paying for itself in these settings. Drafting is not cheap enough to overcome the low acceptance achieved here.

## A/B/C/D effects (median output tok/s; relative to A)

| Workload | A | B (graph only) | C (spec only) | D (graph+spec) | A→B | A→C | A→D | C→D | B→D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Long c1 | 6.664 | 6.708 | 6.480 | 6.516 | +0.7% | -2.8% | -2.2% | +0.6% | -2.9% |
| Medium c1 | 14.945 | 15.173 | 14.032 | 14.240 | +1.5% | -6.1% | -4.7% | +1.5% | -6.1% |
| Medium c2 aggregate | 12.042 | 12.192 | 12.723 | 11.595 | +1.2% | +5.7% | -3.7% | -8.9% | -4.9% |

These are observed comparisons only. B's small c1 improvement is near measurement scale and graph activity is not directly instrumented. C and D show no single-request speculative gain; concurrency-2 C is faster than A, but C's only two-request clients are overlapping and share scheduling, and D's variance is material. These data do not establish causal component-level savings.

## Correctness, errors, graph and timing caveats

- Every intended configuration produced coherent text. A/B/C/D output SHA-256s matched exactly by workload (spec and graph did not alter greedy text in this workload).
- Requests succeeded with `HTTP 200`, finish reason `stop`; all intended configurations have three successful client result records and worker exit status 0. No CUDA/NCCL failures, NaN/Inf reports, protocol/state errors, or worker crashes were found in the collected successful-run logs. The matrix was inadvertently started again after the report commit; that duplicate attempt failed at `A-long-c1-r1` with `launcher required SIGKILL`, exit -9, before making a request. It overwrote that run directory's coordinator/worker logs and left the earlier successful client result/JSONL in place, so those particular client metrics cannot be paired with the current rank logs. The failed attempt is preserved by `error.txt`; exclude it from headline metrics. Other run directories remain the successful benchmark records.
- Expected `tick-stall` warnings appear during long, synchronous mixed prefill chunks (roughly 6.5–7.7 seconds long-context and 3.1–3.3 seconds medium-context). They reflect scheduler wall-time waiting in `mixed`; not CUDA/NCCL errors. This is a major latency component.
- B/D config dumps and launcher env requested TP graph mode; implementation source captures/replays collective-free runs with NCCL outside capture. But the server does not emit capture/replay counters, so capture/replay presence, sustained graph share, and eager fallback fraction remain unverified from runtime evidence. No graph-capture error was observed.
- Server admission→decode start, CUDA stage durations, draft overhead, and verification-kernel durations were not available as reliable per-request metrics in this harness. Do not interpret prompt/TTFT-derived tok/s as device prefill throughput. TTFT and wall are client-observed only.

## Interpretation and recommendation

1. D's measured single-request output rate was 6.516 tok/s (long context; three-run range 6.504–6.541) and 14.240 tok/s (medium; range 14.120–14.278). The medium prompt isolates decode more clearly.
2. Relative to A, D was 2.2% slower at long context and 4.7% slower at medium context. This is not evidence for a production performance win.
3. Graph-requested B improved non-spec output only about 0.7% long and 1.5% medium; not material at this sample count, and activity telemetry is missing.
4. D vs C graph-requested output differed +0.6% long and +1.5% medium; no material measured gain. Concurrency-2 D was lower than C and A medians.
5. Speculation alone regressed c1 output 2.8% long and 6.1% medium. Acceptance (27–33% of drafts; roughly 0.09 accepted draft token per estimated verification step) is poor; accepted output throughput is a small fraction of end-to-end output rate.
6. Max output (not a device-specific inference) remains dominated by target-model execution at each decode step, while long-context requests first incur ~21 s client TTFT and conspicuous multi-second `mixed` stalls. For serving latency, mixed/prefill scheduling is currently a measured limitation; for decode throughput, low-acceptance speculation plus target decode does not reduce work. These measurements do not isolate FFN vs other target layers or draft GPU time, so do not name a kernel bottleneck from this run.
7. Graph+spec is not yet justified as the performance target. Keep it as a candidate for future validation only after graph-activity counters and direct server/device timing are added, and after testing longer outputs that actually reach a decode steady state. Do not optimize a kernel based on the current measurements.

## Reproduction and raw records

Run the benchmark matrix from the repository using `QWEN38_BENCH_ROOT=/home/sime/.hermes/cache/scratch/qwen38-tp2-production-20260928-083340 python3 docs/tp/qwen38-tp2-production-matrix.py`. The matrix script uses fresh servers, span cap 512, disabled prefix cache and 3 repetitions per config/workload. The medium prompt and exact raw sample metadata are stored below the evidence root; see `raw/<A|B|C|D>/<workload>/r<1..3>/`. Preserve and consult coordinator and worker logs alongside JSONL/SSE, rather than relying on this summary alone.
