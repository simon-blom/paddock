# Phase 9 — production TP=2 worker and scheduler integration

Status: batch-one eager serving gate PASS for the pinned Qwen3.8 checkpoint on two Sparks. This is a bounded, serial Phase 9 lane; Phase 10 async/batched execution, graphs, speculation and offload are not part of this result. Work remains uncommitted.

## Integration

Rank 0 boots the ordinary runner/API and engine scheduler. For TP=2 it elects `TpGenerator`, a serial-only `Generator` backed by a GPU-thread-owned `TpCoordinator`; TP=1 continues through the original GPU model factory and batched-capable engine. Rank 1 boots in worker mode and runs `Qwen35TpRank` without an API, tokenizer, scheduler or sampler. Rank 0 authorizes mirrored KV Ensure/Flush operations, sends sequenced token/position/operation commands, waits for rank 1 to prepare a step before entering NCCL work, and waits for completion afterwards. A mismatch, worker error or lost connection poisons the generator rather than downgrading to TP=1. The two ranks compare pinned GGUF SHA-256, CUDA-pack BLAKE3 and context before NCCL initialization. Large tensors and collectives use NCCL, not the TCP control stream. Unsupported TP=2 configurations fail startup before joining the worker.

This lane supports one active slot, token-serial prompt processing, FP16 KV, explicit reset between requests, and no prefix reuse. It retains the sole slot until the next request's mirrored Flush (or model shutdown), rather than performing a separate free at completion. Logical page growth is mirrored. The worker and coordinator own their CUDA/NCCL resources on their respective GPU threads. A bounded control ACK timeout is not a watchdog for a collective that itself stops making progress.

## Two-Spark gate

### Supported two-node startup (upstream-readiness remediation)

The operator-supported worker path is `--tp-worker`; it replaces the old
hand-set `PADDOCK_TP_WORKER_CHILD` recipe, which was never a documented
surface:

```sh
# worker box (192.168.100.11):
./paddock-runner --tp-worker \
  --model /models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --kernel-pack /packs/pd-cuda-sm120.so \
  --tp-master-addr 192.168.100.10 --tp-master-port 11982

# head box (192.168.100.10) - waits for the manual worker:
PADDOCK_TP_NO_SPAWN=1 PADDOCK_TP_SIZE=2 PADDOCK_TP_MASTER_ADDR=0.0.0.0 \
  ./paddock-runner --model ... --kernel-pack ...
```

`--tp-worker` requires `--model`/`--kernel-pack` (or the coordinator-spawn
`PADDOCK_TP_MODEL`/`PADDOCK_TP_PACK` env fallbacks) and the coordinator's
address; missing inputs fail before dialing. The spawned child and the
manual worker run the same worker runtime.

Checkpoint SHA-256 on both ranks: `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`. Pack SHA-256 on both ranks: `059bb62d2e6d863b32ac47208d29da7492618b2fa33015a3ee0ad6fe2a9c4d54`. Release runner SHA-256 on both ranks: `f5bc035c8d3b4e8efaa670766ac44e3cc45b6e5ce459533cad05d8670ebf3f5f`. Head/worker used the Phase 8 NCCL 2.30.4 aliases and the same socket/IB interface pins, with rank 0 bound to 192.168.100.10 and the second Spark reached over SSH at 192.168.100.11 (the control connection appeared to rank 0 from 192.168.100.15). Each run used `--device cuda --max-batch 1 --kv-cache-dtype f16 --no-spec` and a loopback-only HTTP API on rank 0; rank 1 used its manually started worker-child path. Sampling was greedy (`temperature=0, top_k=1`). TP=1 references were started only after both TP=2 processes exited, so the models did not compete for memory.

For the current operator-facing two-node launcher, copy `docs/tp/qwen38-tp2-two-node.env.example` to the ignored `docs/tp/qwen38-tp2-two-node.env`, edit paths as needed, and run `./docs/tp/qwen38-tp2-two-node.sh` on rank 0/head. The launcher starts rank 1 over SSH through the supported public `--tp-worker` form, then runs the coordinator locally. Set `TP_ENV_FILE=/path/to/custom.env` to use another dotenv file; exported shell values take precedence over file values, which take precedence over script defaults.

- With max context 16, two sequential `/v1/completions` requests (`prompt="Hi", max_tokens=2`) each returned HTTP 200, text `", I"`, prompt 1 token, completion 2 tokens, finish `length`. The separately run TP=1 server returned the same text, token counts and finish reason for each request. Both TP=2 processes and the TP=1 process shut down with exit code 0. The second request exercises reset/replay, not a single long-lived KV continuation.
- With max context 32, a TP=2 completion from `"Hi"` with 18 generated tokens returned HTTP 200, text `", I'm a 20-year-old male. I'm 5'10"`, prompt 1 token and completion 18 tokens, finish `length`. This crosses the 16-token logical KV block boundary. The separately run TP=1 server returned identical text, counts and finish reason. Both TP=2 processes and TP=1 exited 0 after graceful shutdown.
- `--max-batch 2` under TP=2 returned exit code 2 with the Phase 9 unsupported-configuration message before binding the control port; there was no silent TP=1 fallback.
- A rank-1 worker deliberately started with a different existing CUDA pack returned exit code 1 with `rank-1 checkpoint, CUDA pack or context disagrees with rank 0`. Rank 0 returned exit code 1 with that error during engine startup and never opened an HTTP API.

Host checks: `cargo check -p paddock-runner --all-targets`, release runner build, all 506 runner library tests, 18 distributed tests, six targeted TP engine tests and `git diff --check` passed. `cargo fmt --all -- --check` reported formatting differences in three unchanged files (`paddock-dist/tests/bootstrap.rs`, `paddock-engine/src/gpu/kquant.rs`, `paddock-engine/src/gpu_model/qwen35/ops.rs`); those unrelated files were left untouched.

## Limits

HTTP parity here is matching output text/token counts, not per-token-ID or full-vocabulary logit parity. Phase 8 independently established full-vocabulary eager-logit parity for five steps and exact rank-1 reset replay. The Phase 9 serving gate has not exercised cancellation mid-collective, a rank-1 crash during GPU execution, a wider batch, longer contexts, throughput, or a protocol-level timeout under a hung NCCL collective. Those are not implied by the observed passing runs.
