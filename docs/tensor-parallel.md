# Tensor-parallel serving

Paddock can split supported CUDA models across multiple rank processes with NCCL.
Rank 0 owns the HTTP/API endpoint; worker ranks execute the same model steps and do
not serve requests themselves.

## Configuration

Rank 0 can be configured in `paddock.toml`:

```toml
[parallel]
tp_size = 2
master_addr = "0.0.0.0"
master_port = 11982
```

Start a worker explicitly on another machine:

```text
paddock-runner --tp-worker \
  --model /path/to/model.gguf \
  --kernel-pack /path/to/pd-cuda-sm120.so \
  --tp-size 2 --tp-rank 1 \
  --tp-master-addr 192.168.100.10 --tp-master-port 11982
```

A worker defaults to local GPU ordinal 0. Use `--gpu <ordinal>` to select another
local CUDA device.

For TP=2, rank 0 may spawn a local worker automatically. Set
`PADDOCK_TP_NO_SPAWN=1` when ranks are placed manually. TP sizes above two use
explicit worker placement.

The generic runtime accepts world sizes 2, 3, 4, 6 and 8. A model adapter may
reject a size when its model geometry cannot be partitioned evenly.

## Runtime contract

Startup validates the rank/world configuration and model identity before serving.
Rank 0 distributes resolved graph mode, KV dtype, model/pack identity and serving
state to workers. Control messages are sequence checked and workers fail closed on
mismatched state rather than silently falling back to single-GPU execution.

CUDA graphs are opt-in for the TP lane with `PADDOCK_TP_GRAPH=1`. The resolved
graph mode is carried in the rank-0 handshake so workers cannot diverge.

`--kv-cache-dtype f16` and `--kv-cache-dtype fp8_e4m3` are supported by the TP
serving contract when the selected model adapter supports them.

## Architecture

The implementation is split into three layers:

- `paddock-dist`: rank/world configuration, bootstrap and framed control protocol.
- `paddock-engine::tp`: model-neutral sharding, paged cache, attention/FFN,
  traversal and serving orchestration.
- model adapters under `gpu_model`: geometry, tensor names and model-specific
  recurrent/state behavior.

The generic layer is world-size agnostic; model adapters own geometry admission.
