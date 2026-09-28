# Generic tensor-parallel architecture plan

Baseline: `review/qwen38-tp2-final` at `4af1b2f`.

## Goal

Tensor parallelism should be a reusable subsystem, not a second implementation
of each model. A conventional model should add TP primarily by declaring shard
layout, local dimensions/head geometry, reductions, and small lifecycle hooks.

Changing TP=2 to TP=3/4 should be a TP-runtime/control-plane change plus tensor
divisibility validation, not a rewrite of every model adapter.

Models must also be able to replace individual generic TP components with
architecture-specific optimized implementations without forking the rest of
the TP stack.

## Design rules

1. Reuse the normal model's existing compute primitives and kernel dispatch.
2. Keep rank/world/collective logic in generic TP modules.
3. Keep sharding policy declarative where possible.
4. Keep exotic algorithms model-specific.
5. Provide default generic FFN/GQA/prefill implementations with narrow override
   points for faster model-specific implementations.
6. Avoid forcing NCCL/rank/world-size into the normal single-device model path.
7. Extract abstractions from the working Qwen3.8 implementation; do not design a
   large trait hierarchy up front.
8. Every extraction must preserve the known Qwen3.8 TP=2 behavioral baseline.

## Current code classification

### Generic TP

- `gpu/distributed.rs::Communicator`: collective API and compute/comm fences.
- tensor shard loading (`TensorSliceRequest`, `ShardKind`).
- rank/world validation.
- row/column-parallel linear patterns.
- all-reduce completion of row-parallel projections.
- most FFN TP behavior.
- most standard GQA projection/output-shard behavior.
- span ownership/collective-order concepts.
- mirrored cache allocation/publication/release concepts.
- TP profiling infrastructure.

### Model declaration / policy

Qwen should eventually supply mostly:

- weight names/references.
- hidden/intermediate dimensions.
- attention head/KV-head geometry.
- positional-encoding and Q/K normalization policy.
- which layers are full-attention vs DeltaNet.
- shard layout declarations.
- optional optimized overrides.

### Genuinely Qwen-specific

- Gated DeltaNet recurrence.
- recurrent matrix state and convolution carry.
- DeltaNet checkpoint semantics.
- Qwen3.8 hybrid-layer sequencing.
- Q/gate projection details and M-RoPE specifics not representable by the
  generic attention spec.
- architecture-specific fast paths retained as explicit overrides.

### Accidental duplication / coupling

- `ffn_tp.rs` hard-codes TP=2 even though its shard loader and collective
  pattern are generic.
- `gqa_tp.rs` hard-codes TP=2/local head division despite shared normal-path
  attention/GEMM primitives.
- `tp_span.rs` owns detailed quantization scratch knowledge that should track
  shared prefill primitives rather than duplicate their assumptions.
- `tp_model.rs` mixes model traversal, TP runtime validation, graph/lane
  mechanics, and Qwen-specific layer policy.
- `tp_serve.rs` is Qwen-specific at its model-facing boundary even though
  much of its mirrored scheduling/cache protocol is generic.
- `tensor_slice.rs` rejected world sizes other than 1/2.
- NCCL construction hard-coded world size 2.

## Target shape

```text
normal/shared model primitives
          ^
          |
    generic TP core
    - topology
    - collectives
    - sharded linear
    - generic FFN
    - generic attention
    - span/prefill
    - cache/state hooks
          ^
          |
 model TP spec / shim
          |
  optional overrides
```

The generic implementation is the default. A model may override a component
(for example attention/GQA or prefill) while continuing to use generic FFN,
cache, collectives, profiling, and serving machinery.

## Implementation sequence

### Phase 1: topology and shard substrate

- Remove TP=2 assumptions from reusable tensor slicing and NCCL construction.
- Introduce `TpTopology` and standard linear shard modes.
- Keep the current TP=2 process bootstrap unchanged until model execution is
  world-size agnostic.

### Phase 2: generic sharded linear / FFN

- Extract common projection loading and row/column-parallel behavior.
- Move the conventional SwiGLU FFN out of Qwen-specific code.
- Qwen supplies names/dimensions/spec only.
- Preserve a model override seam.

### Phase 3: shared prefill staging

- Stop encoding upstream GEMM/quant scratch policy in Qwen-specific TP span
  code.
- Wrap/reuse the same prefill preparation and projection primitives as the
  normal path.
- Ensure future dispatch changes flow into both paths.

### Phase 4: generic standard GQA

- Genericize head partitioning, Q/K/V shard loading, KV paging, attention
  dispatch, output row shard + reduction.
- Keep model policy and optional optimized attention override separate.

### Phase 5: collapse Qwen TP model traversal

- Replace Qwen FFN/GQA implementations with generic components + specs.
- Leave hybrid-layer selection and DeltaNet adapter in Qwen.

### Phase 6: state/cache hook

- Define a narrow generic state snapshot/restore interface.
- Attach DeltaNet state through that hook.
- Move model-independent mirrored cache lifecycle out of the Qwen boundary.

### Phase 7: serving/runtime generalization

- Generalize worker orchestration from one worker to `world_size - 1`.
- Make control fan-out/fan-in rank-generic.
- Preserve rank-symmetric failure semantics.

### Phase 8: prove TP=3/4 and a second model

- TP=3/4 should require no Qwen algorithm changes where geometry divides.
- Add a conventional second model using a small TP spec/shim.
- A second model requiring another full `*_tp.rs` stack is considered a
  failed abstraction.

## Review gates

At every meaningful commit:

- no new model-specific TP=2 constants in generic code.
- normal single-device behavior remains untouched.
- existing Qwen TP=2 output/benchmark baseline remains the acceptance oracle.
- generic code contains no Qwen tensor names or architecture constants.
- an optimized override can replace a component without replacing the whole TP
  runtime.
