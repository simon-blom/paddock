# Where the TP sequence stops without target access — Phase 14/15 blocking analysis

Status: Phase 12/13 target validation is complete for the pinned
Qwen3.8-27B checkpoint. The stop point remains in force: Phase 14 and Phase
15 are still blocked by their target-gated prerequisites below.

Clean checkpoint: Phase 12 validation/report milestone (this commit), with
implementation origin `17efad0`, and Phase 13 implementation `481d092`.
Phase 13 target evidence is limited to the same pinned checkpoint.

## Phase 14 (MoE + expert streaming) — blocked by checkpoint absence + target gates

What exists: the TP=1 qwen35 lane serves MoE checkpoints (expert quartet in
`load.rs`, routing/combine in `forward.rs`, expert streaming/offload behind
`PADDOCK_MOE_HOST`). The TP=2 lane refuses MoE checkpoints by name as of
Phase 13.

Why implementation cannot honestly start host-side:

1. **No MoE qwen35-family GGUF exists locally.** The only MoE artifact on
   disk is `Qwen3.8-Flash-Next-NVFP4` — qwen4_exp (512-expert, multimodal,
   hyper-connections, NVFP4 safetensors), a different architecture and
   format that Phase 14's own text routes to GLM/DeepSeek-style future
   work. A rank-shard expert loader must be written against a real file's
   tensor names/shapes/quant layout; guessing them is fabrication.
2. **Design decisions the plan reserves for a real model**: "prefer
   rank-local expert shards/cache planes where practical; avoid restoring
   full experts on both ranks without a model/format reason" and "benchmark
   TP vs EP rather than assuming one is superior" — both require the model
   and target timings.

Exact target-device gates that block acceptance (in addition to the
standing Phase 12 gates, which apply to any new serve):

- G1: a MoE qwen35 GGUF (e.g. a Qwen3.6-35B-A3B-class file) loaded on both
  Sparks with per-rank expert shards, router replicated, per-expert
  combine folded into the accepted per-layer all-reduce sequence;
- G2: the Phase 10/11 probe discipline (eager-baseline bit-identity,
  promotion/drain/release/reuse) passing on the MoE checkpoint;
- G3: expert-streaming-under-TP behavior verified with
  `PADDOCK_MOE_HOST=1` on the pair: routed-expert planes host-mapped
  rank-locally, no cross-rank expert migration, drain-before-release
  preserved;
- G4: TP-vs-TP=1 MoE parity (T=0 token equality) and per-rank accounting
  cross-check including expert cache bytes.

## Phase 15 (speculative decoding) — blocked by unresolved topology measurement

What exists: TP=1 MTP/DFlash spec paths; the TP=2 gate refuses all spec
(`--no-spec` / `spec=off`, `mtp.is_none()`), and the TP rank loader drops
nextn/MTP blocks (`nextn_predict_layers` subtracted in `load_slots`).

Why implementation cannot honestly start host-side:

1. The plan resolves "target TP only vs target+draft TP; replicated vs
   single-rank draft; accepted-token synchronization; sampled acceptance
   semantics; graph/static-buffer interactions" — each alternative is a
   different implementation, and the plan's own selection criterion is the
   "simplest topology preserving existing user-facing behavior", which is a
   measured judgment, not a static one.
2. Any variant requires loading the checkpoint's MTP/nextn layer(s)
   rank-locally — a load path that has never executed anywhere and whose
   shape decisions can only be validated against the live checkpoint.

Exact target-device gates that block:

- G1: chosen-draft topology measured on the two-Spark pair (draft tokens/s
  vs decode tokens/s for at least the replicated-draft and single-rank-draft
  variants) BEFORE the non-measured variants are built;
- G2: acceptance/parity: sampled-acceptance stream equals TP=1 spec stream
  at T=0 on the pinned checkpoint;
- G3: graph interaction: Stage A graph replays compose with draft/verify
  sequencing on both ranks (Phase 11 Stage A is accepted for the non-spec
  path only);
- G4: drain-before-release/reset/reuse preserved with draft state in the
  slot lifecycle (probe discipline extension).

## What remains actionable without Sparks

- The Phase 12 deferred two-Spark gates (fp8 pair, f16 regression,
  accounting spot-check, wire-compat negative test) — first in queue when a
  Spark frees up.
- The Phase 14/15 gates above, in plan order, after their prerequisites
  (real checkpoint / topology measurement) exist.
- Nothing else in the plan sequence (Phases 12→15) is host-implementable
  without guessing at runtime or format behavior; the deferred TP=1
  baseline (plan §Deferred TP=1 baseline policy) also awaits hardware.
