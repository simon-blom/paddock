# Phase 13 — Broaden qwen35-family coverage

Status: IMPLEMENTATION COMPLETE (host-side scope), HOST VERIFICATION
COMPLETE, two-Spark validation DEFERRED. Most of this phase's checklist is
intrinsically target/model-bound: the only qwen35-family checkpoints that
exist locally are the pinned Qwen3.8-27B UD-Q4_K_M GGUF (already accepted)
and a Qwen3.8-Flash-Next NVFP4 safetensors tree that is a DIFFERENT
architecture (qwen4_exp, Phase 14 scope). No Sparks were available this
session.

Baseline: Phase 12 checkpoint `17efad0` (dtype plumbing + rank-local
accounting, host-verified).

## What the plan asks vs what is host-reachable

Plan items, one by one:

1. **Other qwen35 model sizes.** The TP lane is geometry-pinned at three
   layers, all loud:
   - backbone metadata: `hidden != 5120` refuses (`tp_model.rs`
     `load_slots`, "unsupported Qwen3.8 backbone metadata");
   - attention shape: `GqaGeometry::new` refuses any `head_dim != 256`,
     odd KV heads, incomplete Q/KV groups, or non-256-multiple width
     (`gqa_tp.rs:48-62`);
   - DeltaNet shape: `DeltaTpRank::load` pins `embedding_length=5120`,
     `ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=48`,
     `ssm.conv_kernel=4` against metadata, plus the F32-state requirement
     (`delta_tp.rs:276-290`).
   These pins are now under direct unit test (see host verification). Any
   future qwen35 size work means relaxing these pins against a real
   checkpoint plus kernel-coverage checks - NOT possible honestly without
   the model file and a two-Spark parity run. The architecture to do so is
   unchanged: `load_slots` reads all sizes from GGUF metadata except the
   three pins above.
   Also note `PINNED_SHA256` (tp_serve.rs:35): the serve-level gate pins the
   exact bring-up checkpoint by hash. Any new size needs this replaced by a
   per-checkpoint allowlist - a deliberate, auditable change to be made WITH
   the first new checkpoint, not speculatively.

2. **Other quantizations.** The load path accepts any tensor type with
   `kq_params` coverage or Q8_0 (`validate_dense_gemv_coverage`,
   tp_model.rs:1514+), and refuses anything else loudly with the tensor name
   ("{name}: {ty:?} has no validated quantized repack/dequant dispatch").
   The existing `mixed_quant_types_have_repack_coverage` test pins the IQ4XS/
   IQ4NL/IQ3S/Q3K dispatches. Quantizing the SAME checkpoint differently
   (e.g. a Q8_0 build) flows through the same path; it still needs a
   two-Spark parity run before acceptance, deferred like every target gate.

3. **Safetensors variants.** The TP lane reads GGUF only (`MappedGguf`); a
   safetensors qwen35 would require a mapper, not TP changes. Explicitly out
   of scope here per the plan's "where applicable" - no qwen35 safetensors
   checkpoint exists in this repo's world.

4. **Recurrent-state geometry edge cases.** DeltaGeometry's rank split is
   pinned by tests: value-head banding is complete and disjoint across
   ranks (all 48 heads covered exactly once), channel index sets partition
   0..10240 exactly, and both ranks' per-slot state footprints are
   identical by construction (so `context_mem_bytes` stays exact for any
   slot count). The state-size pin (S=128) also gates the fused vb16 kernel
   election (`dn_vb16` requires state_size==128) - a different S would take
   the eager path, still correct.

5. **Multimodal/qwen35-adjacent paths.** Not relevant: the local
   Flash-Next NVFP4 tree is qwen4_exp (MoE + vision + hyper-connections),
   which Phase 14 owns and which the TP gate already refuses
   (`mmproj.is_some()`, fp8_native, etc. in the runner gate). No code
   change justified.

## Implementation this phase

- `validate_dense_gemv_coverage` now refuses a MoE qwen35 checkpoint BY
  NAME ("this is an MoE qwen35 checkpoint (expert_count present); the TP
  lane serves dense backbones only") instead of the misleading
  "missing tensor blk.0.ffn_gate.weight" the old path produced for
  expert quartet files. This is the one real host-fixable gap the survey
  found: every other refusal already names its cause.
- New unit tests pinning the geometry/quant boundaries other sizes and
  quants will hit (gqa_tp.rs `head_dim_pin_is_exact_and_fp8_halves_kv_bytes`,
  delta_tp.rs `recurrent_state_slots_are_disjoint_and_reset_sized`).

## Host verification (complete)

- `cargo test -p paddock-engine --lib` — 460 passed, 0 failed (2 new tests
  over Phase 12's 458).
- `git diff --check` — clean.

## Deferred two-Spark validation (exact gates)

Nothing NEW is deferred by this phase's code changes (the MoE refusal is
host-verifiable: feed an MoE gguf to the TP gate on any machine and it must
exit 2 with the named error — reproducible without GPUs). The standing
deferred gates from Phase 12 apply unchanged, plus, WHEN a second qwen35
checkpoint/quantization ever lands:

- full Phase 10/11 probe discipline on the new checkpoint (eager baseline
  bit-identity, promotion/drain/release/reuse);
- fp8-vs-f16 and f16-vs-fp32-logit parity sampling on the new checkpoint;
- per-rank accounting cross-check (kv_mem_bytes vs measured pool bytes).

## Remaining limitations

- No other qwen35 size or quantization has ever run under TP in this repo;
  the pins are tested to refuse loudly, but "would work if unpinned" is NOT
  claimed and should not be assumed - the kernel pack instantiates
  256-wide-head qwen3.8 shapes only.
- `PINNED_SHA256` remains a single-checkpoint gate by design.
