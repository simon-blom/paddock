# Qwen3.8 DeltaNet internal prefill profile

Date: 2026-09-28

This diagnostic splits the existing opt-in `PADDOCK_TP_PREFILL_PROFILE=1`
DeltaNet span timing without changing the default execution path. CUDA-event
stage transitions were added at the semantic boundaries available in the
existing code. Fused operations remain grouped and are named accordingly.

## Reproduction

- HEAD before this diagnostic: `d461615966b48bc231dc234d120bc69f9a315d11`
- Runner built and tested from the diagnostic tree: SHA-256
  `29113fa6c78cdde9eda818be9caad0c317e3d7a8124663898495e2821033dc75`
- CUDA pack: `c58133072ed1f339201168b5f5eb7ecf0056ad7edb251e8f1224554d0e9b321e`
- Model: `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`
- Qwen3.8-27B Q4_K_M, F16 KV, max context 65,536, max batch 2,
  `PADDOCK_UNIFIED=1`, speculation off
- Exact Hermes fixture: 22,130 prompt tokens
- TP2 span cap: 192; actual geometry: 116 full/wide spans plus the final
  short spans created by the existing prefix/cache flow. The measured request
  profile was isolated by subtracting the warm-wave cumulative record; the
  final cumulative record contained 22,130 request rows plus 296 warm rows.

The TP1 request returned HTTP 200 with 64 completion tokens and coherent text.
The TP2 request returned HTTP 200 with 64 completion tokens and coherent text.
The first generated token/output matched the validated baseline. TP2 rank 0
and rank 1 both emitted profile summaries with matching span/row geometry; the
worker exited cleanly. No CUDA, NCCL, KV/cache, checkpoint, or protocol errors
were observed.

## Stage comparison

All values are CUDA-event milliseconds converted to seconds. TP1 values are
from the 22,130-row unified-prefill request. TP2 values are rank-0 local
stage deltas for the same request. `Excess` is TP2 local minus TP1.

| DeltaNet stage | TP1 | TP2@192 local | TP2 / TP1 | Excess seconds |
|---|---:|---:|---:|---:|
| Input activation quantization | 0.001 | 0.109 | 109.0x | +0.108 |
| Input QKV projection | 1.811 | 1.463 | 0.808x | -0.348 |
| Conv/split/gate preparation | 1.759 | 13.380 | 7.61x | +11.621 |
| Gate projection | 1.071 | 0.772 | 0.721x | -0.299 |
| Recurrent DeltaNet kernel | 1.050 | 1.011 | 0.963x | -0.039 |
| Norm/gate elementwise | 0.319 | 0.127 | 0.398x | -0.192 |
| Output projection | 1.284 | 1.042 | 0.811x | -0.242 |
| Prelude/other local glue | 0.436 | 0.005 | 0.011x | -0.431 |
| **Internal-stage sum** | **7.730** | **17.909** | **2.317x** | **+10.179** |
| Existing DeltaNet local bucket | 7.702 | 17.999 | 2.337x | +10.297 |
| NCCL | 0 | 2.400 | — | +2.400 |

TP1 reports two adjacent preparation labels: `delta-conv-split-prep`
(1.544 s) and `delta-conv-split-gate-prep` (0.215 s); they are combined above
because TP2's implementation performs the corresponding convolution, split,
and gate-preparation sequence in one fused semantic region. TP2's measured
NCCL was 2.400 s in this diagnostic run versus 2.377 s in the earlier
validated run; this is a profiled diagnostic and is not a headline benchmark.

The internal-stage sum does not exactly equal the historical coarse bucket:
TP1 differs by 0.028 s and TP2 by 0.090 s. This is the expected diagnostic
boundary/measurement residual from replacing one coarse interval with several
CUDA-event intervals and from the existing inter-stage glue. The original
coarse totals remain recorded and are the reconciliation reference.

## Diagnosis

The `conv/split/gate preparation` region accounts for approximately 11.6 s
of the 10.3 s TP2 local-compute excess; the apparent overshoot relative to the
coarse bucket is the small stage-boundary residual above. Every other internal
stage is at least slightly faster on TP2, or contributes only milliseconds.
The recurrent kernel is not the culprit: it is approximately equal (1.050 s
TP1 versus 1.011 s TP2). Likewise, input QKV, gate projection, norm/gate, and
output projection do not explain the excess.

The likely cause is the TP2 span path's per-span DeltaNet convolution/split and
gate-preparation work over the wider 192-row geometry, rather than NCCL or the
recurrent state-update kernel. The code path keeps the recurrent kernel
separate, while the convolution and preparation operations are launched in the
`delta_tp` prefill loop and are grouped by the new diagnostic stage.

Recommended next optimization target: inspect and optimize the TP2
convolution/split/gate-preparation path, starting with dispatch/kernel choice
and launch shape at 192 rows. Do not optimize the recurrent kernel first; the
measurements do not support it. NCCL remains a separate secondary target.

No optimization was implemented in this pass.
