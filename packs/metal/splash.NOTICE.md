# Splash Metal attribution

The validated cooperative-fragment traversal in `splash.metal` is adapted
from Inco AI's Splash, `runtime/metal/kernels/common/q4_mpp_tiles.h`. The
BF16 prefill store and conditional attention-rescaling work also drew on
`runtime/metal/kernels/prefill/linear_q4.metal` and
`runtime/metal/kernels/common/q8_attention_tile.h`, respectively. Revision:
`f58d36ddb046726adc8937ab67d07bdde015c0d6`:
https://github.com/incoai/splash

Copyright Inco AI. Licensed under Apache License 2.0; the full license is
included in the repository's `LICENSE-APACHE`.

Paddock modifications: compact traversal integrated into independently
partitioned quantization-group projections with shared input conversions,
multi-plane epilogues and arbitrary ragged output dimensions. BF16 attention
retains Paddock's page format, fixed interleaved partitions and cache arithmetic;
it does not adopt the reference's INT8 KV representation. This is not
a dependency on or a subprocess invocation of the Splash runtime.
