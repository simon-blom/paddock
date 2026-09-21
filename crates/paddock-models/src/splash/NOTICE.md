# Splash package interoperability

The schema-3 reader implements the format documented by Inco AI's Splash,
source revision `f58d36ddb046726adc8937ab67d07bdde015c0d6`:
https://github.com/incoai/splash

Format references: `runtime/model/WeightStore.cpp`, `QwenTarget.hpp`,
`QwenTarget.cpp`, `Qwen3_8.hpp`, `DFlashDraft.cpp` and `QwenVision.cpp`.
Splash is Copyright Inco AI, Apache-2.0. This is an independently written
Rust interoperability reader, not a vendored copy of the Splash runtime.

The initial reader deliberately pins the audited manifest rather than guessing
the geometry or accepting arbitrary packages. Package storage is distinct from
runtime arithmetic and performance qualification.
