//! Shared model projection primitives.
//!
//! Keep quant/kernel dispatch here when it is independent of a specific model
//! architecture. Model-specific modules may wrap these functions for API
//! compatibility, but generic TP code must not depend back on a model family.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuExecutor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Decode GEMV with per-tensor dispatch.
///
/// Q8_0 uses the repacked Q8 path; K/I-quant weights use the dtype-aware
/// k-quant GEMV. This is model-independent dispatch over `QuantW`.
pub(crate) fn gemv_quant(
    exec: &GpuExecutor,
    w: &QuantW,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => exec.q8_0_gemv_repacked(q, None, x, y)?,
        QuantW::Kq(k) => exec.kquant_gemv(k, x, y)?,
    }
    Ok(())
}
