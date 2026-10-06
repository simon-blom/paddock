//! Generic tensor-parallel projection staging.
//!
//! This owns the scratch geometry required by the shared quantized projection
//! ladders. It is intentionally model-agnostic: models decide which projection
//! policy/fast path to invoke, while scratch layout and capacity live here.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Model-facing prefill projection backend.
///
/// Generic TP owns orchestration and scratch. A model may override this backend
/// to keep an architecture-specific fast path without replacing FFN/GQA,
/// topology, collectives, or cache machinery.
pub(crate) trait ProjectionPrefillBackend {
    fn prepare(
        &self,
        exec: &GpuExecutor,
        staging: &mut ProjectionStaging,
        input: &CudaSlice<f32>,
        in_dim: usize,
        rows: usize,
    ) -> Result<(), GpuModelError>;

    fn project_prepared(
        &self,
        exec: &GpuExecutor,
        staging: &mut ProjectionStaging,
        weight: &QuantW,
        output: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuModelError>;

    fn project(
        &self,
        exec: &GpuExecutor,
        staging: &mut ProjectionStaging,
        weight: &QuantW,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuModelError>;

    fn swiglu_down(
        &self,
        exec: &GpuExecutor,
        staging: &mut ProjectionStaging,
        weight: &QuantW,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        ff: usize,
        rows: usize,
    ) -> Result<(), GpuModelError>;
}

/// Flat MMQ activation-layout sizing. Producer and consumer must use the same
/// padding rule or wide-prefill activations are silently misread.
pub(crate) fn mmq_layout(in_dim: usize, rows: usize) -> (usize, usize) {
    let cols = in_dim.div_ceil(128);
    let rows_pad = rows.next_multiple_of(128).max(128);
    (cols * rows_pad * 144, cols * rows_pad * 4)
}

/// Scratch shared by projections that consume one row-batched activation.
///
/// Both row-major and MMQ layouts are resident because dispatch is selected at
/// execution time from row count, weight type, and available kernels.
pub(crate) struct ProjectionStaging {
    cap: usize,
    max_in: usize,
    pub(crate) xq: CudaSlice<i8>,
    pub(crate) xs: CudaSlice<f32>,
    pub(crate) yq: CudaSlice<u8>,
    pub(crate) xsums: CudaSlice<f32>,
    pub(crate) ssums: CudaSlice<f32>,
    pub(crate) skfix: CudaSlice<f32>,
}

impl ProjectionStaging {
    pub(crate) fn new(
        exec: &GpuExecutor,
        max_in: usize,
        cap: usize,
    ) -> Result<Self, GpuError> {
        if max_in == 0 || cap == 0 {
            return Err(GpuError::Unsupported(
                "projection staging requires nonzero input width and row capacity".into(),
            ));
        }
        let (yq_elems, xsums_elems) = mmq_layout(max_in, cap);
        Ok(Self {
            cap,
            max_in,
            xq: exec.alloc_i8(cap * max_in)?,
            xs: exec.alloc(cap * max_in.div_ceil(32))?,
            yq: exec.alloc_u8(yq_elems)?,
            xsums: exec.alloc(xsums_elems)?,
            ssums: exec.alloc(cap * max_in.div_ceil(16))?,
            // Q8 MMQ stream-K fold scratch: 256 SMs x 128x128 tiles + flags.
            // The projection dispatcher ignores this plane for k-quant rungs.
            skfix: exec.alloc(256 * 128 * 128 + 256)?,
        })
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    pub(crate) fn max_in(&self) -> usize {
        self.max_in
    }

    pub(crate) fn validate(&self, in_dim: usize, rows: usize) -> Result<(), GpuError> {
        if rows == 0 || rows > self.cap || in_dim == 0 || in_dim > self.max_in {
            return Err(GpuError::Unsupported(format!(
                "projection staging capacity exceeded: rows={rows}/{} in_dim={in_dim}/{}",
                self.cap, self.max_in
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmq_layout_padding_is_monotonic_and_shared() {
        let a = mmq_layout(5120, 65);
        let b = mmq_layout(5120, 128);
        let c = mmq_layout(5120, 129);
        assert_eq!(a, b);
        assert!(c.0 > b.0);
        assert!(c.1 > b.1);
    }
}
