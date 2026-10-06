//! Qwen3.8 projection-prefill policy adapter.
//!
//! Generic TP owns scratch and orchestration. This adapter delegates to the
//! existing Qwen35 projection dispatch so measured kernel election remains
//! unchanged while the model-specific FFN/GQA code stops knowing scratch
//! layout details.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuExecutor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::tp::prefill::{ProjectionPrefillBackend, ProjectionStaging};

use super::ops::{prefill_ffn_down_any, prefill_mm_any, prefill_mm_pre_any, prefill_quant};

pub(crate) struct Qwen35PrefillBackend;

impl ProjectionPrefillBackend for Qwen35PrefillBackend {
    fn prepare(
        &self,
        exec: &GpuExecutor,
        staging: &mut ProjectionStaging,
        input: &CudaSlice<f32>,
        in_dim: usize,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        staging.validate(in_dim, rows)?;
        prefill_quant(
            exec,
            &mut staging.xq,
            &mut staging.xs,
            &mut staging.yq,
            input,
            in_dim,
            rows,
        )
    }

    fn project_prepared(
        &self,
        exec: &GpuExecutor,
        staging: &mut ProjectionStaging,
        weight: &QuantW,
        output: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        prefill_mm_pre_any(
            exec,
            weight,
            &staging.xq,
            &staging.xs,
            &staging.yq,
            &mut staging.xsums,
            &mut staging.ssums,
            &mut staging.skfix,
            output,
            rows,
        )
    }

    fn project(
        &self,
        exec: &GpuExecutor,
        staging: &mut ProjectionStaging,
        weight: &QuantW,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        prefill_mm_any(
            exec,
            weight,
            &mut staging.xq,
            &mut staging.xs,
            &mut staging.yq,
            &mut staging.xsums,
            &mut staging.ssums,
            &mut staging.skfix,
            input,
            output,
            rows,
        )
    }

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
    ) -> Result<(), GpuModelError> {
        staging.validate(ff, rows)?;
        prefill_ffn_down_any(
            exec,
            weight,
            &mut staging.xq,
            &mut staging.xs,
            &mut staging.yq,
            &mut staging.xsums,
            &mut staging.ssums,
            &mut staging.skfix,
            gate,
            up,
            output,
            ff,
            rows,
        )
    }
}
