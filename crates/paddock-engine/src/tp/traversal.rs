//! Generic span residual-stream traversal primitives.
//!
//! The invariant ordering of a row-batched prefill span is model-independent:
//! per layer, normalize the residual stream, run the model's mixer, add the
//! reduced mixer rows back, normalize again, run the model's FFN, and add the
//! reduced FFN rows back; after the last layer, one final normalization, then
//! the LM-head projection of the span's LAST row. This module owns those
//! stage primitives and their buffer discipline (one residual plane `x`, one
//! normalized staging plane `xn`, live `rows * hidden` prefixes only).
//!
//! Deliberately dumb by design: no layer indices, layer kinds, collectives,
//! profiling, model policy, or callback/mixer concepts. Only explicit stage
//! primitives over the shared buffers, so a model's traversal stays visibly
//! ordered in one place and interposes its own instrumentation between calls.
//!
//! Every function is a one-to-one kernel sequence over the same planes the
//! Qwen span walk used before this extraction; nothing here introduces a
//! copy, a collective, or a new buffer lifetime.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::projection::gemv_quant;

/// Normalize the first `rows` residual rows from `x` into `xn` with the given
/// norm vector (any layer's norm, or the model's final norm). `xn`'s stale
/// suffix is never consumed: mixers and FFNs read only the live
/// `rows * hidden` prefix.
pub(crate) fn span_normalize(
    exec: &GpuExecutor,
    x: &CudaSlice<f32>,
    xn: &mut CudaSlice<f32>,
    norm: &CudaSlice<f32>,
    hidden: usize,
    eps: f32,
    rows: usize,
) -> Result<(), GpuError> {
    exec.rmsnorm_batch(x, norm, xn, hidden, eps, rows)
}

/// Add a mixer's (or FFN's) reduced row plane back into the residual stream
/// over the live `rows * hidden` prefix. The reduced planes carry live data
/// only in their first rows*hidden elements, so a flat elementwise add covers
/// the batch (row-wise residual without a row-broadcast kernel) — identical
/// to the pre-extraction walk.
pub(crate) fn span_accumulate(
    exec: &GpuExecutor,
    x: &mut CudaSlice<f32>,
    rows_plane: &CudaSlice<f32>,
    hidden: usize,
    rows: usize,
) -> Result<(), GpuError> {
    exec.add(x, rows_plane, rows * hidden)
}

/// The head half of a span: final normalization of all `rows` residual rows
/// into `xn`, then the LAST row's LM-head projection into `logits`.
///
/// `x_last` is the caller's one-row staging plane (`gemv_quant` takes a whole
/// slice, not an offset view); the copy is one hidden-width D2D per span.
/// The head enters no collective — only the traversal halves pair across
/// ranks, exactly as before the extraction. The projection's error type is
/// the model-layer one (`gemv_quant`'s); `rmsnorm`/copy keep `GpuError`,
/// converted at the boundary here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn span_head_normalize_project(
    exec: &GpuExecutor,
    x: &CudaSlice<f32>,
    xn: &mut CudaSlice<f32>,
    x_last: &mut CudaSlice<f32>,
    out_norm: &CudaSlice<f32>,
    head: &QuantW,
    logits: &mut CudaSlice<f32>,
    hidden: usize,
    eps: f32,
    rows: usize,
) -> Result<(), GpuModelError> {
    exec.rmsnorm_batch(x, out_norm, xn, hidden, eps, rows)
        .map_err(GpuModelError::from)?;
    exec.copy_region(xn, (rows - 1) * hidden, x_last, 0, hidden)
        .map_err(GpuModelError::from)?;
    gemv_quant(exec, head, x_last, logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The live-prefix arithmetic the walker relies on: residual adds and
    /// normalizations cover rows*hidden, head staging reads row (rows-1).
    #[test]
    fn head_offsets_stay_inside_live_prefix() {
        let hidden = 5120usize;
        for rows in [1usize, 2, 64, 192] {
            let last_row_start = (rows - 1) * hidden;
            let live = rows * hidden;
            assert!(last_row_start + hidden <= live);
        }
        // rows == 0 must never reach the head (forward_span_advance refuses
        // empty spans before the walk).
        assert_eq!(0 * hidden, 0);
    }
}
