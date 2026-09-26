//! Opt-in per-stage activation trace for the batched TP prefill span
//! prototype (`PADDOCK_TP_ABC_TRACE=1`; probe-only diagnostics).
//!
//! With the env unset every site is a zero-cost early return: no path's
//! behavior, kernel election, sequencing or numerics change in any way, and
//! CUDA graph capture stays legal (no stream ops are issued). With the env
//! set, each site reads back ONE row (row 0 unless a site says otherwise)
//! of the named tensor into a process-local buffer that `tp_trace::take()`
//! drains in host-call order (the B-vs-C probe partitions the drained rows
//! by stage-name prefix - `b.` for the span arm, `c.` for the trusted TP=1
//! prefill - because both arms append into the SAME buffer). The probe
//! diffs the two partitions stage by stage to localize the first material
//! divergence; nothing here feeds back into execution, alters a dispatch,
//! or weakens a guard. Readbacks
//! synchronize the stream, so a traced prefill must run eager (the probe
//! pins `PADDOCK_NO_PREFILL_GRAPH=1`) and no graph capture may be active.
use std::cell::RefCell;

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor};

struct TraceRow {
    #[allow(dead_code)] // read via take()'s mapping
    stage: String,
    #[allow(dead_code)]
    layer: usize,
    #[allow(dead_code)]
    data: Vec<f32>,
}

thread_local! {
    static TRACE: RefCell<Vec<TraceRow>> = const { RefCell::new(Vec::new()) };
}

/// Dev-switch gate: compiled out of hardened builds entirely (`dev_var_os!`),
/// so a hardened binary cannot even arm the trace.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_TP_ABC_TRACE").is_some())
}

/// Capture one row of `plane` at `row_off` (`len` elements) for
/// `stage`/`layer`. No-op unless the trace is armed. Synchronizes the
/// stream (see the module docs) - call sites sit between enqueue phases
/// whose ordering the caller already owns.
pub(crate) fn trace_row(
    e: &GpuExecutor,
    stage: &str,
    layer: usize,
    plane: &CudaSlice<f32>,
    row_off: usize,
    len: usize,
) -> Result<(), GpuError> {
    if !enabled() {
        return Ok(());
    }
    let view = plane.try_slice(row_off..row_off + len).ok_or_else(|| {
        GpuError::Driver(format!(
            "tp_trace {stage}[layer {layer}]: range {row_off}..{} outside plane of {}",
            row_off + len,
            plane.len()
        ))
    })?;
    e.stream.synchronize().map_err(GpuError::from)?;
    let data = e.stream.clone_dtoh(&view).map_err(GpuError::from)?;
    TRACE.with(|t| {
        t.borrow_mut().push(TraceRow {
            stage: stage.to_owned(),
            layer,
            data,
        })
    });
    Ok(())
}

/// Capture the LAST row of a row-batched plane (`rows` logical rows, `len`
/// elements each) for `stage`/`layer`. The row-0 [`trace_row`] sites cannot
/// see position-dependent stages: M-RoPE rotates row r with the row's own
/// position, and row 0 sits at position 0 in every probe case, where all
/// four axes agree regardless of staging. The paired `b.*-last` /
/// `c.*-last` readbacks capture the final row of the pass (row 15 in the
/// 16-row ABC case), which is where a text-position staging bug actually
/// materializes. Same no-op/sync contract as `trace_row`; fails closed on
/// a zero-row call.
pub(crate) fn trace_row_last(
    e: &GpuExecutor,
    stage: &str,
    layer: usize,
    plane: &CudaSlice<f32>,
    rows: usize,
    len: usize,
) -> Result<(), GpuError> {
    if rows == 0 {
        return Err(GpuError::Driver(format!(
            "tp_trace {stage}[layer {layer}]: trace_row_last called with zero rows"
        )));
    }
    trace_row(e, stage, layer, plane, (rows - 1) * len, len)
}

/// Drain the whole buffer in host-call order (each arm calls once, clearing
/// it), so a probe can diff two arms stage by stage. Probe-surface only
/// (used by the qwen35_tp_abc_probe example).
#[allow(dead_code)]
pub fn take() -> Vec<(String, usize, Vec<f32>)> {
    TRACE.with(|t| {
        std::mem::take(&mut *t.borrow_mut())
            .into_iter()
            .map(|r| (r.stage, r.layer, r.data))
            .collect()
    })
}
