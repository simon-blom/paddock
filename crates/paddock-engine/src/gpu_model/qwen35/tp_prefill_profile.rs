//! Opt-in, per-rank CUDA-event timing for one TP prefill span. No events or
//! synchronization are created on the default path. Timings are device elapsed
//! times, not asynchronous host enqueue durations.
use std::cell::RefCell;
use std::collections::BTreeMap;

use cudarc::driver::{sys::CUevent_flags, CudaEvent, CudaStream, DevicePtr, DevicePtrMut};

use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::GpuError;

pub(super) fn enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ProfileError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
}

struct Interval {
    stage: &'static str,
    layer: Option<usize>,
    start: CudaEvent,
    end: CudaEvent,
}

#[derive(Default)]
struct Aggregate {
    spans: usize,
    rows: usize,
    reductions: [usize; 3],
    bytes: [usize; 3],
    stage_ms: BTreeMap<&'static str, f64>,
}

thread_local! {
    static AGGREGATE: RefCell<Aggregate> = RefCell::new(Aggregate::default());
    static SERIAL_AGGREGATE: RefCell<Aggregate> = RefCell::new(Aggregate::default());
}

/// Opt-in event profile for the single-rank batched prefill path. It shares
/// the same CUDA-event and stage-boundary semantics as `SpanProfile`, but has
/// no collective intervals and reports a separate TP1 line.
pub(crate) struct SerialProfile {
    rows: usize,
    intervals: Vec<Interval>,
    pending: Option<(&'static str, CudaEvent)>,
    whole_start: Option<CudaEvent>,
}

impl SerialProfile {
    pub(super) fn new(rows: usize) -> Self {
        Self { rows, intervals: Vec::new(), pending: None, whole_start: None }
    }

    pub(super) fn begin_whole(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.whole_start = Some(SpanProfile::event(stream)?);
        Ok(())
    }

    pub(super) fn stage(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
    ) -> Result<(), GpuError> {
        if let Some((stage, start)) = self.pending.take() {
            self.intervals.push(Interval {
                stage,
                layer: None,
                start,
                end: SpanProfile::event(stream)?,
            });
        }
        self.pending = Some((name, SpanProfile::event(stream)?));
        Ok(())
    }

    pub(super) fn finish(mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if let Some((stage, start)) = self.pending.take() {
            self.intervals.push(Interval {
                stage,
                layer: None,
                start,
                end: SpanProfile::event(stream)?,
            });
        }
        if let Some(start) = self.whole_start.take() {
            self.intervals.push(Interval {
                stage: "whole-prefill",
                layer: None,
                start,
                end: SpanProfile::event(stream)?,
            });
        }
        let terminal = SpanProfile::event(stream)?;
        terminal.synchronize().map_err(GpuError::from)?;
        let mut stage_ms = BTreeMap::<&'static str, f64>::new();
        for interval in &self.intervals {
            let ms = interval.start.elapsed_ms(&interval.end).map_err(GpuError::from)? as f64;
            *stage_ms.entry(interval.stage).or_default() += ms;
        }
        SERIAL_AGGREGATE.with(|state| {
            let mut total = state.borrow_mut();
            total.spans += 1;
            total.rows += self.rows;
            for (stage, ms) in &stage_ms {
                *total.stage_ms.entry(stage).or_default() += ms;
            }
            eprintln!(
                "[TP1-PREFILL-PROFILE] rows={} stage_ms={stage_ms:?} total_chunks={} total_rows={} total_stage_ms={:?}",
                self.rows, total.spans, total.rows, total.stage_ms
            );
        });
        Ok(())
    }
}

pub(crate) struct SpanProfile {
    rank: usize,
    rows: usize,
    intervals: Vec<Interval>,
    pending: Option<(&'static str, Option<usize>, CudaEvent)>,
    whole_start: Option<CudaEvent>,
    reductions: [usize; 3],
    bytes: [usize; 3],
}

impl SpanProfile {
    pub(super) fn new(rank: usize, rows: usize) -> Self {
        Self {
            rank,
            rows,
            intervals: Vec::new(),
            pending: None,
            whole_start: None,
            reductions: [0; 3],
            bytes: [0; 3],
        }
    }

    fn account(&mut self, kind: usize, len: usize) {
        self.reductions[kind] += 1;
        self.bytes[kind] += len * std::mem::size_of::<f32>();
    }

    fn event(stream: &CudaStream) -> Result<CudaEvent, GpuError> {
        let event = stream
            .context()
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(GpuError::from)?;
        event.record(stream).map_err(GpuError::from)?;
        Ok(event)
    }

    pub(super) fn begin_whole(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.whole_start = Some(Self::event(stream)?);
        Ok(())
    }

    /// Transition between adjacent stages on the compute stream. A stage
    /// only starts after the preceding one has been closed at the same event.
    pub(super) fn stage(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        layer: Option<usize>,
    ) -> Result<(), GpuError> {
        if let Some((stage, index, start)) = self.pending.take() {
            self.intervals.push(Interval {
                stage,
                layer: index,
                start,
                end: Self::event(stream)?,
            });
        }
        self.pending = Some((name, layer, Self::event(stream)?));
        Ok(())
    }

    /// Counts the f32 partial submitted to NCCL: the live rows*hidden view
    /// on the span paths (the capacity plane on decode paths).
    pub(super) fn reduce<C: Communicator, S: DevicePtr<f32>, R: DevicePtrMut<f32>>(
        &mut self,
        compute: &CudaStream,
        group: &C,
        src: &S,
        dst: &mut R,
        kind: usize,
        layer: usize,
    ) -> Result<(), ProfileError> {
        if let Some((stage, index, start)) = self.pending.take() {
            self.intervals.push(Interval {
                stage,
                layer: index,
                start,
                end: Self::event(compute)?,
            });
        }
        let wait_start = Self::event(compute)?;
        group.after_compute(compute)?;
        let stream = group.profiling_stream().ok_or_else(|| {
            CollectiveError::InvalidGroup(
                "TP prefill profiler requires a NCCL timing stream".into(),
            )
        })?;
        let nccl_start = Self::event(stream)?;
        group.all_reduce(src, dst)?;
        self.intervals.push(Interval {
            stage: ["gqa-nccl", "delta-nccl", "ffn-nccl"][kind],
            layer: Some(layer),
            start: nccl_start,
            end: Self::event(stream)?,
        });
        group.before_compute(compute)?;
        self.intervals.push(Interval {
            stage: ["gqa-wait", "delta-wait", "ffn-wait"][kind],
            layer: Some(layer),
            start: wait_start,
            end: Self::event(compute)?,
        });
        self.account(kind, src.len());
        Ok(())
    }

    pub(super) fn finish(mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if let Some((stage, layer, start)) = self.pending.take() {
            self.intervals.push(Interval {
                stage,
                layer,
                start,
                end: Self::event(stream)?,
            });
        }
        if let Some(start) = self.whole_start.take() {
            self.intervals.push(Interval {
                stage: "whole-advance",
                layer: None,
                start,
                end: Self::event(stream)?,
            });
        }
        let terminal = Self::event(stream)?;
        // Exactly one enabled-only synchronization at the span boundary;
        // all event pairs stay alive through it (including the NCCL stream).
        terminal.synchronize().map_err(GpuError::from)?;
        let mut layer_ms = BTreeMap::<(&'static str, Option<usize>), f64>::new();
        let mut stage_ms = BTreeMap::<&'static str, f64>::new();
        for interval in &self.intervals {
            let ms = interval
                .start
                .elapsed_ms(&interval.end)
                .map_err(GpuError::from)? as f64;
            *layer_ms
                .entry((interval.stage, interval.layer))
                .or_default() += ms;
            *stage_ms.entry(interval.stage).or_default() += ms;
        }
        AGGREGATE.with(|state| {
            let mut total = state.borrow_mut();
            total.spans += 1;
            total.rows += self.rows;
            for kind in 0..3 {
                total.reductions[kind] += self.reductions[kind];
                total.bytes[kind] += self.bytes[kind];
            }
            for (stage, ms) in &stage_ms {
                *total.stage_ms.entry(stage).or_default() += ms;
            }
            if total.spans == 1 {
                eprintln!("[TP-PREFILL-PROFILE-LAYERS] rank={} layer_ms={layer_ms:?}", self.rank);
            }
            eprintln!(
                "[TP-PREFILL-PROFILE] rank={} rows={} reductions(gqa,delta,ffn)={:?} bytes(gqa,delta,ffn)={:?} stage_ms={stage_ms:?} total_spans={} total_rows={} total_reductions={:?} total_bytes={:?} total_stage_ms={:?}",
                self.rank, self.rows, self.reductions, self.bytes,
                total.spans, total.rows, total.reductions, total.bytes, total.stage_ms
            );
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opt_in_is_exact() {
        assert!(!enabled(None));
        assert!(!enabled(Some("0")));
        assert!(!enabled(Some("true")));
        assert!(enabled(Some("1")));
    }

    #[test]
    fn counts_capacity_bytes_by_kind() {
        let mut profile = SpanProfile::new(1, 3);
        profile.account(0, 16);
        profile.account(1, 32);
        profile.account(1, 32);
        profile.account(2, 16);
        assert_eq!(profile.rows, 3);
        assert_eq!(profile.rank, 1);
        assert_eq!(profile.reductions, [1, 2, 1]);
        assert_eq!(profile.bytes, [16 * 4, 64 * 4, 16 * 4]);
    }
}
