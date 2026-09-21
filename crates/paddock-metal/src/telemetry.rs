//! Process-local observations. No profiling encoders, extra GPU fences,
//! privileged helpers, or inference-thread locks. Command spans are elapsed
//! GPU time, NOT utilization: spans on different queues may overlap.
use objc2_metal::{MTLCommandBuffer, MTLDevice};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static COMMANDS: AtomicU64 = AtomicU64::new(0);
static GPU_NS: AtomicU64 = AtomicU64::new(0);
static LAST_NS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn activate() {
    // Diagnostic A/B switch, read only when creating a queue. Production is on.
    ACTIVE.store(
        std::env::var("PADDOCK_METAL_TELEMETRY").as_deref() != Ok("0"),
        Relaxed,
    );
}

fn span_ns(start: f64, end: f64) -> Option<u64> {
    (start.is_finite() && end.is_finite() && start > 0.0 && end >= start)
        .then_some(((end - start) * 1e9) as u64)
}

pub(crate) fn completed(command: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>) {
    if !ACTIVE.load(Relaxed) {
        return;
    }
    if command.error().is_some() {
        return;
    }
    if let Some(ns) = span_ns(command.GPUStartTime(), command.GPUEndTime()) {
        GPU_NS.fetch_add(ns, Relaxed);
        LAST_NS.store(ns, Relaxed);
        COMMANDS.fetch_add(1, Relaxed);
    }
}

/// The stats thread reads THIS process's device allocations, not global GPU
/// usage. All families (vision, embedding and speech too) share this boundary.
pub fn telemetry_snapshot() -> Option<serde_json::Value> {
    if !ACTIVE.load(Relaxed) {
        return None;
    }
    let device = objc2_metal::MTLCreateSystemDefaultDevice()?;
    let commands = COMMANDS.load(Relaxed);
    Some(serde_json::json!({
        "allocated_bytes": device.currentAllocatedSize(),
        "completed_commands": commands,
        "gpu_seconds_total": GPU_NS.load(Relaxed) as f64 / 1e9,
        "last_command_ms": (commands > 0).then(|| LAST_NS.load(Relaxed) as f64 / 1e6),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_spans_reject_missing_and_invalid_timestamps() {
        assert_eq!(span_ns(2.0, 2.25), Some(250_000_000));
        assert_eq!(span_ns(0.0, 0.0), None);
        assert_eq!(span_ns(2.0, 1.0), None);
        assert_eq!(span_ns(f64::NAN, 2.0), None);
        assert_eq!(span_ns(2.0, f64::INFINITY), None);
    }
}
