//! Documented, unprivileged OS signals. A dispatch memory-pressure source
//! reports transitions, not an initial query. Until an event arrives the value
//! is unknown; free RAM percentages are deliberately not used as a substitute.
use block2::RcBlock;
use dispatch2::{DispatchObject, DispatchRetained, DispatchSource};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering::Relaxed},
};

pub(super) struct Pressure {
    source: DispatchRetained<DispatchSource>,
    memory: Arc<AtomicUsize>,
}

impl Pressure {
    pub(super) fn new() -> Self {
        let memory = Arc::new(AtomicUsize::new(0));
        // SAFETY: public memory-pressure type, no handle, all three public
        // event flags, dispatch's default queue. Configure before activation.
        let source = unsafe {
            DispatchSource::new(
                (&raw const dispatch2::_dispatch_source_type_memorypressure).cast_mut(),
                0,
                1 | 2 | 4,
                None,
            )
        };
        let address = (&*source as *const DispatchSource) as usize;
        let state = memory.clone();
        let handler = RcBlock::new(move || {
            // SAFETY: dispatch retains a source during its event handler;
            // cancel waits for an in-flight handler before final destruction.
            // Capture a non-owning pointer to avoid a source/block retain cycle.
            let data = unsafe { &*(address as *const DispatchSource) }.data();
            state.store(data, Relaxed);
        });
        // SAFETY: block owns its Arc, reads source data only from its handler;
        // dispatch copies the block before this local reference is dropped.
        unsafe { source.set_event_handler_with_block(RcBlock::as_ptr(&handler)) };
        source.activate();
        Self { source, memory }
    }

    pub(super) fn memory(&self) -> Option<&'static str> {
        memory_label(self.memory.load(Relaxed))
    }
}

impl Drop for Pressure {
    fn drop(&mut self) {
        self.source.cancel();
    }
}

fn memory_label(flags: usize) -> Option<&'static str> {
    // Coalesced transitions preserve the most severe reported condition.
    if flags & 4 != 0 {
        Some("critical")
    } else if flags & 2 != 0 {
        Some("warning")
    } else if flags & 1 != 0 {
        Some("normal")
    } else {
        None
    }
}

pub(super) fn thermal() -> Option<&'static str> {
    thermal_label(
        objc2_foundation::NSProcessInfo::processInfo()
            .thermalState()
            .0,
    )
}

fn thermal_label(state: isize) -> Option<&'static str> {
    match state {
        0 => Some("nominal"),
        1 => Some("fair"),
        2 => Some("serious"),
        3 => Some("critical"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_and_future_states_never_become_normal() {
        assert_eq!(memory_label(0), None);
        assert_eq!(memory_label(8), None);
        assert_eq!(memory_label(1), Some("normal"));
        assert_eq!(memory_label(3), Some("warning"));
        assert_eq!(memory_label(7), Some("critical"));
        assert_eq!(thermal_label(0), Some("nominal"));
        assert_eq!(thermal_label(1), Some("fair"));
        assert_eq!(thermal_label(2), Some("serious"));
        assert_eq!(thermal_label(3), Some("critical"));
        assert_eq!(thermal_label(4), None);
    }
    #[test]
    fn repeated_subscription_shutdown_is_safe() {
        for _ in 0..50 {
            drop(Pressure::new());
        }
        assert!(thermal().is_some());
    }

    #[test]
    fn cancellation_releases_callback_state_without_a_retain_cycle() {
        let pressure = Pressure::new();
        let state = Arc::downgrade(&pressure.memory);
        drop(pressure);
        for _ in 0..100 {
            if state.upgrade().is_none() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("dispatch cancellation retained the pressure observer");
    }
}
