//! Unified memory accounting. Metal allocations and CPU prefix caches consume
//! the SAME physical RAM. Never present process-local currentAllocatedSize as
//! machine-wide GPU usage. Free/purgeable pages are a conservative availability
//! sample, not a promise that inactive anonymous pages can be reclaimed for free.
use crate::routes::AppState;

// Only the macOS sampler reads it; off-Mac `sample()` is the `None` stub, and
// an ungated constant is dead code under `-D warnings` there.
#[cfg(target_os = "macos")]
const HEADROOM: u64 = 1 << 30;

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub name: String,
    pub physical: u64,
    pub limit: u64,
    pub available: u64,
}

#[cfg(target_os = "macos")]
pub fn physical_bytes() -> Option<u64> {
    let mut value = 0u64;
    let mut size = std::mem::size_of_val(&value);
    // SAFETY: correctly sized writable output; a read-only, NUL-terminated key.
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && size == 8 && value > 0).then_some(value)
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc exposes mach_task_self through the SDK's legacy binding.
pub fn sample() -> Option<Snapshot> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    use objc2_metal::MTLDevice;
    let device = objc2_metal::MTLCreateSystemDefaultDevice()?;
    if !device.hasUnifiedMemory() {
        return None;
    }
    let physical = physical_bytes()?;
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: host_statistics64 receives the SDK struct and its integer count.
    // Release the send right from mach_host_self on every path.
    let result = unsafe {
        let host = libc::mach_host_self();
        let result = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&raw mut stats).cast(),
            &mut count,
        );
        mach_port_deallocate(libc::mach_task_self(), host);
        result
    };
    if result != libc::KERN_SUCCESS {
        return None;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return None;
    }
    // speculative_count is already included in free_count (Apple's SDK).
    let available = (u64::from(stats.free_count) + u64::from(stats.purgeable_count)) * page as u64;
    Some(Snapshot {
        name: device.name().to_string(),
        physical,
        limit: (device.recommendedMaxWorkingSetSize() * 9 / 10)
            .min(physical.saturating_sub(HEADROOM)),
        available: available.min(physical).saturating_sub(HEADROOM),
    })
}

#[cfg(not(target_os = "macos"))]
pub fn sample() -> Option<Snapshot> {
    None
}

/// Reservations already backed by a live allocation must not be subtracted
/// from the OS's free-page sample a second time. Reserve only their unused
/// headroom there, while the fleet ceiling counts the entire commitment.
pub fn residual(
    snapshot: &Snapshot,
    device_reserved: u64,
    device_used: u64,
    host_reserved: u64,
) -> u64 {
    snapshot
        .limit
        .saturating_sub(device_reserved)
        .saturating_sub(host_reserved)
        .min(
            snapshot
                .available
                .saturating_sub(device_reserved.saturating_sub(device_used))
                .saturating_sub(host_reserved),
        )
}

pub async fn available(state: &AppState, freeing: Option<u16>) -> Option<(Snapshot, u64)> {
    let mut snapshot = sample()?;
    let views = state.supervisor.list().await;
    let recon = state.recon.borrow().clone();
    let mut reserved = 0u64;
    let mut used = 0u64;
    let mut host = 0u64;
    for port in views
        .iter()
        .map(|v| v.port)
        .chain(state.supervisor.spawning_ports())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let live = recon
            .as_ref()
            .as_ref()
            .and_then(|r| r.runners.iter().find(|r| r.port == port))
            .and_then(|r| r.self_mem)
            .unwrap_or(0);
        if Some(port) == freeing {
            snapshot.available = snapshot
                .available
                .saturating_add(live)
                .min(snapshot.physical);
            continue;
        }
        // A deferred save is not the configuration the running process loaded.
        // Ask its own baseline (which survives manager restarts) before trusting
        // a smaller saved ceiling. Unknown/drifted runners reserve the entire
        // backend ceiling until applied; never resell their live headroom.
        let client = paddock_admin::client::AdminClient::new(port);
        let matches =
            tokio::time::timeout(std::time::Duration::from_secs(1), client.config_status())
                .await
                .ok()
                .and_then(Result::ok)
                .is_some_and(|s| {
                    s.restart_required == Some(false)
                        && views.iter().any(|v| v.port == port && v.pid == s.pid)
                });
        let budget = reservation(
            &snapshot,
            state.supervisor.config_vram_budget(port),
            matches,
        );
        reserved = reserved.saturating_add(budget.max(live));
        used = used.saturating_add(live);
        if let Ok(spec) = state
            .supervisor
            .spec_from_config_file(&state.supervisor.server_config_path(port))
            && let Some(kv) = spec.kv_offload.filter(|kv| kv.enabled)
        {
            host = host.saturating_add((kv.ram_gb.max(0.0) * (1u64 << 30) as f64) as u64);
        }
    }
    let free = residual(&snapshot, reserved, used, host);
    Some((snapshot, free))
}

fn reservation(snapshot: &Snapshot, budget_mib: Option<u64>, baseline_matches: bool) -> u64 {
    budget_mib
        .filter(|_| baseline_matches)
        .and_then(|m| m.checked_mul(1 << 20))
        .unwrap_or(snapshot.limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservations_share_ram_without_double_charging_live_gpu_bytes() {
        let s = Snapshot {
            name: "test".into(),
            physical: 100,
            limit: 90,
            available: 60,
        };
        assert_eq!(residual(&s, 40, 30, 0), 50);
        assert_eq!(residual(&s, 40, 30, 15), 35);
        assert_eq!(residual(&s, 100, 30, 0), 0);
        assert_eq!(residual(&s, 10, 10, 100), 0);
        assert_eq!(reservation(&s, Some(1), true), 1 << 20);
        assert_eq!(reservation(&s, Some(1), false), s.limit);
        assert_eq!(reservation(&s, None, true), s.limit);
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn apple_silicon_reports_real_memory() {
        let s = sample().unwrap();
        assert!(s.physical > 1 << 30 && s.limit <= s.physical && s.available <= s.physical);
    }
}
