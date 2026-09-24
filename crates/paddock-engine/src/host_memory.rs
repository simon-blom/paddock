//! Read-only host-memory admission input. Swap is never counted as capacity.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    pub total: u64,
    pub available: u64,
}

#[cfg(target_os = "linux")]
pub fn sample() -> Option<Snapshot> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let read = |key: &str| {
        text.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name == key)
                .then(|| {
                    value
                        .split_whitespace()
                        .next()?
                        .parse::<u64>()
                        .ok()?
                        .checked_mul(1024)
                })
                .flatten()
        })
    };
    Some(Snapshot {
        total: read("MemTotal")?,
        available: read("MemAvailable")?,
    })
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
pub fn sample() -> Option<Snapshot> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let mut total = 0u64;
    let mut size = std::mem::size_of_val(&total);
    // SAFETY: fixed SDK types, valid writable outputs, read-only sysctl.
    if unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut total).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
        || size != 8
    {
        return None;
    }
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: release the host send right on every path.
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
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if result != libc::KERN_SUCCESS || page <= 0 {
        return None;
    }
    // Speculative file pages overlap free_count. Inactive anonymous pages and
    // compressor contents are NOT additional available physical memory.
    let reclaimable = u64::from(stats.free_count)
        .saturating_add(
            u64::from(stats.external_page_count).saturating_sub(stats.speculative_count.into()),
        )
        .saturating_add(stats.purgeable_count.into())
        .saturating_mul(page as u64);
    let unavailable = u64::from(stats.wire_count)
        .saturating_add(stats.compressor_page_count.into())
        .saturating_mul(page as u64);
    Some(Snapshot {
        total,
        available: reclaimable.min(total.saturating_sub(unavailable)),
    })
}

#[cfg(windows)]
pub fn sample() -> Option<Snapshot> {
    #[repr(C)]
    struct MemoryStatus {
        length: u32,
        load: u32,
        total: u64,
        available: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
    }
    let mut status: MemoryStatus = unsafe { std::mem::zeroed() };
    status.length = std::mem::size_of::<MemoryStatus>() as u32;
    // SAFETY: MEMORYSTATUSEX ABI, including its required length field.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(Snapshot {
        total: status.total,
        available: status.available,
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub fn sample() -> Option<Snapshot> {
    None
}

pub fn check(required: u64, available: u64, budget: Option<u64>) -> Result<(), String> {
    if let Some(budget) = budget
        && required > budget
    {
        return Err(format!(
            "model_budget_exceeded: requires {required} bytes, configured budget is {budget} bytes"
        ));
    }
    if required > available {
        return Err(format!(
            "insufficient_memory: requires {required} bytes, currently available {available} bytes; unload another model or retry when memory is available"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn budget_and_current_availability_are_independent() {
        assert!(
            super::check(100, 200, Some(99))
                .unwrap_err()
                .starts_with("model_budget_exceeded:")
        );
        assert!(
            super::check(100, 99, Some(200))
                .unwrap_err()
                .starts_with("insufficient_memory:")
        );
        assert!(super::check(100, 100, Some(100)).is_ok());
    }
    #[test]
    fn real_host_sample_is_bounded() {
        let value = super::sample().expect("supported host memory sample");
        assert!(value.total > 0 && value.available <= value.total);
    }
}
