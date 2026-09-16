//! Memory a unified-memory die can really use, beyond what the OS reports.
//!
//! On an integrated die (DGX Spark GB10, Jetson) "free VRAM" is the OS's
//! MemAvailable (`GpuExecutor::device_mem_info` explains why). That figure has
//! one blind spot big enough to break sizing: the NVIDIA driver (open kernel
//! module >= 590, system-memory pools) keeps the pages an exiting CUDA process
//! frees in its own pools. /proc/meminfo books them as used under no category,
//! MemAvailable leaves them out, and the next CUDA allocation is served from
//! them first.
//!
//! Measured on a Spark on 2026-09-13, Qwen3.8-27B serving through the manager:
//! the first start planned KV against a 76 GiB grant; after one stop/start the
//! grant was 37 GiB, and `max_ctx 65536` refused to start ("needs 126.50 GiB
//! of KV, only 29.92 GiB fits"). No other process held memory - the container
//! held 1.35 GiB of anonymous memory and 31 GiB of page cache - yet
//! MemAvailable read 40 GiB and cuMemGetInfo 2.3 GiB, while a 96 GiB CUDA
//! allocation succeeded. Every restart shrinks the next grant by whatever the
//! last runner left in the pool.
//!
//! The load gate already asks the driver with a trial allocation before it
//! refuses weights. The sizers need an amount rather than a yes, so this
//! measures the reusable pool: allocate in chunks, write each one so its pages
//! really exist, and after every chunk read MemAvailable. A chunk the pool
//! served leaves MemAvailable where it was; a chunk the kernel served lowers it
//! by the chunk's size. Credit is what MemAvailable did not pay for, and the
//! walk stops at the first chunk the kernel mostly paid, so the probe evicts at
//! most about one chunk of page cache. Everything is freed straight after -
//! back into the same pool, where the load's own allocations then find it.
//!
//! Two bounds keep a wrong reading from over-committing the box. The credit
//! can never exceed the memory /proc/meminfo books under no category (minus
//! this process's own reserved pool and a slack for driver and firmware
//! allocations), and it is consumed as this process allocates: each byte our
//! pool gains since the probe is taken off the credit, because the driver
//! serves those allocations from its pool first. If the kernel served them
//! instead, MemAvailable has already dropped and the credit is taken off twice
//! - an undercount, which is today's behaviour, never an overcount.

/// /proc/meminfo, in bytes - only the fields the accounting reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MemInfo {
    pub total: u64,
    pub free: u64,
    pub available: u64,
    buffers: u64,
    cached: u64,
    swap_cached: u64,
    anon: u64,
    slab: u64,
    kernel_stack: u64,
    page_tables: u64,
    sec_page_tables: u64,
    percpu: u64,
    hugetlb: u64,
    huge_total: u64,
    huge_free: u64,
    huge_size: u64,
}

impl MemInfo {
    /// None without MemTotal and MemAvailable - a kernel that old gives the
    /// caller no honest number to start from.
    pub(super) fn parse(text: &str) -> Option<MemInfo> {
        let mut m = MemInfo::default();
        let (mut has_total, mut has_avail) = (false, false);
        // `Slab` is the sum of these two; kept as a fallback for a reading
        // that lacks the total line
        let (mut slab_total, mut s_reclaimable, mut s_unreclaim) = (None, 0u64, 0u64);
        for line in text.lines() {
            let Some((key, rest)) = line.split_once(':') else {
                continue;
            };
            let v: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let kb = v * 1024;
            match key {
                "MemTotal" => (m.total, has_total) = (kb, true),
                "MemFree" => m.free = kb,
                "MemAvailable" => (m.available, has_avail) = (kb, true),
                "Buffers" => m.buffers = kb,
                "Cached" => m.cached = kb,
                "SwapCached" => m.swap_cached = kb,
                "AnonPages" => m.anon = kb,
                "Slab" => slab_total = Some(kb),
                "SReclaimable" => s_reclaimable = kb,
                "SUnreclaim" => s_unreclaim = kb,
                "KernelStack" => m.kernel_stack = kb,
                "PageTables" => m.page_tables = kb,
                "SecPageTables" => m.sec_page_tables = kb,
                "Percpu" => m.percpu = kb,
                "Hugetlb" => m.hugetlb = kb,
                "HugePages_Total" => m.huge_total = v, // a count, not kB
                "HugePages_Free" => m.huge_free = v,
                "Hugepagesize" => m.huge_size = kb,
                _ => {}
            }
        }
        m.slab = slab_total.unwrap_or(s_reclaimable + s_unreclaim);
        (has_total && has_avail).then_some(m)
    }

    /// None off Linux: no integrated NVIDIA die exists anywhere else today.
    pub(super) fn read() -> Option<MemInfo> {
        if !cfg!(target_os = "linux") {
            return None;
        }
        MemInfo::parse(&std::fs::read_to_string("/proc/meminfo").ok()?)
    }

    /// A box with hugetlb pages configured: CUDA can then only use those, so
    /// they are the whole answer (NVIDIA's DGX Spark snippet reads it the
    /// same way), and the driver-pool credit does not apply.
    pub(super) fn hugetlb_configured(&self) -> bool {
        self.huge_total > 0 && self.huge_size > 0
    }

    /// What a CUDA allocation may take per the OS: MemAvailable, the reading
    /// NVIDIA publishes for the DGX Spark and the one llama.cpp, vLLM and
    /// SGLang all take - or the free huge pages on a hugetlb box.
    pub(super) fn usable(&self) -> u64 {
        if self.hugetlb_configured() {
            self.huge_free * self.huge_size
        } else {
            self.available
        }
    }

    /// Bytes the kernel counts as used but files under none of its own
    /// categories: driver allocations, which on a unified-memory die means
    /// live CUDA memory plus the driver's retained pools. The upper bound on
    /// anything the probe may credit. `Cached` already includes shmem.
    pub(super) fn unaccounted(&self) -> u64 {
        self.total
            .saturating_sub(self.free)
            .saturating_sub(self.buffers)
            .saturating_sub(self.cached)
            .saturating_sub(self.swap_cached)
            .saturating_sub(self.anon)
            .saturating_sub(self.slab)
            .saturating_sub(self.kernel_stack)
            .saturating_sub(self.page_tables)
            .saturating_sub(self.sec_page_tables)
            .saturating_sub(self.percpu)
            .saturating_sub(self.hugetlb)
    }
}

/// Probe chunk. Bounds the page cache one probe can evict (the chunk the
/// kernel paid for, at most) and the time it takes (a 2 GiB write is quick
/// on any of these dies).
pub(super) const PROBE_CHUNK: u64 = 2 << 30;

/// Left out of the credit bound for allocations the driver and firmware hold
/// that are neither ours nor reusable: the CUDA context, GSP heaps, display.
pub(super) const DRIVER_SLACK: u64 = 2 << 30;

/// The reusable pool, proven once, then spent as this process allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RetainedCredit {
    /// Bytes the probe's allocations drew without lowering MemAvailable.
    pub proven: u64,
    /// This process's pool usage when the probe ran.
    pub ours_at_probe: u64,
}

impl RetainedCredit {
    /// Credit still standing: what we have allocated since the probe came out
    /// of the pool first.
    pub(super) fn remaining(&self, ours_now: u64) -> u64 {
        self.proven
            .saturating_sub(ours_now.saturating_sub(self.ours_at_probe))
    }
}

/// Measure how much of an allocation of up to `cap` bytes the driver's pool
/// serves. `available` reads MemAvailable, `take` allocates-and-writes one
/// chunk (None = refused), `give_back` frees one. Kept free of CUDA so the
/// walk is testable on any box; `GpuExecutor::retained_headroom` supplies the
/// real calls.
pub(super) fn measure_retained<H>(
    cap: u64,
    chunk: u64,
    mut available: impl FnMut() -> Option<u64>,
    mut take: impl FnMut(u64) -> Option<H>,
    mut give_back: impl FnMut(H),
) -> u64 {
    if cap == 0 || chunk == 0 {
        return 0;
    }
    let Some(mut before) = available() else {
        return 0;
    };
    let mut held = Vec::new();
    let (mut drawn, mut credit) = (0u64, 0u64);
    while drawn < cap {
        let size = chunk.min(cap - drawn);
        let Some(h) = take(size) else {
            break;
        };
        held.push(h);
        drawn += size;
        let Some(now) = available() else {
            break;
        };
        // what the kernel's own memory paid for this chunk
        let paid = before.saturating_sub(now);
        before = now;
        credit += size.saturating_sub(paid);
        if paid > size / 2 {
            // the pool has run dry: every further chunk would evict page cache
            break;
        }
    }
    for h in held {
        give_back(h);
    }
    credit.min(cap)
}

impl super::GpuExecutor {
    /// Driver-retained bytes this process may still count as free on a
    /// unified-memory die, on top of MemAvailable. Measured the first time a
    /// sizer finds MemAvailable binding, capped at `cap` (the most any sizer
    /// could take, so the probe never walks further than could change an
    /// answer), then spent as our pool grows. Must run on the engine thread,
    /// like every sizer that reads `vram_headroom`.
    pub(super) fn retained_headroom(&self, cap: u64) -> u64 {
        let ours = self.process_mem_used().unwrap_or(0);
        let mut slot = self
            .retained
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(c) = *slot {
            return c.remaining(ours);
        }
        let proven = match MemInfo::read() {
            Some(info) if !info.hugetlb_configured() => {
                let bound = info
                    .unaccounted()
                    .saturating_sub(self.pool_reserved_bytes().unwrap_or(0))
                    .saturating_sub(DRIVER_SLACK);
                let t0 = std::time::Instant::now();
                let proven = measure_retained(
                    cap.min(bound),
                    PROBE_CHUNK,
                    || MemInfo::read().map(|m| m.available),
                    |size| {
                        // SAFETY: a fresh allocation on this thread's current
                        // context, written only within its own length, and
                        // freed exactly once - here on a failed write, or by
                        // the give-back below.
                        unsafe {
                            let ptr = cudarc::driver::result::malloc_sync(size as usize).ok()?;
                            // written, so its pages exist: an allocation the
                            // driver only reserved would read as pool-served
                            if cudarc::driver::result::memset_d8_sync(ptr, 0, size as usize)
                                .is_err()
                            {
                                let _ = cudarc::driver::result::free_sync(ptr);
                                return None;
                            }
                            Some(ptr)
                        }
                    },
                    |ptr| {
                        // SAFETY: `ptr` came from the malloc_sync above and
                        // has not been freed.
                        if let Err(e) = unsafe { cudarc::driver::result::free_sync(ptr) } {
                            tracing::warn!(error = %e, "retained-pool probe: free failed");
                        }
                    },
                );
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                tracing::info!(
                    reusable_gib = gib(proven),
                    available_gib = gib(info.available),
                    unaccounted_gib = gib(info.unaccounted()),
                    probe_ms = t0.elapsed().as_millis() as u64,
                    "unified-memory die: memory the driver kept from earlier CUDA processes \
                     measured as reusable - counted as free on top of MemAvailable"
                );
                proven
            }
            _ => 0,
        };
        *slot = Some(RetainedCredit {
            proven,
            ours_at_probe: ours,
        });
        proven
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The lines read off the Spark's /proc/meminfo with no runner up, after
    /// the stop that led to the 37 GiB grant (2026-09-13). Only these were
    /// captured, which is also why `Slab` is missing here.
    const SPARK_AFTER_STOP: &str = "\
MemTotal:       127600816 kB
MemFree:         3139656 kB
MemAvailable:   40174336 kB
Buffers:           13252 kB
Cached:         37694160 kB
Unevictable:       24640 kB
Mlocked:           24640 kB
AnonPages:       1712244 kB
Shmem:              3028 kB
KReclaimable:     477572 kB
SUnreclaim:       520360 kB
NFS_Unstable:          0 kB
VmallocUsed:      101620 kB
Percpu:            27520 kB
CmaTotal:         131072 kB
CmaFree:           69320 kB
HugePages_Total:       0
Hugetlb:               0 kB
";

    #[test]
    fn the_spark_reading_books_the_driver_pool_under_no_category() {
        let m = MemInfo::parse(SPARK_AFTER_STOP).unwrap();
        assert_eq!(m.usable(), 40174336 * 1024);
        // ~80 GiB the kernel counts as used and cannot name - the retained
        // pool, since no CUDA process was running
        let u = m.unaccounted() as f64 / GIB as f64;
        assert!((79.0..82.0).contains(&u), "unaccounted {u:.1} GiB");
    }

    #[test]
    fn hugetlb_boxes_read_the_free_huge_pages() {
        let text = "MemTotal: 1000 kB\nMemAvailable: 900 kB\nHugePages_Total: 8\nHugePages_Free: 3\nHugepagesize: 1048576 kB\n";
        let m = MemInfo::parse(text).unwrap();
        assert!(m.hugetlb_configured());
        assert_eq!(m.usable(), 3 * GIB);
        assert_eq!(MemInfo::parse("MemTotal: 1000 kB\n"), None);
    }

    /// A pool of `pool` bytes in front of a kernel with `kernel` bytes of
    /// MemAvailable: allocations drain the pool first, then the kernel.
    struct PooledBox {
        pool: u64,
        kernel: u64,
        refuse_after: u64,
        live: u64,
        freed: u64,
    }

    impl PooledBox {
        fn run(&mut self, cap: u64) -> u64 {
            let cell = std::cell::RefCell::new(self);
            measure_retained(
                cap,
                PROBE_CHUNK,
                || Some(cell.borrow().kernel),
                |size| {
                    let mut b = cell.borrow_mut();
                    if b.live + size > b.refuse_after {
                        return None;
                    }
                    let from_pool = size.min(b.pool);
                    b.pool -= from_pool;
                    let from_kernel = size - from_pool;
                    if from_kernel > b.kernel {
                        return None;
                    }
                    b.kernel -= from_kernel;
                    b.live += size;
                    Some(size)
                },
                |size| {
                    let mut b = cell.borrow_mut();
                    b.live -= size;
                    b.freed += size;
                    b.pool += size; // freed pages go back to the pool
                },
            )
        }
    }

    #[test]
    fn the_probe_credits_the_pool_and_stops_where_the_kernel_starts_paying() {
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        let credit = b.run(88 * GIB);
        assert_eq!(credit, 56 * GIB);
        // it walked one chunk into the kernel's memory and no further
        assert_eq!(b.kernel, 38 * GIB);
        assert_eq!(b.live, 0, "every chunk is freed");
        assert_eq!(b.freed, 58 * GIB);
    }

    #[test]
    fn the_probe_never_measures_past_its_cap_or_past_a_refusal() {
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(10 * GIB), 10 * GIB);
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: 20 * GIB,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(88 * GIB), 20 * GIB);
        assert_eq!(b.live, 0);
        // an odd cap still ends exactly on it
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(5 * GIB + 7), 5 * GIB + 7);
    }

    #[test]
    fn no_pool_means_no_credit_and_one_chunk_of_cache_at_most() {
        let mut b = PooledBox {
            pool: 0,
            kernel: 80 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(60 * GIB), 0);
        assert_eq!(b.freed, PROBE_CHUNK);
    }

    #[test]
    fn credit_is_spent_by_what_this_process_allocates_afterwards() {
        let c = RetainedCredit {
            proven: 56 * GIB,
            ours_at_probe: GIB,
        };
        assert_eq!(c.remaining(GIB), 56 * GIB);
        // weights and planes land from the pool first
        assert_eq!(c.remaining(26 * GIB), 31 * GIB);
        assert_eq!(c.remaining(80 * GIB), 0);
        // our pool shrinking below the probe's reading adds nothing back
        assert_eq!(c.remaining(0), 56 * GIB);
    }
}
