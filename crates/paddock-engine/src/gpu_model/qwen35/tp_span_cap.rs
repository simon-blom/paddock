//! The TP=2 prefill span row cap: one authoritative, rank-symmetric value.
//!
//! The cap decides how many contiguous prompt rows one whole-model span
//! advance processes. Every consumer (span planes, DeltaNet scratch, the
//! serve-side chunkers) derives its geometry from [`resolved()`] so no magic
//! constant drifts between TP and DeltaNet.
//!
//! Resolution is rank-0-authoritative: the coordinator resolves
//! `PADDOCK_TP_SPAN_CAP` once and ships the value in `TpInit`; the worker
//! never reads the variable itself (the same pattern as kv_dtype, graph mode
//! and checkpoint capacity, so a hand-started remote worker cannot pair a
//! different span geometry and desynchronize the collectives). Invalid or
//! unsupported values fail closed with a clear error instead of truncating.
//!
//! 64 remains the default. The supported experimental widths are 128, 192,
//! 384, 512, 1024, 2048, 4096, and 8192; all are parameterized through the
//! span-sized allocations and remain unvalidated until the sweep below.
use std::sync::atomic::{AtomicUsize, Ordering};

/// The production default: the only GPU-validated span width.
pub const DEFAULT_TP_SPAN_CAP: usize = 64;

/// Largest width the staging and DeltaNet geometry are sized for.
pub const MAX_TP_SPAN_CAP: usize = 8192;

/// Supported experimental widths for the two-node sweep.
pub const SWEEP_CAPS: [usize; 9] = [64, 128, 192, 384, 512, 1024, 2048, 4096, 8192];

static RESOLVED: AtomicUsize = AtomicUsize::new(DEFAULT_TP_SPAN_CAP);

/// The error for an unsupported `PADDOCK_TP_SPAN_CAP` value.
#[derive(Debug, thiserror::Error)]
#[error("PADDOCK_TP_SPAN_CAP={value} unsupported: use one of {allowed:?} (default {default})")]
pub struct SpanCapError {
    pub value: String,
    pub allowed: [usize; 9],
    pub default: usize,
}

/// Parse and install the span cap. `Some("")`/garbage/unset all resolve to
/// the 64 default; a value outside the sweep set is a hard error so a typo
/// can never silently change the geometry.
pub fn parse_span_cap(env: Option<&str>) -> Result<usize, SpanCapError> {
    let Some(raw) = env else {
        return Ok(DEFAULT_TP_SPAN_CAP);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(DEFAULT_TP_SPAN_CAP);
    }
    let value: usize = raw.parse().map_err(|_| SpanCapError {
        value: raw.to_owned(),
        allowed: SWEEP_CAPS,
        default: DEFAULT_TP_SPAN_CAP,
    })?;
    if !SWEEP_CAPS.contains(&value) {
        return Err(SpanCapError {
            value: raw.to_owned(),
            allowed: SWEEP_CAPS,
            default: DEFAULT_TP_SPAN_CAP,
        });
    }
    Ok(value)
}

/// Resolve the cap from the coordinator's env (rank 0 only) and publish it
/// process-wide. Rank 1 must use [`wire_span_cap`] with the `TpInit` value
/// instead of reading the environment.
pub fn resolve_from_env(env: Option<&str>) -> Result<usize, SpanCapError> {
    let cap = parse_span_cap(env)?;
    RESOLVED.store(cap, Ordering::Relaxed);
    Ok(cap)
}

/// Install the rank-0-authoritative value carried by `TpInit`. Fails closed
/// on an unsupported wire value so a mismatched pair can never serve.
pub fn wire_span_cap(wire: usize) -> Result<usize, SpanCapError> {
    if !SWEEP_CAPS.contains(&wire) {
        return Err(SpanCapError {
            value: wire.to_string(),
            allowed: SWEEP_CAPS,
            default: DEFAULT_TP_SPAN_CAP,
        });
    }
    RESOLVED.store(wire, Ordering::Relaxed);
    Ok(wire)
}

/// The authoritative cap every consumer reads. Compile-time constants are
/// gone on purpose: one value decides every span plane and chunker.
#[inline]
pub fn span_cap() -> usize {
    RESOLVED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_empty_and_garbage_default_or_fail() {
        assert_eq!(parse_span_cap(None).unwrap(), 64);
        assert_eq!(parse_span_cap(Some("")).unwrap(), 64);
        assert_eq!(parse_span_cap(Some("  ")).unwrap(), 64);
        let err = parse_span_cap(Some("abc")).unwrap_err();
        assert_eq!(err.value, "abc");
        assert_eq!(
            err.allowed,
            [64, 128, 192, 384, 512, 1024, 2048, 4096, 8192]
        );
    }

    #[test]
    fn only_sweep_values_are_accepted() {
        for v in SWEEP_CAPS {
            assert_eq!(parse_span_cap(Some(&v.to_string())).unwrap(), v);
        }
        for bad in [
            "0", "1", "63", "65", "127", "129", "191", "193", "256", "-1",
        ] {
            assert!(parse_span_cap(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn resolution_is_published_and_wire_gated() {
        // Each test thread mutates the process-wide cell; keep the order
        // explicit so the final state is deterministic for later readers.
        assert_eq!(wire_span_cap(128).unwrap(), 128);
        assert_eq!(span_cap(), 128);
        assert!(wire_span_cap(65).is_err());
        // A refused wire value leaves the last good resolution in place.
        assert_eq!(span_cap(), 128);
        assert_eq!(resolve_from_env(Some("192")).unwrap(), 192);
        assert_eq!(span_cap(), 192);
        assert!(resolve_from_env(Some("999")).is_err());
        assert_eq!(resolve_from_env(None).unwrap(), 64);
        assert_eq!(span_cap(), 64);
    }
}
