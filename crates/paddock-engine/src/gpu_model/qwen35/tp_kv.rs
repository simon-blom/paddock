//! Qwen3.8 policy facade over the generic TP cache lifecycle.
//!
//! Cache/page/checkpoint mechanics live in `gpu_model::tp::cache`. Qwen keeps
//! only the measured policy deciding when a resume is profitable.

pub use crate::gpu_model::tp::cache::{
    Event, MirroredKv, Operation, PrefixProbe, Snapshot, tp_publish_ops,
};
use crate::gpu_model::tp::cache::{ResumePolicy, resume_decision};

/// Qwen3.8's measured resume-profitability policy.
///
/// Structural cache safety is generic; these thresholds remain model/workload
/// policy so another model does not inherit Qwen-specific tuning.
pub fn tp_resume_decision(
    ckpt: Option<(usize, u32)>,
    t_len: usize,
    slots: usize,
) -> usize {
    resume_decision(
        ckpt,
        t_len,
        slots,
        ResumePolicy {
            min_cache_prefix: super::min_cache_prefix(),
            narrow_slots_max: super::resume_live_max(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_resume_policy_preserves_narrow_short_resume() {
        assert_eq!(tp_resume_decision(Some((32, 0)), 100, 2), 32);
        assert_eq!(tp_resume_decision(Some((64, 0)), 64, 2), 0);
    }
}
