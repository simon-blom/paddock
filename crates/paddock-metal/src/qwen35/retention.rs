//! Keep the two current boundaries of a conversation, not every obsolete turn.
//! Global LRU alone lets a fast tool loop evict the other active conversations.
//! No new GPU allocations, per-client identity, or changes to cached arithmetic.
use super::{Checkpoint, multimodal};

pub(super) fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_METAL_CACHE_TRACE").is_some())
}

pub(super) fn superseded(
    checkpoint: &Checkpoint,
    history: &[u32],
    images: &[multimodal::ImageKey],
    keep_from: usize,
) -> bool {
    checkpoint.history.len() < keep_from && covers(checkpoint, history, images)
}

fn covers(checkpoint: &Checkpoint, history: &[u32], images: &[multimodal::ImageKey]) -> bool {
    !checkpoint.history.is_empty()
        && history.starts_with(&checkpoint.history)
        && multimodal::prefix_images_match(&checkpoint.images, images, checkpoint.history.len())
}

/// Prefer an older boundary on the exact same token/image branch. Preserve both
/// current trailing cuts: the shorter one survives chat-template header edits.
/// Empty entries come next, then ordinary LRU for unrelated/new conversations.
/// Reserved GPU copy destinations are never eligible, including aborted restores.
pub(super) fn replacement(
    cache: &[Checkpoint],
    history: &[u32],
    images: &[multimodal::ImageKey],
    keep_from: usize,
) -> Option<usize> {
    cache
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            !c.reserved && !(c.history.len() >= keep_from && covers(c, history, images))
        })
        .min_by_key(|(_, c)| {
            let old = superseded(c, history, images, keep_from);
            (
                if old {
                    0
                } else if c.history.is_empty() {
                    1
                } else {
                    2
                },
                if old {
                    c.history.len() as u64
                } else {
                    c.touched
                },
            )
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(token: u32, n: usize, touched: u64) -> Checkpoint {
        Checkpoint {
            history: vec![token; n],
            touched,
            ..Default::default()
        }
    }

    #[test]
    fn retires_ancestors_before_another_conversation_or_current_backup() {
        let cache = vec![
            checkpoint(1, 64, 1),
            checkpoint(2, 64, 9),
            checkpoint(2, 80, 10),
            checkpoint(2, 96, 11),
        ];
        assert_eq!(replacement(&cache, &[2; 112], &[], 96), Some(1));
        assert!(!superseded(&cache[3], &[2; 112], &[], 96));
        assert_eq!(replacement(&cache, &[3; 112], &[], 96), Some(0));
    }

    #[test]
    fn uses_empty_entries_but_never_reserved_destinations() {
        let mut cache = vec![checkpoint(1, 64, 1), Checkpoint::default()];
        assert_eq!(replacement(&cache, &[2; 112], &[], 96), Some(1));
        cache[1].reserved = true;
        assert_eq!(replacement(&cache, &[2; 112], &[], 96), Some(0));
        cache[0].reserved = true;
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), None);
        assert_eq!(replacement(&[], &[1; 112], &[], 96), None);
    }

    #[test]
    fn skips_optional_capture_instead_of_overwriting_its_current_backup() {
        let mut cache = vec![checkpoint(1, 96, 1), Checkpoint::default()];
        cache[1].reserved = true;
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), None);
        cache[1].reserved = false;
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), Some(1));
    }

    #[test]
    fn four_interleaved_tool_loops_keep_both_boundaries_with_eight_entries() {
        let mut cache: Vec<_> = (0..8).map(|_| Checkpoint::default()).collect();
        let mut clock = 0;
        let mut lengths = [64; 4];
        // A fast branch runs ahead while the other three wait on tools.
        for branch in [0, 1, 2, 3, 0, 0, 0, 2, 0, 3, 1, 1, 0, 2, 3] {
            let first = lengths[branch];
            for cut in [first, first + 16] {
                let history = vec![branch as u32; cut];
                let i = replacement(&cache, &history, &[], first).unwrap();
                clock += 1;
                cache[i] = checkpoint(branch as u32, cut, clock);
            }
            lengths[branch] += 64;
            for (peer, &next) in lengths.iter().enumerate().filter(|(_, n)| **n > 64) {
                for cut in [next - 64, next - 48] {
                    assert!(cache.iter().any(|c| c.history == vec![peer as u32; cut]));
                }
            }
        }
    }

    #[test]
    fn a_diverged_or_shortened_branch_is_not_an_ancestor() {
        let c = checkpoint(2, 64, 1);
        assert!(!superseded(&c, &[2; 48], &[], 96));
        let mut changed = vec![2; 112];
        changed[63] = 3;
        assert!(!superseded(&c, &changed, &[], 96));
        assert!(!superseded(&Checkpoint::default(), &[2; 112], &[], 96));
    }
}
