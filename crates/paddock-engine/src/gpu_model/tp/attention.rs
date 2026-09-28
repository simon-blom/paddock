//! Generic grouped-query-attention tensor-parallel partitioning.

use super::{TpTopology, TpTopologyError};

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum GqaPartitionError {
    #[error(transparent)]
    Topology(#[from] TpTopologyError),
    #[error("attention head counts must be nonzero")]
    EmptyHeads,
    #[error("query heads {heads} must be a multiple of KV heads {kv_heads}")]
    IncompleteGroups { heads: usize, kv_heads: usize },
    #[error("KV heads {kv_heads} cannot split evenly over {world_size} TP ranks")]
    UnevenKvHeads { kv_heads: usize, world_size: usize },
}

/// Complete Q groups follow the rank that owns their KV head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaPartition {
    pub heads: usize,
    pub kv_heads: usize,
    pub local_heads: usize,
    pub local_kv_heads: usize,
    pub q_start: usize,
    pub kv_start: usize,
}

impl GqaPartition {
    pub fn new(
        topology: TpTopology,
        heads: usize,
        kv_heads: usize,
    ) -> Result<Self, GqaPartitionError> {
        if heads == 0 || kv_heads == 0 {
            return Err(GqaPartitionError::EmptyHeads);
        }
        if !heads.is_multiple_of(kv_heads) {
            return Err(GqaPartitionError::IncompleteGroups { heads, kv_heads });
        }
        if !kv_heads.is_multiple_of(topology.world_size()) {
            return Err(GqaPartitionError::UnevenKvHeads {
                kv_heads,
                world_size: topology.world_size(),
            });
        }
        let local_kv_heads = kv_heads / topology.world_size();
        let q_per_kv = heads / kv_heads;
        let local_heads = local_kv_heads
            .checked_mul(q_per_kv)
            .ok_or(TpTopologyError::UnevenShard {
                axis: heads,
                world_size: topology.world_size(),
            })?;
        Ok(Self {
            heads,
            kv_heads,
            local_heads,
            local_kv_heads,
            q_start: topology.rank() * local_heads,
            kv_start: topology.rank() * local_kv_heads,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_groups_partition_over_multiple_world_sizes() {
        for (world, heads, kv_heads) in [(2usize, 24usize, 4usize), (3, 24, 6), (4, 32, 8)] {
            for rank in 0..world {
                let topology = TpTopology::new(rank, world).unwrap();
                let p = GqaPartition::new(topology, heads, kv_heads).unwrap();
                assert_eq!(p.local_kv_heads, kv_heads / world);
                assert_eq!(p.local_heads, heads / world);
                assert_eq!(p.kv_start, rank * p.local_kv_heads);
                assert_eq!(p.q_start, rank * p.local_heads);
            }
        }
    }

    #[test]
    fn rejects_ragged_or_incomplete_groups() {
        let tp3 = TpTopology::new(0, 3).unwrap();
        assert!(matches!(
            GqaPartition::new(tp3, 24, 4),
            Err(GqaPartitionError::UnevenKvHeads { .. })
        ));
        assert!(matches!(
            GqaPartition::new(TpTopology::new(0, 2).unwrap(), 23, 4),
            Err(GqaPartitionError::IncompleteGroups { .. })
        ));
    }
}
