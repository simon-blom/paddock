//! Generic grouped-query-attention tensor-parallel partitioning.

use cudarc::driver::{DevicePtr, DevicePtrMut};
use paddock_models::mapped::MappedGguf;

use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, QuantW};

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

#[derive(Debug, thiserror::Error)]
pub enum AttentionTpError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error(transparent)]
    Topology(#[from] TpTopologyError),
    #[error("tensor-parallel attention: {0}")]
    Shape(String),
}

/// Weight names for the conventional Q/K/V column-parallel + output
/// row-parallel attention decomposition.
#[derive(Debug, Clone, Copy)]
pub struct AttentionWeightNames<'a> {
    pub q: &'a str,
    pub k: &'a str,
    pub v: &'a str,
    pub output: &'a str,
}

/// Generic sharded attention projection weights.
///
/// This does not prescribe attention semantics. A model may use fused Q+gate,
/// custom RoPE/norms, or a completely different attention execution while
/// still reusing the standard sharding contract.
pub struct AttentionTpWeights {
    q: QuantW,
    k: QuantW,
    v: QuantW,
    output: QuantW,
    topology: TpTopology,
}

impl AttentionTpWeights {
    pub fn load(
        exec: &GpuExecutor,
        map: &MappedGguf,
        names: AttentionWeightNames<'_>,
        topology: TpTopology,
    ) -> Result<Self, AttentionTpError> {
        let column = topology.tensor_slice(super::TpLinearMode::ColumnParallel.shard_kind());
        let row = topology.tensor_slice(
            super::TpLinearMode::RowParallel { reduce: true }.shard_kind(),
        );
        Ok(Self {
            q: exec.load_quantw_shard(map, names.q, column)?,
            k: exec.load_quantw_shard(map, names.k, column)?,
            v: exec.load_quantw_shard(map, names.v, column)?,
            output: exec.load_quantw_shard(map, names.output, row)?,
            topology,
        })
    }

    pub fn topology(&self) -> TpTopology {
        self.topology
    }

    pub fn q(&self) -> &QuantW {
        &self.q
    }

    pub fn k(&self) -> &QuantW {
        &self.k
    }

    pub fn v(&self) -> &QuantW {
        &self.v
    }

    pub fn output(&self) -> &QuantW {
        &self.output
    }
}

/// Complete a row-parallel attention output projection.
pub fn reduce_output<C, S, R>(
    exec: &GpuExecutor,
    group: &C,
    topology: TpTopology,
    partial: &S,
    reduced: &mut R,
) -> Result<(), AttentionTpError>
where
    C: Communicator,
    S: DevicePtr<f32>,
    R: DevicePtrMut<f32>,
{
    if TpTopology::from_group(group)? != topology || partial.len() != reduced.len() {
        return Err(AttentionTpError::Shape(
            "attention reduction topology or output width changed".into(),
        ));
    }
    group.after_compute(&exec.stream)?;
    group.all_reduce(partial, reduced)?;
    group.before_compute(&exec.stream)?;
    Ok(())
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
