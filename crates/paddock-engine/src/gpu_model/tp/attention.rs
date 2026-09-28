//! Generic grouped-query-attention tensor-parallel partitioning.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use paddock_models::mapped::MappedGguf;

use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, QuantW};
use crate::gpu_model::tp::cache::MirroredKv;
use crate::kv_pool::BLOCK_TOKENS;

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

/// A persistent paged-KV block table for one TP rank. Logical page ownership
/// remains in `MirroredKv`; only changed slot mappings are uploaded. Every
/// decode still validates geometry and coverage before entering a collective.
pub struct PagedAttentionTable {
    device: CudaSlice<u32>,
    staged_versions: Vec<u64>,
    blocks_per_slot: usize,
    max_ctx: usize,
}

impl PagedAttentionTable {
    pub fn new(exec: &GpuExecutor, slots: usize, max_ctx: usize) -> Result<Self, AttentionTpError> {
        if slots == 0 || max_ctx == 0 || max_ctx > u32::MAX as usize {
            return Err(AttentionTpError::Shape("invalid paged attention geometry".into()));
        }
        let blocks_per_slot = max_ctx.div_ceil(BLOCK_TOKENS);
        let table_len = blocks_per_slot
            .checked_mul(slots)
            .filter(|&len| len <= u32::MAX as usize)
            .ok_or_else(|| AttentionTpError::Shape("paged attention table too large".into()))?;
        Ok(Self {
            device: exec.alloc_u32(table_len)?,
            staged_versions: vec![u64::MAX; slots],
            blocks_per_slot,
            max_ctx,
        })
    }

    pub fn device(&self) -> &CudaSlice<u32> {
        &self.device
    }

    pub fn blocks_per_slot(&self) -> usize {
        self.blocks_per_slot
    }

    pub fn slots(&self) -> usize {
        self.staged_versions.len()
    }

    pub fn stage_slot(
        &mut self,
        exec: &GpuExecutor,
        logical: &MirroredKv,
        slot: usize,
        position: usize,
        pool_blocks: u32,
    ) -> Result<(), AttentionTpError> {
        let cached = *self
            .staged_versions
            .get(slot)
            .ok_or_else(|| AttentionTpError::Shape("slot out of range".into()))?;
        let Some((live, version)) = paged_slot_upload(
            logical,
            slot,
            position,
            pool_blocks,
            self.slots(),
            self.max_ctx,
            self.blocks_per_slot,
            cached,
        )? else {
            return Ok(());
        };
        let start = slot * self.blocks_per_slot;
        if !live.is_empty() {
            exec.stream
                .memcpy_htod(live, &mut self.device.slice_mut(start..start + live.len()))
                .map_err(GpuError::from)?;
        }
        // Commit the generation only after the upload succeeds. A failed copy
        // must retry rather than advertise stale device contents as current.
        self.staged_versions[slot] = version;
        Ok(())
    }
}

/// Pure host-side hot-path decision shared by GPU staging and CPU tests.
fn paged_slot_upload(
    logical: &MirroredKv,
    slot: usize,
    position: usize,
    pool_blocks: u32,
    slots: usize,
    max_ctx: usize,
    blocks_per_slot: usize,
    cached: u64,
) -> Result<Option<(&[u32], u64)>, AttentionTpError> {
    let version = logical
        .validate_device_table_versioned(
            slot,
            position,
            pool_blocks,
            slots,
            max_ctx,
            Some(cached),
        )
        .map_err(|err| AttentionTpError::Shape(err.into()))?;
    if version == cached {
        return Ok(None);
    }
    let live = logical
        .slot_blocks(slot)
        .ok_or_else(|| AttentionTpError::Shape("slot out of range".into()))?;
    if live.len() > blocks_per_slot {
        return Err(AttentionTpError::Shape("block table exceeds slot capacity".into()));
    }
    Ok(Some((live, version)))
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
    use crate::gpu_model::tp::cache::Operation;

    #[test]
    fn paged_upload_tracks_slot_mapping_not_decode_position() {
        let mut kv = MirroredKv::new(4, 2, 48).unwrap();
        kv.authorize(Operation::Ensure { slot: 0, position: 0 })
            .unwrap();
        let (first, version) = paged_slot_upload(&kv, 0, 0, 4, 2, 48, 3, u64::MAX)
            .unwrap()
            .unwrap();
        assert_eq!(first.len(), 1);
        assert!(paged_slot_upload(&kv, 0, 15, 4, 2, 48, 3, version)
            .unwrap()
            .is_none());
        // A cached generation still refuses a missing page or wrong geometry.
        assert!(paged_slot_upload(&kv, 0, 16, 4, 2, 48, 3, version).is_err());
        assert!(paged_slot_upload(&kv, 0, 0, 3, 2, 48, 3, version).is_err());
        assert!(paged_slot_upload(&kv, 2, 0, 4, 2, 48, 3, version).is_err());

        kv.authorize(Operation::Ensure { slot: 0, position: 16 })
            .unwrap();
        let (grown, next) = paged_slot_upload(&kv, 0, 16, 4, 2, 48, 3, version)
            .unwrap()
            .unwrap();
        assert_eq!(grown.len(), 2);
        assert_ne!(next, version);
        assert!(paged_slot_upload(&kv, 0, 16, 4, 2, 48, 1, version).is_err());

        kv.authorize(Operation::Release { slot: 0 }).unwrap();
        kv.authorize(Operation::Ensure { slot: 0, position: 0 })
            .unwrap();
        assert!(paged_slot_upload(&kv, 0, 0, 4, 2, 48, 3, next)
            .unwrap()
            .is_some());
    }

    #[test]
    fn paged_upload_caches_each_slot_independently() {
        let mut kv = MirroredKv::new(4, 2, 48).unwrap();
        for slot in 0..2 {
            kv.authorize(Operation::Ensure { slot, position: 0 })
                .unwrap();
        }
        let (_, v0) = paged_slot_upload(&kv, 0, 0, 4, 2, 48, 3, u64::MAX)
            .unwrap()
            .unwrap();
        let (_, v1) = paged_slot_upload(&kv, 1, 0, 4, 2, 48, 3, u64::MAX)
            .unwrap()
            .unwrap();
        kv.authorize(Operation::Ensure { slot: 1, position: 16 })
            .unwrap();
        assert!(paged_slot_upload(&kv, 0, 0, 4, 2, 48, 3, v0)
            .unwrap()
            .is_none());
        assert!(paged_slot_upload(&kv, 1, 16, 4, 2, 48, 3, v1)
            .unwrap()
            .is_some());
    }

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
