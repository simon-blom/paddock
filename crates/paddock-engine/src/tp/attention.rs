//! Generic grouped-query-attention tensor-parallel partitioning.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use paddock_models::mapped::MappedGguf;

use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, KvDtype, QuantW};
use crate::tp::cache::MirroredKv;
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

/// Shared paged KV append inputs. Both payload planes use the same metadata;
/// the attention kernel can then reuse this exact table and slot stride.
pub struct PagedKvBatch<'a> {
    positions: &'a CudaSlice<u32>,
    slots: &'a CudaSlice<u32>,
    table: &'a CudaSlice<u32>,
    blocks_per_slot: usize,
    kv_dim: usize,
    rows: usize,
    dtype: KvDtype,
}

impl<'a> PagedKvBatch<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        positions: &'a CudaSlice<u32>,
        slots: &'a CudaSlice<u32>,
        table: &'a CudaSlice<u32>,
        blocks_per_slot: usize,
        kv_dim: usize,
        rows: usize,
        dtype: KvDtype,
    ) -> Result<Self, AttentionTpError> {
        validate_paged_kv_metadata(
            positions.len(),
            slots.len(),
            table.len(),
            blocks_per_slot,
            kv_dim,
            rows,
        )?;
        Ok(Self {
            positions,
            slots,
            table,
            blocks_per_slot,
            kv_dim,
            rows,
            dtype,
        })
    }

    pub fn append(
        &self,
        exec: &GpuExecutor,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        kc: &mut CudaSlice<u8>,
        vc: &mut CudaSlice<u8>,
    ) -> Result<(), AttentionTpError> {
        let input_len = self.rows * self.kv_dim; // checked in the constructor
        let block_bytes = BLOCK_TOKENS
            .checked_mul(self.kv_dim)
            .and_then(|n| n.checked_mul(self.dtype.bytes()))
            .ok_or_else(|| AttentionTpError::Shape("KV block size overflow".into()))?;
        validate_paged_kv_payload(k.len(), v.len(), kc.len(), vc.len(), input_len, block_bytes)?;
        self.append_one(exec, k, kc)?;
        self.append_one(exec, v, vc)?;
        Ok(())
    }

    fn append_one(
        &self,
        exec: &GpuExecutor,
        input: &CudaSlice<f32>,
        cache: &mut CudaSlice<u8>,
    ) -> Result<(), GpuError> {
        exec.kv_append_batch_paged(
            input,
            cache,
            self.positions,
            Some(self.slots),
            self.table,
            self.blocks_per_slot,
            self.kv_dim,
            self.rows,
            self.dtype,
        )
    }

    pub fn table(&self) -> (&'a CudaSlice<u32>, usize) {
        (self.table, self.blocks_per_slot)
    }
}

fn validate_paged_kv_metadata(
    positions: usize,
    slots: usize,
    table: usize,
    blocks_per_slot: usize,
    kv_dim: usize,
    rows: usize,
) -> Result<(), AttentionTpError> {
    if rows == 0
        || rows > u32::MAX as usize
        || kv_dim == 0
        || kv_dim > u32::MAX as usize
        || blocks_per_slot == 0
        || blocks_per_slot > u32::MAX as usize
        || rows.checked_mul(kv_dim).is_none()
        || positions < rows
        || slots < rows
        || table == 0
        || !table.is_multiple_of(blocks_per_slot)
    {
        return Err(AttentionTpError::Shape(
            "invalid paged KV append metadata".into(),
        ));
    }
    Ok(())
}

fn validate_paged_kv_payload(
    k_len: usize,
    v_len: usize,
    kc_len: usize,
    vc_len: usize,
    input_len: usize,
    block_bytes: usize,
) -> Result<(), AttentionTpError> {
    if k_len < input_len
        || v_len < input_len
        || kc_len == 0
        || kc_len != vc_len
        || block_bytes == 0
        || !kc_len.is_multiple_of(block_bytes)
    {
        return Err(AttentionTpError::Shape(
            "paged K/V payload geometry mismatch".into(),
        ));
    }
    Ok(())
}

/// Default paged GQA decode: split partial+combine when the model's policy
/// requests multiple fixed splits, otherwise use the single-pass kernel.
/// Models own split-count policy and may override this dispatch entirely.
/// The split count must remain constant across graph replays.
#[allow(clippy::too_many_arguments)]
pub fn decode_paged(
    exec: &GpuExecutor,
    q: &CudaSlice<f32>,
    kc: &CudaSlice<u8>,
    vc: &CudaSlice<u8>,
    sinks: &CudaSlice<f32>,
    attn_o: &mut CudaSlice<f32>,
    attn_ml: &mut CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    positions: &CudaSlice<u32>,
    slots: Option<&CudaSlice<u32>>,
    block_tables: &CudaSlice<u32>,
    blocks_per_slot: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    batch: usize,
    scale: f32,
    dtype: KvDtype,
    n_splits: usize,
) -> Result<(), AttentionTpError> {
    if n_splits == 0 || n_heads == 0 || batch == 0 {
        return Err(AttentionTpError::Shape(
            "empty attention decode geometry".into(),
        ));
    }
    if n_splits > 1 {
        validate_decode_scratch(
            n_heads,
            batch,
            head_dim,
            n_splits,
            attn_o.len(),
            attn_ml.len(),
        )?;
        if !exec.has_attn_partial_batch_paged() {
            return Err(AttentionTpError::Gpu(GpuError::Unsupported(
                "pack lacks the paged attention split-partial kernel".into(),
            )));
        }
        exec.attn_partial_batch_paged(
            q,
            kc,
            vc,
            attn_o,
            attn_ml,
            positions,
            slots,
            block_tables,
            blocks_per_slot,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            0,
            n_splits,
            batch,
            scale,
            dtype,
        )?;
        exec.attn_combine_batch(
            attn_o, attn_ml, sinks, out, n_heads, head_dim, n_splits, batch,
        )?;
    } else {
        exec.attn_decode_batch_paged(
            q,
            kc,
            vc,
            sinks,
            out,
            positions,
            slots,
            block_tables,
            blocks_per_slot,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            0,
            batch,
            scale,
            dtype,
        )?;
    }
    Ok(())
}

fn validate_decode_scratch(
    n_heads: usize,
    batch: usize,
    head_dim: usize,
    n_splits: usize,
    attn_o_len: usize,
    attn_ml_len: usize,
) -> Result<(), AttentionTpError> {
    let partials = n_heads
        .checked_mul(batch)
        .and_then(|n| n.checked_mul(n_splits))
        .ok_or_else(|| AttentionTpError::Shape("attention split count overflow".into()))?;
    let out_len = partials
        .checked_mul(head_dim)
        .ok_or_else(|| AttentionTpError::Shape("attention split output overflow".into()))?;
    let ml_len = partials
        .checked_mul(2)
        .ok_or_else(|| AttentionTpError::Shape("attention split metadata overflow".into()))?;
    if attn_o_len < out_len || attn_ml_len < ml_len {
        return Err(AttentionTpError::Shape(
            "attention split scratch too small".into(),
        ));
    }
    Ok(())
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
    use crate::tp::cache::Operation;

    #[test]
    fn paged_kv_pair_preflights_both_planes_before_append() {
        assert!(validate_paged_kv_metadata(4, 4, 12, 6, 64, 4).is_ok());
        for (positions, slots, table, stride, kv_dim, rows) in [
            (3, 4, 12, 6, 64, 4),
            (4, 3, 12, 6, 64, 4),
            (4, 4, 11, 6, 64, 4),
            (4, 4, 12, 0, 64, 4),
            (4, 4, 12, 6, 0, 4),
            (4, 4, 12, 6, 64, 0),
            (4, 4, 12, 6, usize::MAX, 4),
        ] {
            assert!(
                validate_paged_kv_metadata(positions, slots, table, stride, kv_dim, rows).is_err()
            );
        }
        assert!(validate_paged_kv_payload(256, 256, 2048, 2048, 256, 1024).is_ok());
        for (k, v, kc, vc, input, block) in [
            (255, 256, 2048, 2048, 256, 1024),
            (256, 255, 2048, 2048, 256, 1024),
            (256, 256, 2048, 1024, 256, 1024),
            (256, 256, 2049, 2049, 256, 1024),
            (256, 256, 0, 0, 256, 1024),
            (256, 256, 2048, 2048, 256, 0),
        ] {
            assert!(validate_paged_kv_payload(k, v, kc, vc, input, block).is_err());
        }
    }

    #[test]
    fn decode_split_scratch_covers_local_heads_and_batch() {
        let partials = 3 * 5;
        assert!(validate_decode_scratch(3, 1, 64, 5, partials * 64, partials * 2).is_ok());
        assert!(validate_decode_scratch(3, 1, 64, 5, partials * 64 - 1, partials * 2).is_err());
        assert!(validate_decode_scratch(3, 1, 64, 5, partials * 64, partials * 2 - 1).is_err());
        assert!(validate_decode_scratch(3, 2, 64, 5, partials * 64, partials * 2).is_err());
        assert!(validate_decode_scratch(usize::MAX, 2, 64, 5, usize::MAX, usize::MAX).is_err());
    }

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
