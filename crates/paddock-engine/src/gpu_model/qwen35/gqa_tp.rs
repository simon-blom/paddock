//! Isolated one-token Qwen3.8 GQA TP, with one rank-local KV payload per process.
//!
//! Adapted from ErikBPF/paddock `contrib/tp-09-attention-block` (afedfdd):
//! split complete Q/KV groups, shard interleaved Q/gate output rows and output
//! projection input columns. The fork's same-process Link copies are replaced
//! with the Phase 3 NCCL sum. No production forward or scheduler is changed.
use cudarc::driver::CudaSlice;
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::tensor_slice::{ShardKind, TensorSliceRequest};
use paddock_models::{gguf::Value, mapped::MappedGguf};

use super::ops::{gemv_any, read_sections};
use super::tp_kv::MirroredKv;
use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, KvDtype, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_pool::BLOCK_TOKENS;

#[derive(Debug, thiserror::Error)]
pub enum GqaTpError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] GpuModelError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error("GQA TP: {0}")]
    Shape(String),
}

/// Complete Q groups follow their owning KV head. No ragged or replicated KV
/// policy is implicit: unsupported geometries fail before uploading weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaGeometry {
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub local_heads: usize,
    pub local_kv_heads: usize,
    pub kv_start: usize,
}
impl GqaGeometry {
    pub fn new(
        width: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rank: usize,
    ) -> Result<Self, GqaTpError> {
        if rank >= 2
            || width == 0
            || heads == 0
            || kv_heads == 0
            || head_dim != 256
            || !heads.is_multiple_of(kv_heads)
            || !kv_heads.is_multiple_of(2)
            || !width.is_multiple_of(256)
            || heads.checked_mul(head_dim).is_none()
            || kv_heads.checked_mul(head_dim).is_none()
        {
            return Err(GqaTpError::Shape("TP=2 requires even KV heads, complete Q/KV groups, 256-wide heads and block-aligned width".into()));
        }
        let local_kv_heads = kv_heads / 2;
        let local_heads = local_kv_heads * (heads / kv_heads);
        Ok(Self {
            width,
            heads,
            kv_heads,
            head_dim,
            local_heads,
            local_kv_heads,
            kv_start: rank * local_kv_heads,
        })
    }
    pub fn kv_dim(self) -> usize {
        self.local_kv_heads * self.head_dim
    }
    pub fn q_dim(self) -> usize {
        self.local_heads * self.head_dim
    }
    pub fn kv_bytes(self, max_ctx: usize, dtype: KvDtype) -> Option<usize> {
        max_ctx
            .checked_mul(self.kv_dim())?
            .checked_mul(dtype.bytes())
    }
}

/// `input` is post-attention-norm, identical on both ranks. The output is
/// attention's hidden-width projection (without residual/FFN), identical after
/// NCCL sum. KV buffers are owned locally and never communicated.
pub struct GqaTpRank {
    pub geometry: GqaGeometry,
    weights: [QuantW; 4],
    qnorm: CudaSlice<f32>,
    knorm: CudaSlice<f32>,
    sinks: CudaSlice<f32>,
    qg: CudaSlice<f32>,
    q: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    reduced: CudaSlice<f32>,
    kc: CudaSlice<u8>,
    vc: CudaSlice<u8>,
    positions: CudaSlice<u32>,
    slots: CudaSlice<u32>,
    axes: CudaSlice<u32>,
    block_tables: Option<CudaSlice<u32>>,
    blocks_per_slot: usize,
    slots_count: usize,
    pos: usize,
    max_ctx: usize,
    dtype: KvDtype,
    rank: usize,
    nrot: usize,
    eps: f32,
    yarn: (f32, f32, f32, f32, f32, f32),
    sections: [u32; 4],
}
impl GqaTpRank {
    pub fn load<C: Communicator>(
        e: &GpuExecutor,
        map: &MappedGguf,
        layer: usize,
        group: &C,
        max_ctx: usize,
        dtype: KvDtype,
    ) -> Result<Self, GqaTpError> {
        if group.world_size() != 2
            || group.rank() >= 2
            || max_ctx == 0
            || max_ctx > u32::MAX as usize
        {
            return Err(GqaTpError::Shape(
                "expected TP=2 and nonempty u32 context".into(),
            ));
        }
        let u = |key: &str| -> Result<usize, GqaTpError> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| GqaTpError::Shape(format!("missing or invalid {key}")))
        };
        let f = |key: &str| map.gguf().arch_field(key).and_then(Value::as_f32);
        let g = GqaGeometry::new(
            u("embedding_length")?,
            u("attention.head_count")?,
            u("attention.head_count_kv")?,
            u("attention.key_length")?,
            group.rank(),
        )?;
        let nrot = u("rope.dimension_count")?;
        let eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let base = f("rope.freq_base").unwrap_or(1e7);
        if nrot == 0
            || nrot > g.head_dim
            || nrot % 2 != 0
            || !eps.is_finite()
            || eps <= 0.0
            || !base.is_finite()
            || base <= 0.0
        {
            return Err(GqaTpError::Shape("invalid rope or norm metadata".into()));
        }
        let kv_bytes = g
            .kv_bytes(max_ctx, dtype)
            .ok_or_else(|| GqaTpError::Shape("KV byte count overflow".into()))?;
        let qdim = g
            .heads
            .checked_mul(g.head_dim)
            .ok_or_else(|| GqaTpError::Shape("Q width overflow".into()))?;
        let qg_dim = qdim
            .checked_mul(2)
            .ok_or_else(|| GqaTpError::Shape("Q/gate width overflow".into()))?;
        let name = |part: &str| format!("blk.{layer}.{part}.weight");
        for (part, dims) in [
            ("attn_q", [g.width, qg_dim]),
            ("attn_k", [g.width, g.kv_heads * g.head_dim]),
            ("attn_v", [g.width, g.kv_heads * g.head_dim]),
            ("attn_output", [qdim, g.width]),
            ("attn_q_norm", [g.head_dim, 0]),
            ("attn_k_norm", [g.head_dim, 0]),
        ] {
            let (info, _) = map.tensor_bytes(&name(part)).map_err(GpuError::from)?;
            let expected: Vec<u64> = if dims[1] == 0 {
                vec![dims[0] as u64]
            } else {
                dims.into_iter().map(|v| v as u64).collect()
            };
            if info.dims != expected {
                return Err(GqaTpError::Shape(format!(
                    "{}: unexpected dimensions {:?}",
                    name(part),
                    info.dims
                )));
            }
        }
        let request = |kind| TensorSliceRequest {
            kind,
            rank: group.rank(),
            world_size: 2,
        };
        let weights = [
            e.load_quantw_shard(map, &name("attn_q"), request(ShardKind::OutputRows))?,
            e.load_quantw_shard(map, &name("attn_k"), request(ShardKind::OutputRows))?,
            e.load_quantw_shard(map, &name("attn_v"), request(ShardKind::OutputRows))?,
            e.load_quantw_shard(map, &name("attn_output"), request(ShardKind::InputColumns))?,
        ];
        let train = u("context_length").unwrap_or(max_ctx);
        let yarn = YarnRope::new(nrot, base, 1.0, train, 0.0, 1.0, 32.0, 1.0).kernel_params();
        Ok(Self {
            geometry: g,
            weights,
            qnorm: e.upload(map, &name("attn_q_norm"))?.buf,
            knorm: e.upload(map, &name("attn_k_norm"))?.buf,
            sinks: e.alloc_no_sinks(g.local_heads)?,
            qg: e.alloc(qdim)?,
            q: e.alloc(g.q_dim())?,
            gate: e.alloc(g.q_dim())?,
            k: e.alloc(g.kv_dim())?,
            v: e.alloc(g.kv_dim())?,
            qn: e.alloc(g.q_dim())?,
            kn: e.alloc(g.kv_dim())?,
            attn: e.alloc(g.q_dim())?,
            partial: e.alloc(g.width)?,
            reduced: e.alloc(g.width)?,
            kc: e.alloc_u8(kv_bytes)?,
            vc: e.alloc_u8(kv_bytes)?,
            positions: e.alloc_u32(1)?,
            slots: e.alloc_u32(1)?,
            axes: e.alloc_u32(4)?,
            block_tables: None,
            blocks_per_slot: 0,
            slots_count: 1,
            pos: 0,
            max_ctx,
            dtype,
            rank: group.rank(),
            nrot,
            eps,
            yarn,
            sections: read_sections(map)?,
        })
    }
    /// Isolated paged mode: the caller owns mirrored logical tables; each rank
    /// allocates only its local K/V head payload for the same physical block IDs.
    pub fn load_paged<C: Communicator>(
        e: &GpuExecutor,
        map: &MappedGguf,
        layer: usize,
        group: &C,
        max_ctx: usize,
        dtype: KvDtype,
        blocks: u32,
        slots: usize,
    ) -> Result<Self, GqaTpError> {
        let bps = max_ctx.div_ceil(BLOCK_TOKENS);
        if blocks == 0
            || slots == 0
            || bps == 0
            || bps.checked_mul(slots).is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(GqaTpError::Shape("invalid paged KV capacity".into()));
        }
        let mut rank = Self::load(e, map, layer, group, max_ctx, dtype)?;
        let bytes = rank
            .geometry
            .kv_bytes(
                (blocks as usize)
                    .checked_mul(BLOCK_TOKENS)
                    .ok_or_else(|| GqaTpError::Shape("paged KV size overflow".into()))?,
                dtype,
            )
            .ok_or_else(|| GqaTpError::Shape("paged KV size overflow".into()))?;
        rank.kc = e.alloc_u8(bytes)?;
        rank.vc = e.alloc_u8(bytes)?;
        rank.block_tables = Some(e.alloc_u32(bps * slots)?);
        rank.blocks_per_slot = bps;
        rank.slots_count = slots;
        Ok(rank)
    }
    /// Bytes charged to each rank for one physical block across this GQA layer.
    pub fn local_block_bytes(&self) -> Option<usize> {
        self.geometry
            .kv_bytes(BLOCK_TOKENS, self.dtype)?
            .checked_mul(2)
    }

    /// One physical block's byte stride within this rank's K (or V) slab.
    pub fn block_stride(&self) -> usize {
        BLOCK_TOKENS * self.geometry.kv_dim() * self.dtype.bytes()
    }

    /// The rank-local K and V payload slabs (paged mode: block-strided).
    pub fn kv_slabs(&self) -> (&CudaSlice<u8>, &CudaSlice<u8>) {
        (&self.kc, &self.vc)
    }

    pub fn kv_slabs_mut(&mut self) -> (&mut CudaSlice<u8>, &mut CudaSlice<u8>) {
        (&mut self.kc, &mut self.vc)
    }
    pub fn position(&self) -> usize {
        self.pos
    }
    pub fn reset(&mut self) {
        self.pos = 0;
    } // old KV is masked by position; next write replaces it
    pub fn local_kv_bytes(&self) -> usize {
        self.kc.len() + self.vc.len()
    }

    pub fn forward<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
    ) -> Result<&'a CudaSlice<f32>, GqaTpError> {
        if self.block_tables.is_some() {
            return Err(GqaTpError::Shape("use forward_paged for paged KV".into()));
        }
        self.forward_at(e, group, input, 0, self.pos, None)
    }

    /// Run one rank-0-authorized logical slot/position through paged GPU KV.
    /// Validate the entire read prefix against the mirrored live block pool.
    pub fn forward_paged<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
        slot: usize,
        position: usize,
        logical: &MirroredKv,
    ) -> Result<&'a CudaSlice<f32>, GqaTpError> {
        if self.block_tables.is_none() {
            return Err(GqaTpError::Shape("not a paged GQA rank".into()));
        }
        let stride = BLOCK_TOKENS * self.geometry.kv_dim() * self.dtype.bytes();
        let blocks = u32::try_from(self.kc.len() / stride)
            .map_err(|_| GqaTpError::Shape("KV pool too large".into()))?;
        let table = logical
            .checked_device_table(slot, position, blocks, self.slots_count, self.max_ctx)
            .map_err(|e| GqaTpError::Shape(e.into()))?;
        self.forward_at(e, group, input, slot, position, Some(&table))
    }

    fn forward_at<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
        slot: usize,
        position: usize,
        table: Option<&[u32]>,
    ) -> Result<&'a CudaSlice<f32>, GqaTpError> {
        if group.world_size() != 2
            || group.rank() != self.rank
            || input.len() != self.geometry.width
            || input.context().cu_ctx() != e.stream.context().cu_ctx()
            || position >= self.max_ctx
        {
            return Err(GqaTpError::Shape(
                "rank, input or context limit changed".into(),
            ));
        }
        let g = self.geometry;
        let pos =
            u32::try_from(position).map_err(|_| GqaTpError::Shape("position overflow".into()))?;
        e.stream
            .memcpy_htod(&[pos], &mut self.positions)
            .map_err(GpuError::from)?;
        e.stream
            .memcpy_htod(&[pos; 4], &mut self.axes)
            .map_err(GpuError::from)?;
        e.stream
            .memcpy_htod(&[slot as u32], &mut self.slots)
            .map_err(GpuError::from)?;
        if let Some(host) = table {
            e.stream
                .memcpy_htod(host, self.block_tables.as_mut().expect("paged checked"))
                .map_err(GpuError::from)?;
        }
        gemv_any(e, &self.weights[0], input, &mut self.qg)?;
        gemv_any(e, &self.weights[1], input, &mut self.k)?;
        gemv_any(e, &self.weights[2], input, &mut self.v)?;
        e.split_qg(
            &self.qg,
            &mut self.q,
            &mut self.gate,
            1,
            g.local_heads,
            g.head_dim,
        )?;
        e.rmsnorm_batch(
            &self.q,
            &self.qnorm,
            &mut self.qn,
            g.head_dim,
            self.eps,
            g.local_heads,
        )?;
        e.rmsnorm_batch(
            &self.k,
            &self.knorm,
            &mut self.kn,
            g.head_dim,
            self.eps,
            g.local_kv_heads,
        )?;
        e.mrope(
            &mut self.qn,
            &self.axes,
            1,
            g.local_heads,
            g.head_dim,
            self.nrot,
            self.yarn,
            self.sections,
        )?;
        e.mrope(
            &mut self.kn,
            &self.axes,
            1,
            g.local_kv_heads,
            g.head_dim,
            self.nrot,
            self.yarn,
            self.sections,
        )?;
        if let Some(bt) = self.block_tables.as_ref() {
            e.kv_append_batch_paged(
                &self.kn,
                &mut self.kc,
                &self.positions,
                Some(&self.slots),
                bt,
                self.blocks_per_slot,
                g.kv_dim(),
                1,
                self.dtype,
            )?;
            e.kv_append_batch_paged(
                &self.v,
                &mut self.vc,
                &self.positions,
                Some(&self.slots),
                bt,
                self.blocks_per_slot,
                g.kv_dim(),
                1,
                self.dtype,
            )?;
            e.attn_decode_batch_paged(
                &self.qn,
                &self.kc,
                &self.vc,
                &self.sinks,
                &mut self.attn,
                &self.positions,
                Some(&self.slots),
                bt,
                self.blocks_per_slot,
                g.local_heads,
                g.local_kv_heads,
                g.head_dim,
                g.kv_dim(),
                0,
                1,
                1.0 / (g.head_dim as f32).sqrt(),
                self.dtype,
            )?;
        } else {
            e.kv_append_batch(
                &self.kn,
                &mut self.kc,
                &self.positions,
                Some(&self.slots),
                g.kv_dim(),
                self.max_ctx,
                1,
                self.dtype,
            )?;
            e.kv_append_batch(
                &self.v,
                &mut self.vc,
                &self.positions,
                Some(&self.slots),
                g.kv_dim(),
                self.max_ctx,
                1,
                self.dtype,
            )?;
            e.attn_decode_batch(
                &self.qn,
                &self.kc,
                &self.vc,
                &self.sinks,
                &mut self.attn,
                &self.positions,
                Some(&self.slots),
                g.local_heads,
                g.local_kv_heads,
                g.head_dim,
                self.max_ctx,
                g.kv_dim(),
                0,
                1,
                1.0 / (g.head_dim as f32).sqrt(),
                self.dtype,
            )?;
        }
        e.mul_sigmoid(&mut self.attn, &self.gate, g.q_dim())?;
        gemv_any(e, &self.weights[3], &self.attn, &mut self.partial)?;
        group.after_compute(&e.stream)?;
        group.all_reduce(&self.partial, &mut self.reduced)?;
        group.before_compute(&e.stream)?;
        if table.is_none() {
            self.pos += 1;
        }
        Ok(&self.reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_groups_and_kv_payload() {
        for rank in 0..2 {
            let g = GqaGeometry::new(5120, 24, 4, 256, rank).unwrap();
            assert_eq!(
                (g.local_heads, g.local_kv_heads, g.kv_start),
                (12, 2, rank * 2)
            );
            assert_eq!(g.kv_bytes(64, KvDtype::Fp16), Some(64 * 2 * 256 * 2));
            assert_eq!(
                g.kv_bytes(BLOCK_TOKENS, KvDtype::Fp16)
                    .and_then(|n| n.checked_mul(2)),
                Some(32_768)
            );
        }
        for (h, kv, rank) in [(24, 3, 0), (23, 4, 0), (24, 4, 2), (0, 4, 0)] {
            assert!(GqaGeometry::new(5120, h, kv, 256, rank).is_err());
        }
        assert_eq!(
            GqaGeometry::new(5120, 24, 4, 256, 0)
                .unwrap()
                .kv_bytes(usize::MAX, KvDtype::Fp16),
            None
        );
    }
}
