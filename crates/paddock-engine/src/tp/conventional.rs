//! Conventional (non-gated, standard-RoPE) GQA tensor-parallel rank.
//!
//! The default attention surface: standard Q/K/V column-parallel projections,
//! optional per-head Q/K RMS norms, optional attention sinks, batched YARN
//! RoPE (or none), paged mirrored KV, and the row-parallel output projection
//! with the generic sum reduction. Geometry, sharding, staging, kernel
//! dispatch and the collective all live here; the model supplies names,
//! dimensions and narrow policy hooks.
//!
//! Execution shape mirrors the proven Qwen TP lane: decode advances ONE row
//! per forward with single-row GEMVs, and prompt rows traverse as row-batched
//! spans through the shared projection-prefill backend. Deliberately NOT part
//! of this surface: fused Q+gate projections, M-RoPE axis planes and hybrid
//! layer sequencing — those stay in model adapters (Qwen keeps its gated
//! Q+gate and M-RoPE path in `gpu_model/qwen35/`).

use cudarc::driver::CudaSlice;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;
use paddock_models::tensor_slice::{ShardKind, gguf_shard};

use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::projection::gemv_quant;
use crate::kv_pool::BLOCK_TOKENS;
use crate::tp::attention::{
    AttentionTpError, AttentionTpWeights, AttentionWeightNames, GqaPartition, PagedAttentionTable,
    PagedKvBatch, PagedPrefillPolicy, decode_paged, prefill_paged, reduce_output,
};
use crate::tp::cache::MirroredKv;
use crate::tp::prefill::{ProjectionPrefillBackend, ProjectionStaging};
use crate::tp::{TpLinearMode, TpTopology, TpTopologyError};

#[derive(Debug, thiserror::Error)]
pub enum ConventionalTpError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] GpuModelError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error(transparent)]
    Topology(#[from] TpTopologyError),
    #[error(transparent)]
    Attention(#[from] AttentionTpError),
    #[error("conventional GQA TP: {0}")]
    Shape(String),
}

/// Model declaration for one conventional GQA attention layer.
///
/// All weight names are full checkpoint tensor names for ONE layer; the rank
/// derives its geometry from the tensors themselves and validates it against
/// the declared head counts. Norm and sink names are optional: `None` loads
/// the exact numerical identity (-inf sink vector, no normalization).
pub struct ConventionalAttentionSpec<'a> {
    pub names: AttentionWeightNames<'a>,
    pub q_norm: Option<&'a str>,
    pub k_norm: Option<&'a str>,
    pub sinks: Option<&'a str>,
    /// RMS norm epsilon, used only when a norm name is supplied.
    pub eps: f32,
    /// Batched YARN RoPE applied to Q and K, or `None` for no positional
    /// encoding. Exotic position encodings (M-RoPE, interleaved sections) are
    /// model overrides, not extensions of this spec.
    pub rope: Option<RopeParams>,
    /// Prefill kernel-eligibility policy (F16/tiled preferences). Kernel
    /// availability is checked by the generic dispatch; this only records the
    /// model's measured eligibility decision.
    pub prefill_policy: PagedPrefillPolicy,
}

/// Parameters for the batched YARN RoPE kernels, typically produced host-side
/// by `YarnRope::kernel_params`.
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    pub params: (f32, f32, f32, f32, f32, f32),
    /// `false` = NEOX pair convention (the default for everything this engine
    /// serves except Granite); `true` selects the NORM-convention kernel.
    pub norm_convention: bool,
}

/// Decode split-count policy. The model owns the measured policy; generic TP
/// never invents a split count. The count is fixed at load and sized into the
/// partial scratch, so it stays constant across graph replays.
///
/// Pending seam: no model adapter constructs these variants yet (see the
/// module-level `allow` on `ConventionalGqaRank`).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeSplitPolicy {
    /// Single-pass paged decode kernel.
    SinglePass,
    /// Fixed split partial+combine with `n` splits per head (`n >= 2`).
    Splits(usize),
}

impl DecodeSplitPolicy {
    fn resolve(self) -> Result<usize, ConventionalTpError> {
        match self {
            Self::SinglePass => Ok(1),
            Self::Splits(n) if n > 1 => Ok(n),
            Self::Splits(_) => Err(ConventionalTpError::Shape(
                "decode split policy requires at least two splits; use SinglePass".into(),
            )),
        }
    }
}

/// Rank-local geometry of one conventional GQA attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConventionalGeometry {
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub local_heads: usize,
    pub local_kv_heads: usize,
    pub kv_start: usize,
}

impl ConventionalGeometry {
    /// Derive rank-local geometry from the declared model shape and the
    /// rank's topology. Pure and host-testable; `load` calls this.
    pub fn new(
        topology: TpTopology,
        width: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ConventionalTpError> {
        if width == 0 || head_dim == 0 {
            return Err(ConventionalTpError::Shape(
                "attention width and head_dim must be nonzero".into(),
            ));
        }
        let partition = GqaPartition::new(topology, heads, kv_heads)
            .map_err(|err| ConventionalTpError::Shape(err.to_string()))?;
        Ok(Self {
            width,
            heads,
            kv_heads,
            head_dim,
            local_heads: partition.local_heads,
            local_kv_heads: partition.local_kv_heads,
            kv_start: partition.kv_start,
        })
    }

    pub fn q_dim(self) -> usize {
        self.local_heads * self.head_dim
    }

    pub fn kv_dim(self) -> usize {
        self.local_kv_heads * self.head_dim
    }
}

/// Q and K norm hooks must be supplied together: applying exactly one of the
/// pair silently changes the Q/K numeric relationship every checkpoint that
/// ships both relies on. Pure and host-testable.
fn check_norm_pairing(q_norm: bool, k_norm: bool) -> Result<(), ConventionalTpError> {
    if q_norm != k_norm {
        return Err(ConventionalTpError::Shape(
            "Q and K norms must be supplied together".into(),
        ));
    }
    Ok(())
}

/// Validate a model's attention geometry against a rank's topology BEFORE
/// any weight load: complete Q groups following the rank's KV head, even KV
/// split, nonzero dimensions. This is the load-time refusal rule every
/// conventional model gets from the generic layer (Qwen's GqaGeometry
/// applies the same rule through `GqaPartition`). Pure and host-testable.
/// Pending seam: exercised by model #2's TP shim (`gpu_model/minicpm`).
#[allow(dead_code)]
pub(crate) fn validate_load_geometry(
    topology: TpTopology,
    width: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(), ConventionalTpError> {
    ConventionalGeometry::new(topology, width, heads, kv_heads, head_dim).map(|_| ())
}

/// Contiguous-span range validation shared by prefill staging. Pure.
fn validate_span_range(
    position: usize,
    rows: usize,
    max_ctx: usize,
) -> Result<(), ConventionalTpError> {
    if rows == 0
        || position
            .checked_add(rows)
            .is_none_or(|end| end > max_ctx)
    {
        return Err(ConventionalTpError::Shape("span geometry out of range".into()));
    }
    Ok(())
}

/// Row-batched prefill scratch for the conventional attention rank: the
/// projection/norm/attention planes, the capacity-sized all-reduce pair, and
/// the span-global metadata planes (per-row positions, slot ids).
pub struct ConventionalPrefillScratch {
    pub(crate) cap: usize,
    pub(crate) q: CudaSlice<f32>,
    pub(crate) k: CudaSlice<f32>,
    pub(crate) v: CudaSlice<f32>,
    pub(crate) qn: CudaSlice<f32>,
    pub(crate) kn: CudaSlice<f32>,
    pub(crate) attn: CudaSlice<f32>,
    pub(crate) partial: CudaSlice<f32>,
    pub(crate) reduced: CudaSlice<f32>,
    pub(crate) positions: CudaSlice<u32>,
    pub(crate) slots: CudaSlice<u32>,
}

/// Pending seam: no model adapter constructs the scratch yet (see the
/// module-level `allow` on `ConventionalGqaRank`).
#[allow(dead_code)]
impl ConventionalPrefillScratch {
    pub fn new(
        exec: &GpuExecutor,
        geometry: ConventionalGeometry,
        cap: usize,
    ) -> Result<Self, GpuError> {
        if cap == 0 || geometry.width == 0 || geometry.q_dim() == 0 || geometry.kv_dim() == 0 {
            return Err(GpuError::Unsupported(
                "attention prefill scratch requires nonzero geometry".into(),
            ));
        }
        Ok(Self {
            cap,
            q: exec.alloc(cap * geometry.q_dim())?,
            k: exec.alloc(cap * geometry.kv_dim())?,
            v: exec.alloc(cap * geometry.kv_dim())?,
            qn: exec.alloc(cap * geometry.q_dim())?,
            kn: exec.alloc(cap * geometry.kv_dim())?,
            attn: exec.alloc(cap * geometry.q_dim())?,
            partial: exec.alloc(cap * geometry.width)?,
            reduced: exec.alloc(cap * geometry.width)?,
            positions: exec.alloc_u32(cap)?,
            slots: exec.alloc_u32(cap)?,
        })
    }
}

/// One rank of a conventional GQA attention layer over paged mirrored KV.
///
/// Execution is split into a staging phase, a collective-free run, and the
/// output reduction, so a graph-capturing caller can replay the run with
/// re-staged inputs exactly like the Qwen adapter does.
pub struct ConventionalGqaRank {
    geometry: ConventionalGeometry,
    weights: AttentionTpWeights,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    sinks: CudaSlice<f32>,
    rope: Option<RopeParams>,
    eps: f32,
    prefill_policy: PagedPrefillPolicy,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    attn_o: CudaSlice<f32>,
    attn_ml: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    reduced: CudaSlice<f32>,
    positions: CudaSlice<u32>,
    slots: CudaSlice<u32>,
    block_table: PagedAttentionTable,
    kc: CudaSlice<u8>,
    vc: CudaSlice<u8>,
    max_ctx: usize,
    dtype: KvDtype,
    n_splits: usize,
}

fn dims_2d(map: &MappedGguf, name: &str) -> Result<[usize; 2], ConventionalTpError> {
    let (info, _) = map.tensor_bytes(name).map_err(GpuError::from)?;
    if info.dims.len() != 2 {
        return Err(ConventionalTpError::Shape(format!("{name} must be 2-D")));
    }
    Ok([
        usize::try_from(info.dims[0])
            .map_err(|_| ConventionalTpError::Shape(format!("{name}: input overflow")))?,
        usize::try_from(info.dims[1])
            .map_err(|_| ConventionalTpError::Shape(format!("{name}: output overflow")))?,
    ])
}

fn dims_1d(map: &MappedGguf, name: &str) -> Result<usize, ConventionalTpError> {
    let (info, _) = map.tensor_bytes(name).map_err(GpuError::from)?;
    if info.dims.len() != 1 {
        return Err(ConventionalTpError::Shape(format!("{name} must be 1-D")));
    }
    usize::try_from(info.dims[0])
        .map_err(|_| ConventionalTpError::Shape(format!("{name}: length overflow")))
}

/// Rank-local load of a 1-D f32 tensor. `Replicated` keeps the whole vector;
/// `OutputRows` slices the head axis so each rank holds its own heads' sinks.
fn load_f32_1d(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    kind: ShardKind,
    topology: TpTopology,
) -> Result<CudaSlice<f32>, ConventionalTpError> {
    let len = dims_1d(map, name)?;
    let local = match kind {
        ShardKind::Replicated => len,
        ShardKind::OutputRows => topology.local_len(len)?,
        _ => {
            return Err(ConventionalTpError::Shape(format!(
                "{name}: unsupported shard kind"
            )))
        }
    };
    let bytes: std::borrow::Cow<'_, [u8]> = if local == len {
        let (info, bytes) = map.tensor_bytes(name).map_err(GpuError::from)?;
        if info.ggml_type != GgmlType::F32 {
            return Err(ConventionalTpError::Shape(format!(
                "{name}: expected f32 tensor"
            )));
        }
        std::borrow::Cow::Borrowed(bytes)
    } else {
        let (ty, shard) = gguf_shard(map, name, topology.tensor_slice(kind))
            .map_err(|err| ConventionalTpError::Shape(err.to_string()))?;
        if ty != GgmlType::F32 {
            return Err(ConventionalTpError::Shape(format!(
                "{name}: expected f32 tensor"
            )));
        }
        shard.bytes
    };
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if values.len() != local {
        return Err(ConventionalTpError::Shape(format!(
            "{name}: expected {local} local elements, read {}",
            values.len()
        )));
    }
    Ok(exec.stream.clone_htod(&values).map_err(GpuError::from)?)
}

/// Documented pending seam: the conventional surface is the generic default
/// for a future second model; Qwen's gated adapter does not call it yet, so
/// these items are dead until that phase lands (review handoff, "next work"
/// #2). Remove narrowly when a real production caller exists.
#[allow(dead_code)]
impl ConventionalGqaRank {
    /// Load one layer for this rank's shard of the communicator.
    ///
    /// `heads`, `kv_heads` and `head_dim` are model-declared; the hidden width
    /// is derived from the Q tensor and cross-checked across all four
    /// projections. The rank-local KV pools cover `pool_blocks` physical
    /// blocks whose IDs are shared with the mirrored pool.
    #[allow(clippy::too_many_arguments)]
    pub fn load<C: Communicator>(
        exec: &GpuExecutor,
        map: &MappedGguf,
        spec: ConventionalAttentionSpec<'_>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        group: &C,
        max_ctx: usize,
        slots: usize,
        pool_blocks: u32,
        dtype: KvDtype,
        splits: DecodeSplitPolicy,
    ) -> Result<Self, ConventionalTpError> {
        let topology = TpTopology::from_group(group)?;
        if max_ctx == 0
            || max_ctx > u32::MAX as usize
            || slots == 0
            || pool_blocks == 0
            || pool_blocks as usize > u32::MAX as usize
            || pool_blocks as usize * BLOCK_TOKENS > u32::MAX as usize
        {
            return Err(ConventionalTpError::Shape(
                "invalid attention capacity".into(),
            ));
        }
        let n_splits = splits.resolve()?;
        let q_dim = heads.checked_mul(head_dim).filter(|&n| n > 0).ok_or_else(|| {
            ConventionalTpError::Shape("Q width overflow".into())
        })?;
        let kv_dim = kv_heads.checked_mul(head_dim).filter(|&n| n > 0).ok_or_else(|| {
            ConventionalTpError::Shape("KV width overflow".into())
        })?;
        check_norm_pairing(spec.q_norm.is_some(), spec.k_norm.is_some())?;
        let [width, q_out] = dims_2d(map, spec.names.q)?;
        if q_out != q_dim
            || dims_2d(map, spec.names.k)? != [width, kv_dim]
            || dims_2d(map, spec.names.v)? != [width, kv_dim]
            || dims_2d(map, spec.names.output)? != [q_dim, width]
        {
            return Err(ConventionalTpError::Shape(
                "attention projection dimensions disagree with declared heads".into(),
            ));
        }
        for name in [spec.q_norm, spec.k_norm] {
            if let Some(name) = name {
                if dims_1d(map, name)? != head_dim {
                    return Err(ConventionalTpError::Shape(format!(
                        "{name}: expected head_dim"
                    )));
                }
            }
        }
        if let Some(name) = spec.sinks {
            if dims_1d(map, name)? != heads {
                return Err(ConventionalTpError::Shape(format!(
                    "{name}: expected {heads} heads"
                )));
            }
        }
        let geometry = ConventionalGeometry::new(topology, width, heads, kv_heads, head_dim)?;
        let weights = AttentionTpWeights::load(exec, map, spec.names, topology)?;
        let q_norm = match spec.q_norm {
            Some(name) => Some(exec.upload(map, name)?.buf),
            None => None,
        };
        let k_norm = match spec.k_norm {
            Some(name) => Some(exec.upload(map, name)?.buf),
            None => None,
        };
        let sinks = match spec.sinks {
            Some(name) => load_f32_1d(
                exec,
                map,
                name,
                TpLinearMode::ColumnParallel.shard_kind(),
                topology,
            )?,
            None => exec.alloc_no_sinks(geometry.local_heads)?,
        };
        let pool_bytes = pool_blocks
            .checked_mul(BLOCK_TOKENS as u32)
            .and_then(|n| usize::try_from(n).ok())
            .and_then(|tokens| tokens.checked_mul(geometry.kv_dim()))
            .and_then(|elems| elems.checked_mul(dtype.bytes()))
            .ok_or_else(|| ConventionalTpError::Shape("KV pool byte count overflow".into()))?;
        Ok(Self {
            geometry,
            weights,
            q_norm,
            k_norm,
            sinks,
            rope: spec.rope,
            eps: spec.eps,
            prefill_policy: spec.prefill_policy,
            q: exec.alloc(geometry.q_dim())?,
            k: exec.alloc(geometry.kv_dim())?,
            v: exec.alloc(geometry.kv_dim())?,
            qn: exec.alloc(geometry.q_dim())?,
            kn: exec.alloc(geometry.kv_dim())?,
            attn: exec.alloc(geometry.q_dim())?,
            attn_o: exec.alloc(geometry.local_heads * n_splits * head_dim)?,
            attn_ml: exec.alloc(geometry.local_heads * n_splits * 2)?,
            partial: exec.alloc(width)?,
            reduced: exec.alloc(width)?,
            positions: exec.alloc_u32(1)?,
            slots: exec.alloc_u32(1)?,
            block_table: PagedAttentionTable::new(exec, slots, max_ctx)?,
            kc: exec.alloc_u8(pool_bytes)?,
            vc: exec.alloc_u8(pool_bytes)?,
            max_ctx,
            dtype,
            n_splits,
        })
    }

    /// The topology these weights were sharded for.
    pub fn topology(&self) -> TpTopology {
        self.weights.topology()
    }

    pub fn geometry(&self) -> ConventionalGeometry {
        self.geometry
    }

    /// Bytes charged to each rank for one physical block across this layer.
    pub fn local_block_bytes(&self) -> Option<usize> {
        self.geometry
            .kv_dim()
            .checked_mul(BLOCK_TOKENS)?
            .checked_mul(self.dtype.bytes())?
            .checked_mul(2)
    }

    /// One physical block's byte stride within this rank's K (or V) slab.
    pub fn block_stride(&self) -> usize {
        BLOCK_TOKENS * self.geometry.kv_dim() * self.dtype.bytes()
    }

    /// The rank-local K and V payload slabs.
    pub fn kv_slabs(&self) -> (&CudaSlice<u8>, &CudaSlice<u8>) {
        (&self.kc, &self.vc)
    }

    fn validate_group<C: Communicator>(&self, group: &C) -> Result<(), ConventionalTpError> {
        if TpTopology::from_group(group)? != self.topology() {
            return Err(ConventionalTpError::Shape(
                "communicator topology changed after attention load".into(),
            ));
        }
        Ok(())
    }

    /// Stage one decode row: validate the slot's block coverage against the
    /// mirrored logical pool and upload only changed mappings, then upload
    /// the position and slot id. All ranks call this with the same row, so
    /// validation fails symmetrically before any rank enters a collective.
    pub fn stage_row(
        &mut self,
        exec: &GpuExecutor,
        logical: &MirroredKv,
        slot: usize,
        position: usize,
    ) -> Result<(), ConventionalTpError> {
        if position >= self.max_ctx {
            return Err(ConventionalTpError::Shape("row position out of range".into()));
        }
        let pool_blocks = u32::try_from(self.kc.len() / self.block_stride())
            .map_err(|_| ConventionalTpError::Shape("KV pool too large".into()))?;
        self.block_table
            .stage_slot(exec, logical, slot, position, pool_blocks)?;
        let pos =
            u32::try_from(position).map_err(|_| ConventionalTpError::Shape("position overflow".into()))?;
        let slot =
            u32::try_from(slot).map_err(|_| ConventionalTpError::Shape("slot id exceeds u32".into()))?;
        exec.stream
            .memcpy_htod(&[pos], &mut self.positions)
            .map_err(GpuError::from)?;
        exec.stream
            .memcpy_htod(&[slot], &mut self.slots)
            .map_err(GpuError::from)?;
        Ok(())
    }

    /// The collective-free one-row decode run: Q/K/V projections, optional
    /// per-head norms, optional YARN RoPE, one paged KV pair append, paged
    /// decode (single-pass or fixed split+combine) and the row-parallel
    /// output projection into the partial plane. Suitable for graph capture
    /// with re-staged inputs.
    pub fn decode_run(
        &mut self,
        exec: &GpuExecutor,
        input: &CudaSlice<f32>,
    ) -> Result<(), ConventionalTpError> {
        if input.len() != self.geometry.width
            || input.context().cu_ctx() != exec.stream.context().cu_ctx()
        {
            return Err(ConventionalTpError::Shape(
                "attention input width/context changed".into(),
            ));
        }
        let g = self.geometry;
        gemv_quant(exec, self.weights.q(), input, &mut self.q)?;
        gemv_quant(exec, self.weights.k(), input, &mut self.k)?;
        gemv_quant(exec, self.weights.v(), input, &mut self.v)?;
        // Work planes are used when a norm or rope transform must be applied;
        // with neither, the raw projections feed attention directly (no copy,
        // no identity kernel).
        let transformed = self.q_norm.is_some() || self.rope.is_some();
        if transformed {
            match (&self.q_norm, &self.k_norm) {
                (Some(qn_w), Some(kn_w)) => {
                    exec.rmsnorm_batch(
                        &self.q,
                        qn_w,
                        &mut self.qn,
                        g.head_dim,
                        self.eps,
                        g.local_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &self.k,
                        kn_w,
                        &mut self.kn,
                        g.head_dim,
                        self.eps,
                        g.local_kv_heads,
                    )?;
                }
                (None, None) => {
                    exec.stream
                        .memcpy_dtod(&self.q, &mut self.qn)
                        .map_err(GpuError::from)?;
                    exec.stream
                        .memcpy_dtod(&self.k, &mut self.kn)
                        .map_err(GpuError::from)?;
                }
                _ => {
                    return Err(ConventionalTpError::Shape(
                        "Q and K norms must be supplied together".into(),
                    ));
                }
            }
            if let Some(rope) = self.rope {
                self.apply_rope(exec, rope)?;
            }
        }
        let (q_in, k_in) = if transformed {
            (&self.qn, &self.kn)
        } else {
            (&self.q, &self.k)
        };
        let kv = PagedKvBatch::new(
            &self.positions,
            &self.slots,
            self.block_table.device(),
            self.block_table.blocks_per_slot(),
            g.kv_dim(),
            1,
            self.dtype,
        )?;
        kv.append(exec, k_in, &self.v, &mut self.kc, &mut self.vc)?;
        let (bt, blocks_per_slot) = kv.table();
        decode_paged(
            exec,
            q_in,
            &self.kc,
            &self.vc,
            &self.sinks,
            &mut self.attn_o,
            &mut self.attn_ml,
            &mut self.attn,
            &self.positions,
            Some(&self.slots),
            bt,
            blocks_per_slot,
            g.local_heads,
            g.local_kv_heads,
            g.head_dim,
            g.kv_dim(),
            1,
            1.0 / (g.head_dim as f32).sqrt(),
            self.dtype,
            self.n_splits,
        )?;
        gemv_quant(exec, self.weights.output(), &self.attn, &mut self.partial)?;
        Ok(())
    }

    fn apply_rope(
        &mut self,
        exec: &GpuExecutor,
        rope: RopeParams,
    ) -> Result<(), ConventionalTpError> {
        let g = self.geometry;
        let f = |exec: &GpuExecutor, x: &mut CudaSlice<f32>, heads: usize| {
            if rope.norm_convention {
                exec.rope_yarn_batch_norm(x, &self.positions, heads, g.head_dim, rope.params, 1)
            } else {
                exec.rope_yarn_batch(x, &self.positions, heads, g.head_dim, rope.params, 1)
            }
        };
        f(exec, &mut self.qn, g.local_heads)?;
        f(exec, &mut self.kn, g.local_kv_heads)?;
        Ok(())
    }

    /// Complete the row-parallel output projection with the generic sum
    /// reduction over the live `width` prefix.
    pub fn decode_finish<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
    ) -> Result<&'a CudaSlice<f32>, ConventionalTpError> {
        self.validate_group(group)?;
        reduce_output(
            exec,
            group,
            self.topology(),
            &self.partial,
            &mut self.reduced,
        )?;
        Ok(&self.reduced)
    }

    /// Convenience path: stage, run and reduce one decode row in one call.
    /// The returned slice is the attention layer's hidden-width projection,
    /// identical on every rank after the sum reduction.
    pub fn decode<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
        logical: &MirroredKv,
        slot: usize,
        position: usize,
    ) -> Result<&'a CudaSlice<f32>, ConventionalTpError> {
        self.stage_row(exec, logical, slot, position)?;
        self.decode_run(exec, input)?;
        self.decode_finish(exec, group)
    }

    /// Stage the span-global metadata: validate the slot's block coverage
    /// through the span's last logical position and upload the per-row
    /// positions and slot ids into the scratch planes. Same mirrored
    /// semantics as [`Self::stage_row`], pointed at the span scratch.
    pub fn stage_span_metadata(
        &mut self,
        exec: &GpuExecutor,
        logical: &MirroredKv,
        slot: usize,
        position: usize,
        rows: usize,
        scratch: &mut ConventionalPrefillScratch,
    ) -> Result<(), ConventionalTpError> {
        if rows > scratch.cap {
            return Err(ConventionalTpError::Shape(
                "attention span rows exceed configured scratch".into(),
            ));
        }
        validate_span_range(position, rows, self.max_ctx)?;
        let pool_blocks = u32::try_from(self.kc.len() / self.block_stride())
            .map_err(|_| ConventionalTpError::Shape("KV pool too large".into()))?;
        // Validation covers the span's last logical position, matching the
        // Qwen span staging contract.
        self.block_table
            .stage_slot(exec, logical, slot, position + rows - 1, pool_blocks)?;
        let positions: Vec<u32> = (0..rows as u32).map(|r| position as u32 + r).collect();
        let slot_id = u32::try_from(slot)
            .map_err(|_| ConventionalTpError::Shape("slot id exceeds u32".into()))?;
        exec.stream
            .memcpy_htod(&positions, &mut scratch.positions.slice_mut(0..rows))
            .map_err(GpuError::from)?;
        exec.stream
            .memcpy_htod(&vec![slot_id; rows], &mut scratch.slots.slice_mut(0..rows))
            .map_err(GpuError::from)?;
        Ok(())
    }

    /// Collective-free row-batched prefill span: `rows` prompt rows of ONE
    /// slot at contiguous positions through this layer. Row order is
    /// preserved end-to-end: row `r` lands at `reduced[r * width ..]`.
    /// Call [`Self::stage_span_metadata`] first (or re-stage per replay).
    pub(crate) fn prefill_local<B: ProjectionPrefillBackend>(
        &mut self,
        exec: &GpuExecutor,
        backend: &B,
        input: &CudaSlice<f32>,
        rows: usize,
        scratch: &mut ConventionalPrefillScratch,
        staging: &mut ProjectionStaging,
    ) -> Result<(), ConventionalTpError> {
        if rows == 0
            || rows > scratch.cap
            || input.len() < rows * self.geometry.width
            || input.context().cu_ctx() != exec.stream.context().cu_ctx()
        {
            return Err(ConventionalTpError::Shape(
                "attention prefill rows/input exceed configured scratch".into(),
            ));
        }
        check_norm_pairing(self.q_norm.is_some(), self.k_norm.is_some())?;
        let g = self.geometry;
        staging.validate(g.width, rows)?;
        backend.prepare(exec, staging, input, g.width, rows)?;
        backend.project_prepared(exec, staging, self.weights.q(), &mut scratch.q, rows)?;
        backend.project_prepared(exec, staging, self.weights.k(), &mut scratch.k, rows)?;
        backend.project_prepared(exec, staging, self.weights.v(), &mut scratch.v, rows)?;
        // Work planes are used when a norm or rope transform must be applied;
        // with neither, the raw projections feed attention directly.
        let transformed = self.q_norm.is_some() || self.rope.is_some();
        if transformed {
            match (&self.q_norm, &self.k_norm) {
                (Some(qn_w), Some(kn_w)) => {
                    exec.rmsnorm_batch(
                        &scratch.q,
                        qn_w,
                        &mut scratch.qn,
                        g.head_dim,
                        self.eps,
                        rows * g.local_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &scratch.k,
                        kn_w,
                        &mut scratch.kn,
                        g.head_dim,
                        self.eps,
                        rows * g.local_kv_heads,
                    )?;
                }
                (None, None) => {
                    exec.stream
                        .memcpy_dtod(&scratch.q.slice(0..rows * g.q_dim()), &mut scratch.qn)
                        .map_err(GpuError::from)?;
                    exec.stream
                        .memcpy_dtod(&scratch.k.slice(0..rows * g.kv_dim()), &mut scratch.kn)
                        .map_err(GpuError::from)?;
                }
                _ => {
                    return Err(ConventionalTpError::Shape(
                        "Q and K norms must be supplied together".into(),
                    ));
                }
            }
            if let Some(rope) = self.rope {
                let f = |exec: &GpuExecutor,
                         x: &mut CudaSlice<f32>,
                         heads: usize|
                 -> Result<(), GpuError> {
                    if rope.norm_convention {
                        exec.rope_yarn_batch_norm(
                            x,
                            &scratch.positions,
                            heads,
                            g.head_dim,
                            rope.params,
                            rows,
                        )
                    } else {
                        exec.rope_yarn_batch(
                            x,
                            &scratch.positions,
                            heads,
                            g.head_dim,
                            rope.params,
                            rows,
                        )
                    }
                };
                f(exec, &mut scratch.qn, g.local_heads)?;
                f(exec, &mut scratch.kn, g.local_kv_heads)?;
            }
        }
        let (q_in, k_in) = if transformed {
            (&scratch.qn, &scratch.kn)
        } else {
            (&scratch.q, &scratch.k)
        };
        let kv = PagedKvBatch::new(
            &scratch.positions,
            &scratch.slots,
            self.block_table.device(),
            self.block_table.blocks_per_slot(),
            g.kv_dim(),
            rows,
            self.dtype,
        )?;
        kv.append(exec, k_in, &scratch.v, &mut self.kc, &mut self.vc)?;
        let (bt, blocks_per_slot) = kv.table();
        prefill_paged(
            exec,
            q_in,
            &self.kc,
            &self.vc,
            &self.sinks,
            &mut scratch.attn,
            &scratch.positions,
            &scratch.slots,
            bt,
            blocks_per_slot,
            g.local_heads,
            g.local_kv_heads,
            g.head_dim,
            g.kv_dim(),
            rows,
            1.0 / (g.head_dim as f32).sqrt(),
            self.dtype,
            self.prefill_policy,
        )?;
        // Q/K/V share staging prepared from the layer input above, but the
        // row-parallel output projection consumes the post-attention
        // activation. Re-prepare staging from attention before projecting O.
        backend.project(
            exec,
            staging,
            self.weights.output(),
            &scratch.attn,
            &mut scratch.partial,
            rows,
        )?;
        Ok(())
    }

    /// Finish the row-batched prefill span with the generic sum reduction
    /// over the live `rows * width` prefix.
    pub fn prefill_finish<'a, C: Communicator>(
        &'a mut self,
        exec: &GpuExecutor,
        group: &C,
        rows: usize,
        scratch: &'a mut ConventionalPrefillScratch,
    ) -> Result<&'a CudaSlice<f32>, ConventionalTpError> {
        self.validate_group(group)?;
        let live = rows
            .checked_mul(self.geometry.width)
            .filter(|&n| n > 0 && n <= scratch.partial.len())
            .ok_or_else(|| {
                ConventionalTpError::Shape("live prefix exceeds partial plane".into())
            })?;
        reduce_output(
            exec,
            group,
            self.topology(),
            &scratch.partial.slice(0..live),
            &mut scratch.reduced.slice_mut(0..live),
        )?;
        Ok(&scratch.reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A conventional layer's rank-local geometry follows the complete-group
    /// partition at every world size, exactly like the FFN shard pattern.
    #[test]
    fn conventional_geometry_partitions_complete_groups() {
        for (world, heads, kv_heads) in [(2usize, 24usize, 4usize), (4, 32, 8), (3, 24, 6)] {
            for rank in 0..world {
                let topology = TpTopology::new(rank, world).unwrap();
                let g = ConventionalGeometry::new(topology, 5120, heads, kv_heads, 128).unwrap();
                assert_eq!(g.local_kv_heads, kv_heads / world);
                assert_eq!(g.local_heads, heads / world);
                assert_eq!(g.kv_start, rank * g.local_kv_heads);
                assert_eq!(g.q_dim(), g.local_heads * 128);
                assert_eq!(g.kv_dim(), g.local_kv_heads * 128);
            }
        }
    }

    #[test]
    fn conventional_geometry_rejects_ragged_or_incomplete_groups() {
        let tp3 = TpTopology::new(0, 3).unwrap();
        assert!(ConventionalGeometry::new(tp3, 5120, 24, 4, 128).is_err());
        let tp2 = TpTopology::new(0, 2).unwrap();
        assert!(ConventionalGeometry::new(tp2, 5120, 23, 4, 128).is_err());
        assert!(ConventionalGeometry::new(tp2, 0, 24, 4, 128).is_err());
        assert!(ConventionalGeometry::new(tp2, 5120, 24, 4, 0).is_err());
    }

    #[test]
    fn norm_pairing_requires_both_or_neither() {
        assert!(check_norm_pairing(true, true).is_ok());
        assert!(check_norm_pairing(false, false).is_ok());
        assert!(check_norm_pairing(true, false).is_err());
        assert!(check_norm_pairing(false, true).is_err());
    }

    #[test]
    fn span_range_bounds_position_and_rows() {
        assert!(validate_span_range(0, 1, 48).is_ok());
        assert!(validate_span_range(16, 32, 48).is_ok());
        assert!(validate_span_range(0, 0, 48).is_err());
        assert!(validate_span_range(48, 1, 48).is_err());
        // Overflowing end position must refuse, not wrap.
        assert!(validate_span_range(usize::MAX - 2, 4, usize::MAX).is_err());
    }

    #[test]
    fn split_policy_resolves_fixed_counts_only() {
        assert_eq!(DecodeSplitPolicy::SinglePass.resolve().unwrap(), 1);
        assert_eq!(DecodeSplitPolicy::Splits(8).resolve().unwrap(), 8);
        // A one-split policy is a policy bug: decode_paged treats 1 as
        // single-pass, so the split scratch would never be used.
        assert!(DecodeSplitPolicy::Splits(1).resolve().is_err());
        assert!(DecodeSplitPolicy::Splits(0).resolve().is_err());
    }

    /// Until a production model exercises ConventionalGqaRank on GPU, keep a
    /// structural regression guard on the prefill O projection: it must
    /// re-prepare staging from the attention activation instead of reusing the
    /// hidden-state staging left by Q/K/V.
    #[test]
    fn prefill_output_projection_reprepares_from_attention() {
        let source = include_str!("conventional.rs");
        let start = source
            .find("row-parallel output projection consumes the post-attention")
            .expect("output projection marker");
        let tail = &source[start..];
        let end = tail.find("Ok(())").expect("end of prefill_local");
        let block = &tail[..end];

        assert!(block.contains("backend.project("));
        assert!(block.contains("&scratch.attn"));
        assert!(!block.contains("backend.project_prepared("));
    }
}
