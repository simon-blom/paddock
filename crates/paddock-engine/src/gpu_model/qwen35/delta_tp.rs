//! Isolated Qwen3.8 DeltaNet TP: decode and bounded native-span prefill.
//!
//! Adapted from Erik Bogado / ErikBPF/paddock `contrib/tp-10-deltanet-decode`
//! (8495d2d) and `contrib/tp-11-deltanet-prefill` (02ba232), MIT OR Apache-2.0.
//! Their complete key-group/value-head tiling and state/conv slicing inform
//! this rank-process port. The fork's same-process Link/copy/reduce and model
//! runner are not used: only the final projected hidden rows meet via NCCL.
use cudarc::driver::CudaSlice;
use paddock_models::ggml_type::GgmlType;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;
use paddock_models::tensor_slice::{ShardKind, TensorSliceRequest, gguf_shard};

use super::ops::{gemv_any, prefill_mm_any};
use crate::gpu::distributed::{CollectiveError, Communicator};
use crate::gpu::{GpuError, GpuExecutor, QuantW, RepackedQ8};
use crate::gpu_model::gpt_oss::GpuModelError;

const WIDTH: usize = 5120;
const S: usize = 128;
const NK: usize = 16;
const NV: usize = 48;
const CONV_K: usize = 4;
/// Span row cap of the DeltaNet batched prefill primitive. The whole-model
/// TP prefill span's cap (tp_span.rs `TP_SPAN_CAP`) is guarded against this
/// value by a source-reading test, since this constant stays private.
const SPAN_CAP: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum DeltaTpError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] GpuModelError),
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error("DeltaNet TP: {0}")]
    Shape(String),
}

/// Whole key groups: global value head `band*NK + key` reads that key.
/// Local ordering must retain three complete value bands for the existing
/// deltanet_split_gqa_norm and recurrent kernels to preserve their tiling.
#[derive(Debug, Clone)]
pub struct DeltaGeometry {
    pub key_heads: Vec<usize>,
    pub value_heads: Vec<usize>,
    pub channels: Vec<usize>,
}
impl DeltaGeometry {
    pub fn new(rank: usize) -> Result<Self, DeltaTpError> {
        if rank >= 2 {
            return Err(DeltaTpError::Shape("expected TP=2 rank 0 or 1".into()));
        }
        let key_heads: Vec<usize> = (rank * (NK / 2)..(rank + 1) * (NK / 2)).collect();
        let value_heads: Vec<usize> = (0..NV / NK)
            .flat_map(|band| key_heads.iter().map(move |&h| band * NK + h))
            .collect();
        let channels: Vec<usize> = key_heads
            .iter()
            .flat_map(|&h| h * S..(h + 1) * S)
            .chain(
                key_heads
                    .iter()
                    .flat_map(|&h| (NK + h) * S..(NK + h + 1) * S),
            )
            .chain(
                value_heads
                    .iter()
                    .flat_map(|&h| (2 * NK + h) * S..(2 * NK + h + 1) * S),
            )
            .collect();
        Ok(Self {
            key_heads,
            value_heads,
            channels,
        })
    }
    pub fn keys(&self) -> usize {
        self.key_heads.len()
    }
    pub fn values(&self) -> usize {
        self.value_heads.len()
    }
    pub fn mixed(&self) -> usize {
        self.channels.len()
    }
    pub fn value_dim(&self) -> usize {
        self.values() * S
    }
    pub fn recurrent_elements(&self) -> usize {
        self.values() * S * S
    }
    pub fn conv_elements(&self) -> usize {
        (CONV_K - 1) * self.mixed()
    }
}

/// Gather complete GGML blocks BEFORE device upload. `rows=true`: output
/// channels; `rows=false`: input columns of every output row. The Phase-4
/// replicated adapter validates source geometry/layout/length first.
fn select_matrix(
    map: &MappedGguf,
    name: &str,
    dims: [usize; 2],
    indices: &[usize],
    rows: bool,
) -> Result<(GgmlType, Vec<u8>), DeltaTpError> {
    let (ty, full) = gguf_shard(
        map,
        name,
        TensorSliceRequest {
            kind: ShardKind::Replicated,
            rank: 0,
            world_size: 1,
        },
    )
    .map_err(|e| DeltaTpError::Shape(e.to_string()))?;
    if full.dims != dims || indices.is_empty() || indices.windows(2).any(|w| w[0] >= w[1]) {
        return Err(DeltaTpError::Shape(format!(
            "{name}: dimensions or channel order invalid"
        )));
    }
    let (elems, bytes) = ty
        .block_layout()
        .ok_or_else(|| DeltaTpError::Shape(format!("{name}: unknown GGML layout")))?;
    let stride = dims[0] / elems * bytes;
    let source = full.bytes.as_ref();
    let mut selected = Vec::new();
    if rows {
        if indices.last().is_none_or(|&i| i >= dims[1]) {
            return Err(DeltaTpError::Shape(format!(
                "{name}: output row out of range"
            )));
        }
        selected.reserve(indices.len() * stride);
        for &i in indices {
            selected.extend_from_slice(&source[i * stride..(i + 1) * stride]);
        }
    } else {
        if !indices.len().is_multiple_of(elems)
            || indices.last().is_none_or(|&i| i >= dims[0])
            || indices
                .chunks_exact(elems)
                .any(|c| c[0] % elems != 0 || c.iter().enumerate().any(|(i, &x)| x != c[0] + i))
        {
            return Err(DeltaTpError::Shape(format!(
                "{name}: input columns cut or reorder GGML blocks"
            )));
        }
        selected.reserve(indices.len() / elems * bytes * dims[1]);
        for row in source.chunks_exact(stride) {
            for block in indices.chunks_exact(elems) {
                let start = block[0] / elems * bytes;
                selected.extend_from_slice(&row[start..start + bytes]);
            }
        }
    }
    Ok((ty, selected))
}

fn f32_select(
    e: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    size: usize,
    indices: &[usize],
) -> Result<CudaSlice<f32>, DeltaTpError> {
    let (info, bytes) = map.tensor_bytes(name).map_err(GpuError::from)?;
    let expected_dims = if name.ends_with("ssm_conv1d.weight") {
        vec![CONV_K as u64, ((2 * NK + NV) * S) as u64]
    } else {
        vec![size as u64]
    };
    if info.ggml_type != GgmlType::F32
        || info.dims != expected_dims
        || bytes.len() != size * 4
        || indices.iter().any(|&i| i >= size)
    {
        return Err(DeltaTpError::Shape(format!(
            "{name}: expected finite F32 vector of length {size}"
        )));
    }
    let data: Vec<f32> = indices
        .iter()
        .map(|&i| {
            let start = 4 * i;
            f32::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ])
        })
        .collect();
    if data.iter().any(|x| !x.is_finite()) {
        return Err(DeltaTpError::Shape(format!("{name}: nonfinite metadata")));
    }
    Ok(e.to_device(&data)?)
}

struct PrefillGemm {
    xq: CudaSlice<i8>,
    xs: CudaSlice<f32>,
    yq: CudaSlice<u8>,
    xsums: CudaSlice<f32>,
    ssums: CudaSlice<f32>,
    skfix: CudaSlice<f32>,
}

impl PrefillGemm {
    fn new(e: &GpuExecutor) -> Result<Self, DeltaTpError> {
        // All three rank-local projections have input width <= WIDTH. The
        // span cap is <=64, so prefill_mm_any uses strided int8 + dp4a/mma.
        Ok(Self {
            xq: e.alloc_i8(SPAN_CAP * WIDTH)?,
            xs: e.alloc(SPAN_CAP * WIDTH / 32)?,
            yq: e.alloc_u8(1)?,
            xsums: e.alloc(1)?,
            ssums: e.alloc(SPAN_CAP * WIDTH / 16)?,
            skfix: e.alloc(1)?,
        })
    }

    fn project(
        &mut self,
        e: &GpuExecutor,
        w: &QuantW,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), DeltaTpError> {
        prefill_mm_any(
            e,
            w,
            &mut self.xq,
            &mut self.xs,
            &mut self.yq,
            &mut self.xsums,
            &mut self.ssums,
            &mut self.skfix,
            x,
            y,
            rows,
        )?;
        Ok(())
    }
}

struct Span {
    input: CudaSlice<f32>,
    mixed: CudaSlice<f32>,
    ext: CudaSlice<f32>,
    conv_out: CudaSlice<f32>,
    convolved: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    gate_one: CudaSlice<f32>,
    beta_one: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    z: CudaSlice<f32>,
    core: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    reduced: CudaSlice<f32>,
}
impl Span {
    fn new(e: &GpuExecutor, g: &DeltaGeometry) -> Result<Self, DeltaTpError> {
        let (n, mixed, value, heads) = (SPAN_CAP, g.mixed(), g.value_dim(), g.values());
        Ok(Self {
            input: e.alloc(WIDTH)?,
            mixed: e.alloc(n * mixed)?,
            ext: e.alloc((n + CONV_K - 1) * mixed)?,
            conv_out: e.alloc((n + CONV_K - 1) * mixed)?,
            convolved: e.alloc(n * mixed)?,
            q: e.alloc(n * value)?,
            k: e.alloc(n * value)?,
            v: e.alloc(n * value)?,
            gate: e.alloc(n * heads)?,
            beta: e.alloc(n * heads)?,
            gate_one: e.alloc(heads)?,
            beta_one: e.alloc(heads)?,
            attn: e.alloc(n * value)?,
            z: e.alloc(n * value)?,
            core: e.alloc(n * value)?,
            partial: e.alloc(n * WIDTH)?,
            reduced: e.alloc(n * WIDTH)?,
        })
    }
}

/// Input rows are already pre-attention-normalized and MIRRORED. Output is
/// the mixer projection without residual or post-norm; caller alone owns those.
/// Persistent conv and recurrent payloads are rank-local, F32 on BOTH oracle
/// and TP paths. No full model state or worker execution protocol is installed.
pub struct DeltaTpRank {
    pub geometry: DeltaGeometry,
    weights: [QuantW; 3], // in_qkv, gate, out
    alpha: RepackedQ8,
    beta_w: RepackedQ8,
    conv_weight: CudaSlice<f32>,
    a: CudaSlice<f32>,
    dt: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    recurrent: CudaSlice<f32>,
    conv: CudaSlice<f32>,
    /// Slot 0 lives in recurrent/conv; other slots swap into that pair for a step.
    slot_states: Vec<(CudaSlice<f32>, CudaSlice<f32>)>,
    span: Span,
    prefill_gemm: Option<PrefillGemm>,
    eps: f32,
    rank: usize,
}
impl DeltaTpRank {
    pub fn load<C: Communicator>(
        e: &GpuExecutor,
        map: &MappedGguf,
        layer: usize,
        group: &C,
    ) -> Result<Self, DeltaTpError> {
        if group.world_size() != 2 {
            return Err(DeltaTpError::Shape("expected TP=2".into()));
        }
        let g = DeltaGeometry::new(group.rank())?;
        if GpuExecutor::dn_state_esz() != 4 {
            return Err(DeltaTpError::Shape(
                "F32 recurrent state required for oracle parity".into(),
            ));
        }
        for (field, expected) in [
            ("embedding_length", WIDTH),
            ("ssm.state_size", S),
            ("ssm.group_count", NK),
            ("ssm.time_step_rank", NV),
            ("ssm.conv_kernel", CONV_K),
        ] {
            if map.gguf().arch_field(field).and_then(Value::as_u64) != Some(expected as u64) {
                return Err(DeltaTpError::Shape(format!("unsupported {field}")));
            }
        }
        let eps = map
            .gguf()
            .arch_field("attention.layer_norm_rms_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-6);
        if !eps.is_finite() || eps <= 0.0 {
            return Err(DeltaTpError::Shape("invalid norm epsilon".into()));
        }
        let name = |part: &str| format!("blk.{layer}.{part}");
        let mut ab = Vec::new();
        for part in ["ssm_alpha.weight", "ssm_beta.weight"] {
            let (ty, bytes) = select_matrix(map, &name(part), [WIDTH, NV], &g.value_heads, true)?;
            if ty != GgmlType::Q8_0 {
                return Err(DeltaTpError::Shape(format!(
                    "{part}: only Q8_0 fused gate supported"
                )));
            }
            ab.push(e.repack_q8_blocks(&bytes, vec![WIDTH, g.values()])?);
        }
        let mut projections = Vec::new();
        for (part, dims, indices, rows) in [
            (
                "attn_qkv.weight",
                [WIDTH, (2 * NK + NV) * S],
                g.channels.clone(),
                true,
            ),
            (
                "attn_gate.weight",
                [WIDTH, NV * S],
                g.value_heads
                    .iter()
                    .flat_map(|&h| h * S..(h + 1) * S)
                    .collect(),
                true,
            ),
            (
                "ssm_out.weight",
                [NV * S, WIDTH],
                g.value_heads
                    .iter()
                    .flat_map(|&h| h * S..(h + 1) * S)
                    .collect(),
                false,
            ),
        ] {
            let (ty, bytes) = select_matrix(map, &name(part), dims, &indices, rows)?;
            if crate::gpu::kq_params(ty).is_none() {
                return Err(DeltaTpError::Shape(format!(
                    "{part}: unsupported quant type {ty:?}"
                )));
            }
            let dims = if rows {
                vec![dims[0], indices.len()]
            } else {
                vec![indices.len(), dims[1]]
            };
            projections.push(QuantW::Kq(e.repack_kquant_raw(
                &bytes,
                dims,
                ty,
                &name(part),
            )?));
        }
        let conv_indices: Vec<usize> = g
            .channels
            .iter()
            .flat_map(|&c| c * CONV_K..(c + 1) * CONV_K)
            .collect();
        let mut r = Self {
            span: Span::new(e, &g)?,
            prefill_gemm: None,
            recurrent: e.alloc(g.recurrent_elements())?,
            conv: e.alloc(g.conv_elements())?,
            slot_states: Vec::new(),
            conv_weight: f32_select(
                e,
                map,
                &name("ssm_conv1d.weight"),
                (2 * NK + NV) * S * CONV_K,
                &conv_indices,
            )?,
            a: f32_select(e, map, &name("ssm_a"), NV, &g.value_heads)?,
            dt: f32_select(e, map, &name("ssm_dt.bias"), NV, &g.value_heads)?,
            norm: f32_select(
                e,
                map,
                &name("ssm_norm.weight"),
                S,
                &(0..S).collect::<Vec<_>>(),
            )?,
            weights: projections
                .try_into()
                .map_err(|_| DeltaTpError::Shape("projection count".into()))?,
            alpha: ab.remove(0),
            beta_w: ab.remove(0),
            geometry: g,
            eps,
            rank: group.rank(),
        };
        r.reset(e)?;
        Ok(r)
    }
    pub fn reset(&mut self, e: &GpuExecutor) -> Result<(), DeltaTpError> {
        if self.recurrent.context().cu_ctx() != e.stream.context().cu_ctx() {
            return Err(DeltaTpError::Shape("foreign executor".into()));
        }
        e.stream
            .memset_zeros(&mut self.recurrent)
            .map_err(GpuError::from)?;
        e.stream
            .memset_zeros(&mut self.conv)
            .map_err(GpuError::from)?;
        Ok(())
    }
    pub fn enable_slots(&mut self, e: &GpuExecutor, slots: usize) -> Result<(), DeltaTpError> {
        if slots == 0 || !self.slot_states.is_empty() {
            return Err(DeltaTpError::Shape(
                "invalid or repeated slot allocation".into(),
            ));
        }
        for _ in 1..slots {
            let mut recurrent = e.alloc(self.geometry.recurrent_elements())?;
            let mut conv = e.alloc(self.geometry.conv_elements())?;
            e.stream
                .memset_zeros(&mut recurrent)
                .map_err(GpuError::from)?;
            e.stream.memset_zeros(&mut conv).map_err(GpuError::from)?;
            self.slot_states.push((recurrent, conv));
        }
        Ok(())
    }
    pub fn reset_slot(&mut self, e: &GpuExecutor, slot: usize) -> Result<(), DeltaTpError> {
        if slot == 0 {
            self.reset(e)
        } else {
            let (recurrent, conv) = self
                .slot_states
                .get_mut(slot - 1)
                .ok_or_else(|| DeltaTpError::Shape("slot out of range".into()))?;
            e.stream.memset_zeros(recurrent).map_err(GpuError::from)?;
            e.stream.memset_zeros(conv).map_err(GpuError::from)?;
            Ok(())
        }
    }
    /// Swap a slot's persistent state pair into the home buffers so the
    /// collective-free run (and any capture of it) addresses this slot's
    /// state. Slot 0 lives in place and is a no-op. Mirrors `decode_slot`'s
    /// eager swap exactly; `swap_state_out` is its inverse.
    pub(crate) fn swap_state_in(&mut self, slot: usize) -> Result<(), DeltaTpError> {
        if slot == 0 {
            return Ok(());
        }
        let state = self
            .slot_states
            .get_mut(slot - 1)
            .ok_or_else(|| DeltaTpError::Shape("slot out of range".into()))?;
        std::mem::swap(&mut self.recurrent, &mut state.0);
        std::mem::swap(&mut self.conv, &mut state.1);
        Ok(())
    }
    /// Swap the (advanced) state pair back to its slot's storage. The swap
    /// is its own inverse given the same slot.
    pub(crate) fn swap_state_out(&mut self, slot: usize) -> Result<(), DeltaTpError> {
        self.swap_state_in(slot)
    }
    pub fn decode_slot<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
        slot: usize,
    ) -> Result<&'a CudaSlice<f32>, DeltaTpError> {
        if slot == 0 {
            return self.decode(e, group, input);
        }
        let state = self
            .slot_states
            .get_mut(slot - 1)
            .ok_or_else(|| DeltaTpError::Shape("slot out of range".into()))?;
        std::mem::swap(&mut self.recurrent, &mut state.0);
        std::mem::swap(&mut self.conv, &mut state.1);
        let result = self.forward(e, group, input, 1, slot, false, 0).map(|_| ());
        let state = &mut self.slot_states[slot - 1];
        std::mem::swap(&mut self.recurrent, &mut state.0);
        std::mem::swap(&mut self.conv, &mut state.1);
        result?;
        Ok(&self.span.reduced)
    }
    pub fn state(&self) -> (&CudaSlice<f32>, &CudaSlice<f32>) {
        (&self.recurrent, &self.conv)
    }

    /// A slot's persistent state buffers: slot 0 works in place on the home
    /// pair, other slots on their `slot_states` entry.
    pub fn slot_state(&self, slot: usize) -> Option<(&CudaSlice<f32>, &CudaSlice<f32>)> {        if slot == 0 {
            Some((&self.recurrent, &self.conv))
        } else {
            self.slot_states
                .get(slot - 1)
                .map(|(recurrent, conv)| (recurrent, conv))
        }
    }

    pub fn slot_state_mut(
        &mut self,
        slot: usize,
    ) -> Option<(&mut CudaSlice<f32>, &mut CudaSlice<f32>)> {
        if slot == 0 {
            Some((&mut self.recurrent, &mut self.conv))
        } else {
            self.slot_states
                .get_mut(slot - 1)
                .map(|(recurrent, conv)| (recurrent, conv))
        }
    }
    pub fn local_state_bytes(&self) -> usize {
        (self.recurrent.len() + self.conv.len()) * 4
    }
    pub fn decode<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
    ) -> Result<&'a CudaSlice<f32>, DeltaTpError> {
        self.forward(e, group, input, 1, 0, false, 0)
    }
    pub fn prefill<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<&'a CudaSlice<f32>, DeltaTpError> {
        self.forward(e, group, input, rows, 0, true, 0)
    }

    /// Slot-addressed span prefill (prototype): identical math to `prefill`
    /// with the addressed slot's recurrent/conv pair swapped in for the span
    /// and swapped back after it. `layer` feeds the probe-only
    /// `PADDOCK_TP_ABC_TRACE` substage readbacks.
    pub(crate) fn prefill_slot<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
        slot: usize,
        rows: usize,
        layer: usize,
    ) -> Result<&'a CudaSlice<f32>, DeltaTpError> {
        if slot == 0 {
            return self.forward(e, group, input, rows, slot, true, layer);
        }
        let state = self
            .slot_states
            .get_mut(slot - 1)
            .ok_or_else(|| DeltaTpError::Shape("slot out of range".into()))?;
        std::mem::swap(&mut self.recurrent, &mut state.0);
        std::mem::swap(&mut self.conv, &mut state.1);
        let result = self
            .forward(e, group, input, rows, slot, true, layer)
            .map(|_| ());
        let state = &mut self.slot_states[slot - 1];
        std::mem::swap(&mut self.recurrent, &mut state.0);
        std::mem::swap(&mut self.conv, &mut state.1);
        result?;
        Ok(&self.span.reduced)
    }
    fn forward<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
        input: &CudaSlice<f32>,
        rows: usize,
        slot: usize,
        prefill: bool,
        layer: usize,
    ) -> Result<&'a CudaSlice<f32>, DeltaTpError> {
        let world = group.world_size();
        let rank = group.rank();
        let input_len = input.len();
        let context_match = input.context().cu_ctx() == e.stream.context().cu_ctx();
        if world != 2 || rank != self.rank || rows == 0 || rows > SPAN_CAP
            || input_len < rows * WIDTH || !context_match
        {
            return Err(DeltaTpError::Shape(format!(
                "rank/input/span mismatch: expected world=2 rank={} input_len=rows*{} span_len=1..{} state_pair={}; actual world={} rank={} rows={} input_len={} span_len={} context_match={}",
                self.rank, WIDTH, SPAN_CAP, if slot == 0 { "home" } else { "slot" },
                world, rank, rows, input_len, rows, context_match
            )));
        }
        if prefill {
            if self.prefill_gemm.is_none() {
                self.prefill_gemm = Some(PrefillGemm::new(e)?);
            }
            self.prefill_run(e, input, rows, layer)?;
            self.finish_partial_prefill(e, rows)?;
        } else {
            self.decode_run(e, input, rows)?;
            self.finish_partial(e, rows)?;
        }
        group.after_compute(&e.stream)?;
        group.all_reduce(&self.span.partial, &mut self.span.reduced)?;
        group.before_compute(&e.stream)?;
        Ok(&self.span.reduced)
    }

    /// Stage the out-projection into the capacity-sized `partial`: zero the
    /// whole buffer first (the unused suffix rides every all-reduce), then
    /// the row-parallel down GEMV per row. `forward` runs this before its
    /// collective; the graphed path bakes it into the capture.
    pub(crate) fn finish_partial(
        &mut self,
        e: &GpuExecutor,
        rows: usize,
    ) -> Result<(), DeltaTpError> {
        let g = &self.geometry;
        let s = &mut self.span;
        // The capacity-sized NCCL buffer includes an unused suffix for short
        // spans. Initialize that suffix before reducing the entire buffer.
        e.stream
            .memset_zeros(&mut s.partial)
            .map_err(GpuError::from)?;
        if rows == 1 {
            gemv_any(e, &self.weights[2], &s.core, &mut s.partial)?;
        } else {
            for t in 0..rows {
                e.copy_region(&s.core, t * g.value_dim(), &mut s.input, 0, g.value_dim())?;
                gemv_any(e, &self.weights[2], &s.input, &mut s.convolved)?;
                e.copy_region(&s.convolved, 0, &mut s.partial, t * WIDTH, WIDTH)?;
            }
        }
        Ok(())
    }

    fn finish_partial_prefill(&mut self, e: &GpuExecutor, rows: usize) -> Result<(), DeltaTpError> {
        // NCCL reduces the entire capacity buffer, including the unused rows.
        e.stream
            .memset_zeros(&mut self.span.partial)
            .map_err(GpuError::from)?;
        self.prefill_gemm
            .as_mut()
            .expect("prefill scratch allocated")
            .project(e, &self.weights[2], &self.span.core, &mut self.span.partial, rows)?;
        Ok(())
    }

    /// The post-run NCCL fences + all-reduce over the staged `partial`;
    /// returns the reduced output. `forward` calls this after
    /// `finish_partial`; the graphed path runs the fences + all-reduce
    /// inline (its capture already staged `partial`).
    pub(crate) fn finish<'a, C: Communicator>(
        &'a mut self,
        e: &GpuExecutor,
        group: &C,
    ) -> Result<&'a CudaSlice<f32>, DeltaTpError> {
        let s = &mut self.span;
        group.after_compute(&e.stream)?;
        group.all_reduce(&s.partial, &mut s.reduced)?;
        group.before_compute(&e.stream)?;
        Ok(&s.reduced)
    }

    /// The pre-collective DeltaNet run through `core` (mixed projection,
    /// causal conv, QKV split + norms, alpha/beta gate, gated recurrent
    /// state, gated norm). `forward` stages `partial` from `core` and runs
    /// the NCCL reduce after it; Phase 11 graph capture records exactly this
    /// run - the recurrent/conv buffers it bakes are whatever the caller
    /// swapped in, so per-slot captures key on the swapped state pair.
    pub(crate) fn decode_run(
        &mut self,
        e: &GpuExecutor,
        input: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), DeltaTpError> {
        self.run(e, input, rows, false, 0)
    }

    fn prefill_run(
        &mut self,
        e: &GpuExecutor,
        input: &CudaSlice<f32>,
        rows: usize,
        layer: usize,
    ) -> Result<(), DeltaTpError> {
        self.run(e, input, rows, true, layer)
    }

    fn run(
        &mut self,
        e: &GpuExecutor,
        input: &CudaSlice<f32>,
        rows: usize,
        prefill: bool,
        layer: usize,
    ) -> Result<(), DeltaTpError> {
        let g = &self.geometry;
        let s = &mut self.span;
        if prefill {
            let mm = self.prefill_gemm.as_mut().expect("prefill scratch allocated");
            mm.project(e, &self.weights[0], input, &mut s.mixed, rows)?;
            mm.project(e, &self.weights[1], input, &mut s.z, rows)?;
            // [PADDOCK_TP_ABC_TRACE] substage readbacks (row 0): the
            // rank-local mixed shard and gate (z) shard, plus the whole-row
            // mixer input, echoed under the c.* name for the host compare.
            super::tp_trace::trace_row(e, "b.dn-in", layer, input, 0, WIDTH)?;
            super::tp_trace::trace_row(e, "b.dn-mixed", layer, &s.mixed, 0, g.mixed())?;
            super::tp_trace::trace_row(e, "b.dn-z", layer, &s.z, 0, g.value_dim())?;
        }
        // One row keeps the existing conv_step arithmetic in both modes.
        // Multi-row spans keep the existing causal-conv/state path.
        if rows == 1 {
            if !prefill {
                gemv_any(e, &self.weights[0], input, &mut s.mixed)?;
            }
            e.conv_step(
                &mut self.conv,
                &s.mixed,
                &self.conv_weight,
                &mut s.convolved,
                g.mixed(),
                CONV_K,
            )?;
            e.deltanet_split_gqa_norm(
                &s.convolved,
                &mut s.q,
                &mut s.k,
                &mut s.v,
                1,
                g.keys(),
                g.values(),
                S,
            )?;
            e.deltanet_alpha_beta_gate(
                &self.alpha,
                &self.beta_w,
                input,
                &self.a,
                &self.dt,
                &mut s.gate,
                &mut s.beta,
                g.values(),
            )?;
            if !prefill {
                gemv_any(e, &self.weights[1], input, &mut s.z)?;
            }
        } else {
            for t in 0..rows {
                e.copy_region(input, t * WIDTH, &mut s.input, 0, WIDTH)?;
                if !prefill {
                    gemv_any(e, &self.weights[0], &s.input, &mut s.convolved)?;
                    e.copy_region(&s.convolved, 0, &mut s.mixed, t * g.mixed(), g.mixed())?;
                    gemv_any(e, &self.weights[1], &s.input, &mut s.core)?;
                    e.copy_region(&s.core, 0, &mut s.z, t * g.value_dim(), g.value_dim())?;
                }
                // [PADDOCK_TP_ABC_TRACE] the remaining PER-ROW fused
                // alpha/beta gate work (deltanet_alpha_beta_gate): row 0 of
                // the span only, so a probe sees exactly this site's values.
                if t == 0 {
                    super::tp_trace::trace_row(e, "b.dn-in", layer, &s.input, 0, WIDTH)?;
                }
                e.deltanet_alpha_beta_gate(
                    &self.alpha,
                    &self.beta_w,
                    &s.input,
                    &self.a,
                    &self.dt,
                    &mut s.gate_one,
                    &mut s.beta_one,
                    g.values(),
                )?;
                if t == 0 {
                    super::tp_trace::trace_row(
                        e,
                        "b.dn-gate",
                        layer,
                        &s.gate_one,
                        0,
                        g.values(),
                    )?;
                    super::tp_trace::trace_row(
                        e,
                        "b.dn-beta",
                        layer,
                        &s.beta_one,
                        0,
                        g.values(),
                    )?;
                }
                e.copy_region(&s.gate_one, 0, &mut s.gate, t * g.values(), g.values())?;
                e.copy_region(&s.beta_one, 0, &mut s.beta, t * g.values(), g.values())?;
            }
            e.copy_region(&self.conv, 0, &mut s.ext, 0, g.conv_elements())?;
            e.copy_region(&s.mixed, 0, &mut s.ext, g.conv_elements(), rows * g.mixed())?;
            e.causal_conv1d_silu(
                &s.ext,
                &self.conv_weight,
                &mut s.conv_out,
                rows + CONV_K - 1,
                g.mixed(),
                CONV_K,
            )?;
            e.copy_region(
                &s.conv_out,
                g.conv_elements(),
                &mut s.convolved,
                0,
                rows * g.mixed(),
            )?;
            e.copy_region(
                &s.ext,
                rows * g.mixed(),
                &mut self.conv,
                0,
                g.conv_elements(),
            )?;
            e.deltanet_split_gqa_norm(
                &s.convolved,
                &mut s.q,
                &mut s.k,
                &mut s.v,
                rows,
                g.keys(),
                g.values(),
                S,
            )?;
            if prefill {
                // [PADDOCK_TP_ABC_TRACE] conv output, split+L2-norm q/k/v
                // (row 0) on the batched span path.
                super::tp_trace::trace_row(e, "b.dn-conv", layer, &s.convolved, 0, g.mixed())?;
                super::tp_trace::trace_row(e, "b.dn-q", layer, &s.q, 0, g.value_dim())?;
                super::tp_trace::trace_row(e, "b.dn-k", layer, &s.k, 0, g.value_dim())?;
                super::tp_trace::trace_row(e, "b.dn-v", layer, &s.v, 0, g.value_dim())?;
            }
        }
        e.gated_delta_recurrent_v2(
            &s.q,
            &s.k,
            &s.v,
            &s.gate,
            &s.beta,
            None,
            &mut self.recurrent,
            0,
            None,
            &mut s.attn,
            1,
            rows,
            g.values(),
            S,
        )?;
        if prefill {
            // [PADDOCK_TP_ABC_TRACE] recurrence output (row 0) and the gated
            // norm's `core` (row 0) - the out projection's input.
            super::tp_trace::trace_row(e, "b.dn-rec", layer, &s.attn, 0, g.value_dim())?;
        }
        e.gated_rmsnorm(
            &s.attn,
            &s.z,
            &self.norm,
            &mut s.core,
            rows * g.values(),
            S,
            self.eps,
        )?;
        if prefill {
            super::tp_trace::trace_row(e, "b.dn-core", layer, &s.core, 0, g.value_dim())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn head_groups_and_state_are_rank_local() {
        let a = DeltaGeometry::new(0).unwrap();
        let b = DeltaGeometry::new(1).unwrap();
        assert_eq!(
            (
                a.keys(),
                a.values(),
                a.mixed(),
                a.recurrent_elements(),
                a.conv_elements()
            ),
            (8, 24, 5120, 24 * 128 * 128, 3 * 5120)
        );
        assert!(DeltaGeometry::new(2).is_err());
        let mut heads = a.value_heads.clone();
        heads.extend(b.value_heads.iter().copied());
        heads.sort_unstable();
        assert_eq!(heads, (0..48).collect::<Vec<_>>());
        for g in [&a, &b] {
            assert!(
                g.value_heads
                    .iter()
                    .all(|h| g.key_heads.contains(&(h % NK)))
            );
            assert_eq!((g.recurrent_elements() + g.conv_elements()) * 4, 1_634_304);
        }
        let mut channels = a.channels.clone();
        channels.extend(b.channels.iter().copied());
        channels.sort_unstable();
        assert_eq!(channels, (0..10240).collect::<Vec<_>>());
    }

    #[test]
    fn recurrent_state_slots_are_disjoint_and_reset_sized() {
        // Rank-local slot state: each rank owns exactly half the value-head
        // band set; the pairs never overlap and together cover all 48 heads
        // (the cross-rank completeness side is pinned above). The per-slot
        // state footprint is the Phase 12 accounting input for DeltaNet
        // layers - local_state_bytes must equal both slots' live state, so
        // context_mem_bytes stays exact regardless of slot count.
        let a = DeltaGeometry::new(0).unwrap();
        let b = DeltaGeometry::new(1).unwrap();
        assert!(!a.value_heads.iter().any(|h| b.value_heads.contains(h)));
        // Conv window is (conv_k - 1) * mixed width per slot; recurrent is
        // value_heads * S * S. Both derive from the same geometry, so the
        // two slots' bytes double the single-slot figure exactly.
        assert_eq!(a.conv_elements(), b.conv_elements());
        assert_eq!(a.recurrent_elements(), b.recurrent_elements());
    }
}
