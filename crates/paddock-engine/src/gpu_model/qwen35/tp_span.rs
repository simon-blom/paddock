//! Bounded batched-prefill span scratch for the Qwen3.8 TP rank.
//!
//! One lazily-allocated plane set per rank serving the production span
//! advance/head path and the `forward_prefill_span` probe wrapper:
//! whole-model activation planes, the row-batched GQA/FFN mixer planes,
//! and one shared quantized-GEMM staging set. Capacity is fixed at
//! [`tp_span_cap::span_cap()`] rows so memory use is bounded and explicit
//! (~16 MB at cap 64, ~25 MB at cap 192 at the Qwen3.8 geometry). Nothing
//! here exists until the first span forward, so the one-token decode paths
//! never pay for it.
//!
//! The staging set carries BOTH quantize layouts: the rows<=64 strided
//! band (`quantize_q8` + the dp4a/mma GEMM rungs) and the rows>64 flat
//! mmq band (`yq`/`xsums` + the W4A8-tile/mmq rungs) plus the Q8_0
//! stream-K fold scratch (`skfix`). Every plane is sized at allocation
//! from the resolved cap, and every consumer asserts `rows` against it —
//! there are no one-element placeholder planes left to read stale bytes.
use cudarc::driver::CudaSlice;

use super::gqa_tp::GqaGeometry;
use super::tp_span_cap::span_cap;
use crate::gpu::{GpuError, GpuExecutor};
use crate::gpu_model::tp::prefill::ProjectionStaging;
pub(crate) use crate::gpu_model::tp::ffn::SwiGluPrefillScratch as SpanFfn;
pub(crate) use crate::gpu_model::tp::prefill::ProjectionStaging as SpanGemmStaging;

/// Whole-model activation planes for one span: residual stream, normalized
/// rows, and the span's token ids.
pub(crate) struct SpanAct {
    pub(crate) x: CudaSlice<f32>,
    pub(crate) xn: CudaSlice<f32>,
    pub(crate) tokens: CudaSlice<u32>,
    /// One-row staging for the final-row head GEMV: `gemv_any` takes a
    /// `&CudaSlice` (not an offset view), so the last normalized row is
    /// copied here (one 20 KB D2D per span) before the head projection.
    pub(crate) x_last: CudaSlice<f32>,
}

/// Row-batched GQA mixer planes for one span: projections, per-head norms,
/// attention output, and the capacity-sized all-reduce pair. The position,
/// slot, axis and block-table planes are staged per span (one upload covers
/// the whole row batch).
pub(crate) struct SpanGqa {
    pub(crate) cap: usize,
    /// The (slot, position, rows) whose metadata is currently resident in
    /// the device planes below. None until the first stage of a span; a
    /// change re-stages. GQA-layer-local buffers (projections, norms,
    /// attention) are NOT covered by this key - only the span-global
    /// metadata planes shared by every GQA layer of one traversal.
    pub(crate) staged: Option<(usize, usize, usize)>,
    pub(crate) qg: CudaSlice<f32>,
    pub(crate) q: CudaSlice<f32>,
    pub(crate) gate: CudaSlice<f32>,
    pub(crate) k: CudaSlice<f32>,
    pub(crate) v: CudaSlice<f32>,
    pub(crate) qn: CudaSlice<f32>,
    pub(crate) kn: CudaSlice<f32>,
    pub(crate) attn: CudaSlice<f32>,
    pub(crate) partial: CudaSlice<f32>,
    pub(crate) reduced: CudaSlice<f32>,
    pub(crate) positions: CudaSlice<u32>,
    pub(crate) slots: CudaSlice<u32>,
    pub(crate) axes: CudaSlice<u32>,
    pub(crate) block_table: CudaSlice<u32>,
}

/// The complete span plane set owned by each `Qwen35TpRank`, including
/// the dedicated prefill-lane rank.
pub(super) struct TpSpanPlanes {
    pub(super) act: SpanAct,
    pub(super) gqa: SpanGqa,
    pub(super) ffn: SpanFfn,
    pub(crate) q: ProjectionStaging,
}

impl TpSpanPlanes {
    /// Allocate every span plane at the resolved cap. `hidden` is the
    /// backbone width, `g` the (uniform) GQA geometry, `local_ff` the FFN
    /// shard width, and `table_len` the full mirrored block-table length
    /// (bps * slots). `rows` beyond `cap` are refused by every consumer
    /// (model, mixers, chunkers); nothing truncates.
    pub(super) fn new(
        e: &GpuExecutor,
        hidden: usize,
        g: &GqaGeometry,
        local_ff: usize,
        table_len: usize,
    ) -> Result<Self, GpuError> {
        let cap = span_cap();
        let q_dim = g.q_dim();
        let kv_dim = g.kv_dim();
        let width = g.width;
        // The strided staging covers the widest projected input across the
        // GQA and FFN planes: hidden for Q/K/V and gate/up, the local Q width
        // for the attention output projection, the local FF width for down.
        let max_in = width.max(q_dim).max(local_ff);
        Ok(Self {
            act: SpanAct {
                x: e.alloc(cap * hidden)?,
                xn: e.alloc(cap * hidden)?,
                tokens: e.alloc_u32(cap)?,
                x_last: e.alloc(hidden)?,
            },
            gqa: SpanGqa {
                cap,
                staged: None,
                qg: e.alloc(cap * 2 * q_dim)?,
                q: e.alloc(cap * q_dim)?,
                gate: e.alloc(cap * q_dim)?,
                k: e.alloc(cap * kv_dim)?,
                v: e.alloc(cap * kv_dim)?,
                qn: e.alloc(cap * q_dim)?,
                kn: e.alloc(cap * kv_dim)?,
                attn: e.alloc(cap * q_dim)?,
                partial: e.alloc(cap * width)?,
                reduced: e.alloc(cap * width)?,
                positions: e.alloc_u32(cap)?,
                slots: e.alloc_u32(cap)?,
                axes: e.alloc_u32(4 * cap)?,
                block_table: e.alloc_u32(table_len)?,
            },
            ffn: SpanFfn::new(e, cap, local_ff, hidden)?,
            q: ProjectionStaging::new(e, max_in, cap)?,
        })
    }
}
