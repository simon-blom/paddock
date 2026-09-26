//! Bounded batched-prefill span scratch for the Qwen3.8 TP rank.
//!
//! One lazily-allocated plane set per rank serving the production span
//! advance/head path and the `forward_prefill_span` probe wrapper:
//! whole-model activation planes, the row-batched GQA/FFN mixer planes,
//! and one shared quantized-GEMM staging set. Capacity is fixed at
//! [`TP_SPAN_CAP`] rows so memory use is bounded and explicit (~16 MB at
//! the Qwen3.8 geometry). Nothing here exists until the
//! first span forward, so the one-token decode paths never pay for it.
//!
//! The staging set rides the rows<=64 strided quantize band exclusively
//! (`quantize_q8` + the mma/dp4a GEMM rungs); the flat-mmq producers (`yq`,
//! `xsums`) and the stream-K fold scratch (`skfix`) only engage above 64 rows,
//! so those planes are one-element stand-ins while the cap holds.
use cudarc::driver::CudaSlice;

use super::gqa_tp::GqaGeometry;
use crate::gpu::{GpuError, GpuExecutor};

/// Row cap of the batched prefill span. It must not exceed the
/// DeltaNet primitive's own cap: the whole-model traversal reuses
/// `DeltaTpRank::prefill(rows)` and a wider cap would fail mid-span.
/// Guarded by a compile-time assertion.
pub(super) const TP_SPAN_CAP: usize = 64;
// The whole-model traversal passes this count directly to DeltaNet prefill.
const _: () = assert!(TP_SPAN_CAP <= super::delta_tp::SPAN_CAP);

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

/// Row-batched FFN planes for one span: gate/up activations and the
/// capacity-sized all-reduce pair.
pub(crate) struct SpanFfn {
    pub(crate) cap: usize,
    pub(crate) gate: CudaSlice<f32>,
    pub(crate) up: CudaSlice<f32>,
    pub(crate) partial: CudaSlice<f32>,
    pub(crate) reduced: CudaSlice<f32>,
}

/// Quantized-GEMM staging shared by every projection of one span pass (GQA
/// Q/K/V/output and FFN gate/up/down): the strided int8 activation pair plus
/// the per-16 k-quant sums plane.
pub(crate) struct SpanGemmStaging {
    pub(crate) xq: CudaSlice<i8>,
    pub(crate) xs: CudaSlice<f32>,
    pub(crate) yq: CudaSlice<u8>,
    pub(crate) xsums: CudaSlice<f32>,
    pub(crate) ssums: CudaSlice<f32>,
    pub(crate) skfix: CudaSlice<f32>,
}

/// The complete span plane set owned by each `Qwen35TpRank`, including
/// the dedicated prefill-lane rank.
pub(super) struct TpSpanPlanes {
    pub(super) act: SpanAct,
    pub(super) gqa: SpanGqa,
    pub(super) ffn: SpanFfn,
    pub(crate) q: SpanGemmStaging,
}

impl TpSpanPlanes {
    /// Allocate every span plane at the fixed cap. `hidden` is the backbone
    /// width, `g` the (uniform) GQA geometry, `local_ff` the FFN shard width,
    /// and `table_len` the full mirrored block-table length (bps * slots).
    pub(super) fn new(
        e: &GpuExecutor,
        hidden: usize,
        g: &GqaGeometry,
        local_ff: usize,
        table_len: usize,
    ) -> Result<Self, GpuError> {
        let cap = TP_SPAN_CAP;
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
            ffn: SpanFfn {
                cap,
                gate: e.alloc(cap * local_ff)?,
                up: e.alloc(cap * local_ff)?,
                partial: e.alloc(cap * hidden)?,
                reduced: e.alloc(cap * hidden)?,
            },
            q: SpanGemmStaging {
                xq: e.alloc_i8(cap * max_in)?,
                xs: e.alloc(cap * max_in / 32)?,
                // Only the > 64-row rungs touch these three; the cap is pinned
                // to the DeltaNet primitive's 64 (see TP_SPAN_CAP).
                yq: e.alloc_u8(1)?,
                xsums: e.alloc(1)?,
                ssums: e.alloc(cap * max_in / 16)?,
                skfix: e.alloc(1)?,
            },
        })
    }
}
