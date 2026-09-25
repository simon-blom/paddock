//! Text-encoder ops - ModernBERT (the encoder under the Laya decision model)
//! and the decision head around it. Kernel side: `packs/cuda/src/encoder.cuh`
//! and `attn/varlen.cuh`, slots 671-678.
//!
//! The unit of work is a PACKED batch of variable-length sequences: every
//! plane is `[rows][channels]` with sequence `s` owning rows
//! `cu[s]..cu[s + 1]`, no padding between them. The GEMMs are the ordinary
//! f16-landing tensor-core ones (slot 618 and its fused epilogues 624 / 671 /
//! 672); what lives here is the attention that knows where one sequence ends
//! and the next begins, and the seams no other family needed.
//!
//! Buffers are checked against the geometry before any launch - a short slice
//! is a logic error here, not something to hand the driver.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// Query rows one attention block owns - must match `PD_VL_QT`.
pub const ENC_ATTN_QTILE: usize = 64;
/// How a tile descriptor packs its sequence - must match `PD_VL_TILE_SHIFT`.
pub const ENC_ATTN_TILE_SHIFT: u32 = 12;

impl GpuExecutor {
    /// True when the loaded pack carries the whole text-encoder lane. The
    /// slots landed together, so one missing means a pack older than it.
    pub fn has_text_encoder(&self) -> bool {
        let k = &self.kernels;
        k.f16_gemm_h.is_some()
            && k.f16_gemm_h_gelu.is_some()
            && k.f16_gemm_h_relu.is_some()
            && k.f16_gemm_h_geglu.is_some()
            && k.enc_attn_h.is_some()
            && k.enc_embed_ln.is_some()
            && k.laya_head_entry.is_some()
            && k.gather_rows.is_some()
            && k.laya_rowdot.is_some()
            && k.laya_act_head.is_some()
            && k.dp_res_ls_ln_h.is_some()
    }

    /// `y16 = relu(W x + bias)` at f16 - [`Self::matvec_batch_f16_h`] with the
    /// bias (optional) and ReLU in the epilogue, before the one round.
    pub fn matvec_batch_f16_h_relu(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_relu
            .ok_or(GpuError::MissingOp("f16_gemm_h_relu"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if !in_dim.is_multiple_of(8) {
            return Err(oob("f16_gemm_h_relu: in_dim must be a multiple of 8"));
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim
            || bias.is_some_and(|b| b.len() < out_dim)
        {
            return Err(oob("f16_gemm_h_relu: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-relu", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 671); bounds checked above, null bias = none
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// ModernBERT's MLP up projection with the GLU folded into the landing:
    /// `y16[b][f] = gelu(input_f) * gate_f`, `w` being `[in, 2F]` re-laid in
    /// 16-row blocks (see `gpu_model::laya::load`). `y16` is `[batch, F]`.
    pub fn matvec_batch_f16_h_geglu(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_geglu
            .ok_or(GpuError::MissingOp("f16_gemm_h_geglu"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if !in_dim.is_multiple_of(8) || !out_dim.is_multiple_of(16) {
            return Err(oob(
                "f16_gemm_h_geglu: in_dim a multiple of 8, out_dim of 16",
            ));
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim / 2
        {
            return Err(oob("f16_gemm_h_geglu: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-geglu", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (yp, _g3) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 672); bounds checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Bidirectional attention over packed sequences, straight off the fused
    /// `[rows][3][heads][hd]` projection landing into `out` `[rows][heads][hd]`.
    /// `cu` holds `n_seq + 1` row offsets, `tiles` one descriptor per
    /// [`ENC_ATTN_QTILE`] query rows of each sequence (`seq << 12 | tile`).
    /// `rope` = (cos, sin) tables `[max_pos][hd/2]`, positions relative to the
    /// sequence start; `bias` = the q|k|v in-projection bias; `window` 0 is
    /// full attention, else `|i - j| <= window`.
    #[allow(clippy::too_many_arguments)]
    pub fn enc_attn_h(
        &self,
        qkv: &CudaSlice<f16>,
        cu: &CudaSlice<u32>,
        tiles: &CudaSlice<u32>,
        n_tiles: usize,
        rope: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        bias: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n_heads: usize,
        head_dim: usize,
        window: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .enc_attn_h
            .ok_or(GpuError::MissingOp("enc_attn_h"))?;
        let d = n_heads * head_dim;
        if qkv.len() < rows * 3 * d
            || out.len() < rows * d
            || tiles.len() < n_tiles
            || bias.is_some_and(|b| b.len() < 3 * d)
        {
            return Err(oob("enc_attn_h: buffers under the attention geometry"));
        }
        let (qp, _g1) = qkv.device_ptr(&self.stream);
        let (cp, _g2) = cu.device_ptr(&self.stream);
        let (tp, _g3) = tiles.device_ptr(&self.stream);
        let rp = rope.map(|(c, s)| (c.device_ptr(&self.stream), s.device_ptr(&self.stream)));
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (cos_p, sin_p) = rp.as_ref().map_or((0, 0), |((c, _), (s, _))| (*c, *s));
        // SAFETY: ABI contract (slot 673); bounds checked above; the caller
        // builds cu/tiles for these rows and keeps rope positions in-table
        check(unsafe {
            f(
                qp as *const _,
                cp as *const _,
                tp as *const _,
                n_tiles as u32,
                cos_p as *const _,
                sin_p as *const _,
                bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                op as *mut _,
                n_heads as u32,
                head_dim as u32,
                window as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Token gather + the embedding LayerNorm: `x = LN(E[ids])` (f32, the
    /// residual) and `x16` its f16 twin. `emb` is `[vocab, d]` f16.
    #[allow(clippy::too_many_arguments)]
    pub fn enc_embed_ln(
        &self,
        emb: &CudaSlice<f16>,
        ids: &CudaSlice<u32>,
        w: &CudaSlice<f32>,
        b: Option<&CudaSlice<f32>>,
        x: &mut CudaSlice<f32>,
        x16: &mut CudaSlice<f16>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .enc_embed_ln
            .ok_or(GpuError::MissingOp("enc_embed_ln"))?;
        if ids.len() < rows
            || w.len() < d
            || b.is_some_and(|b| b.len() < d)
            || x.len() < rows * d
            || x16.len() < rows * d
        {
            return Err(oob("enc_embed_ln: buffers under the row geometry"));
        }
        let (ep, _g1) = emb.device_ptr(&self.stream);
        let (ip, _g2) = ids.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let bp = b.map(|b| b.device_ptr(&self.stream));
        let (xp, _g4) = x.device_ptr_mut(&self.stream);
        let (hp, _g5) = x16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 674); bounds checked above - every id is
        // < vocab by the tokenizer's construction
        check(unsafe {
            f(
                ep as *const _,
                ip as *const _,
                wp as *const _,
                bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                xp as *mut _,
                hp as *mut _,
                rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// The decision head's entry, in place on the residual: `x = LN(x; fw) +
    /// temb[rtype[row]]`, then `n16 = LN(x; w1, b1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn laya_head_entry(
        &self,
        x: &mut CudaSlice<f32>,
        fw: &CudaSlice<f32>,
        temb: &CudaSlice<f32>,
        rtype: &CudaSlice<u32>,
        w1: &CudaSlice<f32>,
        b1: &CudaSlice<f32>,
        n16: &mut CudaSlice<f16>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .laya_head_entry
            .ok_or(GpuError::MissingOp("laya_head_entry"))?;
        if x.len() < rows * d
            || n16.len() < rows * d
            || rtype.len() < rows
            || fw.len() < d
            || w1.len() < d
            || b1.len() < d
            || temb.len() < 3 * d
        {
            return Err(oob("laya_head_entry: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (fp, _g2) = fw.device_ptr(&self.stream);
        let (tp, _g3) = temb.device_ptr(&self.stream);
        let (rp, _g4) = rtype.device_ptr(&self.stream);
        let (wp, _g5) = w1.device_ptr(&self.stream);
        let (bp, _g6) = b1.device_ptr(&self.stream);
        let (np, _g7) = n16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 675); bounds checked above, every type < 3
        check(unsafe {
            f(
                xp as *mut _,
                fp as *const _,
                tp as *const _,
                rp as *const _,
                wp as *const _,
                bp as *const _,
                np as *mut _,
                rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// `dst[i] = src[idx[i]]` over rows of `width` halves.
    pub fn gather_rows_f16(
        &self,
        src: &CudaSlice<f16>,
        idx: &CudaSlice<u32>,
        dst: &mut CudaSlice<f16>,
        n: usize,
        width: usize,
    ) -> Result<(), GpuError> {
        if idx.len() < n || dst.len() < n * width || !(width * 2).is_multiple_of(16) {
            return Err(oob("gather_rows_f16: buffers under the gather geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        self.gather_rows_raw(sp, idx, dp, n, width * 2)
    }

    /// `dst[i] = src[idx[i]]` over rows of `width` floats.
    pub fn gather_rows_f32(
        &self,
        src: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        dst: &mut CudaSlice<f32>,
        n: usize,
        width: usize,
    ) -> Result<(), GpuError> {
        if idx.len() < n || dst.len() < n * width || !(width * 4).is_multiple_of(16) {
            return Err(oob("gather_rows_f32: buffers under the gather geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        self.gather_rows_raw(sp, idx, dp, n, width * 4)
    }

    fn gather_rows_raw(
        &self,
        src: u64,
        idx: &CudaSlice<u32>,
        dst: u64,
        n: usize,
        row_bytes: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gather_rows
            .ok_or(GpuError::MissingOp("gather_rows"))?;
        let (ip, _g) = idx.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 676); the typed wrappers checked the
        // destination, every index is a row of src by the caller's build
        check(unsafe {
            f(
                src as *const _,
                ip as *const _,
                dst as *mut _,
                n as u32,
                row_bytes as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One output per row: `out[i] = g[i] . w + b` (the scorer's Linear(d, 1)).
    pub fn laya_rowdot(
        &self,
        g: &CudaSlice<f16>,
        w: &CudaSlice<f32>,
        b: f32,
        out: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .laya_rowdot
            .ok_or(GpuError::MissingOp("laya_rowdot"))?;
        if g.len() < m * n || w.len() < n || out.len() < m {
            return Err(oob("laya_rowdot: buffers under the row geometry"));
        }
        let (gp, _g1) = g.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 677); bounds checked above
        check(unsafe {
            f(
                gp as *const _,
                wp as *const _,
                b,
                op as *mut _,
                m as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The Laya act/escalate head: one block a question, off its [CLS] row
    /// (`xcls[x_row0 + q]`) and the softmax of its own option logits
    /// (`logits[qoff[q]..qoff[q + 1]]`); `out` is `[nq][n_act]` probabilities.
    #[allow(clippy::too_many_arguments)]
    pub fn laya_act_head(
        &self,
        logits: &CudaSlice<f32>,
        qoff: &CudaSlice<u32>,
        xcls: &CudaSlice<f32>,
        x_row0: usize,
        w0: &CudaSlice<f16>,
        b0: &CudaSlice<f32>,
        w2: &CudaSlice<f16>,
        b2: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        nq: usize,
        d: usize,
        hid: usize,
        n_act: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .laya_act_head
            .ok_or(GpuError::MissingOp("laya_act_head"))?;
        if qoff.len() < nq + 1
            || xcls.len() < (x_row0 + nq) * d
            || w0.len() < hid * (d + 4)
            || b0.len() < hid
            || w2.len() < n_act * hid
            || b2.len() < n_act
            || out.len() < nq * n_act
        {
            return Err(oob("laya_act_head: buffers under the head geometry"));
        }
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (qp, _g2) = qoff.device_ptr(&self.stream);
        let (xp, _g3) = xcls.device_ptr(&self.stream);
        let xp = xp + (x_row0 * d * std::mem::size_of::<f32>()) as u64;
        let (w0p, _g4) = w0.device_ptr(&self.stream);
        let (b0p, _g5) = b0.device_ptr(&self.stream);
        let (w2p, _g6) = w2.device_ptr(&self.stream);
        let (b2p, _g7) = b2.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 678); bounds checked above, qoff indexes
        // inside logits by the caller's build
        check(unsafe {
            f(
                lp as *const _,
                qp as *const _,
                xp as *const _,
                w0p as *const _,
                b0p as *const _,
                w2p as *const _,
                b2p as *const _,
                op as *mut _,
                nq as u32,
                d as u32,
                hid as u32,
                n_act as u32,
                self.stream_ptr(),
            )
        })
    }
}
