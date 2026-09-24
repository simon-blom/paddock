//! QSA - Qwen3.8-Flash-Next's sparse attention - op wrappers (pack slots
//! 661-...; `packs/cuda/src/attn/qsa.cuh`).
//!
//! Part 1, the indexer: the query norm, and the compressed key cache every
//! attention layer keeps beside its KV - one bf16 row of 128 per 4-token
//! block, pooled from a small per-slot ring of raw keys. Semantics follow
//! the Flash-Next QSA design note. Parity-gated against
//! `paddock_kernels::reference::qwen4exp` in `tests/gpu_qwen4exp_ops.rs`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::types::KvDtype;
use super::{GpuError, GpuExecutor};

/// Positions a QSA block spans (the checkpoint's `compress_ratio`).
pub const QSA_BLOCK: usize = 4;

/// Which kernel serves [`GpuExecutor::q4x_qsa_logits`] and
/// [`GpuExecutor::q4x_qsa_attn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QsaRoute {
    /// The f32 SIMT kernels (slots 664, 666): the parity anchors, any shape.
    Simt,
    /// Tensor cores (slots 670, 669): bf16 scores, f16 attention - the dense
    /// f16 attention kernels' class; for the shapes the `*_fits` accept.
    Mma,
}

impl QsaRoute {
    /// Shapes the tensor-core attention takes: 4-token blocks, a kv group's
    /// heads in one 16-row MMA tile, 128- or 256-dim heads.
    pub fn attn_fits(nh: usize, nkv: usize, hd: usize, cr: usize) -> bool {
        cr == 4 && nkv > 0 && nh.is_multiple_of(nkv) && nh / nkv <= 16 && (hd == 128 || hd == 256)
    }

    /// Shapes the tensor-core scores take: the model's indexer, 4 heads of
    /// 128 over 4-token blocks.
    pub fn logits_fits(heads: usize, hd: usize, cr: usize) -> bool {
        heads == 4 && hd == 128 && cr == 4
    }
}

impl GpuExecutor {
    /// True when the pack carries the QSA attention ops (slots 666-667).
    pub fn has_qsa_attn(&self) -> bool {
        self.kernels.q4x_qsa_attn.is_some() && self.kernels.q4x_qsa_combine.is_some()
    }

    /// True when the pack carries the tensor-core QSA attention (slot 669).
    pub fn has_qsa_attn_mma(&self) -> bool {
        self.kernels.q4x_qsa_attn_mma.is_some() && self.kernels.q4x_qsa_combine.is_some()
    }

    /// True when the pack carries the tensor-core QSA scores (slot 670).
    pub fn has_qsa_logits_mma(&self) -> bool {
        self.kernels.q4x_qsa_logits_mma.is_some()
    }

    /// True when the pack carries the QSA selection ops (slots 664-665).
    pub fn has_qsa_select(&self) -> bool {
        self.kernels.q4x_qsa_logits.is_some() && self.kernels.q4x_qsa_topk.is_some()
    }

    /// True when the pack carries the QSA indexer ops (slots 661-663).
    pub fn has_qsa_indexer(&self) -> bool {
        self.kernels.q4x_idx_q.is_some()
            && self.kernels.q4x_idx_pool.is_some()
            && self.kernels.q4x_idx_store.is_some()
    }

    /// The indexer query, normalized: row `r`'s `heads` query heads of `hd`
    /// sit at `src[r*ld + h*hd ..]` (the fused q|k projection row); `dst`
    /// gets `[rows, heads, hd]` contiguous - per-head f32 RMS with the (1+w)
    /// weight `w` [hd], rounded to bf16 - ready for the rotary.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_idx_q(
        &self,
        src: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        rows: usize,
        heads: usize,
        hd: usize,
        ld: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_idx_q
            .ok_or(GpuError::MissingOp("q4x_idx_q"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (dp, _g3) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                sp as *const _,
                wp as *const _,
                dp as *mut _,
                rows as u32,
                heads as u32,
                hd as u32,
                ld as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Pool every row that closes a block (`(pos+1) % cr == 0`): its `cr`
    /// raw keys - from this launch's rows (`raw[r*ld + koff ..]`, the run's
    /// own earlier rows: same slot, consecutive positions) or else the slot's
    /// `ring` ([slots, ring_len, hd] f32) - meaned in f32, rounded to bf16,
    /// (1+w)-normalized with `w` [hd], rounded to bf16, into `stage[r]`
    /// ([rows, hd]) with the block's first position in `spos` ([4, rows],
    /// the rotary's axis-major layout). Other rows are left untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_idx_pool(
        &self,
        raw: &CudaSlice<f32>,
        ring: &CudaSlice<f32>,
        pos: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        w: &CudaSlice<f32>,
        stage: &mut CudaSlice<f32>,
        spos: &mut CudaSlice<u32>,
        rows: usize,
        hd: usize,
        ld: usize,
        koff: usize,
        ring_len: usize,
        cr: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_idx_pool
            .ok_or(GpuError::MissingOp("q4x_idx_pool"))?;
        let (rp, _g1) = raw.device_ptr(&self.stream);
        let (gp, _g2) = ring.device_ptr(&self.stream);
        let (pp, _g3) = pos.device_ptr(&self.stream);
        let (lp, _g4) = slots.device_ptr(&self.stream);
        let (wp, _g5) = w.device_ptr(&self.stream);
        let (stp, _g6) = stage.device_ptr_mut(&self.stream);
        let (spp, _g7) = spos.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                rp as *const _,
                gp as *const _,
                pp as *const _,
                lp as *const _,
                wp as *const _,
                stp as *mut _,
                spp as *mut _,
                rows as u32,
                hd as u32,
                ld as u32,
                koff as u32,
                ring_len as u32,
                cr as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// After the rotary ran over `stage`: rows that closed a block write it
    /// (bf16) to `cache` ([slots, cap, hd]) at block `pos / cr`; every row
    /// files its raw key in the slot's `ring` at `pos % ring_len`, unless a
    /// later row of this launch takes that entry (only a run's last
    /// `ring_len` rows land). `ring_len` must be >= `cr` + the deepest verify
    /// chunk, so a rejected draft never aliases a committed position.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_idx_store(
        &self,
        raw: &CudaSlice<f32>,
        stage: &CudaSlice<f32>,
        pos: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        cache: &mut CudaSlice<half::bf16>,
        ring: &mut CudaSlice<f32>,
        rows: usize,
        hd: usize,
        ld: usize,
        koff: usize,
        ring_len: usize,
        cr: usize,
        cap: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_idx_store
            .ok_or(GpuError::MissingOp("q4x_idx_store"))?;
        let (rp, _g1) = raw.device_ptr(&self.stream);
        let (sp, _g2) = stage.device_ptr(&self.stream);
        let (pp, _g3) = pos.device_ptr(&self.stream);
        let (lp, _g4) = slots.device_ptr(&self.stream);
        let (cp, _g5) = cache.device_ptr_mut(&self.stream);
        let (gp, _g6) = ring.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                rp as *const _,
                sp as *const _,
                pp as *const _,
                lp as *const _,
                cp as *mut _,
                gp as *mut _,
                rows as u32,
                hd as u32,
                ld as u32,
                koff as u32,
                ring_len as u32,
                cr as u32,
                cap as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Block scores for launch rows `row0 .. row0+rows`: for each, every
    /// complete block `b < nb = (pos+1)/cr` of its slot's compressed `cache`
    /// ([slots, cap, hd] bf16) scored as `sum_h relu(q_h . k_b)` with its
    /// `heads` query vectors (`q` [launch rows, heads, hd]) into
    /// `scores[(r-row0)*cap + b]`. Rows with `nb <= k` select everything and
    /// are skipped. The grid is fixed by `cap` (graph-safe across positions).
    /// `route` picks the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_qsa_logits(
        &self,
        route: QsaRoute,
        q: &CudaSlice<f32>,
        cache: &CudaSlice<half::bf16>,
        pos: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        scores: &mut CudaSlice<f32>,
        row0: usize,
        rows: usize,
        heads: usize,
        hd: usize,
        cap: usize,
        cr: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = match route {
            QsaRoute::Simt => self
                .kernels
                .q4x_qsa_logits
                .ok_or(GpuError::MissingOp("q4x_qsa_logits"))?,
            QsaRoute::Mma => self
                .kernels
                .q4x_qsa_logits_mma
                .ok_or(GpuError::MissingOp("q4x_qsa_logits_mma"))?,
        };
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (cp, _g2) = cache.device_ptr(&self.stream);
        let (pp, _g3) = pos.device_ptr(&self.stream);
        let (lp, _g4) = slots.device_ptr(&self.stream);
        let (sp, _g5) = scores.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                qp as *const _,
                cp as *const _,
                pp as *const _,
                lp as *const _,
                sp as *mut _,
                row0 as u32,
                rows as u32,
                heads as u32,
                hd as u32,
                cap as u32,
                cr as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Per launch row `row0 .. row0+rows`: the top `k` block ids of its
    /// scores (`scores[(r-row0)*cap ..]`, the first `nb = (pos+1)/cr`), in
    /// ascending order, into `sel[r*k ..]`, and the count (`min(k, nb)`) into
    /// `cnt[r]`. Exact radix select; the lowest ids win ties.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_qsa_topk(
        &self,
        scores: &CudaSlice<f32>,
        pos: &CudaSlice<u32>,
        sel: &mut CudaSlice<u32>,
        cnt: &mut CudaSlice<u32>,
        row0: usize,
        rows: usize,
        cap: usize,
        cr: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_qsa_topk
            .ok_or(GpuError::MissingOp("q4x_qsa_topk"))?;
        let (sp, _g1) = scores.device_ptr(&self.stream);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        let (lp, _g3) = sel.device_ptr_mut(&self.stream);
        let (np, _g4) = cnt.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                sp as *const _,
                pp as *const _,
                lp as *mut _,
                np as *mut _,
                row0 as u32,
                rows as u32,
                cap as u32,
                cr as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Attention over each row's selection, through `route`'s kernel: row `r` (position `pos[r]`, slot
    /// `slots[r]`) attends to the tokens of blocks `sel[r*k .. +cnt[r]]`
    /// (`cr` tokens each) and its tail `[cr*nb, pos]`, through the lane's
    /// slot-major `kc`/`vc` caches ([slots*max_ctx, nkv*hd], `kv_dtype`).
    /// `q` is [rows, nh, hd] (normed, rotated); partials land in `part_o`
    /// ([rows, nkv, splits, nh/nkv, hd]) and `part_ml` (the same with (m, l)).
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_qsa_attn(
        &self,
        route: QsaRoute,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        pos: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        sel: &CudaSlice<u32>,
        cnt: &CudaSlice<u32>,
        part_o: &mut CudaSlice<f32>,
        part_ml: &mut CudaSlice<f32>,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        max_ctx: usize,
        k: usize,
        cr: usize,
        splits: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = match route {
            QsaRoute::Simt => self
                .kernels
                .q4x_qsa_attn
                .ok_or(GpuError::MissingOp("q4x_qsa_attn"))?,
            QsaRoute::Mma => self
                .kernels
                .q4x_qsa_attn_mma
                .ok_or(GpuError::MissingOp("q4x_qsa_attn_mma"))?,
        };
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (pp, _g4) = pos.device_ptr(&self.stream);
        let (lp, _g5) = slots.device_ptr(&self.stream);
        let (sp, _g6) = sel.device_ptr(&self.stream);
        let (np, _g7) = cnt.device_ptr(&self.stream);
        let (op, _g8) = part_o.device_ptr_mut(&self.stream);
        let (mp, _g9) = part_ml.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                pp as *const _,
                lp as *const _,
                sp as *const _,
                np as *const _,
                op as *mut _,
                mp as *mut _,
                rows as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
                max_ctx as u32,
                k as u32,
                cr as u32,
                splits as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Merge `q4x_qsa_attn`'s split partials into `out` [rows, nh, hd].
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_qsa_combine(
        &self,
        part_o: &CudaSlice<f32>,
        part_ml: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        splits: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_qsa_combine
            .ok_or(GpuError::MissingOp("q4x_qsa_combine"))?;
        let (op, _g1) = part_o.device_ptr(&self.stream);
        let (mp, _g2) = part_ml.device_ptr(&self.stream);
        let (dp, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                op as *const _,
                mp as *const _,
                dp as *mut _,
                rows as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
                splits as u32,
                self.stream_ptr(),
            )
        })
    }
}
