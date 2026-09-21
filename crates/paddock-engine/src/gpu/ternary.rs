//! The batch-1 decode lane of PrismML's dense ternary packing (PTQ1_0) and
//! the per-128 int8 activation quantizer it eats: pack slots 627 / 628. See
//! `packs/cuda/src/quant/ternary.cuh` for why the lane exists and what the
//! coarser activation scale costs.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use paddock_models::ggml_type::GgmlType;

/// Shared memory a block of the lane needs for `in_dim` (mirrors
/// `pd_trn_smem_bytes`): the 1 KB trit table, then 74 words a super-block.
fn ternary_smem_bytes(in_dim: usize) -> usize {
    1024 + (in_dim / 256) * 74 * 4
}

impl GpuExecutor {
    /// The pack carries the dedicated ternary decode lane and its quantizer.
    pub fn has_ternary_gemv_b128(&self) -> bool {
        self.kernels.ternary_gemv_b128.is_some() && self.kernels.quantize_q8_b128.is_some()
    }

    /// `w` can ride [`Self::ternary_gemv_b128`]: a PTQ1_0 plane whose rows
    /// are whole super-blocks and whose staging fits a block's shared memory.
    pub fn ternary_gemv_b128_fits(&self, w: &RepackedKQ) -> bool {
        self.has_ternary_gemv_b128()
            && w.ty == GgmlType::Ptq1_0
            && w.dims[0].is_multiple_of(256)
            && ternary_smem_bytes(w.dims[0]) <= 96 * 1024
    }

    /// int8 activations with one scale per 128: `q` holds `n` values, `scale`
    /// `n / 128`. Sound on Hadamard-rotated rows only.
    pub fn quantize_q8_b128(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8_b128
            .ok_or(GpuError::MissingOp("quantize_q8_b128"))?;
        debug_assert!(n.is_multiple_of(128));
        debug_assert!(x.len() >= n && q.len() >= n && scale.len() >= n / 128);
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The pack launches several ternary planes (or a gate|up pair with
    /// SwiGLU) off one staged row (slot 630).
    pub fn has_ternary_gemv_b128_multi(&self) -> bool {
        self.kernels.ternary_gemv_b128_multi.is_some()
    }

    /// Planes that may share one [`Self::ternary_gemv_b128_multi`] launch:
    /// all on the lane, one input width, row counts in multiples of 8.
    pub fn ternary_multi_fits(&self, planes: &[&RepackedKQ]) -> bool {
        self.has_ternary_gemv_b128_multi()
            && (1..=3).contains(&planes.len())
            && planes.iter().all(|w| {
                self.ternary_gemv_b128_fits(w)
                    && w.dims[0] == planes[0].dims[0]
                    && w.dims[1].is_multiple_of(8)
            })
    }

    /// `y_i = W_i xq` for up to three planes reading the same staged row, in
    /// one launch. Per row bit-identical to [`Self::ternary_gemv_b128`].
    pub fn ternary_gemv_b128_multi(
        &self,
        planes: &mut [(&RepackedKQ, &mut CudaSlice<f32>)],
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ternary_gemv_b128_multi
            .ok_or(GpuError::MissingOp("ternary_gemv_b128_multi"))?;
        let n = planes.len();
        if !(1..=3).contains(&n) {
            return Err(GpuError::Unsupported(format!(
                "{n} planes in one ternary launch"
            )));
        }
        let in_dim = planes[0].0.dims[0];
        debug_assert!(xq.len() >= in_dim && xs.len() >= in_dim / 128);
        let (xqp, _gx) = xq.device_ptr(&self.stream);
        let (xsp, _gs) = xs.device_ptr(&self.stream);
        let mut dp = [std::ptr::null::<core::ffi::c_void>(); 3];
        let mut rp = [std::ptr::null::<core::ffi::c_void>(); 3];
        let mut yp = [std::ptr::null_mut::<core::ffi::c_void>(); 3];
        let mut rows = [0u32; 3];
        let mut guards = Vec::with_capacity(9);
        for (i, (w, y)) in planes.iter_mut().enumerate() {
            debug_assert!(w.dims[0] == in_dim && y.len() >= w.dims[1]);
            let (d, g1) = w.data.device_ptr(&self.stream);
            let (r, g2) = w.scales.device_ptr(&self.stream);
            let (yy, g3) = y.device_ptr_mut(&self.stream);
            dp[i] = d as *const _;
            rp[i] = r as *const _;
            yp[i] = yy as *mut _;
            rows[i] = w.dims[1] as u32;
            guards.push(g1);
            guards.push(g2);
            guards.push(g3);
        }
        // SAFETY: pack ABI v1 contract; pointers + stream live across the
        // call (the guards above outlive it); unused planes pass null
        check(unsafe {
            f(
                dp[0],
                rp[0],
                dp[1],
                rp[1],
                dp[2],
                rp[2],
                xqp as *const _,
                xsp as *const _,
                yp[0],
                yp[1],
                yp[2],
                in_dim as u32,
                rows[0],
                rows[1],
                rows[2],
                n as u32,
                0,
                self.stream_ptr(),
            )
        })
    }

    /// `y = silu(gate xq) * (up xq)`: the FFN's first half in one launch.
    /// Bit-identical to the two GEMVs followed by `swiglu`.
    pub fn ternary_glu_b128(
        &self,
        gate: &RepackedKQ,
        up: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ternary_gemv_b128_multi
            .ok_or(GpuError::MissingOp("ternary_gemv_b128_multi"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        debug_assert!(up.dims[0] == in_dim && up.dims[1] == ff && y.len() >= ff);
        debug_assert!(xq.len() >= in_dim && xs.len() >= in_dim / 128);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gr, _g2) = gate.scales.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (ur, _g4) = up.scales.device_ptr(&self.stream);
        let (xqp, _g5) = xq.device_ptr(&self.stream);
        let (xsp, _g6) = xs.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                gd as *const _,
                gr as *const _,
                ud as *const _,
                ur as *const _,
                std::ptr::null(),
                std::ptr::null(),
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                in_dim as u32,
                ff as u32,
                ff as u32,
                0,
                2,
                1,
                self.stream_ptr(),
            )
        })
    }

    /// `y = W xq` for one row of per-128 int8 activations.
    pub fn ternary_gemv_b128(
        &self,
        w: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        self.ternary_gemv_b128_layout(w, xq, xs, y, false)
    }

    /// [`Self::ternary_gemv_b128`], optionally over the kernel's column-major
    /// operand layout - its probe comparand; serving never asks for it. A
    /// per-call switch so a bench can interleave the two inside one process.
    pub fn ternary_gemv_b128_layout(
        &self,
        w: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        column_major: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ternary_gemv_b128
            .ok_or(GpuError::MissingOp("ternary_gemv_b128"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(xq.len() >= in_dim && xs.len() >= in_dim / 128 && y.len() >= out_dim);
        let (raw_id, _, _) = kq_layout(w.ty).ok_or(GpuError::MissingOp("ternary layout"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                raw_id | ((column_major as u32) << 20),
                self.stream_ptr(),
            )
        })
    }
}
