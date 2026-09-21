//! The blockwise Walsh-Hadamard rotation a rotated-basis checkpoint asks of
//! its runtime (PrismML's Bonsai files, `prism.hadamard.*`): pack slot 626.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

/// Value-head geometry of a gated-delta-net output projection's input, for
/// files that set `prism.hadamard.gdn_v_grouped`: the engine keeps the heads
/// tiled (`head = k + n_k * r`), the rotated weight expects them grouped
/// (`head = r + rep * k`), so the rotation permutes whole heads on the way in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HadamardGdnHeads {
    pub head_dim: usize,
    pub n_k: usize,
    pub rep: usize,
}

impl GpuExecutor {
    /// The pack carries the rotation (slot 626).
    pub fn has_hadamard(&self) -> bool {
        self.kernels.hadamard_rows.is_some()
    }

    /// `y = H(s * x)` over each `block`-wide strip of every `width`-wide row:
    /// what a rotated matmul's input goes through. `signs` holds `width`
    /// values of +-1. `gdn` permutes value heads tiled -> grouped first.
    #[allow(clippy::too_many_arguments)]
    pub fn hadamard_rotate(
        &self,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        rows: usize,
        width: usize,
        block: usize,
        gdn: Option<HadamardGdnHeads>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .hadamard_rows
            .ok_or(GpuError::MissingOp("hadamard_rows"))?;
        debug_assert!(x.len() >= rows * width && y.len() >= rows * width);
        debug_assert!(signs.len() >= width);
        let (hd, nk, rep) = gdn.map_or((0, 0, 0), |g| (g.head_dim, g.n_k, g.rep));
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (yp, _g2) = y.device_ptr_mut(&self.stream);
        let (sp, _g3) = signs.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                xp as *const _,
                yp as *mut _,
                sp as *const _,
                rows as u32,
                width as u32,
                block as u32,
                0,
                hd as u32,
                nk as u32,
                rep as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The pack carries the rotation fused with the per-128 int8 quantizer
    /// (slot 629).
    pub fn has_hadamard_q8_b128(&self) -> bool {
        self.kernels.hadamard_rows_q8_b128.is_some()
    }

    /// One launch for `y = H(s * P x)` AND its per-128 int8 form (`xq`,
    /// `xs` = `width / 128` scales a row): what a rotated model's batch-1
    /// ternary lane reads. `y` takes the f32 rows as well; `None` rotates
    /// `x` in place (not with `gdn`, whose heads cross strips).
    /// Bit-identical to `hadamard_rotate` followed by `quantize_q8_b128`.
    #[allow(clippy::too_many_arguments)]
    pub fn hadamard_rotate_q8_b128(
        &self,
        x: &mut CudaSlice<f32>,
        y: Option<&mut CudaSlice<f32>>,
        signs: &CudaSlice<f32>,
        xq: &mut CudaSlice<i8>,
        xs: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        block: usize,
        gdn: Option<HadamardGdnHeads>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .hadamard_rows_q8_b128
            .ok_or(GpuError::MissingOp("hadamard_rows_q8_b128"))?;
        debug_assert!(x.len() >= rows * width && signs.len() >= width);
        debug_assert!(xq.len() >= rows * width && xs.len() >= rows * width / 128);
        if gdn.is_some() && y.is_none() {
            return Err(GpuError::Unsupported(
                "the head regroup cannot rotate in place".into(),
            ));
        }
        let (hd, nk, rep) = gdn.map_or((0, 0, 0), |g| (g.head_dim, g.n_k, g.rep));
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (sp, _g2) = signs.device_ptr(&self.stream);
        let (qp, _g3) = xq.device_ptr_mut(&self.stream);
        let (scp, _g4) = xs.device_ptr_mut(&self.stream);
        let yg = y.map(|y| {
            debug_assert!(y.len() >= rows * width);
            y.device_ptr_mut(&self.stream)
        });
        let yp = yg.as_ref().map_or(xp, |(p, _)| *p);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the
        // call. x == y is the documented in-place form.
        check(unsafe {
            f(
                xp as *const _,
                yp as *mut _,
                sp as *const _,
                qp as *mut _,
                scp as *mut _,
                rows as u32,
                width as u32,
                block as u32,
                hd as u32,
                nk as u32,
                rep as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The rotation in place: `x <- H(s * x)`, or with `inverse` the lookup
    /// side `x <- s * H(x)` that brings a row of a rotated table (the token
    /// embedding) back into the model's own basis.
    pub fn hadamard_rotate_inplace(
        &self,
        x: &mut CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        rows: usize,
        width: usize,
        block: usize,
        inverse: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .hadamard_rows
            .ok_or(GpuError::MissingOp("hadamard_rows"))?;
        debug_assert!(x.len() >= rows * width && signs.len() >= width);
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (sp, _g2) = signs.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract; x is read and written by the same
        // thread per element set, which the kernel documents as safe
        check(unsafe {
            f(
                xp as *const _,
                xp as *mut _,
                sp as *const _,
                rows as u32,
                width as u32,
                block as u32,
                inverse as u32,
                0,
                0,
                0,
                self.stream_ptr(),
            )
        })
    }
}
