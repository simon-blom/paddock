//! Image VAE conv glue (pack segment `conv/vae.cuh`, slots 641-643). The
//! convolutions themselves are im2row + the f16 GEMM (`f16_gemm`); these are
//! the pieces around them that the Wan-shaped residual VAE needs.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::check;
use super::{GpuError, GpuExecutor};

impl GpuExecutor {
    /// Per-pixel channel RMSNorm of an NHWC f32 `[rows][c]` plane
    /// (`normalize(x, dim=C) * sqrt(C) * gamma`), SiLU when `act`, written
    /// f16.
    pub fn vae_norm_f16(
        &self,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        c: usize,
        act: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vae_norm_f16
            .ok_or(GpuError::MissingOp("vae_norm_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (gp, _g2) = gamma.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; pointers + stream live across the call.
        check(unsafe {
            f(
                xp as *const _,
                gp as *const _,
                op as *mut _,
                rows as u32,
                c as u32,
                u32::from(act),
                self.stream_ptr(),
            )
        })
    }

    /// 3x3 / stride 1 / pad 1 im2row of output rows `y0 .. y0 + ny` from an
    /// f16 NHWC `[h][w][c]` source into `[ny * w_out][9c]` f16 staging;
    /// `up2` reads the source through a nearest-exact 2x upsample.
    #[allow(clippy::too_many_arguments)]
    pub fn vae_im2row3(
        &self,
        src: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        h: usize,
        w: usize,
        c: usize,
        y0: usize,
        ny: usize,
        up2: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vae_im2row3
            .ok_or(GpuError::MissingOp("vae_im2row3"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the entry refuses a stripe past the grid.
        check(unsafe {
            f(
                sp as *const _,
                op as *mut _,
                h as u32,
                w as u32,
                c as u32,
                y0 as u32,
                ny as u32,
                u32::from(up2),
                self.stream_ptr(),
            )
        })
    }

    /// The encoder's downsampler im2row: 3x3 / stride 2 with the (0, 1, 0, 1)
    /// zero pad, output rows `y0 .. y0 + ny` of the `h/2 x w/2` grid from an
    /// f16 NHWC `[h][w][c]` source into `[ny * w/2][9c]` f16 staging.
    #[allow(clippy::too_many_arguments)]
    pub fn vae_im2row3_down(
        &self,
        src: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        h: usize,
        w: usize,
        c: usize,
        y0: usize,
        ny: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vae_im2row3_down
            .ok_or(GpuError::MissingOp("vae_im2row3_down"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the entry refuses odd planes and a stripe
        // past the grid.
        check(unsafe {
            f(
                sp as *const _,
                op as *mut _,
                h as u32,
                w as u32,
                c as u32,
                y0 as u32,
                ny as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The AvgDown shortcut of an encoder down block added into an f32 NHWC
    /// `[h/fs][w/fs][c_out]` plane from `[h][w][c_in]`; `ft` is the temporal
    /// factor (a zero frame is padded in front when 2), `fs` the spatial one.
    #[allow(clippy::too_many_arguments)]
    pub fn vae_avgdown_add(
        &self,
        out: &mut CudaSlice<f32>,
        input: &CudaSlice<f32>,
        h: usize,
        w: usize,
        c_in: usize,
        c_out: usize,
        ft: usize,
        fs: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vae_avgdown_add
            .ok_or(GpuError::MissingOp("vae_avgdown_add"))?;
        let (op, _g1) = out.device_ptr_mut(&self.stream);
        let (ip, _g2) = input.device_ptr(&self.stream);
        // SAFETY: ABI contract; the entry refuses a shape that does not tile.
        check(unsafe {
            f(
                op as *mut _,
                ip as *const _,
                h as u32,
                w as u32,
                c_in as u32,
                c_out as u32,
                ft as u32,
                fs as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The DupUp shortcut added into an f32 NHWC `[2h][2w][c_out]` plane
    /// from `[h][w][c_in]`; `ft` is the temporal factor, `repeats` the
    /// channel repeat (`c_out * ft * 4 == c_in * repeats`).
    #[allow(clippy::too_many_arguments)]
    pub fn vae_dupup_add(
        &self,
        out: &mut CudaSlice<f32>,
        input: &CudaSlice<f32>,
        h: usize,
        w: usize,
        c_in: usize,
        c_out: usize,
        ft: usize,
        repeats: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vae_dupup_add
            .ok_or(GpuError::MissingOp("vae_dupup_add"))?;
        let (op, _g1) = out.device_ptr_mut(&self.stream);
        let (ip, _g2) = input.device_ptr(&self.stream);
        // SAFETY: ABI contract; the entry refuses a shape that does not tile.
        check(unsafe {
            f(
                op as *mut _,
                ip as *const _,
                h as u32,
                w as u32,
                c_in as u32,
                c_out as u32,
                ft as u32,
                repeats as u32,
                self.stream_ptr(),
            )
        })
    }
}
