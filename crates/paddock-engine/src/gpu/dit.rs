//! Diffusion-transformer glue (pack segment `dit.cuh`, slots 632-640): the
//! image-generation lane's own small ops. The heavy work of a DiT step - the
//! block GEMMs, the target-block attention, the modulated layernorm - runs on
//! lanes that already exist; these are the model's residue.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::check;
use super::{GpuError, GpuExecutor};

impl GpuExecutor {
    /// 3-axis complex rope in place over `[rows][n_heads][hd]` f32. `pos` is
    /// `[rows][3]` i32 (frame, h, w); `axes` the per-axis pair widths summing
    /// to `hd`; `theta` the base (10000 for Qwen-Image).
    #[allow(clippy::too_many_arguments)]
    pub fn dit_rope(
        &self,
        x: &mut CudaSlice<f32>,
        pos: &CudaSlice<i32>,
        rows: usize,
        n_heads: usize,
        hd: usize,
        axes: [usize; 3],
        theta: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_rope
            .ok_or(GpuError::MissingOp("dit_rope"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        // SAFETY: ABI contract; the entry refuses an axis split that does
        // not tile the head.
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                rows as u32,
                n_heads as u32,
                hd as u32,
                axes[0] as u32,
                axes[1] as u32,
                axes[2] as u32,
                theta,
                self.stream_ptr(),
            )
        })
    }

    /// Initial latent noise in torch's CUDA `randn` layout, packed
    /// `[token][channel]`. `offset` counts prior draws on the same seed
    /// (0 for the first image of a request).
    pub fn dit_philox_randn(
        &self,
        out: &mut CudaSlice<f32>,
        seed: u64,
        offset: u32,
        n_tokens: usize,
        channels: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_philox_randn
            .ok_or(GpuError::MissingOp("dit_philox_randn"))?;
        let (op, _g) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; pointer + stream live across the call.
        check(unsafe {
            f(
                op as *mut _,
                seed as u32,
                (seed >> 32) as u32,
                offset,
                n_tokens as u32,
                channels as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `x[r][c] += g[c] * y[r][c]` over `[rows][n]` f32 - the gated residual.
    pub fn dit_gated_add(
        &self,
        x: &mut CudaSlice<f32>,
        y: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_gated_add
            .ok_or(GpuError::MissingOp("dit_gated_add"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (yp, _g2) = y.device_ptr(&self.stream);
        let (gp, _g3) = g.device_ptr(&self.stream);
        // SAFETY: ABI contract; pointers + stream live across the call.
        check(unsafe {
            f(
                xp as *mut _,
                yp as *const _,
                gp as *const _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Bare SiLU in place over `n` f32 values.
    pub fn dit_silu(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_silu
            .ok_or(GpuError::MissingOp("dit_silu"))?;
        let (xp, _g) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; pointer + stream live across the call.
        check(unsafe { f(xp as *mut _, n as u32, self.stream_ptr()) })
    }

    /// Row softmax in place over `[rows][n]` f32, `scale` applied before the
    /// max.
    pub fn dit_softmax_rows(
        &self,
        x: &mut CudaSlice<f32>,
        rows: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_softmax_rows
            .ok_or(GpuError::MissingOp("dit_softmax_rows"))?;
        let (xp, _g) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; pointer + stream live across the call.
        check(unsafe {
            f(
                xp as *mut _,
                rows as u32,
                n as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// `[rows][cols]` f16 -> `[cols][rows]` f16.
    pub fn dit_transpose_f16(
        &self,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f16>,
        rows: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_transpose_f16
            .ok_or(GpuError::MissingOp("dit_transpose_f16"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; pointers + stream live across the call.
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                rows as u32,
                cols as u32,
                self.stream_ptr(),
            )
        })
    }

    /// An f32 `[rows][3C]` qkv landing into three f16 `[rows][C]` planes,
    /// q scaled by `qscale` before its round.
    #[allow(clippy::too_many_arguments)]
    pub fn dit_split3_f16(
        &self,
        src: &CudaSlice<f32>,
        q: &mut CudaSlice<f16>,
        k: &mut CudaSlice<f16>,
        v: &mut CudaSlice<f16>,
        rows: usize,
        c: usize,
        qscale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_split3_f16
            .ok_or(GpuError::MissingOp("dit_split3_f16"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (kp, _g3) = k.device_ptr_mut(&self.stream);
        let (vp, _g4) = v.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; pointers + stream live across the call.
        check(unsafe {
            f(
                sp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                rows as u32,
                c as u32,
                qscale,
                self.stream_ptr(),
            )
        })
    }

    /// `x[r][c] = x[r][c] * a[c] + b[c]` over `[rows][n]` f32.
    pub fn dit_affine_cols(
        &self,
        x: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_affine_cols
            .ok_or(GpuError::MissingOp("dit_affine_cols"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (ap, _g2) = a.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        // SAFETY: ABI contract; pointers + stream live across the call.
        check(unsafe {
            f(
                xp as *mut _,
                ap as *const _,
                bp as *const _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Decoded f32 samples to 8-bit: `round((x / 2 + 0.5) * 255)` after the
    /// [-1, 1] clamp, channel order kept.
    pub fn dit_to_u8(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dit_to_u8
            .ok_or(GpuError::MissingOp("dit_to_u8"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; pointers + stream live across the call.
        check(unsafe { f(xp as *const _, op as *mut _, n as u32, self.stream_ptr()) })
    }

    /// Whether the pack carries the image-generation glue (slots 632-643).
    pub fn has_dit(&self) -> bool {
        self.kernels.dit_rope.is_some()
            && self.kernels.dit_philox_randn.is_some()
            && self.kernels.vae_dupup_add.is_some()
    }
}
