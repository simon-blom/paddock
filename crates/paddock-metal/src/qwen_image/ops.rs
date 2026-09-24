//! Native Metal primitives used by the bounded VAE graph. Typed buffers keep
//! element offsets distinct from byte offsets; all host access is fenced.
use crate::device::{Buffer, MetalDevice, MetalError, Result};
use half::f16;
use objc2_metal::MTLBuffer;
use std::{marker::PhantomData, ops::Deref};

pub(super) struct Tensor<T> {
    pub buffer: Buffer,
    _element: PhantomData<T>,
}
impl<T> Deref for Tensor<T> {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        &self.buffer
    }
}
pub(super) struct Ops {
    pub device: MetalDevice,
}

impl Ops {
    pub fn alloc<T>(&self, n: usize) -> Result<Tensor<T>> {
        let bytes = n
            .checked_mul(size_of::<T>())
            .ok_or_else(|| MetalError::Memory("image shape overflow".into()))?;
        Ok(Tensor {
            buffer: self.device.alloc(bytes)?,
            _element: PhantomData,
        })
    }
    pub fn alloc_f16(&self, n: usize) -> Result<Tensor<f16>> {
        self.alloc(n)
    }
    pub fn alloc_u8(&self, n: usize) -> Result<Tensor<u8>> {
        self.alloc(n)
    }
    pub fn to_device(&self, x: &[f32]) -> Result<Tensor<f32>> {
        self.upload(x)
    }
    pub fn f16_to_device(&self, x: &[f16]) -> Result<Tensor<f16>> {
        self.upload(x)
    }
    fn upload<T: Copy>(&self, x: &[T]) -> Result<Tensor<T>> {
        // Used only for initialized, padding-free f32 and IEEE f16 slices.
        let bytes = unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), size_of_val(x)) };
        Ok(Tensor {
            buffer: self.device.upload(bytes)?,
            _element: PhantomData,
        })
    }
    pub fn copy_region<T>(
        &self,
        src: &Tensor<T>,
        so: usize,
        dst: &mut Tensor<T>,
        doff: usize,
        n: usize,
    ) -> Result<()> {
        let s = size_of::<T>();
        self.device
            .copy_regions(&[(&src.buffer, so * s, &dst.buffer, doff * s, n * s)])
    }
    pub fn to_host_u8_len(&self, x: &Tensor<u8>, n: usize) -> Result<Vec<u8>> {
        if n > x.len() {
            return Err(MetalError::Model("image readback range".into()));
        }
        Ok(
            unsafe { std::slice::from_raw_parts(x.raw.contents().as_ptr().cast::<u8>(), n) }
                .to_vec(),
        )
    }
    pub fn run(
        &self,
        name: &str,
        bufs: &[&Buffer],
        p: &[u32],
        groups: [usize; 3],
        threads: usize,
    ) -> Result<()> {
        let cmd = self.device.begin()?;
        cmd.dispatch(name, bufs, p, groups, threads);
        cmd.finish()?;
        Ok(())
    }
    pub fn convert_f32_f16(&self, x: &Tensor<f32>, y: &mut Tensor<f16>, n: usize) -> Result<()> {
        self.run(
            "qi_store_half",
            &[x, y],
            &[n as u32, 0],
            [n.div_ceil(256), 1, 1],
            256,
        )
    }
    pub fn bias_add(
        &self,
        x: &mut Tensor<f32>,
        b: &Tensor<f32>,
        rows: usize,
        c: usize,
    ) -> Result<()> {
        self.run(
            "qi_bias",
            &[x, b],
            &[(rows * c) as u32, c as u32],
            [(rows * c).div_ceil(256), 1, 1],
            256,
        )
    }
    pub fn scale_add(
        &self,
        x: &mut Tensor<f32>,
        y: &Tensor<f32>,
        scale: f32,
        n: usize,
    ) -> Result<()> {
        self.run(
            "residual",
            &[x, y],
            &[n as u32, scale.to_bits()],
            [n.div_ceil(256), 1, 1],
            256,
        )
    }
    pub fn add(&self, x: &mut Tensor<f32>, y: &Tensor<f32>, n: usize) -> Result<()> {
        self.scale_add(x, y, 1., n)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn f16_gemm(
        &self,
        w: &Tensor<f16>,
        x: &Tensor<f16>,
        y: &mut Tensor<f32>,
        k: usize,
        n: usize,
        m: usize,
        beta: f32,
    ) -> Result<()> {
        let cmd = self.device.begin()?;
        // Zero bias, so the existing fused residual epilogue implements beta=1.
        let bias = self.device.alloc(n * 4)?;
        cmd.dispatch(
            "vis_mm64",
            &[w, x, y, &bias],
            &[k as u32, n as u32, m as u32, if beta == 0. { 0 } else { 2 }],
            [n.div_ceil(64), m.div_ceil(64), 1],
            128,
        );
        cmd.finish()?;
        Ok(())
    }
    pub fn matvec_f32_raw(
        &self,
        w: &Tensor<f32>,
        k: usize,
        n: usize,
        x: &Tensor<f32>,
        y: &mut Tensor<f32>,
        m: usize,
    ) -> Result<()> {
        self.run(
            "qi_f32_mm",
            &[w, x, y],
            &[k as u32, n as u32, m as u32, 0],
            [n.div_ceil(64), m.div_ceil(32), 1],
            128,
        )
    }
    pub fn dit_affine_cols(
        &self,
        x: &mut Tensor<f32>,
        s: &Tensor<f32>,
        b: &Tensor<f32>,
        m: usize,
        n: usize,
    ) -> Result<()> {
        self.run(
            "qi_affine",
            &[x, s, b],
            &[(m * n) as u32, n as u32],
            [(m * n).div_ceil(256), 1, 1],
            256,
        )
    }
    pub fn vae_norm_f16(
        &self,
        x: &Tensor<f32>,
        g: &Tensor<f32>,
        y: &mut Tensor<f16>,
        m: usize,
        c: usize,
        act: bool,
    ) -> Result<()> {
        self.run(
            "qi_vae_norm",
            &[x, g, y],
            &[c as u32, u32::from(act)],
            [m, 1, 1],
            256,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn im2row(
        &self,
        x: &Tensor<f16>,
        y: &mut Tensor<f16>,
        h: usize,
        w: usize,
        c: usize,
        y0: usize,
        ny: usize,
        mode: u32,
    ) -> Result<()> {
        let wo = match mode {
            1 => w * 2,
            2 => w / 2,
            _ => w,
        };
        self.run(
            "qi_im2row",
            &[x, y],
            &[h as u32, w as u32, c as u32, y0 as u32, ny as u32, mode],
            [(ny * wo * 9 * c).div_ceil(256), 1, 1],
            256,
        )
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn vae_im2row3(
        &self,
        x: &Tensor<f16>,
        y: &mut Tensor<f16>,
        h: usize,
        w: usize,
        c: usize,
        y0: usize,
        ny: usize,
        up: bool,
    ) -> Result<()> {
        self.im2row(x, y, h, w, c, y0, ny, u32::from(up))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn vae_dupup_add(
        &self,
        y: &mut Tensor<f32>,
        x: &Tensor<f32>,
        h: usize,
        w: usize,
        ci: usize,
        co: usize,
        ft: usize,
        repeats: usize,
    ) -> Result<()> {
        self.run(
            "qi_dupup",
            &[y, x],
            &[
                h as u32,
                w as u32,
                ci as u32,
                co as u32,
                ft as u32,
                repeats as u32,
            ],
            [(4 * h * w * co).div_ceil(256), 1, 1],
            256,
        )
    }
    pub fn dit_to_u8(&self, x: &Tensor<f32>, y: &mut Tensor<u8>, n: usize) -> Result<()> {
        self.run(
            "qi_pixels",
            &[x, y],
            &[n as u32],
            [n.div_ceil(256), 1, 1],
            256,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dit_split3_f16(
        &self,
        x: &Tensor<f32>,
        q: &mut Tensor<f16>,
        k: &mut Tensor<f16>,
        v: &mut Tensor<f16>,
        rows: usize,
        c: usize,
        scale: f32,
    ) -> Result<()> {
        self.run(
            "qi_split3",
            &[x, q, k, v],
            &[rows as u32, c as u32, scale.to_bits()],
            [(rows * c).div_ceil(256), 1, 1],
            256,
        )
    }
    pub fn dit_transpose_f16(
        &self,
        x: &Tensor<f16>,
        y: &mut Tensor<f16>,
        rows: usize,
        c: usize,
    ) -> Result<()> {
        self.run(
            "qi_transpose",
            &[x, y],
            &[rows as u32, c as u32],
            [(rows * c).div_ceil(256), 1, 1],
            256,
        )
    }
    pub fn dit_softmax_rows(
        &self,
        x: &mut Tensor<f32>,
        rows: usize,
        c: usize,
        scale: f32,
    ) -> Result<()> {
        self.run(
            "qi_softmax",
            &[x],
            &[c as u32, scale.to_bits()],
            [rows, 1, 1],
            256,
        )
    }
}
