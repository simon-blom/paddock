//! The Wan-shaped residual VAE's building blocks, shared by the decoder
//! (`vae.rs`) and the encoder (`vae_encode.rs`): the conv / residual /
//! attention blocks as loaded from the official F32 safetensors (conv
//! weights as f16 GEMM planes in the im2row order, biases and norm gammas
//! f32) and the ops over NHWC f32 planes that run them. Every conv is
//! stripe-tiled im2row + the f16 GEMM, so the 9x staging plane is bounded
//! by a stripe whatever the plane.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::safetensors::{SafetensorsFile, StDtype};

use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;

/// Bytes of f16 staging one conv stripe may use: bounds `ny * W * 9C * 2`.
const STRIPE_BYTES: usize = 256 << 20;
/// Bytes of f32 attention scores one query chunk of a mid block may hold
/// (`rows x positions`): the 29584 latent positions of a 2752^2 picture
/// would otherwise be a 3.5 GB plane.
const ATTN_SCORE_BYTES: usize = 256 << 20;

/// A 3x3 conv: `[c_out][9 * c_in]` f16 (tap-outer) + f32 bias.
pub(super) struct Conv3 {
    pub w: CudaSlice<f16>,
    pub b: CudaSlice<f32>,
    pub c_in: usize,
    pub c_out: usize,
}

/// A 1x1 conv: `[c_out][c_in]` f16 + f32 bias.
pub(super) struct Conv1 {
    pub w: CudaSlice<f16>,
    pub b: CudaSlice<f32>,
    pub c_in: usize,
    pub c_out: usize,
}

pub(super) struct ResBlock {
    pub norm1: CudaSlice<f32>,
    pub conv1: Conv3,
    pub norm2: CudaSlice<f32>,
    pub conv2: Conv3,
    pub shortcut: Option<Conv1>,
}

pub(super) struct AttnBlock {
    pub norm: CudaSlice<f32>,
    pub qkv: Conv1,
    pub proj: Conv1,
    pub c: usize,
}

/// An F32 tensor from the file as host f32.
pub(super) fn f32_of(
    st: &SafetensorsFile,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), GpuModelError> {
    let (t, bytes) = st
        .bytes(name)
        .ok_or_else(|| GpuModelError::MissingMeta(format!("vae tensor {name}")))?;
    if t.dtype != StDtype::F32 {
        return Err(GpuModelError::Unsupported(format!(
            "vae tensor {name} is {:?}, expected F32",
            t.dtype
        )));
    }
    let v = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok((v, t.shape.clone()))
}

/// The loaders, counting device bytes into `bytes` as they go.
pub(super) struct Loader<'a> {
    pub exec: Arc<GpuExecutor>,
    pub st: &'a SafetensorsFile,
    pub bytes: u64,
}

impl Loader<'_> {
    pub fn vec(&self, name: &str) -> Result<CudaSlice<f32>, GpuModelError> {
        let (v, _) = f32_of(self.st, name)?;
        Ok(self.exec.to_device(&v)?)
    }

    pub fn conv3(&mut self, base: &str) -> Result<Conv3, GpuModelError> {
        let (w, shape) = f32_of(self.st, &format!("{base}.weight"))?;
        let (b, _) = f32_of(self.st, &format!("{base}.bias"))?;
        let [c_out, c_in, kh, kw] = [shape[0], shape[1], shape[2], shape[3]];
        if kh != 3 || kw != 3 {
            return Err(GpuModelError::Unsupported(format!(
                "vae {base}: {kh}x{kw}, expected 3x3"
            )));
        }
        // [o][ci][ky][kx] -> [o][(ky*3+kx)*c_in + ci]
        let mut g = vec![f16::ZERO; c_out * 9 * c_in];
        for o in 0..c_out {
            for ci in 0..c_in {
                for t in 0..9 {
                    g[o * 9 * c_in + t * c_in + ci] = f16::from_f32(w[(o * c_in + ci) * 9 + t]);
                }
            }
        }
        self.bytes += (g.len() * 2 + b.len() * 4) as u64;
        Ok(Conv3 {
            w: self.exec.f16_to_device(&g)?,
            b: self.exec.to_device(&b)?,
            c_in,
            c_out,
        })
    }

    /// A 1x1 conv; `rows` keeps only the first output channels (None = all)
    /// - the encoder's quant conv is read for its mean half alone.
    pub fn conv1(&mut self, base: &str, rows: Option<usize>) -> Result<Conv1, GpuModelError> {
        let (w, shape) = f32_of(self.st, &format!("{base}.weight"))?;
        let (b, _) = f32_of(self.st, &format!("{base}.bias"))?;
        let (c_out_all, c_in) = (shape[0], shape[1]);
        let c_out = rows.unwrap_or(c_out_all).min(c_out_all);
        let g: Vec<f16> = w[..c_out * c_in]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        self.bytes += (g.len() * 2 + c_out * 4) as u64;
        Ok(Conv1 {
            w: self.exec.f16_to_device(&g)?,
            b: self.exec.to_device(&b[..c_out])?,
            c_in,
            c_out,
        })
    }

    pub fn res(
        &mut self,
        base: &str,
        c_in: usize,
        c_out: usize,
    ) -> Result<ResBlock, GpuModelError> {
        Ok(ResBlock {
            norm1: self.vec(&format!("{base}.norm1.gamma"))?,
            conv1: self.conv3(&format!("{base}.conv1"))?,
            norm2: self.vec(&format!("{base}.norm2.gamma"))?,
            conv2: self.conv3(&format!("{base}.conv2"))?,
            shortcut: if c_in != c_out {
                Some(self.conv1(&format!("{base}.conv_shortcut"), None)?)
            } else {
                None
            },
        })
    }

    pub fn attn(&mut self, base: &str, c: usize) -> Result<AttnBlock, GpuModelError> {
        Ok(AttnBlock {
            norm: self.vec(&format!("{base}.norm.gamma"))?,
            qkv: self.conv1(&format!("{base}.to_qkv"), None)?,
            proj: self.conv1(&format!("{base}.proj"), None)?,
            c,
        })
    }
}

/// 1x1 conv over `[rows][c_in]` f16 -> `[rows][c_out]` f32 (+ bias).
pub(super) fn conv1(
    exec: &GpuExecutor,
    x16: &CudaSlice<f16>,
    rows: usize,
    cv: &Conv1,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let mut y = exec.alloc(rows * cv.c_out)?;
    exec.f16_gemm(&cv.w, x16, &mut y, cv.c_in, cv.c_out, rows, 0.0)?;
    exec.bias_add(&mut y, &cv.b, rows, cv.c_out)?;
    Ok(y)
}

/// How a 3x3 conv reads its source: the plain stride-1 / pad-1 conv, the
/// same through a nearest-exact 2x upsample (output 2h x 2w), or the
/// encoder's stride-2 downsample with its (0, 1, 0, 1) zero pad (output
/// h/2 x w/2).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Conv3Mode {
    Same,
    Up2,
    Down2,
}

/// 3x3 conv over an f16 NHWC `[h][w][c_in]` source, stripe by stripe, into
/// a new f32 plane; `residual` (same shape as the output) is added.
pub(super) fn conv3(
    exec: &GpuExecutor,
    src16: &CudaSlice<f16>,
    h: usize,
    w: usize,
    cv: &Conv3,
    mode: Conv3Mode,
    residual: Option<&CudaSlice<f32>>,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let (h_out, w_out) = match mode {
        Conv3Mode::Same => (h, w),
        Conv3Mode::Up2 => (2 * h, 2 * w),
        Conv3Mode::Down2 => (h / 2, w / 2),
    };
    let k = 9 * cv.c_in;
    let ny = (STRIPE_BYTES / (w_out * k * 2)).clamp(1, h_out);
    let mut stage = exec.alloc_f16(ny * w_out * k)?;
    let mut ys = exec.alloc(ny * w_out * cv.c_out)?;
    let mut out = exec.alloc(h_out * w_out * cv.c_out)?;
    let mut y0 = 0;
    while y0 < h_out {
        let rows_y = ny.min(h_out - y0);
        let rows = rows_y * w_out;
        match mode {
            Conv3Mode::Down2 => {
                exec.vae_im2row3_down(src16, &mut stage, h, w, cv.c_in, y0, rows_y)?
            }
            _ => exec.vae_im2row3(
                src16,
                &mut stage,
                h,
                w,
                cv.c_in,
                y0,
                rows_y,
                mode == Conv3Mode::Up2,
            )?,
        }
        let beta = match residual {
            Some(r) => {
                exec.copy_region(r, y0 * w_out * cv.c_out, &mut ys, 0, rows * cv.c_out)?;
                1.0
            }
            None => 0.0,
        };
        exec.f16_gemm(&cv.w, &stage, &mut ys, k, cv.c_out, rows, beta)?;
        exec.bias_add(&mut ys, &cv.b, rows, cv.c_out)?;
        exec.copy_region(&ys, 0, &mut out, y0 * w_out * cv.c_out, rows * cv.c_out)?;
        y0 += rows_y;
    }
    Ok(out)
}

/// One residual block over an NHWC f32 plane.
pub(super) fn resblock(
    exec: &GpuExecutor,
    x: &CudaSlice<f32>,
    h: usize,
    w: usize,
    rb: &ResBlock,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let p = h * w;
    let (c_in, c_out) = (rb.conv1.c_in, rb.conv1.c_out);
    let mut n16 = exec.alloc_f16(p * c_in)?;
    exec.vae_norm_f16(x, &rb.norm1, &mut n16, p, c_in, true)?;
    let t = conv3(exec, &n16, h, w, &rb.conv1, Conv3Mode::Same, None)?;
    drop(n16);
    let mut n16b = exec.alloc_f16(p * c_out)?;
    exec.vae_norm_f16(&t, &rb.norm2, &mut n16b, p, c_out, true)?;
    drop(t);
    let shortcut = match &rb.shortcut {
        None => None,
        Some(sc) => {
            let mut x16 = exec.alloc_f16(p * c_in)?;
            exec.convert_f32_f16(x, &mut x16, p * c_in)?;
            Some(conv1(exec, &x16, p, sc)?)
        }
    };
    conv3(
        exec,
        &n16b,
        h,
        w,
        &rb.conv2,
        Conv3Mode::Same,
        Some(shortcut.as_ref().unwrap_or(x)),
    )
}

/// A mid block's single-head attention over every pixel, the score plane
/// in query chunks: `[rows][p] = q_chunk . k^T` (k as the weight operand),
/// softmax per row, then the chunk's rows of P V - the same arithmetic per
/// element as one whole score plane, a bounded piece of it at a time.
pub(super) fn attention(
    exec: &GpuExecutor,
    x: &CudaSlice<f32>,
    h: usize,
    w: usize,
    ab: &AttnBlock,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let (p, c) = (h * w, ab.c);
    let mut n16 = exec.alloc_f16(p * c)?;
    exec.vae_norm_f16(x, &ab.norm, &mut n16, p, c, false)?;
    let qkv = conv1(exec, &n16, p, &ab.qkv)?;
    drop(n16);
    let mut q16 = exec.alloc_f16(p * c)?;
    let mut k16 = exec.alloc_f16(p * c)?;
    let mut v16 = exec.alloc_f16(p * c)?;
    exec.dit_split3_f16(
        &qkv,
        &mut q16,
        &mut k16,
        &mut v16,
        p,
        c,
        1.0 / (c as f32).sqrt(),
    )?;
    drop(qkv);
    let mut vt = exec.alloc_f16(c * p)?;
    exec.dit_transpose_f16(&v16, &mut vt, p, c)?;
    drop(v16);
    let mut o = exec.alloc(p * c)?;
    let chunk = (ATTN_SCORE_BYTES / (p * 4)).clamp(1, p);
    let mut q0 = 0;
    while q0 < p {
        let rows = chunk.min(p - q0);
        let mut qc = exec.alloc_f16(rows * c)?;
        exec.copy_region(&q16, q0 * c, &mut qc, 0, rows * c)?;
        let mut s = exec.alloc(rows * p)?;
        exec.f16_gemm(&k16, &qc, &mut s, c, p, rows, 0.0)?;
        exec.dit_softmax_rows(&mut s, rows, p, 1.0)?;
        let mut p16 = exec.alloc_f16(rows * p)?;
        exec.convert_f32_f16(&s, &mut p16, rows * p)?;
        drop(s);
        let mut oc = exec.alloc(rows * c)?;
        exec.f16_gemm(&vt, &p16, &mut oc, p, c, rows, 0.0)?;
        exec.copy_region(&oc, 0, &mut o, q0 * c, rows * c)?;
        q0 += rows;
    }
    let mut o16 = exec.alloc_f16(p * c)?;
    exec.convert_f32_f16(&o, &mut o16, p * c)?;
    drop(o);
    let mut y = conv1(exec, &o16, p, &ab.proj)?;
    exec.add(&mut y, x, p * c)?;
    Ok(y)
}
