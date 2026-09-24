//! Block-diffusion canvas ops (slots 647-650, `packs/cuda/src/diffusion/
//! canvas.cuh`): the transposed bf16 embedding plane the self-conditioning
//! matmul runs on, the per-row canvas sampler, the per-canvas
//! entropy-bounded accept step, and the label-id column gather.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

/// What one canvas's accept step reports back. The type lives in
/// `generator` (it builds for every backend); this keeps the `gpu::`
/// spelling the CUDA lane uses.
pub use crate::generator::CanvasStatus;

impl GpuExecutor {
    /// True when the pack carries the canvas op set (all four slots - a pack
    /// with some of them is a mismatched build, not a partial capability)
    /// AND the `win_pos` prefill entries: a canvas row's sliding window
    /// starts at its own position, not the block end, and only those
    /// entries can express that - without them the first rows of every
    /// block past ~768 prompt tokens would silently lose their oldest keys.
    pub fn has_canvas_ops(&self) -> bool {
        self.kernels.q8_embed_transpose_bf16.is_some()
            && self.kernels.canvas_sample.is_some()
            && self.kernels.canvas_accept.is_some()
            && self.kernels.gather_cols.is_some()
            && self.has_win_pos()
    }

    /// True when the pack carries the four `win_pos` prefill entries (slots
    /// 651-654). Callers with a bidirectional span pass their true positions
    /// through them; on an older pack the family keeps the bound-derived
    /// floor it always had (exact for causal rows).
    pub fn has_win_pos(&self) -> bool {
        self.kernels.attn_prefill_wp.is_some()
            && self.kernels.attn_prefill_f16_wp.is_some()
            && self.kernels.attn_prefill_f16_paged_wp.is_some()
            && self.kernels.attn_prefill_f16_paged2_wp.is_some()
    }

    /// The raw Q8_0 embedding `[vocab][embd]` dequantized into a TRANSPOSED
    /// bf16 plane `dst = [embd][vocab]` (`vocab * embd * 2` bytes): the NT
    /// weight `bf16_gemm` multiplies a `[rows][vocab]` activation with to
    /// get `[rows][embd]` - `softmax(logits) @ E`, the self-conditioning
    /// signal. embd must be a multiple of 32 (one Q8_0 block).
    pub fn q8_embed_transpose_bf16(
        &self,
        q8: &QuantTensor,
        dst: &mut CudaSlice<u8>,
        vocab: usize,
        embd: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_embed_transpose_bf16
            .ok_or(GpuError::MissingOp("q8_embed_transpose_bf16"))?;
        if q8.ty != paddock_models::ggml_type::GgmlType::Q8_0 {
            return Err(GpuError::Unsupported(format!(
                "embedding transpose wants a Q8_0 plane, got {:?}",
                q8.ty
            )));
        }
        if q8.bytes.len() < vocab * (embd / 32) * 34 || dst.len() < vocab * embd * 2 {
            return Err(GpuError::Driver(
                "embedding transpose: source or destination plane too small".into(),
            ));
        }
        let (sp, _g1) = q8.bytes.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                vocab as u32,
                embd as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The canvas row sampler over softcapped `logits` (`[rows][vocab]` f32,
    /// overwritten with the normalized probs): per row one Gumbel-max
    /// categorical draw at `inv_t[row]` (0 = deterministic, the argmax), the
    /// raw argmax, and the entropy of the temperature-scaled distribution.
    /// Draws are Philox4x32-10 keyed on `(seed, offset, row * vocab + i)`, so
    /// `offset` must differ per (slot, step) for independent rows.
    #[allow(clippy::too_many_arguments)]
    pub fn canvas_sample(
        &self,
        logits: &mut CudaSlice<f32>,
        inv_t: &CudaSlice<f32>,
        seed: u64,
        offset: u32,
        out_sample: &mut CudaSlice<u32>,
        out_argmax: &mut CudaSlice<u32>,
        out_entropy: &mut CudaSlice<f32>,
        rows: usize,
        vocab: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .canvas_sample
            .ok_or(GpuError::MissingOp("canvas_sample"))?;
        if logits.len() < rows * vocab
            || inv_t.len() < rows
            || out_sample.len() < rows
            || out_argmax.len() < rows
            || out_entropy.len() < rows
        {
            return Err(GpuError::Driver(
                "canvas_sample: a plane is too small".into(),
            ));
        }
        let (lp, _g1) = logits.device_ptr_mut(&self.stream);
        let (tp, _g2) = inv_t.device_ptr(&self.stream);
        let (sp, _g3) = out_sample.device_ptr_mut(&self.stream);
        let (ap, _g4) = out_argmax.device_ptr_mut(&self.stream);
        let (ep, _g5) = out_entropy.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                lp as *mut _,
                tp as *const _,
                seed as u32,
                (seed >> 32) as u32,
                offset,
                sp as *mut _,
                ap as *mut _,
                ep as *mut _,
                rows as u32,
                vocab as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One canvas's entropy-bounded accept step (see the kernel doc in
    /// abi.rs): `canvas` (`[w]`) becomes the next step's input, `hist`
    /// (`[stab][w]`, 0xFFFFFFFF before an entry exists) rolls, and `status`
    /// (`[4]` u32) receives the words `CanvasStatus::from_words` decodes.
    /// `offset` must differ per (slot, step) for independent re-noise draws.
    #[allow(clippy::too_many_arguments)]
    pub fn canvas_accept(
        &self,
        entropy: &CudaSlice<f32>,
        sampled: &CudaSlice<u32>,
        argmax: &CudaSlice<u32>,
        canvas: &mut CudaSlice<u32>,
        hist: &mut CudaSlice<u32>,
        status: &mut CudaSlice<u32>,
        w: usize,
        vocab: usize,
        stab: usize,
        step: u32,
        bound: f32,
        conf: f32,
        seed: u64,
        offset: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .canvas_accept
            .ok_or(GpuError::MissingOp("canvas_accept"))?;
        if w == 0 || w > 1024 {
            return Err(GpuError::Unsupported(format!(
                "canvas_accept: {w} positions (1..=1024)"
            )));
        }
        if entropy.len() < w
            || sampled.len() < w
            || argmax.len() < w
            || canvas.len() < w
            || hist.len() < stab * w
            || status.len() < 4
        {
            return Err(GpuError::Driver(
                "canvas_accept: a plane is too small".into(),
            ));
        }
        let (ep, _g1) = entropy.device_ptr(&self.stream);
        let (sp, _g2) = sampled.device_ptr(&self.stream);
        let (ap, _g3) = argmax.device_ptr(&self.stream);
        let (cp, _g4) = canvas.device_ptr_mut(&self.stream);
        let (hp, _g5) = hist.device_ptr_mut(&self.stream);
        let (stp, _g6) = status.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                ep as *const _,
                sp as *const _,
                ap as *const _,
                cp as *mut _,
                hp as *mut _,
                stp as *mut _,
                w as u32,
                vocab as u32,
                stab as u32,
                step,
                bound,
                conf,
                seed as u32,
                (seed >> 32) as u32,
                offset,
                self.stream_ptr(),
            )
        })
    }

    /// `out[r][j] = src[r][ids[j]]` for `k` column ids over an `[rows][n]`
    /// plane - the structured read's label probabilities.
    pub fn gather_cols(
        &self,
        src: &CudaSlice<f32>,
        ids: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gather_cols
            .ok_or(GpuError::MissingOp("gather_cols"))?;
        if src.len() < rows * n || ids.len() < k || out.len() < rows * k {
            return Err(GpuError::Driver("gather_cols: a plane is too small".into()));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (ip, _g2) = ids.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                sp as *const _,
                ip as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }
}
