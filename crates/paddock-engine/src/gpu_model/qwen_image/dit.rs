//! The single-stream DiT forward: the prefix pass (once per prompt) and the
//! target pass (once per step). Both walk the same 32 blocks; they differ in
//! which modulation row they read (t = 0 for the prefix, the sampled t for
//! the target), in their attention (causal over the text for the prefix,
//! everything for the target) and in what they keep (the prefix its K/V, the
//! target its velocity).
//!
//! A block, as diffusers writes it:
//!   x += tanh(g1) * attn((1 + s1) * LN(x))
//!   x += tanh(g2) * mlp((1 + s2) * LN(x))
//! with LN affine-free and (s1, g1, s2, g2) one shared 4 x hidden modulation
//! computed from the timestep for the whole network. That is the existing
//! layernorm with `1 + s` as its weight vector and zeros as its bias, the
//! prefill GEMM helpers the LLM families use, per-head RMSNorm on q/k, the
//! 3-axis rope, and the gated add - nothing bespoke but the rope and the
//! gate.
//!
//! Two plane classes, one forward: the Q8_0 / k-quant files run their block
//! GEMMs on the int8 and W4A8 prefill lanes off int8-staged activations; the
//! F16 file (the same-weights parity yardstick against stable-diffusion.cpp)
//! runs them on the in-house f16 GEMM off f16-staged activations. The block
//! loop stages once per LN and dispatches per plane.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;

use crate::gpu::{DeviceTensor, GpuExecutor, HalfTensor, KvDtype, QuantTensor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{
    gemv_any, prefill_ffn_down_any, prefill_mm_any, prefill_mm_pre_any, prefill_quant,
};

use super::sampler::timestep_embedding;
use super::{AXES_DIMS_ROPE, ROPE_THETA};

pub struct Hparams {
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub n_layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub context_dim: usize,
    pub eps: f32,
}

/// A matmul plane: quantized (Q8_0 or a k-quant, on the int8 / W4A8 lanes)
/// or f16 (the parity file, on the f16 GEMM).
pub enum Plane {
    Quant(QuantW),
    Half(HalfTensor),
}

impl Plane {
    pub fn bytes(&self) -> u64 {
        match self {
            Plane::Quant(q) => q.bytes(),
            Plane::Half(h) => h.bytes() as u64,
        }
    }

    fn is_half(&self) -> bool {
        matches!(self, Plane::Half(_))
    }
}

pub struct DitBlock {
    pub wq: Plane,
    pub wk: Plane,
    pub wv: Plane,
    pub wo: Plane,
    pub norm_q: DeviceTensor,
    pub norm_k: DeviceTensor,
    pub gate: Plane,
    pub up: Plane,
    pub down: Plane,
}

/// The modulation vectors for one timestep, ready for the block loop:
/// `1 + s` as layernorm weights, `tanh(g)` as gates, plus `1 + s_out` for
/// the final norm.
struct ModVecs {
    w1: CudaSlice<f32>,
    g1: CudaSlice<f32>,
    w2: CudaSlice<f32>,
    g2: CudaSlice<f32>,
    w_out: CudaSlice<f32>,
}

/// A prompt's prefix, encoded: its post-rope K/V per layer, and its length.
pub struct Prefix {
    pub len: usize,
    /// The rope cursor after the prefix - the target block's frame index.
    /// The row count for a text-only prefix; with condition images, text
    /// rows + `max(lw, lh)` per image, the model's own rule.
    pub frame: usize,
    k: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
}

/// One block of the prefix, in sequence order: the VL encoder's hidden
/// rows for a run of text tokens (through `txt_in`), or a condition
/// image's packed latents (through `img_in`), which fill the `<|image_pad|>`
/// slots of that run - four positions per VL token, raster order over the
/// latent grid - and attend to each other bidirectionally.
pub enum PrefixSegment<'a> {
    Text {
        hidden: &'a CudaSlice<f32>,
        /// first row of `hidden` this run starts at
        row0: usize,
        rows: usize,
    },
    Image {
        latents: &'a CudaSlice<f32>,
        lw: usize,
        lh: usize,
    },
}

impl PrefixSegment<'_> {
    fn rows(&self) -> usize {
        match self {
            PrefixSegment::Text { rows, .. } => *rows,
            PrefixSegment::Image { lw, lh, .. } => lw * lh,
        }
    }
}

/// The staged activation a GEMM reads: int8 (+ scales, sums, the stream-k
/// fixup) for the quantized lanes, f16 for the f16 lane.
struct Stage {
    xq: CudaSlice<i8>,
    xs: CudaSlice<f32>,
    yq: CudaSlice<u8>,
    x16: CudaSlice<f16>,
    skfix: CudaSlice<f32>,
    xsums: CudaSlice<f32>,
    ssums: CudaSlice<f32>,
}

/// Row scratch for a forward of up to `rows` tokens.
struct Scratch {
    rows: usize,
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    st: Stage,
}

/// What a target forward needs beyond the row scratch: the layout's rope
/// positions and the `[prefix | target]` K/V the attention reads.
struct Target {
    lw: usize,
    lh: usize,
    prefix_len: usize,
    /// the rope frame the stored positions were built for
    frame: usize,
    pos: CudaSlice<i32>,
    kv_k: CudaSlice<f32>,
    kv_v: CudaSlice<f32>,
}

/// Stage `x` `[rows][in_dim]` f32 for the block GEMMs of this plane class.
fn stage(
    exec: &GpuExecutor,
    st: &mut Stage,
    half: bool,
    x: &CudaSlice<f32>,
    in_dim: usize,
    rows: usize,
) -> Result<(), GpuModelError> {
    if half {
        exec.convert_f32_f16(x, &mut st.x16, rows * in_dim)?;
        Ok(())
    } else {
        prefill_quant(exec, &mut st.xq, &mut st.xs, &mut st.yq, x, in_dim, rows)
    }
}

/// `y = W x` off the staged activation.
fn mm(
    exec: &GpuExecutor,
    st: &mut Stage,
    w: &Plane,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    match w {
        Plane::Quant(q) => prefill_mm_pre_any(
            exec,
            q,
            &st.xq,
            &st.xs,
            &st.yq,
            &mut st.xsums,
            &mut st.ssums,
            &mut st.skfix,
            y,
            rows,
        ),
        Plane::Half(h) => Ok(exec.f16_gemm(&h.buf, &st.x16, y, h.dims[0], h.dims[1], rows, 0.0)?),
    }
}

/// `y = W x` from an unstaged f32 `x` (the attention output).
fn mm_from(
    exec: &GpuExecutor,
    st: &mut Stage,
    w: &Plane,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    match w {
        Plane::Quant(q) => prefill_mm_any(
            exec,
            q,
            &mut st.xq,
            &mut st.xs,
            &mut st.yq,
            &mut st.xsums,
            &mut st.ssums,
            &mut st.skfix,
            x,
            y,
            rows,
        ),
        Plane::Half(h) => {
            exec.convert_f32_f16(x, &mut st.x16, rows * h.dims[0])?;
            Ok(exec.f16_gemm(&h.buf, &st.x16, y, h.dims[0], h.dims[1], rows, 0.0)?)
        }
    }
}

/// `y = W_down (silu(gate) * up)`; `gate` is consumed.
#[allow(clippy::too_many_arguments)]
fn ffn_down(
    exec: &GpuExecutor,
    st: &mut Stage,
    w: &Plane,
    gate: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    ff: usize,
    rows: usize,
) -> Result<(), GpuModelError> {
    match w {
        Plane::Quant(q) => prefill_ffn_down_any(
            exec,
            q,
            &mut st.xq,
            &mut st.xs,
            &mut st.yq,
            &mut st.xsums,
            &mut st.ssums,
            &mut st.skfix,
            gate,
            up,
            y,
            ff,
            rows,
        ),
        Plane::Half(h) => {
            exec.swiglu(gate, up, rows * ff)?;
            exec.convert_f32_f16(gate, &mut st.x16, rows * ff)?;
            Ok(exec.f16_gemm(&h.buf, &st.x16, y, h.dims[0], h.dims[1], rows, 0.0)?)
        }
    }
}

/// One-row `y = W x` for the timestep planes.
fn gemv(
    exec: &GpuExecutor,
    w: &Plane,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    match w {
        Plane::Quant(q) => gemv_any(exec, q, x, y),
        Plane::Half(h) => {
            let mut x16 = exec.alloc_f16(h.dims[0])?;
            exec.convert_f32_f16(x, &mut x16, h.dims[0])?;
            Ok(exec.matvec_batch_f16(h, &x16, y, 1)?)
        }
    }
}

/// The modulation for one sigma: sinusoid -> linear_1 -> SiLU -> linear_2
/// -> SiLU -> (modulation.1 | norm_out.linear), then the `1 + s` / `tanh(g)`
/// forms the blocks consume. A free function so `new` can build the t = 0
/// row before the model exists.
#[allow(clippy::too_many_arguments)]
fn build_modvecs(
    exec: &GpuExecutor,
    d: usize,
    t_lin1: &Plane,
    t_lin2: &Plane,
    modulation: &Plane,
    norm_out_lin: &DeviceTensor,
    ones: &CudaSlice<f32>,
    sigma: f32,
) -> Result<ModVecs, GpuModelError> {
    let sinus = exec.to_device(&timestep_embedding(sigma, 256))?;
    let mut t1 = exec.alloc(d)?;
    gemv(exec, t_lin1, &sinus, &mut t1)?;
    exec.dit_silu(&mut t1, d)?;
    let mut temb = exec.alloc(d)?;
    gemv(exec, t_lin2, &t1, &mut temb)?;
    exec.dit_silu(&mut temb, d)?; // = silu(temb), what both heads read
    let mut m = exec.alloc(4 * d)?;
    gemv(exec, modulation, &temb, &mut m)?;
    let mut w_out = exec.alloc(d)?;
    exec.matvec_f32_batch(norm_out_lin, &temb, &mut w_out, 1)?;
    let mut w1 = exec.alloc(d)?;
    let mut g1 = exec.alloc(d)?;
    let mut w2 = exec.alloc(d)?;
    let mut g2 = exec.alloc(d)?;
    exec.copy_region(&m, 0, &mut w1, 0, d)?;
    exec.copy_region(&m, d, &mut g1, 0, d)?;
    exec.copy_region(&m, 2 * d, &mut w2, 0, d)?;
    exec.copy_region(&m, 3 * d, &mut g2, 0, d)?;
    for w in [&mut w1, &mut w2, &mut w_out] {
        exec.bias_add(w, ones, 1, d)?; // 1 + scale
    }
    for g in [&mut g1, &mut g2] {
        exec.softcap(g, d, 1.0)?; // tanh
    }
    Ok(ModVecs {
        w1,
        g1,
        w2,
        g2,
        w_out,
    })
}

pub struct DitModel {
    exec: Arc<GpuExecutor>,
    pub hp: Hparams,
    blocks: Vec<DitBlock>,
    /// every block plane is f16 (the parity file) rather than quantized
    half: bool,
    t_lin1: Plane,
    t_lin2: Plane,
    modulation: Plane,
    norm_out_lin: DeviceTensor,
    proj_out: DeviceTensor,
    img_in: QuantTensor,
    txt_in_layer: QuantTensor,
    txt_out_layer: QuantTensor,
    txt_norm: CudaSlice<f32>,
    pub weights_bytes: u64,
    zeros: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    sinks: CudaSlice<f32>,
    /// the t = 0 modulation every prefix token reads - a constant. Shared
    /// so a block walk can hold it while it borrows the model mutably.
    mod0: Arc<ModVecs>,
    scratch: Option<Scratch>,
    target: Option<Target>,
}

impl DitModel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        exec: Arc<GpuExecutor>,
        hp: Hparams,
        blocks: Vec<DitBlock>,
        t_lin1: Plane,
        t_lin2: Plane,
        modulation: Plane,
        norm_out_lin: DeviceTensor,
        proj_out: DeviceTensor,
        img_in: QuantTensor,
        txt_in_layer: QuantTensor,
        txt_out_layer: QuantTensor,
        txt_norm: CudaSlice<f32>,
        weights_bytes: u64,
    ) -> Result<Self, GpuModelError> {
        // one class per file: the staging is decided once per LN, not per plane
        let half = blocks[0].wq.is_half();
        if blocks.iter().any(|b| {
            [&b.wq, &b.wk, &b.wv, &b.wo, &b.gate, &b.up, &b.down]
                .iter()
                .any(|p| p.is_half() != half)
        }) {
            return Err(GpuModelError::Unsupported(
                "qwen-image: the block planes mix f16 and quantized tensors; a file is one or the other".into(),
            ));
        }
        let d = hp.hidden;
        let zeros = exec.alloc(d)?;
        let ones = exec.to_device(&vec![1.0f32; d])?;
        let sinks = exec.alloc_no_sinks(hp.n_heads)?;
        let mod0 = Arc::new(build_modvecs(
            &exec,
            d,
            &t_lin1,
            &t_lin2,
            &modulation,
            &norm_out_lin,
            &ones,
            0.0,
        )?);
        Ok(Self {
            exec,
            hp,
            blocks,
            half,
            t_lin1,
            t_lin2,
            modulation,
            norm_out_lin,
            proj_out,
            img_in,
            txt_in_layer,
            txt_out_layer,
            txt_norm,
            weights_bytes,
            zeros,
            ones,
            sinks,
            mod0,
            scratch: None,
            target: None,
        })
    }

    fn modulation_for(&self, sigma: f32) -> Result<ModVecs, GpuModelError> {
        build_modvecs(
            &self.exec,
            self.hp.hidden,
            &self.t_lin1,
            &self.t_lin2,
            &self.modulation,
            &self.norm_out_lin,
            &self.ones,
            sigma,
        )
    }

    fn ensure_scratch(&mut self, rows: usize) -> Result<(), GpuModelError> {
        if self.scratch.as_ref().is_some_and(|s| s.rows >= rows) {
            return Ok(());
        }
        self.scratch = None;
        let e = &self.exec;
        let (d, ff) = (self.hp.hidden, self.hp.ff);
        let wide = d.max(ff);
        self.scratch = Some(Scratch {
            rows,
            x: e.alloc(rows * d)?,
            xn: e.alloc(rows * d)?,
            q: e.alloc(rows * d)?,
            k: e.alloc(rows * d)?,
            v: e.alloc(rows * d)?,
            attn: e.alloc(rows * d)?,
            proj: e.alloc(rows * d)?,
            gate: e.alloc(rows * ff)?,
            up: e.alloc(rows * ff)?,
            st: Stage {
                xq: e.alloc_i8(rows * wide)?,
                xs: e.alloc(rows * wide / 32)?,
                yq: e.alloc_u8(wide.div_ceil(128) * rows.next_multiple_of(128) * 144)?,
                x16: e.alloc_f16(rows * wide)?,
                skfix: e.alloc(256 * 128 * 128 + 256)?,
                xsums: e.alloc(wide.div_ceil(128) * rows.next_multiple_of(128) * 4)?,
                ssums: e.alloc(rows * wide / 16)?,
            },
        });
        Ok(())
    }

    /// The block's attention half up to (and excluding) the attention
    /// itself: LN + modulate, q/k/v projections, q/k norms, rope. Leaves q/k/v
    /// in the scratch. `own_pos` is the rope positions to use; `None` means
    /// the prepared target layout's.
    fn block_qkv(
        &mut self,
        li: usize,
        m: &ModVecs,
        own_pos: Option<&CudaSlice<i32>>,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (d, h, hd, eps) = (
            self.hp.hidden,
            self.hp.n_heads,
            self.hp.head_dim,
            self.hp.eps,
        );
        let half = self.half;
        let blk = &self.blocks[li];
        let rope_pos = own_pos.unwrap_or_else(|| &self.target.as_ref().expect("target").pos);
        let sc = self.scratch.as_mut().expect("scratch");
        exec.layernorm(&sc.x, &m.w1, &self.zeros, &mut sc.xn, rows, d, eps)?;
        if li == 0 {
            super::dump(&exec, "L0 ln", &sc.xn, rows * d);
        }
        stage(&exec, &mut sc.st, half, &sc.xn, d, rows)?;
        if li == 0 {
            super::dump(&exec, "L0 xs", &sc.st.xs, rows * d / 32);
        }
        mm(&exec, &mut sc.st, &blk.wq, &mut sc.q, rows)?;
        if li == 0 {
            super::dump(&exec, "L0 q gemm", &sc.q, rows * d);
        }
        mm(&exec, &mut sc.st, &blk.wk, &mut sc.k, rows)?;
        mm(&exec, &mut sc.st, &blk.wv, &mut sc.v, rows)?;
        exec.rmsnorm_batch_inplace(&mut sc.q, &blk.norm_q.buf, hd, eps, rows * h)?;
        exec.rmsnorm_batch_inplace(&mut sc.k, &blk.norm_k.buf, hd, eps, rows * h)?;
        if li == 0 {
            super::dump(&exec, "L0 q norm", &sc.q, rows * d);
        }
        exec.dit_rope(&mut sc.q, rope_pos, rows, h, hd, AXES_DIMS_ROPE, ROPE_THETA)?;
        exec.dit_rope(&mut sc.k, rope_pos, rows, h, hd, AXES_DIMS_ROPE, ROPE_THETA)?;
        Ok(())
    }

    /// The block's tail after the attention output landed in `attn`: the
    /// output projection, the gated residual, then the MLP half.
    fn block_tail(&mut self, li: usize, m: &ModVecs, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (d, ff, eps) = (self.hp.hidden, self.hp.ff, self.hp.eps);
        let half = self.half;
        let blk = &self.blocks[li];
        let sc = self.scratch.as_mut().expect("scratch");
        mm_from(&exec, &mut sc.st, &blk.wo, &sc.attn, &mut sc.proj, rows)?;
        exec.dit_gated_add(&mut sc.x, &sc.proj, &m.g1, rows, d)?;
        exec.layernorm(&sc.x, &m.w2, &self.zeros, &mut sc.xn, rows, d, eps)?;
        stage(&exec, &mut sc.st, half, &sc.xn, d, rows)?;
        mm(&exec, &mut sc.st, &blk.gate, &mut sc.gate, rows)?;
        mm(&exec, &mut sc.st, &blk.up, &mut sc.up, rows)?;
        ffn_down(
            &exec,
            &mut sc.st,
            &blk.down,
            &mut sc.gate,
            &sc.up,
            &mut sc.proj,
            ff,
            rows,
        )?;
        exec.dit_gated_add(&mut sc.x, &sc.proj, &m.g2, rows, d)?;
        Ok(())
    }

    /// Encode a prompt's hidden states `[rows][context_dim]` into its prefix
    /// K/V. Text only for now: positions are `(i, i, i)`, attention is
    /// causal, modulation is the t = 0 row.
    /// Run the prefix - text rows and condition-image blocks in sequence
    /// order - through the 32 blocks once, modulated at t = 0, and keep
    /// every layer's post-rope K/V for the target steps. Attention is
    /// block-causal: a text row reads everything up to itself, an image row
    /// reads everything up to the END of its block (the block sees itself
    /// whole), and no prefix row ever sees the target. On the LLM lane's
    /// contiguous cache that is one per-row read bound, so it costs nothing
    /// a text-only prefix did not.
    pub fn encode_prefix(
        &mut self,
        segments: &[PrefixSegment<'_>],
    ) -> Result<Prefix, GpuModelError> {
        let rows: usize = segments.iter().map(PrefixSegment::rows).sum();
        if rows == 0 {
            return Err(GpuModelError::Unsupported("an empty prefix".into()));
        }
        self.ensure_scratch(rows)?;
        let exec = self.exec.clone();
        let (d, h, hd, eps) = (
            self.hp.hidden,
            self.hp.n_heads,
            self.hp.head_dim,
            self.hp.eps,
        );
        let n_layers = self.hp.n_layers;
        let scale = 1.0 / (hd as f32).sqrt();

        // The rows' inputs, rope positions and read bounds, segment by
        // segment. Text: txt_in (zero-centred RMSNorm -> in_layer ->
        // GELU(tanh) -> out_layer), every rope axis at the cursor, one
        // cursor step per row. Image: img_in, frame frozen at the cursor with
        // h / w centred on zero over the grid, then the cursor advances by
        // max(lw, lh) - the model's rule, the same one the target follows.
        let mut rope_pos: Vec<i32> = Vec::with_capacity(3 * rows);
        let mut read_pos: Vec<u32> = Vec::with_capacity(rows);
        let mut cursor = 0usize;
        let mut off = 0usize;
        for seg in segments {
            let n = seg.rows();
            match seg {
                PrefixSegment::Text {
                    hidden,
                    row0,
                    rows: n,
                } => {
                    let ctx = self.hp.context_dim;
                    let mut run = exec.alloc(n * ctx)?;
                    exec.copy_region(hidden, row0 * ctx, &mut run, 0, n * ctx)?;
                    let mut xn = exec.alloc(n * ctx)?;
                    let mut proj = exec.alloc(n * d)?;
                    let mut x = exec.alloc(n * d)?;
                    exec.rmsnorm_batch(&run, &self.txt_norm, &mut xn, ctx, eps, *n)?;
                    exec.bf16_gemm_tile(&self.txt_in_layer, None, &xn, &mut proj, d, *n)?;
                    exec.gelu(&mut proj, n * d)?;
                    exec.bf16_gemm_tile(&self.txt_out_layer, None, &proj, &mut x, d, *n)?;
                    let sc = self.scratch.as_mut().expect("scratch");
                    exec.copy_region(&x, 0, &mut sc.x, off * d, n * d)?;
                    for i in 0..*n {
                        let p = (cursor + i) as i32;
                        rope_pos.extend_from_slice(&[p, p, p]);
                        read_pos.push((off + i) as u32);
                    }
                    cursor += n;
                }
                PrefixSegment::Image { latents, lw, lh } => {
                    let mut x = exec.alloc(n * d)?;
                    exec.bf16_gemm_tile(&self.img_in, None, latents, &mut x, d, n)?;
                    let sc = self.scratch.as_mut().expect("scratch");
                    exec.copy_region(&x, 0, &mut sc.x, off * d, n * d)?;
                    rope_pos.extend(grid_positions(*lw, *lh, cursor));
                    let end = (off + n - 1) as u32;
                    read_pos.extend(std::iter::repeat_n(end, n));
                    cursor += lw.max(lh);
                }
            }
            off += n;
        }

        // the LLM lane's contiguous cache at slot 0, f16 keys, written at
        // the row index and read to each row's own bound
        let ctx = rows.next_multiple_of(64);
        let mut cache_k = exec.alloc_u8(ctx * d * 2)?;
        let mut cache_v = exec.alloc_u8(ctx * d * 2)?;
        let positions: Vec<u32> = (0..rows as u32).collect();
        let d_pos = exec.to_device_u32(&positions)?;
        let d_read = exec.to_device_u32(&read_pos)?;
        let d_slots = exec.alloc_u32(rows)?; // zeros -> slot 0
        let d_rope = exec.to_device_i32(&rope_pos)?;

        let m0 = self.mod0.clone();
        let mut pk = Vec::with_capacity(n_layers);
        let mut pv = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            self.block_qkv(li, &m0, Some(&d_rope), rows)?;
            let sc = self.scratch.as_mut().expect("scratch");
            exec.kv_append_batch(
                &sc.k,
                &mut cache_k,
                &d_pos,
                Some(&d_slots),
                d,
                ctx,
                rows,
                KvDtype::Fp16,
            )?;
            exec.kv_append_batch(
                &sc.v,
                &mut cache_v,
                &d_pos,
                Some(&d_slots),
                d,
                ctx,
                rows,
                KvDtype::Fp16,
            )?;
            exec.attn_decode_batch(
                &sc.q,
                &cache_k,
                &cache_v,
                &self.sinks,
                &mut sc.attn,
                &d_read,
                Some(&d_slots),
                h,
                h,
                hd,
                ctx,
                d,
                0,
                rows,
                scale,
                KvDtype::Fp16,
            )?;
            // keep this layer's post-rope K/V for every later step
            let mut lk = exec.alloc(rows * d)?;
            let mut lv = exec.alloc(rows * d)?;
            exec.copy_region(&sc.k, 0, &mut lk, 0, rows * d)?;
            exec.copy_region(&sc.v, 0, &mut lv, 0, rows * d)?;
            pk.push(lk);
            pv.push(lv);
            self.block_tail(li, &m0, rows)?;
        }
        Ok(Prefix {
            len: rows,
            frame: cursor,
            k: pk,
            v: pv,
        })
    }

    /// Size the target-pass state for an `lw x lh` latent grid behind a
    /// prefix of up to `prefix_len` tokens whose rope cursor ends at
    /// `frame`: the target's rope positions (frame frozen there, h/w
    /// centred on zero) and the `[prefix | target]` K/V planes one layer's
    /// attention reads.
    pub fn prepare_target(
        &mut self,
        lw: usize,
        lh: usize,
        prefix_len: usize,
        frame: usize,
    ) -> Result<(), GpuModelError> {
        let n = lw * lh;
        self.ensure_scratch(n)?;
        if self.target.as_ref().is_some_and(|t| {
            t.lw == lw && t.lh == lh && t.prefix_len >= prefix_len && t.frame == frame
        }) {
            return Ok(());
        }
        let d = self.hp.hidden;
        let total = prefix_len + n;
        self.target = Some(Target {
            lw,
            lh,
            prefix_len,
            frame,
            pos: self.exec.to_device_i32(&target_positions(lw, lh, frame))?,
            kv_k: self.exec.alloc(total * d)?,
            kv_v: self.exec.alloc(total * d)?,
        });
        Ok(())
    }

    /// One denoising step: the target block's velocity at `sigma` given its
    /// packed latents `[n][in_channels]`, into `out` `[n][out_channels]`.
    pub fn step(
        &mut self,
        prefix: &Prefix,
        latents: &CudaSlice<f32>,
        sigma: f32,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (d, h, hd, eps) = (
            self.hp.hidden,
            self.hp.n_heads,
            self.hp.head_dim,
            self.hp.eps,
        );
        let n_layers = self.hp.n_layers;
        let scale = 1.0 / (hd as f32).sqrt();
        let (lw, lh, prepared, frame) = {
            let tg = self.target.as_ref().expect("prepare_target first");
            (tg.lw, tg.lh, tg.prefix_len, tg.frame)
        };
        let n = lw * lh;
        let l = prefix.len;
        if l > prepared {
            return Err(GpuModelError::Unsupported(format!(
                "prefix of {l} tokens exceeds the prepared {prepared}"
            )));
        }
        let m = self.modulation_for(sigma)?;
        // the target's frame is where this prefix's rope cursor ended; a
        // prefix that ends elsewhere than the layout was prepared for
        // (guidance's negative, a different prompt length) gets its own
        // positions
        let own_pos = (prefix.frame != frame)
            .then(|| exec.to_device_i32(&target_positions(lw, lh, prefix.frame)))
            .transpose()?;

        {
            let sc = self.scratch.as_mut().expect("scratch");
            exec.bf16_gemm_tile(&self.img_in, None, latents, &mut sc.x, d, n)?;
            super::dump(&exec, "img_in", &sc.x, n * d);
        }
        for li in 0..n_layers {
            self.block_qkv(li, &m, own_pos.as_ref(), n)?;
            let sc = self.scratch.as_mut().expect("scratch");
            let tg = self.target.as_mut().expect("target");
            if li == 0 || li + 1 == n_layers {
                super::dump(&exec, &format!("L{li} q"), &sc.q, n * d);
                super::dump(&exec, &format!("L{li} k"), &sc.k, n * d);
            }
            // [prefix | target] keys and values for this layer
            exec.copy_region(&prefix.k[li], 0, &mut tg.kv_k, 0, l * d)?;
            exec.copy_region(&prefix.v[li], 0, &mut tg.kv_v, 0, l * d)?;
            exec.copy_region(&sc.k, 0, &mut tg.kv_k, l * d, n * d)?;
            exec.copy_region(&sc.v, 0, &mut tg.kv_v, l * d, n * d)?;
            exec.vision_attn_xq_at(
                &sc.q,
                &tg.kv_k,
                &tg.kv_v,
                &mut sc.attn,
                0,
                n,
                l + n,
                h,
                hd,
                scale,
            )?;
            if li == 0 || li + 1 == n_layers {
                super::dump(&exec, &format!("L{li} attn"), &sc.attn, n * d);
            }
            self.block_tail(li, &m, n)?;
            if li == 0 || li + 1 == n_layers {
                let sc = self.scratch.as_ref().expect("scratch");
                super::dump(&exec, &format!("L{li} x"), &sc.x, n * d);
            }
        }
        // norm_out (scale only) then proj_out
        let sc = self.scratch.as_mut().expect("scratch");
        exec.layernorm(&sc.x, &m.w_out, &self.zeros, &mut sc.xn, n, d, eps)?;
        exec.matvec_f32_batch(&self.proj_out, &sc.xn, out, n)?;
        super::dump(&exec, "velocity", out, n * self.hp.out_channels);
        Ok(())
    }
}

/// An image block's rope positions: frame frozen at `frame` (the cursor
/// where the block starts), height and width centred on zero, row-major
/// over the grid. The target block and every condition image alike.
fn grid_positions(lw: usize, lh: usize, frame: usize) -> Vec<i32> {
    let mut pos = Vec::with_capacity(3 * lw * lh);
    let (h0, w0) = (-((lh - lh / 2) as i32), -((lw - lw / 2) as i32));
    for y in 0..lh as i32 {
        for x in 0..lw as i32 {
            pos.extend_from_slice(&[frame as i32, h0 + y, w0 + x]);
        }
    }
    pos
}

/// The target block's rope positions: the grid at the prefix's end.
fn target_positions(lw: usize, lh: usize, frame: usize) -> Vec<i32> {
    grid_positions(lw, lh, frame)
}
