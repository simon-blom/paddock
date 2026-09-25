//! Laya - a decision model: ModernBERT (large for the English and
//! typed-decisions checkpoints, mmBERT-base for the multilingual one) under a
//! trained head that scores every option of a typed question at its own
//! `[MASK]`. Nothing is generated. A question is one bidirectional sequence,
//! `[CLS] <type> question: ... [SEP] [MASK] option0 [MASK] option1 ... [SEP]
//! state [SEP]`, and its answer is the softmax over its markers' logits (the
//! runner applies the checkpoint's fitted temperature and shapes the Jev
//! answer).
//!
//! The serving unit is a PASS of packed sequences: every question of every
//! request the queue held, back to back with no padding (the reference's
//! battery runs 15..512 tokens in one request, so a [batch, max_len] layout
//! would be mostly pad). Every op is row-batched over the packed rows; the
//! attention is the one op that knows where a sequence ends
//! (`gpu::enc_attn_h`, a per-sequence tile list).
//!
//!   ids -> embed + norm -> x (f32 residual)
//!   28 x { qkv GEMM -> attention (rope: 160k on global layers, 10k on the
//!          local ones with |i - j| <= 64) -> o GEMM -> x += o; mlp_norm
//!          -> Wi GEMM landing gelu(in) * gate -> Wo GEMM -> x += y; next norm }
//!   head entry: X = final_norm(x) + type_emb[question type]
//!   2 x post-norm-free torch TransformerEncoderLayer (norm_first, ReLU FFN
//!        of 4d, full attention with the in_proj bias); the LAST one runs its
//!        attention over every row and the rest over only the rows anything
//!        reads - the option markers and the [CLS] rows - which is most of
//!        that layer's FFN saved
//!   scorer: LN -> Linear(d, d) + GELU -> Linear(d, 1) at each marker
//!   act head: [X at CLS | 4 features of the option distribution] -> 256 ->
//!        GELU -> n_act, softmax
//!
//! Reference: the model's own package (github.com/NandhaKishorM/laya,
//! Apache-2.0) over transformers' ModernBERT, in fp32 on the GPU - there is
//! no llama.cpp for a model like this, so its outputs are the parity oracle
//! (a reference run records them; `tests/gpu_laya_golden.rs` gates on them). Its
//! own serving default is bf16 autocast, which moves logits by up to 0.076 on
//! the battery; fp16 autocast by 0.015.
//!
//! Precision class: f16 weights (the checkpoint's own dtype - nothing is
//! converted), f16 GEMM operands and landings, f32 accumulate, f32 residual
//! stream and every norm in f32 - fp16 autocast's class, with each fused
//! epilogue rounding once where autocast rounds twice.
//!
//! Batch invariance: an answer does not depend on what else shared its pass.
//! The f16-landing GEMMs never split K (one accumulation order whatever the
//! row count), the attention's work per (sequence, query tile, head) is laid
//! from the sequence start, and everything else is row- or question-local.
//! Gated in the golden test (one request alone vs packed with the battery).

mod forward;
mod load;

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::laya::LayaConfig;

use crate::gpu::{GpuExecutor, HalfTensor};

pub use crate::gpu_model::gpt_oss::GpuModelError;

/// The act head's hidden width (`nn.Linear(d + 4, 256)` in the reference).
pub const ACT_HIDDEN: usize = 256;

/// One ModernBERT layer.
struct EncLayer {
    /// None on layer 0 (its attn_norm is Identity: the embedding norm's
    /// output goes straight into the qkv GEMM)
    attn_norm: Option<CudaSlice<f32>>,
    wqkv: HalfTensor,
    wo: HalfTensor,
    mlp_norm: CudaSlice<f32>,
    /// `[d, 2F]`, rows re-laid in 16-row blocks for the GEGLU landing
    wi: HalfTensor,
    wo2: HalfTensor,
    global: bool,
}

/// One torch `TransformerEncoderLayer(norm_first=True)` of the head.
struct HeadLayer {
    n1w: CudaSlice<f32>,
    n1b: CudaSlice<f32>,
    in_w: HalfTensor,
    in_b: CudaSlice<f32>,
    out_w: HalfTensor,
    out_b: CudaSlice<f32>,
    n2w: CudaSlice<f32>,
    n2b: CudaSlice<f32>,
    l1w: HalfTensor,
    l1b: CudaSlice<f32>,
    l2w: HalfTensor,
    l2b: CudaSlice<f32>,
}

/// One loaded checkpoint.
pub struct GpuLaya {
    exec: Arc<GpuExecutor>,
    cfg: LayaConfig,
    emb: CudaSlice<f16>,
    emb_norm: CudaSlice<f32>,
    layers: Vec<EncLayer>,
    final_norm: CudaSlice<f32>,
    /// cos/sin `[max_len][hd/2]` for the global and the local rope bases
    rope_g: (CudaSlice<f32>, CudaSlice<f32>),
    rope_l: (CudaSlice<f32>, CudaSlice<f32>),
    temb: CudaSlice<f32>,
    head: Vec<HeadLayer>,
    s0w: CudaSlice<f32>,
    s0b: CudaSlice<f32>,
    s1w: HalfTensor,
    s1b: CudaSlice<f32>,
    s3w: CudaSlice<f32>,
    s3b: f32,
    a0w: CudaSlice<f16>,
    a0b: CudaSlice<f32>,
    a2w: CudaSlice<f16>,
    a2b: CudaSlice<f32>,
    /// the residual seam (slot 622) takes a bias and a scale per element;
    /// the encoder's residual has neither
    zeros: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    weight_bytes: u64,
}

impl GpuLaya {
    pub fn config(&self) -> &LayaConfig {
        &self.cfg
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn executor(&self) -> &Arc<GpuExecutor> {
        &self.exec
    }
    /// The workspace dims this checkpoint needs: (d, widest f16 plane a row).
    pub fn plane_dims(&self) -> (usize, usize) {
        let d = self.cfg.encoder.hidden;
        (d, (3 * d).max(self.cfg.encoder.intermediate).max(4 * d))
    }
}

/// One question's sequence as the runner built it.
pub struct LayaSeq<'a> {
    pub ids: &'a [u32],
    /// each option's `[MASK]` position inside the sequence
    pub markers: &'a [u32],
    /// 0 choice, 1 score, 2 noul (`paddock_models::laya::QTYPE_*`)
    pub qtype: u32,
}

/// What one pass returns, sequence-major.
#[derive(Debug, Clone)]
pub struct LayaOut {
    /// every sequence's option logits, back to back (raw - no temperature)
    pub logits: Vec<f32>,
    /// `offsets[s]..offsets[s + 1]` are sequence s's logits
    pub offsets: Vec<usize>,
    /// `[seqs][n_act]` act-head probabilities
    pub act: Vec<f32>,
    pub n_act: usize,
    /// tokens through the encoder
    pub rows: usize,
}

/// Every buffer a pass touches, sized once. Shared by every checkpoint on the
/// engine thread (a pass runs one checkpoint), so it is sized for the widest.
pub struct LayaWorkspace {
    rows_cap: usize,
    seq_cap: usize,
    gather_cap: usize,
    tile_cap: usize,
    d: usize,
    wide: usize,
    ids: CudaSlice<u32>,
    rtype: CudaSlice<u32>,
    cu: CudaSlice<u32>,
    tiles: CudaSlice<u32>,
    gidx: CudaSlice<u32>,
    qoff: CudaSlice<u32>,
    x: CudaSlice<f32>,
    n16: CudaSlice<f16>,
    wide16: CudaSlice<f16>,
    att: CudaSlice<f16>,
    proj: CudaSlice<f16>,
    xg: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    act: CudaSlice<f32>,
    bytes: u64,
}

impl LayaWorkspace {
    /// Most act outputs a question carries (the kernel's cap).
    const MAX_ACT: usize = 8;

    /// Bytes a workspace of this shape holds - the admission gate's input.
    pub fn bytes_for(d: usize, wide: usize, rows_cap: usize, seq_cap: usize) -> u64 {
        let gather_cap = rows_cap / 2 + seq_cap;
        let tile_cap = rows_cap / crate::gpu::ENC_ATTN_QTILE + seq_cap;
        let u32s = 2 * rows_cap + 2 * (seq_cap + 1) + tile_cap + gather_cap;
        let f32s = rows_cap * d + gather_cap * d + gather_cap + seq_cap * Self::MAX_ACT;
        let f16s = rows_cap * (3 * d + wide);
        (u32s * 4 + f32s * 4 + f16s * 2) as u64
    }

    /// `rows_cap` tokens a pass, `seq_cap` sequences (questions) a pass,
    /// planes `d` wide with an f16 scratch of `wide` a row.
    pub fn new(
        exec: &GpuExecutor,
        d: usize,
        wide: usize,
        rows_cap: usize,
        seq_cap: usize,
    ) -> Result<Self, GpuModelError> {
        let gather_cap = rows_cap / 2 + seq_cap;
        let tile_cap = rows_cap / crate::gpu::ENC_ATTN_QTILE + seq_cap;
        Ok(Self {
            rows_cap,
            seq_cap,
            gather_cap,
            tile_cap,
            d,
            wide,
            ids: exec.alloc_u32(rows_cap)?,
            rtype: exec.alloc_u32(rows_cap)?,
            cu: exec.alloc_u32(seq_cap + 1)?,
            tiles: exec.alloc_u32(tile_cap)?,
            gidx: exec.alloc_u32(gather_cap)?,
            qoff: exec.alloc_u32(seq_cap + 1)?,
            x: exec.alloc(rows_cap * d)?,
            n16: exec.alloc_f16(rows_cap * d)?,
            wide16: exec.alloc_f16(rows_cap * wide)?,
            att: exec.alloc_f16(rows_cap * d)?,
            proj: exec.alloc_f16(rows_cap * d)?,
            xg: exec.alloc(gather_cap * d)?,
            logits: exec.alloc(gather_cap)?,
            act: exec.alloc(seq_cap * Self::MAX_ACT)?,
            bytes: Self::bytes_for(d, wide, rows_cap, seq_cap),
        })
    }

    pub fn rows_cap(&self) -> usize {
        self.rows_cap
    }
    pub fn seq_cap(&self) -> usize {
        self.seq_cap
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Rope as transformers builds it for ModernBERT, in f32: `inv_freq[j] =
/// 1 / base^(2j / dim)`, angle = `inv_freq[j] * pos` (one f32 product - the
/// reference's K=1 matmul), cos / sin of that. rotate_half pairs `(j, j +
/// dim/2)` with the same angle, so the table is `[positions][dim/2]`.
pub(crate) fn rope_table(positions: usize, dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / base.powf((2 * j) as f32 / dim as f32))
        .collect();
    let mut cos = vec![0f32; positions * half];
    let mut sin = vec![0f32; positions * half];
    for p in 0..positions {
        for j in 0..half {
            let a = inv[j] * p as f32;
            cos[p * half + j] = a.cos();
            sin[p * half + j] = a.sin();
        }
    }
    (cos, sin)
}

/// The GEGLU landing's weight order: `Wi` `[2F, d]` (rows 0..F the GLU
/// input, F..2F its gate) re-laid so each 16-row block holds 8 input rows
/// and then their 8 gate rows - out-rows g and g + 8 of a block meet in one
/// lane of the GEMM's C fragment. Row-order only; every value is untouched.
pub(crate) fn geglu_relay<T: Copy>(wi: &[T], f: usize, d: usize) -> Vec<T> {
    debug_assert_eq!(wi.len(), 2 * f * d);
    debug_assert!(f.is_multiple_of(8));
    let mut out = Vec::with_capacity(wi.len());
    for b in 0..f / 8 {
        for j in 0..8 {
            let r = 8 * b + j;
            out.extend_from_slice(&wi[r * d..(r + 1) * d]);
        }
        for j in 0..8 {
            let r = f + 8 * b + j;
            out.extend_from_slice(&wi[r * d..(r + 1) * d]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geglu_relay_puts_each_gate_eight_rows_after_its_input() {
        let (f, d) = (16usize, 3usize);
        let wi: Vec<u32> = (0..2 * f * d).map(|i| i as u32).collect();
        let r = geglu_relay(&wi, f, d);
        let row = |v: &[u32], i: usize| v[i * d..(i + 1) * d].to_vec();
        for blk in 0..f / 8 {
            for g in 0..8 {
                // out-row 16b + g is input feature 8b + g, 16b + 8 + g its gate
                assert_eq!(row(&r, 16 * blk + g), row(&wi, 8 * blk + g));
                assert_eq!(row(&r, 16 * blk + 8 + g), row(&wi, f + 8 * blk + g));
            }
        }
    }

    #[test]
    fn rope_table_matches_the_reference_frequencies() {
        let (c, s) = rope_table(4, 64, 10000.0);
        // position 0 is the identity rotation
        assert!(c[..32].iter().all(|&x| x == 1.0) && s[..32].iter().all(|&x| x == 0.0));
        // dim 0 turns one radian a position
        assert!((c[32] - 1f32.cos()).abs() < 1e-7 && (s[32] - 1f32.sin()).abs() < 1e-7);
        // the last pair turns 1 / 10000^(62/64) a position
        let w = 1.0f32 / 10000f32.powf(62.0 / 64.0);
        assert!((s[3 * 32 + 31] - (3.0 * w).sin()).abs() < 1e-7);
    }
}
