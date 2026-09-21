//! The forward pass, for a batch of chips at once.
//!
//!   u8 chips [B, S, S, bands]
//!   -> patch stem: normalize + im2row (GPU) -> GEMM -> + {patch bias | class
//!      token | register tokens}                            [B*T, d]   T = G*G + 1 + R
//!   -> N x pre-LN LayerScale block, every plane between the GEMMs at f16:
//!        LN -> fused qkv GEMM -> split + q/v bias + table rope (patch rows)
//!        -> full bidirectional attention within each chip -> o GEMM
//!        -> x += ls1*(o + b); LN -> up GEMM (bias + GELU in its landing) -> down
//!        GEMM -> x += ls2*(down + b); next LN
//!      after block d in stage_depths: tap = project_i(x)   [B*T, w_i]
//!   -> decoder, deepest tap first:
//!        x0 = blend_0(tap_0's patch grid)                  [B, G, G, w_0]
//!        x_i = blend_i(convT_i(x_{i-1}) + bilinear(tap_i's patch grid))
//!   -> stacked 1x1 heads -> argmax u8 + height f32         [B, 8G, 8G]
//!
//! Every op is row-batched, so B scales the pass as pure row counts and there
//! is one launch train whatever B is. Nothing here allocates and nothing reads
//! the host until the outputs are copied back.
//!
//! The taps are the block outputs, un-normed - transformers' encoder captures
//! hidden states with `tie_last_hidden_states=False`, so depth N is the raw
//! output of block N and `backbone.norm` never touches what the decoder sees.
//! That tensor ships in the checkpoint and is deliberately not loaded.

use cudarc::driver::CudaSlice;
use half::f16;

use super::load::GN_EPS;
use super::{GpuDinov3Seg, GpuModelError, SegOutput};
use crate::gpu::{GpuExecutor, HalfTensor};

/// One backbone GEMM onto the half interface. `land32` is `None` wherever the
/// f16-landing GEMM is the device's elected route - every validated part - and
/// the f32 plane a non-elected device lands on first otherwise.
fn gemm_h(
    exec: &GpuExecutor,
    w: &HalfTensor,
    x16: &CudaSlice<f16>,
    y16: &mut CudaSlice<f16>,
    rows: usize,
    land32: Option<&mut CudaSlice<f32>>,
) -> Result<(), GpuModelError> {
    match land32 {
        None => exec.matvec_batch_f16_h(w, x16, y16, rows)?,
        Some(y32) => {
            exec.matvec_batch_f16(w, x16, y32, rows)?;
            exec.convert_f32_f16(y32, y16, rows * w.dims[1])?;
        }
    }
    Ok(())
}

impl GpuDinov3Seg {
    /// Segment `chips` chips. `pixels` is `chips * side * side * bands` bytes,
    /// chip-major HWC in the config's band order. `want_logits` also returns
    /// the biased class logits at f16 - the parity gate's view, 20x the bytes
    /// of the class raster, not something a production sweep should ask for.
    pub fn segment(
        &mut self,
        pixels: &[u8],
        chips: usize,
        want_logits: bool,
    ) -> Result<SegOutput, GpuModelError> {
        let side = self.cfg.out_size;
        let ncls = self.cfg.n_classes;
        if chips == 0 {
            return Ok(SegOutput {
                chips: 0,
                size: side,
                classes: Vec::new(),
                height: Vec::new(),
                logits: want_logits.then(Vec::new),
            });
        }
        if chips > self.ws.cap {
            return Err(GpuModelError::BatchTooLarge {
                got: chips,
                max: self.ws.cap,
            });
        }
        if pixels.len() != chips * self.chip_bytes() {
            return Err(GpuModelError::Unsupported(format!(
                "dinov3: {} bytes for {chips} chip(s), expected {} ({}x{}x{} u8 each)",
                pixels.len(),
                chips * self.chip_bytes(),
                self.cfg.image_size,
                self.cfg.image_size,
                self.cfg.channels
            )));
        }
        let out_px = side * side;
        if want_logits && self.ws.logits.is_none() {
            self.ws.logits = Some(self.exec.alloc_f16(self.ws.cap * out_px * ncls)?);
        }

        self.exec.upload_u8(pixels, &mut self.ws.px)?;
        self.backbone(chips)?;
        self.decoder(chips, want_logits)?;

        let n = chips * out_px;
        let classes = self.exec.to_host_u8_len(&self.ws.cls, n)?;
        let height = self.exec.to_host_len(&self.ws.height, n)?;
        let logits: Option<Vec<f16>> = match (&self.ws.logits, want_logits) {
            (Some(l), true) => Some(self.exec.to_host_f16_len(l, n * ncls)?),
            _ => None,
        };
        // A NaN anywhere upstream reaches both heads (they are linear in the
        // same plane), so the regression raster is the canary for the chip.
        // Height is unbounded by design; non-FINITE is the only refusal.
        if let Some(bad) = height.iter().position(|h| !h.is_finite()) {
            return Err(GpuModelError::Unsupported(format!(
                "dinov3: non-finite output in chip {} - an activation left f16's range; \
                 this chip cannot be served in the f16 class",
                bad / out_px
            )));
        }
        Ok(SegOutput {
            chips,
            size: side,
            classes,
            height,
            logits,
        })
    }

    fn backbone(&mut self, chips: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let cfg = &self.cfg;
        let (d, ffn, hd, heads) = (cfg.hidden, cfg.intermediate, cfg.head_dim(), cfg.n_heads);
        let t = cfg.tokens();
        let n_rope = cfg.grid() * cfg.grid();
        let rows = chips * t;
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = cfg.eps;
        let ws = &mut self.ws;

        // ---- stem ----
        exec.dp_u8_patch_rows(
            &ws.px,
            &mut ws.s16,
            &cfg.mean,
            &cfg.std,
            chips,
            cfg.image_size,
            cfg.patch,
            t,
        )?;
        exec.matvec_batch_f16(&self.patch_w, &ws.s16, &mut ws.x, rows)?;
        exec.add_rows_bcast(&mut ws.x, &self.embed_add, rows, t, d)?;

        let first = &self.blocks[0].ln1;
        exec.whisper_ln_f16(&ws.x, &first.w, &first.b, &mut ws.n16, rows, d, eps)?;

        let n_layer = self.blocks.len();
        for li in 0..n_layer {
            let blk = &self.blocks[li];
            gemm_h(
                &exec,
                &blk.wqkv,
                &ws.n16,
                &mut ws.qkv,
                rows,
                ws.land32.as_mut(),
            )?;
            // 1/sqrt(hd) rides the split: q is scaled before its one round to
            // f16, which is the round the attention kernel would have done
            exec.dp_qkv_split_rope_h(
                &ws.qkv,
                &blk.bq,
                &blk.bv,
                &self.rope_cos,
                &self.rope_sin,
                &mut ws.q,
                &mut ws.k,
                &mut ws.v,
                d,
                hd,
                rows,
                t,
                n_rope,
                scale,
            )?;
            // grid.z strides q/k/v by t*heads*hd - the chip-major layout the
            // split just wrote - so each chip attends only to itself. It lands
            // straight on the o projection's input: no f32 plane, no convert.
            exec.vision_attn_h(&ws.q, &ws.k, &ws.v, &mut ws.s16, t, t, heads, hd, chips)?;
            gemm_h(
                &exec,
                &blk.wo,
                &ws.s16,
                &mut ws.proj,
                rows,
                ws.land32.as_mut(),
            )?;
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.bo,
                &blk.ls1,
                &blk.ln2.w,
                &blk.ln2.b,
                &mut ws.n16,
                rows,
                d,
                eps,
            )?;

            if self.fuse_gelu {
                // bias + GELU ride the landing: one round, no pass over ff
                exec.matvec_batch_f16_h_gelu(&blk.up, &ws.n16, &mut ws.ff, &blk.up_b, rows)?;
            } else {
                gemm_h(
                    &exec,
                    &blk.up,
                    &ws.n16,
                    &mut ws.ff,
                    rows,
                    ws.land32.as_mut(),
                )?;
                exec.dp_gelu_bias_h(&mut ws.ff, &blk.up_b, rows, ffn)?;
            }
            gemm_h(
                &exec,
                &blk.down,
                &ws.ff,
                &mut ws.proj,
                rows,
                ws.land32.as_mut(),
            )?;
            // the seam also lands the next block's pre-norm. After the last
            // block there is no next norm the decoder wants (see the module
            // note), so the final seam reuses the block's own ln2 purely to
            // satisfy the kernel - n16 is dead from there on.
            let next = if li + 1 < n_layer {
                &self.blocks[li + 1].ln1
            } else {
                &blk.ln2
            };
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.down_b,
                &blk.ls2,
                &next.w,
                &next.b,
                &mut ws.n16,
                rows,
                d,
                eps,
            )?;

            // ---- tap: project this depth's hidden state for the decoder ----
            // Projected now so the four full hidden states are never kept -
            // only their narrow projections are. s16 is free here; n16 is not.
            if let Some(si) = cfg.stage_depths.iter().position(|&dep| dep == li + 1) {
                exec.convert_f32_f16(&ws.x, &mut ws.s16, rows * d)?;
                exec.matvec_batch_f16(&self.stages[si].project, &ws.s16, &mut ws.taps[si], rows)?;
            }
        }
        Ok(())
    }

    fn decoder(&mut self, chips: usize, want_logits: bool) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let cfg = &self.cfg;
        let t = cfg.tokens();
        let grid = cfg.grid();
        let ws = &mut self.ws;

        let mut side = grid;
        for (si, st) in self.stages.iter().enumerate() {
            let c = st.width;
            if si > 0 {
                // x_{i-1} sits in h16 at f16, which is what the GEMM eats
                let (up, up_b) = (
                    st.up.as_ref().expect("stage > 0 has a transposed conv"),
                    st.up_b.as_ref().expect("stage > 0 has its bias"),
                );
                exec.matvec_batch_f16(up, &ws.h16, &mut ws.g, chips * side * side)?;
                exec.dp_convt2_skip(
                    &ws.g,
                    up_b,
                    &ws.taps[si],
                    &mut ws.xa,
                    chips,
                    side,
                    side,
                    c,
                    grid,
                    t,
                )?;
                side *= 2;
            }
            let px = side * side;
            // conv a: its input is f32 - the tap itself at stage 0 (read in
            // place out of the token plane, t rows per chip), the seam's
            // landing after that
            if si == 0 {
                let pb = st
                    .project_b
                    .as_ref()
                    .expect("stage 0 keeps its project bias");
                exec.bias_add(&mut ws.taps[0], pb, chips * t, c)?;
                exec.dp_im2row3_f32(&ws.taps[0], &mut ws.s16, chips, side, side, c, t)?;
            } else {
                exec.dp_im2row3_f32(&ws.xa, &mut ws.s16, chips, side, side, c, px)?;
            }
            let a = &st.blend[0];
            exec.matvec_batch_f16(&a.w, &ws.s16, &mut ws.ya, chips * px)?;
            exec.dp_group_norm_gelu_f16(
                &ws.ya,
                &a.b,
                &a.gn.w,
                &a.gn.b,
                &mut ws.h16,
                &mut ws.gn_part,
                &mut ws.gn_stat,
                chips,
                px,
                c,
                a.groups,
                GN_EPS,
            )?;
            // conv b: off the f16 plane the norm just wrote - a pure gather
            let b = &st.blend[1];
            exec.dp_im2row3_f16(&ws.h16, &mut ws.s16, chips, side, side, c, px)?;
            exec.matvec_batch_f16(&b.w, &ws.s16, &mut ws.ya, chips * px)?;
            exec.dp_group_norm_gelu_f16(
                &ws.ya,
                &b.b,
                &b.gn.w,
                &b.gn.b,
                &mut ws.h16,
                &mut ws.gn_part,
                &mut ws.gn_stat,
                chips,
                px,
                c,
                b.groups,
                GN_EPS,
            )?;
        }

        // ---- heads: one GEMM for both, one pass for bias + argmax + split ----
        let rows = chips * side * side;
        exec.matvec_batch_f16(&self.out_w, &ws.h16, &mut ws.o, rows)?;
        let logits = if want_logits {
            ws.logits.as_mut()
        } else {
            None
        };
        exec.dp_seg_heads(
            &ws.o,
            &self.out_b,
            &mut ws.cls,
            &mut ws.height,
            logits,
            rows,
            cfg.n_classes,
        )?;
        Ok(())
    }
}
