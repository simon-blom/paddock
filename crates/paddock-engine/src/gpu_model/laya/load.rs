//! Checkpoint -> device planes. `model.safetensors` is f16 throughout (plus
//! the unused `temperature` buffer, f32): every GEMM plane goes up as it
//! stands - `nn.Linear`'s `[out, in]` IS the GEMM layout - except `Wi`, whose
//! rows are re-laid for the GEGLU landing. Norm weights, biases, the type
//! embedding and the scorer's last row are widened to f32 exactly (f16 ->
//! f32 is exact); the embedding table stays f16 and is gathered from.

use std::sync::Arc;

use half::f16;
use paddock_models::laya::LayaConfig;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::{ACT_HIDDEN, EncLayer, GpuLaya, GpuModelError, HeadLayer, geglu_relay, rope_table};
use crate::gpu::{GpuExecutor, HalfTensor};

struct Reader<'a> {
    st: &'a ShardedSafetensors,
    exec: &'a GpuExecutor,
    bytes: u64,
}

impl Reader<'_> {
    /// Raw f16 values of a tensor whose shape must match exactly.
    fn halves(&self, name: &str, shape: &[usize]) -> Result<Vec<f16>, GpuModelError> {
        let (t, b) = self
            .st
            .bytes(name)
            .ok_or_else(|| GpuModelError::MissingMeta(format!("laya tensor {name}")))?;
        if t.dtype != StDtype::F16 || t.shape != shape {
            return Err(GpuModelError::Unsupported(format!(
                "laya {name}: {:?} {:?} (want F16 {shape:?})",
                t.dtype, t.shape
            )));
        }
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
            .collect())
    }

    fn f32s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, GpuModelError> {
        Ok(self
            .halves(name, shape)?
            .iter()
            .map(|h| h.to_f32())
            .collect())
    }

    fn dev32(&mut self, host: &[f32]) -> Result<cudarc::driver::CudaSlice<f32>, GpuModelError> {
        self.bytes += (host.len() * 4) as u64;
        Ok(self.exec.to_device(host)?)
    }

    fn vec(
        &mut self,
        name: &str,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>, GpuModelError> {
        let v = self.f32s(name, &[n])?;
        self.dev32(&v)
    }

    fn dev16(&mut self, host: &[f16]) -> Result<cudarc::driver::CudaSlice<f16>, GpuModelError> {
        self.bytes += (host.len() * 2) as u64;
        Ok(self.exec.f16_to_device(host)?)
    }

    /// `nn.Linear` weight `[out, in]` -> a GEMM plane (`dims = [in, out]`).
    fn linear(
        &mut self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<HalfTensor, GpuModelError> {
        if !in_dim.is_multiple_of(8) {
            return Err(GpuModelError::Unsupported(format!(
                "laya {name}: GEMM input width {in_dim} is not a multiple of 8"
            )));
        }
        let h = self.halves(name, &[out_dim, in_dim])?;
        Ok(HalfTensor {
            buf: self.dev16(&h)?,
            dims: vec![in_dim, out_dim],
        })
    }
}

impl GpuLaya {
    /// Load one checkpoint onto `exec`. The workspace is not part of it - the
    /// engine thread sizes one for every checkpoint it holds.
    pub fn load(exec: Arc<GpuExecutor>, cfg: &LayaConfig) -> Result<Self, GpuModelError> {
        if !exec.has_text_encoder() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates the text-encoder lane (slots 671-678) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        if !exec.f16_gemm_h_elected() {
            // cc 10.0 elects the tcgen05 f32 arms; the encoder is built on the
            // f16 landings and their fused epilogues only
            return Err(GpuModelError::Unsupported(
                "laya: this device's elected f16 GEMM route has no f16 landing (cc 10.0) - \
                 the text-encoder lane is not validated there"
                    .into(),
            ));
        }
        let e = &cfg.encoder;
        let (d, f, hd) = (e.hidden, e.intermediate, e.head_dim());
        if !f.is_multiple_of(8) || !d.is_multiple_of(8) {
            return Err(GpuModelError::Unsupported(format!(
                "laya: hidden {d} / intermediate {f} must be multiples of 8"
            )));
        }
        if cfg.head_layers == 0 {
            return Err(GpuModelError::Unsupported(
                "laya: a head with no transformer layers is not built (every shipped \
                 checkpoint has two)"
                    .into(),
            ));
        }
        if cfg.n_act > 8 {
            return Err(GpuModelError::Unsupported(format!(
                "laya: {} act outputs (the act head takes up to 8)",
                cfg.n_act
            )));
        }
        let st = ShardedSafetensors::open_dir(&cfg.dir)
            .map_err(|e| GpuModelError::Unsupported(format!("laya safetensors: {e}")))?;
        let file_bytes = std::fs::metadata(cfg.weights_path()).map_or(0, |m| m.len());
        exec.vram_load_gate(file_bytes, &format!("laya ({})", cfg.name))
            .map_err(GpuModelError::WontFit)?;
        // single-stream engine - must precede every alloc
        exec.disable_event_tracking();

        let mut r = Reader {
            st: &st,
            exec: &exec,
            bytes: 0,
        };
        let emb_h = r.halves("encoder.embeddings.tok_embeddings.weight", &[e.vocab, d])?;
        let emb = r.dev16(&emb_h)?;
        drop(emb_h);
        let emb_norm = r.vec("encoder.embeddings.norm.weight", d)?;

        let mut layers = Vec::with_capacity(e.n_layer);
        for i in 0..e.n_layer {
            let p = format!("encoder.layers.{i}");
            let attn_norm = if i == 0 {
                None
            } else {
                Some(r.vec(&format!("{p}.attn_norm.weight"), d)?)
            };
            let wi_raw = r.halves(&format!("{p}.mlp.Wi.weight"), &[2 * f, d])?;
            let wi = HalfTensor {
                buf: r.dev16(&geglu_relay(&wi_raw, f, d))?,
                dims: vec![d, 2 * f],
            };
            layers.push(EncLayer {
                attn_norm,
                wqkv: r.linear(&format!("{p}.attn.Wqkv.weight"), d, 3 * d)?,
                wo: r.linear(&format!("{p}.attn.Wo.weight"), d, d)?,
                mlp_norm: r.vec(&format!("{p}.mlp_norm.weight"), d)?,
                wi,
                wo2: r.linear(&format!("{p}.mlp.Wo.weight"), f, d)?,
                global: e.global[i],
            });
        }
        let final_norm = r.vec("encoder.final_norm.weight", d)?;

        // rope: positions up to the checkpoint's own sequence budget - the
        // runner never builds a longer sequence
        let (cg, sg) = rope_table(cfg.max_len, hd, e.rope_theta_global);
        let (cl, sl) = rope_table(cfg.max_len, hd, e.rope_theta_local);
        let rope_g = (r.dev32(&cg)?, r.dev32(&sg)?);
        let rope_l = (r.dev32(&cl)?, r.dev32(&sl)?);

        let temb_h = r.f32s("type_emb.weight", &[3, d])?;
        let temb = r.dev32(&temb_h)?;

        let mut head = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            let p = format!("head.layers.{i}");
            head.push(HeadLayer {
                n1w: r.vec(&format!("{p}.norm1.weight"), d)?,
                n1b: r.vec(&format!("{p}.norm1.bias"), d)?,
                in_w: r.linear(&format!("{p}.self_attn.in_proj_weight"), d, 3 * d)?,
                in_b: r.vec(&format!("{p}.self_attn.in_proj_bias"), 3 * d)?,
                out_w: r.linear(&format!("{p}.self_attn.out_proj.weight"), d, d)?,
                out_b: r.vec(&format!("{p}.self_attn.out_proj.bias"), d)?,
                n2w: r.vec(&format!("{p}.norm2.weight"), d)?,
                n2b: r.vec(&format!("{p}.norm2.bias"), d)?,
                l1w: r.linear(&format!("{p}.linear1.weight"), d, 4 * d)?,
                l1b: r.vec(&format!("{p}.linear1.bias"), 4 * d)?,
                l2w: r.linear(&format!("{p}.linear2.weight"), 4 * d, d)?,
                l2b: r.vec(&format!("{p}.linear2.bias"), d)?,
            });
        }

        let s0w = r.vec("scorer.0.weight", d)?;
        let s0b = r.vec("scorer.0.bias", d)?;
        let s1w = r.linear("scorer.1.weight", d, d)?;
        let s1b = r.vec("scorer.1.bias", d)?;
        let s3 = r.f32s("scorer.3.weight", &[1, d])?;
        let s3w = r.dev32(&s3)?;
        let s3b = r.f32s("scorer.3.bias", &[1])?[0];

        let a0 = r.halves("act_head.0.weight", &[ACT_HIDDEN, d + 4])?;
        let a0w = r.dev16(&a0)?;
        let a0b = r.vec("act_head.0.bias", ACT_HIDDEN)?;
        let a2 = r.halves("act_head.2.weight", &[cfg.n_act, ACT_HIDDEN])?;
        let a2w = r.dev16(&a2)?;
        let a2b = r.vec("act_head.2.bias", cfg.n_act)?;

        let zeros = r.dev32(&vec![0f32; 4 * d])?;
        let ones = r.dev32(&vec![1f32; 4 * d])?;
        let weight_bytes = r.bytes;
        Ok(Self {
            exec,
            cfg: cfg.clone(),
            emb,
            emb_norm,
            layers,
            final_norm,
            rope_g,
            rope_l,
            temb,
            head,
            s0w,
            s0b,
            s1w,
            s1b,
            s3w,
            s3b,
            a0w,
            a0b,
            a2w,
            a2b,
            zeros,
            ones,
            weight_bytes,
        })
    }
}
