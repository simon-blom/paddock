//! Validate the complete original safetensors inventory before allocating GPU
//! memory. Dense matrices stay F16; only small norms/biases widen to F32.
use super::*;
use half::f16;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

pub(super) struct Source {
    st: ShardedSafetensors,
    pub resident_bytes: u64,
}
fn upload_f32(device: &MetalDevice, values: &[f32]) -> Result<Buffer> {
    device.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn schema(cfg: &LayaConfig) -> Vec<(String, Vec<usize>, bool)> {
    let e = &cfg.encoder;
    let d = e.hidden;
    let f = e.intermediate;
    let mut s = vec![];
    let mut add = |name: String, shape: Vec<usize>, wide: bool| s.push((name, shape, wide));
    add(
        "encoder.embeddings.tok_embeddings.weight".into(),
        vec![e.vocab, d],
        false,
    );
    add("encoder.embeddings.norm.weight".into(), vec![d], true);
    for i in 0..e.n_layer {
        let p = format!("encoder.layers.{i}");
        if i > 0 {
            add(format!("{p}.attn_norm.weight"), vec![d], true);
        }
        for (name, shape, wide) in [
            ("attn.Wqkv.weight", vec![3 * d, d], false),
            ("attn.Wo.weight", vec![d, d], false),
            ("mlp_norm.weight", vec![d], true),
            ("mlp.Wi.weight", vec![2 * f, d], false),
            ("mlp.Wo.weight", vec![d, f], false),
        ] {
            add(format!("{p}.{name}"), shape, wide);
        }
    }
    add("encoder.final_norm.weight".into(), vec![d], true);
    add("type_emb.weight".into(), vec![3, d], true);
    for i in 0..cfg.head_layers {
        let p = format!("head.layers.{i}");
        for n in ["norm1", "norm2"] {
            for tail in ["weight", "bias"] {
                add(format!("{p}.{n}.{tail}"), vec![d], true);
            }
        }
        for (n, k, out) in [
            ("self_attn.in_proj", d, 3 * d),
            ("self_attn.out_proj", d, d),
            ("linear1", d, 4 * d),
            ("linear2", 4 * d, d),
        ] {
            let sep = if n.ends_with("in_proj") { "_" } else { "." };
            add(format!("{p}.{n}{sep}weight"), vec![out, k], false);
            add(format!("{p}.{n}{sep}bias"), vec![out], true);
        }
    }
    for tail in ["weight", "bias"] {
        add(format!("scorer.0.{tail}"), vec![d], true);
    }
    for (n, k, out) in [
        ("scorer.1", d, d),
        ("scorer.3", d, 1),
        ("act_head.0", d + 4, 256),
        ("act_head.2", 256, cfg.n_act),
    ] {
        add(format!("{n}.weight"), vec![out, k], n == "scorer.3");
        add(format!("{n}.bias"), vec![out], true);
    }
    add("temperature".into(), vec![3], false);
    s
}
impl Source {
    pub fn open(cfg: &LayaConfig) -> Result<Self> {
        let e = &cfg.encoder;
        if ![768, 1024].contains(&e.hidden)
            || e.head_dim() != 64
            || cfg.head_layers == 0
            || !(1..=8).contains(&cfg.n_act)
            || e.n_layer == 0
            || e.intermediate == 0
            || !e.intermediate.is_multiple_of(64)
            || cfg.max_len > PASS_TOKENS
            || e.vocab <= 6
        {
            return Err(error("unsupported ModernBERT/head dimensions"));
        }
        let st = ShardedSafetensors::open_dir(&cfg.dir).map_err(|e| error(e.to_string()))?;
        let schema = schema(cfg);
        if st.names().count() != schema.len() {
            return Err(error("unexpected tensor inventory"));
        }
        let mut resident_bytes = 0;
        for (name, shape, wide) in schema {
            let (info, bytes) = st
                .bytes(&name)
                .ok_or_else(|| error(format!("missing {name}")))?;
            if name == "temperature" && info.dtype == StDtype::F32 {
                if info.shape != [3]
                    || bytes.len() != 12
                    || bytes
                        .chunks_exact(4)
                        .any(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
                {
                    return Err(error("temperature: expected three finite F32 values"));
                }
                continue;
            }
            let size = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d));
            if info.dtype != StDtype::F16 || info.shape != shape || size != Some(bytes.len()) {
                return Err(error(format!("{name}: expected F16 {shape:?}")));
            }
            if bytes
                .chunks_exact(2)
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7c00 == 0x7c00)
            {
                return Err(error(format!("{name}: nonfinite weight")));
            }
            // The temperature is applied by the shared host calibration layer;
            // scorer.3.bias is passed as a constant, not resident GPU storage.
            if name != "temperature" && name != "scorer.3.bias" {
                resident_bytes += bytes.len() as u64 * if wide { 2 } else { 1 };
            }
        }
        resident_bytes += (4 * e.hidden * 4 + cfg.max_len * 32 * 2 * 4 * 2) as u64; // zero bias + both cos/sin tables
        Ok(Self { st, resident_bytes })
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.st
            .bytes(name)
            .map(|(_, b)| b)
            .ok_or_else(|| error(format!("missing {name}")))
    }
    fn floats(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self
            .bytes(name)?
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect())
    }
    fn vector(&self, device: &MetalDevice, name: &str) -> Result<Buffer> {
        upload_f32(device, &self.floats(name)?)
    }
    fn linear(&self, device: &MetalDevice, name: &str, k: usize, n: usize) -> Result<Linear> {
        Ok(Linear {
            w: device.upload(self.bytes(name)?)?,
            k,
            n,
        })
    }
    pub fn load(&self, device: &MetalDevice, cfg: &LayaConfig) -> Result<Model> {
        let e = &cfg.encoder;
        let d = e.hidden;
        let f = e.intermediate;
        let v = |name: &str| self.vector(device, name);
        let lin = |name: &str, k, n| self.linear(device, name, k, n);
        let norm = |p: &str| -> Result<Norm> {
            Ok(Norm {
                w: v(&format!("{p}.weight"))?,
                b: v(&format!("{p}.bias"))?,
            })
        };
        let mut layers = vec![];
        for i in 0..e.n_layer {
            let p = format!("encoder.layers.{i}");
            layers.push(EncoderLayer {
                attn_norm: if i > 0 {
                    Some(v(&format!("{p}.attn_norm.weight"))?)
                } else {
                    None
                },
                qkv: lin(&format!("{p}.attn.Wqkv.weight"), d, 3 * d)?,
                out: lin(&format!("{p}.attn.Wo.weight"), d, d)?,
                mlp_norm: v(&format!("{p}.mlp_norm.weight"))?,
                up: lin(&format!("{p}.mlp.Wi.weight"), d, f)?,
                down: lin(&format!("{p}.mlp.Wo.weight"), f, d)?,
                global: e.global[i],
            });
        }
        let mut head = vec![];
        for i in 0..cfg.head_layers {
            let p = format!("head.layers.{i}");
            head.push(HeadLayer {
                n1: norm(&format!("{p}.norm1"))?,
                n2: norm(&format!("{p}.norm2"))?,
                qkv: lin(&format!("{p}.self_attn.in_proj_weight"), d, 3 * d)?,
                qb: v(&format!("{p}.self_attn.in_proj_bias"))?,
                out: lin(&format!("{p}.self_attn.out_proj.weight"), d, d)?,
                ob: v(&format!("{p}.self_attn.out_proj.bias"))?,
                up: lin(&format!("{p}.linear1.weight"), d, 4 * d)?,
                ub: v(&format!("{p}.linear1.bias"))?,
                down: lin(&format!("{p}.linear2.weight"), 4 * d, d)?,
                db: v(&format!("{p}.linear2.bias"))?,
            });
        }
        let rope = |theta: f32| -> Result<Buffer> {
            let mut table = Vec::with_capacity(cfg.max_len * 64);
            for pos in 0..cfg.max_len {
                for j in 0..32 {
                    let a = pos as f32 * (1.0 / theta.powf(j as f32 / 32.0));
                    table.extend([a.cos(), a.sin()]);
                }
            }
            upload_f32(device, &table)
        };
        Ok(Model {
            cfg: cfg.clone(),
            layers,
            head,
            emb: device.upload(self.bytes("encoder.embeddings.tok_embeddings.weight")?)?,
            emb_norm: v("encoder.embeddings.norm.weight")?,
            final_norm: v("encoder.final_norm.weight")?,
            rope_g: rope(e.rope_theta_global)?,
            rope_l: rope(e.rope_theta_local)?,
            type_emb: v("type_emb.weight")?,
            score_norm: norm("scorer.0")?,
            score: lin("scorer.1.weight", d, d)?,
            score_bias: v("scorer.1.bias")?,
            score_w: v("scorer.3.weight")?,
            score_b: self.floats("scorer.3.bias")?[0],
            act_w0: device.upload(self.bytes("act_head.0.weight")?)?,
            act_b0: v("act_head.0.bias")?,
            act_w2: device.upload(self.bytes("act_head.2.weight")?)?,
            act_b2: v("act_head.2.bias")?,
            zeros: upload_f32(device, &vec![0.0; 4 * d])?,
        })
    }
}
