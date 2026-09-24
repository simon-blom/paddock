//! Qwen3-VL conditioning before final RMSNorm (not pooled embeddings/logits).
use super::*;
use crate::weights::projections;
use paddock_models::{gguf::Value, mapped::MappedGguf};

#[cfg(test)]
fn trace(x: &Buffer, name: &str, rows: usize) -> Result<()> {
    if let Some(dir) = std::env::var_os("PADDOCK_QI_TEXT_TRACE") {
        let values = unsafe { x.read_f32(0, rows * 4096) };
        std::fs::write(
            Path::new(&dir).join(format!("text-{name}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(model_error)?;
    }
    Ok(())
}

#[cfg(test)]
fn trace_stage<'a>(
    exec: &'a Ops,
    cmd: Commands<'a>,
    first: bool,
    x: &Buffer,
    name: &str,
    count: usize,
) -> Result<Commands<'a>> {
    if first && let Some(dir) = std::env::var_os("PADDOCK_QI_TEXT_TRACE") {
        cmd.finish()?;
        let values = if x.len() == count * 2 {
            unsafe { x.read_u32(count.div_ceil(2)) }
                .into_iter()
                .flat_map(|v| [v as u16, (v >> 16) as u16])
                .take(count)
                .map(|v| half::f16::from_bits(v).to_f32())
                .collect()
        } else {
            unsafe { x.read_f32(0, count) }
        };
        std::fs::write(
            Path::new(&dir).join(format!("text-first-{name}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(model_error)?;
        exec.device.begin()
    } else {
        Ok(cmd)
    }
}

struct Layer {
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    qnorm: Weight,
    knorm: Weight,
    o: Weight,
    ffn_norm: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}
pub(super) struct TextEncoder {
    embedding: Weight,
    layers: Vec<Layer>,
    eps: f32,
    rope: f32,
    context: usize,
}
impl TextEncoder {
    pub fn load_mlx(exec: &Ops, path: &Path, context: usize) -> Result<Self> {
        if !(1..=8192).contains(&context) {
            return Err(error("image text context must be 1..=8192"));
        }
        let v = mlx::config(path, serde_json::json!({"model_type":"qwen3_vl"}))?;
        let tc = &v["text_config"];
        if tc["rope_scaling"]
            != serde_json::json!({"mrope_interleaved":true,"mrope_section":[24,20,20],"rope_type":"default"})
        {
            return Err(error(
                "unsupported image text multimodal rotary configuration",
            ));
        }
        for (key, expected) in serde_json::json!({"hidden_size":4096,"num_hidden_layers":36,
            "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,
            "intermediate_size":12288,"vocab_size":151936,"rms_norm_eps":1e-6,
            "rope_theta":5000000,"attention_bias":false,"hidden_act":"silu"})
        .as_object()
        .expect("fixed schema")
        {
            if tc.get(key) != Some(expected) {
                return Err(error(format!("image MLX text_config.{key}")));
            }
        }
        let source = mlx::Source::open(path)?;
        let root = "language_model.model";
        let load = |name: &str, shape: &[usize]| {
            source.weight(&exec.device, &format!("{root}.{name}.weight"), shape)
        };
        let embedding = load("embed_tokens", &[4096, 151936])?;
        let mut layers = Vec::with_capacity(36);
        for i in 0..36 {
            let l = |name: &str, shape: &[usize]| load(&format!("layers.{i}.{name}"), shape);
            layers.push(Layer {
                norm: l("input_layernorm", &[4096])?,
                q: l("self_attn.q_proj", &[4096, 4096])?,
                k: l("self_attn.k_proj", &[4096, 1024])?,
                v: l("self_attn.v_proj", &[4096, 1024])?,
                qnorm: l("self_attn.q_norm", &[128])?,
                knorm: l("self_attn.k_norm", &[128])?,
                o: l("self_attn.o_proj", &[4096, 4096])?,
                ffn_norm: l("post_attention_layernorm", &[4096])?,
                gate: l("mlp.gate_proj", &[4096, 12288])?,
                up: l("mlp.up_proj", &[4096, 12288])?,
                down: l("mlp.down_proj", &[12288, 4096])?,
            });
        }
        Ok(Self {
            embedding,
            layers,
            eps: 1e-6,
            rope: 5000000.,
            context,
        })
    }
    pub fn load(exec: &Ops, path: &Path, context: usize) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(model_error)?;
        let g = map.gguf();
        if g.architecture() != Some("qwen3vl") {
            return Err(error("Qwen-Image requires the Qwen3-VL-8B text encoder"));
        }
        for (key, value) in [
            ("embedding_length", 4096),
            ("block_count", 36),
            ("attention.head_count", 32),
            ("attention.head_count_kv", 8),
            ("attention.key_length", 128),
            ("feed_forward_length", 12288),
        ] {
            if g.arch_field(key).and_then(Value::as_u64) != Some(value) {
                return Err(error(format!("unexpected Qwen3-VL {key}")));
            }
        }
        if context == 0 || context > 8192 {
            return Err(error("image text context must be 1..=8192"));
        }
        let number = |name: &str| {
            g.arch_field(name)
                .and_then(Value::as_f32)
                .filter(|v| v.is_finite() && *v > 0.)
                .ok_or_else(|| error(format!("missing {name}")))
        };
        let rope = number("rope.freq_base")?;
        let eps = number("attention.layer_norm_rms_epsilon")?;
        let vocab = map
            .tensor_info("token_embd.weight")
            .and_then(|t| t.dims.get(1))
            .copied()
            .ok_or_else(|| error("missing text embeddings"))? as usize;
        let load = |n: &str, d: &[usize]| Weight::load(&exec.device, &map, n, d);
        let embedding = load("token_embd.weight", &[4096, vocab])?;
        let mut layers = Vec::with_capacity(36);
        for i in 0..36 {
            let l = |n: &str, d: &[usize]| load(&format!("blk.{i}.{n}.weight"), d);
            layers.push(Layer {
                norm: l("attn_norm", &[4096])?,
                q: l("attn_q", &[4096, 4096])?,
                k: l("attn_k", &[4096, 1024])?,
                v: l("attn_v", &[4096, 1024])?,
                qnorm: l("attn_q_norm", &[128])?,
                knorm: l("attn_k_norm", &[128])?,
                o: l("attn_output", &[4096, 4096])?,
                ffn_norm: l("ffn_norm", &[4096])?,
                gate: l("ffn_gate", &[4096, 12288])?,
                up: l("ffn_up", &[4096, 12288])?,
                down: l("ffn_down", &[12288, 4096])?,
            });
        }
        Ok(Self {
            embedding,
            layers,
            eps,
            rope,
            context,
        })
    }
    #[cfg(test)]
    pub fn encode(
        &self,
        exec: &Ops,
        ids: &[u32],
        drop: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Buffer> {
        self.encode_vl(exec, ids, drop, &[], &[], cancelled)
    }
    pub fn validate(&self, ids: &[u32], drop: usize) -> Result<()> {
        if ids.len() > self.context
            || drop >= ids.len()
            || ids.iter().any(|&t| t as usize >= self.embedding.n)
        {
            return Err(error(
                "image prompt exceeds text encoder context or vocabulary",
            ));
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn encode_vl(
        &self,
        exec: &Ops,
        ids: &[u32],
        drop: usize,
        images: &[&vision::Output],
        runs: &[(usize, usize)],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Buffer> {
        self.validate(ids, drop)?;
        if images.len() != runs.len()
            || images.iter().zip(runs).any(|(im, &(off, len))| {
                off < drop
                    || off.checked_add(len).is_none_or(|end| end > ids.len())
                    || im.nx * im.ny != len
                    || im.deepstack.len() != 3
            })
        {
            return Err(error("invalid text vision splice"));
        }
        let rows = ids.len();
        if rows > self.context
            || drop >= rows
            || ids.iter().any(|&t| t as usize >= self.embedding.n)
        {
            return Err(error(
                "image prompt exceeds text encoder context or vocabulary",
            ));
        }
        let d = &exec.device;
        let positions = if images.is_empty() {
            None
        } else {
            Some(words(
                d,
                &conditioning::text_positions(
                    rows,
                    runs,
                    &images.iter().map(|im| (im.nx, im.ny)).collect::<Vec<_>>(),
                ),
            )?)
        };
        let ids = words(d, ids)?;
        let meta = words(
            d,
            &(0..rows).flat_map(|i| [0, i as u32]).collect::<Vec<_>>(),
        )?;
        let tiles = tiles(d, rows)?;
        let alloc = |n| d.alloc(rows * n * 4);
        let x = alloc(4096)?;
        let norm = alloc(4096)?;
        let q = alloc(4096)?;
        let k = alloc(1024)?;
        let v = alloc(1024)?;
        let qh = d.alloc(rows * 4096 * 2)?;
        let kh = d.alloc(rows * 1024 * 2)?;
        let vh = d.alloc(rows * 1024 * 2)?;
        let attn = alloc(4096)?;
        let delta = alloc(4096)?;
        let gate = alloc(12288)?;
        let up = alloc(12288)?;
        let gemm = d.alloc(mlx::workspace(rows))?;
        let cmd = d.begin()?;
        let affine = self.embedding.ty == crate::affine::AFFINE4;
        cmd.dispatch(
            if affine { "mlx_embed" } else { "embed" },
            &[&self.embedding.buffer, &ids, &x],
            &[
                4096,
                rows as u32,
                if affine {
                    self.embedding.n as u32
                } else {
                    self.embedding.ty
                },
                1f32.to_bits(),
            ],
            [(rows * 4096).div_ceil(256), 1, 1],
            256,
        );
        for (im, &(off, len)) in images.iter().zip(runs) {
            cmd.dispatch(
                "qi_text_splice",
                &[&im.embd, &x],
                &[len as u32, off as u32, 0, u32::from(affine)],
                [(len * 4096).div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.finish()?;
        #[cfg(test)]
        trace(&x, "embed", rows)?;
        for (layer, l) in self.layers.iter().enumerate() {
            check_cancelled(cancelled)?;
            let cmd = d.begin()?;
            cmd.dispatch(
                "rms",
                &[&x, &l.norm.buffer, &norm],
                &[4096, l.norm.ty, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &norm,
                "norm",
                rows * 4096,
            )?;
            projections(
                &cmd,
                &[(&l.q, &q), (&l.k, &k), (&l.v, &v)],
                &norm,
                rows,
                &gemm,
            );
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &q,
                "q",
                rows * 4096,
            )?;
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &k,
                "k",
                rows * 1024,
            )?;
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &v,
                "v",
                rows * 1024,
            )?;
            for (input, n, out, heads, store) in
                [(&q, &l.qnorm, &qh, 32, 0), (&k, &l.knorm, &kh, 8, 1)]
            {
                cmd.dispatch(
                    if positions.is_some() {
                        "qi_text_mrope"
                    } else if affine {
                        "qi_mlx_text_rope"
                    } else {
                        "qwen3_head_rope"
                    },
                    &[
                        input,
                        &n.buffer,
                        positions.as_ref().unwrap_or(&meta),
                        out,
                        &v,
                        &vh,
                    ],
                    &[
                        heads,
                        n.ty,
                        self.rope.to_bits(),
                        self.eps.to_bits(),
                        store,
                        u32::from(affine),
                    ],
                    [heads as usize, rows, 1],
                    32,
                );
            }
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &qh,
                "qh",
                rows * 4096,
            )?;
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &kh,
                "kh",
                rows * 1024,
            )?;
            if affine && rows <= 256 {
                cmd.dispatch(
                    "qi_text_attention",
                    &[&qh, &kh, &vh, &attn],
                    &[],
                    [32, rows, 1],
                    32,
                );
            } else {
                cmd.dispatch(
                    "qwen3_attention",
                    &[&qh, &kh, &vh, &meta, &meta, &attn, &tiles],
                    &[32, 8, 0, (1f32 / 128f32.sqrt()).to_bits()],
                    [32, rows.div_ceil(32), 1],
                    128,
                );
            }
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &attn,
                "attn",
                rows * 4096,
            )?;
            project(&l.o, &cmd, &attn, &delta, rows, &gemm);
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &delta,
                "o",
                rows * 4096,
            )?;
            cmd.dispatch(
                if affine {
                    "qi_mlx_residual"
                } else {
                    "residual"
                },
                &[&x, &delta],
                &[(rows * 4096) as u32, 1f32.to_bits()],
                [(rows * 4096).div_ceil(256), 1, 1],
                256,
            );
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &x,
                "residual",
                rows * 4096,
            )?;
            cmd.dispatch(
                "rms",
                &[&x, &l.ffn_norm.buffer, &norm],
                &[4096, l.ffn_norm.ty, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            projections(&cmd, &[(&l.gate, &gate), (&l.up, &up)], &norm, rows, &gemm);
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &gate,
                "gate",
                rows * 12288,
            )?;
            cmd.dispatch(
                if affine { "qi_mlx_swiglu" } else { "swiglu" },
                &[&gate, &up],
                &[(rows * 12288) as u32],
                [(rows * 12288).div_ceil(256), 1, 1],
                256,
            );
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &gate,
                "swiglu",
                rows * 12288,
            )?;
            project(&l.down, &cmd, &gate, &delta, rows, &gemm);
            #[cfg(test)]
            let cmd = trace_stage(
                exec,
                cmd,
                std::ptr::eq(l, &self.layers[0]),
                &delta,
                "down",
                rows * 4096,
            )?;
            cmd.dispatch(
                if affine {
                    "qi_mlx_residual"
                } else {
                    "residual"
                },
                &[&x, &delta],
                &[(rows * 4096) as u32, 1f32.to_bits()],
                [(rows * 4096).div_ceil(256), 1, 1],
                256,
            );
            if layer < 3 {
                for (im, &(off, len)) in images.iter().zip(runs) {
                    cmd.dispatch(
                        "qi_text_splice",
                        &[&im.deepstack[layer], &x],
                        &[len as u32, off as u32, 1, u32::from(affine)],
                        [(len * 4096).div_ceil(256), 1, 1],
                        256,
                    );
                }
            }
            cmd.finish()?;
            #[cfg(test)]
            trace(
                &x,
                &format!(
                    "layer-{:02}",
                    self.layers
                        .iter()
                        .position(|item| std::ptr::eq(item, l))
                        .unwrap()
                ),
                rows,
            )?;
        }
        let out = d.alloc((rows - drop) * 4096 * 4)?;
        d.copy_regions(&[(&x, drop * 4096 * 4, &out, 0, out.len())])?;
        Ok(out)
    }
}
