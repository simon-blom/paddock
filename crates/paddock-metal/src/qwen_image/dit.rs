//! Single-stream DiT with immutable prefix K/V and reusable target workspace.
//! All 32 layers are device-side; no latent/logit readback between steps.
use super::*;
use paddock_models::mapped::MappedGguf;

fn project(w: &Weight, cmd: &Commands<'_>, x: &Buffer, y: &Buffer, rows: usize, gemm: &Buffer) {
    if w.ty == crate::affine::AFFINE4 {
        mlx::project(cmd, &[(w, y)], x, rows, gemm);
    } else {
        super::project(w, cmd, x, y, rows, gemm);
    }
}

struct Block {
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    qn: Weight,
    kn: Weight,
    gate_up: Weight,
    down: Weight,
}
pub(super) struct Dit {
    mlx: bool,
    blocks: Vec<Block>,
    img: Weight,
    txt_norm: Weight,
    txt_in: Weight,
    txt_out: Weight,
    time1: Weight,
    time2: Weight,
    modulation: Weight,
    out_norm: Weight,
    out: Weight,
}
pub(super) struct Prefix {
    pub len: usize,
    pub frame: usize,
    k: Vec<Buffer>,
    v: Vec<Buffer>,
}
#[cfg(test)]
impl Prefix {
    pub fn storage_bytes(&self) -> usize {
        self.k.iter().chain(&self.v).map(Buffer::len).sum()
    }
}
pub(super) struct Scratch {
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qh: Buffer,
    attn: Buffer,
    delta: Buffer,
    gate_up: Buffer,
    gate: Buffer,
    gemm: Buffer,
    joined_k: Buffer,
    joined_v: Buffer,
}
impl Scratch {
    pub fn new(d: &MetalDevice, rows: usize, prefix: usize) -> Result<Self> {
        let a = |c| d.alloc(rows * c * 4);
        Ok(Self {
            x: a(4096)?,
            norm: a(4096)?,
            q: a(4096)?,
            k: a(4096)?,
            v: a(4096)?,
            qh: d.alloc(rows * 4096 * 2)?,
            attn: a(4096)?,
            delta: a(4096)?,
            gate_up: a(24576)?,
            gate: a(12288)?,
            gemm: d.alloc(mlx::workspace(rows))?,
            joined_k: d.alloc((rows + prefix) * 4096 * 2)?,
            joined_v: d.alloc((rows + prefix) * 4096 * 2)?,
        })
    }
}
impl Dit {
    pub fn load_mlx(d: &MetalDevice, path: &Path) -> Result<Self> {
        mlx::config(
            path,
            serde_json::json!({"_class_name":"QwenImage21Transformer2DModel",
            "attention_head_dim":128,"axes_dims_rope":[16,56,56],"context_in_dim":4096,
            "in_channels":64,"out_channels":64,"num_attention_heads":32,"num_layers":32,
            "patch_size":1,"mlp_ratio":3,"eps":1e-6,"causal_condition":true}),
        )?;
        let s = mlx::Source::open(path)?;
        let load = |n: &str, shape: &[usize]| s.weight(d, &format!("{n}.weight"), shape);
        let mut blocks = Vec::with_capacity(32);
        for i in 0..32 {
            let base = format!("transformer_blocks.{i}");
            let l = |n: &str, shape: &[usize]| load(&format!("{base}.{n}"), shape);
            blocks.push(Block {
                q: l("attn.to_q", &[4096, 4096])?,
                k: l("attn.to_k", &[4096, 4096])?,
                v: l("attn.to_v", &[4096, 4096])?,
                o: l("attn.to_out.0", &[4096, 4096])?,
                qn: l("attn.norm_q", &[128])?,
                kn: l("attn.norm_k", &[128])?,
                gate_up: s.gate_up(d, &format!("{base}.img_mlp"))?,
                down: l("img_mlp.out", &[12288, 4096])?,
            });
        }
        Ok(Self {
            mlx: true,
            blocks,
            img: load("img_in", &[64, 4096])?,
            txt_norm: load("txt_in.text_norm", &[4096])?,
            txt_in: load("txt_in.in_layer", &[4096, 4096])?,
            txt_out: load("txt_in.out_layer", &[4096, 4096])?,
            time1: load("time_text_embed.linear_1", &[256, 4096])?,
            time2: load("time_text_embed.linear_2", &[4096, 4096])?,
            modulation: load("modulation.0", &[4096, 16384])?,
            out_norm: load("norm_out.linear", &[4096, 4096])?,
            out: load("proj_out", &[4096, 64])?,
        })
    }
    pub fn load(d: &MetalDevice, path: &Path) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(model_error)?;
        let prefix = ["model.diffusion_model.", ""]
            .into_iter()
            .find(|p| map.tensor_info(&format!("{p}img_in.weight")).is_some())
            .ok_or_else(|| error("not a Qwen-Image-2.1 DiT"))?;
        let load = |n: &str, shape: &[usize]| Weight::load(d, &map, &format!("{prefix}{n}"), shape);
        let mut blocks = Vec::with_capacity(32);
        for i in 0..32 {
            let l = |n: &str, shape: &[usize]| {
                load(&format!("transformer_blocks.{i}.{n}.weight"), shape)
            };
            blocks.push(Block {
                q: l("attn.to_q", &[4096, 4096])?,
                k: l("attn.to_k", &[4096, 4096])?,
                v: l("attn.to_v", &[4096, 4096])?,
                o: l("attn.to_out.0", &[4096, 4096])?,
                qn: l("attn.norm_q", &[128])?,
                kn: l("attn.norm_k", &[128])?,
                gate_up: l("img_mlp.gate_up", &[4096, 24576])?,
                down: l("img_mlp.out", &[12288, 4096])?,
            });
        }
        Ok(Self {
            mlx: false,
            blocks,
            img: load("img_in.weight", &[64, 4096])?,
            txt_norm: load("txt_in.text_norm.weight", &[4096])?,
            txt_in: load("txt_in.in_layer.weight", &[4096, 4096])?,
            txt_out: load("txt_in.out_layer.weight", &[4096, 4096])?,
            time1: load(
                "time_text_embed.timestep_embedder.linear_1.weight",
                &[256, 4096],
            )?,
            time2: load(
                "time_text_embed.timestep_embedder.linear_2.weight",
                &[4096, 4096],
            )?,
            modulation: load("modulation.1.weight", &[4096, 16384])?,
            out_norm: load("norm_out.linear.weight", &[4096, 4096])?,
            out: load("proj_out.weight", &[4096, 64])?,
        })
    }
    fn modulation(&self, exec: &Ops, sigma: f32) -> Result<(Buffer, Buffer)> {
        let d = &exec.device;
        // The MLX graph evaluates [sampled-t, zero] together. Its two-row
        // vector contraction is not interchangeable with the one-row bias
        // reduction. Keep that contract even when caching the t=0 prefix.
        let rows = if self.mlx { 2 } else { 1 };
        let mut embedding = paddock_engine::image::schedule::timestep_embedding(sigma, 256);
        if self.mlx {
            embedding.extend(paddock_engine::image::schedule::timestep_embedding(0., 256));
        }
        let emb = exec.to_device(&embedding)?;
        let t = d.alloc(rows * 4096 * 4)?;
        let t2 = d.alloc(rows * 4096 * 4)?;
        let m = d.alloc(rows * 16384 * 4)?;
        let out = d.alloc(rows * 4096 * 4)?;
        let gemm = d.alloc(4096 * 128 * 2)?;
        let cmd = d.begin()?;
        if self.mlx {
            cmd.dispatch("qi_round_bf", &[&emb], &[512], [2, 1, 1], 256);
        }
        super::project(&self.time1, &cmd, &emb, &t, rows, &gemm);
        cmd.dispatch(
            "qi_activation",
            &[&t],
            &[(rows * 4096) as u32, u32::from(self.mlx) * 2],
            [rows * 16, 1, 1],
            256,
        );
        super::project(&self.time2, &cmd, &t, &t2, rows, &gemm);
        cmd.dispatch(
            "qi_activation",
            &[&t2],
            &[(rows * 4096) as u32, u32::from(self.mlx) * 2],
            [rows * 16, 1, 1],
            256,
        );
        super::project(&self.modulation, &cmd, &t2, &m, rows, &gemm);
        super::project(&self.out_norm, &cmd, &t2, &out, rows, &gemm);
        cmd.finish()?;
        Ok((m, out))
    }
    #[cfg(test)]
    pub fn prefix(
        &self,
        exec: &Ops,
        hidden: &Buffer,
        rows: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Prefix> {
        self.prefix_vl(
            exec,
            hidden,
            rows,
            &[conditioning::Segment::Text { start: 0, rows }],
            &[],
            cancelled,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prefix_vl(
        &self,
        exec: &Ops,
        hidden: &Buffer,
        text_rows: usize,
        segments: &[conditioning::Segment],
        references: &[Tensor<f32>],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Prefix> {
        let rows = segments
            .iter()
            .map(conditioning::Segment::rows)
            .sum::<usize>();
        if rows == 0 || rows > 32768 || text_rows > rows || hidden.len() != text_rows * 4096 * 4 {
            return Err(error("invalid multimodal image prefix geometry"));
        }
        let d = &exec.device;
        let sc = Scratch::new(d, rows, 0)?;
        let cmd = d.begin()?;
        cmd.dispatch(
            "qi_norm",
            &[hidden, &self.txt_norm.buffer, &sc.norm],
            &[
                4096,
                self.txt_norm.ty,
                1e-6f32.to_bits(),
                u32::from(self.mlx) * 2,
                0,
            ],
            [text_rows, 1, 1],
            256,
        );
        project(&self.txt_in, &cmd, &sc.norm, &sc.delta, text_rows, &sc.gemm);
        cmd.dispatch(
            "qi_activation",
            &[&sc.delta],
            &[(text_rows * 4096) as u32, 1 + u32::from(self.mlx) * 2],
            [(text_rows * 4096).div_ceil(256), 1, 1],
            256,
        );
        project(
            &self.txt_out,
            &cmd,
            &sc.delta,
            &sc.norm,
            text_rows,
            &sc.gemm,
        );
        cmd.finish()?;
        let mut off = 0;
        for seg in segments {
            check_cancelled(cancelled)?;
            let n = seg.rows();
            match *seg {
                conditioning::Segment::Text { start, rows } => {
                    if start.checked_add(rows).is_none_or(|end| end > text_rows) {
                        return Err(error("text segment outside conditioning"));
                    }
                    d.copy_regions(&[(
                        &sc.norm,
                        start * 4096 * 4,
                        &sc.x,
                        off * 4096 * 4,
                        n * 4096 * 4,
                    )])?;
                }
                conditioning::Segment::Image { index, .. } => {
                    let latent = references
                        .get(index)
                        .filter(|z| z.len() == n * 64 * 4)
                        .ok_or_else(|| error("invalid reference latent segment"))?;
                    let cmd = d.begin()?;
                    project(&self.img, &cmd, latent, &sc.delta, n, &sc.gemm);
                    cmd.finish()?;
                    d.copy_regions(&[(&sc.delta, 0, &sc.x, off * 4096 * 4, n * 4096 * 4)])?;
                }
            }
            off += n;
        }
        let (positions, bounds, frame) = conditioning::prefix_layout(segments);
        let pos = words(d, &positions)?;
        let meta = words(d, &bounds)?;
        let tiles = tiles(d, rows)?;
        let (m, _) = self.modulation(exec, 0.)?;
        let mut prefix = Prefix {
            len: rows,
            frame,
            k: Vec::with_capacity(32),
            v: Vec::with_capacity(32),
        };
        for b in &self.blocks {
            check_cancelled(cancelled)?;
            let k = d.alloc(rows * 4096 * 2)?;
            let v = d.alloc(rows * 4096 * 2)?;
            let cmd = d.begin()?;
            #[cfg(test)]
            let trace = diagnostic::BlockTrace::new(
                d,
                "prefix",
                serde_json::json!({"rows":rows,"offset":0,"sigma":0.0}),
            )?;
            self.block(
                &cmd,
                b,
                &sc,
                &m,
                &pos,
                &meta,
                &tiles,
                &k,
                &v,
                rows,
                0,
                #[cfg(test)]
                trace.as_ref(),
            )?;
            cmd.finish()?;
            #[cfg(test)]
            if let Some(trace) = trace {
                trace.save()?;
            }
            prefix.k.push(k);
            prefix.v.push(v);
        }
        Ok(prefix)
    }
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        cmd: &Commands<'_>,
        b: &Block,
        sc: &Scratch,
        m: &Buffer,
        pos: &Buffer,
        meta: &Buffer,
        tiles: &Buffer,
        k: &Buffer,
        v: &Buffer,
        rows: usize,
        offset: usize,
        #[cfg(test)] trace: Option<&diagnostic::BlockTrace<'_>>,
    ) -> Result<()> {
        macro_rules! capture {
            ($name:expr, $buffer:expr, $count:expr, $half:expr) => {
                #[cfg(test)]
                if let Some(trace) = trace {
                    trace.snapshot(cmd, $name, $buffer, $count, $half)?;
                }
            };
        }
        capture!("input", &sc.x, rows * 4096, false);
        capture!("modulation", m, 16384, false);
        cmd.dispatch(
            "qi_norm",
            &[&sc.x, m, &sc.norm],
            &[4096, 0, 1e-6f32.to_bits(), 1 + u32::from(self.mlx) * 2, 0],
            [rows, 1, 1],
            256,
        );
        capture!("norm1", &sc.norm, rows * 4096, false);
        mlx::project(
            cmd,
            &[(&b.q, &sc.q), (&b.k, &sc.k), (&b.v, &sc.v)],
            &sc.norm,
            rows,
            &sc.gemm,
        );
        capture!("q", &sc.q, rows * 4096, false);
        capture!("k", &sc.k, rows * 4096, false);
        capture!("v", &sc.v, rows * 4096, false);
        for (x, n, y, off) in [(&sc.q, &b.qn, &sc.qh, 0), (&sc.k, &b.kn, k, offset)] {
            cmd.dispatch(
                if self.mlx {
                    "qi_mlx_head_rope"
                } else {
                    "qi_head_rope"
                },
                &[x, &n.buffer, pos, y],
                &[32, n.ty, off as u32],
                [32, rows, 1],
                32,
            );
        }
        cmd.dispatch(
            "qi_store_half",
            &[&sc.v, v],
            &[(rows * 4096) as u32, (offset * 4096) as u32],
            [(rows * 4096).div_ceil(256), 1, 1],
            256,
        );
        capture!("qrope", &sc.qh, rows * 4096, true);
        capture!("krope", k, (rows + offset) * 4096, true);
        capture!("values", v, (rows + offset) * 4096, true);
        let deep = cmd.tensor_accelerated() && rows >= 1024;
        cmd.dispatch(
            if deep {
                "qi_attention_deep"
            } else {
                "qi_attention"
            },
            &[&sc.qh, k, v, meta, meta, &sc.attn, tiles],
            &[32, 32, 0, (1f32 / 128f32.sqrt()).to_bits()],
            [32, rows.div_ceil(if deep { 16 } else { 32 }), 1],
            128,
        );
        capture!("attention", &sc.attn, rows * 4096, false);
        project(&b.o, cmd, &sc.attn, &sc.delta, rows, &sc.gemm);
        capture!("o", &sc.delta, rows * 4096, false);
        cmd.dispatch(
            if self.mlx {
                "qi_mlx_gated_add"
            } else {
                "qi_gated_add"
            },
            &[&sc.x, &sc.delta, m],
            &[(rows * 4096) as u32, 4096, 4096],
            [(rows * 4096).div_ceil(256), 1, 1],
            256,
        );
        capture!("residual1", &sc.x, rows * 4096, false);
        cmd.dispatch(
            "qi_norm",
            &[&sc.x, m, &sc.norm],
            &[
                4096,
                0,
                1e-6f32.to_bits(),
                1 + u32::from(self.mlx) * 2,
                8192,
            ],
            [rows, 1, 1],
            256,
        );
        capture!("norm2", &sc.norm, rows * 4096, false);
        project(&b.gate_up, cmd, &sc.norm, &sc.gate_up, rows, &sc.gemm);
        capture!("gate_up", &sc.gate_up, rows * 24576, false);
        cmd.dispatch(
            if self.mlx {
                "qi_mlx_gate_up"
            } else {
                "qi_swiglu"
            },
            &[&sc.gate_up, &sc.gate],
            &[rows as u32, 12288],
            [(rows * 12288).div_ceil(256), 1, 1],
            256,
        );
        capture!("gate", &sc.gate, rows * 12288, false);
        project(&b.down, cmd, &sc.gate, &sc.delta, rows, &sc.gemm);
        capture!("down", &sc.delta, rows * 4096, false);
        cmd.dispatch(
            if self.mlx {
                "qi_mlx_gated_add"
            } else {
                "qi_gated_add"
            },
            &[&sc.x, &sc.delta, m],
            &[(rows * 4096) as u32, 4096, 12288],
            [(rows * 4096).div_ceil(256), 1, 1],
            256,
        );
        capture!("residual2", &sc.x, rows * 4096, false);
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        exec: &Ops,
        prefix: &Prefix,
        latent: &Buffer,
        sc: &Scratch,
        sigma: f32,
        lw: usize,
        lh: usize,
        out: &Buffer,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<()> {
        let d = &exec.device;
        let rows = lw * lh;
        if sc.joined_k.len() < (prefix.len + rows) * 4096 * 2
            || sc.joined_v.len() < (prefix.len + rows) * 4096 * 2
        {
            return Err(error(
                "image attention workspace is smaller than its prefix and target",
            ));
        }
        let pos = words(
            d,
            &(0..rows)
                .flat_map(|i| {
                    [
                        prefix.frame as u32,
                        ((i / lw) as i32 - (lh - lh / 2) as i32) as u32,
                        ((i % lw) as i32 - (lw - lw / 2) as i32) as u32,
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        let meta = words(
            d,
            &(0..rows)
                .flat_map(|_| [0, (prefix.len + rows - 1) as u32])
                .collect::<Vec<_>>(),
        )?;
        let tiles = tiles(d, rows)?;
        let (m, out_norm) = self.modulation(exec, sigma)?;
        let cmd = d.begin()?;
        project(&self.img, &cmd, latent, &sc.x, rows, &sc.gemm);
        cmd.finish()?;
        for (i, b) in self.blocks.iter().enumerate() {
            check_cancelled(cancelled)?;
            let cmd = d.begin()?;
            #[cfg(test)]
            let trace = diagnostic::BlockTrace::new(
                d,
                "target",
                serde_json::json!({"rows":rows,"offset":prefix.len,"sigma":sigma,
                                  "latent_width":lw,"latent_height":lh}),
            )?;
            // Only the immutable prefix is retained per layer. The target's
            // K/V are consumed by this layer and then reused by the next.
            // Copy packed bits, preserving every BF16/F16 value exactly.
            let words = prefix.len * 4096 / 2;
            cmd.dispatch(
                "qi_prefix_copy",
                &[&prefix.k[i], &prefix.v[i], &sc.joined_k, &sc.joined_v],
                &[words as u32],
                [words.div_ceil(256), 1, 1],
                256,
            );
            self.block(
                &cmd,
                b,
                sc,
                &m,
                &pos,
                &meta,
                &tiles,
                &sc.joined_k,
                &sc.joined_v,
                rows,
                prefix.len,
                #[cfg(test)]
                trace.as_ref(),
            )?;
            cmd.finish()?;
            #[cfg(test)]
            if let Some(trace) = trace {
                trace.save()?;
            }
        }
        let cmd = d.begin()?;
        cmd.dispatch(
            "qi_norm",
            &[&sc.x, &out_norm, &sc.norm],
            &[4096, 0, 1e-6f32.to_bits(), 1 + u32::from(self.mlx) * 2, 0],
            [rows, 1, 1],
            256,
        );
        project(&self.out, &cmd, &sc.norm, out, rows, &sc.gemm);
        cmd.finish()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Metal and PADDOCK_QI_MLX checkpoint"]
    fn mlx_zero_time_prefix_modulation_is_independent_of_target_sigma() {
        let root = std::env::var("PADDOCK_QI_MLX").expect("MLX checkpoint");
        let e = Ops {
            device: MetalDevice::new(None).unwrap(),
        };
        let dit = Dit::load_mlx(&e.device, &Path::new(&root).join("transformer")).unwrap();
        let (cached, cached_out) = dit.modulation(&e, 0.).unwrap();
        let expected = unsafe { cached.read_f32(0, 16384) };
        let expected_out = unsafe { cached_out.read_f32(0, 4096) };
        for sigma in [0., 0.02, 0.123, 0.75, 1.] {
            let (joint, joint_out) = dit.modulation(&e, sigma).unwrap();
            // modulation() fences before returning. The second row is the
            // reference graph's zero-time prefix, independent of target t.
            assert_eq!(unsafe { joint.read_f32(16384, 16384) }, expected);
            assert_eq!(unsafe { joint_out.read_f32(4096, 4096) }, expected_out);
        }
    }
}
