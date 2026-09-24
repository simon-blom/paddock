//! Qwen3-VL-8B vision conditioning, including all three DeepStack mergers.
//! Packed MLX matrices stay packed; GGUF uses its own F16 companion. Inputs
//! have already been resized by the shared image API. No second resize or
//! colour/alpha loss is introduced here.
use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
const E: usize = 1152;
const F: usize = 4304;

struct Linear {
    w: Weight,
    b: Weight,
}
struct Norm {
    w: Weight,
    b: Weight,
}
struct Block {
    n1: Norm,
    qkv: Linear,
    out: Linear,
    n2: Norm,
    up: Linear,
    down: Linear,
}
struct Merger {
    norm: Norm,
    up: Linear,
    down: Linear,
}
pub(super) struct Vision {
    patch: Weight,
    patch1: Option<Weight>,
    bias: Weight,
    pos: Buffer,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
    mean: [f32; 3],
    std: [f32; 3],
}
pub(super) struct Output {
    pub embd: Buffer,
    pub deepstack: Vec<Buffer>,
    pub nx: usize,
    pub ny: usize,
}

fn vector(d: &MetalDevice, w: Weight) -> Result<Weight> {
    if w.ty == 0 {
        return Ok(w);
    }
    if !matches!(w.ty, 1 | 30) {
        return Err(error("vision vector must be dense"));
    }
    let buffer = d.alloc(w.k * 4)?;
    let bad = words(d, &[0])?;
    let cmd = d.begin()?;
    cmd.dispatch(
        "vis_cast",
        &[&w.buffer, &buffer, &bad],
        &[w.k as u32, w.ty, 0],
        [w.k.div_ceil(256), 1, 1],
        256,
    );
    cmd.finish()?;
    if unsafe { bad.read_u32(1)[0] } != 0 {
        return Err(error("nonfinite vision vector"));
    }
    Ok(Weight {
        buffer,
        ty: 0,
        k: w.k,
        n: 1,
    })
}
impl Vision {
    pub fn load_mlx(d: &MetalDevice, path: &Path) -> Result<Self> {
        let cfg = mlx::config(path, serde_json::json!({"model_type":"qwen3_vl"}))?;
        let expected = serde_json::json!({"depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,"out_hidden_size":4096,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,"num_position_embeddings":2304,"deepstack_visual_indexes":[8,16,24],"hidden_act":"gelu_pytorch_tanh"});
        for (k, v) in expected.as_object().expect("schema") {
            if cfg["vision_config"].get(k) != Some(v) {
                return Err(error(format!("unsupported image vision_config.{k}")));
            }
        }
        let s = mlx::Source::open(path)?;
        let load =
            |name: &str, shape: &[usize]| s.weight(d, &format!("vision_tower.{name}"), shape);
        let mut patch = load("patch_embed.proj.weight", &[16, 16, 2, 3, E])?;
        patch.k = 1536;
        patch.n = E;
        let bias = vector(d, load("patch_embed.proj.bias", &[E])?)?;
        let pw = load("pos_embed.weight", &[E, 2304])?;
        let ids = words(d, &(0..2304).collect::<Vec<_>>())?;
        let pos = d.alloc(2304 * E * 4)?;
        let cmd = d.begin()?;
        cmd.dispatch(
            if pw.ty == crate::affine::AFFINE4 {
                "mlx_embed"
            } else {
                "embed"
            },
            &[&pw.buffer, &ids, &pos],
            &[
                E as u32,
                2304,
                if pw.ty == crate::affine::AFFINE4 {
                    2304
                } else {
                    pw.ty
                },
                1f32.to_bits(),
            ],
            [(2304 * E).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        let norm = |base: &str, width| -> Result<Norm> {
            Ok(Norm {
                w: vector(d, load(&format!("{base}.weight"), &[width])?)?,
                b: vector(d, load(&format!("{base}.bias"), &[width])?)?,
            })
        };
        let linear = |base: &str, k, n| -> Result<Linear> {
            Ok(Linear {
                w: load(&format!("{base}.weight"), &[k, n])?,
                b: vector(d, load(&format!("{base}.bias"), &[n])?)?,
            })
        };
        let mut blocks = Vec::new();
        for i in 0..27 {
            let b = format!("blocks.{i}");
            blocks.push(Block {
                n1: norm(&format!("{b}.norm1"), E)?,
                qkv: linear(&format!("{b}.attn.qkv"), E, 3 * E)?,
                out: linear(&format!("{b}.attn.proj"), E, E)?,
                n2: norm(&format!("{b}.norm2"), E)?,
                up: linear(&format!("{b}.mlp.linear_fc1"), E, F)?,
                down: linear(&format!("{b}.mlp.linear_fc2"), F, E)?,
            });
        }
        let merger = |base: &str, width| -> Result<Merger> {
            Ok(Merger {
                norm: norm(&format!("{base}.norm"), width)?,
                up: linear(&format!("{base}.linear_fc1"), 4 * E, 4 * E)?,
                down: linear(&format!("{base}.linear_fc2"), 4 * E, 4096)?,
            })
        };
        Ok(Self {
            patch,
            patch1: None,
            bias,
            pos,
            blocks,
            merger: merger("merger", E)?,
            deepstack: (0..3)
                .map(|i| merger(&format!("deepstack_merger_list.{i}"), 4 * E))
                .collect::<Result<_>>()?,
            mean: [0.5; 3],
            std: [0.5; 3],
        })
    }
    pub fn load(d: &MetalDevice, path: &Path) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(model_error)?;
        let meta = &map.gguf().metadata;
        if meta
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .is_some_and(|v| v != 1e-6)
        {
            return Err(error("unsupported vision norm epsilon"));
        }
        if map.gguf().architecture() != Some("clip")
            || meta.get("clip.projector_type").and_then(Value::as_str) != Some("qwen3vl_merger")
        {
            return Err(error("editing requires the Qwen3-VL vision companion"));
        }
        for (k, v) in [
            ("block_count", 27),
            ("embedding_length", E as u64),
            ("attention.head_count", 16),
            ("patch_size", 16),
        ] {
            if meta
                .get(&format!("clip.vision.{k}"))
                .and_then(Value::as_u64)
                != Some(v)
            {
                return Err(error(format!("unsupported image vision {k}")));
            }
        }
        let Some(Value::Array(flags)) = meta.get("clip.vision.is_deepstack_layers") else {
            return Err(error("editing requires DeepStack taps"));
        };
        let taps: Vec<_> = flags
            .iter()
            .enumerate()
            .filter_map(|(i, v)| matches!(v, Value::Bool(true)).then_some(i))
            .collect();
        if taps != [8, 16, 24] {
            return Err(error("editing requires DeepStack at layers 8, 16, 24"));
        }
        let load = |name: &str, shape: &[usize]| -> Result<Weight> {
            let w = Weight::load(d, &map, name, shape)?;
            if !matches!(w.ty, 0 | 1 | 30) {
                return Err(error("editing GGUF tower must be F32/F16/BF16"));
            }
            Ok(w)
        };
        let norm = |base: &str, width| -> Result<Norm> {
            Ok(Norm {
                w: vector(d, load(&format!("{base}.weight"), &[width])?)?,
                b: vector(d, load(&format!("{base}.bias"), &[width])?)?,
            })
        };
        let linear = |base: &str, k, n| -> Result<Linear> {
            Ok(Linear {
                w: load(&format!("{base}.weight"), &[k, n])?,
                b: vector(d, load(&format!("{base}.bias"), &[n])?)?,
            })
        };
        let mut patch = load("v.patch_embd.weight", &[16, 16, 3, E])?;
        patch.k = 768;
        patch.n = E;
        let mut patch1 = load("v.patch_embd.weight.1", &[16, 16, 3, E])?;
        patch1.k = 768;
        patch1.n = E;
        let mut blocks = Vec::new();
        for i in 0..27 {
            let b = format!("v.blk.{i}");
            blocks.push(Block {
                n1: norm(&format!("{b}.ln1"), E)?,
                qkv: linear(&format!("{b}.attn_qkv"), E, 3 * E)?,
                out: linear(&format!("{b}.attn_out"), E, E)?,
                n2: norm(&format!("{b}.ln2"), E)?,
                up: linear(&format!("{b}.ffn_up"), E, F)?,
                down: linear(&format!("{b}.ffn_down"), F, E)?,
            });
        }
        let mut posw = load("v.position_embd.weight", &[E, 2304])?;
        posw.k = E * 2304;
        posw.n = 1;
        let arr = |name: &str| -> Result<[f32; 3]> {
            let Some(Value::Array(a)) = meta.get(name) else {
                return Err(error(format!("missing {name}")));
            };
            let a = a
                .iter()
                .map(Value::as_f32)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| error(name))?;
            a.try_into().map_err(|_| error(name))
        };
        let mean = arr("clip.vision.image_mean")?;
        let std = arr("clip.vision.image_std")?;
        if mean.iter().chain(&std).any(|v| !v.is_finite()) || std.iter().any(|v| *v <= 0.) {
            return Err(error("invalid vision normalization"));
        }
        Ok(Self {
            patch,
            patch1: Some(patch1),
            bias: vector(d, load("v.patch_embd.bias", &[E])?)?,
            pos: vector(d, posw)?.buffer,
            blocks,
            merger: Merger {
                norm: norm("v.post_ln", E)?,
                up: linear("mm.0", 4 * E, 4 * E)?,
                down: linear("mm.2", 4 * E, 4096)?,
            },
            deepstack: [8, 16, 24]
                .into_iter()
                .map(|i| {
                    let b = format!("v.deepstack.{i}");
                    Ok(Merger {
                        norm: norm(&format!("{b}.norm"), 4 * E)?,
                        up: linear(&format!("{b}.fc1"), 4 * E, 4 * E)?,
                        down: linear(&format!("{b}.fc2"), 4 * E, 4096)?,
                    })
                })
                .collect::<Result<_>>()?,
            mean,
            std,
        })
    }
    pub fn encode(
        &self,
        exec: &Ops,
        rgba: &Buffer,
        width: usize,
        height: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Output> {
        check_cancelled(cancelled)?;
        let d = &exec.device;
        let (pw, ph) = (width / 16, height / 16);
        let rows = pw * ph;
        if width == 0
            || height == 0
            || width > 2752
            || height > 2752
            || !width.is_multiple_of(32)
            || !height.is_multiple_of(32)
            || rgba.len() != width * height * 16
        {
            return Err(error("invalid vision reference dimensions"));
        }
        let alloc = |n| d.alloc(rows * n * 4);
        let x = alloc(E)?;
        let stage = alloc(4 * E)?;
        let delta = alloc(E)?;
        let qkv = alloc(3 * E)?;
        let q = d.alloc((rows + 64) * 1280 * 2)?;
        let k = d.alloc(q.len())?;
        let v = d.alloc(q.len())?;
        let xy = words(
            d,
            &(0..rows)
                .flat_map(|r| {
                    [
                        ((r / 4 % (pw / 2)) * 2 + r % 2) as u32,
                        ((r / 4 / (pw / 2)) * 2 + r % 4 / 2) as u32,
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        let ts = words(
            d,
            &(0..rows)
                .step_by(32)
                .flat_map(|r| [r as u32, (rows - r).min(32) as u32, 0, rows as u32])
                .collect::<Vec<_>>(),
        )?;
        // All tower contractions consume F32 directly; no language-model
        // activation conversion or split-K workspace is needed.
        let gemm = d.alloc(4)?;
        let cmd = d.begin()?;
        let p = [
            width as u32,
            height as u32,
            self.patch.k as u32,
            self.mean[0].to_bits(),
            self.mean[1].to_bits(),
            self.mean[2].to_bits(),
            self.std[0].to_bits(),
            self.std[1].to_bits(),
            self.std[2].to_bits(),
        ];
        cmd.dispatch(
            "qi_vision_patches",
            &[rgba, &stage],
            &p,
            [(rows * self.patch.k).div_ceil(256), 1, 1],
            256,
        );
        project(&self.patch, &cmd, &stage, &x, rows, &gemm);
        if let Some(patch1) = &self.patch1 {
            project(patch1, &cmd, &stage, &delta, rows, &gemm);
            cmd.dispatch(
                "residual",
                &[&x, &delta],
                &[(rows * E) as u32, 1f32.to_bits()],
                [(rows * E).div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.dispatch(
            "qi_vision_position",
            &[&x, &self.bias.buffer, &self.pos],
            &[pw as u32, ph as u32],
            [(rows * E).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        let mut deepstack = Vec::with_capacity(3);
        for (i, b) in self.blocks.iter().enumerate() {
            check_cancelled(cancelled)?;
            let cmd = d.begin()?;
            b.n1.run(&cmd, &x, &stage, rows);
            b.qkv.run(&cmd, &stage, &qkv, rows, &gemm, 0);
            cmd.dispatch(
                "vis_qkv",
                &[&qkv, &xy, &q, &k, &v],
                &[rows as u32],
                [((rows + 64) * 1280).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "vis_attention",
                &[&q, &k, &v, &delta, &ts],
                &[0],
                [16, rows.div_ceil(32), 1],
                64,
            );
            b.out.run(&cmd, &delta, &stage, rows, &gemm, 0);
            cmd.dispatch(
                "residual",
                &[&x, &stage],
                &[(rows * E) as u32, 1f32.to_bits()],
                [(rows * E).div_ceil(256), 1, 1],
                256,
            );
            b.n2.run(&cmd, &x, &delta, rows);
            b.up.run(&cmd, &delta, &stage, rows, &gemm, 1);
            b.down.run(&cmd, &stage, &delta, rows, &gemm, 0);
            cmd.dispatch(
                "residual",
                &[&x, &delta],
                &[(rows * E) as u32, 1f32.to_bits()],
                [(rows * E).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
            if let Some(tap) = [8, 16, 24].iter().position(|at| *at == i) {
                deepstack.push(self.deepstack[tap].run(exec, &x, &stage, &qkv, &gemm, rows)?);
            }
        }
        check_cancelled(cancelled)?;
        let embd = self.merger.run(exec, &x, &stage, &qkv, &gemm, rows)?;
        let bad = words(d, &[0])?;
        let cmd = d.begin()?;
        for out in std::iter::once(&embd).chain(&deepstack) {
            let n = out.len() / 4;
            cmd.dispatch(
                "vis_finite",
                &[out, &bad],
                &[n as u32],
                [n.div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.finish()?;
        if unsafe { bad.read_u32(1)[0] } != 0 {
            return Err(error("nonfinite reference vision features"));
        }
        Ok(Output {
            embd,
            deepstack,
            nx: pw / 2,
            ny: ph / 2,
        })
    }
}
impl Norm {
    fn run(&self, cmd: &Commands<'_>, x: &Buffer, y: &Buffer, rows: usize) {
        cmd.dispatch(
            "vis_ln",
            &[x, &self.w.buffer, &self.b.buffer, y],
            &[self.w.k as u32, 1e-6f32.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
}
impl Linear {
    fn run(
        &self,
        cmd: &Commands<'_>,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
        gemm: &Buffer,
        gelu: u32,
    ) {
        if self.w.ty == crate::affine::AFFINE4 {
            cmd.dispatch(
                "qi_vision_affine",
                &[&self.w.buffer, x, y],
                &[self.w.k as u32, self.w.n as u32, rows as u32],
                [self.w.n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
        } else {
            project(&self.w, cmd, x, y, rows, gemm);
        }
        cmd.dispatch(
            "qi_bias",
            &[y, &self.b.buffer],
            &[(rows * self.w.n) as u32, self.w.n as u32],
            [(rows * self.w.n).div_ceil(256), 1, 1],
            256,
        );
        if gelu != 0 {
            cmd.dispatch(
                "qi_activation",
                &[y],
                &[(rows * self.w.n) as u32, 1],
                [(rows * self.w.n).div_ceil(256), 1, 1],
                256,
            );
        }
    }
}
impl Merger {
    fn run(
        &self,
        exec: &Ops,
        x: &Buffer,
        stage: &Buffer,
        mid: &Buffer,
        gemm: &Buffer,
        rows: usize,
    ) -> Result<Buffer> {
        let out = exec.device.alloc(rows / 4 * 4096 * 4)?;
        let cmd = exec.device.begin()?;
        self.norm.run(&cmd, x, stage, rows * E / self.norm.w.k);
        self.up.run(&cmd, stage, mid, rows / 4, gemm, 1);
        self.down.run(&cmd, mid, &out, rows / 4, gemm, 0);
        cmd.finish()?;
        Ok(out)
    }
}
