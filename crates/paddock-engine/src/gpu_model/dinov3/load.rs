//! Checkpoint -> device planes. `model.safetensors` (465 tensors, all BF16)
//! plus `config.json`, which is not optional: the decoder's depths and widths,
//! the band statistics and the class names live only there.
//!
//! Three kinds of plane come out of this:
//!   - GEMM weights, bf16 -> f16 bit-exactly (`bf16_to_f16_exact` refuses a
//!     plane that would overflow), some of them re-laid first - always on the
//!     raw bf16 words, so a re-lay can never be a rounding;
//!   - everything a fused kernel reads per element (norm weights, biases,
//!     LayerScale lambdas, the token table) widened to f32 exactly;
//!   - the rope table, which is not in the file at all.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::dinov3::Dinov3SegConfig;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::{Block, ConvGn, GpuDinov3Seg, GpuModelError, Norm, RopeTable, Stage, Workspace};
use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::st_load::{bf16_to_f16_exact, bf16_to_f32};

/// torch's `GroupNorm(min(32, w), w)` - the group count is a function of the
/// stage width in the training graph and is not written to the config.
fn groups_for(width: usize) -> usize {
    width.min(32)
}

/// GroupNorm's eps in the training graph: torch's default, never overridden.
pub(super) const GN_EPS: f32 = 1e-5;

struct Reader<'a> {
    st: &'a ShardedSafetensors,
    exec: &'a GpuExecutor,
    bytes: u64,
}

impl Reader<'_> {
    /// Raw bf16 words of a tensor whose shape must match exactly.
    fn words(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, GpuModelError> {
        let (t, b) = self
            .st
            .bytes(name)
            .ok_or_else(|| GpuModelError::MissingMeta(format!("dinov3 tensor {name}")))?;
        if t.dtype != StDtype::Bf16 || t.shape != shape {
            return Err(GpuModelError::Unsupported(format!(
                "dinov3 {name}: {:?} {:?} (want BF16 {shape:?})",
                t.dtype, t.shape
            )));
        }
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
    }

    /// A per-element plane, widened to f32 exactly.
    fn f32s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, GpuModelError> {
        let w = self.words(name, shape)?;
        let raw: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok(bf16_to_f32(&raw))
    }

    fn dev(&mut self, host: &[f32]) -> Result<CudaSlice<f32>, GpuModelError> {
        self.bytes += (host.len() * 4) as u64;
        Ok(self.exec.to_device(host)?)
    }

    fn vec(&mut self, name: &str, n: usize) -> Result<CudaSlice<f32>, GpuModelError> {
        let v = self.f32s(name, &[n])?;
        self.dev(&v)
    }

    fn norm(&mut self, prefix: &str, n: usize) -> Result<Norm, GpuModelError> {
        Ok(Norm {
            w: self.vec(&format!("{prefix}.weight"), n)?,
            b: self.vec(&format!("{prefix}.bias"), n)?,
        })
    }

    /// bf16 words already laid `[out][in]` row-major -> an f16 GEMM plane.
    fn plane(
        &mut self,
        words: &[u16],
        in_dim: usize,
        out_dim: usize,
        what: &str,
    ) -> Result<HalfTensor, GpuModelError> {
        debug_assert_eq!(words.len(), in_dim * out_dim);
        // the mma GEMM stages 16-byte f16 rows; a ragged in_dim would drop to
        // the portable fallback silently and cost this tower its throughput
        if !in_dim.is_multiple_of(8) {
            return Err(GpuModelError::Unsupported(format!(
                "dinov3 {what}: GEMM input width {in_dim} is not a multiple of 8"
            )));
        }
        let raw: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
        let h = bf16_to_f16_exact(&raw, what)?;
        self.bytes += (h.len() * 2) as u64;
        Ok(HalfTensor {
            buf: self.exec.f16_to_device(&h)?,
            dims: vec![in_dim, out_dim],
        })
    }

    /// `nn.Linear` / 1x1 conv: `[out, in(,1,1)]` is already the GEMM layout.
    fn linear(
        &mut self,
        name: &str,
        shape: &[usize],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<HalfTensor, GpuModelError> {
        let w = self.words(name, shape)?;
        self.plane(&w, in_dim, out_dim, name)
    }

    /// 3x3 conv `[out][in][ky][kx]` -> `[out][ky][kx][in]`: the im2row is
    /// TAP-outer (col = (ky*3+kx)*C + c) so its loads run contiguous, and the
    /// weight is re-laid once here to match.
    fn conv3(&mut self, name: &str, c: usize) -> Result<HalfTensor, GpuModelError> {
        let w = self.words(name, &[c, c, 3, 3])?;
        let mut out = vec![0u16; w.len()];
        for o in 0..c {
            for i in 0..c {
                for t in 0..9 {
                    out[(o * 9 + t) * c + i] = w[(o * c + i) * 9 + t];
                }
            }
        }
        self.plane(&out, 9 * c, c, name)
    }

    /// 2x2 / stride-2 transposed conv, torch layout `[in][out][ky][kx]`, as the
    /// GEMM it is: output row `(ky*2+kx)*C_out + co`, column `ci`. Each input
    /// pixel produces its four output pixels' worth in one row.
    fn convt2(&mut self, name: &str, cin: usize, cout: usize) -> Result<HalfTensor, GpuModelError> {
        let w = self.words(name, &[cin, cout, 2, 2])?;
        let mut out = vec![0u16; w.len()];
        for ci in 0..cin {
            for co in 0..cout {
                for t in 0..4 {
                    out[(t * cout + co) * cin + ci] = w[(ci * cout + co) * 4 + t];
                }
            }
        }
        self.plane(&out, cin, 4 * cout, name)
    }
}

impl GpuDinov3Seg {
    /// Load from a checkpoint directory. `max_batch` sizes the resident
    /// workspace - the most chips one pass will ever be handed.
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_batch: usize,
    ) -> Result<Self, GpuModelError> {
        let cfg = Dinov3SegConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("dinov3 config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("dinov3 safetensors: {e}")))?;
        if !exec.has_dense_pred() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates the dense-prediction lane (slots 610-617) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        if !exec.has_f16_gemm() {
            return Err(GpuModelError::Unsupported(
                "dinov3 needs the f16 tensor-core GEMM (slot 383), which this pack lacks".into(),
            ));
        }
        if !exec.has_dense_pred_h() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates the half activation interface (slots 618-623) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        // The half interface's own geometry: the GEMM ring stages 16-byte units
        // (every backbone in_dim a multiple of 8) and the f16 attention moves 8
        // dims a step. Every DINOv3 size satisfies both; a config that does not
        // is refused here, by name, instead of failing a launch mid-pass.
        let hd = cfg.head_dim();
        if cfg.hidden % 8 != 0
            || cfg.intermediate % 8 != 0
            || hd % 8 != 0
            || !(16..=128).contains(&hd)
        {
            return Err(GpuModelError::Unsupported(format!(
                "dinov3: hidden {} / intermediate {} / head_dim {hd} - the backbone needs \
                 hidden and intermediate in multiples of 8 and a head_dim that is a multiple \
                 of 8 in 16..=128",
                cfg.hidden, cfg.intermediate
            )));
        }
        // Where the f16-landing GEMM is not the device's elected route the
        // backbone keeps the f32 landing and converts - one more plane.
        let h_landing = exec.f16_gemm_h_elected();
        // A pack from before slot 624 keeps the seam pass; same numbers to the
        // golden pixel, one more walk over the FFN plane.
        let fuse_gelu = h_landing && exec.has_f16_gemm_h_gelu();
        let max_batch = max_batch.max(1);

        // Admission: weights AND the workspace, because here the workspace is
        // the bigger half at any useful batch and it is just as resident.
        let file_bytes: u64 = std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        let ws_bytes = Workspace::bytes_for(&cfg, max_batch, h_landing);
        exec.vram_load_gate(file_bytes + ws_bytes, "tic-forestry (dinov3)")
            .map_err(GpuModelError::WontFit)?;
        // single-stream engine - must precede every alloc
        exec.disable_event_tracking();

        let (d, ffn, p, ch) = (cfg.hidden, cfg.intermediate, cfg.patch, cfg.channels);
        let tokens = cfg.tokens();
        let grid = cfg.grid();
        let mut r = Reader {
            st: &st,
            exec: &exec,
            bytes: 0,
        };

        // ---- stem ----
        // [d, ch, p, p] flattens to the im2row column order as it stands
        let patch_w = r.linear(
            "backbone.embeddings.patch_embeddings.weight",
            &[d, ch, p, p],
            ch * p * p,
            d,
        )?;
        let patch_b = r.f32s("backbone.embeddings.patch_embeddings.bias", &[d])?;
        let cls = r.f32s("backbone.embeddings.cls_token", &[1, 1, d])?;
        let reg = r.f32s(
            "backbone.embeddings.register_tokens",
            &[1, cfg.n_registers, d],
        )?;
        // patches first, then class, then registers (see the module note)
        let mut embed_add = Vec::with_capacity(tokens * d);
        for _ in 0..grid * grid {
            embed_add.extend_from_slice(&patch_b);
        }
        embed_add.extend_from_slice(&cls);
        embed_add.extend_from_slice(&reg);
        let embed_add = r.dev(&embed_add)?;

        let rope = RopeTable::new(grid, cfg.head_dim(), cfg.rope_theta);
        let rope_cos = r.dev(&rope.cos)?;
        let rope_sin = r.dev(&rope.sin)?;

        // ---- blocks ----
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let l = |s: &str| format!("backbone.model.layer.{i}.{s}");
            let mut qkv = r.words(&l("attention.q_proj.weight"), &[d, d])?;
            qkv.extend(r.words(&l("attention.k_proj.weight"), &[d, d])?);
            qkv.extend(r.words(&l("attention.v_proj.weight"), &[d, d])?);
            // k_proj carries no bias in a DINOv3 checkpoint. If one ever shows
            // up the graph below would silently drop it, so say so instead.
            if st.bytes(&l("attention.k_proj.bias")).is_some() {
                return Err(GpuModelError::Unsupported(format!(
                    "dinov3 layer {i}: k_proj has a bias, which this graph does not apply"
                )));
            }
            blocks.push(Block {
                ln1: r.norm(&l("norm1"), d)?,
                wqkv: r.plane(&qkv, d, 3 * d, &l("attention.qkv"))?,
                bq: r.vec(&l("attention.q_proj.bias"), d)?,
                bv: r.vec(&l("attention.v_proj.bias"), d)?,
                wo: r.linear(&l("attention.o_proj.weight"), &[d, d], d, d)?,
                bo: r.vec(&l("attention.o_proj.bias"), d)?,
                ls1: r.vec(&l("layer_scale1.lambda1"), d)?,
                ln2: r.norm(&l("norm2"), d)?,
                up: r.linear(&l("mlp.up_proj.weight"), &[ffn, d], d, ffn)?,
                up_b: r.vec(&l("mlp.up_proj.bias"), ffn)?,
                down: r.linear(&l("mlp.down_proj.weight"), &[d, ffn], ffn, d)?,
                down_b: r.vec(&l("mlp.down_proj.bias"), d)?,
                ls2: r.vec(&l("layer_scale2.lambda1"), d)?,
            });
        }

        // ---- decoder ----
        let mut stages = Vec::with_capacity(cfg.widths.len());
        for (i, &w) in cfg.widths.iter().enumerate() {
            let project = r.linear(&format!("head.project.{i}.weight"), &[w, d, 1, 1], d, w)?;
            let pb = r.f32s(&format!("head.project.{i}.bias"), &[w])?;
            let (project_b, up, up_b) = if i == 0 {
                (Some(r.dev(&pb)?), None, None)
            } else {
                let prev = cfg.widths[i - 1];
                let up = r.convt2(&format!("head.up.{}.weight", i - 1), prev, w)?;
                let ub = r.f32s(&format!("head.up.{}.bias", i - 1), &[w])?;
                let sum: Vec<f32> = ub.iter().zip(&pb).map(|(a, b)| a + b).collect();
                (None, Some(up), Some(r.dev(&sum)?))
            };
            // blend = Sequential(conv, gn, gelu, conv, gn, gelu): 0,1 and 3,4
            let mut cg = |ci: usize, gi: usize| -> Result<ConvGn, GpuModelError> {
                Ok(ConvGn {
                    w: r.conv3(&format!("head.blend.{i}.{ci}.weight"), w)?,
                    b: r.vec(&format!("head.blend.{i}.{ci}.bias"), w)?,
                    gn: r.norm(&format!("head.blend.{i}.{gi}"), w)?,
                    groups: groups_for(w),
                })
            };
            let blend = [cg(0, 1)?, cg(3, 4)?];
            stages.push(Stage {
                width: w,
                project,
                project_b,
                up,
                up_b,
                blend,
            });
        }

        // ---- heads, stacked: class rows then the height row ----
        let last = *cfg.widths.last().expect("config check: at least one stage");
        let ncls = cfg.n_classes;
        let mut ow = r.words("head.head_class.weight", &[ncls, last, 1, 1])?;
        ow.extend(r.words("head.head_height.weight", &[1, last, 1, 1])?);
        let out_w = r.plane(&ow, last, ncls + 1, "head.heads")?;
        let mut ob = r.f32s("head.head_class.bias", &[ncls])?;
        ob.extend(r.f32s("head.head_height.bias", &[1])?);
        let out_b = r.dev(&ob)?;

        let weight_bytes = r.bytes;
        let ws = Workspace::new(&exec, &cfg, max_batch, h_landing)?;
        tracing::info!(
            layers = cfg.n_layer,
            hidden = d,
            tokens,
            bands = ch,
            stages = cfg.widths.len(),
            max_batch,
            weights_mib = weight_bytes >> 20,
            workspace_mib = ws.bytes >> 20,
            "dinov3 dense-prediction model resident"
        );
        Ok(Self {
            exec,
            cfg,
            patch_w,
            embed_add,
            rope_cos,
            rope_sin,
            blocks,
            stages,
            out_w,
            out_b,
            fuse_gelu,
            ws,
            weight_bytes,
        })
    }
}

impl Workspace {
    /// Element counts per chip, in one place so the admission estimate and the
    /// allocation cannot drift apart.
    fn plan(cfg: &Dinov3SegConfig) -> Plan {
        let t = cfg.tokens();
        let d = cfg.hidden;
        let g2 = cfg.grid() * cfg.grid();
        // stage i runs at (grid << i)^2 pixels and widths[i] channels
        let stage_px = |i: usize| g2 << (2 * i);
        let plane = (0..cfg.widths.len())
            .map(|i| stage_px(i) * cfg.widths[i])
            .max()
            .unwrap_or(0);
        let im2row = (0..cfg.widths.len())
            .map(|i| stage_px(i) * 9 * cfg.widths[i])
            .max()
            .unwrap_or(0);
        let part = (0..cfg.widths.len())
            .map(|i| GpuExecutor::dp_group_norm_part_len(1, stage_px(i), cfg.widths[i]))
            .max()
            .unwrap_or(0);
        let out_px = cfg.out_size * cfg.out_size;
        Plan {
            px: cfg.image_size * cfg.image_size * cfg.channels,
            // also the attention landing and the tap's f16 view, one row plane
            s16: im2row
                .max(t * d)
                .max(t * cfg.channels * cfg.patch * cfg.patch),
            row: t * d,
            ff: t * cfg.intermediate,
            taps: cfg.widths.iter().map(|w| t * w).collect(),
            plane,
            part,
            stat: 2 * cfg.widths.iter().map(|w| groups_for(*w)).max().unwrap_or(1),
            o: out_px * (cfg.n_classes + 1),
            out_px,
        }
    }

    /// `h_landing` false adds the f32 plane a non-elected device lands its
    /// backbone GEMMs on before converting (the widest one: the ffn's).
    pub(super) fn bytes_for(cfg: &Dinov3SegConfig, chips: usize, h_landing: bool) -> u64 {
        let p = Self::plan(cfg);
        // f32: the residual stream, then the decoder's planes
        let f32s = p.row
            + if h_landing { 0 } else { p.ff }
            + p.taps.iter().sum::<usize>()
            + 3 * p.plane
            + p.part
            + p.stat
            + p.o
            + p.out_px;
        // f16: n16, the three-wide qkv landing, q, k, v, proj - eight row
        // planes - and the ffn plane
        let f16s = p.s16 + 8 * p.row + p.ff + p.plane;
        (chips * (f32s * 4 + f16s * 2 + p.px + p.out_px)) as u64
    }

    fn new(
        exec: &GpuExecutor,
        cfg: &Dinov3SegConfig,
        cap: usize,
        h_landing: bool,
    ) -> Result<Self, GpuModelError> {
        let p = Self::plan(cfg);
        let f = |n: usize| exec.alloc(cap * n);
        let h = |n: usize| exec.alloc_f16(cap * n);
        Ok(Self {
            cap,
            px: exec.alloc_u8(cap * p.px)?,
            s16: h(p.s16)?,
            n16: h(p.row)?,
            x: f(p.row)?,
            qkv: h(3 * p.row)?,
            q: h(p.row)?,
            k: h(p.row)?,
            v: h(p.row)?,
            proj: h(p.row)?,
            ff: h(p.ff)?,
            land32: if h_landing { None } else { Some(f(p.ff)?) },
            taps: p.taps.iter().map(|n| f(*n)).collect::<Result<_, _>>()?,
            ya: f(p.plane)?,
            xa: f(p.plane)?,
            g: f(p.plane)?,
            h16: h(p.plane)?,
            gn_part: f(p.part)?,
            gn_stat: f(p.stat)?,
            o: f(p.o)?,
            cls: exec.alloc_u8(cap * p.out_px)?,
            height: f(p.out_px)?,
            logits: None,
            bytes: Self::bytes_for(cfg, cap, h_landing),
        })
    }
}

struct Plan {
    px: usize,
    s16: usize,
    row: usize,
    ff: usize,
    taps: Vec<usize>,
    plane: usize,
    part: usize,
    stat: usize,
    o: usize,
    out_px: usize,
}
