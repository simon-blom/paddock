//! Native affine4 text + BF16 vision ingestion. No GGUF transcode, framework
//! dependency, persistent dequantized language weights or host tensor math.
use super::*;
use paddock_models::{
    mlx::{MultimodalConfig, MultimodalFamily},
    safetensors::{ShardedSafetensors, StDtype},
};
use std::{cell::RefCell, collections::HashSet};

pub(super) struct Source {
    map: ShardedSafetensors,
    used: RefCell<HashSet<String>>,
}
impl Source {
    pub(super) fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            map: ShardedSafetensors::open_dir(path)
                .map_err(|e| MetalError::Model(e.to_string()))?,
            used: RefCell::new(HashSet::new()),
        })
    }
    pub(super) fn bytes(&self) -> u64 {
        self.map.total_len()
    }
    pub(super) fn packed(
        &self,
        d: &MetalDevice,
        base: &str,
        k: usize,
        shape: &[usize],
        bits: usize,
    ) -> Result<Weight> {
        if !matches!(bits, 4 | 8)
            || k == 0
            || !k.is_multiple_of(64)
            || shape.is_empty()
            || shape.contains(&0)
        {
            return Err(MetalError::Model(format!(
                "{base}: invalid packed shape/bits"
            )));
        }
        let n = shape
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or_else(|| MetalError::Model("affine shape overflow".into()))?;
        let mut parts = Vec::new();
        for (suffix, dtype, last) in [
            ("weight", StDtype::U32, k / (32 / bits)),
            ("scales", StDtype::Bf16, k / 64),
            ("biases", StDtype::Bf16, k / 64),
        ] {
            let name = format!("{base}.{suffix}");
            let (info, bytes) = self
                .map
                .bytes(&name)
                .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
            let mut expected = shape.to_vec();
            expected.push(last);
            if info.dtype != dtype || info.shape != expected {
                return Err(MetalError::Model(format!(
                    "{name}: expected {dtype:?} {expected:?}"
                )));
            }
            if suffix != "weight"
                && bytes
                    .chunks_exact(2)
                    .any(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x7f80) == 0x7f80)
            {
                return Err(MetalError::Model(format!(
                    "{name}: nonfinite quantization parameters"
                )));
            }
            parts.push(bytes);
            self.used.borrow_mut().insert(name);
        }
        Ok(Weight {
            buffer: d.upload_parts(&parts)?,
            ty: if bits == 4 { 0x100 } else { 0x108 },
            k,
            n,
        })
    }
    pub(super) fn affine(&self, d: &MetalDevice, base: &str, k: usize, n: usize) -> Result<Weight> {
        let w = crate::affine::load(d, &self.map, &format!("{base}.weight"), k, n)?;
        for suffix in ["weight", "scales", "biases"] {
            self.used.borrow_mut().insert(format!("{base}.{suffix}"));
        }
        Ok(w)
    }
    pub(super) fn raw(
        &self,
        d: &MetalDevice,
        name: &str,
        shape: &[usize],
        f32_out: bool,
    ) -> Result<Weight> {
        let (info, data) = self
            .map
            .bytes(name)
            .ok_or_else(|| MetalError::Model(format!("missing MLX tensor {name}")))?;
        if info.shape != shape
            || !matches!(info.dtype, StDtype::Bf16 | StDtype::F32)
            || (!f32_out && info.dtype != StDtype::Bf16)
        {
            return Err(MetalError::Model(format!(
                "{name}: unsupported dtype/shape {:?} {:?}, expected {shape:?}",
                info.dtype, info.shape
            )));
        }
        let source = d.upload_parts(&[data])?;
        let count: usize = shape.iter().product();
        let ty = if f32_out { 0 } else { 30 };
        let buffer = d.alloc(count * if f32_out { 4 } else { 2 })?;
        let bad = d.upload(&0u32.to_le_bytes())?;
        let cmd = d.begin()?;
        cmd.dispatch(
            "vis_cast",
            &[&source, &buffer, &bad],
            &[
                count as u32,
                if info.dtype == StDtype::Bf16 { 30 } else { 0 },
                ty,
            ],
            [count.div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        if unsafe { bad.read_u32(1)[0] } != 0 {
            return Err(MetalError::Model(format!("nonfinite MLX tensor {name}")));
        }
        self.used.borrow_mut().insert(name.to_owned());
        Ok(Weight {
            buffer,
            ty: if f32_out { 0 } else { 0x101 },
            k: *shape.last().unwrap_or(&1),
            n: if shape.len() == 2 { shape[0] } else { 1 },
        })
    }
    fn norm(&self, d: &MetalDevice, base: &str, n: usize, centered: bool) -> Result<Weight> {
        let w = self.raw(d, &format!("{base}.weight"), &[n], true)?;
        if centered {
            let cmd = d.begin()?;
            cmd.dispatch(
                "gmlx_centered_weights",
                &[&w.buffer],
                &[n as u32],
                [n.div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
        }
        Ok(w)
    }
    pub(super) fn finish(&self) -> Result<()> {
        let used = self.used.borrow();
        let mut extra: Vec<_> = self.map.names().filter(|n| !used.contains(*n)).collect();
        extra.sort();
        if !extra.is_empty() {
            return Err(MetalError::Model(format!(
                "unconsumed MLX tensors (refusing an unknown graph): {:?}",
                &extra[..extra.len().min(8)]
            )));
        }
        Ok(())
    }
}

/// BF16 tower matrices have a separate arithmetic tag from GGUF BF16: both
/// operands and the result round at the checkpoint's MLX operation boundary.
pub(super) fn dense(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    out: &Buffer,
    bias: Option<&Weight>,
    rows: usize,
    epilogue: u32,
) {
    assert_eq!(w.ty, 0x101);
    cmd.dispatch(
        "gmlx_bmm",
        &[&w.buffer, x, out, bias.map_or(&w.buffer, |b| &b.buffer)],
        &[
            w.k as u32,
            w.n as u32,
            rows as u32,
            if epilogue == 0 {
                u32::from(bias.is_some())
            } else {
                epilogue
            },
        ],
        [w.n.div_ceil(32), rows.div_ceil(32), 1],
        128,
    );
}

impl Gemma4 {
    pub(super) fn load_mlx(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let cfg = MultimodalConfig::read(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let muse = cfg.family == MultimodalFamily::Muse30;
        if context == 0 || context > cfg.context || max_batch == 0 || max_batch >= CHUNK {
            return Err(MetalError::Model(
                "MLX context exceeds training limit, or batch outside 1..512".into(),
            ));
        }
        let source = Source {
            map: ShardedSafetensors::open_dir(path)
                .map_err(|e| MetalError::Model(e.to_string()))?,
            used: RefCell::new(HashSet::new()),
        };
        let (width, ff, count, vocab, window, capacity) = if muse {
            (6656, 19968, 52, 202048, 2048, muse::IMAGE_CHUNK)
        } else {
            (5376, 21504, 60, 262144, 1024, CHUNK)
        };
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let ring = context
            .min(window + capacity)
            .next_multiple_of(BLOCK_TOKENS);
        let blocks = page_stride
            .checked_mul(max_batch * 2 + 1)
            .filter(|&n| n <= u32::MAX as usize / BLOCK_TOKENS)
            .ok_or_else(|| MetalError::Memory("MLX KV size overflow".into()))?;
        let (global_width, sliding_width, global_layers) = if muse {
            (256, 256, 13)
        } else {
            (2048, 4096, 10)
        };
        let global_bytes = blocks * BLOCK_TOKENS * global_width * 2;
        let sliding_bytes = ring * max_batch * 2 * sliding_width * 2;
        let kv_bytes =
            (global_bytes * global_layers * 2 + sliding_bytes * (count - global_layers) * 2) as u64;
        let q_width = if muse { 4096 } else { 16384 };
        let shapes = [
            (width, ff),
            (ff, width),
            (width, q_width),
            (q_width, width),
            (width, if muse { 4096 } else { 8192 }),
            (if muse { 4096 } else { 8192 }, width),
            (width, global_width),
            (width, sliding_width),
            (width, vocab),
        ];
        let gemm_bytes = shapes
            .into_iter()
            .map(|(k, n)| crate::affine::workspace_bytes(k, n, capacity))
            .max()
            .expect("nonempty projection shape inventory");
        let part_hd = if muse { 128 } else { 512 };
        let scratch_bytes =
            (capacity * (width * 5 + q_width * 3 + sliding_width * 2 + ff * 2 + 9) * 4
                + max_batch * (page_stride + vocab + HEADS * SPLITS * (part_hd + 2)) * 4
                + gemm_bytes) as u64;
        let device = MetalDevice::new(budget)?;
        // Source bytes cover the embedded tower too. Vectors expand to F32,
        // while large matrix weights remain packed/BF16. Reserve that bounded
        // overhead before any upload; MetalDevice enforces the exact grant.
        let required = source
            .map
            .total_len()
            .saturating_add(kv_bytes)
            .saturating_add(scratch_bytes)
            .saturating_add(256 << 20);
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "MLX text + vision + bounded KV/scratch need at most {:.2} GiB, grant {:.2} GiB",
                required as f64 / (1u64 << 30) as f64,
                device.budget_bytes() as f64 / (1u64 << 30) as f64
            )));
        }
        let embedding =
            source.affine(&device, "language_model.model.embed_tokens", width, vocab)?;
        let output = if muse {
            Some(source.affine(&device, "language_model.lm_head", width, vocab)?)
        } else {
            None
        };
        let output_norm = source.norm(&device, "language_model.model.norm", width, false)?;
        // MLX RoPE is generated from metadata on GPU; no GGUF-specific table.
        let factors = Weight {
            buffer: device.alloc(4)?,
            ty: 0,
            k: 1,
            n: 1,
        };
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let sliding = (i + 1) % if muse { 4 } else { 6 } != 0;
            let (hd, kh) = if muse {
                (128, 2)
            } else if sliding {
                (256, 16)
            } else {
                (512, 4)
            };
            let prefix = format!("language_model.model.layers.{i}");
            let mat = |name: &str, k, n| source.affine(&device, &format!("{prefix}.{name}"), k, n);
            let norm = |name: &str, n, centered| {
                source.norm(&device, &format!("{prefix}.{name}"), n, centered)
            };
            let scale = if muse {
                1.
            } else {
                let w = source.raw(&device, &format!("{prefix}.layer_scalar"), &[1], true)?;
                unsafe { w.buffer.read_f32(0, 1)[0] }
            };
            let dummy_norm = || -> Result<Weight> {
                Ok(Weight {
                    buffer: device.alloc(4)?,
                    ty: 0,
                    k: 1,
                    n: 1,
                })
            };
            let bytes = if sliding { sliding_bytes } else { global_bytes };
            layers.push(Layer {
                heads: HEADS,
                moe: None,
                norm: norm("input_layernorm", width, muse)?,
                post_attn: norm("post_attention_layernorm", width, muse)?,
                ffn_norm: norm("pre_feedforward_layernorm", width, muse)?,
                post_ffn: norm("post_feedforward_layernorm", width, muse)?,
                q: mat("self_attn.q_proj", width, HEADS * hd)?,
                k: mat("self_attn.k_proj", width, kh * hd)?,
                v: if muse || sliding {
                    Some(mat("self_attn.v_proj", width, kh * hd)?)
                } else {
                    None
                },
                o: mat("self_attn.o_proj", HEADS * hd, width)?,
                attn_gate: if muse {
                    Some(mat("self_attn.gate_proj", width, HEADS * hd)?)
                } else {
                    None
                },
                q_norm: if muse {
                    dummy_norm()?
                } else {
                    norm("self_attn.q_norm", hd, false)?
                },
                k_norm: if muse {
                    dummy_norm()?
                } else {
                    norm("self_attn.k_norm", hd, false)?
                },
                gate: mat("mlp.gate_proj", width, ff)?,
                up: mat("mlp.up_proj", width, ff)?,
                down: mat("mlp.down_proj", ff, width)?,
                scale,
                sliding,
                keys: device.alloc(bytes)?,
                values: device.alloc(bytes)?,
            });
        }
        let vision = if muse {
            tower::Tower::Muse(Box::new(muse_vision::Vision::load_mlx(&device, &source)?))
        } else {
            tower::Tower::Gemma(Box::new(vision::Vision::load_mlx(&device, &source)?))
        };
        source.finish()?;
        let weight_bytes = device.allocated_bytes() - kv_bytes;
        let a = |n| device.alloc(capacity * n * 4);
        let scratch = Scratch {
            rows: capacity,
            ids: a(1)?,
            meta: a(2)?,
            limits: a(1)?,
            pages: device.alloc(max_batch * page_stride * 4)?,
            outputs: a(1)?,
            tiles: a(2)?,
            decode_rows: a(1)?,
            x: a(width)?,
            norm: a(width)?,
            delta: a(width)?,
            q: a(q_width)?,
            k: a(sliding_width)?,
            v: a(sliding_width)?,
            attn: a(q_width)?,
            attn_gate: a(q_width)?,
            parts: device.alloc(max_batch * HEADS * SPLITS * (part_hd + 2) * 4)?,
            gate: a(ff)?,
            up: a(ff)?,
            gemm: device.alloc(gemm_bytes)?,
            logits: device.alloc(max_batch * vocab * 4)?,
        };
        // Marker ids are read from the checkpoint tokenizer, not inferred
        // from an artifact directory name. The runner uses this same vocab.
        let tokenizer: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("tokenizer.json"))
                .map_err(|e| MetalError::Model(e.to_string()))?,
        )
        .map_err(|e| MetalError::Model(e.to_string()))?;
        let marker = |text| {
            tokenizer["added_tokens"]
                .as_array()
                .and_then(|a| a.iter().find(|v| v["content"] == text))
                .and_then(|v| v["id"].as_u64())
                .and_then(|n| u32::try_from(n).ok())
        };
        let image_markers = if muse {
            marker("<|image_start|>").zip(marker("<|image_end|>"))
        } else {
            marker("<|image>").zip(marker("<image|>"))
        };
        if image_markers.is_none() {
            return Err(MetalError::Model(
                "MLX tokenizer lacks image markers".into(),
            ));
        }
        Ok(Self {
            diffusion: None,
            muse,
            mlx: true,
            device,
            embedding,
            output,
            output_norm,
            factors,
            layers,
            scratch,
            moe_scratch: None,
            mtp: None,
            dflash: None,
            vision: Some(vision),
            image_markers,
            image_cache: Vec::new(),
            image_cache_reused: 0,
            encoding: VecDeque::new(),
            spec: None,
            verifying: false,
            greedy_verify: false,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            cache: (0..max_batch).map(|_| Checkpoint::default()).collect(),
            clock: 0,
            pending: VecDeque::new(),
            prefill_phase: None,
            pool: KvPool::with_blocks(blocks as u32),
            width,
            ff,
            vocab,
            context,
            window,
            ring,
            page_stride,
            eps: cfg.eps,
            rope: cfg.rope,
            softcap: cfg.softcap,
            logit_scale: cfg.logit_scale,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
