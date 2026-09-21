use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
impl Vision {
    pub(in crate::unlimited_ocr) fn load(d: &MetalDevice, path: &Path) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        if map.gguf().architecture() != Some("clip")
            || meta.get("clip.projector_type").and_then(Value::as_str) != Some("deepseekocr")
            || !matches!(meta.get("clip.has_vision_encoder"), Some(Value::Bool(true)))
            || !matches!(meta.get("clip.use_gelu"), Some(Value::Bool(true)))
            || meta
                .get("clip.vision.attention.layer_norm_epsilon")
                .and_then(Value::as_f32)
                != Some(1e-6)
        {
            return Err(error("requires Unlimited-OCR DeepEncoder companion"));
        }
        for (key, want) in [
            ("block_count", 24),
            ("embedding_length", 1024),
            ("attention.head_count", 16),
            ("sam.block_count", 12),
            ("sam.embedding_length", 768),
            ("sam.head_count", 12),
            ("window_size", 14),
            ("preproc_min_tiles", 2),
            ("preproc_max_tiles", 32),
            ("projection_dim", 1280),
            ("projector.scale_factor", 1),
        ] {
            if meta
                .get(&format!("clip.vision.{key}"))
                .and_then(Value::as_u64)
                != Some(want)
            {
                return Err(error(format!("unsupported tower {key}")));
            }
        }
        for key in ["image_mean", "image_std"] {
            if !matches!(meta.get(&format!("clip.vision.{key}")),Some(Value::Array(a)) if a.len()==3 && a.iter().all(|x|x.as_f32()==Some(0.5)))
            {
                return Err(error("unsupported image normalization"));
            }
        }
        let mut schema = Vec::new();
        for (name, shape, ty) in [
            ("v.sam.patch_embd.weight", vec![16, 16, 3, 768], 1),
            ("v.sam.patch_embd.bias", vec![768], 0),
            ("v.sam.pos_embd.weight", vec![768, 64, 64, 1], 0),
            ("v.sam.neck.0.weight", vec![1, 1, 768, 256], 0),
            ("v.sam.neck.2.weight", vec![3, 3, 256, 256], 0),
            ("v.sam.net_2.weight", vec![3, 3, 256, 512], 0),
            ("v.sam.net_3.weight", vec![3, 3, 512, 1024], 0),
            ("v.class_embd", vec![1024], 0),
            ("v.position_embd.weight", vec![1024, 257], 0),
            // Validated but dead: CLIP receives SAM embeddings, never pixels.
            ("v.patch_embd.weight", vec![14, 14, 3, 1024], 0),
            ("mm.model.fc.weight", vec![2048, 1280], 1),
            ("mm.model.fc.bias", vec![1280], 0),
            ("v.image_newline", vec![1280], 0),
            ("v.view_seperator", vec![1280, 1], 0),
        ] {
            schema.push((name.into(), shape, ty));
        }
        for (name, width) in [
            ("v.sam.neck.1", 256),
            ("v.sam.neck.3", 256),
            ("v.pre_ln", 1024),
        ] {
            for suffix in ["weight", "bias"] {
                schema.push((format!("{name}.{suffix}"), vec![width], 0));
            }
        }
        for (sam, layers, e, ff) in [(true, 12, 768, 3072), (false, 24, 1024, 4096)] {
            for i in 0..layers {
                let prefix = if sam {
                    format!("v.sam.blk.{i}")
                } else {
                    format!("v.blk.{i}")
                };
                for name in if sam {
                    ["pre_ln", "post_ln"]
                } else {
                    ["ln1", "ln2"]
                } {
                    for suffix in ["weight", "bias"] {
                        schema.push((format!("{prefix}.{name}.{suffix}"), vec![e], 0));
                    }
                }
                let names = if sam {
                    ["attn.qkv", "attn.out", "mlp.lin1", "mlp.lin2"]
                } else {
                    ["attn_qkv", "attn_out", "ffn_up", "ffn_down"]
                };
                for (name, k, n) in [
                    (names[0], e, 3 * e),
                    (names[1], e, e),
                    (names[2], e, ff),
                    (names[3], ff, e),
                ] {
                    schema.push((format!("{prefix}.{name}.weight"), vec![k, n], 1));
                    schema.push((format!("{prefix}.{name}.bias"), vec![n], 0));
                }
                if sam {
                    for axis in ["h", "w"] {
                        schema.push((
                            format!("{prefix}.attn.pos_{axis}.weight"),
                            vec![64, if i % 3 == 2 { 127 } else { 27 }],
                            0,
                        ));
                    }
                }
            }
        }
        let payload = super::super::load::validate(&map, &schema)? - 14 * 14 * 3 * 1024 * 4;
        if d.allocated_bytes() + payload + Self::workspace_bound() > d.budget_bytes() {
            return Err(MetalError::Memory(
                "Unlimited-OCR tower + workspace exceed grant".into(),
            ));
        }
        let w = |n: &str, s: &[usize]| Weight::load(d, &map, n, s);
        let norm = |n: &str, e: usize| -> Result<Norm> {
            Ok(Norm {
                w: w(&format!("{n}.weight"), &[e])?,
                b: w(&format!("{n}.bias"), &[e])?,
            })
        };
        let linear = |n: &str, k: usize, e: usize| -> Result<Linear> {
            Ok(Linear {
                w: w(&format!("{n}.weight"), &[k, e])?,
                b: w(&format!("{n}.bias"), &[e])?,
            })
        };
        let mut sam = Vec::new();
        let mut clip = Vec::new();
        for (is_sam, layers, e, ff) in [(true, 12, 768, 3072), (false, 24, 1024, 4096)] {
            for i in 0..layers {
                let p = if is_sam {
                    format!("v.sam.blk.{i}")
                } else {
                    format!("v.blk.{i}")
                };
                let names = if is_sam {
                    [
                        "pre_ln", "post_ln", "attn.qkv", "attn.out", "mlp.lin1", "mlp.lin2",
                    ]
                } else {
                    ["ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down"]
                };
                let b = Block {
                    ln1: norm(&format!("{p}.{}", names[0]), e)?,
                    ln2: norm(&format!("{p}.{}", names[1]), e)?,
                    qkv: linear(&format!("{p}.{}", names[2]), e, 3 * e)?,
                    out: linear(&format!("{p}.{}", names[3]), e, e)?,
                    up: linear(&format!("{p}.{}", names[4]), e, ff)?,
                    down: linear(&format!("{p}.{}", names[5]), ff, e)?,
                    relative: if is_sam {
                        Some((
                            w(
                                &format!("{p}.attn.pos_h.weight"),
                                &[64, if i % 3 == 2 { 127 } else { 27 }],
                            )?,
                            w(
                                &format!("{p}.attn.pos_w.weight"),
                                &[64, if i % 3 == 2 { 127 } else { 27 }],
                            )?,
                        ))
                    } else {
                        None
                    },
                };
                if is_sam { sam.push(b) } else { clip.push(b) }
            }
        }
        let conv = |name: &str, taps: usize, cin: usize, cout: usize| -> Result<Weight> {
            let mut x = w(name, &[taps, taps, cin, cout])?;
            x.k = taps * taps * cin;
            x.n = cout;
            Ok(x)
        };
        Ok(Self {
            patch: Linear {
                w: conv("v.sam.patch_embd.weight", 16, 3, 768)?,
                b: w("v.sam.patch_embd.bias", &[768])?,
            },
            sam_pos: w("v.sam.pos_embd.weight", &[768, 64, 64, 1])?,
            sam,
            clip,
            neck0: conv("v.sam.neck.0.weight", 1, 768, 256)?,
            neck1: norm("v.sam.neck.1", 256)?,
            neck2: conv("v.sam.neck.2.weight", 3, 256, 256)?,
            neck3: norm("v.sam.neck.3", 256)?,
            net2: conv("v.sam.net_2.weight", 3, 256, 512)?,
            net3: conv("v.sam.net_3.weight", 3, 512, 1024)?,
            cls: w("v.class_embd", &[1024])?,
            clip_pos: w("v.position_embd.weight", &[1024, 257])?,
            pre: norm("v.pre_ln", 1024)?,
            projector: linear("mm.model.fc", 2048, 1280)?,
            nl: w("v.image_newline", &[1280])?,
            separator: w("v.view_seperator", &[1280, 1])?,
        })
    }
}
