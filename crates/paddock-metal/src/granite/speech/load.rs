use super::*;
impl Tower {
    pub(super) fn load(d: &MetalDevice, path: &Path, plus: bool) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        if map.gguf().architecture() != Some("clip")
            || meta.get("clip.projector_type").and_then(Value::as_str) != Some("granite_speech")
            || !matches!(meta.get("clip.has_audio_encoder"), Some(Value::Bool(true)))
            || meta
                .get("clip.audio.attention.layer_norm_epsilon")
                .and_then(Value::as_f32)
                != Some(1e-5)
        {
            return Err(error("requires Granite Speech F16/F32 companion"));
        }
        for (key, n) in [
            ("projection_dim", 2048),
            ("embedding_length", E),
            ("feed_forward_length", F),
            ("block_count", 16),
            ("attention.head_count", 8),
            ("num_mel_bins", 160),
            ("chunk_size", 200),
            ("conv_kernel_size", 15),
            ("max_pos_emb", 512),
            ("projector.window_size", 15),
            ("projector.downsample_rate", 5),
            ("projector.head_count", 16),
        ] {
            if meta
                .get(&format!("clip.audio.{key}"))
                .and_then(Value::as_u64)
                != Some(n as u64)
            {
                return Err(error(format!("unsupported {key}")));
            }
        }
        let features = match meta.get("clip.audio.feature_layer") {
            None => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_u64().ok_or_else(|| error("invalid feature layer")))
                .collect::<Result<Vec<_>>>()?,
            _ => return Err(error("invalid feature layer")),
        };
        if features != if plus { vec![3] } else { vec![] } {
            return Err(error(
                "base/Plus companion mismatch (expected Plus layer-3 capture only on Plus)",
            ));
        }
        let mut schema = Vec::<(String, Vec<usize>, u32)>::new();
        let mut norm = |name: String, width: usize| {
            for tail in ["weight", "bias"] {
                schema.push((format!("{name}.{tail}"), vec![width], 0));
            }
        };
        for i in 0..16 {
            for n in ["ffn_norm", "ln1", "norm_conv", "ffn_norm_1", "ln2"] {
                norm(format!("a.blk.{i}.{n}"), E);
            }
            norm(format!("a.blk.{i}.conv_norm"), 2048);
        }
        norm("a.proj_norm".into(), E);
        for i in 0..2 {
            for n in ["self_attn_norm", "cross_attn_norm", "ffn_norm"] {
                norm(format!("a.proj_blk.{i}.{n}"), E);
            }
        }
        let mut linear = |name: String, k: usize, n: usize, ty: u32| {
            schema.push((format!("{name}.weight"), vec![k, n], ty));
            schema.push((format!("{name}.bias"), vec![n], 0));
        };
        for (name, k, n) in [
            ("input_projection", 160, E),
            ("enc_ctc_out", E, 348),
            ("enc_ctc_out_mid", 348, E),
            ("proj_linear", E, 2048),
        ] {
            linear(format!("a.{name}"), k, n, 1);
        }
        for i in 0..16 {
            for (name, k, n, ty) in [
                ("ffn_up", E, F, 1),
                ("ffn_down", F, E, 1),
                ("ffn_up_1", E, F, 1),
                ("ffn_down_1", F, E, 1),
                ("attn_out", E, E, 1),
                ("conv_pw1", E, 4096, 0),
                ("conv_pw2", 2048, E, 0),
            ] {
                linear(format!("a.blk.{i}.{name}"), k, n, ty);
            }
        }
        for i in 0..2 {
            for name in [
                "self_attn_q",
                "self_attn_k",
                "self_attn_v",
                "self_attn_out",
                "cross_attn_q",
                "cross_attn_out",
            ] {
                linear(format!("a.proj_blk.{i}.{name}"), E, E, 1);
            }
            for name in ["cross_attn_k", "cross_attn_v"] {
                linear(
                    format!("a.proj_blk.{i}.{name}"),
                    if plus { 2048 } else { E },
                    E,
                    1,
                );
            }
            linear(format!("a.proj_blk.{i}.ffn_up"), E, F, 1);
            linear(format!("a.proj_blk.{i}.ffn_down"), F, E, 1);
        }
        schema.push(("a.proj_query".into(), vec![E, 3, 1], 0));
        for i in 0..16 {
            for name in ["attn_q", "attn_k", "attn_v"] {
                schema.push((format!("a.blk.{i}.{name}.weight"), vec![E, E], 1));
            }
            schema.push((format!("a.blk.{i}.attn_rel_pos_emb"), vec![128, 1025], 0));
            schema.push((format!("a.blk.{i}.conv_dw.weight"), vec![15, 2048], 0));
        }
        if map.gguf().tensors.len() != schema.len() {
            return Err(error("unexpected tensor inventory"));
        }
        let mut bytes = 0u64;
        for (name, dims, ty) in &schema {
            let (info, raw) = map.tensor_bytes(name).map_err(|e| error(e.to_string()))?;
            if info.dims.iter().map(|&n| n as usize).collect::<Vec<_>>() != *dims
                || info.raw_type != *ty
                || raw.len() != dims.iter().product::<usize>() * if *ty == 0 { 4 } else { 2 }
            {
                return Err(error(format!("{name}: incompatible shape/type/span")));
            }
            // Storage validation only. Every actual contraction/nonlinearity
            // runs on the GPU; reject non-finite file values before admission.
            let invalid = if *ty == 0 {
                raw.chunks_exact(4).any(|b| {
                    u32::from_le_bytes(b.try_into().expect("four bytes")) & 0x7f800000 == 0x7f800000
                })
            } else {
                raw.chunks_exact(2).any(|b| {
                    u16::from_le_bytes(b.try_into().expect("two bytes")) & 0x7c00 == 0x7c00
                })
            };
            if invalid {
                return Err(error(format!("{name}: non-finite weight")));
            }
            bytes += raw.len() as u64;
        }
        if d.allocated_bytes() + bytes + WORKSPACE > d.budget_bytes() {
            return Err(MetalError::Memory(
                "Granite Speech weights + workspace exceed grant".into(),
            ));
        }
        let weight = |n: &str, dims: &[usize]| Weight::load(d, &map, n, dims);
        let norm = |n: &str, width: usize| -> Result<Norm> {
            Ok(Norm {
                w: weight(&format!("{n}.weight"), &[width])?,
                b: weight(&format!("{n}.bias"), &[width])?,
            })
        };
        let linear = |n: &str, k: usize, out: usize| -> Result<Linear> {
            Ok(Linear {
                w: weight(&format!("{n}.weight"), &[k, out])?,
                b: weight(&format!("{n}.bias"), &[out])?,
            })
        };
        let mut blocks = Vec::new();
        for i in 0..16 {
            let p = format!("a.blk.{i}");
            let mut qkv = Vec::new();
            for name in ["attn_q", "attn_k", "attn_v"] {
                qkv.extend_from_slice(
                    map.tensor_bytes(&format!("{p}.{name}.weight"))
                        .map_err(|e| error(e.to_string()))?
                        .1,
                );
            }
            let rel = map
                .tensor_bytes(&format!("{p}.attn_rel_pos_emb"))
                .map_err(|e| error(e.to_string()))?
                .1;
            blocks.push(Block {
                ff1: norm(&format!("{p}.ffn_norm"), E)?,
                up1: linear(&format!("{p}.ffn_up"), E, F)?,
                down1: linear(&format!("{p}.ffn_down"), F, E)?,
                attn: norm(&format!("{p}.ln1"), E)?,
                qkv: Weight {
                    buffer: d.upload(&qkv)?,
                    ty: 1,
                    k: E,
                    n: 3 * E,
                },
                // Only +/-199 can be addressed within a 200-frame block.
                // Keep a symmetric +/-200 slab by byte slicing, not recasting.
                rel: Weight {
                    buffer: d.upload(&rel[312 * 128 * 4..713 * 128 * 4])?,
                    ty: 0,
                    k: 128,
                    n: 401,
                },
                out: linear(&format!("{p}.attn_out"), E, E)?,
                conv: norm(&format!("{p}.norm_conv"), E)?,
                pw1: linear(&format!("{p}.conv_pw1"), E, 4096)?,
                dw: weight(&format!("{p}.conv_dw.weight"), &[15, 2048])?,
                bn: norm(&format!("{p}.conv_norm"), 2048)?,
                pw2: linear(&format!("{p}.conv_pw2"), 2048, E)?,
                ff2: norm(&format!("{p}.ffn_norm_1"), E)?,
                up2: linear(&format!("{p}.ffn_up_1"), E, F)?,
                down2: linear(&format!("{p}.ffn_down_1"), F, E)?,
                post: norm(&format!("{p}.ln2"), E)?,
            });
        }
        let attention = |p: &str, kind: &str, k: usize| -> Result<Attention> {
            Ok(Attention {
                q: linear(&format!("{p}.{kind}_q"), E, E)?,
                k: linear(&format!("{p}.{kind}_k"), k, E)?,
                v: linear(&format!("{p}.{kind}_v"), k, E)?,
                out: linear(&format!("{p}.{kind}_out"), E, E)?,
                norm: norm(&format!("{p}.{kind}_norm"), E)?,
            })
        };
        let mut projectors = Vec::new();
        for i in 0..2 {
            let p = format!("a.proj_blk.{i}");
            projectors.push(Projector {
                sa: attention(&p, "self_attn", E)?,
                ca: attention(&p, "cross_attn", if plus { 2048 } else { E })?,
                up: linear(&format!("{p}.ffn_up"), E, F)?,
                down: linear(&format!("{p}.ffn_down"), F, E)?,
                norm: norm(&format!("{p}.ffn_norm"), E)?,
            });
        }
        let raw = weight("a.proj_query", &[E, 3, 1])?;
        let qnorm = norm("a.proj_norm", E)?;
        let queries = d.alloc(3 * E * 4)?;
        let c = d.begin()?;
        Self::norm(&c, &qnorm, &raw.buffer, &queries, 3, 1e-12);
        c.finish()?;
        Ok(Self {
            input: linear("a.input_projection", 160, E)?,
            blocks,
            ctc: linear("a.enc_ctc_out", E, 348)?,
            mid: linear("a.enc_ctc_out_mid", 348, E)?,
            queries,
            projectors,
            output: linear("a.proj_linear", E, 2048)?,
            plus,
        })
    }
}
