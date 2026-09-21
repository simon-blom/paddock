use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};
impl Tower {
    pub(in crate::qwen3_asr) fn load(d: &MetalDevice, path: &Path) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        if map.gguf().architecture() != Some("clip")
            || meta
                .get("clip.audio.projector_type")
                .and_then(Value::as_str)
                != Some("qwen3a")
            || !matches!(meta.get("clip.has_audio_encoder"), Some(Value::Bool(true)))
            || meta
                .get("clip.audio.attention.layer_norm_epsilon")
                .and_then(Value::as_f32)
                != Some(1e-5)
        {
            return Err(error("requires Qwen3-ASR BF16 audio companion"));
        }
        for (k, n) in [
            ("block_count", 24),
            ("embedding_length", 1024),
            ("attention.head_count", 16),
            ("feed_forward_length", 4096),
            ("projection_dim", 2048),
            ("num_mel_bins", 128),
        ] {
            if meta.get(&format!("clip.audio.{k}")).and_then(Value::as_u64) != Some(n) {
                return Err(error(format!("unsupported audio {k}")));
            }
        }
        let mut schema = vec![
            ("a.position_embd.weight".into(), vec![1024, 1500], 0),
            ("a.conv_out.weight".into(), vec![7680, 1024], 1),
        ];
        for (i, ch) in [(1, 1), (2, 480), (3, 480)] {
            schema.push((format!("a.conv2d.{i}.weight"), vec![3, 3, ch, 480], 1));
            schema.push((format!("a.conv2d.{i}.bias"), vec![1, 1, 480], 0));
        }
        for i in 0..24 {
            for n in ["ln1", "ln2"] {
                for tail in ["weight", "bias"] {
                    schema.push((format!("a.blk.{i}.{n}.{tail}"), vec![1024], 0));
                }
            }
            for (n, k, out) in [
                ("attn_q", 1024, 1024),
                ("attn_k", 1024, 1024),
                ("attn_v", 1024, 1024),
                ("attn_out", 1024, 1024),
                ("ffn_up", 1024, 4096),
                ("ffn_down", 4096, 1024),
            ] {
                schema.push((format!("a.blk.{i}.{n}.weight"), vec![k, out], 30));
                schema.push((format!("a.blk.{i}.{n}.bias"), vec![out], 0));
            }
        }
        for tail in ["weight", "bias"] {
            schema.push((format!("a.post_ln.{tail}"), vec![1024], 0));
        }
        for (name, n) in [("mm.a.mlp.1", 1024), ("mm.a.mlp.2", 2048)] {
            schema.push((format!("{name}.weight"), vec![1024, n], 30));
            schema.push((format!("{name}.bias"), vec![n], 0));
        }
        let bytes = super::super::load::validate(&map, &schema)?;
        if d.allocated_bytes() + bytes + WORKSPACE > d.budget_bytes() {
            return Err(MetalError::Memory(
                "audio weights + workspace exceed grant".into(),
            ));
        }
        let weight = |n: &str, dims: &[usize]| Weight::load(d, &map, n, dims);
        let norm = |n: &str| -> Result<Norm> {
            Ok(Norm {
                w: weight(&format!("{n}.weight"), &[1024])?,
                b: weight(&format!("{n}.bias"), &[1024])?,
            })
        };
        let linear = |n: &str, k: usize, out: usize| -> Result<Linear> {
            Ok(Linear {
                w: weight(&format!("{n}.weight"), &[k, out])?,
                b: weight(&format!("{n}.bias"), &[out])?,
            })
        };
        // Concatenation is raw byte movement only; preserve every original
        // BF16 weight and F32 bias. No duplicate Q/K/V weights stay resident.
        let fused = |prefix: &str| -> Result<Linear> {
            let mut weights = Vec::with_capacity(3 * 1024 * 1024 * 2);
            let mut biases = Vec::with_capacity(3 * 1024 * 4);
            for plane in ["q", "k", "v"] {
                weights.extend_from_slice(
                    map.tensor_bytes(&format!("{prefix}.attn_{plane}.weight"))
                        .map_err(|e| error(e.to_string()))?
                        .1,
                );
                biases.extend_from_slice(
                    map.tensor_bytes(&format!("{prefix}.attn_{plane}.bias"))
                        .map_err(|e| error(e.to_string()))?
                        .1,
                );
            }
            Ok(Linear {
                w: Weight {
                    buffer: d.upload(&weights)?,
                    ty: 30,
                    k: 1024,
                    n: 3072,
                },
                b: Weight {
                    buffer: d.upload(&biases)?,
                    ty: 0,
                    k: 3072,
                    n: 1,
                },
            })
        };
        let mut conv = Vec::new();
        for (i, ch) in [(1, 1), (2, 480), (3, 480)] {
            let mut w = weight(&format!("a.conv2d.{i}.weight"), &[3, 3, ch, 480])?;
            w.k = ch * 9;
            w.n = 480;
            conv.push(Linear {
                w,
                b: weight(&format!("a.conv2d.{i}.bias"), &[1, 1, 480])?,
            });
        }
        let mut blocks = Vec::new();
        for i in 0..24 {
            let p = format!("a.blk.{i}");
            blocks.push(Block {
                ln1: norm(&format!("{p}.ln1"))?,
                ln2: norm(&format!("{p}.ln2"))?,
                qkv: fused(&p)?,
                out: linear(&format!("{p}.attn_out"), 1024, 1024)?,
                up: linear(&format!("{p}.ffn_up"), 1024, 4096)?,
                down: linear(&format!("{p}.ffn_down"), 4096, 1024)?,
            });
        }
        Ok(Self {
            bf16_activations: false,
            conv,
            conv_out: weight("a.conv_out.weight", &[7680, 1024])?,
            pos: weight("a.position_embd.weight", &[1024, 1500])?,
            blocks,
            post: norm("a.post_ln")?,
            up: linear("mm.a.mlp.1", 1024, 1024)?,
            down: linear("mm.a.mlp.2", 1024, 2048)?,
        })
    }
}
