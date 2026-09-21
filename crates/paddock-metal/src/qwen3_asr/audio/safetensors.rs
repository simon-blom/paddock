//! Aligner's integrated tower. Same graph as ASR, BF16 convs/linears, 1024
//! output width. No quantization or duplicate Q/K/V resident planes.
use super::*;
use crate::qwen3_asr::safetensors::{Schema, Weights};
pub(in crate::qwen3_asr) fn schema() -> Schema {
    let mut s = Vec::new();
    for (i, ch) in [(1, 1), (2, 480), (3, 480)] {
        s.push((
            format!("model.audio_tower.conv2d{i}.weight"),
            vec![480, ch, 3, 3],
        ));
        s.push((format!("model.audio_tower.conv2d{i}.bias"), vec![480]));
    }
    s.push(("model.audio_tower.conv_out.weight".into(), vec![1024, 7680]));
    for i in 0..24 {
        for n in ["self_attn_layer_norm", "final_layer_norm"] {
            for tail in ["weight", "bias"] {
                s.push((
                    format!("model.audio_tower.layers.{i}.{n}.{tail}"),
                    vec![1024],
                ));
            }
        }
        for (n, k, out) in [
            ("self_attn.q_proj", 1024, 1024),
            ("self_attn.k_proj", 1024, 1024),
            ("self_attn.v_proj", 1024, 1024),
            ("self_attn.out_proj", 1024, 1024),
            ("fc1", 1024, 4096),
            ("fc2", 4096, 1024),
        ] {
            let p = format!("model.audio_tower.layers.{i}.{n}");
            s.push((format!("{p}.weight"), vec![out, k]));
            s.push((format!("{p}.bias"), vec![out]));
        }
    }
    for tail in ["weight", "bias"] {
        s.push((format!("model.audio_tower.ln_post.{tail}"), vec![1024]));
    }
    for i in [1, 2] {
        s.push((
            format!("model.multi_modal_projector.linear_{i}.weight"),
            vec![1024, 1024],
        ));
        s.push((
            format!("model.multi_modal_projector.linear_{i}.bias"),
            vec![1024],
        ));
    }
    s
}
impl Tower {
    pub(in crate::qwen3_asr) fn st_schema() -> Schema {
        schema()
    }
    pub(in crate::qwen3_asr) fn load_st(d: &MetalDevice, w: &Weights) -> Result<Self> {
        let norm = |p: &str| -> Result<Norm> {
            Ok(Norm {
                w: w.vector(d, &format!("{p}.weight"), 1024)?,
                b: w.vector(d, &format!("{p}.bias"), 1024)?,
            })
        };
        let linear = |p: &str, k, n| -> Result<Linear> {
            Ok(Linear {
                w: w.plane(d, &format!("{p}.weight"), k, n)?,
                b: w.vector(d, &format!("{p}.bias"), n)?,
            })
        };
        let conv = [(1, 1), (2, 480), (3, 480)]
            .into_iter()
            .map(|(i, ch)| linear(&format!("model.audio_tower.conv2d{i}"), ch * 9, 480))
            .collect::<Result<Vec<_>>>()?;
        let mut blocks = Vec::new();
        for i in 0..24 {
            let p = format!("model.audio_tower.layers.{i}");
            let names = |tail| ["q", "k", "v"].map(|a| format!("{p}.self_attn.{a}_proj.{tail}"));
            blocks.push(Block {
                ln1: norm(&format!("{p}.self_attn_layer_norm"))?,
                ln2: norm(&format!("{p}.final_layer_norm"))?,
                qkv: Linear {
                    w: w.fused(d, &names("weight"), 1024, 3072, false)?,
                    b: w.fused(d, &names("bias"), 3072, 1, true)?,
                },
                out: linear(&format!("{p}.self_attn.out_proj"), 1024, 1024)?,
                up: linear(&format!("{p}.fc1"), 1024, 4096)?,
                down: linear(&format!("{p}.fc2"), 4096, 1024)?,
            });
        }
        let pos = d.alloc(13 * 1024 * 4)?;
        let c = d.begin()?;
        c.dispatch(
            "qalign_sinusoid",
            &[&pos],
            &[0],
            [(13usize * 1024).div_ceil(256), 1, 1],
            256,
        );
        c.finish()?;
        Ok(Self {
            bf16_activations: true,
            conv,
            blocks,
            conv_out: w.plane(d, "model.audio_tower.conv_out.weight", 7680, 1024)?,
            pos: Weight {
                buffer: pos,
                ty: 0,
                k: 1024,
                n: 13,
            },
            post: norm("model.audio_tower.ln_post")?,
            up: linear("model.multi_modal_projector.linear_1", 1024, 1024)?,
            down: linear("model.multi_modal_projector.linear_2", 1024, 1024)?,
        })
    }
}
