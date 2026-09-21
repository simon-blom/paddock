//! MLX's own image geometry and embedded tower, not the GGUF compatibility
//! processor (whose centering and activation metadata differ).
use super::super::mlx::Source;
use super::*;

pub(in super::super) fn resize(w: usize, h: usize) -> Result<(usize, usize)> {
    if w == 0 || h == 0 || w > 65536 || h > 65536 {
        return Err(error("invalid MLX Gemma image edges"));
    }
    let scale = (280.0 * 2304.0 / (w as f64 * h as f64)).sqrt();
    let mut tw = (scale * w as f64 / 48.).floor() as usize * 48;
    let mut th = (scale * h as f64 / 48.).floor() as usize * 48;
    if th == 0 {
        th = 48;
        tw = (w / h).min(280) * 48;
    }
    if tw == 0 {
        tw = 48;
        th = (h / w).min(280) * 48;
    }
    if tw == 0 || th == 0 || tw * th > 280 * 2304 {
        return Err(error("MLX Gemma aspect ratio exceeds patch budget"));
    }
    Ok((tw, th))
}
impl Vision {
    pub(in super::super) fn load_mlx(d: &MetalDevice, s: &Source) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..27 {
            let prefix = format!("vision_tower.encoder.layers.{i}");
            let norm = |name: &str, n| s.raw(d, &format!("{prefix}.{name}.weight"), &[n], true);
            let mat = |name: &str, k, n| {
                s.raw(d, &format!("{prefix}.{name}.linear.weight"), &[n, k], false)
            };
            blocks.push(Block {
                norm: norm("input_layernorm", E)?,
                post: norm("post_attention_layernorm", E)?,
                ffn_norm: norm("pre_feedforward_layernorm", E)?,
                ffn_post: norm("post_feedforward_layernorm", E)?,
                q: mat("self_attn.q_proj", E, E)?,
                k: mat("self_attn.k_proj", E, E)?,
                v: mat("self_attn.v_proj", E, E)?,
                q_norm: norm("self_attn.q_norm", 72)?,
                k_norm: norm("self_attn.k_norm", 72)?,
                out: mat("self_attn.o_proj", E, E)?,
                gate: mat("mlp.gate_proj", E, F)?,
                up: mat("mlp.up_proj", E, F)?,
                down: mat("mlp.down_proj", F, E)?,
            });
        }
        Ok(Self {
            mlx: true,
            blocks,
            eps: 1e-6,
            quick_gelu: false,
            patch: s.raw(
                d,
                "vision_tower.patch_embedder.input_proj.weight",
                &[E, 768],
                false,
            )?,
            pos: s.raw(
                d,
                "vision_tower.patch_embedder.position_embedding_table",
                &[2, 10240, E],
                true,
            )?,
            bias: s.raw(d, "vision_tower.std_bias", &[E], true)?,
            scale: s.raw(d, "vision_tower.std_scale", &[E], true)?,
            projection: s.affine(d, "embed_vision.embedding_projection", E, 5376)?,
        })
    }
}
