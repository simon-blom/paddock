//! Read the upstream unpermuted ViT and its quantized adapter unchanged.
//! In particular retain the two-frame patch projection; the GGUF exporter
//! folds it for still images, which is a different BF16 contraction order.
use super::super::mlx::Source;
use super::*;
impl Vision {
    pub(in super::super) fn load_mlx(d: &MetalDevice, s: &Source) -> Result<Self> {
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                w: s.raw(d, &format!("{name}.weight"), &[E], true)?,
                b: s.raw(d, &format!("{name}.bias"), &[E], true)?,
            })
        };
        let mut blocks = Vec::new();
        for i in 0..50 {
            let prefix = format!("vision_tower.layers.{i}");
            let mat =
                |name: &str, k, n| s.raw(d, &format!("{prefix}.{name}.weight"), &[n, k], false);
            let bias = |name: &str, n| s.raw(d, &format!("{prefix}.{name}.bias"), &[n], true);
            blocks.push(Block {
                ln1: norm(&format!("{prefix}.norm1"))?,
                ln2: norm(&format!("{prefix}.norm2"))?,
                q: mat("attn.q_proj", E, E)?,
                k: mat("attn.k_proj", E, E)?,
                v: mat("attn.v_proj", E, E)?,
                qb: bias("attn.q_proj", E)?,
                kb: bias("attn.k_proj", E)?,
                vb: bias("attn.v_proj", E)?,
                out: mat("attn.proj", E, E)?,
                ob: bias("attn.proj", E)?,
                up: mat("mlp.fc1", E, F)?,
                ub: bias("mlp.fc1", F)?,
                down: mat("mlp.fc2", F, E)?,
                db: bias("mlp.fc2", E)?,
            });
        }
        Ok(Self {
            mlx: true,
            blocks,
            eps: 1e-5,
            patch: s.raw(
                d,
                "vision_tower.patch_embedder.patch_embedding.weight",
                &[E, 1176],
                false,
            )?,
            pos: s.raw(
                d,
                "vision_tower.patch_embedder.position_embedding_table.weight",
                &[1024, E],
                true,
            )?,
            pre: norm("vision_tower.ln_pre")?,
            post: norm("vision_tower.ln_post")?,
            mm0: s.affine(d, "vision_adapter.fc1", 6144, 4096)?,
            mm1: s.affine(d, "vision_adapter.fc2", 4096, 4096)?,
            mm2: s.affine(d, "vision_projection", 4096, 6656)?,
        })
    }
}
