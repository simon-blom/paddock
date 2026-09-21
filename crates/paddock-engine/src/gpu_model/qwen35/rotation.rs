//! Rotated-basis checkpoints on the qwen35 family (PrismML's Bonsai GGUFs,
//! `prism.hadamard.*` - the contract itself is `paddock_models::hadamard`).
//!
//! The file's linear weights are stored as `W' = W S H` per 1024-wide strip
//! of the input axis, so each of them has to be fed `x' = H(s * x)`. The
//! residual stream, the norms, attention and the gated-delta-net state all
//! stay in the model's own basis: the rotation lives only between the last op
//! that produces a linear's input and the matmul. Per layer that is four
//! buffers - the pre-mixer norm (q/k/v or in_qkv + gate read it), the mixer
//! output (wo, or ssm_out with its head permutation), the post norm (ffn gate
//! and up) and the SwiGLU product (ffn down) - plus the final norm in front of
//! the head, and the inverse on what the embedding lookup returns.
//!
//! Two things keep that list this short:
//!   - weights sharing an input share one rotation, so the buffer is rotated
//!     IN PLACE once and every consumer reads the rotated rows. That needs
//!     every consumer of the buffer to agree, which is why `load` refuses a
//!     file that rotates some of a family's linears and not others.
//!   - ssm_alpha / ssm_beta read the pre-mixer norm too and ship UNrotated
//!     (BF16). Rather than keep a second copy of the normed rows around for
//!     two 48-row projections, the loader moves those two weights into the
//!     rotated basis once (`row' = H(s * row)`, the same op, on the GPU):
//!     `W x = (W S H)(H S x)` exactly, H being orthonormal and symmetric, and
//!     the f32 rounding of it is three orders under the int8 activation
//!     noise every other projection of the layer already carries.

use crate::gpu::{GpuExecutor, HadamardGdnHeads};
use crate::gpu_model::gpt_oss::GpuModelError;
use cudarc::driver::CudaSlice;
use paddock_models::hadamard::HadamardSpec;
use paddock_models::mapped::MappedGguf;

/// Model geometry the contract is checked against.
pub(super) struct RotationGeometry {
    pub n_layers: usize,
    pub full_attn_interval: usize,
    pub embd: usize,
    pub q_dim: usize,
    pub value_dim: usize,
    pub ff: usize,
    /// gated-delta-net value head width, key-head count, value-head count
    pub state_size: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub moe: bool,
    pub n_nextn: usize,
    pub fp8_native: bool,
}

/// The live rotation of one loaded model.
pub(crate) struct Rotation {
    block: usize,
    /// device +-1 vectors by input width
    signs: Vec<(usize, CudaSlice<f32>)>,
    /// tiled -> grouped value-head permutation in front of ssm_out
    gdn: Option<HadamardGdnHeads>,
    /// the token embedding stores rotated rows
    embd_inverse: bool,
}

fn refuse(msg: String) -> GpuModelError {
    GpuModelError::Unsupported(format!("rotated-basis file (prism.hadamard.*): {msg}"))
}

impl Rotation {
    /// Read the file's contract. `Ok(None)` for an ordinary file; for a
    /// rotated one either everything checks out - and the executor is told the
    /// family applies the rotation, which is what lets the weight loaders
    /// touch the file at all - or the load stops here naming the reason.
    pub(super) fn load(
        exec: &GpuExecutor,
        map: &MappedGguf,
        g: &RotationGeometry,
    ) -> Result<Option<Self>, GpuModelError> {
        let spec = HadamardSpec::from_gguf(map.gguf()).map_err(|e| refuse(e.to_string()))?;
        let Some(spec) = spec else { return Ok(None) };

        if !exec.has_hadamard() {
            return Err(refuse(
                "the kernel pack has no Walsh-Hadamard rotation (slot 626) - rebuild packs/cuda"
                    .into(),
            ));
        }
        if !matches!(spec.block, 256 | 512 | 1024 | 2048 | 4096) {
            return Err(refuse(format!(
                "block_size {} is outside the rotation kernel's 256..4096",
                spec.block
            )));
        }
        // lanes that would read these weights without passing a rotating site
        if g.moe {
            return Err(refuse(
                "the MoE expert lanes do not rotate their inputs yet".into(),
            ));
        }
        if g.n_nextn > 0 {
            return Err(refuse(
                "the nextn/MTP draft block does not rotate its inputs yet".into(),
            ));
        }
        if g.fp8_native {
            return Err(refuse(
                "an fp8-native snapshot cannot stand in for rotated weights".into(),
            ));
        }

        // Every linear the family runs, and nothing else: a buffer is rotated
        // once for all of its consumers, so a partial list cannot be served.
        let mut want: Vec<String> = Vec::with_capacity(7 * g.n_layers + 1);
        for i in 0..g.n_layers {
            let full = (i + 1) % g.full_attn_interval == 0;
            let mixer: &[&str] = if full {
                &["attn_q", "attn_k", "attn_v", "attn_output"]
            } else {
                &["attn_qkv", "attn_gate", "ssm_out"]
            };
            for kind in mixer.iter().chain(&["ffn_gate", "ffn_up", "ffn_down"]) {
                want.push(format!("blk.{i}.{kind}.weight"));
            }
        }
        want.push("output.weight".to_owned());
        for name in &want {
            if !spec.weights.contains(name) {
                return Err(refuse(format!(
                    "{name} is not listed as rotated, and this family rotates a layer's inputs \
                     for all of its linears at once"
                )));
            }
            if map.tensor_info(name).is_none() {
                return Err(refuse(format!("{name} is listed but not in the file")));
            }
        }
        if spec.weights.len() != want.len() {
            let extra = spec
                .weights
                .iter()
                .filter(|n| !want.contains(n))
                .min()
                .cloned()
                .unwrap_or_default();
            return Err(refuse(format!(
                "{extra} is listed as rotated but this family has no rotating site for it"
            )));
        }

        let mut signs: Vec<(usize, CudaSlice<f32>)> = Vec::new();
        for width in [g.embd, g.q_dim, g.value_dim, g.ff] {
            if signs.iter().any(|(w, _)| *w == width) {
                continue;
            }
            if !width.is_multiple_of(spec.block) {
                return Err(refuse(format!(
                    "block_size {} does not divide the input width {width}",
                    spec.block
                )));
            }
            let host = spec.sign_vector(width).map_err(|e| refuse(e.to_string()))?;
            let mut dev = exec.alloc(width)?;
            exec.upload_f32(&host, &mut dev)?;
            signs.push((width, dev));
        }

        let gdn = if spec.gdn_v_grouped {
            if g.n_k_heads == 0
                || !g.n_v_heads.is_multiple_of(g.n_k_heads)
                || g.state_size * g.n_v_heads != g.value_dim
            {
                return Err(refuse(format!(
                    "gdn_v_grouped with value heads {} / key heads {} / head width {}",
                    g.n_v_heads, g.n_k_heads, g.state_size
                )));
            }
            Some(HadamardGdnHeads {
                head_dim: g.state_size,
                n_k: g.n_k_heads,
                rep: g.n_v_heads / g.n_k_heads,
            })
        } else {
            None
        };

        exec.acknowledge_rotated_basis();
        tracing::info!(
            "qwen35: rotated-basis checkpoint (prism.hadamard v1) - {} weights behind a \
             block-{} Walsh-Hadamard rotation, {} sign vectors, embedding inverse {}, \
             GDN head regroup {}",
            spec.weights.len(),
            spec.block,
            signs.len(),
            if spec.embd_inverse { "on" } else { "off" },
            if gdn.is_some() { "on" } else { "off" },
        );
        Ok(Some(Self {
            block: spec.block,
            signs,
            gdn,
            embd_inverse: spec.embd_inverse,
        }))
    }

    fn signs(&self, width: usize) -> Result<&CudaSlice<f32>, GpuModelError> {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, s)| s)
            .ok_or_else(|| refuse(format!("no sign vector was loaded for width {width}")))
    }

    /// `x <- H(s * x)`, in place, over `rows` rows of `width`: the input of a
    /// rotated matmul (or of several that share it).
    pub(crate) fn rotate(
        &self,
        exec: &GpuExecutor,
        x: &mut CudaSlice<f32>,
        width: usize,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        exec.hadamard_rotate_inplace(x, self.signs(width)?, rows, width, self.block, false)?;
        Ok(())
    }

    /// The gated-delta-net output projection's input: `y = H(s * P x)`, P the
    /// tiled -> grouped value-head regroup when the file asks for it. Never
    /// in place (whole heads cross strips), so the caller hands a second
    /// plane of the same size and ssm_out reads `y`.
    pub(crate) fn rotate_ssm_out(
        &self,
        exec: &GpuExecutor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        width: usize,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        exec.hadamard_rotate(x, y, self.signs(width)?, rows, width, self.block, self.gdn)?;
        Ok(())
    }

    /// [`Self::rotate`] that also stages the rotated rows as per-128 int8
    /// (`xq`, `xs`) in the same launch - the batch-1 ternary lane's operand.
    /// `x` still ends up rotated in place, so an f32 reader of the buffer
    /// (alpha / beta, a consumer that stages for itself) stays right.
    pub(crate) fn rotate_q8(
        &self,
        exec: &GpuExecutor,
        x: &mut CudaSlice<f32>,
        xq: &mut CudaSlice<i8>,
        xs: &mut CudaSlice<f32>,
        width: usize,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        let signs = self.signs(width)?;
        exec.hadamard_rotate_q8_b128(x, None, signs, xq, xs, rows, width, self.block, None)?;
        Ok(())
    }

    /// [`Self::rotate_ssm_out`] with the int8 operand staged in the same
    /// launch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rotate_ssm_out_q8(
        &self,
        exec: &GpuExecutor,
        x: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        xq: &mut CudaSlice<i8>,
        xs: &mut CudaSlice<f32>,
        width: usize,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        let signs = self.signs(width)?;
        exec.hadamard_rotate_q8_b128(x, Some(y), signs, xq, xs, rows, width, self.block, self.gdn)?;
        Ok(())
    }

    /// What the embedding lookup returned, back in the model's basis:
    /// `x <- s * H(x)`. A no-op for a file whose table is not rotated.
    pub(crate) fn after_embed(
        &self,
        exec: &GpuExecutor,
        x: &mut CudaSlice<f32>,
        width: usize,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        if self.embd_inverse {
            exec.hadamard_rotate_inplace(x, self.signs(width)?, rows, width, self.block, true)?;
        }
        Ok(())
    }
}

/// `rot.rotate(..)` when the model is rotated, nothing otherwise - the form
/// every site in the layer walks uses.
pub(crate) fn rotate_opt(
    rot: Option<&Rotation>,
    exec: &GpuExecutor,
    x: &mut CudaSlice<f32>,
    width: usize,
    rows: usize,
) -> Result<(), GpuModelError> {
    match rot {
        Some(r) => r.rotate(exec, x, width, rows),
        None => Ok(()),
    }
}
