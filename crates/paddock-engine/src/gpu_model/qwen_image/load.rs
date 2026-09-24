//! The DiT's weights from the GGUF. The unsloth conversion carries NO
//! metadata (zero KV pairs, no `general.architecture`), so the file is
//! identified and measured by its tensor names and shapes; the sd.cpp
//! converter it came through also FUSES the SwiGLU as one
//! `img_mlp.gate_up.weight` plane whose first `ff` output rows are the gate
//! and the rest the up projection (`ggml_ext_chunk(.., 2, 0)`, parts[0] =
//! gate). The official split layout (`gate_layer` / `proj`) loads too.
//!
//! Every matmul plane dispatches on its tensor type: F16 becomes an f16
//! plane on the f16 GEMM (the parity file), anything else goes through the
//! per-tensor quant dispatch (`load_quantw`) - the Q8_0 file is Q8_0
//! throughout, the Q4_K_M file mixes Q4_K / Q5_K / Q6_K / Q8_0 per tensor.
//! The four bf16 planes (`img_in`, `txt_in.*`) stay bf16 on the bf16 dense
//! lane; the f32 ones (norms, `norm_out.linear`) are f32.

use std::sync::Arc;

use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

use half::f16;

use crate::gpu::{GpuError, GpuExecutor, HalfTensor, QuantTensor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::dit::{DitBlock, DitModel, Hparams, Plane};

/// The tensor-name prefix the conversion writes; the second form is what a
/// direct diffusers -> GGUF export would carry.
const PREFIXES: [&str; 2] = ["model.diffusion_model.", ""];

/// Does this GGUF hold a Qwen-Image-2.1-shaped DiT? Decided by names, since
/// the file says nothing about itself.
pub fn is_dit_gguf(map: &MappedGguf) -> bool {
    PREFIXES.iter().any(|p| {
        map.tensor_info(&format!("{p}img_in.weight")).is_some()
            && map
                .tensor_info(&format!("{p}modulation.1.weight"))
                .is_some()
            && map
                .tensor_info(&format!("{p}transformer_blocks.0.attn.to_q.weight"))
                .is_some()
    })
}

/// Bytes one row of `in_dim` weights takes in `ty` - what slicing a fused
/// plane by output rows needs.
fn row_bytes(ty: GgmlType, in_dim: usize) -> Result<usize, GpuModelError> {
    Ok(match ty {
        GgmlType::Q8_0 => in_dim / 32 * 34,
        GgmlType::F16 | GgmlType::Bf16 => in_dim * 2,
        GgmlType::F32 => in_dim * 4,
        t => {
            let (_, block_bytes, _) = crate::gpu::kq_params(t).ok_or_else(|| {
                GpuModelError::Unsupported(format!("qwen-image: no row size for {t:?}"))
            })?;
            in_dim / 256 * block_bytes
        }
    })
}

impl DitModel {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuModelError> {
        exec.vram_load_gate(map.total_len(), "qwen-image")
            .map_err(GpuModelError::WontFit)?;
        exec.disable_event_tracking();

        let prefix = PREFIXES
            .iter()
            .copied()
            .find(|p| map.tensor_info(&format!("{p}img_in.weight")).is_some())
            .ok_or_else(|| {
                GpuModelError::Unsupported(
                    "not a Qwen-Image DiT GGUF: no img_in.weight under any known prefix".into(),
                )
            })?;
        let name = |s: &str| format!("{prefix}{s}");
        let dims = |s: &str| -> Result<Vec<usize>, GpuModelError> {
            let info = map
                .tensor_info(&name(s))
                .ok_or_else(|| GpuModelError::MissingMeta(name(s)))?;
            Ok(info.dims.iter().map(|&d| d as usize).collect())
        };

        // geometry from the shapes (GGUF dims are [in, out])
        let img_in = dims("img_in.weight")?;
        let (in_channels, hidden) = (img_in[0], img_in[1]);
        let context_dim = dims("txt_in.in_layer.weight")?[0];
        let head_dim = dims("transformer_blocks.0.attn.norm_q.weight")?[0];
        let out_channels = dims("proj_out.weight")?[1];
        let (ff, fused_mlp) =
            match map.tensor_info(&name("transformer_blocks.0.img_mlp.gate_up.weight")) {
                Some(t) => (t.dims[1] as usize / 2, true),
                None => (dims("transformer_blocks.0.img_mlp.proj.weight")?[1], false),
            };
        let n_layers = (0..)
            .take_while(|i| {
                map.tensor_info(&name(&format!("transformer_blocks.{i}.attn.to_q.weight")))
                    .is_some()
            })
            .count();
        if hidden % head_dim != 0 || n_layers == 0 {
            return Err(GpuModelError::Unsupported(format!(
                "qwen-image: hidden {hidden} / head {head_dim} / {n_layers} layers - not a shape this family knows"
            )));
        }
        let axes = super::AXES_DIMS_ROPE;
        if axes.iter().sum::<usize>() != head_dim {
            return Err(GpuModelError::Unsupported(format!(
                "qwen-image: head dim {head_dim} does not match the 2.1 rope split {axes:?}"
            )));
        }
        let hp = Hparams {
            hidden,
            n_heads: hidden / head_dim,
            head_dim,
            ff,
            n_layers,
            in_channels,
            out_channels,
            context_dim,
            eps: 1e-6,
        };
        tracing::info!(
            layers = n_layers,
            hidden,
            heads = hp.n_heads,
            ff,
            fused_mlp,
            "qwen-image DiT geometry"
        );

        // per-tensor dispatch: F16 planes ride the f16 GEMM (the parity
        // file), everything else the per-type quant lanes
        let plane = |s: &str| -> Result<Plane, GpuModelError> {
            let n = name(s);
            let ty = map
                .tensor_info(&n)
                .ok_or_else(|| GpuModelError::MissingMeta(n.clone()))?
                .ggml_type;
            Ok(match ty {
                GgmlType::F16 => Plane::Half(exec.upload_f16(map, &n)?),
                _ => Plane::Quant(exec.load_quantw(map, &n)?),
            })
        };
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let b = |s: &str| format!("transformer_blocks.{i}.{s}");
            let (gate, up) = if fused_mlp {
                split_fused(&exec, map, &name(&b("img_mlp.gate_up.weight")), ff)?
            } else {
                (
                    plane(&b("img_mlp.gate_layer.weight"))?,
                    plane(&b("img_mlp.proj.weight"))?,
                )
            };
            blocks.push(DitBlock {
                wq: plane(&b("attn.to_q.weight"))?,
                wk: plane(&b("attn.to_k.weight"))?,
                wv: plane(&b("attn.to_v.weight"))?,
                wo: plane(&b("attn.to_out.0.weight"))?,
                norm_q: exec.upload(map, &name(&b("attn.norm_q.weight")))?,
                norm_k: exec.upload(map, &name(&b("attn.norm_k.weight")))?,
                gate,
                up,
                down: plane(&b("img_mlp.out.weight"))?,
            });
        }

        // the small planes around the blocks
        let t_lin1 = plane("time_text_embed.timestep_embedder.linear_1.weight")?;
        let t_lin2 = plane("time_text_embed.timestep_embedder.linear_2.weight")?;
        let modulation = plane("modulation.1.weight")?;
        let norm_out_lin = exec.upload(map, &name("norm_out.linear.weight"))?;
        // proj_out is 4096 -> 64: too narrow for the mmq tile to be worth
        // anything, and one exact f32 GEMM per step over N x 64 is noise
        let proj_out = exec.upload(map, &name("proj_out.weight"))?;
        let img_in = bf16_plane(&exec, map, &name("img_in.weight"))?;
        let txt_in_layer = bf16_plane(&exec, map, &name("txt_in.in_layer.weight"))?;
        let txt_out_layer = bf16_plane(&exec, map, &name("txt_in.out_layer.weight"))?;
        // zero-centred RMSNorm: the file stores scale - 1; fold the + 1 once
        let (mut tn, _) =
            crate::gpu_model::qwen35::vision::host_f32(map, &name("txt_in.text_norm.weight"))?;
        for v in &mut tn {
            *v += 1.0;
        }
        let txt_norm = exec.to_device(&tn)?;
        let weights_bytes = blocks
            .iter()
            .map(|b| {
                b.wq.bytes()
                    + b.wk.bytes()
                    + b.wv.bytes()
                    + b.wo.bytes()
                    + b.gate.bytes()
                    + b.up.bytes()
                    + b.down.bytes()
            })
            .sum::<u64>()
            + t_lin1.bytes()
            + t_lin2.bytes()
            + modulation.bytes()
            + (norm_out_lin.buf.len() + proj_out.buf.len()) as u64 * 4
            + (img_in.bytes.len() + txt_in_layer.bytes.len() + txt_out_layer.bytes.len()) as u64;

        DitModel::new(
            exec,
            hp,
            blocks,
            t_lin1,
            t_lin2,
            modulation,
            norm_out_lin,
            proj_out,
            img_in,
            txt_in_layer,
            txt_out_layer,
            txt_norm,
            weights_bytes,
        )
    }
}

/// A bf16 GEMM plane for the bf16 dense lane. The known files ship these as
/// BF16; anything else is refused rather than silently widened.
fn bf16_plane(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
) -> Result<QuantTensor, GpuModelError> {
    let info = map
        .tensor_info(name)
        .ok_or_else(|| GpuModelError::MissingMeta(name.to_owned()))?;
    if info.ggml_type != GgmlType::Bf16 {
        return Err(GpuModelError::Unsupported(format!(
            "qwen-image: {name} is {:?}, expected BF16",
            info.ggml_type
        )));
    }
    if !exec.has_bf16_dense() {
        return Err(GpuModelError::Unsupported(
            "qwen-image: this pack has no bf16 dense lane".into(),
        ));
    }
    Ok(exec.upload_raw(map, name)?)
}

/// Split the fused `[in][2ff]` gate|up plane into two resident planes by
/// output rows. Quant blocks never straddle a row, so a byte slice by rows
/// is exact for every format.
fn split_fused(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    ff: usize,
) -> Result<(Plane, Plane), GpuModelError> {
    let (info, bytes) = map.tensor_bytes(name).map_err(GpuError::from)?;
    let in_dim = info.dims[0] as usize;
    let rb = row_bytes(info.ggml_type, in_dim)?;
    let half = rb * ff;
    if bytes.len() != half * 2 {
        return Err(GpuModelError::Unsupported(format!(
            "qwen-image: {name} has {} bytes, expected {} for [{in_dim}][{}]",
            bytes.len(),
            half * 2,
            2 * ff
        )));
    }
    let plane = |b: &[u8], what: &str| -> Result<Plane, GpuModelError> {
        Ok(match info.ggml_type {
            GgmlType::Q8_0 => Plane::Quant(QuantW::Q8(exec.repack_q8_blocks(b, vec![in_dim, ff])?)),
            GgmlType::F16 => {
                let host: Vec<f16> = b
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16::from_le_bytes(*c))
                    .collect();
                Plane::Half(HalfTensor {
                    buf: exec.f16_to_device(&host)?,
                    dims: vec![in_dim, ff],
                })
            }
            ty if crate::gpu::kq_params(ty).is_some() => Plane::Quant(QuantW::Kq(
                exec.repack_kquant_raw(b, vec![in_dim, ff], ty, what)?,
            )),
            ty => {
                return Err(GpuModelError::Unsupported(format!(
                    "qwen-image: {name} is {ty:?}; fused planes load as F16, Q8_0 or a k-quant"
                )));
            }
        })
    };
    Ok((plane(&bytes[..half], "gate")?, plane(&bytes[half..], "up")?))
}
