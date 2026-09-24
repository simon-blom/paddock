//! MLX-community Qwen-Image-2.1 affine-4/group-64 checkpoint contract.
//! Packed codes and BF16 scales/biases remain packed on the device. This is
//! ingestion, not a conversion through GGUF or a full-model FP16 expansion.
use super::*;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};
use serde_json::{Value, json};

pub(super) fn config(dir: &Path, expected: Value) -> Result<Value> {
    config_file(&dir.join("config.json"), expected)
}

fn config_file(path: &Path, expected: Value) -> Result<Value> {
    let value: Value =
        serde_json::from_slice(&std::fs::read(path).map_err(model_error)?).map_err(model_error)?;
    check_config(path, &value, &expected)?;
    Ok(value)
}

fn check_config(path: &Path, value: &Value, expected: &Value) -> Result<()> {
    for (key, wanted) in expected.as_object().expect("fixed config contract") {
        if value.get(key) != Some(wanted) {
            return Err(error(format!("{}: unsupported {key}", path.display())));
        }
    }
    Ok(())
}

pub(super) fn pipeline(root: &Path) -> Result<()> {
    config_file(
        &root.join("model_index.json"),
        json!({
            "_class_name":"QwenImage21Pipeline",
            "scheduler":["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "transformer":["diffusers", "QwenImage21Transformer2DModel"],
            "text_encoder":["transformers", "Qwen3VLForConditionalGeneration"],
            "processor":["transformers", "Qwen3VLProcessor"],
            "vae":["diffusers", "AutoencoderKLQwenImage21"]
        }),
    )?;
    config_file(
        &root.join("scheduler/scheduler_config.json"),
        json!({
            "_class_name":"FlowMatchEulerDiscreteScheduler",
            "base_image_seq_len":256, "max_image_seq_len":8192,
            "base_shift":0.5, "max_shift":0.9, "num_train_timesteps":1000,
            "shift":1.0, "shift_terminal":0.02, "time_shift_type":"exponential",
            "use_dynamic_shifting":true, "invert_sigmas":false,
            "stochastic_sampling":false, "use_beta_sigmas":false,
            "use_exponential_sigmas":false, "use_karras_sigmas":false
        }),
    )?;
    Ok(())
}

pub(super) struct Source(pub ShardedSafetensors);

/// The DiT's reference contracts joint [prefix | image] rows. Splitting out
/// its immutable prefix must not change the K reduction to BF16 split-K
/// partials based on the now-short row count. Full F32 accumulation is used
/// for both cached prefix and target; the language encoder keeps its own policy.
pub(super) fn project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
) {
    project_grouped(cmd, planes, input, rows, workspace, 3);
}

#[cfg(test)]
pub(super) fn project_separate(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
) {
    project_grouped(cmd, planes, input, rows, workspace, 1);
}

fn project_grouped(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
    group_size: usize,
) {
    assert!(!planes.is_empty());
    let k = planes[0].0.k;
    if planes
        .iter()
        .any(|(w, _)| w.ty != crate::affine::AFFINE4 || w.k != k)
    {
        crate::weights::projections(cmd, planes, input, rows, workspace);
        return;
    }
    let elements = k.next_multiple_of(128) * rows.next_multiple_of(128);
    assert!(workspace.len() >= elements * 2);
    cmd.dispatch(
        "mlx_input",
        &[input, workspace],
        &[k as u32, rows as u32],
        [elements.div_ceil(256), 1, 1],
        256,
    );
    // Independent Q/K/V planes use one grid and one prepared activation.
    // Grid partitioning changes neither the contraction nor output rounding.
    for group in planes.chunks(group_size) {
        let (w, out) = group[0];
        let (w1, o1) = group.get(1).copied().unwrap_or((w, out));
        let (w2, o2) = group.get(2).copied().unwrap_or((w, out));
        let tile = if rows <= 32 {
            32
        } else if cmd.tensor_accelerated() && rows.is_multiple_of(256) && rows >= 512 {
            256
        } else {
            64
        };
        cmd.dispatch(
            match tile {
                32 => "mlx_affine_tile32",
                256 => "mlx_affine_prefill_store256",
                _ => "mlx_affine_prefill_load32",
            },
            &[&w.buffer, &w1.buffer, &w2.buffer, workspace, out, o1, o2],
            &[
                k as u32,
                w.n as u32,
                if group.len() > 1 { w1.n as u32 } else { 0 },
                if group.len() > 2 { w2.n as u32 } else { 0 },
                rows as u32,
            ],
            [
                group.iter().map(|(w, _)| w.n.div_ceil(32)).sum(),
                rows.div_ceil(tile),
                1,
            ],
            128,
        );
    }
}

impl Source {
    pub fn open(dir: &Path) -> Result<Self> {
        config(
            dir,
            json!({"mlx_format": true,
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}),
        )?;
        Ok(Self(
            ShardedSafetensors::open_dir(dir).map_err(model_error)?,
        ))
    }
    pub fn weight(&self, d: &MetalDevice, name: &str, shape: &[usize]) -> Result<Weight> {
        let (info, bytes) = self
            .0
            .bytes(name)
            .ok_or_else(|| error(format!("missing MLX tensor {name}")))?;
        let k = shape[0];
        let n = shape.get(1).copied().unwrap_or(1);
        if info.dtype == StDtype::U32 && shape.len() == 2 {
            return crate::affine::load(d, &self.0, name, k, n);
        }
        let expected: Vec<_> = shape.iter().rev().copied().collect();
        if info.shape != expected
            || !matches!(info.dtype, StDtype::Bf16 | StDtype::F32 | StDtype::F16)
        {
            return Err(error(format!(
                "{name}: expected dense {expected:?}, got {info:?}"
            )));
        }
        Ok(Weight {
            buffer: d.upload(bytes)?,
            ty: match info.dtype {
                StDtype::Bf16 => 30,
                StDtype::F16 => 1,
                _ => 0,
            },
            k,
            n,
        })
    }
    /// Byte-only concatenation into [gate; up], independently for the three
    /// affine planes. No rounding, dequantization, or duplicate residency.
    pub fn gate_up(&self, d: &MetalDevice, base: &str) -> Result<Weight> {
        let mut parts = Vec::with_capacity(6);
        for (suffix, dtype, columns) in [
            ("weight", StDtype::U32, 512),
            ("scales", StDtype::Bf16, 64),
            ("biases", StDtype::Bf16, 64),
        ] {
            for projection in ["gate_layer", "proj"] {
                let name = format!("{base}.{projection}.{suffix}");
                let (info, bytes) = self
                    .0
                    .bytes(&name)
                    .ok_or_else(|| error(format!("missing {name}")))?;
                if info.dtype != dtype || info.shape != [12288, columns] {
                    return Err(error(format!("{name}: invalid affine gate/up shape")));
                }
                parts.push(bytes);
            }
        }
        Ok(Weight {
            buffer: d.upload_parts(&parts)?,
            ty: crate::affine::AFFINE4,
            k: 4096,
            n: 24576,
        })
    }
}

pub(super) fn workspace(rows: usize) -> usize {
    // Covers fused gate/up, QKV, down, plus the narrow image input/output.
    [
        (4096, 24576),
        (4096, 4096),
        (4096, 1024),
        (12288, 4096),
        (64, 4096),
        (4096, 64),
    ]
    .into_iter()
    .map(|(k, n)| crate::affine::workspace_bytes(k, n, rows))
    .max()
    .unwrap_or(0)
    .max(rows.next_multiple_of(128) * 12288 * 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_contract_rejects_incompatible_packs() {
        let expected = json!({"mlx_format":true,
            "quantization":{"bits":4,"group_size":64,"mode":"affine"}});
        assert!(check_config(Path::new("config.json"), &expected, &expected).is_ok());
        for (key, value) in [
            ("bits", json!(8)),
            ("group_size", json!(128)),
            ("mode", json!("mxfp4")),
        ] {
            let mut wrong = expected.clone();
            wrong["quantization"][key] = value;
            assert!(check_config(Path::new("config.json"), &wrong, &expected).is_err());
        }
        assert!(check_config(Path::new("config.json"), &json!({}), &expected).is_err());
    }
}
