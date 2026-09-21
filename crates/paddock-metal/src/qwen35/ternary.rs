//! GGUF ternary storage and rotation, without transcoding into MLX or dense weights.
use super::*;
use paddock_models::{gguf::GgufFile, hadamard::HadamardSpec, mapped::MappedGguf};
use std::collections::{BTreeMap, HashSet};

pub(super) const PTQ1: u32 = 143;
const GROUPED: u32 = PTQ1 | 0x100;

fn error(message: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Bonsai PTQ1 GGUF: {}", message.into()))
}

pub(super) fn validate(gguf: &GgufFile, g: Geometry, nextn: u64) -> Result<Option<HadamardSpec>> {
    let Some(spec) = HadamardSpec::from_gguf(gguf).map_err(|e| error(e.to_string()))? else {
        return Ok(None);
    };
    if g != Geometry::DENSE_27B
        || nextn != 0
        || spec.block != 1024
        || !spec.embd_inverse
        || !spec.gdn_v_grouped
    {
        return Err(error(
            "requires dense 27B, block-1024 rotation, inverse embedding, grouped GDN and no MTP",
        ));
    }
    let mut expected = HashSet::from(["output.weight".to_owned()]);
    for i in 0..g.layers {
        let mixer: &[&str] = if (i + 1) % 4 == 0 {
            &["attn_q", "attn_k", "attn_v", "attn_output"]
        } else {
            &["attn_qkv", "attn_gate", "ssm_out"]
        };
        for kind in mixer.iter().chain(&["ffn_gate", "ffn_up", "ffn_down"]) {
            expected.insert(format!("blk.{i}.{kind}.weight"));
        }
    }
    if spec.weights != expected {
        return Err(error(
            "rotation weight_names do not match every supported linear",
        ));
    }
    for name in expected
        .iter()
        .map(String::as_str)
        .chain(["token_embd.weight"])
    {
        let tensor = gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| error(format!("missing {name}")))?;
        if tensor.raw_type != PTQ1 || tensor.dims.len() != 2 {
            return Err(error(format!("{name}: expected a packed PTQ1_0 matrix")));
        }
    }
    for tensor in &gguf.tensors {
        if tensor.raw_type == PTQ1
            && tensor.name != "token_embd.weight"
            && !expected.contains(&tensor.name)
        {
            return Err(error(format!(
                "{}: PTQ1 outside the declared rotated matrices",
                tensor.name
            )));
        }
    }
    for width in [g.width, g.heads * 256, g.ff] {
        spec.sign_vector(width).map_err(|e| error(e.to_string()))?;
    }
    Ok(Some(spec))
}

pub(super) fn load(
    device: &MetalDevice,
    map: &MappedGguf,
    name: &str,
    dims: &[usize],
) -> Result<Weight> {
    let (tensor, bytes) = map.tensor_bytes(name).map_err(|e| error(e.to_string()))?;
    if tensor.raw_type != PTQ1 {
        return Weight::load(device, map, name, dims);
    }
    if tensor
        .dims
        .iter()
        .copied()
        .ne(dims.iter().map(|&d| d as u64))
        || dims.len() != 2
        || !dims[0].is_multiple_of(128)
        || dims[0]
            .checked_mul(dims[1])
            .and_then(|n| (n / 128).checked_mul(28))
            != Some(bytes.len())
    {
        return Err(error(format!("{name}: invalid PTQ1 shape or byte count")));
    }
    Ok(Weight {
        buffer: device.upload(bytes)?,
        ty: if name.ends_with("ssm_out.weight") {
            GROUPED
        } else {
            PTQ1
        },
        k: dims[0],
        n: dims[1],
    })
}

pub(super) struct Ternary {
    signs: BTreeMap<usize, Buffer>,
    rotated: Buffer,
}

impl Ternary {
    pub(super) fn reserve_rows(&mut self, device: &MetalDevice, rows: usize) -> Result<()> {
        let bytes = rows
            .checked_mul(17408 * 4)
            .ok_or_else(|| MetalError::Memory("Bonsai rotation workspace overflow".into()))?;
        if bytes > self.rotated.len() {
            // Allocate first: a failed grant must preserve the working buffer.
            self.rotated = device.alloc(bytes)?;
        }
        Ok(())
    }

    pub(super) fn new(device: &MetalDevice, spec: &HadamardSpec) -> Result<Self> {
        let mut signs = BTreeMap::new();
        for width in [5120, 6144, 17408] {
            let values = spec.sign_vector(width).map_err(|e| error(e.to_string()))?;
            signs.insert(
                width,
                device.upload_parts(&[&values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()])?,
            );
        }
        Ok(Self {
            signs,
            rotated: device.alloc(CHUNK * 17408 * 4)?,
        })
    }
    pub(super) fn embed(
        &self,
        cmd: &Commands<'_>,
        w: &Weight,
        ids: &Buffer,
        out: &Buffer,
        rows: usize,
    ) {
        cmd.dispatch(
            "ptq1_embed",
            &[&w.buffer, ids, &self.signs[&w.k], out],
            &[w.k as u32, w.n as u32, rows as u32],
            [w.k / 1024, rows, 1],
            256,
        );
    }
    pub(super) fn project(
        &self,
        cmd: &Commands<'_>,
        planes: &[(&Weight, &Buffer)],
        input: &Buffer,
        rows: usize,
        workspace: &Buffer,
        head: bool,
    ) {
        if !planes.iter().any(|(w, _)| matches!(w.ty, PTQ1 | GROUPED)) {
            if !planes.iter().all(|(w, _)| w.ty == 30) {
                return projection::project(cmd, planes, input, rows, workspace);
            }
            for &(w, out) in planes {
                cmd.dispatch(
                    "ptq1_gate",
                    &[&w.buffer, input, out],
                    &[w.k as u32, w.n as u32, rows as u32],
                    [w.n, rows, 1],
                    256,
                );
            }
            return;
        }
        let k = planes[0].0.k;
        assert!(rows > 0 && rows * k * 4 <= self.rotated.len());
        assert!(
            planes
                .iter()
                .all(|(w, _)| w.k == k && w.ty == planes[0].0.ty)
        );
        cmd.dispatch(
            if planes[0].0.ty == GROUPED {
                "ptq1_rotate_grouped"
            } else {
                "bonsai_rotate"
            },
            &[input, &self.signs[&k], &self.rotated],
            &[k as u32, rows as u32],
            [k / 1024, rows, 1],
            256,
        );
        let all = [(0, rows, 1)];
        let spans = if head {
            &all[..]
        } else {
            cmd.projection_rows().unwrap_or(&all)
        };
        let mut end = 0;
        for &(first, count, role) in spans {
            assert!(count > 0 && first == end && count <= rows - first);
            end = first + count;
            for &(w, out) in planes {
                let (kernel, columns, tile) = if head || role == 1 {
                    (
                        [
                            "ptq1_vectors1",
                            "ptq1_vectors2",
                            "ptq1_vectors3",
                            "ptq1_vectors4",
                        ][count.min(4) - 1],
                        4,
                        count.min(4),
                    )
                } else if count <= 16 {
                    ("ptq1_mm16", 32, 16)
                } else if count <= 32 {
                    ("ptq1_mm32", 32, 32)
                } else {
                    ("ptq1_mm64", 32, 64)
                };
                cmd.dispatch_at(
                    kernel,
                    &[&w.buffer, &self.rotated, out],
                    &[0, first * k * 4, first * w.n * 4],
                    &[k as u32, w.n as u32, count as u32],
                    [w.n.div_ceil(columns), count.div_ceil(tile), 1],
                    128,
                );
            }
        }
        assert_eq!(end, rows);
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    #[test]
    fn ptq1_rotation_workspace_grows_transactionally_and_checks_overflow() {
        let device = MetalDevice::new(Some(2 << 20)).unwrap();
        let mut state = Ternary {
            signs: BTreeMap::new(),
            rotated: device.alloc(17408 * 4).unwrap(),
        };
        state.reserve_rows(&device, 4).unwrap();
        let bytes = state.rotated.len();
        let allocated = device.allocated_bytes();
        assert_eq!(bytes, 4 * 17408 * 4);
        state.reserve_rows(&device, 2).unwrap();
        assert!(state.reserve_rows(&device, 128).is_err());
        assert!(state.reserve_rows(&device, usize::MAX).is_err());
        assert_eq!(state.rotated.len(), bytes);
        assert_eq!(device.allocated_bytes(), allocated);
    }
}
