//! Opt-in same-input DiT operator captures. Copies stay on the command stream;
//! host reads occur only after its normal completion fence. Not a timing path.
use super::*;
use std::{cell::RefCell, path::PathBuf};

pub(super) struct BlockTrace<'a> {
    device: &'a MetalDevice,
    directory: PathBuf,
    buffers: RefCell<Vec<(String, Buffer, bool, usize)>>,
}

impl<'a> BlockTrace<'a> {
    pub fn new(
        device: &'a MetalDevice,
        kind: &str,
        metadata: serde_json::Value,
    ) -> Result<Option<Self>> {
        let Some(root) = std::env::var_os("PADDOCK_QI_DIT_TRACE") else {
            return Ok(None);
        };
        let directory = Path::new(&root).join(kind);
        // One immutable capture per run, never overwrite the first step with
        // another sigma or a later layer/conditioning branch.
        if directory.exists() {
            return Ok(None);
        }
        std::fs::create_dir(&directory).map_err(model_error)?;
        std::fs::write(
            directory.join("metadata.json"),
            serde_json::to_vec(&metadata).map_err(model_error)?,
        )
        .map_err(model_error)?;
        Ok(Some(Self {
            device,
            directory,
            buffers: RefCell::new(Vec::new()),
        }))
    }

    pub fn snapshot(
        &self,
        cmd: &Commands<'_>,
        name: &str,
        buffer: &Buffer,
        count: usize,
        half: bool,
    ) -> Result<()> {
        let bytes = count * if half { 2 } else { 4 };
        assert!(bytes <= buffer.len() && bytes.is_multiple_of(4));
        let copy = self.device.alloc(bytes)?;
        cmd.dispatch(
            "uov_copy",
            &[buffer, &copy],
            &[(bytes / 4) as u32],
            [(bytes / 4).div_ceil(256), 1, 1],
            256,
        );
        self.buffers
            .borrow_mut()
            .push((name.into(), copy, half, count));
        Ok(())
    }

    pub fn save(self) -> Result<()> {
        for (name, buffer, half, count) in self.buffers.into_inner() {
            let values = if half {
                unsafe { buffer.read_u32(count / 2) }
                    .into_iter()
                    .flat_map(|v| [v as u16, (v >> 16) as u16])
                    .map(|v| half::f16::from_bits(v).to_f32())
                    .collect()
            } else {
                unsafe { buffer.read_f32(0, count) }
            };
            assert!(values.iter().all(|v| v.is_finite()), "nonfinite {name}");
            std::fs::write(
                self.directory.join(format!("{name}.f32")),
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .map_err(model_error)?;
        }
        Ok(())
    }
}
