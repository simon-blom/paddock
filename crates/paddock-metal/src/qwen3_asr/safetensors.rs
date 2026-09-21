//! Strict BF16 ingestion: raw planes stay BF16; small vectors widen on GPU.
//! Validate the complete inventory, spans and nonfinite bit patterns before
//! allocating anything. This is storage validation, never host model math.
use super::*;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};
pub(super) type Schema = Vec<(String, Vec<usize>)>;
pub(super) struct Weights {
    st: ShardedSafetensors,
    pub bytes: u64,
}
impl Weights {
    pub fn open(dir: &Path, schema: &Schema) -> Result<Self> {
        let st = ShardedSafetensors::open_dir(dir).map_err(|e| error(e.to_string()))?;
        if st.names().count() != schema.len() {
            return Err(error("aligner: unexpected tensor inventory"));
        }
        let mut total = 0;
        for (name, shape) in schema {
            let (t, b) = st
                .bytes(name)
                .ok_or_else(|| error(format!("missing {name}")))?;
            let expected = shape.iter().try_fold(2usize, |n, &v| n.checked_mul(v));
            if t.dtype != StDtype::Bf16 || t.shape != *shape || expected != Some(b.len()) {
                return Err(error(format!("{name}: requires BF16 {shape:?}")));
            }
            if b.chunks_exact(2)
                .any(|p| u16::from_le_bytes([p[0], p[1]]) & 0x7f80 == 0x7f80)
            {
                return Err(error(format!("{name}: nonfinite BF16 weight")));
            }
            total += b.len() as u64;
        }
        Ok(Self { st, bytes: total })
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.st
            .bytes(name)
            .map(|(_, b)| b)
            .ok_or_else(|| error(format!("missing {name}")))
    }
    pub fn plane(&self, d: &MetalDevice, name: &str, k: usize, n: usize) -> Result<Weight> {
        Ok(Weight {
            buffer: d.upload(self.bytes(name)?)?,
            ty: 30,
            k,
            n,
        })
    }
    pub fn fused(
        &self,
        d: &MetalDevice,
        names: &[String],
        k: usize,
        n: usize,
        widen: bool,
    ) -> Result<Weight> {
        let parts = names
            .iter()
            .map(|n| self.bytes(n))
            .collect::<Result<Vec<_>>>()?;
        let raw = d.upload_parts(&parts)?;
        if !widen {
            return Ok(Weight {
                buffer: raw,
                ty: 30,
                k,
                n,
            });
        }
        let out = d.alloc(raw.len() * 2)?;
        let c = d.begin()?;
        c.dispatch(
            "qalign_widen",
            &[&raw, &out],
            &[(raw.len() / 2) as u32],
            [(raw.len() / 2).div_ceil(256), 1, 1],
            256,
        );
        c.finish()?;
        Ok(Weight {
            buffer: out,
            ty: 0,
            k,
            n,
        })
    }
    pub fn vector(&self, d: &MetalDevice, name: &str, n: usize) -> Result<Weight> {
        self.fused(d, &[name.to_owned()], n, 1, true)
    }
}
