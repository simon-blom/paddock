//! Audited schema-3 Splash package reader. No Python, checkpoint execution,
//! requantization, or temporary expanded checkpoint. See splash/NOTICE.md.
use crate::safetensors::StError;
use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

pub const REVISION: &str = "9d27070b71f7142c6b6025f03ac011d70a73cb48";
pub const MANIFEST_SHA256: &str =
    "40121d0d933fb27a7206248a592451f9ebbd73671f19ce3162cf27a7db0968d5";
const ALIGN: usize = 16384;
mod vision;
pub use vision::Vision;
fn bad(message: impl Into<String>) -> StError {
    StError::Header(format!("Splash: {}", message.into()))
}
fn align(n: usize) -> Result<usize, StError> {
    n.checked_add(ALIGN - 1)
        .map(|n| n & !(ALIGN - 1))
        .ok_or_else(|| bad("section overflow"))
}
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Deserialize)]
pub struct Artifact {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    artifacts: Vec<Artifact>,
}

/// Only the audited dense package is admitted. Pinning covers all geometry,
/// provenance and tensor semantics, including precombined RMS weights.
pub fn inventory(root: &Path) -> Result<Vec<Artifact>, StError> {
    let file = root.join("manifest.json");
    if std::fs::metadata(&file)?.len() > 1 << 20 {
        return Err(bad("manifest too large"));
    }
    let bytes = std::fs::read(file)?;
    if sha(&bytes) != MANIFEST_SHA256 {
        return Err(bad("unsupported or changed package manifest"));
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| bad(e.to_string()))?;
    Ok(manifest.artifacts)
}

pub fn files(root: &Path) -> Result<Vec<PathBuf>, StError> {
    let mut paths = vec![root.join("manifest.json")];
    paths.extend(inventory(root)?.into_iter().map(|a| root.join(a.path)));
    Ok(paths)
}

pub fn tokenizer_dir(root: &Path) -> Result<PathBuf, StError> {
    for a in inventory(root)?
        .iter()
        .filter(|a| a.path.starts_with("tokenizer/"))
    {
        let path = root.join(&a.path);
        if std::fs::metadata(&path)?.len() != a.size
            || a.size > 32 << 20
            || sha(&std::fs::read(&path)?) != a.sha256
        {
            return Err(bad(format!("{} tokenizer integrity mismatch", a.path)));
        }
    }
    Ok(root.join("tokenizer"))
}

#[derive(Clone)]
enum Layout {
    Affine {
        codes: Range<usize>,
        scales: Range<usize>,
        biases: Range<usize>,
        stored_n: usize,
        first: usize,
        tiled: bool,
    },
    Small {
        data: Range<usize>,
        f32: bool,
    },
}

#[derive(Clone)]
pub struct Tensor {
    map: Arc<Mmap>,
    dims: Vec<usize>,
    layout: Layout,
}
impl Tensor {
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn affine(&self) -> bool {
        matches!(self.layout, Layout::Affine { .. })
    }
    pub fn bf16_bytes(&self) -> Option<&[u8]> {
        match &self.layout {
            Layout::Small { data, f32: false } => Some(&self.map[data.clone()]),
            _ => None,
        }
    }
    pub fn output_bytes(&self) -> usize {
        let count: usize = self.dims.iter().product();
        if self.affine() {
            count / 16 * 9
        } else {
            count * 4
        }
    }
    pub fn tiled_bytes(&self) -> usize {
        self.dims[0] * self.dims[1].div_ceil(256) * 256 / 16 * 9
    }
    pub fn copy_tiled(&self, out: &mut [u8]) -> Result<(), StError> {
        let Layout::Affine {
            codes,
            scales,
            biases,
            stored_n,
            first,
            tiled,
        } = &self.layout
        else {
            return Err(bad("expected affine"));
        };
        if out.len() != self.tiled_bytes() {
            return Err(bad("tiled destination size mismatch"));
        }
        let (k, n) = (self.dims[0], self.dims[1]);
        let padded = n.div_ceil(256) * 256;
        let groups = k / 64;
        if *tiled && *first == 0 && n == padded && *stored_n == padded {
            // Already in the GPU's native tile order. Copy complete planes,
            // avoiding millions of tiny row/group copies at model startup.
            let (w, params) = out.split_at_mut(k * padded / 2);
            let (s, b) = params.split_at_mut(k * padded / 32);
            w.copy_from_slice(&self.map[codes.clone()]);
            s.copy_from_slice(&self.map[scales.clone()]);
            b.copy_from_slice(&self.map[biases.clone()]);
            return Ok(());
        }
        out.fill(0);
        for row in 0..n {
            let r = row + first;
            if r >= *stored_n {
                return Err(bad("tiled source row out of bounds"));
            }
            for g in 0..groups {
                let src = if *tiled {
                    (r / 256 * groups + g) * 256 + r % 256
                } else {
                    r * groups + g
                };
                let dst = (row / 256 * groups + g) * 256 + row % 256;
                out[dst * 32..dst * 32 + 32].copy_from_slice(
                    &self.map[codes.start + src * 32..codes.start + src * 32 + 32],
                );
                let s = k * padded / 2 + dst * 2;
                let b = k * padded / 2 + k * padded / 32 + dst * 2;
                out[s..s + 2]
                    .copy_from_slice(&self.map[scales.start + src * 2..scales.start + src * 2 + 2]);
                out[b..b + 2]
                    .copy_from_slice(&self.map[biases.start + src * 2..biases.start + src * 2 + 2]);
            }
        }
        Ok(())
    }
    /// Byte-only tile permutation into the existing Metal affine planes.
    /// Small BF16 tensors widen exactly. Sanitized RMS weights are already
    /// multiplicative scales, just like the existing native MLX checkpoint.
    pub fn copy_native(&self, out: &mut [u8]) -> Result<(), StError> {
        if out.len() != self.output_bytes() {
            return Err(bad("output size mismatch"));
        }
        match &self.layout {
            Layout::Small { data, f32 } => {
                let width = if *f32 { 4 } else { 2 };
                let (words, _) = out.as_chunks_mut::<4>();
                for (src, dst) in self.map[data.clone()].chunks_exact(width).zip(words) {
                    let value = if *f32 {
                        f32::from_le_bytes([src[0], src[1], src[2], src[3]])
                    } else {
                        f32::from_bits((u16::from_le_bytes([src[0], src[1]]) as u32) << 16)
                    };
                    if !value.is_finite() {
                        return Err(bad("nonfinite small weight"));
                    }
                    *dst = value.to_le_bytes();
                }
            }
            Layout::Affine {
                codes,
                scales,
                biases,
                stored_n,
                first,
                tiled,
            } => {
                let (k, n) = (self.dims[0], self.dims[1]);
                let groups = k / 64;
                let (w, params) = out.split_at_mut(k * n / 2);
                let (s, b) = params.split_at_mut(k * n / 32);
                for row in 0..n {
                    let source_row = row + first;
                    if source_row >= *stored_n {
                        return Err(bad("affine row out of bounds"));
                    }
                    for g in 0..groups {
                        let src = if *tiled {
                            (source_row / 256 * groups + g) * 256 + source_row % 256
                        } else {
                            source_row * groups + g
                        };
                        let dst = row * groups + g;
                        w[dst * 32..dst * 32 + 32].copy_from_slice(
                            &self.map[codes.start + src * 32..codes.start + src * 32 + 32],
                        );
                        s[dst * 2..dst * 2 + 2].copy_from_slice(
                            &self.map[scales.start + src * 2..scales.start + src * 2 + 2],
                        );
                        b[dst * 2..dst * 2 + 2].copy_from_slice(
                            &self.map[biases.start + src * 2..biases.start + src * 2 + 2],
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

struct Packed {
    map: Arc<Mmap>,
    offset: usize,
}
impl Packed {
    fn open(
        root: &Path,
        a: &Artifact,
        magic: &[u8; 8],
        layer: u32,
        kind: u32,
    ) -> Result<Self, StError> {
        let file = File::open(root.join(&a.path))?;
        if !file.metadata()?.is_file() || file.metadata()?.len() != a.size {
            return Err(bad(format!("{} size mismatch", a.path)));
        }
        // SAFETY: read-only mapping; callers must not mutate checkpoints while
        // loaded, as with the existing GGUF/safetensors mapping contract.
        let map = unsafe { Mmap::map(&file)? };
        validate_header(&map, magic, layer, kind)?;
        if sha(&map) != a.sha256 {
            return Err(bad(format!("{} SHA-256 mismatch", a.path)));
        }
        Ok(Self {
            map: Arc::new(map),
            offset: 16,
        })
    }
    fn section(&mut self, bytes: usize) -> Result<Range<usize>, StError> {
        let start = align(self.offset)?;
        let end = start
            .checked_add(bytes)
            .filter(|&n| bytes > 0 && n <= self.map.len())
            .ok_or_else(|| bad("truncated section"))?;
        self.offset = end;
        Ok(start..end)
    }
    fn small(&mut self, dims: &[usize], f32: bool) -> Result<Tensor, StError> {
        let data = self.section(dims.iter().product::<usize>() * if f32 { 4 } else { 2 })?;
        Ok(Tensor {
            map: self.map.clone(),
            dims: dims.to_vec(),
            layout: Layout::Small { data, f32 },
        })
    }
    fn affine(&mut self, k: usize, n: usize, components: bool) -> Result<Tensor, StError> {
        if k == 0 || !k.is_multiple_of(64) || n == 0 || (!components && !n.is_multiple_of(256)) {
            return Err(bad("invalid affine geometry"));
        }
        let elements = k
            .checked_mul(n)
            .ok_or_else(|| bad("affine size overflow"))?;
        let (codes, scales, biases) = if components {
            (
                self.section(elements / 2)?,
                self.section(elements / 32)?,
                self.section(elements / 32)?,
            )
        } else {
            let all = self.section(elements / 16 * 9)?;
            let w = all.start..all.start + elements / 2;
            let s = w.end..w.end + elements / 32;
            let b = s.end..all.end;
            (w, s, b)
        };
        Ok(Tensor {
            map: self.map.clone(),
            dims: vec![k, n],
            layout: Layout::Affine {
                codes,
                scales,
                biases,
                stored_n: n,
                first: 0,
                tiled: !components,
            },
        })
    }
    fn finish(self) -> Result<(), StError> {
        if align(self.offset)? != self.map.len() {
            return Err(bad("unconsumed packed sections"));
        }
        Ok(())
    }
}

fn validate_header(bytes: &[u8], magic: &[u8; 8], layer: u32, kind: u32) -> Result<(), StError> {
    if bytes.len() < ALIGN
        || !bytes.len().is_multiple_of(ALIGN)
        || &bytes[..8] != magic
        || u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) != layer
        || u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) != kind
    {
        return Err(bad("invalid packed size, magic, layer or type"));
    }
    Ok(())
}

fn rows(t: &Tensor, first: usize, n: usize) -> Result<Tensor, StError> {
    let mut result = t.clone();
    let Layout::Affine {
        first: start,
        stored_n,
        ..
    } = &mut result.layout
    else {
        return Err(bad("expected affine projection"));
    };
    if first.checked_add(n).is_none_or(|end| end > *stored_n) || n == 0 {
        return Err(bad("invalid fused projection slice"));
    }
    *start = first;
    result.dims[1] = n;
    Ok(result)
}

pub struct Target {
    tensors: HashMap<String, Tensor>,
    bytes: u64,
}
impl Target {
    pub fn open(root: &Path) -> Result<Self, StError> {
        let entries = inventory(root)?;
        let find = |name: &str| {
            entries
                .iter()
                .find(|a| a.path == name)
                .ok_or_else(|| bad(format!("missing {name}")))
        };
        let mut tensors = HashMap::new();
        let mut bytes = 0;
        for i in 0..64 {
            let full = (i + 1) % 4 == 0;
            let a = find(&format!("target/layer-{i}.bin"))?;
            bytes += a.size;
            let mut file = Packed::open(root, a, b"MDFL0006", i, u32::from(full))?;
            let mut insert = |name: &str, t| {
                tensors.insert(format!("blk.{i}.{name}"), t);
            };
            insert("attn_norm.weight", file.small(&[5120], false)?);
            let input = file.affine(5120, if full { 14336 } else { 16640 }, false)?;
            if full {
                for (name, first, n) in [
                    ("attn_q.weight", 0, 12288),
                    ("attn_k.weight", 12288, 1024),
                    ("attn_v.weight", 13312, 1024),
                ] {
                    insert(name, rows(&input, first, n)?);
                }
                insert("attn_q_norm.weight", file.small(&[256], false)?);
                insert("attn_k_norm.weight", file.small(&[256], false)?);
                insert("attn_output.weight", file.affine(6144, 5120, false)?);
            } else {
                for (name, first, n) in [
                    ("attn_qkv.weight", 0, 10240),
                    ("attn_gate.weight", 10240, 6144),
                    ("ssm_beta.weight", 16384, 48),
                    ("ssm_alpha.weight", 16432, 48),
                ] {
                    insert(name, rows(&input, first, n)?);
                }
                insert("ssm_conv1d.weight", file.small(&[4, 10240], false)?);
                insert("ssm_a", file.small(&[48], true)?);
                insert("ssm_dt.bias", file.small(&[48], false)?);
                insert("ssm_norm.weight", file.small(&[128], false)?);
                insert("ssm_out.weight", file.affine(6144, 5120, false)?);
            }
            insert("post_attention_norm.weight", file.small(&[5120], false)?);
            insert("ffn_gate.weight", file.affine(5120, 17408, false)?);
            insert("ffn_up.weight", file.affine(5120, 17408, false)?);
            insert("ffn_down.weight", file.affine(17408, 5120, false)?);
            file.finish()?;
        }
        let a = find("target/head.bin")?;
        bytes += a.size;
        let mut file = Packed::open(root, a, b"MDFL0002", 64, 2)?;
        tensors.insert("output_norm.weight".into(), file.small(&[5120], false)?);
        tensors.insert("output.weight".into(), file.affine(5120, 248320, false)?);
        file.finish()?;
        let a = find("target/embedding.bin")?;
        bytes += a.size;
        let mut file = Packed::open(root, a, b"MDFE0001", 248320, 5120)?;
        tensors.insert("token_embd.weight".into(), file.affine(5120, 248320, true)?);
        file.finish()?;
        Ok(Self { tensors, bytes })
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor, StError> {
        self.tensors
            .get(name)
            .ok_or_else(|| bad(format!("missing {name}")))
    }
    pub fn total_len(&self) -> u64 {
        self.bytes
    }
}

pub struct Draft {
    tensors: HashMap<String, Tensor>,
}
impl Draft {
    pub fn open(root: &Path) -> Result<Self, StError> {
        let entries = inventory(root)?;
        let find = |name: &str| {
            entries
                .iter()
                .find(|a| a.path == name)
                .ok_or_else(|| bad(format!("missing {name}")))
        };
        let mut tensors = HashMap::new();
        for i in 0..5 {
            let mut f = Packed::open(
                root,
                find(&format!("draft/layer-{i}.bin"))?,
                b"MDFD0004",
                i,
                0,
            )?;
            let mut put = |name: &str, t| {
                tensors.insert(format!("blk.{i}.{name}"), t);
            };
            put("attn_norm.weight", f.small(&[5120], false)?);
            put("attn_conv_base", f.small(&[5120, 2, 2], false)?);
            put("attn_conv_proj.weight", f.affine(5120, 1280, false)?);
            let qkv = f.affine(5120, 6144, false)?;
            for (name, start, n) in [
                ("attn_q.weight", 0, 4096),
                ("attn_k.weight", 4096, 1024),
                ("attn_v.weight", 5120, 1024),
            ] {
                put(name, rows(&qkv, start, n)?);
            }
            put("attn_q_norm.weight", f.small(&[128], false)?);
            put("attn_k_norm.weight", f.small(&[128], false)?);
            put("attn_output.weight", f.affine(4096, 5120, false)?);
            put("ffn_norm.weight", f.small(&[5120], false)?);
            put("ffn_conv_base", f.small(&[5120, 2, 2], false)?);
            put("ffn_conv_proj.weight", f.affine(5120, 1280, false)?);
            put("ffn_gate.weight", f.affine(5120, 17408, false)?);
            put("ffn_up.weight", f.affine(5120, 17408, false)?);
            put("ffn_down.weight", f.affine(17408, 5120, false)?);
            f.finish()?;
        }
        let mut f = Packed::open(root, find("draft/model.bin")?, b"MDFD0004", 5, 1)?;
        tensors.insert("fc.weight".into(), f.affine(25600, 5120, false)?);
        tensors.insert("enc.output_norm.weight".into(), f.small(&[5120], false)?);
        tensors.insert("output_norm.weight".into(), f.small(&[5120], false)?);
        tensors.insert("selector_hidden.weight".into(), f.affine(5120, 256, false)?);
        tensors.insert(
            "selector_predecessor.weight".into(),
            f.small(&[256, 248320], false)?,
        );
        tensors.insert(
            "selector_successor.weight".into(),
            f.small(&[256, 248320], false)?,
        );
        f.finish()?;
        Ok(Self { tensors })
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor, StError> {
        self.tensors
            .get(name)
            .ok_or_else(|| bad(format!("missing {name}")))
    }
}

#[cfg(test)]
#[path = "splash/tests.rs"]
mod tests;
