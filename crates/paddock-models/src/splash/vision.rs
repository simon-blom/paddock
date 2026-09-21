//! BF16 vision sections mapped to the canonical Qwen tower tensor order.
use super::*;

struct Plane {
    map: Arc<Mmap>,
    data: Range<usize>,
    dims: Vec<usize>,
    stored_k: usize,
    k: usize,
    n: usize,
    temporal: Option<usize>,
}
impl Plane {
    fn copy(&self, out: &mut [u8], half: bool) -> Result<(), StError> {
        let size = if half { 2 } else { 4 };
        if out.len() != self.k * self.n * size {
            return Err(bad("vision output size mismatch"));
        }
        for row in 0..self.n {
            for col in 0..self.k {
                // The patch encoder is [output, channel, time, y, x]. Each
                // still image supplies both temporal planes independently.
                let c = self
                    .temporal
                    .map_or(col, |t| col / 256 * 512 + t * 256 + col % 256);
                let offset = self.data.start + (row * self.stored_k + c) * 2;
                let bits = u16::from_le_bytes([self.map[offset], self.map[offset + 1]]);
                if bits & 0x7f80 == 0x7f80 {
                    return Err(bad("nonfinite vision weight"));
                }
                let dst = (row * self.k + col) * size;
                if half {
                    out[dst..dst + 2].copy_from_slice(&bits.to_le_bytes());
                } else {
                    out[dst..dst + 4].copy_from_slice(&((bits as u32) << 16).to_le_bytes());
                }
            }
        }
        Ok(())
    }
}

pub struct Vision {
    planes: HashMap<String, Plane>,
}
impl Vision {
    pub fn open(root: &Path) -> Result<Self, StError> {
        let entries = inventory(root)?;
        let a = entries
            .iter()
            .find(|a| a.path == "vision/model.bin")
            .ok_or_else(|| bad("missing vision"))?;
        let mut f = Packed::open(root, a, b"MDFV0001", 27, 0)?;
        let mut planes = HashMap::new();
        let patch = f.section(1152 * 1536 * 2)?;
        for t in 0..2 {
            planes.insert(
                if t == 0 {
                    "v.patch_embd.weight".into()
                } else {
                    "v.patch_embd.weight.1".into()
                },
                Plane {
                    map: f.map.clone(),
                    data: patch.clone(),
                    dims: vec![16, 16, 3, 1152],
                    stored_k: 1536,
                    k: 768,
                    n: 1152,
                    temporal: Some(t),
                },
            );
        }
        let mut put =
            |name: &str, dims: &[usize], stored_k: usize, stored_n: usize| -> Result<(), StError> {
                let k = dims[0];
                let n = *dims.get(1).unwrap_or(&1);
                let data = f.section(stored_k * stored_n * 2)?;
                planes.insert(
                    name.into(),
                    Plane {
                        map: f.map.clone(),
                        data,
                        dims: dims.to_vec(),
                        stored_k,
                        k,
                        n,
                        temporal: None,
                    },
                );
                Ok(())
            };
        put("v.patch_embd.bias", &[1152], 1152, 1)?;
        put("v.position_embd.weight", &[1152, 2304], 1152, 2304)?;
        for i in 0..27 {
            for (name, k, n, sk, sn) in [
                ("ln1.weight", 1152, 1, 1152, 1),
                ("ln1.bias", 1152, 1, 1152, 1),
                ("attn_qkv.weight", 1152, 3456, 1152, 3456),
                ("attn_qkv.bias", 3456, 1, 3456, 1),
                ("attn_out.weight", 1152, 1152, 1152, 1152),
                ("attn_out.bias", 1152, 1, 1152, 1),
                ("ln2.weight", 1152, 1, 1152, 1),
                ("ln2.bias", 1152, 1, 1152, 1),
                ("ffn_up.weight", 1152, 4304, 1152, 4352),
                ("ffn_up.bias", 4304, 1, 4352, 1),
                ("ffn_down.weight", 4304, 1152, 4352, 1152),
                ("ffn_down.bias", 1152, 1, 1152, 1),
            ] {
                let dims = if n == 1 { vec![k] } else { vec![k, n] };
                put(&format!("v.blk.{i}.{name}"), &dims, sk, sn)?;
            }
        }
        for (name, k, n) in [
            ("v.post_ln.weight", 1152, 1),
            ("v.post_ln.bias", 1152, 1),
            ("mm.0.weight", 4608, 4608),
            ("mm.0.bias", 4608, 1),
            ("mm.2.weight", 4608, 5120),
            ("mm.2.bias", 5120, 1),
        ] {
            let dims = if n == 1 { vec![k] } else { vec![k, n] };
            put(name, &dims, k, n)?;
        }
        f.finish()?;
        Ok(Self { planes })
    }
    pub fn copy(
        &self,
        name: &str,
        dims: &[usize],
        half: bool,
        out: &mut [u8],
    ) -> Result<(), StError> {
        let p = self
            .planes
            .get(name)
            .ok_or_else(|| bad(format!("missing {name}")))?;
        if p.dims != dims {
            return Err(bad(format!("vision {name} dimensions mismatch")));
        }
        p.copy(out, half)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_and_temporal_channels_are_not_pixels() {
        let mut map = memmap2::MmapMut::map_anon(1536 * 2).unwrap();
        for (i, b) in map.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            *b = (0x3f00 + (i % 256) as u16).to_le_bytes();
        }
        let p = Plane {
            map: Arc::new(map.make_read_only().unwrap()),
            data: 0..3072,
            dims: vec![768, 1],
            stored_k: 1536,
            k: 768,
            n: 1,
            temporal: Some(1),
        };
        let mut bytes = vec![0; 1536];
        p.copy(&mut bytes, true).unwrap();
        for c in 0..3 {
            assert_eq!(
                &bytes[c * 512..c * 512 + 512],
                &p.map[c * 1024 + 512..c * 1024 + 1024]
            );
        }
        let mut p = p;
        p.k = 48;
        p.temporal = None;
        p.copy(&mut [0; 96], true).unwrap();
        assert!(p.copy(&mut [0; 95], true).is_err());
    }
    #[test]
    #[ignore = "requires PADDOCK_SPLASH_MODEL and PADDOCK_METAL_MLX_MODEL"]
    fn packed_vision_matches_checkpoint() {
        let packed =
            Vision::open(Path::new(&std::env::var("PADDOCK_SPLASH_MODEL").unwrap())).unwrap();
        let mlx = crate::safetensors::ShardedSafetensors::open_dir(Path::new(
            &std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap(),
        ))
        .unwrap();
        for (name, p) in &packed.planes {
            let key = match name.as_str() {
                "v.patch_embd.weight" | "v.patch_embd.weight.1" => {
                    "vision_tower.patch_embed.proj.weight".into()
                }
                "v.patch_embd.bias" => "vision_tower.patch_embed.proj.bias".into(),
                "v.position_embd.weight" => "vision_tower.pos_embed.weight".into(),
                "v.post_ln.weight" => "vision_tower.merger.norm.weight".into(),
                "v.post_ln.bias" => "vision_tower.merger.norm.bias".into(),
                n if n.starts_with("mm.") => n
                    .replacen("mm.0.", "vision_tower.merger.linear_fc1.", 1)
                    .replacen("mm.2.", "vision_tower.merger.linear_fc2.", 1),
                n => n
                    .replacen("v.blk.", "vision_tower.blocks.", 1)
                    .replace("ln1.", "norm1.")
                    .replace("ln2.", "norm2.")
                    .replace("attn_qkv.", "attn.qkv.")
                    .replace("attn_out.", "attn.proj.")
                    .replace("ffn_up.", "mlp.linear_fc1.")
                    .replace("ffn_down.", "mlp.linear_fc2."),
            };
            let (info, source) = mlx.bytes(&key).unwrap();
            let mut out = vec![0; p.k * p.n * 2];
            p.copy(&mut out, true).unwrap();
            if let Some(time) = p.temporal {
                // MLX conv3d is channels-last [out,time,y,x,channel].
                assert_eq!(info.shape, vec![1152, 2, 16, 16, 3]);
                for row in 0..1152 {
                    for c in 0..3 {
                        for spatial in 0..256 {
                            let src = (((row * 2 + time) * 256 + spatial) * 3 + c) * 2;
                            let dst = (row * 768 + c * 256 + spatial) * 2;
                            assert_eq!(
                                &out[dst..dst + 2],
                                &source[src..src + 2],
                                "{name} patch coordinate"
                            );
                        }
                    }
                }
            } else {
                assert!(out == source, "vision tensor differs: {name} / {key}");
            }
        }
    }
}
