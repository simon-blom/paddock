//! Storage adapter: one dense graph, original GGUF or packed affine MLX.
//! Only descriptors are normalized in memory; no checkpoint conversion,
//! repacking, requantization or persistent expanded projection weights.
use super::*;
use paddock_models::{
    ggml_type::GgmlType,
    gguf::{GgufFile, TensorInfo},
    mlx::MiniCpmConfig,
    safetensors::{ShardedSafetensors, StDtype},
};
use std::collections::{HashMap, HashSet};

pub(super) enum Source {
    Gguf(MappedGguf),
    Mlx {
        map: ShardedSafetensors,
        header: GgufFile,
        names: HashMap<String, String>,
    },
}
impl Source {
    pub(super) fn open(path: &Path) -> Result<Self> {
        if !path.is_dir() {
            return MappedGguf::open(path)
                .map(Self::Gguf)
                .map_err(|e| MetalError::Model(e.to_string()));
        }
        let cfg = MiniCpmConfig::read(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let map =
            ShardedSafetensors::open_dir(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let mut header = GgufFile {
            version: 3,
            alignment: 32,
            data_offset: 0,
            metadata: HashMap::new(),
            tensors: Vec::new(),
        };
        header
            .metadata
            .insert("general.architecture".into(), Value::Str("llama".into()));
        for (key, n) in [
            ("embedding_length", 2048),
            ("attention.head_count", 16),
            ("attention.head_count_kv", 2),
            ("block_count", 42),
            ("feed_forward_length", 6144),
            ("context_length", cfg.context),
            ("rope.dimension_count", 128),
        ] {
            header
                .metadata
                .insert(format!("llama.{key}"), Value::U64(n as u64));
        }
        for (key, n) in [
            ("attention.layer_norm_rms_epsilon", cfg.eps),
            ("rope.freq_base", cfg.rope),
        ] {
            header
                .metadata
                .insert(format!("llama.{key}"), Value::F32(n));
        }
        let mut names = HashMap::new();
        let mut expected = HashSet::new();
        let mut add = |name: String, base: String, dims: &[u64]| -> Result<()> {
            let affine = dims.len() == 2;
            for suffix in if affine {
                &["weight", "scales", "biases"][..]
            } else {
                &["weight"][..]
            } {
                let key = format!("{base}.{suffix}");
                let (info, _) = map
                    .bytes(&key)
                    .ok_or_else(|| MetalError::Model(format!("missing {key}")))?;
                let shape = if affine {
                    vec![
                        dims[1] as usize,
                        dims[0] as usize / if *suffix == "weight" { 8 } else { 64 },
                    ]
                } else {
                    vec![dims[0] as usize]
                };
                let dtype = if affine && *suffix == "weight" {
                    StDtype::U32
                } else {
                    StDtype::Bf16
                };
                if info.shape != shape || info.dtype != dtype {
                    return Err(MetalError::Model(format!(
                        "{key}: expected {dtype:?} {shape:?}, got {:?} {:?}",
                        info.dtype, info.shape
                    )));
                }
                expected.insert(key);
            }
            header.tensors.push(TensorInfo {
                name: name.clone(),
                dims: dims.to_vec(),
                ggml_type: GgmlType::F32,
                raw_type: 0,
                offset: 0,
            });
            names.insert(name, format!("{base}.weight"));
            Ok(())
        };
        add(
            "token_embd.weight".into(),
            "model.embed_tokens".into(),
            &[2048, 130560],
        )?;
        add("output.weight".into(), "lm_head".into(), &[2048, 130560])?;
        add("output_norm.weight".into(), "model.norm".into(), &[2048])?;
        for i in 0..42 {
            for (dst, src, dims) in [
                ("attn_norm", "input_layernorm", vec![2048]),
                ("ffn_norm", "post_attention_layernorm", vec![2048]),
                ("attn_q", "self_attn.q_proj", vec![2048, 2048]),
                ("attn_k", "self_attn.k_proj", vec![2048, 256]),
                ("attn_v", "self_attn.v_proj", vec![2048, 256]),
                ("attn_output", "self_attn.o_proj", vec![2048, 2048]),
                ("ffn_gate", "mlp.gate_proj", vec![2048, 6144]),
                ("ffn_up", "mlp.up_proj", vec![2048, 6144]),
                ("ffn_down", "mlp.down_proj", vec![6144, 2048]),
            ] {
                add(
                    format!("blk.{i}.{dst}.weight"),
                    format!("model.layers.{i}.{src}"),
                    &dims,
                )?;
            }
        }
        let mut extra: Vec<_> = map.names().filter(|n| !expected.contains(*n)).collect();
        extra.sort();
        if !extra.is_empty() {
            return Err(MetalError::Model(format!(
                "MiniCPM MLX unconsumed tensors: {:?}",
                &extra[..extra.len().min(8)]
            )));
        }
        Ok(Self::Mlx { map, header, names })
    }
    pub(super) fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx { .. })
    }
    pub(super) fn gguf(&self) -> &GgufFile {
        match self {
            Self::Gguf(m) => m.gguf(),
            Self::Mlx { header, .. } => header,
        }
    }
    pub(super) fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.gguf().tensors.iter().find(|t| t.name == name)
    }
    pub(super) fn weight_budget_bytes(&self) -> u64 {
        match self {
            Self::Gguf(m) => m.total_len(),
            Self::Mlx { map, header, .. } => {
                // Norms are the only expanded tensors (BF16 -> F32). Include
                // that delta in admission, not just in the allocation ledger.
                // Retaining file-header bytes makes this a conservative bound
                // with room for each small temporary norm upload.
                map.total_len()
                    + header
                        .tensors
                        .iter()
                        .filter(|t| t.dims.len() == 1)
                        .map(|t| t.dims[0] * 2)
                        .sum::<u64>()
            }
        }
    }
    pub(super) fn load(&self, d: &MetalDevice, name: &str, dims: &[usize]) -> Result<Weight> {
        let Self::Mlx { map, names, .. } = self else {
            let Self::Gguf(m) = self else { unreachable!() };
            return Weight::load(d, m, name, dims);
        };
        let key = names
            .get(name)
            .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
        if dims.len() == 2 {
            return crate::affine::load(d, map, key, dims[0], dims[1]);
        }
        let (_, data) = map
            .bytes(key)
            .ok_or_else(|| MetalError::Model(format!("missing {key}")))?;
        let input = d.upload(data)?;
        let buffer = d.alloc(dims[0] * 4)?;
        let bad = d.upload(&0u32.to_le_bytes())?;
        let cmd = d.begin()?;
        cmd.dispatch(
            "vis_cast",
            &[&input, &buffer, &bad],
            &[dims[0] as u32, 30, 0],
            [dims[0].div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        if unsafe { bad.read_u32(1)[0] } != 0 {
            return Err(MetalError::Model(format!("nonfinite {key}")));
        }
        Ok(Weight {
            buffer,
            ty: 0,
            k: dims[0],
            n: 1,
        })
    }
}
