# TP model #2 selection: MiniCPM5-2B

Decision date: 2026-09-29 (overnight run). Branch: `review/tp-model2-proof`.

## Candidates evaluated

1. **openbmb/MiniCPM5-2B** (SELECTED) — Paddock already serves it as a plain
   `llama`-architecture file through the granite family (`granite/mod.rs`
   documents MiniCPM5-2B as the first plain-llama file served; the loader
   gates what "llama" means). Ships as one 4.8 GB safetensors file
   (`model-00000-of-00001.safetensors`).
2. **ibm-granite/granite-4.1-3b** — dense, full attention, clean scalars, but
   head_dim **64** (2560/40) where MiniCPM is head_dim **128**.
3. Rejected per instructions: Qwen3 0.6B (non-autoregressive embedding/rerank),
   any hybrid/DeltaNet/MoE family.

## Why MiniCPM5-2B

- **Kernel coverage**: head_dim 128 rides the pack's existing decode and
  prefill instantiations (`muse` hd128 G16 decode class, hd128 prefill f16
  paged). Granite's hd64 prefill rides a 64-instantiated arm with an fp8 gap
  (hd 64 is f16-KV-only per the granite dispatch comment) — a constraint
  model #2 does not need to be born with.
- **Geometry fits the generic partition exactly**: 16 Q heads / 2 KV heads /
  head_dim 128 → TP=2 = 8 Q heads / 1 KV head per rank, complete groups,
  kv_dim 128. TP=4 would be rejected by the KV-head divisibility gate (2 not
  divisible by 4) — the correct behavior for this checkpoint, same shape of
  proof as Qwen's TP=3 refusal.
- **fp8 KV eligibility**: group size 8 passes Qwen's `pf_attn_dtype_ok` fp8
  rule (groups 4|6|8) — both KV dtypes available on day one.
- **Small and conventional**: 42 layers, hidden 2048, ffn 6144, SwiGLU,
  RMSNorm eps 1e-6, rope_theta 5e6, vocab 130,560, untied head. All four
  granite multipliers at identity (`llama` arch → embedding/residual/logit
  scales 1.0, attention scale 1/sqrt(128)) — the model contributes policy
  defaults that are already the generic defaults, which is exactly what the
  `tp::conventional` surface describes.
- **Checkpoint format**: safetensors directory (same directory layout the
  runner already accepts for granite), no conversion or requantization task.

## Where it rides

| Component | Source |
| --- | --- |
| `TpTopology`, sharded linear loading | `tp/mod.rs`, `paddock_models` tensor slicing |
| `ConventionalGqaRank` | `tp/conventional.rs` (hd 128, optional norms — llama has none, optional sinks — llama has none, YARN RoPE) |
| `SwiGluTpRank` | `tp/ffn.rs` |
| Traversal primitives | `tp/traversal.rs` |
| Serving/runtime | `tp/serve.rs` (`ServeModel` bound to a `MiniCpmTpRank`) |
| Mirrored KV / checkpoints | `tp/cache.rs` |

Model-specific code should be: a `tp` spec (tensor names `blk.{layer}.{q,k,v,output,ffn_*}.weight`,
norms `blk.{layer}.attn_{q,k}_norm.weight` **do not exist** in llama →
`None`), rope params from metadata, identity multipliers, and the shim wiring
`ServeModel`. If anything forces duplicating scheduler/cache/FFN code, the
generic layer gets fixed instead.

## Download record

- HF repo: `openbmb/MiniCPM5-2B`
- Files: `config.json`, `generation_config.json`, `model-00000-of-00001.safetensors`,
  `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `chat_template.jinja`
- Identity: `LlamaForCausalLM` / `llama`, 42L, hidden 2048, 16q/2kv, hd 128,
  ffn 6144, vocab 130,560, rope_theta 5,000,000, rms_eps 1e-06, ctx 131,072
- Total: ~4.8 GB (bf16 safetensors; the weights file is 5,033,557,096 bytes)
- Local path: `~/models/minicpm5-2b/` → `/media/sime/KINGSTON/models/minicpm5-2b/` (KINGSTON volume, symlinked)
- Safetensors weights SHA-256: `14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c`
  (verified byte-for-byte against HF's LFS manifest for the file)

### Revision: the TP lane rides the official GGUF

The TP serving path hashes and shards a single-file **GGUF** (`TpInit`
identity + `load_quantw_shard`), and MiniCPM5-2B on CUDA is served from GGUF
through the granite/llama graph. `openbmb` ships official GGUF conversions,
so the safetensors directory above is reference material only and the TP
checkpoint is:

- HF repo: `openbmb/MiniCPM5-2B-GGUF`, file `MiniCPM5-2B-Q8_0.gguf`
- Local: `/media/sime/KINGSTON/models/minicpm5-2b/MiniCPM5-2B-Q8_0.gguf`
  (2,679,710,688 bytes, ~2.5 GB)
- SHA-256: `c5415f8989bf88a8288f1b55a3cc371af53c07b0faa220a63bd7a990cfaba078`
  (verified against HF's LFS manifest)
- Q8_0 is the proven TP weight class (the pinned Qwen3.8 TP checkpoint packs
  the same numeric format); metadata confirms `general.architecture = llama`,
  42 blocks, 16q/2kv hd 128, ffn 6144, ctx 131,072, rope base 5e6, eps 1e-6,
  untied `output.weight` Q8_0 [2048, 130560]
- Tensor map (381 tensors): `model.embed_tokens.weight`, `lm_head.weight`
  (untied), per-layer `model.layers.{i}.{self_attn.{q,k,v,o}_proj,
  mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}.weight`,
  `model.norm.weight`. **No** `attn_q/k_norm` (llama has no per-head norms) and
  **no** attention biases — the exact shape `tp::conventional`'s optional
  hooks default to.
