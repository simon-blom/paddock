use super::*;
use crate::device::MetalError;
use paddock_models::safetensors::AlignerConfig;
use std::path::Path;
impl Qwen3Aligner {
    pub fn load(dir: &Path, context: usize, budget: Option<u64>) -> Result<Self> {
        objc2::rc::autoreleasepool(|_| Self::load_inner(dir, context, budget))
    }
    fn load_inner(dir: &Path, context: usize, budget: Option<u64>) -> Result<Self> {
        let cfg =
            AlignerConfig::read(&dir.join("config.json")).map_err(|e| error(e.to_string()))?;
        if !cfg.vanilla_graph
            || !cfg.vanilla_frontend
            || context == 0
            || context > 8192
            || cfg.n_layer != 28
            || cfg.hidden != WIDTH
            || cfg.n_heads != 16
            || cfg.n_kv_heads != 8
            || cfg.head_dim != 128
            || cfg.intermediate != FF
            || cfg.vocab != VOCAB
            || cfg.max_pos != 8192
            || cfg.eps != 1e-6
            || cfg.rope_theta != 1e6
            || cfg.a_layers != 24
            || cfg.a_dmodel != 1024
            || cfg.a_heads != 16
            || cfg.a_ffn != 4096
            || cfg.a_out_dim != WIDTH
            || cfg.a_mels != 128
            || cfg.a_ch != 480
            || cfg.a_max_pos != 13
            || cfg.audio_token_id != super::super::AUDIO
            || cfg.timestamp_token_id != TIMESTAMP
            || cfg.n_labels != LABELS
            || cfg.segment_ms != 80.
        {
            return Err(error(
                "requires exact Qwen3-ForcedAligner-0.6B BF16 graph, context 1..8192",
            ));
        }
        let mut schema = audio::Tower::st_schema();
        schema.extend([
            (
                "model.language_model.embed_tokens.weight".into(),
                vec![VOCAB, WIDTH],
            ),
            ("model.language_model.norm.weight".into(), vec![WIDTH]),
            ("score.weight".into(), vec![LABELS, WIDTH]),
        ]);
        for i in 0..28 {
            let p = format!("model.language_model.layers.{i}");
            for (n, d) in [
                ("input_layernorm", WIDTH),
                ("post_attention_layernorm", WIDTH),
                ("self_attn.q_norm", 128),
                ("self_attn.k_norm", 128),
            ] {
                schema.push((format!("{p}.{n}.weight"), vec![d]));
            }
            for (n, k, out) in [
                ("self_attn.q_proj", WIDTH, 2048),
                ("self_attn.k_proj", WIDTH, 1024),
                ("self_attn.v_proj", WIDTH, 1024),
                ("self_attn.o_proj", 2048, WIDTH),
                ("mlp.gate_proj", WIDTH, FF),
                ("mlp.up_proj", WIDTH, FF),
                ("mlp.down_proj", FF, WIDTH),
            ] {
                schema.push((format!("{p}.{n}.weight"), vec![out, k]));
            }
        }
        let weights = safetensors::Weights::open(dir, &schema)?;
        let device = MetalDevice::new(budget)?;
        // Vector widening, generated positions and transient small uploads.
        // Packed decoder scratch is flat across layers; no persistent KV.
        let required =
            weights.bytes + (2 << 20) + Scratch::bytes(context) as u64 + audio::WORKSPACE;
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "aligner weights + packed decoder + audio workspace need {required} bytes"
            )));
        }
        let embedding = weights.plane(
            &device,
            "model.language_model.embed_tokens.weight",
            WIDTH,
            VOCAB,
        )?;
        let final_norm = weights.vector(&device, "model.language_model.norm.weight", WIDTH)?;
        let score = weights.plane(&device, "score.weight", WIDTH, LABELS)?;
        let mut layers = Vec::new();
        for i in 0..28 {
            let p = format!("model.language_model.layers.{i}");
            let norm = |s, d| weights.vector(&device, &format!("{p}.{s}.weight"), d);
            let w = |s, k, n| weights.plane(&device, &format!("{p}.{s}.weight"), k, n);
            layers.push(Layer {
                norm: norm("input_layernorm", WIDTH)?,
                post: norm("post_attention_layernorm", WIDTH)?,
                qnorm: norm("self_attn.q_norm", 128)?,
                knorm: norm("self_attn.k_norm", 128)?,
                q: w("self_attn.q_proj", WIDTH, 2048)?,
                k: w("self_attn.k_proj", WIDTH, 1024)?,
                v: w("self_attn.v_proj", WIDTH, 1024)?,
                o: w("self_attn.o_proj", 2048, WIDTH)?,
                gate: w("mlp.gate_proj", WIDTH, FF)?,
                up: w("mlp.up_proj", WIDTH, FF)?,
                down: w("mlp.down_proj", FF, WIDTH)?,
            });
        }
        let tower = audio::Tower::load_st(&device, &weights)?;
        let weight_bytes = device.allocated_bytes();
        let scratch = Scratch::new(&device, context)?;
        tracing::info!(
            weight_bytes,
            workspace_bytes = Scratch::bytes(context),
            context,
            "experimental Metal BF16 forced aligner loaded; no load-time requantization"
        );
        Ok(Self {
            device,
            embedding,
            final_norm,
            score,
            layers,
            tower,
            scratch,
            context,
            weight_bytes,
        })
    }
}
impl Scratch {
    fn sizes(rows: usize) -> [usize; 14] {
        [
            rows * WIDTH * 4,
            rows * WIDTH * 4,
            rows * 2048 * 4,
            rows * 1024 * 4,
            rows * 1024 * 4,
            (rows + 32) * 2048 * 2,
            (rows + 32) * 1024 * 2,
            (rows + 32) * 1024 * 2,
            rows * 2048 * 4,
            rows * WIDTH * 4,
            rows * FF * 4,
            rows * FF * 4,
            HEAD_ROWS * WIDTH * 4,
            HEAD_ROWS * LABELS * 4,
        ]
    }
    pub(super) fn bytes(rows: usize) -> usize {
        // Includes live ids/meta/tiles/timestamp descriptors and compact bins.
        Self::sizes(rows).iter().sum::<usize>() + rows * 32
    }
    fn new(d: &MetalDevice, rows: usize) -> Result<Self> {
        let sizes = Self::sizes(rows);
        let a = |i| d.alloc(sizes[i]);
        Ok(Self {
            x: a(0)?,
            norm: a(1)?,
            q: a(2)?,
            k: a(3)?,
            v: a(4)?,
            qh: a(5)?,
            kh: a(6)?,
            vh: a(7)?,
            attn: a(8)?,
            delta: a(9)?,
            gate: a(10)?,
            up: a(11)?,
            selected: a(12)?,
            logits: a(13)?,
        })
    }
}
