//! Packed native Laya decision inference. The shared decision service owns
//! admission/routing; every checkpoint reuses one bounded Metal workspace.
//! No Python runtime, generation loop or CPU encoder fallback.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_engine::decision::{DecisionBackend, DecisionInfo, LayaOut, LayaSeq};
use paddock_models::laya::{Checkpoint, LayaBundle, LayaConfig};
use std::path::Path;

mod forward;
mod load;
#[cfg(test)]
mod tests;

pub const PASS_TOKENS: usize = 4096;
pub const PASS_SEQUENCES: usize = 128;
const QUERY_TILE: usize = 32;
fn error(message: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Laya: {}", message.into()))
}

struct Linear {
    w: Buffer,
    k: usize,
    n: usize,
}
impl Linear {
    fn run(
        &self,
        c: &Commands<'_>,
        input: &Buffer,
        out: &Buffer,
        bias: &Buffer,
        rows: usize,
        op: u32,
    ) {
        c.dispatch(
            if op == 4 { "laya_geglu" } else { "laya_mm" },
            &[&self.w, input, out, bias],
            &[self.k as u32, self.n as u32, rows as u32, op],
            [self.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
}
struct Norm {
    w: Buffer,
    b: Buffer,
}
struct EncoderLayer {
    attn_norm: Option<Buffer>,
    qkv: Linear,
    out: Linear,
    mlp_norm: Buffer,
    up: Linear,
    down: Linear,
    global: bool,
}
struct HeadLayer {
    n1: Norm,
    qkv: Linear,
    qb: Buffer,
    out: Linear,
    ob: Buffer,
    n2: Norm,
    up: Linear,
    ub: Buffer,
    down: Linear,
    db: Buffer,
}
struct Model {
    cfg: LayaConfig,
    emb: Buffer,
    emb_norm: Buffer,
    layers: Vec<EncoderLayer>,
    final_norm: Buffer,
    rope_g: Buffer,
    rope_l: Buffer,
    type_emb: Buffer,
    head: Vec<HeadLayer>,
    score_norm: Norm,
    score: Linear,
    score_bias: Buffer,
    score_w: Buffer,
    score_b: f32,
    act_w0: Buffer,
    act_b0: Buffer,
    act_w2: Buffer,
    act_b2: Buffer,
    zeros: Buffer,
}

struct Workspace {
    rows: usize,
    sequences: usize,
    gather: usize,
    tile_count: usize,
    ids: Buffer,
    meta: Buffer,
    tiles: Buffer,
    indices: Buffer,
    offsets: Buffer,
    x: Buffer,
    xg: Buffer,
    n: Buffer,
    wide: Buffer,
    att: Buffer,
    proj: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    logits: Buffer,
    act: Buffer,
}
impl Workspace {
    fn sizes(d: usize, wide: usize, rows: usize, seq: usize) -> [usize; 16] {
        let gather = rows / 2 + seq;
        [
            rows * 4,
            rows * 8,
            (rows.div_ceil(QUERY_TILE) + seq) * 16,
            gather * 4,
            (seq + 1) * 4,
            rows * d * 4,
            gather * d * 4,
            rows * d * 2,
            rows * wide * 2,
            rows * d * 2,
            rows * d * 2,
            rows * d * 2,
            rows * d * 2,
            rows * d * 2,
            gather * 4,
            seq * 8 * 4,
        ]
    }
    fn new(device: &MetalDevice, d: usize, wide: usize, rows: usize, seq: usize) -> Result<Self> {
        let [
            ids,
            meta,
            tiles,
            indices,
            offsets,
            x,
            xg,
            n,
            wide,
            att,
            proj,
            q,
            k,
            v,
            logits,
            act,
        ] = Self::sizes(d, wide, rows, seq).map(|n| device.alloc(n));
        Ok(Self {
            rows,
            sequences: seq,
            gather: rows / 2 + seq,
            tile_count: rows.div_ceil(QUERY_TILE) + seq,
            ids: ids?,
            meta: meta?,
            tiles: tiles?,
            indices: indices?,
            offsets: offsets?,
            x: x?,
            xg: xg?,
            n: n?,
            wide: wide?,
            att: att?,
            proj: proj?,
            q: q?,
            k: k?,
            v: v?,
            logits: logits?,
            act: act?,
        })
    }
}

/// All three official checkpoints, sharing the same execution queue and scratch.
pub struct Laya {
    device: MetalDevice,
    models: Vec<(Checkpoint, Model)>,
    ws: Workspace,
    weight_bytes: u64,
    workspace_bytes: u64,
}
impl Laya {
    pub fn load(dir: &Path, budget: Option<u64>) -> Result<Self> {
        let bundle = LayaBundle::read(dir).map_err(|e| error(e.to_string()))?;
        let sources = bundle
            .checkpoints
            .iter()
            .map(|(_, cfg)| load::Source::open(cfg))
            .collect::<Result<Vec<_>>>()?;
        let d = bundle
            .checkpoints
            .iter()
            .map(|(_, c)| c.encoder.hidden)
            .max()
            .ok_or_else(|| error("empty bundle"))?;
        let wide = bundle
            .checkpoints
            .iter()
            .map(|(_, c)| (4 * c.encoder.hidden).max(c.encoder.intermediate))
            .max()
            .ok_or_else(|| error("empty bundle"))?;
        let workspace_bytes = Workspace::sizes(d, wide, PASS_TOKENS, PASS_SEQUENCES)
            .iter()
            .sum::<usize>() as u64;
        let weight_bytes = sources.iter().map(|s| s.resident_bytes).sum::<u64>();
        let device = MetalDevice::new_planned(budget, weight_bytes + workspace_bytes)?;
        let models = bundle
            .checkpoints
            .iter()
            .zip(&sources)
            .map(|((ck, cfg), s)| Ok((*ck, s.load(&device, cfg)?)))
            .collect::<Result<Vec<_>>>()?;
        drop(sources);
        let actual_weights = device.allocated_bytes();
        debug_assert_eq!(actual_weights, weight_bytes);
        let ws = Workspace::new(&device, d, wide, PASS_TOKENS, PASS_SEQUENCES)?;
        let mut result = Self {
            device,
            models,
            ws,
            weight_bytes,
            workspace_bytes,
        };
        // Kernel specialization and first-touch belong to load, not a user's read.
        for (ck, _) in &bundle.checkpoints {
            result.run(
                *ck,
                &[LayaSeq {
                    ids: &[1, 2, 3, 4, 5, 6],
                    markers: &[2, 4],
                    qtype: 0,
                }],
            )?;
        }
        Ok(result)
    }
    pub fn run(&mut self, checkpoint: Checkpoint, seqs: &[LayaSeq<'_>]) -> Result<LayaOut> {
        let model = &self
            .models
            .iter()
            .find(|(c, _)| *c == checkpoint)
            .ok_or_else(|| error("checkpoint not loaded"))?
            .1;
        model.forward(&self.device, &mut self.ws, seqs)
    }
}
impl DecisionBackend for Laya {
    fn info(&self) -> DecisionInfo {
        DecisionInfo {
            checkpoints: self
                .models
                .iter()
                .map(|(c, m)| (*c, m.cfg.clone()))
                .collect(),
            rows_cap: self.ws.rows,
            seq_cap: self.ws.sequences,
            weight_bytes: self.weight_bytes,
            workspace_bytes: self.workspace_bytes,
        }
    }
    fn forward(
        &mut self,
        checkpoint: Checkpoint,
        seqs: &[LayaSeq<'_>],
    ) -> std::result::Result<LayaOut, String> {
        self.run(checkpoint, seqs).map_err(|e| e.to_string())
    }
}
