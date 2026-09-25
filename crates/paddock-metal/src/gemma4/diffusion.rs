//! Block diffusion shares the family's paged causal prefix. Canvas KV lives
//! only in append slack, never in a published prefix; causal commit overwrites
//! it. Sampler probabilities and self-conditioning stay on the GPU.
use super::*;
use paddock_engine::generator::{CanvasReadOut, CanvasStatus, CanvasTickReq};
use paddock_models::mapped::MappedGguf;

pub(super) const WIDTH: usize = 256;
const VOCAB: usize = 262144;
const HIDDEN: usize = 2816;
const FF: usize = 2112;
const PARTS: usize = 8;

struct Canvas {
    ids: Vec<u32>,
    step: u32,
    seeded: bool,
    probs: Buffer,
    picks: Buffer,
    draws: Buffer,
    entropy: Buffer,
    next: Buffer,
    previous: Buffer,
    status: Buffer,
}
pub(super) struct Pass {
    handle: usize,
    pub slot: usize,
    pub base: usize,
    pub offset: usize,
    pub width: usize,
}
pub(super) struct Lane {
    pre: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    pub logits: Buffer,
    parts: Buffer,
    soft: Buffer,
    norm: Buffer,
    gate_out: Buffer,
    up_out: Buffer,
    delta: Buffer,
    canvases: Vec<Option<Canvas>>,
    pub pass: Vec<Pass>,
}
pub(super) fn project(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    out: &Buffer,
    rows: usize,
    mlx: bool,
) {
    if mlx
        && rows
            < if w.k <= 2048 && w.n <= 2048 {
                33
            } else if w.k <= 4096 && w.n <= 4096 {
                25
            } else {
                13
            }
    {
        cmd.dispatch(
            "dg_affine_vector",
            &[&w.buffer, x, out],
            &[w.k as u32, w.n as u32, rows as u32, w.ty],
            [w.n.div_ceil(4), rows, 1],
            128,
        );
        return;
    }
    let kernel = match w.ty {
        0 => "dg_project_f32",
        6 => "dg_project_q5",
        8 => "dg_project_q8",
        12 => "dg_project_q4",
        14 => "dg_project_q6",
        0x100 => "dg_project_a4",
        0x108 => "dg_project_a8",
        _ => "dg_project",
    };
    cmd.dispatch(
        kernel,
        &[&w.buffer, x, out],
        &[w.k as u32, w.n as u32, rows as u32, w.ty, u32::from(mlx)],
        [w.n.div_ceil(32), rows.div_ceil(16), 1],
        128,
    );
}

impl Lane {
    pub(super) fn weight_bytes(&self) -> u64 {
        [&self.pre, &self.gate, &self.up, &self.down]
            .iter()
            .map(|w| w.buffer.len() as u64)
            .sum()
    }
    pub(super) fn workspace_bytes(slots: usize) -> u64 {
        ((CHUNK + slots * WIDTH) * VOCAB * 4
            + WIDTH * (HIDDEN * (PARTS + 3) + FF * 2) * 4
            + slots * WIDTH * 24
            + 4096) as u64
    }
    pub(super) fn new(
        d: &MetalDevice,
        slots: usize,
        pre: Weight,
        gate: Weight,
        up: Weight,
        down: Weight,
    ) -> Result<Self> {
        if slots == 0 || slots > 8 {
            return Err(MetalError::Model(
                "DiffusionGemma supports 1..8 concurrent slots".into(),
            ));
        }
        Ok(Self {
            pre,
            gate,
            up,
            down,
            logits: d.alloc(CHUNK * VOCAB * 4)?,
            parts: d.alloc(PARTS * WIDTH * HIDDEN * 4)?,
            soft: d.alloc(WIDTH * HIDDEN * 4)?,
            norm: d.alloc(WIDTH * HIDDEN * 4)?,
            gate_out: d.alloc(WIDTH * FF * 4)?,
            up_out: d.alloc(WIDTH * FF * 4)?,
            delta: d.alloc(WIDTH * HIDDEN * 4)?,
            canvases: (0..slots).map(|_| None).collect(),
            pass: Vec::new(),
        })
    }
    pub(super) fn load_gguf(d: &MetalDevice, map: &MappedGguf, slots: usize) -> Result<Self> {
        if map
            .gguf()
            .metadata
            .get("diffusion.canvas_length")
            .and_then(|v| v.as_u64())
            != Some(WIDTH as u64)
            || map
                .gguf()
                .arch_field("attention.causal")
                .is_none_or(|v| !matches!(v, paddock_models::gguf::Value::Bool(false)))
        {
            return Err(MetalError::Model(
                "DiffusionGemma requires a 256-row bidirectional canvas".into(),
            ));
        }
        Self::new(
            d,
            slots,
            Weight::load(d, map, "self_cond_pre_norm.weight", &[HIDDEN])?,
            Weight::load(d, map, "self_cond_gate.weight", &[HIDDEN, FF])?,
            Weight::load(d, map, "self_cond_up.weight", &[HIDDEN, FF])?,
            Weight::load(d, map, "self_cond_down.weight", &[FF, HIDDEN])?,
        )
    }
    pub(super) fn reset(&mut self) {
        self.pass.clear();
        for c in &mut self.canvases {
            *c = None;
        }
    }
    pub(super) fn preamble(
        &self,
        cmd: &Commands<'_>,
        embedding: &Weight,
        x: &Buffer,
        eps: f32,
        mlx: bool,
    ) {
        for pass in &self.pass {
            let c = self.canvases[pass.handle]
                .as_ref()
                .expect("validated canvas");
            let rows = pass.width;
            if c.step > 0 {
                cmd.dispatch(
                    match embedding.ty {
                        14 => "dg_soft_embed_q6",
                        8 => "dg_soft_embed_q8",
                        0x108 => "dg_soft_embed_a8",
                        _ => "dg_soft_embed",
                    },
                    &[&embedding.buffer, &c.probs, &self.parts],
                    &[
                        HIDDEN as u32,
                        VOCAB as u32,
                        rows as u32,
                        embedding.ty,
                        PARTS as u32,
                    ],
                    [HIDDEN.div_ceil(32), rows.div_ceil(16), PARTS],
                    128,
                );
                cmd.dispatch(
                    "dg_soft_fold",
                    &[&self.parts, &self.soft],
                    &[HIDDEN as u32, rows as u32, PARTS as u32, u32::from(mlx)],
                    [(rows * HIDDEN).div_ceil(256), 1, 1],
                    256,
                );
                cmd.dispatch(
                    if mlx { "gmlx_norm" } else { "rms" },
                    &[&self.soft, &self.pre.buffer, &self.norm],
                    &[HIDDEN as u32, 0, eps.to_bits()],
                    [rows, 1, 1],
                    if mlx { HIDDEN.div_ceil(128) * 32 } else { 256 },
                );
                project(cmd, &self.gate, &self.norm, &self.gate_out, rows, mlx);
                project(cmd, &self.up, &self.norm, &self.up_out, rows, mlx);
                cmd.dispatch(
                    if mlx { "gmlx_geglu" } else { "gemma_geglu" },
                    &[&self.gate_out, &self.up_out],
                    &[(rows * FF) as u32],
                    [(rows * FF).div_ceil(256), 1, 1],
                    256,
                );
                project(cmd, &self.down, &self.gate_out, &self.delta, rows, mlx);
            } else {
                cmd.dispatch(
                    "dg_zero",
                    &[&self.delta],
                    &[(rows * HIDDEN) as u32],
                    [(rows * HIDDEN).div_ceil(256), 1, 1],
                    256,
                );
            }
            cmd.dispatch(
                "dg_add_norm",
                &[x, &self.delta],
                &[
                    HIDDEN as u32,
                    eps.to_bits(),
                    pass.offset as u32,
                    u32::from(mlx),
                ],
                [rows, 1, 1],
                if mlx { HIDDEN.div_ceil(128) * 32 } else { 256 },
            );
        }
    }
}

// Same counter layout as the CUDA canvas, independent of batch slot/arrival.
fn random(seed: u64, offset: u32, index: u32) -> u32 {
    let mut c = [offset, 0, index, 0];
    let mut k = [seed as u32, (seed >> 32) as u32];
    for _ in 0..10 {
        let a = c[0] as u64 * 0xD2511F53;
        let b = c[2] as u64 * 0xCD9E8D57;
        c = [
            (b >> 32) as u32 ^ c[1] ^ k[0],
            b as u32,
            (a >> 32) as u32 ^ c[3] ^ k[1],
            a as u32,
        ];
        k[0] = k[0].wrapping_add(0x9E3779B9);
        k[1] = k[1].wrapping_add(0xBB67AE85);
    }
    c[0]
}
pub(super) fn noise(w: usize, seed: u64, offset: u32) -> Vec<u32> {
    (0..w)
        .map(|i| ((random(seed, offset, i as u32) as u64 * VOCAB as u64) >> 32) as u32)
        .collect()
}
impl Gemma4 {
    fn lane(&self) -> Result<&Lane> {
        self.diffusion
            .as_ref()
            .ok_or_else(|| MetalError::Model("not DiffusionGemma".into()))
    }
    pub(super) fn open_canvas(&mut self, width: usize) -> Result<usize> {
        if width == 0 || width > WIDTH {
            return Err(MetalError::Model("canvas width outside 1..256".into()));
        }
        let h = self
            .lane()?
            .canvases
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| MetalError::Memory("all canvas slots are leased".into()))?;
        let d = &self.device;
        let a = || d.alloc(width * 4);
        let c = Canvas {
            ids: vec![0; width],
            step: 0,
            seeded: false,
            probs: d.alloc(width * self.vocab * 4)?,
            picks: a()?,
            draws: a()?,
            entropy: a()?,
            next: a()?,
            previous: a()?,
            status: d.alloc(16)?,
        };
        self.diffusion.as_mut().expect("validated lane").canvases[h] = Some(c);
        Ok(h)
    }
    pub(super) fn set_canvas(&mut self, h: usize, ids: &[u32]) -> Result<()> {
        let vocab = self.vocab;
        let c = self
            .diffusion
            .as_mut()
            .and_then(|d| d.canvases.get_mut(h))
            .and_then(Option::as_mut)
            .ok_or_else(|| MetalError::Model("invalid canvas handle".into()))?;
        if ids.len() != c.ids.len() || ids.iter().any(|&id| id as usize >= vocab) {
            return Err(MetalError::Model("invalid canvas tokens".into()));
        }
        c.ids.copy_from_slice(ids);
        c.seeded = true;
        Ok(())
    }
    pub(super) fn close_canvas(&mut self, h: usize) {
        if let Some(c) = self.diffusion.as_mut().and_then(|d| d.canvases.get_mut(h)) {
            *c = None;
        }
    }
    pub(super) fn pin_canvas(&mut self, h: usize, positions: &[u32], ids: &[u32]) -> Result<()> {
        let c = self
            .diffusion
            .as_mut()
            .and_then(|d| d.canvases.get_mut(h))
            .and_then(Option::as_mut)
            .ok_or_else(|| MetalError::Model("invalid canvas handle".into()))?;
        // Validate the entire edit before mutating; keep the probability plane
        // and step intact for the next self-conditioning pass.
        if !c.seeded
            || positions.len() != ids.len()
            || positions.iter().any(|&p| p as usize >= c.ids.len())
            || ids.iter().any(|&id| id as usize >= self.vocab)
        {
            return Err(MetalError::Model("invalid canvas pins".into()));
        }
        for (&p, &id) in positions.iter().zip(ids) {
            c.ids[p as usize] = id;
        }
        Ok(())
    }
    pub(super) fn tick_canvases(&mut self, ticks: &[CanvasTickReq]) -> Result<Vec<CanvasStatus>> {
        self.require_committed()?;
        let lane = self.lane()?;
        let mut rows = Vec::new();
        let mut passes = Vec::new();
        let mut seen = vec![false; self.slots.len()];
        let mut handles = Vec::new();
        for t in ticks {
            let c = lane
                .canvases
                .get(t.handle)
                .and_then(Option::as_ref)
                .ok_or_else(|| MetalError::Model("invalid canvas handle".into()))?;
            if !c.seeded
                || t.slot >= seen.len()
                || seen[t.slot]
                || handles.contains(&t.handle)
                || self.slots[t.slot].history.len() != t.base
                || t.base
                    .checked_add(c.ids.len())
                    .is_none_or(|n| n > self.context)
                || t.temperature.is_some_and(|v| !v.is_finite() || v < 0.)
                || c.step >= 48
                || self.pending.iter().any(|p| p.slot == t.slot)
                || self.image_slot_pending(t.slot)
            {
                return Err(MetalError::Model(
                    "invalid canvas slot, position, temperature or lifecycle".into(),
                ));
            }
            seen[t.slot] = true;
            handles.push(t.handle);
            passes.push(Pass {
                handle: t.handle,
                slot: t.slot,
                base: t.base,
                offset: rows.len(),
                width: c.ids.len(),
            });
            rows.extend(
                c.ids
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| (t.slot, id, (t.base + i) as u32)),
            );
        }
        if rows.is_empty() || rows.len() > CHUNK {
            return Err(MetalError::Model(
                "canvas tick exceeds bounded row capacity".into(),
            ));
        }
        self.diffusion.as_mut().expect("lane").pass = passes;
        let forward = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>());
        let passes = std::mem::take(&mut self.diffusion.as_mut().expect("lane").pass);
        forward?;
        let lane = self.diffusion.as_ref().expect("lane");
        let cmd = self.device.begin()?;
        for (t, p) in ticks.iter().zip(&passes) {
            let c = lane.canvases[t.handle].as_ref().expect("validated handle");
            copy_words(
                &cmd,
                &lane.logits,
                &c.probs,
                p.offset * self.vocab,
                0,
                p.width * self.vocab,
            );
            let schedule = 0.4 + 0.4 * (48 - c.step) as f32 / 48.;
            let mut inv = match t.temperature {
                Some(0.) => -1. / schedule,
                Some(v) => 1. / v,
                None => 1. / schedule,
            };
            // A structured read needs distributions, never categorical draws.
            // Preserve its temperature while skipping a vocab-wide Philox pass.
            if !t.accept {
                inv = -inv.abs();
            }
            cmd.dispatch(
                "dg_sample",
                &[&c.probs, &c.picks, &c.draws, &c.entropy],
                &[
                    self.vocab as u32,
                    inv.to_bits(),
                    t.seed as u32,
                    (t.seed >> 32) as u32,
                    c.step * 2,
                ],
                [p.width, 1, 1],
                256,
            );
            if t.accept {
                cmd.dispatch(
                    "dg_accept",
                    &[
                        &c.entropy,
                        &c.picks,
                        &c.draws,
                        &c.next,
                        &c.previous,
                        &c.status,
                    ],
                    &[
                        p.width as u32,
                        self.vocab as u32,
                        t.seed as u32,
                        (t.seed >> 32) as u32,
                        c.step * 2 + 1,
                        c.step,
                    ],
                    [1, 1, 1],
                    256,
                );
            }
        }
        self.last_gpu_seconds += cmd.finish()?;
        let lane = self.diffusion.as_mut().expect("lane");
        let mut results = Vec::new();
        for t in ticks {
            let c = lane.canvases[t.handle].as_mut().expect("validated handle");
            c.step += 1;
            let entropy = unsafe { c.entropy.read_f32(0, c.ids.len()) };
            if entropy.iter().any(|v| !v.is_finite()) {
                return Err(MetalError::Model("nonfinite canvas distribution".into()));
            }
            if t.accept {
                c.ids = unsafe { c.next.read_u32(c.ids.len()) };
                let w = unsafe { c.status.read_u32(4) };
                results.push(CanvasStatus::from_words([w[0], w[1], w[2], w[3]]));
            } else {
                results.push(CanvasStatus {
                    converged: true,
                    stable: false,
                    n_accepted: 0,
                    mean_entropy: entropy.iter().sum::<f32>() / entropy.len() as f32,
                });
            }
        }
        Ok(results)
    }
    pub(super) fn canvas_output(&self, h: usize, labels: &[u32]) -> Result<CanvasReadOut> {
        let c = self
            .lane()?
            .canvases
            .get(h)
            .and_then(Option::as_ref)
            .ok_or_else(|| MetalError::Model("invalid canvas handle".into()))?;
        if c.step == 0 || labels.len() > 4096 || labels.iter().any(|&id| id as usize >= self.vocab)
        {
            return Err(MetalError::Model(
                "canvas has no result or label set is invalid".into(),
            ));
        }
        let probs = if labels.is_empty() {
            Vec::new()
        } else {
            let ids = self.device.alloc(labels.len() * 4)?;
            unsafe {
                ids.write_u32(labels);
            }
            let out = self.device.alloc(c.ids.len() * labels.len() * 4)?;
            let cmd = self.device.begin()?;
            cmd.dispatch(
                "dg_labels",
                &[&c.probs, &ids, &out],
                &[c.ids.len() as u32, self.vocab as u32, labels.len() as u32],
                [(c.ids.len() * labels.len()).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
            unsafe { out.read_f32(0, c.ids.len() * labels.len()) }
        };
        Ok(CanvasReadOut {
            probs,
            entropy: unsafe { c.entropy.read_f32(0, c.ids.len()) },
            argmax: unsafe { c.picks.read_u32(c.ids.len()) },
        })
    }
    pub(super) fn commit_canvas(&mut self, slot: usize, base: usize, ids: &[u32]) -> Result<()> {
        if self.diffusion.is_none()
            || ids.is_empty()
            || ids.len() > WIDTH
            || self.slots.get(slot).is_none_or(|s| s.history.len() != base)
        {
            return Err(MetalError::Model("invalid canvas commit".into()));
        }
        let rows = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (slot, id, (base + i) as u32))
            .collect::<Vec<_>>();
        self.execute(&rows, &[])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_affine8_embedding_retains_scale() {
        let d = MetalDevice::new(None).unwrap();
        let mut bytes = vec![1u8; HIDDEN * 2];
        bytes.extend((0..HIDDEN * 2 / 64).flat_map(|_| 0x3f80u16.to_le_bytes()));
        bytes.extend(vec![0u8; HIDDEN * 2 / 64 * 2]);
        let w = d.upload(&bytes).unwrap();
        let ids = d.upload(&1u32.to_le_bytes()).unwrap();
        let out = d.alloc(HIDDEN * 4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "dg_embed",
            &[&w, &ids, &out],
            &[HIDDEN as u32, 1, 2, 0x108, 1],
            [HIDDEN.div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        let values = unsafe { out.read_f32(0, HIDDEN) };
        assert_eq!(values[0], 53.0);
        assert!(values.iter().all(|&v| v == 53.0));
    }
    #[test]
    fn philox_has_stable_slot_independent_counters() {
        assert_eq!(random(0, 0, 0), 0x6627e8d5);
        let a = noise(256, 42, 0);
        assert_eq!(a, noise(256, 42, 0));
        assert_ne!(a, noise(256, 42, 1));
        assert!(a.iter().all(|&v| v < 262144));
    }
    #[test]
    fn q5_tail_projection_keeps_high_bits_and_signed_codes() {
        let d = MetalDevice::new(None).unwrap();
        let mut bytes = Vec::new();
        for _ in 0..4 {
            bytes.extend(0x3c00u16.to_le_bytes());
            bytes.extend(0xffff0000u32.to_le_bytes());
            bytes.extend((0..16u8).map(|i| i | (i << 4)));
        }
        let w = d.upload(&bytes).unwrap();
        let x = d
            .upload(
                &(0..32)
                    .flat_map(|i| ((i as f32 - 8.) / 16.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let out = d.alloc(16).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "dg_project",
            &[&w, &x, &out],
            &[32, 4, 1, 6, 0],
            [1, 1, 1],
            128,
        );
        cmd.finish().unwrap();
        let expected = (0..32)
            .map(|i| ((i as f32 - 8.) / 16.) * (i as f32 - 16.))
            .sum::<f32>();
        assert!(
            unsafe { out.read_f32(0, 4) }
                .iter()
                .all(|v| (v - expected).abs() < 1e-5)
        );
    }
    #[test]
    fn mlx_attention_rounds_scores_and_probabilities_without_a_score_plane() {
        let d = MetalDevice::new(None).unwrap();
        let bf = |x: f32| {
            let b = x.to_bits();
            ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
        };
        let back = |b: u16| f32::from_bits(u32::from(b) << 16);
        for hd in [256, 512] {
            let kh = if hd == 256 { 8 } else { 2 };
            let mut queries = vec![0f32; 16 * hd];
            for h in 0..16 {
                queries[h * hd] = 1.;
            }
            let mut keys = vec![0u16; 2 * kh * hd];
            let mut values = vec![0u16; 2 * kh * hd];
            for t in 0..2 {
                for h in 0..kh {
                    keys[(t * kh + h) * hd] = bf(if t == 0 { 0.7 } else { 1.3 });
                    for j in 0..hd {
                        values[(t * kh + h) * hd + j] = bf(if t == 0 { 2. } else { -1. });
                    }
                }
            }
            let q = d
                .upload(
                    &queries
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let k = d
                .upload(
                    &keys
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let v = d
                .upload(
                    &values
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let words = |w: &[u32]| {
                d.upload(&w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                    .unwrap()
            };
            let meta = words(&[0, 0]);
            let pages = words(&[0]);
            let tiles = words(&[0, 1]);
            let limits = words(&[1]);
            let out = d.alloc(16 * hd * 4).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if hd == 256 {
                    "dg_attention256"
                } else {
                    "dg_attention512"
                },
                &[&q, &k, &v, &meta, &pages, &out, &tiles, &limits],
                &[
                    16,
                    kh as u32,
                    1,
                    if hd == 256 { 1024 } else { 0 },
                    1536,
                    1,
                    1,
                    0,
                ],
                [16, 1, 1],
                128,
            );
            cmd.finish().unwrap();
            let e = (back(bf(0.7)) - back(bf(1.3))).exp();
            let expected = 2. * back(bf(e / (e + 1.))) - back(bf(1. / (e + 1.)));
            assert!(
                unsafe { out.read_f32(0, 16 * hd) }
                    .iter()
                    .all(|x| (x - expected).abs() < 1e-5),
                "head dimension {hd}"
            );
        }
    }
    #[test]
    fn decoder_tiles_share_the_retained_prefix_window_in_both_formats() {
        let d = MetalDevice::new(None).unwrap();
        let hd = 256;
        let base = 1100;
        let end = base + 2;
        let upload_words = |w: &[u32]| {
            d.upload(&w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        // Uniform attention makes the inclusion/exclusion boundary observable
        // without conflating it with projection or softmax reduction drift.
        let q = d.upload(&vec![0; 2 * hd * 4]).unwrap();
        let k = d.upload(&vec![0; end * hd * 2]).unwrap();
        let meta = upload_words(&[0, base as u32, 0, (base + 1) as u32]);
        let pages = upload_words(&[0]);
        let tiles = upload_words(&[0, 1, 1, 1]);
        let limits = upload_words(&[(end - 1) as u32; 2]);
        let out = d.alloc(2 * hd * 4).unwrap();
        for mlx in [false, true] {
            let mut values = vec![0u16; end * hd];
            for (t, x) in [(base - 1024, 1024f32), (base - 1023, 1.), (base + 1, 2.)] {
                let bits = if mlx {
                    (x.to_bits() >> 16) as u16
                } else {
                    half::f16::from_f32(x).to_bits()
                };
                values[t * hd..(t + 1) * hd].fill(bits);
            }
            let v = d
                .upload(
                    &values
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if mlx {
                    "dg_attention256"
                } else {
                    "dg_gguf_attention256"
                },
                &[&q, &k, &v, &meta, &pages, &out, &tiles, &limits],
                &[1, 1, 1, 1024, 1536, 1, u32::from(mlx), base as u32],
                [1, 2, 1],
                128,
            );
            cmd.finish().unwrap();
            let mut p = 1f32 / 1025.;
            if mlx {
                let b = p.to_bits();
                p = f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000);
            }
            let values = unsafe { out.read_f32(0, 2 * hd) };
            assert!(
                values.iter().all(|x| (x - 3. * p).abs() < 1e-6),
                "MLX={mlx}: {} {}",
                values[0],
                values[hd]
            );
        }
    }
    #[test]
    #[ignore = "requires downloaded DiffusionGemma; set PADDOCK_DG_MODEL"]
    fn live_read_and_generation() {
        let path = std::env::var("PADDOCK_DG_MODEL").unwrap();
        let path = Path::new(&path);
        let tok = if path.is_dir() {
            paddock_tokenizer::GgufTokenizer::from_hf_dir(path).unwrap()
        } else {
            let map = MappedGguf::open(path).unwrap();
            paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).unwrap()
        };
        let prompt = tok
            .encode("<bos><|turn>user\nIs water wet?<turn|>\n<|turn>model\n")
            .unwrap();
        let mut canvas = tok
            .encode("<|channel>thought\n<channel|>answer: yes<turn|>")
            .unwrap();
        canvas.resize(16, 0);
        let started = std::time::Instant::now();
        let mut m = Gemma4::load(path, 4096, 1, None).unwrap();
        eprintln!(
            "load_s={:.3} device_bytes={:?}",
            started.elapsed().as_secs_f64(),
            m.device_mem_used()
        );
        let started = std::time::Instant::now();
        m.forward_prefill(0, &prompt).unwrap();
        eprintln!("prefill_s={:.3}", started.elapsed().as_secs_f64());
        // All invalid lifecycle operations fail before submitting GPU work;
        // releasing a failed/abandoned read must make its lease reusable.
        assert!(m.canvas_open(0).is_err());
        assert!(m.canvas_open(WIDTH + 1).is_err());
        let handle = m.canvas_open(canvas.len()).unwrap();
        assert!(m.canvas_open(canvas.len()).is_err());
        let tick = CanvasTickReq {
            handle,
            slot: 0,
            base: prompt.len(),
            temperature: Some(1.),
            seed: 0,
            accept: false,
        };
        assert!(m.canvas_tick(std::slice::from_ref(&tick)).is_err());
        assert!(m.canvas_result(handle, &[]).is_err());
        assert!(m.canvas_set(handle, &[u32::MAX; 16]).is_err());
        m.canvas_set(handle, &canvas).unwrap();
        assert!(m.canvas_tick(&[tick, tick]).is_err());
        for invalid in [
            CanvasTickReq { slot: 1, ..tick },
            CanvasTickReq {
                base: prompt.len() + 1,
                ..tick
            },
            CanvasTickReq {
                temperature: Some(f32::NAN),
                ..tick
            },
        ] {
            assert!(m.canvas_tick(&[invalid]).is_err());
        }
        assert_eq!(m.slots[0].history, prompt);
        m.canvas_close(handle);
        m.canvas_close(handle);
        assert!(m.canvas_set(handle, &canvas).is_err());
        let labels = [
            *tok.encode(" yes").unwrap().last().unwrap(),
            *tok.encode(" no").unwrap().last().unwrap(),
        ];
        let started = std::time::Instant::now();
        let read = m.canvas_read(prompt.len(), &canvas, &labels).unwrap();
        eprintln!(
            "read_s={:.3} text={:?} entropy={:?}",
            started.elapsed().as_secs_f64(),
            tok.decode(&read.argmax, false).unwrap(),
            read.entropy
        );
        assert!(
            read.probs
                .iter()
                .all(|p| p.is_finite() && (0. ..=1.).contains(p))
        );
        assert_eq!(m.slots[0].history, prompt);
        assert_eq!(
            read.argmax[6], labels[0],
            "the fixed answer slot must read yes"
        );
        if path.is_dir() {
            assert_eq!(
                read.argmax,
                [
                    1, 506, 1, 818, 3890, 236787, 11262, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ],
                "MLX 0.32.2 / mlx-vlm 0.7.1 fixed-canvas oracle"
            );
        }
        let again = m.canvas_read(prompt.len(), &canvas, &labels).unwrap();
        assert_eq!(read.argmax, again.argmax);
        assert_eq!(read.probs, again.probs);
        if std::env::var_os("PADDOCK_DG_TRACE").is_some() {
            return;
        }
        let started = std::time::Instant::now();
        let (ids, steps) = m.canvas_block(prompt.len(), 256, Some(0.), 42).unwrap();
        let text = tok.decode(&ids, false).unwrap();
        assert!(
            text.starts_with("<|channel>thought\n<channel|>"),
            "invalid answer scaffold: {text}"
        );
        assert!(
            text.to_lowercase().contains("water"),
            "generation did not address the prompt: {text}"
        );
        eprintln!(
            "generation_s={:.3} steps={steps} text={:?}",
            started.elapsed().as_secs_f64(),
            text
        );
        assert_eq!(m.slots[0].history, prompt);
        m.canvas_commit(prompt.len(), &ids).unwrap();
        assert_eq!(&m.slots[0].history[prompt.len()..], ids);
    }
}
