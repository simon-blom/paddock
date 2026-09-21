//! Packed audio encoder: 100-frame padded conv chunks, independent 104-row
//! attention windows, fused QKV and producer bias/residual epilogues. A wave
//! shares projections across requests but never attention. One bounded conv
//! stage or transformer block per step leaves a decode scheduling boundary.
use super::*;
use paddock_engine::audio::{MelFeatures, audio_token_count};
mod load;
mod safetensors;
const E: usize = 1024;
const GROUP: usize = 8;
pub(super) const MAX_FRAMES: usize = 12000;
// Conservative device grant includes one packed wave, all final span buffers,
// and maximum conv gather. Host preprocessing buffers are separately bounded
// by request/slot limits; this is not a host+device unified-memory estimate.
pub(super) const WORKSPACE: u64 = 512 << 20;
struct Norm {
    w: Weight,
    b: Weight,
}
struct Linear {
    w: Weight,
    b: Weight,
}
struct Block {
    ln1: Norm,
    ln2: Norm,
    qkv: Linear,
    out: Linear,
    up: Linear,
    down: Linear,
}
pub(super) struct Tower {
    bf16_activations: bool,
    conv: Vec<Linear>,
    conv_out: Weight,
    pos: Weight,
    blocks: Vec<Block>,
    post: Norm,
    up: Linear,
    down: Linear,
}
pub(super) struct Job {
    sizes: Vec<usize>,
    rows: usize,
    chunks: usize,
    chunk: usize,
    stage: usize,
    layer: usize,
    mel: Buffer,
    map: Buffer,
    tiles: Buffer,
    gather: Buffer,
    conv: Buffer,
    projected: Buffer,
    x: Buffer,
    norm: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    ff: Buffer,
    output: Buffer,
}
pub(super) fn validate(m: &MelFeatures) -> Result<()> {
    if m.n_frames == 0
        || m.n_frames > MAX_FRAMES
        || m.data.len() != m.n_frames.div_ceil(100) * 100 * 128
        || m.n_samples > MAX_FRAMES * 160
        || m.data.iter().any(|v| !v.is_finite())
    {
        return Err(error(
            "invalid mel plane or audio exceeds 120-second implementation ceiling",
        ));
    }
    Ok(())
}
impl Tower {
    pub(super) fn start(&self, d: &MetalDevice, inputs: &[MelFeatures]) -> Result<Job> {
        self.start_refs(d, &inputs.iter().collect::<Vec<_>>())
    }
    pub(super) fn start_refs(&self, d: &MetalDevice, inputs: &[&MelFeatures]) -> Result<Job> {
        if inputs.is_empty() || inputs.len() > 16 {
            return Err(error("audio wave requires 1..16 clips"));
        }
        let mut mel = Vec::new();
        let mut mapping = Vec::new();
        let mut tiles = Vec::new();
        let mut sizes = Vec::new();
        let mut rows = 0;
        for m in inputs {
            validate(m)?;
            let n = audio_token_count(m.n_frames);
            let chunks = m.n_frames.div_ceil(100);
            if mel.len() / 128 + chunks * 100 > MAX_FRAMES {
                return Err(error("audio wave exceeds packed frame budget"));
            }
            mel.extend_from_slice(&m.data);
            mapping
                .extend((0..chunks * 13).map(|i| if i < n { (rows + i) as u32 } else { u32::MAX }));
            for w in (0..n).step_by(104) {
                let count = (n - w).min(104);
                for q in (0..count).step_by(32) {
                    tiles.extend([
                        (rows + w + q) as u32,
                        (count - q).min(32) as u32,
                        (rows + w) as u32,
                        count as u32,
                    ]);
                }
            }
            sizes.push(n);
            rows += n;
        }
        if d.allocated_bytes() + WORKSPACE > d.budget_bytes() {
            return Err(MetalError::Memory(
                "audio wave workspace exceeds grant".into(),
            ));
        }
        let f = |x: &[f32]| -> Result<Buffer> {
            let b = d.alloc(x.len() * 4)?;
            unsafe {
                b.write_u32(&x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
            Ok(b)
        };
        let u = |x: &[u32]| -> Result<Buffer> {
            let b = d.alloc(x.len() * 4)?;
            unsafe {
                b.write_u32(x);
            }
            Ok(b)
        };
        // The second conv has the largest gather: 8 * 25 * 32 * (480*9).
        // Resident for this wave across every chunk group; no stage alloc churn.
        Ok(Job {
            sizes,
            rows,
            chunks: mel.len() / 128 / 100,
            chunk: 0,
            stage: 0,
            layer: 0,
            mel: f(&mel)?,
            map: u(&mapping)?,
            tiles: u(&tiles)?,
            gather: d.alloc(GROUP * 25 * 32 * 4320 * if self.bf16_activations { 2 } else { 4 })?,
            conv: d.alloc(GROUP * 50 * 64 * 480 * 4)?,
            projected: d.alloc(GROUP * 13 * E * 4)?,
            x: d.alloc(rows * E * 4)?,
            norm: d.alloc(rows * E * 4)?,
            qkv: d.alloc(rows * E * 3 * 4)?,
            q: d.alloc((rows + 64) * E * 2)?,
            k: d.alloc((rows + 64) * E * 2)?,
            v: d.alloc((rows + 64) * E * 2)?,
            attn: d.alloc(rows * E * 4)?,
            ff: d.alloc(rows * 4096 * 4)?,
            output: d.alloc(rows * self.down.w.n * 4)?,
        })
    }
    fn norm(&self, c: &Commands<'_>, p: &Norm, x: &Buffer, y: &Buffer, rows: usize) {
        c.dispatch(
            if self.bf16_activations {
                "qalign_ln"
            } else {
                "vis_ln"
            },
            &[x, &p.w.buffer, &p.b.buffer, y],
            &[E as u32, 1e-5f32.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    fn plain(
        &self,
        c: &Commands<'_>,
        w: &Weight,
        b: &Buffer,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
        epilogue: u32,
    ) {
        c.dispatch(
            if self.bf16_activations {
                "qalign_project"
            } else if w.ty == 1 {
                "qasr_half_mm"
            } else {
                "vis_bmm32"
            },
            &[&w.buffer, x, y, b],
            &[w.k as u32, w.n as u32, rows as u32, epilogue],
            [w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
    fn linear(
        &self,
        c: &Commands<'_>,
        p: &Linear,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
        epilogue: u32,
    ) {
        self.plain(c, &p.w, &p.b.buffer, x, y, rows, epilogue);
    }
    fn gelu(&self, c: &Commands<'_>, x: &Buffer, n: usize) {
        c.dispatch(
            if self.bf16_activations {
                "qalign_gelu"
            } else {
                "uov_activation"
            },
            &[x],
            &[n as u32, 0],
            [n.div_ceil(256), 1, 1],
            256,
        );
    }
    fn gelu_bf(c: &Commands<'_>, x: &Buffer, out: &Buffer, n: usize) {
        c.dispatch(
            "qalign_gelu_bf",
            &[x, out],
            &[n as u32],
            [n.div_ceil(256), 1, 1],
            256,
        );
    }
    pub(super) fn step(&self, d: &MetalDevice, j: &mut Job) -> Result<Option<Vec<Buffer>>> {
        let c = d.begin()?;
        if self.bf16_activations && j.chunk == 0 && j.stage == 0 && j.layer == 0 {
            c.dispatch(
                "qalign_round",
                &[&j.mel],
                &[(j.mel.len() / 4) as u32],
                [(j.mel.len() / 4).div_ceil(256), 1, 1],
                256,
            );
        }
        if j.chunk < j.chunks {
            let count = (j.chunks - j.chunk).min(GROUP);
            if j.stage < 3 {
                let (h, w, ch) = [(128u32, 100u32, 1u32), (64, 50, 480), (32, 25, 480)][j.stage];
                let rows = count * h.div_ceil(2) as usize * w.div_ceil(2) as usize;
                c.dispatch(
                    if self.bf16_activations {
                        "qalign_conv_rows"
                    } else {
                        "qasr_conv_rows"
                    },
                    &[if j.stage == 0 { &j.mel } else { &j.conv }, &j.gather],
                    &[h, w, ch, count as u32, j.stage as u32, j.chunk as u32],
                    [(rows * ch as usize * 9).div_ceil(256), 1, 1],
                    256,
                );
                self.linear(&c, &self.conv[j.stage], &j.gather, &j.conv, rows, 1);
                self.gelu(&c, &j.conv, rows * 480);
                j.stage += 1;
            } else {
                c.dispatch(
                    if self.bf16_activations {
                        "qalign_flatten"
                    } else {
                        "qasr_flatten"
                    },
                    &[&j.conv, &j.gather],
                    &[count as u32],
                    [(count * 13 * 7680).div_ceil(256), 1, 1],
                    256,
                );
                self.plain(
                    &c,
                    &self.conv_out,
                    &self.pos.buffer,
                    &j.gather,
                    &j.projected,
                    count * 13,
                    0,
                );
                c.dispatch(
                    if self.bf16_activations {
                        "qalign_position"
                    } else {
                        "qasr_position"
                    },
                    &[&j.projected, &self.pos.buffer, &j.x, &j.map],
                    &[(count * 13) as u32, (j.chunk * 13) as u32],
                    [(count * 13 * E).div_ceil(256), 1, 1],
                    256,
                );
                j.chunk += count;
                j.stage = 0;
            }
        } else if j.layer < 24 {
            let l = &self.blocks[j.layer];
            self.norm(&c, &l.ln1, &j.x, &j.norm, j.rows);
            self.linear(&c, &l.qkv, &j.norm, &j.qkv, j.rows, 1);
            c.dispatch(
                if self.bf16_activations {
                    "qalign_heads"
                } else {
                    "qasr_heads"
                },
                &[&j.qkv, &j.q, &j.k, &j.v],
                &[j.rows as u32],
                [((j.rows + 64) * E).div_ceil(256), 1, 1],
                256,
            );
            c.dispatch(
                if self.bf16_activations {
                    "qalign_audio_attention"
                } else {
                    "qasr_attention"
                },
                &[&j.q, &j.k, &j.v, &j.attn, &j.tiles],
                &[if self.bf16_activations { 30 } else { 0 }],
                [16, j.tiles.len() / 16, 1],
                64,
            );
            self.linear(&c, &l.out, &j.attn, &j.x, j.rows, 2);
            self.norm(&c, &l.ln2, &j.x, &j.norm, j.rows);
            self.linear(&c, &l.up, &j.norm, &j.ff, j.rows, 1);
            if self.bf16_activations {
                Self::gelu_bf(&c, &j.ff, &j.gather, j.rows * 4096);
                self.linear(&c, &l.down, &j.gather, &j.x, j.rows, 2);
            } else {
                self.gelu(&c, &j.ff, j.rows * 4096);
                self.linear(&c, &l.down, &j.ff, &j.x, j.rows, 2);
            }
            j.layer += 1;
        } else {
            self.norm(&c, &self.post, &j.x, &j.norm, j.rows);
            self.linear(&c, &self.up, &j.norm, &j.attn, j.rows, 1);
            if self.bf16_activations {
                Self::gelu_bf(&c, &j.attn, &j.norm, j.rows * E);
                self.linear(&c, &self.down, &j.norm, &j.output, j.rows, 1);
            } else {
                self.gelu(&c, &j.attn, j.rows * E);
                self.linear(&c, &self.down, &j.attn, &j.output, j.rows, 1);
            }
            let mut outputs = Vec::new();
            let mut offset = 0;
            let width = self.down.w.n;
            for &n in &j.sizes {
                let out = d.alloc(n * width * 4)?;
                c.dispatch(
                    "qasr_extract",
                    &[&j.output, &out],
                    &[(n * width) as u32, (offset * width) as u32],
                    [(n * width).div_ceil(256), 1, 1],
                    256,
                );
                outputs.push(out);
                offset += n;
            }
            c.finish()?;
            #[cfg(test)]
            if self.bf16_activations {
                super::aligner::trace("audio-output", &j.output, j.rows * self.down.w.n);
            }
            return Ok(Some(outputs));
        }
        c.finish()?;
        #[cfg(test)]
        if self.bf16_activations && j.chunk == 0 && j.stage > 0 {
            super::aligner::trace(
                &format!("conv-{}", j.stage),
                &j.conv,
                [50 * 64 * 480, 25 * 32 * 480, 13 * 16 * 480][j.stage - 1],
            );
            if j.stage == 1 {
                super::aligner::trace("mel", &j.mel, j.mel.len() / 4);
            }
        }
        #[cfg(test)]
        if self.bf16_activations && j.chunk == j.chunks {
            super::aligner::trace(&format!("audio-{}", j.layer), &j.x, j.rows * E);
        }
        Ok(None)
    }
}
