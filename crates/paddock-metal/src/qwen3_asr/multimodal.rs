use super::*;
use paddock_engine::audio::{MelFeatures, audio_token_count};
use paddock_engine::{
    generator::{GenError, Generator, MmAdmit},
    service::MmChunk,
};
pub(super) struct AudioSpan {
    pub offset: usize,
    pub tokens: usize,
    pub embd: Buffer,
}
pub(super) struct Layout {
    pub audio: Vec<AudioSpan>,
}
struct Request {
    slot: usize,
    ids: Vec<u32>,
    spans: Vec<(usize, usize, usize)>,
}
pub(super) struct Encoding {
    requests: Vec<Request>,
    clips: Vec<MelFeatures>,
    job: Option<audio::Job>,
}
impl Encoding {
    pub(super) fn owns(&self, slot: usize) -> bool {
        self.requests.iter().any(|r| r.slot == slot)
    }
    pub(super) fn abort(&mut self, slot: usize) {
        self.requests.retain(|r| r.slot != slot);
    }
    pub(super) fn empty(&self) -> bool {
        self.requests.is_empty()
    }
}
impl Qwen3Asr {
    pub fn attach_audio(&mut self, path: &Path) -> Result<()> {
        if self.audio.is_some()
            || !self.encoding.is_empty()
            || !self.pending.is_empty()
            || self.slots.iter().any(|s| !s.history.is_empty())
        {
            return Err(error("attach audio once, before admission"));
        }
        let before = self.device.allocated_bytes();
        let v = audio::Tower::load(&self.device, path)?;
        self.weight_bytes += self.device.allocated_bytes() - before;
        self.audio = Some(v);
        Ok(())
    }
    fn plan(&self, slot: usize, chunks: Vec<MmChunk>) -> Result<(Request, Vec<MelFeatures>)> {
        if slot >= self.slots.len()
            || self.pending.iter().any(|p| p.slot == slot)
            || self.encoding.iter().any(|e| e.owns(slot))
            || self.audio.is_none()
        {
            return Err(error("invalid/busy multimodal slot or missing audio tower"));
        }
        let mut r = Request {
            slot,
            ids: Vec::new(),
            spans: Vec::new(),
        };
        let mut clips = Vec::new();
        let mut frames = 0;
        for c in chunks {
            match c {
                MmChunk::Text(tokens) => {
                    if tokens.iter().any(|&t| t as usize >= VOCAB || t == AUDIO) {
                        return Err(error("invalid token or unmatched audio placeholder"));
                    }
                    r.ids.extend(tokens);
                }
                MmChunk::Audio { samples, mel } => {
                    if samples.len() > audio::MAX_FRAMES * 160
                        || samples.iter().any(|x| !x.is_finite())
                    {
                        return Err(error(
                            "invalid samples or audio exceeds 120-second implementation ceiling",
                        ));
                    }
                    let m = match mel {
                        Some(m) => m,
                        None => {
                            paddock_engine::audio::qwen3_asr_features(&samples).map_err(error)?
                        }
                    };
                    audio::validate(&m)?;
                    frames += m.n_frames.div_ceil(100) * 100;
                    if frames > audio::MAX_FRAMES || clips.len() >= 16 {
                        return Err(error("request exceeds audio frame/clip budget"));
                    }
                    let n = audio_token_count(m.n_frames);
                    let offset = r.ids.len();
                    if offset + n > self.context {
                        return Err(error("audio-expanded prompt exceeds context"));
                    }
                    r.ids.resize(offset + n, AUDIO);
                    r.spans.push((clips.len(), offset, n));
                    clips.push(m);
                }
                _ => {
                    return Err(error(
                        "Qwen3-ASR accepts audio and text, not image/crop/pixel directives",
                    ));
                }
            }
            if r.ids.len() > self.context {
                return Err(error("audio-expanded prompt exceeds context"));
            }
        }
        if clips.is_empty() {
            return Err(error("request needs audio"));
        }
        Ok((r, clips))
    }
    pub(super) fn admit_audio(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        let mut out = Vec::new();
        let mut wave = Encoding {
            requests: Vec::new(),
            clips: Vec::new(),
            job: None,
        };
        let mut frames = 0;
        for (slot, chunks) in items {
            let planned = if wave.owns(slot) {
                Err(error("duplicate audio slot"))
            } else {
                self.plan(slot, chunks)
            };
            match planned {
                Err(e) => out.push((slot, MmAdmit::Failed(e.into()))),
                Ok((mut r, clips)) => {
                    let count = clips
                        .iter()
                        .map(|m| m.n_frames.div_ceil(100) * 100)
                        .sum::<usize>();
                    if frames + count > audio::MAX_FRAMES || wave.clips.len() + clips.len() > 16 {
                        self.encoding.push_back(wave);
                        wave = Encoding {
                            requests: Vec::new(),
                            clips: Vec::new(),
                            job: None,
                        };
                        frames = 0;
                    }
                    for span in &mut r.spans {
                        span.0 += wave.clips.len();
                    }
                    let s = &mut self.slots[slot];
                    s.table.clear(&mut self.pool);
                    s.history.clear();
                    s.mm = None;
                    s.reused = 0;
                    wave.requests.push(r);
                    wave.clips.extend(clips);
                    frames += count;
                    out.push((slot, MmAdmit::Encoding));
                }
            }
        }
        if !wave.empty() {
            self.encoding.push_back(wave);
        }
        out
    }
    pub(super) fn encode_audio(&mut self) -> Vec<(usize, MmAdmit)> {
        // The engine is a long-lived Rust worker, not a Cocoa run loop. Drain
        // autoreleased submission objects at each scheduling boundary; their
        // retained buffers/results still live through their owning request.
        objc2::rc::autoreleasepool(|_| self.encode_audio_inner())
    }
    fn encode_audio_inner(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut e) = self.encoding.pop_front() else {
            return Vec::new();
        };
        let result = (|| -> Result<Option<Vec<Buffer>>> {
            let v = self
                .audio
                .as_ref()
                .ok_or_else(|| error("missing audio tower"))?;
            if e.job.is_none() {
                e.job = Some(v.start(&self.device, &e.clips)?);
                e.clips.clear();
                return Ok(None);
            }
            v.step(&self.device, e.job.as_mut().expect("started"))
        })();
        match result {
            Ok(None) => {
                // Only one tower workspace is live at once. A packed wave
                // still yields after each block; the language lane can run
                // before the scheduler calls us again.
                self.encoding.push_front(e);
                Vec::new()
            }
            Err(err) => e
                .requests
                .into_iter()
                .map(|r| (r.slot, MmAdmit::Failed(GenError::Backend(err.to_string()))))
                .collect(),
            Ok(Some(outputs)) => {
                let mut outputs = outputs.into_iter().map(Some).collect::<Vec<_>>();
                e.requests
                    .into_iter()
                    .map(|r| {
                        let clips = r
                            .spans
                            .into_iter()
                            .map(|(i, offset, tokens)| AudioSpan {
                                offset,
                                tokens,
                                embd: outputs[i].take().expect("unique clip"),
                            })
                            .collect();
                        self.slots[r.slot].mm = Some(Layout { audio: clips });
                        self.pending.push_back(Pending {
                            slot: r.slot,
                            work: r.ids.len(),
                            tokens: r.ids,
                            offset: 0,
                        });
                        (r.slot, MmAdmit::Queued)
                    })
                    .collect()
            }
        }
    }
    pub(super) fn prefill_audio(
        &mut self,
        slot: usize,
        chunks: Vec<MmChunk>,
    ) -> std::result::Result<(Vec<f32>, usize), GenError> {
        for (_, a) in self.admit_audio(vec![(slot, chunks)]) {
            if let MmAdmit::Failed(e) = a {
                return Err(e);
            }
        }
        while self.encoding.iter().any(|e| e.owns(slot)) {
            for (s, a) in self.encode_audio() {
                if s == slot
                    && let MmAdmit::Failed(e) = a
                {
                    return Err(e);
                }
            }
        }
        loop {
            let (_, done) = self.forward_mixed(&[], CHUNK)?;
            for (s, logits, n) in done {
                if s == slot {
                    return Ok((logits, n));
                }
            }
            if !self.pending.iter().any(|p| p.slot == slot) {
                return Err(GenError::Backend("audio prompt disappeared".into()));
            }
        }
    }
}
