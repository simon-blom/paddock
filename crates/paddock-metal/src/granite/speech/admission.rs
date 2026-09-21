use super::*;
use paddock_engine::audio::{MelFeatures, granite::audio_token_count};
use paddock_engine::{
    generator::{GenError, Generator, MmAdmit},
    service::MmChunk,
};
pub(in crate::granite) struct Span {
    pub offset: usize,
    pub tokens: usize,
    pub embd: Buffer,
}
struct Request {
    slot: usize,
    ids: Vec<u32>,
    spans: Vec<(usize, usize, usize)>,
}
pub(in crate::granite) struct Encoding {
    requests: Vec<Request>,
    clips: Vec<MelFeatures>,
    job: Option<super::Job>,
}
impl Encoding {
    pub(in crate::granite) fn owns(&self, slot: usize) -> bool {
        self.requests.iter().any(|r| r.slot == slot)
    }
    pub(in crate::granite) fn abort(&mut self, slot: usize) {
        self.requests.retain(|r| r.slot != slot);
    }
    pub(in crate::granite) fn empty(&self) -> bool {
        self.requests.is_empty()
    }
}
impl Granite {
    pub fn requires_audio(&self) -> bool {
        self.audio_id.is_some()
    }
    pub fn attach_audio(&mut self, path: &Path) -> Result<()> {
        if self.audio.is_some()
            || self.cold.is_some()
            || self.vision.is_some()
            || self.audio_id != Some(100352)
            || self.width != 2048
            || !self.deepstack.is_empty()
            || !self.audio_encoding.is_empty()
            || !self.pending.is_empty()
            || self.slots.iter().any(|s| !s.history.is_empty())
        {
            return Err(error("attach audio once, before admission"));
        }
        self.source_versions.extend(crate::offload::versions(path)?);
        let before = self.device.allocated_bytes();
        let v = super::Tower::load(&self.device, path, self.speech_plus)?;
        self.weight_bytes += self.device.allocated_bytes() - before;
        self.audio = Some(v);
        Ok(())
    }
    fn plan(&self, slot: usize, chunks: Vec<MmChunk>) -> Result<(Request, Vec<MelFeatures>)> {
        if slot >= self.slots.len()
            || self.pending.iter().any(|p| p.slot == slot)
            || self.audio_encoding.iter().any(|e| e.owns(slot))
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
                    if tokens.iter().any(|&t| t as usize >= 100353 || t == 100352) {
                        return Err(error("invalid token or unmatched audio placeholder"));
                    }
                    r.ids.extend(tokens);
                }
                MmChunk::Audio { samples, mel } => {
                    if samples.len() > super::MAX_FRAMES * 320
                        || samples.iter().any(|x| !x.is_finite())
                    {
                        return Err(error(
                            "invalid samples or audio exceeds 120-second implementation ceiling",
                        ));
                    }
                    let m = match mel {
                        Some(m) => m,
                        None => paddock_engine::audio::granite::speech_features(&samples)
                            .map_err(error)?,
                    };
                    super::validate(&m)?;
                    if m.n_samples != samples.len() {
                        return Err(error("precomputed mel/sample length mismatch"));
                    }
                    frames += m.n_frames;
                    if frames > super::MAX_FRAMES || clips.len() >= 16 {
                        return Err(error("request exceeds audio frame/clip budget"));
                    }
                    let n = audio_token_count(m.n_frames);
                    let offset = r.ids.len();
                    if offset + n > self.context {
                        return Err(error("audio-expanded prompt exceeds context"));
                    }
                    r.ids.resize(offset + n, 100352);
                    r.spans.push((clips.len(), offset, n));
                    clips.push(m);
                }
                _ => {
                    return Err(error(
                        "Granite Speech accepts audio and text, not image/crop/pixel directives",
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
    pub(in crate::granite) fn admit_audio(
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
                    let count = clips.iter().map(|m| m.n_frames).sum::<usize>();
                    if frames + count > super::MAX_FRAMES || wave.clips.len() + clips.len() > 16 {
                        self.audio_encoding.push_back(wave);
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
                    s.audio.clear();
                    s.radix_tokens.clear();
                    s.reused = 0;
                    wave.requests.push(r);
                    wave.clips.extend(clips);
                    frames += count;
                    out.push((slot, MmAdmit::Encoding));
                }
            }
        }
        if !wave.empty() {
            self.audio_encoding.push_back(wave);
        }
        out
    }
    pub(in crate::granite) fn encode_audio(&mut self) -> Vec<(usize, MmAdmit)> {
        // The engine is a long-lived Rust worker, not a Cocoa run loop. Drain
        // autoreleased submission objects at each scheduling boundary; their
        // retained buffers/results still live through their owning request.
        objc2::rc::autoreleasepool(|_| {
            // One tiny phase per language tick tied new-audio TTFT to 53
            // decode forwards, not the tower's compute cost. Spend a small
            // wall-time budget instead, then yield for decoding/cancellation.
            // A single bounded phase is non-preemptible and may exceed 4 ms
            // on the largest wave. No kernel order or precision changes.
            let start = std::time::Instant::now();
            loop {
                let out = self.encode_audio_inner();
                if !out.is_empty()
                    || self.audio_encoding.is_empty()
                    || start.elapsed() >= std::time::Duration::from_millis(4)
                {
                    return out;
                }
            }
        })
    }
    fn encode_audio_inner(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut e) = self.audio_encoding.pop_front() else {
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
                self.audio_encoding.push_front(e);
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
                            .map(|(i, offset, tokens)| Span {
                                offset,
                                tokens,
                                embd: outputs[i].take().expect("unique clip"),
                            })
                            .collect();
                        self.slots[r.slot].audio = clips;
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
    pub(in crate::granite) fn prefill_audio(
        &mut self,
        slot: usize,
        chunks: Vec<MmChunk>,
    ) -> std::result::Result<(Vec<f32>, usize), GenError> {
        for (_, a) in self.admit_audio(vec![(slot, chunks)]) {
            if let MmAdmit::Failed(e) = a {
                return Err(e);
            }
        }
        while self.audio_encoding.iter().any(|e| e.owns(slot)) {
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

impl Granite {
    pub(in crate::granite) fn audio_slot_pending(&self, slot: usize) -> bool {
        self.audio_encoding.iter().any(|e| e.owns(slot))
    }
    pub(in crate::granite) fn abort_audio(&mut self, slot: usize) {
        for e in &mut self.audio_encoding {
            e.abort(slot);
        }
        self.audio_encoding.retain(|e| !e.empty());
    }
    pub(in crate::granite) fn inject_audio(&self, c: &Commands<'_>, rows: &[(usize, u32, u32)]) {
        for (slot, s) in self.slots.iter().enumerate() {
            for span in &s.audio {
                if rows.iter().any(|r| {
                    r.0 == slot
                        && r.2 as usize >= span.offset
                        && (r.2 as usize) < span.offset + span.tokens
                }) {
                    c.dispatch(
                        "gs_inject",
                        &[&span.embd, &self.scratch.meta, &self.scratch.x],
                        &[
                            rows.len() as u32,
                            slot as u32,
                            span.offset as u32,
                            span.tokens as u32,
                            self.embedding_scale.to_bits(),
                        ],
                        [(rows.len() * 2048).div_ceil(256), 1, 1],
                        256,
                    );
                }
            }
        }
    }
}
