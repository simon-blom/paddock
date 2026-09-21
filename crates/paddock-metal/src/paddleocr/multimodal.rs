use super::*;
use paddock_engine::{
    generator::{GenError, Generator, MmAdmit},
    service::MmChunk,
};
pub(super) struct Image {
    pub offset: usize,
    pub tokens: usize,
    pub embd: Buffer,
}
pub(super) struct Layout {
    positions: Vec<[u32; 4]>,
    next: u32,
    pub images: Vec<Image>,
}
impl Layout {
    pub(super) fn position(&self, pos: usize) -> [u32; 4] {
        self.positions.get(pos).copied().unwrap_or_else(|| {
            let n = self.next + (pos - self.positions.len()) as u32;
            [n, n, n, 0]
        })
    }
}
struct Request {
    slot: usize,
    ids: Vec<u32>,
    positions: Vec<[u32; 4]>,
    next: u32,
    spans: Vec<(usize, usize, usize)>,
}
pub(super) struct Encoding {
    requests: Vec<Request>,
    images: Vec<vision::Input>,
    job: Option<vision::Job>,
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
impl PaddleOcr {
    pub fn attach_vision(&mut self, path: &Path) -> Result<()> {
        if self.vision.is_some()
            || !self.encoding.is_empty()
            || !self.pending.is_empty()
            || self.slots.iter().any(|s| !s.history.is_empty())
        {
            return Err(error("attach vision once, before admission"));
        }
        let before = self.device.allocated_bytes();
        let v = vision::Vision::load(&self.device, path)?;
        self.weight_bytes += self.device.allocated_bytes() - before;
        self.vision = Some(v);
        Ok(())
    }
    fn plan(&self, slot: usize, chunks: Vec<MmChunk>) -> Result<(Request, Vec<vision::Input>)> {
        if slot >= self.slots.len()
            || self.pending.iter().any(|p| p.slot == slot)
            || self.encoding.iter().any(|e| e.owns(slot))
            || self.vision.is_none()
        {
            return Err(error("invalid/busy multimodal slot or missing tower"));
        }
        let (mut min, mut max) = (112896usize, 1003520usize);
        for c in &chunks {
            if let MmChunk::VisionPixels {
                min_pixels,
                max_pixels,
            } = c
            {
                if let Some(n) = min_pixels {
                    min = usize::try_from(*n).map_err(|_| error("pixel budget overflow"))?;
                }
                if let Some(n) = max_pixels {
                    max = usize::try_from(*n).map_err(|_| error("pixel budget overflow"))?;
                }
            }
        }
        let mut r = Request {
            slot,
            ids: Vec::new(),
            positions: Vec::new(),
            next: 0,
            spans: Vec::new(),
        };
        let mut images = Vec::new();
        for c in chunks {
            match c {
                MmChunk::Text(tokens) => {
                    if tokens.iter().any(|&t| t as usize >= VOCAB || t == IMAGE) {
                        return Err(error("invalid token or unmatched image placeholder"));
                    }
                    for t in tokens {
                        r.ids.push(t);
                        r.positions.push([r.next, r.next, r.next, 0]);
                        r.next += 1;
                    }
                }
                MmChunk::Image { rgb, w, h } => {
                    if w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
                        return Err(error("RGB size mismatch"));
                    }
                    let (tw, th) = vision::resize(w, h, min, max)?;
                    let (nx, ny) = (tw / 28, th / 28);
                    let n = nx * ny;
                    let offset = r.ids.len();
                    if offset + n > self.context {
                        return Err(error("image-expanded prompt exceeds context"));
                    }
                    for i in 0..n {
                        r.ids.push(IMAGE);
                        r.positions.push([
                            r.next,
                            r.next + (i / nx) as u32,
                            r.next + (i % nx) as u32,
                            0,
                        ]);
                    }
                    r.next += nx.max(ny) as u32;
                    r.spans.push((images.len(), offset, n));
                    images.push(vision::Input { rgb, w, h, tw, th });
                }
                MmChunk::VisionPixels { .. } => {}
                _ => {
                    return Err(error(
                        "audio and crop directives are not supported by this tower",
                    ));
                }
            }
            if r.ids.len() > self.context {
                return Err(error("image-expanded prompt exceeds context"));
            }
        }
        if images.is_empty()
            || images.iter().map(vision::Input::patches).sum::<usize>() > vision::MAX_PATCHES
        {
            return Err(error(
                "request needs images within the 16384-patch aggregate budget",
            ));
        }
        Ok((r, images))
    }
    pub(super) fn admit_images(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        let mut out = Vec::new();
        let mut wave = Encoding {
            requests: Vec::new(),
            images: Vec::new(),
            job: None,
        };
        let mut patches = 0;
        for (slot, chunks) in items {
            let planned = if wave.owns(slot) {
                Err(error("duplicate image slot"))
            } else {
                self.plan(slot, chunks)
            };
            match planned {
                Err(e) => out.push((slot, MmAdmit::Failed(e.into()))),
                Ok((mut r, images)) => {
                    let count = images.iter().map(vision::Input::patches).sum::<usize>();
                    if patches + count > vision::MAX_PATCHES {
                        self.encoding.push_back(wave);
                        wave = Encoding {
                            requests: Vec::new(),
                            images: Vec::new(),
                            job: None,
                        };
                        patches = 0;
                    }
                    for span in &mut r.spans {
                        span.0 += wave.images.len();
                    }
                    let s = &mut self.slots[slot];
                    s.table.clear(&mut self.pool);
                    s.history.clear();
                    s.mm = None;
                    s.reused = 0;
                    wave.requests.push(r);
                    wave.images.extend(images);
                    patches += count;
                    out.push((slot, MmAdmit::Encoding));
                }
            }
        }
        if !wave.empty() {
            self.encoding.push_back(wave);
        }
        out
    }
    pub(super) fn encode_images(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut e) = self.encoding.pop_front() else {
            return Vec::new();
        };
        let result = (|| -> Result<Option<Vec<Buffer>>> {
            let v = self.vision.as_ref().ok_or_else(|| error("missing tower"))?;
            if e.job.is_none() {
                e.job = Some(v.start(&self.device, &e.images)?);
                e.images.clear();
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
                        let images = r
                            .spans
                            .into_iter()
                            .map(|(i, offset, tokens)| Image {
                                offset,
                                tokens,
                                embd: outputs[i].take().expect("unique image"),
                            })
                            .collect();
                        self.slots[r.slot].mm = Some(Layout {
                            positions: r.positions,
                            next: r.next,
                            images,
                        });
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
    pub(super) fn prefill_images(
        &mut self,
        slot: usize,
        chunks: Vec<MmChunk>,
    ) -> std::result::Result<(Vec<f32>, usize), GenError> {
        for (_, a) in self.admit_images(vec![(slot, chunks)]) {
            if let MmAdmit::Failed(e) = a {
                return Err(e);
            }
        }
        while self.encoding.iter().any(|e| e.owns(slot)) {
            for (s, a) in self.encode_images() {
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
                return Err(GenError::Backend("image prompt disappeared".into()));
            }
        }
    }
}
