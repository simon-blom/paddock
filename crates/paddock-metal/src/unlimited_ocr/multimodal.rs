use super::*;
use paddock_engine::{
    generator::{GenError, Generator, MmAdmit},
    service::{MmChunk, OcrCropMode},
};
pub(super) struct Image {
    pub offset: usize,
    pub tokens: usize,
    pub embd: Buffer,
}
pub(super) struct Layout {
    pub images: Vec<Image>,
}
pub(super) struct Encoding {
    slot: usize,
    ids: Vec<u32>,
    spans: Vec<(usize, usize)>,
    inputs: VecDeque<vision::Input>,
    job: Option<vision::Job>,
    outputs: Vec<Buffer>,
    aborted: bool,
}
impl Encoding {
    pub fn owns(&self, slot: usize) -> bool {
        self.slot == slot && !self.aborted
    }
    pub fn abort(&mut self, slot: usize) {
        if self.slot == slot {
            self.aborted = true;
        }
    }
    pub fn empty(&self) -> bool {
        self.aborted
    }
}
// Same deterministic geometric election as the CUDA family. Equal-count
// aspect ties are ordered by (tiles,cols,rows), not Python set hash order.
fn grid(w: usize, h: usize) -> (usize, usize) {
    if w <= 640 && h <= 640 {
        return (0, 0);
    }
    let mut candidates = Vec::new();
    for c in 1..=32 {
        for r in 1..=32 {
            if (2..=32).contains(&(c * r)) {
                candidates.push((c * r, c, r));
            }
        }
    }
    candidates.sort_unstable();
    let mut best = (1, 1);
    let mut delta = f64::INFINITY;
    for (tiles, c, r) in candidates {
        let diff = (w as f64 / h as f64 - c as f64 / r as f64).abs();
        if diff < delta || (diff == delta && (w * h) as f64 > 0.5 * 640. * 640. * tiles as f64) {
            best = (c, r);
            delta = diff;
        }
    }
    best
}
impl UnlimitedOcr {
    pub fn attach_vision(&mut self, path: &Path) -> Result<()> {
        if self.vision.is_some()
            || !self.pending.is_empty()
            || !self.encoding.is_empty()
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
    fn plan(&self, slot: usize, chunks: Vec<MmChunk>) -> Result<Encoding> {
        if slot >= self.slots.len()
            || self.pending.iter().any(|p| p.slot == slot)
            || self.encoding.iter().any(|e| e.owns(slot))
            || self.vision.is_none()
        {
            return Err(error("invalid/busy image slot or missing tower"));
        }
        let count = chunks
            .iter()
            .filter(|c| matches!(c, MmChunk::Image { .. }))
            .count();
        if count == 0 || count > vision::MAX_IMAGES {
            return Err(error("request needs 1..16 images"));
        }
        let base = count > 1
            || chunks
                .iter()
                .any(|c| matches!(c, MmChunk::OcrCrop(OcrCropMode::Base)));
        let mut e = Encoding {
            slot,
            ids: Vec::new(),
            spans: Vec::new(),
            inputs: VecDeque::new(),
            job: None,
            outputs: Vec::new(),
            aborted: false,
        };
        let mut image_tokens = 0;
        let mut rgb_bytes = 0;
        for c in chunks {
            match c {
                MmChunk::Text(ids) => {
                    if ids.iter().any(|&t| t as usize >= VOCAB || t == IMAGE) {
                        return Err(error("invalid token or unmatched image placeholder"));
                    }
                    e.ids.extend(ids);
                }
                MmChunk::Image { rgb, w, h } => {
                    if w == 0
                        || h == 0
                        || w > 8192
                        || h > 8192
                        || w.checked_mul(h).and_then(|x| x.checked_mul(3)) != Some(rgb.len())
                        || w.max(h) > w.min(h) * 512
                    {
                        return Err(error("invalid RGB dimensions or aspect"));
                    }
                    rgb_bytes += rgb.len();
                    if rgb_bytes > 128 * 1024 * 1024 {
                        return Err(error("request RGB exceeds 128 MiB"));
                    }
                    let (cols, rows) = if base { (0, 0) } else { grid(w, h) };
                    let input = vision::Input {
                        rgb,
                        w,
                        h,
                        cols,
                        rows,
                    };
                    let n = input.tokens();
                    image_tokens += n;
                    if image_tokens > vision::MAX_TOKENS || e.ids.len() + n > self.context {
                        return Err(error("expanded images exceed context/token budget"));
                    }
                    e.spans.push((e.ids.len(), n));
                    e.ids.resize(e.ids.len() + n, IMAGE);
                    e.inputs.push_back(input);
                }
                MmChunk::OcrCrop(_) => {}
                _ => {
                    return Err(error(
                        "audio and arbitrary pixel budgets unsupported by DeepEncoder",
                    ));
                }
            }
            if e.ids.len() > self.context {
                return Err(error("expanded prompt exceeds context"));
            }
        }
        Ok(e)
    }
    pub(super) fn admit_images(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        items
            .into_iter()
            .map(|(slot, chunks)| match self.plan(slot, chunks) {
                Err(e) => (slot, MmAdmit::Failed(e.into())),
                Ok(e) => {
                    let s = &mut self.slots[slot];
                    s.table.clear(&mut self.pool);
                    s.history.clear();
                    s.mm = None;
                    s.reused = 0;
                    s.prompt = e.ids.len();
                    self.encoding.push_back(e);
                    (slot, MmAdmit::Encoding)
                }
            })
            .collect()
    }
    pub(super) fn encode_images(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut e) = self.encoding.pop_front() else {
            return Vec::new();
        };
        let result = (|| -> Result<Option<Buffer>> {
            let v = self.vision.as_ref().ok_or_else(|| error("missing tower"))?;
            if e.job.is_none() {
                let input = e.inputs.pop_front().ok_or_else(|| error("missing image"))?;
                e.job = Some(v.start(&self.device, input)?);
                return Ok(None);
            }
            v.step(&self.device, e.job.as_mut().expect("started"))
        })();
        match result {
            Err(err) => vec![(e.slot, MmAdmit::Failed(err.into()))],
            Ok(output) => {
                if let Some(b) = output {
                    e.outputs.push(b);
                    e.job = None;
                }
                if e.job.is_none() && e.inputs.is_empty() {
                    let images = e
                        .spans
                        .into_iter()
                        .zip(e.outputs)
                        .map(|((offset, tokens), embd)| Image {
                            offset,
                            tokens,
                            embd,
                        })
                        .collect();
                    self.slots[e.slot].mm = Some(Layout { images });
                    self.pending.push_back(Pending {
                        slot: e.slot,
                        work: e.ids.len(),
                        tokens: e.ids,
                        offset: 0,
                    });
                    vec![(e.slot, MmAdmit::Queued)]
                } else {
                    self.encoding.push_front(e);
                    Vec::new()
                }
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry() {
        assert_eq!(grid(640, 640), (0, 0));
        assert_eq!(grid(1240, 1754), (2, 3));
        assert_eq!(3 * 10 * (2 * 10 + 1) + 273, 903);
    }
}
