//! Image-aware admission and prefix identity. Encoder jobs yield between
//! blocks; language prefill never cuts a bidirectional image span in half.
use super::*;
use paddock_engine::{generator::MmAdmit, service::MmChunk};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, PartialEq, Eq)]
pub(super) struct ImageKey {
    hash: u64,
    rgb: Arc<[u8]>,
    w: usize,
    h: usize,
    offset: usize,
    tokens: usize,
}
impl ImageKey {
    fn same_image(&self, other: &Self) -> bool {
        self.hash == other.hash && self.w == other.w && self.h == other.h && self.rgb == other.rgb
    }
}
pub(super) struct CachedImage {
    key: ImageKey,
    output: vision::Output,
    touched: u64,
}
pub(super) struct Layout {
    pub(super) ids: Vec<u32>,
    pub(super) keys: Vec<ImageKey>,
    pub(super) images: Vec<vision::Output>,
}
impl Layout {
    pub(super) fn limit(&self, pos: usize) -> usize {
        self.keys
            .iter()
            .find(|k| pos >= k.offset && pos < k.offset + k.tokens)
            .map_or(pos, |k| k.offset + k.tokens - 1)
    }
    /// Walk a proposed cut backwards, or admit one whole image if it is the
    /// very next unit and fits scratch. The service's token grant is soft;
    /// the allocation bound is hard. Decode riders retain their own rows.
    pub(super) fn grant(&self, offset: usize, wanted: usize, capacity: usize) -> usize {
        let end = (offset + wanted.min(capacity)).min(self.ids.len());
        if let Some(k) = self
            .keys
            .iter()
            .find(|k| end > k.offset && end < k.offset + k.tokens)
        {
            if k.offset == offset && k.tokens <= capacity {
                k.tokens
            } else {
                k.offset - offset
            }
        } else {
            end - offset
        }
    }
}
pub(super) fn prefix_cut(old: &[ImageKey], new: &[ImageKey], mut cut: usize) -> usize {
    for key in old.iter().chain(new) {
        if key.offset >= cut {
            continue;
        }
        if !old.contains(key) || !new.contains(key) || cut < key.offset + key.tokens {
            cut = cut.min(key.offset);
        }
    }
    cut
}
pub(super) struct Encoding {
    requests: Vec<(usize, Layout)>,
    unique: Vec<ImageKey>,
    outputs: Vec<Option<vision::Output>>,
    missing: Vec<usize>,
    in_flight: Vec<usize>,
    job: Option<tower::Job>,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(s.into())
}

#[cfg(test)]
#[path = "image_layout_tests.rs"]
mod tests;
impl Gemma4 {
    pub fn attach_vision(&mut self, path: &Path) -> Result<()> {
        self.require_committed()?;
        if self.vision.is_some()
            || self.image_markers.is_none()
            || self.slots.iter().any(|s| !s.history.is_empty())
            || !self.encoding.is_empty()
            || !self.pending.is_empty()
        {
            return Err(error(
                "attach vision once before prefill, with image markers present",
            ));
        }
        let before = self.device.allocated_bytes();
        let vision = tower::Tower::load(&self.device, path, self.width, self.muse)?;
        self.weight_bytes += self.device.allocated_bytes() - before;
        self.vision = Some(vision);
        Ok(())
    }
    pub fn image_cache_reuses(&self) -> u64 {
        self.image_cache_reused
    }
    pub(super) fn image_slot_pending(&self, slot: usize) -> bool {
        self.encoding
            .iter()
            .any(|e| e.requests.iter().any(|r| r.0 == slot))
    }
    fn layout(&self, chunks: Vec<MmChunk>) -> Result<Layout> {
        if self.vision.is_none() {
            return Err(error("configure --mmproj for images"));
        }
        let (begin, end) = self
            .image_markers
            .ok_or_else(|| error("image markers missing"))?;
        let mut layout = Layout {
            ids: Vec::new(),
            keys: Vec::new(),
            images: Vec::new(),
        };
        for chunk in chunks {
            match chunk {
                MmChunk::Text(ids) => {
                    if ids.iter().any(|&t| t as usize >= self.vocab) {
                        return Err(error("invalid multimodal text token"));
                    }
                    layout.ids.extend(ids);
                }
                MmChunk::Image { rgb, w, h } => {
                    let (tw, th) = if self.muse {
                        muse_vision::resize(w, h)?
                    } else if self.mlx {
                        vision::resize_mlx(w, h)?
                    } else {
                        vision::resize(w, h)?
                    };
                    if rgb.len() != w * h * 3 {
                        return Err(error("RGB byte count mismatch"));
                    }
                    let tokens = tw * th / if self.muse { 784 } else { 2304 };
                    if layout.ids.len().saturating_add(tokens + 2) > self.context {
                        return Err(error("image-expanded prompt exceeds context"));
                    }
                    layout.ids.push(begin);
                    let offset = layout.ids.len();
                    let mut hash = std::collections::hash_map::DefaultHasher::new();
                    rgb.hash(&mut hash);
                    w.hash(&mut hash);
                    h.hash(&mut hash);
                    layout.keys.push(ImageKey {
                        hash: hash.finish(),
                        rgb: rgb.into(),
                        w,
                        h,
                        offset,
                        tokens,
                    });
                    layout.ids.resize(offset + tokens, 0);
                    layout.ids.push(end);
                }
                _ => {
                    return Err(error(
                        "this Metal family supports text/images, not audio or OCR directives",
                    ));
                }
            }
            if layout.ids.len() > self.context {
                return Err(error("image-expanded prompt exceeds context"));
            }
        }
        if layout.keys.is_empty() {
            return Err(error("multimodal prompt has no images"));
        }
        Ok(layout)
    }
    pub(super) fn inject_images(&self, cmd: &Commands<'_>, rows: &[(usize, u32, u32)]) {
        for (slot, state) in self.slots.iter().enumerate() {
            let Some(layout) = &state.mm else {
                continue;
            };
            for (key, image) in layout.keys.iter().zip(&layout.images) {
                if let Some(first) = rows.iter().position(|r| {
                    r.0 == slot
                        && r.2 as usize >= key.offset
                        && (r.2 as usize) < key.offset + key.tokens
                }) {
                    let offset = rows[first].2 as usize - key.offset;
                    let count = rows[first..]
                        .iter()
                        .take_while(|r| r.0 == slot && (r.2 as usize) < key.offset + key.tokens)
                        .count();
                    copy_words(
                        cmd,
                        &image.embd,
                        &self.scratch.x,
                        offset * self.width,
                        first * self.width,
                        count * self.width,
                    );
                }
            }
        }
    }
    fn copy_image(&self, image: &vision::Output) -> Result<vision::Output> {
        let embd = self.device.alloc(image.embd.len())?;
        let cmd = self.device.begin()?;
        copy_words(&cmd, &image.embd, &embd, 0, 0, embd.len() / 4);
        cmd.finish()?;
        Ok(vision::Output {
            embd,
            tokens: image.tokens,
        })
    }
    pub(super) fn admit_images(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        let mut verdicts = Vec::new();
        let mut wave = Encoding {
            requests: Vec::new(),
            unique: Vec::new(),
            outputs: Vec::new(),
            missing: Vec::new(),
            in_flight: Vec::new(),
            job: None,
        };
        for (slot, chunks) in items {
            let result = (|| {
                self.require_committed()?;
                if slot >= self.slots.len()
                    || self.pending.iter().any(|p| p.slot == slot)
                    || self.image_slot_pending(slot)
                    || wave.requests.iter().any(|r| r.0 == slot)
                {
                    return Err(error("invalid or already-admitted image slot"));
                }
                let layout = self.layout(chunks)?;
                for key in &layout.keys {
                    if !wave.unique.iter().any(|k| key.same_image(k)) {
                        wave.unique.push(key.clone());
                    }
                }
                wave.requests.push((slot, layout));
                Ok(())
            })();
            verdicts.push((
                slot,
                match result {
                    Ok(()) => MmAdmit::Encoding,
                    Err(e) => MmAdmit::Failed(e.into()),
                },
            ));
        }
        if !wave.requests.is_empty() {
            self.encoding.push_back(wave);
        }
        verdicts
    }
    fn encode_wave(&mut self, wave: &mut Encoding) -> Result<bool> {
        let started = Instant::now();
        // DFlash's full-block latency class also bounds encoder work. Dense
        // riders still run every tick (spec is deferred during encoding).
        // Gemma and non-DFlash Muse retain their qualified 96/160 ms limits.
        let (encoder_ms, tick_ms) = if self.muse && self.dflash.is_some() {
            (240, 300)
        } else {
            (96, 160)
        };
        let quantum = Duration::from_millis(encoder_ms).min(
            Duration::from_millis(tick_ms)
                .saturating_sub(Duration::from_secs_f64(self.last_gpu_seconds)),
        );
        if wave.outputs.is_empty() {
            for (i, key) in wave.unique.iter().enumerate() {
                if let Some(at) = self.image_cache.iter().position(|c| key.same_image(&c.key)) {
                    self.clock += 1;
                    self.image_cache[at].touched = self.clock;
                    self.image_cache_reused += 1;
                    wave.outputs
                        .push(Some(self.copy_image(&self.image_cache[at].output)?));
                } else {
                    wave.outputs.push(None);
                    wave.missing.push(i);
                }
            }
        }
        if wave.job.is_none() && !wave.missing.is_empty() {
            let mut patches = 0;
            let count = wave
                .missing
                .iter()
                .take_while(|&&i| {
                    patches += wave.unique[i].tokens * if self.muse { 4 } else { 9 };
                    patches
                        <= if self.muse {
                            muse_vision::MAX_PATCHES
                        } else {
                            vision::MAX_PATCHES
                        }
                })
                .count();
            wave.in_flight = wave.missing.drain(..count).collect();
            let images = wave
                .in_flight
                .iter()
                .map(|&i| {
                    let k = &wave.unique[i];
                    (&*k.rgb, k.w, k.h)
                })
                .collect::<Vec<_>>();
            wave.job = Some(
                self.vision
                    .as_ref()
                    .expect("attached")
                    .start(&self.device, &images)?,
            );
            if started.elapsed() >= quantum {
                return Ok(false);
            }
        }
        if let Some(job) = &mut wave.job {
            let outputs = loop {
                if let Some(outputs) = self.vision.as_ref().expect("attached").step(
                    &self.device,
                    job,
                    quantum.saturating_sub(started.elapsed()),
                )? {
                    break outputs;
                }
                if started.elapsed().saturating_add(job.cost()) >= quantum {
                    return Ok(false);
                }
            };
            tracing::info!(
                images = outputs.len(),
                gpu_ms = job.gpu_seconds() * 1000.,
                family = if self.muse { "Muse" } else { "Gemma" },
                "Metal vision wave encoded"
            );
            for (&i, output) in wave.in_flight.iter().zip(outputs) {
                let size = output.embd.len() + wave.unique[i].rgb.len();
                while !self.image_cache.is_empty()
                    && (self.image_cache.len() >= 16
                        || self
                            .image_cache
                            .iter()
                            .map(|c| c.output.embd.len() + c.key.rgb.len())
                            .sum::<usize>()
                            .saturating_add(size)
                            > 256 << 20)
                {
                    let at = self
                        .image_cache
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, c)| c.touched)
                        .expect("nonempty")
                        .0;
                    self.image_cache.swap_remove(at);
                }
                if size <= 256 << 20 {
                    self.clock += 1;
                    let cached = CachedImage {
                        key: wave.unique[i].clone(),
                        output: self.copy_image(&output)?,
                        touched: self.clock,
                    };
                    self.image_cache.push(cached);
                }
                wave.outputs[i] = Some(output);
            }
            wave.job = None;
            wave.in_flight.clear();
            if !wave.missing.is_empty() {
                return Ok(false);
            }
        }
        for (_, layout) in &mut wave.requests {
            for key in &layout.keys {
                let i = wave
                    .unique
                    .iter()
                    .position(|k| key.same_image(k))
                    .expect("known image");
                layout
                    .images
                    .push(self.copy_image(wave.outputs[i].as_ref().expect("encoded"))?);
            }
        }
        Ok(true)
    }
    pub(super) fn step_images(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut wave) = self.encoding.pop_front() else {
            return Vec::new();
        };
        match self.encode_wave(&mut wave) {
            Ok(false) => {
                self.encoding.push_front(wave);
                Vec::new()
            }
            Err(e) => wave
                .requests
                .into_iter()
                .map(|(slot, _)| {
                    (
                        slot,
                        MmAdmit::Failed(match &e {
                            MetalError::Memory(_) => GenError::OutOfMemory,
                            _ => GenError::Backend(e.to_string()),
                        }),
                    )
                })
                .collect(),
            Ok(true) => wave
                .requests
                .into_iter()
                .map(|(slot, layout)| {
                    let tokens = layout.ids.clone();
                    let result = self.prepare_mm(slot, &tokens, Some(layout));
                    (
                        slot,
                        match result {
                            Ok(n) => {
                                self.pending.push_back(Pending {
                                    slot,
                                    work: tokens.len() - n,
                                    tokens,
                                    offset: n,
                                });
                                MmAdmit::Queued
                            }
                            Err(e) => MmAdmit::Failed(e.into()),
                        },
                    )
                })
                .collect(),
        }
    }
    pub(super) fn abort_images(&mut self, slot: usize) {
        for wave in &mut self.encoding {
            wave.requests.retain(|r| r.0 != slot);
        }
        self.encoding.retain(|wave| !wave.requests.is_empty());
    }
    pub(super) fn prefill_images(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize)> {
        if !self.pending.is_empty() || !self.encoding.is_empty() {
            return Err(error("synchronous images require an empty admission queue"));
        }
        for (_, v) in self.admit_images(vec![(slot, chunks.to_vec())]) {
            if let MmAdmit::Failed(e) = v {
                return Err(error(e.to_string()));
            }
        }
        while self.image_slot_pending(slot) {
            for (_, v) in self.step_images() {
                if let MmAdmit::Failed(e) = v {
                    return Err(error(e.to_string()));
                }
            }
        }
        loop {
            let (_, done) = self
                .forward_mixed(&[], CHUNK)
                .map_err(|e| error(e.to_string()))?;
            if let Some((_, logits, n)) = done.into_iter().find(|r| r.0 == slot) {
                return Ok((logits, n));
            }
        }
    }
}
