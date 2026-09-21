//! Image admission, exact-content cache identity and chunk-global DeepStack
//! injection. Virtual radix symbols are outside the text vocabulary and never
//! recycled. A hash hit is verified against dimensions and every RGB byte.
use super::*;
use paddock_engine::{generator::MmAdmit, service::MmChunk};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
    time::{Duration, Instant},
};
#[derive(Clone)]
pub(super) struct Key {
    hash: u64,
    rgb: Arc<[u8]>,
    w: usize,
    h: usize,
    offset: usize,
    tokens: usize,
}
impl Key {
    fn same(&self, other: &Self) -> bool {
        self.hash == other.hash && self.w == other.w && self.h == other.h && self.rgb == other.rgb
    }
}
pub(super) struct Layout {
    ids: Vec<u32>,
    keys: Vec<Key>,
    images: Vec<(u32, Arc<vision::Features>)>,
}
impl Layout {
    pub(super) fn radix_tokens(&self) -> Vec<u32> {
        let mut ids = self.ids.clone();
        for (k, (id, _)) in self.keys.iter().zip(&self.images) {
            ids[k.offset..k.offset + k.tokens].fill(*id);
        }
        ids
    }
}
pub(super) struct Cached {
    key: Key,
    id: u32,
    features: Arc<vision::Features>,
}
pub(super) struct Encoding {
    requests: Vec<(usize, Layout)>,
    unique: Vec<Key>,
    outputs: Vec<Option<(u32, Arc<vision::Features>)>>,
    missing: Vec<usize>,
    flight: Vec<usize>,
    job: Option<vision::Job>,
}
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Granite images: {}", s.into()))
}
impl Granite {
    pub fn attach_vision(&mut self, path: &Path) -> Result<()> {
        if self.vision.is_some()
            || self.cold.is_some()
            || self.image_id.is_none()
            || self.deepstack.is_empty()
            || self.slots.iter().any(|s| !s.history.is_empty())
            || !self.pending.is_empty()
            || !self.encoding.is_empty()
        {
            return Err(error(
                "attach a Granite vision companion once, before admission, to its matching text checkpoint",
            ));
        }
        self.source_versions.extend(crate::offload::versions(path)?);
        let before = self.device.allocated_bytes();
        let v = vision::Vision::load(&self.device, path, self.width)?;
        self.weight_bytes += self.device.allocated_bytes() - before;
        self.vision = Some(v);
        Ok(())
    }
    pub fn image_cache_reuses(&self) -> u64 {
        self.image_cache_reused
    }
    pub(super) fn image_slot_pending(&self, slot: usize) -> bool {
        self.encoding
            .iter()
            .any(|w| w.requests.iter().any(|r| r.0 == slot))
    }
    fn layout(&self, chunks: Vec<MmChunk>) -> Result<Layout> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| error("configure --mmproj for images"))?;
        let mut layout = Layout {
            ids: Vec::new(),
            keys: Vec::new(),
            images: Vec::new(),
        };
        for chunk in chunks {
            match chunk {
                MmChunk::Text(ids) => {
                    if ids.iter().any(|&t| t as usize >= self.vocab) {
                        return Err(error("invalid text token"));
                    }
                    layout.ids.extend(ids);
                }
                MmChunk::Image { rgb, w, h } => {
                    if w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
                        return Err(error("RGB size mismatch"));
                    }
                    let plan = vision.plan(w, h)?;
                    let n = plan.n_tokens();
                    let offset = layout.ids.len();
                    if offset.saturating_add(n) > self.context {
                        return Err(error("image-expanded prompt exceeds context"));
                    }
                    let mut hash = std::collections::hash_map::DefaultHasher::new();
                    rgb.hash(&mut hash);
                    w.hash(&mut hash);
                    h.hash(&mut hash);
                    layout.keys.push(Key {
                        hash: hash.finish(),
                        rgb: rgb.into(),
                        w,
                        h,
                        offset,
                        tokens: n,
                    });
                    layout
                        .ids
                        .resize(offset + n, self.image_id.expect("validated"));
                }
                _ => {
                    return Err(error(
                        "Granite Metal supports text/images, not audio or OCR directives",
                    ));
                }
            }
            if layout.ids.len() > self.context {
                return Err(error("image-expanded prompt exceeds context"));
            }
        }
        if layout.keys.is_empty() {
            return Err(error("multimodal prompt contains no images"));
        }
        Ok(layout)
    }
    pub(super) fn inject_images(
        &self,
        cmd: &Commands<'_>,
        rows: &[(usize, u32, u32)],
        stream: usize,
    ) -> bool {
        let mut changed = false;
        for (slot, s) in self.slots.iter().enumerate() {
            let Some(layout) = &s.mm else {
                continue;
            };
            for (k, (_, image)) in layout.keys.iter().zip(&layout.images) {
                if !rows.iter().any(|r| {
                    r.0 == slot && r.2 as usize >= k.offset && (r.2 as usize) < k.offset + k.tokens
                }) {
                    continue;
                }
                cmd.dispatch(
                    if stream == 0 { "vis_inject" } else { "grv_add" },
                    &[&image.streams[stream], &self.scratch.meta, &self.scratch.x],
                    &[
                        self.width as u32,
                        rows.len() as u32,
                        slot as u32,
                        k.offset as u32,
                        k.tokens as u32,
                    ],
                    [(rows.len() * self.width).div_ceil(256), 1, 1],
                    256,
                );
                changed = true;
            }
        }
        changed
    }
    pub(super) fn has_image_rows(&self, rows: &[(usize, u32, u32)]) -> bool {
        rows.iter().any(|r| {
            self.slots[r.0].mm.as_ref().is_some_and(|l| {
                l.keys
                    .iter()
                    .any(|k| r.2 as usize >= k.offset && (r.2 as usize) < k.offset + k.tokens)
            })
        })
    }
    pub(super) fn admit_images(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        let mut wave = Encoding {
            requests: Vec::new(),
            unique: Vec::new(),
            outputs: Vec::new(),
            missing: Vec::new(),
            flight: Vec::new(),
            job: None,
        };
        let mut out = Vec::new();
        for (slot, chunks) in items {
            let result = (|| {
                if slot >= self.slots.len()
                    || self.pending.iter().any(|p| p.slot == slot)
                    || self.image_slot_pending(slot)
                    || wave.requests.iter().any(|r| r.0 == slot)
                {
                    return Err(error("invalid or already-admitted slot"));
                }
                let layout = self.layout(chunks)?;
                for k in &layout.keys {
                    if !wave.unique.iter().any(|u| k.same(u)) {
                        wave.unique.push(k.clone());
                    }
                }
                wave.requests.push((slot, layout));
                Ok(())
            })();
            out.push((
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
        out
    }
    fn encode_wave(&mut self, w: &mut Encoding) -> Result<bool> {
        let started = Instant::now();
        // Work-conserving block-level yields. Not a hardware ceiling: the
        // complete TTFT/stream-gap ladder qualifies this budget, not the timer.
        let quantum = Duration::from_millis(64).min(
            Duration::from_millis(96)
                .saturating_sub(Duration::from_secs_f64(self.last_gpu_seconds)),
        );
        if w.outputs.is_empty() {
            for (i, key) in w.unique.iter().enumerate() {
                if let Some(at) = self.image_cache.iter().position(|c| key.same(&c.key)) {
                    let cached = self.image_cache.remove(at).expect("cache hit");
                    w.outputs
                        .push(Some((cached.id, Arc::clone(&cached.features))));
                    self.image_cache.push_back(cached);
                    self.image_cache_reused += 1;
                } else {
                    w.outputs.push(None);
                    w.missing.push(i);
                }
            }
        }
        if w.job.is_none() && !w.missing.is_empty() {
            let v = self.vision.as_ref().expect("attached");
            let mut tiles = 0;
            let mut count = 0;
            for &i in &w.missing {
                let k = &w.unique[i];
                let n = v.plan(k.w, k.h)?.n_tiles();
                if tiles + n > vision::MAX_TILES {
                    break;
                }
                tiles += n;
                count += 1;
            }
            w.flight = w.missing.drain(..count).collect();
            let images = w
                .flight
                .iter()
                .map(|&i| {
                    let k = &w.unique[i];
                    (&*k.rgb, k.w, k.h)
                })
                .collect::<Vec<_>>();
            w.job = Some(v.start(&self.device, &images)?);
            if started.elapsed() >= quantum {
                return Ok(false);
            }
        }
        if let Some(job) = &mut w.job {
            let outputs = loop {
                if let Some(o) = self
                    .vision
                    .as_ref()
                    .expect("attached")
                    .step(&self.device, job)?
                {
                    break o;
                }
                if started.elapsed().saturating_add(job.cost) >= quantum {
                    return Ok(false);
                }
            };
            tracing::info!(
                images = outputs.len(),
                tiles_gpu_ms = job.gpu_seconds * 1000.,
                "Granite Metal vision wave encoded"
            );
            for (&i, features) in w.flight.iter().zip(outputs) {
                if features.tokens != w.unique[i].tokens || features.streams.len() != 8 {
                    return Err(error("projected row/stream count drift"));
                }
                let id = self.next_image_symbol;
                self.next_image_symbol = id
                    .checked_add(1)
                    .ok_or_else(|| error("image identity space exhausted; restart runner"))?;
                let size = features.bytes() + w.unique[i].rgb.len();
                while !self.image_cache.is_empty()
                    && (self.image_cache.len() >= 16
                        || self
                            .image_cache
                            .iter()
                            .map(|c| c.features.bytes() + c.key.rgb.len())
                            .sum::<usize>()
                            .saturating_add(size)
                            > 512 << 20)
                {
                    self.image_cache.pop_front();
                }
                if size <= 512 << 20 {
                    self.image_cache.push_back(Cached {
                        key: w.unique[i].clone(),
                        id,
                        features: Arc::clone(&features),
                    });
                }
                w.outputs[i] = Some((id, features));
            }
            w.job = None;
            w.flight.clear();
            if !w.missing.is_empty() {
                return Ok(false);
            }
        }
        for (_, l) in &mut w.requests {
            for k in &l.keys {
                let i = w
                    .unique
                    .iter()
                    .position(|u| k.same(u))
                    .expect("known image");
                let (id, features) = w.outputs[i].as_ref().expect("encoded");
                l.images.push((*id, Arc::clone(features)));
            }
        }
        Ok(true)
    }
    pub(super) fn step_images(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut w) = self.encoding.pop_front() else {
            return Vec::new();
        };
        match self.encode_wave(&mut w) {
            Ok(false) => {
                self.encoding.push_front(w);
                Vec::new()
            }
            Err(e) => w
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
            Ok(true) => w
                .requests
                .into_iter()
                .map(|(slot, layout)| {
                    let tokens = layout.ids.clone();
                    let result = self.prepare_layout(slot, &tokens, Some(layout));
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
        for w in &mut self.encoding {
            w.requests.retain(|r| r.0 != slot);
        }
        self.encoding.retain(|w| !w.requests.is_empty());
    }
    pub(super) fn prefill_images(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize)> {
        if !self.pending.is_empty() || !self.encoding.is_empty() {
            return Err(error("synchronous images require empty admission queue"));
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
