//! Backend-neutral forced alignment: bounded FIFO coalescing, no artificial
//! batching delay, and cancellation at native stage boundaries. CUDA retains
//! its single-request graph. One thread owns each native GPU model.
use crate::audio::MelFeatures;
use std::sync::{
    Arc,
    mpsc::{Sender, channel},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

pub struct AlignReq {
    pub ids: Vec<u32>,
    pub mel: MelFeatures,
    pub splice_at: usize,
    pub n_audio: usize,
    pub ts_rows: Vec<usize>,
}
#[derive(Clone, Copy)]
pub struct AlignLimits {
    pub batch: usize,
    pub rows: usize,
    pub frames: usize,
}
pub trait AlignBackend {
    fn limits(&self) -> AlignLimits;
    fn validate(&self, req: &AlignReq) -> Result<(), String>;
    /// Preserve input order, including canceled entries in a packed wave.
    fn run_batch(
        &mut self,
        reqs: &[&AlignReq],
        canceled: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<Result<Vec<u32>, String>>, String>;
}
#[cfg(feature = "cuda")]
impl AlignBackend for crate::gpu_model::qwen3_asr::GpuQwen3Asr {
    fn limits(&self) -> AlignLimits {
        AlignLimits {
            batch: 1,
            rows: usize::MAX,
            frames: usize::MAX,
        }
    }
    fn validate(&self, r: &AlignReq) -> Result<(), String> {
        if r.ids.is_empty() || r.ts_rows.is_empty() || r.ts_rows.iter().any(|&i| i >= r.ids.len()) {
            return Err("invalid alignment token rows".into());
        }
        Ok(())
    }
    fn run_batch(
        &mut self,
        reqs: &[&AlignReq],
        canceled: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<Result<Vec<u32>, String>>, String> {
        Ok(reqs
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if canceled(i) {
                    return Err("alignment canceled".into());
                }
                self.align_bins(&r.ids, &r.mel, r.splice_at, r.n_audio, &r.ts_rows)
                    .map_err(|e| e.to_string())
            })
            .collect())
    }
}
struct Job {
    req: AlignReq,
    reply: oneshot::Sender<Result<Vec<u32>, String>>,
    // Retain admission until GPU work releases, even after HTTP cancellation.
    _permit: OwnedSemaphorePermit,
}
#[derive(Clone)]
pub struct Aligner {
    tx: Sender<Job>,
    admission: Arc<Semaphore>,
}
impl Aligner {
    pub fn spawn<F, B>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<B, String> + Send + 'static,
        B: AlignBackend + 'static,
    {
        let (tx, rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        std::thread::Builder::new()
            .name("paddock-aligner".into())
            .spawn(move || {
                let mut model = match build() {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let limits = model.limits();
                if limits.batch == 0 || limits.rows == 0 || limits.frames == 0 {
                    let _ = ready_tx.send(Err("invalid aligner scheduling limits".into()));
                    return;
                }
                let _ = ready_tx.send(Ok(()));
                let mut pending = None;
                loop {
                    let first = match pending.take().or_else(|| rx.recv().ok()) {
                        Some(j) => j,
                        None => break,
                    };
                    let mut batch = Vec::new();
                    let (mut rows, mut frames) = (0usize, 0usize);
                    let mut next = Some(first);
                    while let Some(j) = next.take() {
                        if !j.reply.is_closed() {
                            let valid = model.validate(&j.req).and_then(|_| {
                                if j.req.ids.len() > limits.rows
                                    || j.req.mel.n_frames.div_ceil(100).saturating_mul(100)
                                        > limits.frames
                                {
                                    Err("alignment exceeds batch capacity".into())
                                } else {
                                    Ok(())
                                }
                            });
                            if let Err(e) = valid {
                                let _ = j.reply.send(Err(e));
                            } else {
                                let f = j.req.mel.n_frames.div_ceil(100).saturating_mul(100);
                                if rows.saturating_add(j.req.ids.len()) > limits.rows
                                    || frames.saturating_add(f) > limits.frames
                                {
                                    pending = Some(j);
                                    break;
                                }
                                rows += j.req.ids.len();
                                frames += f;
                                batch.push(j);
                                if batch.len() == limits.batch {
                                    break;
                                }
                            }
                        }
                        next = rx.try_recv().ok();
                    }
                    if batch.is_empty() {
                        continue;
                    }
                    let reqs = batch.iter().map(|j| &j.req).collect::<Vec<_>>();
                    let result = model.run_batch(&reqs, &|i| batch[i].reply.is_closed());
                    let results = match result {
                        Ok(r) if r.len() == batch.len() => r,
                        Ok(_) => (0..batch.len())
                            .map(|_| Err("aligner returned the wrong batch size".into()))
                            .collect(),
                        Err(e) => (0..batch.len()).map(|_| Err(e.clone())).collect(),
                    };
                    for (j, r) in batch.into_iter().zip(results) {
                        let r = r.and_then(|b| {
                            if b.len() == j.req.ts_rows.len() {
                                Ok(b)
                            } else {
                                Err("aligner returned the wrong timestamp count".into())
                            }
                        });
                        let _ = j.reply.send(r);
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            admission: Arc::new(Semaphore::new(8)),
        })
    }
    /// HTTP reserves before decode/mel preprocessing. Saturation fails
    /// explicitly instead of letting queued audio grow without bound.
    pub fn reserve(&self) -> Result<OwnedSemaphorePermit, String> {
        self.admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| "aligner busy; retry later".into())
    }
    pub async fn align(&self, req: AlignReq) -> Result<Vec<u32>, String> {
        self.align_reserved(req, self.reserve()?).await
    }
    pub async fn align_reserved(
        &self,
        req: AlignReq,
        permit: OwnedSemaphorePermit,
    ) -> Result<Vec<u32>, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job {
                req,
                reply,
                _permit: permit,
            })
            .map_err(|_| "aligner thread gone".to_string())?;
        rx.await
            .map_err(|_| "aligner dropped the request".to_string())?
    }
}
#[cfg(test)]
mod tests;
