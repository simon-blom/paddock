//! Dense-prediction service seam - the `encoder.rs` shape: a dedicated CUDA
//! thread owning the model, oneshot request/response, no decode loop.
//!
//! The workload is batch over an area. A request is "these N chips", N from
//! one to thousands, and nobody is waiting on any single chip - so the unit
//! the thread schedules is the chip, not the request. Every pass packs up to
//! `max_batch` chips in arrival order, across request boundaries: a 3-chip
//! request and the head of a 5000-chip one share a pass, a big request spans
//! many passes, and a request is answered when its last chip lands. That is
//! the whole scheduler. It keeps the GEMMs at the row count they are fast at
//! whatever the callers' request sizes are, which is what continuous batching
//! is for a model with no sequence dimension.
//!
//! Every pass runs at one width. A short pass - the tail of a sweep, a lone
//! chip - is padded to `max_batch` with blank chips whose rasters are thrown
//! away. The reason is reproducibility: a chip's raster is bit-stable for a
//! given pass width, but the tensor-core GEMM chooses its tiling from how full
//! its grid is, so the same chip served in a pass of one and in a pass of
//! sixteen can differ in a handful of near-tie boundary pixels. With
//! coalescing, the width a chip lands in is an accident of what else was
//! queued that millisecond - and a county re-run that disagrees with itself,
//! however slightly, is not something a geodata product gets to shrug at. So
//! the width is a constant of the endpoint, and a raster is a function of the
//! chip and the endpoint's config, nothing else. The cost is that a lone chip
//! pays for a full pass; a full queue pays nothing. An endpoint meant for
//! one-at-a-time use should simply be configured narrow.
//!
//! Passes are synchronous: upload, forward, read back, next. The GPU idles
//! for the host's share of that - a few MB each way against a hundred-plus
//! milliseconds of compute per pass - so the submit/collect overlap
//! `encoder.rs` runs is not built here. If the backbone ever gets fast enough
//! for that gap to show, that is the pattern.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use half::f16;
use paddock_models::dinov3::Dinov3SegConfig;
use tokio::sync::oneshot;

use crate::gpu_model::dinov3::GpuDinov3Seg;

/// One request: `chips * side * side * bands` bytes of u8 HWC, chip-major.
pub struct SegRequest {
    pub pixels: Vec<u8>,
    pub chips: usize,
    /// also return the biased class logits at f16 - 20x the class raster
    pub want_logits: bool,
}

/// One request's rasters, chip-major, in request order.
#[derive(Debug)]
pub struct SegReply {
    pub chips: usize,
    pub size: usize,
    pub n_classes: usize,
    /// `[chips][size][size]`
    pub classes: Vec<u8>,
    /// `[chips][size][size]`, metres, unclamped
    pub height: Vec<f32>,
    /// `[chips][size][size][n_classes]`
    pub logits: Option<Vec<f16>>,
}

/// What the runner needs to describe the endpoint without touching the GPU.
pub struct SegInfo {
    pub config: Dinov3SegConfig,
    pub max_batch: usize,
    pub weight_bytes: u64,
    pub workspace_bytes: u64,
}

struct Job {
    req: SegRequest,
    /// chips already through the model
    done: usize,
    out: SegReply,
    reply: oneshot::Sender<Result<SegReply, String>>,
}

/// Handle to the segmenter thread. Cloneable; chips are served FIFO.
#[derive(Clone)]
pub struct Segmenter {
    tx: Sender<(SegRequest, oneshot::Sender<Result<SegReply, String>>)>,
    info: Arc<SegInfo>,
}

impl Segmenter {
    /// Spawn the segmenter thread. `build` constructs the model on that thread
    /// (the CUDA context binds to it) and may fail; spawn blocks until it has
    /// and propagates the error - same contract as `Encoder::spawn`.
    pub fn spawn<F>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<GpuDinov3Seg, String> + Send + 'static,
    {
        let (tx, rx) = channel();
        let (ready_tx, ready_rx) = channel::<Result<SegInfo, String>>();
        std::thread::Builder::new()
            .name("paddock-segmenter".into())
            .spawn(move || {
                let model = match build() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(SegInfo {
                            config: m.config().clone(),
                            max_batch: m.max_batch(),
                            weight_bytes: m.weight_bytes(),
                            workspace_bytes: m.workspace_bytes(),
                        }));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                serve(model, rx);
            })
            .map_err(|e| e.to_string())?;
        let info = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            info: Arc::new(info),
        })
    }

    pub fn info(&self) -> &SegInfo {
        &self.info
    }

    /// Segment a request's chips; resolves when the last one has landed.
    pub async fn segment(&self, req: SegRequest) -> Result<SegReply, String> {
        let chip =
            self.info.config.image_size * self.info.config.image_size * self.info.config.channels;
        if req.pixels.len() != req.chips * chip {
            return Err(format!(
                "{} bytes for {} chip(s), expected {} each",
                req.pixels.len(),
                req.chips,
                chip
            ));
        }
        let (tx, rx) = oneshot::channel();
        self.tx
            .send((req, tx))
            .map_err(|_| "segmenter thread gone".to_string())?;
        rx.await
            .map_err(|_| "segmenter dropped the request".to_string())?
    }
}

type Incoming = (SegRequest, oneshot::Sender<Result<SegReply, String>>);

fn serve(mut model: GpuDinov3Seg, rx: Receiver<Incoming>) {
    let cap = model.max_batch();
    let chip = model.chip_bytes();
    let size = model.config().out_size;
    let ncls = model.config().n_classes;
    let px = size * size;
    let mut queue: VecDeque<Job> = VecDeque::new();
    let mut stage = Vec::with_capacity(cap * chip);

    let admit = |queue: &mut VecDeque<Job>, (req, reply): Incoming| {
        let n = req.chips;
        let out = SegReply {
            chips: n,
            size,
            n_classes: ncls,
            classes: Vec::with_capacity(n * px),
            height: Vec::with_capacity(n * px),
            logits: req.want_logits.then(|| Vec::with_capacity(n * px * ncls)),
        };
        if n == 0 {
            let _ = reply.send(Ok(out));
        } else {
            queue.push_back(Job {
                req,
                done: 0,
                out,
                reply,
            });
        }
    };

    loop {
        // block only when there is nothing to run; otherwise just sweep in
        // whatever arrived while the last pass was on the GPU
        if queue.is_empty() {
            match rx.recv() {
                Ok(j) => admit(&mut queue, j),
                Err(_) => return,
            }
        }
        loop {
            match rx.try_recv() {
                Ok(j) => admit(&mut queue, j),
                Err(TryRecvError::Empty) => break,
                // senders gone: finish what is queued, then the recv above exits
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // a caller that went away takes its unserved chips with it - a county
        // sweep cancelled at the client must not keep the GPU for an hour
        queue.retain(|j| !j.reply.is_closed());
        if queue.is_empty() {
            continue;
        }

        // ---- pack one pass, FIFO across request boundaries ----
        stage.clear();
        let mut take: Vec<usize> = Vec::new(); // chips taken from queue[i]
        let mut packed = 0usize;
        let mut logits = false;
        for job in queue.iter() {
            if packed == cap {
                break;
            }
            let n = (job.req.chips - job.done).min(cap - packed);
            stage.extend_from_slice(&job.req.pixels[job.done * chip..(job.done + n) * chip]);
            logits |= job.req.want_logits;
            take.push(n);
            packed += n;
        }

        // blank chips up to the endpoint's one pass width (see the module note)
        stage.resize(cap * chip, 0);

        match model.segment(&stage, cap, logits) {
            Ok(out) => {
                let mut at = 0usize;
                for (job, &n) in queue.iter_mut().zip(&take) {
                    job.out
                        .classes
                        .extend_from_slice(&out.classes[at * px..(at + n) * px]);
                    job.out
                        .height
                        .extend_from_slice(&out.height[at * px..(at + n) * px]);
                    if let (Some(dst), Some(src)) = (job.out.logits.as_mut(), out.logits.as_ref()) {
                        dst.extend_from_slice(&src[at * px * ncls..(at + n) * px * ncls]);
                    }
                    job.done += n;
                    at += n;
                }
                while queue.front().is_some_and(|j| j.done == j.req.chips) {
                    let j = queue.pop_front().expect("front checked");
                    let _ = j.reply.send(Ok(j.out));
                }
            }
            Err(e) => {
                // A pass fails as a unit, so every request with a chip in it
                // fails - and says which pass, not just that one did. Requests
                // behind it were not touched and keep their place.
                let msg = e.to_string();
                for _ in 0..take.len() {
                    if let Some(j) = queue.pop_front() {
                        let _ = j.reply.send(Err(msg.clone()));
                    }
                }
            }
        }
    }
}
