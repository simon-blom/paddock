//! Decision-model service seam (Laya) - the `segment.rs` shape: a dedicated
//! GPU thread owning every loaded checkpoint and one shared workspace,
//! oneshot request/response, no decode loop.
//!
//! The workload is many short sequences. A request is a handful of
//! questions, each one sequence of 15 to 1024 tokens, and a busy endpoint has
//! many requests in flight - so the unit the thread schedules is the
//! SEQUENCE, not the request. Every pass takes the checkpoint of the oldest
//! waiting request and packs whole sequences of that checkpoint in arrival
//! order, across request boundaries, until the next one would overflow the
//! workspace (tokens, sequences). A request is answered when its last
//! sequence lands. That is continuous batching for a model with no decode
//! step: the GEMMs run at the row count the queue offers, not at one
//! request's.
//!
//! Unlike the segmenter, no pass is padded to a fixed width: the text
//! encoder is batch-invariant by construction (the f16-landing GEMMs never
//! split K, the attention's work is laid per sequence, everything else is
//! row- or question-local), so a question's probabilities are a function of
//! the question and the checkpoint, never of what else rode its pass. The
//! golden test holds that bit-exact.
//!
//! Passes are synchronous: upload the index planes, forward, read back a few
//! kilobytes of logits. The host share is microseconds against milliseconds
//! of GPU work, so there is no submit/collect overlap to build.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, SyncSender, channel, sync_channel};
use std::time::Instant;

use paddock_models::laya::{Checkpoint, LayaConfig};
use tokio::sync::oneshot;

#[cfg(feature = "cuda")]
use crate::gpu_model::laya::{GpuLaya, LayaWorkspace};

/// Borrowed packed input, shared by native CUDA and Metal implementations.
pub struct LayaSeq<'a> {
    pub ids: &'a [u32],
    pub markers: &'a [u32],
    pub qtype: u32,
}

#[derive(Debug, Clone)]
pub struct LayaOut {
    pub logits: Vec<f32>,
    pub offsets: Vec<usize>,
    pub act: Vec<f32>,
    pub n_act: usize,
    pub rows: usize,
}

/// Constructed and owned on the decision thread. Scheduling, cancellation,
/// calibration and language routing do not depend on the GPU API.
pub trait DecisionBackend {
    fn info(&self) -> DecisionInfo;
    fn forward(&mut self, checkpoint: Checkpoint, seqs: &[LayaSeq<'_>]) -> Result<LayaOut, String>;
}

#[cfg(feature = "cuda")]
struct CudaDecision {
    models: Vec<(Checkpoint, GpuLaya)>,
    ws: LayaWorkspace,
}

#[cfg(feature = "cuda")]
impl DecisionBackend for CudaDecision {
    fn info(&self) -> DecisionInfo {
        DecisionInfo {
            checkpoints: self
                .models
                .iter()
                .map(|(c, m)| (*c, m.config().clone()))
                .collect(),
            rows_cap: self.ws.rows_cap(),
            seq_cap: self.ws.seq_cap(),
            weight_bytes: self.models.iter().map(|(_, m)| m.weight_bytes()).sum(),
            workspace_bytes: self.ws.bytes(),
        }
    }

    fn forward(&mut self, checkpoint: Checkpoint, seqs: &[LayaSeq<'_>]) -> Result<LayaOut, String> {
        let model = self
            .models
            .iter()
            .find(|(c, _)| *c == checkpoint)
            .ok_or_else(|| format!("the {} checkpoint is not loaded", checkpoint.name()))?;
        model
            .1
            .forward(&mut self.ws, seqs)
            .map_err(|e| e.to_string())
    }
}

/// One question's sequence, as the runner built it.
#[derive(Debug, Clone)]
pub struct DecisionSeq {
    pub ids: Vec<u32>,
    /// each option's `[MASK]` position inside the sequence
    pub markers: Vec<u32>,
    /// `paddock_models::laya::QTYPE_*`
    pub qtype: u32,
}

pub struct DecisionRequest {
    pub checkpoint: Checkpoint,
    pub seqs: Vec<DecisionSeq>,
}

/// One request's outputs, in its sequence order.
#[derive(Debug, Clone, Default)]
pub struct DecisionReply {
    /// per sequence: its option logits, raw (the runner applies temperature)
    pub logits: Vec<Vec<f32>>,
    /// per sequence: the act head's probabilities
    pub act: Vec<Vec<f32>>,
    /// tokens this request put through the encoder
    pub tokens: usize,
    /// passes this request's sequences rode, and their summed GPU wall
    pub passes: usize,
    pub gpu_ms: f64,
}

/// What the runner needs to describe the endpoint without touching the GPU.
pub struct DecisionInfo {
    pub checkpoints: Vec<(Checkpoint, LayaConfig)>,
    pub rows_cap: usize,
    pub seq_cap: usize,
    pub weight_bytes: u64,
    pub workspace_bytes: u64,
}

impl DecisionInfo {
    pub fn config(&self, c: Checkpoint) -> Option<&LayaConfig> {
        self.checkpoints
            .iter()
            .find(|(k, _)| *k == c)
            .map(|(_, v)| v)
    }
}

struct Job {
    req: DecisionRequest,
    /// sequences already through the model
    done: usize,
    out: DecisionReply,
    reply: oneshot::Sender<Result<DecisionReply, String>>,
}

type Incoming = (
    DecisionRequest,
    oneshot::Sender<Result<DecisionReply, String>>,
);
const QUEUE_CAP: usize = 128;

/// Handle to the decision thread. Cloneable; sequences are served FIFO.
#[derive(Clone)]
pub struct Decider {
    tx: SyncSender<Incoming>,
    info: Arc<DecisionInfo>,
    metrics: Arc<crate::metrics::EngineMetrics>,
}

impl Decider {
    /// Spawn the decision thread. `build` constructs the checkpoints and the
    /// workspace on that thread (the CUDA context binds to it) and may fail;
    /// spawn blocks until it has and propagates the error.
    #[cfg(feature = "cuda")]
    pub fn spawn<F>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<(Vec<(Checkpoint, GpuLaya)>, LayaWorkspace), String> + Send + 'static,
    {
        Self::spawn_backend(move || {
            let (models, ws) = build()?;
            Ok(CudaDecision { models, ws })
        })
    }

    pub fn spawn_backend<F, B>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<B, String> + Send + 'static,
        B: DecisionBackend + 'static,
    {
        let (tx, rx) = sync_channel(QUEUE_CAP);
        let metrics = Arc::new(crate::metrics::EngineMetrics::default());
        let worker_metrics = Arc::clone(&metrics);
        let (ready_tx, ready_rx) = channel::<Result<DecisionInfo, String>>();
        std::thread::Builder::new()
            .name("paddock-decision".into())
            .spawn(move || {
                let backend = match build() {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let info = backend.info();
                if info.rows_cap == 0 || info.seq_cap == 0 {
                    let _ = ready_tx.send(Err("decision backend has an empty workspace".into()));
                    return;
                }
                worker_metrics
                    .weights_mem_bytes
                    .store(info.weight_bytes, Relaxed);
                worker_metrics
                    .model_mem_bytes
                    .store(info.weight_bytes + info.workspace_bytes, Relaxed);
                let _ = ready_tx.send(Ok(info));
                serve(backend, rx, worker_metrics);
            })
            .map_err(|e| e.to_string())?;
        let info = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            info: Arc::new(info),
            metrics,
        })
    }

    pub fn info(&self) -> &DecisionInfo {
        &self.info
    }
    pub fn metrics(&self) -> Arc<crate::metrics::EngineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Run a request's sequences; resolves when the last one has landed.
    pub async fn decide(&self, req: DecisionRequest) -> Result<DecisionReply, String> {
        let Some(cfg) = self.info.config(req.checkpoint) else {
            return Err(format!(
                "the {} checkpoint is not loaded",
                req.checkpoint.name()
            ));
        };
        if let Some(s) = req.seqs.iter().find(|s| {
            s.ids.is_empty()
                || s.ids.len() > cfg.max_len
                || s.ids.len() > self.info.rows_cap
                || s.markers.is_empty()
                || s.markers.len() + 1 > self.info.rows_cap / 2 + self.info.seq_cap
                || s.qtype > 2
                || s.ids.iter().any(|&id| id as usize >= cfg.encoder.vocab)
                || s.markers.iter().any(|&m| m as usize >= s.ids.len())
        }) {
            return Err(format!(
                "a sequence of {} tokens with {} options (the checkpoint takes 1..={} tokens \
                 and at least one option)",
                s.ids.len(),
                s.markers.len(),
                cfg.max_len
            ));
        }
        let (tx, rx) = oneshot::channel();
        self.tx.try_send((req, tx)).map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(_) => {
                "decision queue is full; retry shortly".to_string()
            }
            std::sync::mpsc::TrySendError::Disconnected(_) => "decision thread gone".to_string(),
        })?;
        rx.await
            .map_err(|_| "decision thread dropped the request".to_string())?
    }
}

fn serve(
    mut backend: impl DecisionBackend,
    rx: Receiver<Incoming>,
    metrics: Arc<crate::metrics::EngineMetrics>,
) {
    let info = backend.info();
    let (rows_cap, seq_cap) = (info.rows_cap, info.seq_cap);
    // the head's last layer continues on markers + [CLS] rows; bound them the
    // way the workspace was sized (see LayaWorkspace::new)
    let gather_cap = rows_cap / 2 + seq_cap;
    let mut queue: VecDeque<Job> = VecDeque::new();

    let admit = |queue: &mut VecDeque<Job>, (req, reply): Incoming| {
        let n = req.seqs.len();
        let out = DecisionReply {
            logits: Vec::with_capacity(n),
            act: Vec::with_capacity(n),
            ..Default::default()
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
        if queue.is_empty() {
            match rx.recv() {
                Ok(j) => admit(&mut queue, j),
                Err(_) => return,
            }
        }
        // sweep in whatever arrived while the last pass was on the GPU; a
        // disconnect finishes what is queued, then the recv above exits
        while queue.len() < QUEUE_CAP {
            match rx.try_recv() {
                Ok(j) => admit(&mut queue, j),
                Err(_) => break,
            }
        }
        // a caller that went away takes its unserved sequences with it
        queue.retain(|j| !j.reply.is_closed());
        let Some(front) = queue.front() else {
            continue;
        };
        let ck = front.req.checkpoint;
        if info.config(ck).is_none() {
            let j = queue.pop_front().expect("front checked");
            let _ = j
                .reply
                .send(Err(format!("the {} checkpoint is not loaded", ck.name())));
            continue;
        }

        // ---- pack one pass: this checkpoint's sequences, FIFO ----
        // take[i] = (job index, sequences taken from it)
        let mut take: Vec<(usize, usize)> = Vec::new();
        let (mut rows, mut nseq, mut gat) = (0usize, 0usize, 0usize);
        'pack: for (ji, job) in queue.iter().enumerate() {
            if job.req.checkpoint != ck {
                continue;
            }
            let mut n = 0usize;
            for s in &job.req.seqs[job.done..] {
                let (r, gg) = (s.ids.len(), s.markers.len() + 1);
                if nseq > 0 && (rows + r > rows_cap || nseq + 1 > seq_cap || gat + gg > gather_cap)
                {
                    if n > 0 {
                        take.push((ji, n));
                    }
                    break 'pack;
                }
                rows += r;
                nseq += 1;
                gat += gg;
                n += 1;
            }
            if n > 0 {
                take.push((ji, n));
            }
        }
        let seqs: Vec<LayaSeq> = take
            .iter()
            .flat_map(|&(ji, n)| {
                let job = &queue[ji];
                job.req.seqs[job.done..job.done + n]
                    .iter()
                    .map(|s| LayaSeq {
                        ids: &s.ids,
                        markers: &s.markers,
                        qtype: s.qtype,
                    })
            })
            .collect();
        let t0 = Instant::now();
        metrics.active_slots.store(nseq as u32, Relaxed);
        metrics.phase.store(crate::metrics::PHASE_PREFILL, Relaxed);
        let result = backend.forward(ck, &seqs);
        metrics.phase.store(crate::metrics::PHASE_IDLE, Relaxed);
        metrics.active_slots.store(0, Relaxed);
        if result.is_ok() {
            metrics.prefill_tokens_total.fetch_add(rows as u64, Relaxed);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(seqs);
        match result {
            Ok(out) => {
                let mut s = 0usize;
                for &(ji, n) in &take {
                    let job = &mut queue[ji];
                    for k in 0..n {
                        let q = s + k;
                        job.out
                            .logits
                            .push(out.logits[out.offsets[q]..out.offsets[q + 1]].to_vec());
                        job.out
                            .act
                            .push(out.act[q * out.n_act..(q + 1) * out.n_act].to_vec());
                        job.out.tokens += job.req.seqs[job.done + k].ids.len();
                    }
                    job.done += n;
                    job.out.passes += 1;
                    job.out.gpu_ms += ms;
                    s += n;
                }
            }
            Err(e) => {
                // a pass fails as a unit: every request with a sequence in it
                // fails, and says so; the rest keep their place
                let msg = e.to_string();
                for &(ji, _) in take.iter().rev() {
                    if let Some(j) = queue.remove(ji) {
                        let _ = j.reply.send(Err(msg.clone()));
                    }
                }
                continue;
            }
        }
        // answer every finished request, wherever it sits in the queue
        let mut i = 0usize;
        while i < queue.len() {
            if queue[i].done == queue[i].req.seqs.len() {
                let j = queue.remove(i).expect("index checked");
                let _ = j.reply.send(Ok(j.out));
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_models::laya::ModernBertConfig;
    struct Fake;
    fn config() -> LayaConfig {
        LayaConfig {
            dir: Default::default(),
            name: "test".into(),
            head_layers: 2,
            n_act: 2,
            max_len: 4,
            head_max_len: 2,
            temperature: [1.0; 3],
            temperature_by_options: vec![],
            encoder: ModernBertConfig {
                hidden: 64,
                n_layer: 1,
                n_heads: 1,
                intermediate: 64,
                vocab: 16,
                global: vec![true],
                window: 1,
                rope_theta_global: 160000.0,
                rope_theta_local: 10000.0,
                eps: 1e-5,
                max_position: 4,
            },
        }
    }
    impl DecisionBackend for Fake {
        fn info(&self) -> DecisionInfo {
            DecisionInfo {
                checkpoints: vec![
                    (Checkpoint::English, config()),
                    (Checkpoint::Multilingual, config()),
                ],
                rows_cap: 4,
                seq_cap: 2,
                weight_bytes: 100,
                workspace_bytes: 20,
            }
        }
        fn forward(&mut self, ck: Checkpoint, seqs: &[LayaSeq<'_>]) -> Result<LayaOut, String> {
            assert!(seqs.len() <= 2 && seqs.iter().map(|s| s.ids.len()).sum::<usize>() <= 4);
            let mut out = LayaOut {
                logits: vec![],
                offsets: vec![0],
                act: vec![],
                n_act: 2,
                rows: 0,
            };
            for s in seqs {
                out.logits
                    .extend(s.markers.iter().map(|&m| s.ids[m as usize] as f32));
                out.offsets.push(out.logits.len());
                out.rows += s.ids.len();
                out.act.extend(if ck == Checkpoint::English {
                    [1.0, 0.0]
                } else {
                    [0.0, 1.0]
                });
            }
            Ok(out)
        }
    }
    fn seq() -> DecisionSeq {
        DecisionSeq {
            ids: vec![1, 2],
            markers: vec![1],
            qtype: 0,
        }
    }
    #[tokio::test]
    async fn decision_service_splits_requests_and_reports_actual_memory_without_cuda() {
        let d = Decider::spawn_backend(|| Ok(Fake)).unwrap();
        let (a, b) = tokio::join!(
            d.decide(DecisionRequest {
                checkpoint: Checkpoint::English,
                seqs: vec![seq(); 5]
            }),
            d.decide(DecisionRequest {
                checkpoint: Checkpoint::Multilingual,
                seqs: vec![seq(); 2]
            })
        );
        let a = a.unwrap();
        let b = b.unwrap();
        assert_eq!(a.logits, vec![vec![2.0]; 5]);
        assert_eq!(a.tokens, 10);
        assert_eq!(a.passes, 3);
        assert_eq!(b.act, vec![vec![0.0, 1.0]; 2]);
        assert_eq!(d.metrics().model_mem_bytes.load(Relaxed), 120);
        assert_eq!(d.metrics().prefill_tokens_total.load(Relaxed), 14);
        assert_eq!(d.metrics().active_slots.load(Relaxed), 0);
    }
    #[tokio::test]
    async fn decision_service_rejects_invalid_gpu_indices_and_handles_empty_requests() {
        let d = Decider::spawn_backend(|| Ok(Fake)).unwrap();
        for s in [
            DecisionSeq {
                ids: vec![],
                ..seq()
            },
            DecisionSeq {
                ids: vec![16, 1],
                ..seq()
            },
            DecisionSeq {
                markers: vec![2],
                ..seq()
            },
            DecisionSeq { qtype: 3, ..seq() },
            DecisionSeq {
                markers: vec![0; 5],
                ..seq()
            },
            DecisionSeq {
                ids: vec![1; 5],
                ..seq()
            },
        ] {
            assert!(
                d.decide(DecisionRequest {
                    checkpoint: Checkpoint::English,
                    seqs: vec![s]
                })
                .await
                .is_err()
            );
        }
        assert!(
            d.decide(DecisionRequest {
                checkpoint: Checkpoint::TypedDecisions,
                seqs: vec![seq()]
            })
            .await
            .is_err()
        );
        let out = d
            .decide(DecisionRequest {
                checkpoint: Checkpoint::English,
                seqs: vec![],
            })
            .await
            .unwrap();
        assert!(out.logits.is_empty());
        assert_eq!(out.passes, 0);
    }
    #[test]
    fn cancelled_requests_are_never_run() {
        let (tx, rx) = sync_channel(2);
        let (reply, wait) = oneshot::channel();
        tx.send((
            DecisionRequest {
                checkpoint: Checkpoint::English,
                seqs: vec![seq()],
            },
            reply,
        ))
        .unwrap();
        drop(wait);
        drop(tx);
        let metrics = Arc::new(crate::metrics::EngineMetrics::default());
        serve(Fake, rx, metrics.clone());
        assert_eq!(metrics.prefill_tokens_total.load(Relaxed), 0);
    }
    #[tokio::test]
    async fn queue_overload_never_blocks_an_async_executor() {
        let (tx, _rx) = sync_channel(1);
        let (reply, _wait) = oneshot::channel();
        tx.send((
            DecisionRequest {
                checkpoint: Checkpoint::English,
                seqs: vec![seq()],
            },
            reply,
        ))
        .unwrap();
        let d = Decider {
            tx,
            info: Arc::new(Fake.info()),
            metrics: Default::default(),
        };
        let out = d
            .decide(DecisionRequest {
                checkpoint: Checkpoint::English,
                seqs: vec![seq()],
            })
            .await;
        assert!(out.unwrap_err().starts_with("decision queue is full"));
    }
}
