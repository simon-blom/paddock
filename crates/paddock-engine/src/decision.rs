//! Decision-model service seam (Laya) - the `segment.rs` shape: a dedicated
//! CUDA thread owning every loaded checkpoint and one shared workspace,
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
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use paddock_models::laya::{Checkpoint, LayaConfig};
use tokio::sync::oneshot;

use crate::gpu_model::laya::{GpuLaya, LayaSeq, LayaWorkspace};

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

/// Handle to the decision thread. Cloneable; sequences are served FIFO.
#[derive(Clone)]
pub struct Decider {
    tx: Sender<Incoming>,
    info: Arc<DecisionInfo>,
}

impl Decider {
    /// Spawn the decision thread. `build` constructs the checkpoints and the
    /// workspace on that thread (the CUDA context binds to it) and may fail;
    /// spawn blocks until it has and propagates the error.
    pub fn spawn<F>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<(Vec<(Checkpoint, GpuLaya)>, LayaWorkspace), String> + Send + 'static,
    {
        let (tx, rx) = channel();
        let (ready_tx, ready_rx) = channel::<Result<DecisionInfo, String>>();
        std::thread::Builder::new()
            .name("paddock-decision".into())
            .spawn(move || {
                let (models, ws) = match build() {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(DecisionInfo {
                    checkpoints: models
                        .iter()
                        .map(|(c, m)| (*c, m.config().clone()))
                        .collect(),
                    rows_cap: ws.rows_cap(),
                    seq_cap: ws.seq_cap(),
                    weight_bytes: models.iter().map(|(_, m)| m.weight_bytes()).sum(),
                    workspace_bytes: ws.bytes(),
                }));
                serve(models, ws, rx);
            })
            .map_err(|e| e.to_string())?;
        let info = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            info: Arc::new(info),
        })
    }

    pub fn info(&self) -> &DecisionInfo {
        &self.info
    }

    /// Run a request's sequences; resolves when the last one has landed.
    pub async fn decide(&self, req: DecisionRequest) -> Result<DecisionReply, String> {
        let Some(cfg) = self.info.config(req.checkpoint) else {
            return Err(format!(
                "the {} checkpoint is not loaded",
                req.checkpoint.name()
            ));
        };
        if let Some(s) = req
            .seqs
            .iter()
            .find(|s| s.ids.is_empty() || s.ids.len() > cfg.max_len || s.markers.is_empty())
        {
            return Err(format!(
                "a sequence of {} tokens with {} options (the checkpoint takes 1..={} tokens \
                 and at least one option)",
                s.ids.len(),
                s.markers.len(),
                cfg.max_len
            ));
        }
        let (tx, rx) = oneshot::channel();
        self.tx
            .send((req, tx))
            .map_err(|_| "decision thread gone".to_string())?;
        rx.await
            .map_err(|_| "decision thread dropped the request".to_string())?
    }
}

fn serve(models: Vec<(Checkpoint, GpuLaya)>, mut ws: LayaWorkspace, rx: Receiver<Incoming>) {
    let (rows_cap, seq_cap) = (ws.rows_cap(), ws.seq_cap());
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
        while let Ok(j) = rx.try_recv() {
            admit(&mut queue, j);
        }
        // a caller that went away takes its unserved sequences with it
        queue.retain(|j| !j.reply.is_closed());
        let Some(front) = queue.front() else {
            continue;
        };
        let ck = front.req.checkpoint;
        let Some(model) = models.iter().find(|(c, _)| *c == ck).map(|(_, m)| m) else {
            let j = queue.pop_front().expect("front checked");
            let _ = j
                .reply
                .send(Err(format!("the {} checkpoint is not loaded", ck.name())));
            continue;
        };

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
        let result = model.forward(&mut ws, &seqs);
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
