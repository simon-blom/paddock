//! Rank-0-authoritative TP=2 serving over the distributed control channel.
//! The production scheduler owns rank 0's dense slot plan; rank 1 replays the
//! ordered live rows and mirrored KV operations. Each row uses eager kernels.
use std::{io::Read, net::TcpStream, path::Path, sync::Arc, time::Duration};

use cudarc::driver::CudaEvent;
use paddock_dist::{
    config::Resolved,
    protocol::{ControlMessage, receive_nccl_id, send_nccl_id},
    worker::shutdown_worker,
};
use paddock_models::mapped::MappedGguf;

use super::{
    tp_kv::{Event, MirroredKv, Operation},
    tp_model::Qwen35TpRank,
};
use crate::{
    generator::{GenError, Generator, RowSample, SampledStep},
    gpu::{
        GpuExecutor, KvDtype,
        distributed::{NcclCommunicator, create_unique_id},
    },
};

const PINNED_SHA256: &str = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482";
const READY_TIMEOUT: Duration = Duration::from_secs(600);
const STEP_TIMEOUT: Duration = Duration::from_secs(120);

fn hashes(model: &Path, pack: &Path) -> Result<(String, String), String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(model).map_err(|e| format!("model open: {e}"))?;
    let mut sha = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("model read: {e}"))?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    let checkpoint: String = sha.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if checkpoint != PINNED_SHA256 {
        return Err(format!(
            "Phase 9 requires pinned Qwen3.8 GGUF SHA-256 {PINNED_SHA256}, got {checkpoint}"
        ));
    }
    let mut file = std::fs::File::open(pack).map_err(|e| format!("CUDA pack open: {e}"))?;
    let mut hasher = blake3::Hasher::new();
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("CUDA pack read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok((checkpoint, hasher.finalize().to_hex().to_string()))
}

fn ready(stream: &mut TcpStream, sequence: u64) -> Result<(), String> {
    match ControlMessage::from_stream(stream).map_err(|e| e.to_string())? {
        ControlMessage::TpReady { sequence: got } if got == sequence => Ok(()),
        ControlMessage::TpError { reason } | ControlMessage::Reject { reason } => Err(reason),
        other => Err(format!(
            "rank 1 reply out of order (expected {sequence}): {other:?}"
        )),
    }
}

fn prepared(stream: &mut TcpStream, sequence: u64) -> Result<(), String> {
    match ControlMessage::from_stream(stream).map_err(|e| e.to_string())? {
        ControlMessage::TpPrepared { sequence: got } if got == sequence => Ok(()),
        ControlMessage::TpError { reason } | ControlMessage::Reject { reason } => Err(reason),
        other => Err(format!("rank 1 did not prepare step {sequence}: {other:?}")),
    }
}

// The scheduler sends dense rows through its high-water slot; position 0 is
// a hole, including a newly admitted slot that has not finished prefill.
fn active_rows(
    tokens: &[u32],
    positions: &[u32],
    slots: usize,
) -> Result<Vec<(usize, u32, usize)>, String> {
    if tokens.len() != positions.len() || tokens.is_empty() || tokens.len() > slots {
        return Err("TP batch width mismatch".into());
    }
    Ok(positions
        .iter()
        .enumerate()
        .filter_map(|(slot, &position)| {
            (position != 0).then_some((slot, tokens[slot], position as usize))
        })
        .collect())
}

fn validate_sampled_rows(
    rows: &[(usize, u32, usize)],
    plans: &[RowSample],
    width: usize,
) -> Result<(), String> {
    if plans.len() != width || rows.iter().any(|&(slot, _, _)| slot >= width) {
        return Err("TP sampling plan width mismatch".into());
    }
    let mut active = vec![false; width];
    for &(slot, _, _) in rows {
        active[slot] = true;
    }
    for (slot, plan) in plans.iter().enumerate() {
        match (active[slot], plan) {
            (false, RowSample::Hole) | (true, RowSample::Host) => {}
            (true, RowSample::Device(plan)) => {
                super::tp_model::tp_sample_params(*plan).map_err(|e| e.to_string())?;
            }
            _ => return Err("TP sampling plan disagrees with live slots".into()),
        }
    }
    Ok(())
}

fn logical(max_ctx: usize, slots: usize) -> Result<MirroredKv, String> {
    let blocks = u32::try_from(
        max_ctx
            .div_ceil(crate::kv_pool::BLOCK_TOKENS)
            .checked_mul(slots)
            .ok_or("KV block count overflow")?,
    )
    .map_err(|_| "KV block count overflow")?;
    MirroredKv::new(blocks, slots, max_ctx).map_err(str::to_owned)
}

struct PipeFlight {
    slots: Vec<usize>,
    width: usize,
    plane: usize,
    events: Vec<(usize, CudaEvent)>,
}

/// Runs on the production engine thread; owns the control stream until drop.
pub struct TpCoordinator {
    stream: TcpStream,
    group: NcclCommunicator,
    model: Qwen35TpRank,
    logical: MirroredKv,
    positions: Vec<usize>,
    occupied: Vec<bool>,
    sequence: u64,
    pipe: Option<PipeFlight>,
    poisoned: Option<String>,
}

impl TpCoordinator {
    pub fn load(
        mut stream: TcpStream,
        resolved: &Resolved,
        model: &Path,
        pack: &Path,
        gpu: usize,
        max_ctx: usize,
        slots: usize,
    ) -> Result<Self, String> {
        if !resolved.is_coordinator() || max_ctx == 0 || !(1..=2).contains(&slots) {
            return Err("Phase 9 requires TP=2 rank 0 and nonzero context".into());
        }
        stream
            .set_read_timeout(Some(READY_TIMEOUT))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(STEP_TIMEOUT))
            .map_err(|e| e.to_string())?;
        let (checkpoint_sha256, pack_blake3) = hashes(model, pack)?;
        ControlMessage::TpInit {
            checkpoint_sha256,
            pack_blake3,
            max_ctx,
            slots,
        }
        .to_stream(&mut stream)
        .map_err(|e| e.to_string())?;
        // Rank 1 checks identity before NCCL is initialized.
        ready(&mut stream, 0)?;
        let id = create_unique_id().map_err(|e| e.to_string())?;
        send_nccl_id(&mut stream, &id).map_err(|e| e.to_string())?;
        let exec = Arc::new(GpuExecutor::new(gpu, pack).map_err(|e| e.to_string())?);
        let group = NcclCommunicator::from_resolved(Some(resolved), exec.stream.context(), id)
            .map_err(|e| e.to_string())?
            .ok_or("TP group absent")?;
        let map = MappedGguf::open(model).map_err(|e| e.to_string())?;
        let model = Qwen35TpRank::load_slots(exec, &map, &group, max_ctx, KvDtype::Fp16, slots)
            .map_err(|e| e.to_string())?;
        ready(&mut stream, 1)?;
        stream
            .set_read_timeout(Some(STEP_TIMEOUT))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            group,
            model,
            logical: logical(max_ctx, slots)?,
            positions: vec![0; slots],
            occupied: vec![false; slots],
            sequence: 1,
            pipe: None,
            poisoned: None,
        })
    }

    fn exchange(&mut self, msg: ControlMessage) -> Result<(), String> {
        msg.to_stream(&mut self.stream).map_err(|e| e.to_string())?;
        self.sequence += 1;
        ready(&mut self.stream, self.sequence)
    }

    fn reset_both(&mut self) -> Result<(), String> {
        let event = self
            .logical
            .authorize(Operation::Flush)
            .map_err(str::to_owned)?;
        let kv_event = serde_json::to_value(&event).map_err(|e| e.to_string())?;
        self.exchange(ControlMessage::TpReset {
            sequence: self.sequence + 1,
            kv_event,
        })?;
        self.model.reset().map_err(|e| e.to_string())?;
        self.positions.fill(0);
        self.occupied.fill(false);
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.run_rows(&[(0, token, self.positions[0])], 0)
    }

    fn release(&mut self, occupied: &[bool]) -> Result<(), String> {
        if occupied.len() != self.occupied.len() {
            return Err("TP occupancy width mismatch".into());
        }
        let slots: Vec<usize> = self
            .occupied
            .iter()
            .enumerate()
            .filter_map(|(i, was)| (*was && !occupied[i]).then_some(i))
            .collect();
        if slots.is_empty() {
            return Ok(());
        }
        let kv_events = slots
            .iter()
            .map(|&slot| {
                self.logical
                    .authorize(Operation::Release { slot })
                    .map_err(str::to_owned)
                    .and_then(|event| serde_json::to_value(event).map_err(|e| e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.exchange(ControlMessage::TpRelease {
            sequence: self.sequence + 1,
            slots: slots.clone(),
            kv_events,
        })?;
        for slot in slots {
            self.model.reset_slot(slot).map_err(|e| e.to_string())?;
            self.positions[slot] = 0;
            self.occupied[slot] = false;
        }
        Ok(())
    }

    fn batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, String> {
        let rows = active_rows(tokens, positions, self.positions.len())?;
        if rows.is_empty() {
            return Ok(vec![0.0; tokens.len() * self.model.vocab()]);
        }
        self.run_rows(&rows, tokens.len())
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if slot >= self.positions.len()
            || tokens.is_empty()
            || self.occupied[slot]
            || tokens.len() >= self.model.max_ctx()
        {
            return Err("TP prefill slot or prompt invalid".into());
        }
        let mut last = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            last = self.run_rows(&[(slot, token, position)], 0)?;
        }
        Ok(last)
    }

    // Validate the entire tick before authorizing any KV mutation or sending
    // control. Active pipe members keep executing as dummies after completion;
    // the scheduler ignores their Hole results until the segment drains.
    fn pipe_plans(
        slots: &[usize],
        width: usize,
        plans: &[RowSample],
    ) -> Result<Vec<crate::sampler::DevicePlan>, String> {
        if plans.len() != width || slots.is_empty() {
            return Err("TP pipe plan width mismatch".into());
        }
        let mut selected = Vec::with_capacity(slots.len());
        for (i, plan) in plans.iter().enumerate() {
            if slots.contains(&i) {
                let device = match plan {
                    RowSample::Device(p) => *p,
                    RowSample::Hole => crate::sampler::DevicePlan::Greedy,
                    RowSample::Host => return Err("TP pipe cannot read host logits".into()),
                };
                super::tp_model::tp_sample_params(device).map_err(|e| e.to_string())?;
                selected.push(device);
            } else if !matches!(plan, RowSample::Hole) {
                return Err("TP pipe plan disagrees with holes".into());
            }
        }
        Ok(selected)
    }

    fn pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), String> {
        if self.pipe.is_some() || !self.model.supports_device_sampling() {
            return Err("TP pipe unavailable or already in flight".into());
        }
        let rows = active_rows(tokens, positions, self.positions.len())?;
        let slots: Vec<_> = rows.iter().map(|row| row.0).collect();
        // Begin cannot contain a dead member; next ticks can carry dummy holes.
        validate_sampled_rows(&rows, plans, tokens.len())?;
        let devices = Self::pipe_plans(&slots, tokens.len(), plans)?;
        if rows.is_empty()
            || rows
                .iter()
                .any(|&(slot, _, pos)| pos != self.positions[slot] || pos >= self.model.max_ctx())
        {
            return Err("TP pipe begin position invalid".into());
        }
        let kv_events = rows
            .iter()
            .map(|&(slot, _, position)| {
                self.logical
                    .authorize(Operation::Ensure { slot, position })
                    .map_err(str::to_owned)
                    .and_then(|event| serde_json::to_value(event).map_err(|e| e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let seq = self.sequence + 1;
        ControlMessage::TpPipeBegin {
            sequence: seq,
            rows: rows.clone(),
            kv_events,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, seq)?;
        let mut events = Vec::with_capacity(rows.len());
        for (&(slot, token, position), plan) in rows.iter().zip(devices) {
            let event = self
                .model
                .forward_host_to_feedback(
                    &self.group,
                    &self.logical,
                    token,
                    position,
                    slot,
                    0,
                    plan,
                )
                .map_err(|e| e.to_string())?;
            events.push((slot, event));
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        self.sequence = seq;
        self.pipe = Some(PipeFlight {
            slots,
            width: tokens.len(),
            plane: 0,
            events,
        });
        tracing::info!(sequence = seq, "TP decode pipe began");
        Ok(())
    }

    fn pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, String> {
        let flight = self.pipe.as_ref().ok_or("TP pipe next without begin")?;
        let devices = Self::pipe_plans(&flight.slots, flight.width, plans)?;
        let rows = flight
            .slots
            .iter()
            .map(|&slot| (slot, self.positions[slot]))
            .collect::<Vec<_>>();
        if rows.iter().any(|&(_, pos)| pos >= self.model.max_ctx()) {
            return Err("TP pipe reached context limit; drain before next tick".into());
        }
        let source_plane = flight.plane;
        let next_plane = source_plane ^ 1;
        let kv_events = rows
            .iter()
            .map(|&(slot, position)| {
                self.logical
                    .authorize(Operation::Ensure { slot, position })
                    .map_err(str::to_owned)
                    .and_then(|event| serde_json::to_value(event).map_err(|e| e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let seq = self.sequence + 1;
        ControlMessage::TpPipeNext {
            sequence: seq,
            rows: rows.clone(),
            source_plane,
            next_plane,
            kv_events,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, seq)?;
        let mut events = Vec::with_capacity(rows.len());
        for (&(slot, position), plan) in rows.iter().zip(devices) {
            let event = self
                .model
                .forward_feedback_to_feedback(
                    &self.group,
                    &self.logical,
                    slot,
                    position,
                    source_plane,
                    next_plane,
                    plan,
                )
                .map_err(|e| e.to_string())?
                .ok_or("rank 0 missing feedback event")?;
            events.push((slot, event));
            self.positions[slot] += 1;
        }
        // Rank 1 has enqueued the next tick before acknowledging the old one.
        ready(&mut self.stream, self.sequence)?;
        let old = self.pipe.as_mut().ok_or("TP pipe disappeared")?;
        let mut ids = vec![0; old.width];
        for (slot, event) in &old.events {
            ids[*slot] = self
                .model
                .feedback_id_after(event, *slot, source_plane)
                .map_err(|e| e.to_string())?;
        }
        old.events = events;
        old.plane = next_plane;
        self.sequence = seq;
        Ok(ids)
    }

    fn pipe_drain(&mut self) -> Result<Vec<u32>, String> {
        let flight = self.pipe.as_ref().ok_or("TP pipe drain without begin")?;
        let seq = self.sequence + 1;
        ControlMessage::TpPipeDrain { sequence: seq }
            .to_stream(&mut self.stream)
            .map_err(|e| e.to_string())?;
        ready(&mut self.stream, self.sequence)?;
        ready(&mut self.stream, seq)?;
        let mut ids = vec![0; flight.width];
        for (slot, event) in &flight.events {
            ids[*slot] = self
                .model
                .feedback_id_after(event, *slot, flight.plane)
                .map_err(|e| e.to_string())?;
        }
        self.group
            .stream()
            .synchronize()
            .map_err(|e| e.to_string())?;
        self.model.synchronize().map_err(|e| e.to_string())?;
        self.sequence = seq;
        self.pipe = None;
        tracing::info!(sequence = seq, "TP decode pipe drained");
        Ok(ids)
    }

    fn sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, String> {
        if !self.model.supports_device_sampling() {
            return Err("TP device sampler unavailable".into());
        }
        let rows = active_rows(tokens, positions, self.positions.len())?;
        validate_sampled_rows(&rows, plans, tokens.len())?;
        if rows.is_empty() {
            return Ok(SampledStep {
                ids: vec![0; tokens.len()],
                host_rows: Vec::new(),
            });
        }
        self.run_rows_impl(&rows, tokens.len(), Some(plans))
    }

    fn run_rows(&mut self, rows: &[(usize, u32, usize)], width: usize) -> Result<Vec<f32>, String> {
        let result = self.run_rows_impl(rows, width, None)?;
        if width == 0 {
            return Ok(result
                .host_rows
                .into_iter()
                .next()
                .ok_or("TP missing serial logits")?
                .1);
        }
        let mut logits = vec![0.0; width * self.model.vocab()];
        for (slot, row) in result.host_rows {
            logits[slot * self.model.vocab()..(slot + 1) * self.model.vocab()]
                .copy_from_slice(&row);
        }
        Ok(logits)
    }

    fn run_rows_impl(
        &mut self,
        rows: &[(usize, u32, usize)],
        width: usize,
        plans: Option<&[RowSample]>,
    ) -> Result<SampledStep, String> {
        if rows.is_empty()
            || rows.len() > self.positions.len()
            || rows.windows(2).any(|w| w[0].0 >= w[1].0)
        {
            return Err("TP batch has no rows or unordered slots".into());
        }
        for &(slot, _, position) in rows {
            if slot >= self.positions.len()
                || position != self.positions[slot]
                || position >= self.model.max_ctx()
            {
                return Err("TP slot position disagrees with rank-0 plan".into());
            }
        }
        if let Some(plans) = plans {
            validate_sampled_rows(rows, plans, width)?;
        }
        let kv_events = rows
            .iter()
            .map(|&(slot, _, position)| {
                self.logical
                    .authorize(Operation::Ensure { slot, position })
                    .map_err(str::to_owned)
                    .and_then(|event| serde_json::to_value(event).map_err(|e| e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ControlMessage::TpBatch {
            sequence: self.sequence + 1,
            rows: rows.to_vec(),
            kv_events,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, self.sequence + 1)?;
        let mut step = SampledStep {
            ids: vec![0; width],
            host_rows: Vec::new(),
        };
        for &(slot, token, position) in rows {
            match plans.map(|p| p[slot]).unwrap_or(RowSample::Host) {
                RowSample::Host => {
                    let row = self
                        .model
                        .forward_token_slot(&self.group, &self.logical, token, position, slot)
                        .map_err(|e| e.to_string())?;
                    step.host_rows.push((slot, row));
                }
                RowSample::Device(plan) => {
                    step.ids[slot] = self
                        .model
                        .forward_token_sampled_slot(
                            &self.group,
                            &self.logical,
                            token,
                            position,
                            slot,
                            plan,
                        )
                        .map_err(|e| e.to_string())?;
                }
                RowSample::Hole => return Err("TP live slot was planned as a hole".into()),
            }
        }
        self.sequence += 1;
        ready(&mut self.stream, self.sequence)?;
        for &(slot, _, _) in rows {
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        Ok(step)
    }
}

enum Command {
    Reset,
    Step(u32),
    Prefill(usize, Vec<u32>),
    Batch(Vec<u32>, Vec<u32>),
    BatchSampled(Vec<u32>, Vec<u32>, Vec<RowSample>),
    PipeBegin(Vec<u32>, Vec<u32>, Vec<RowSample>),
    PipeNext(Vec<RowSample>),
    PipeDrain,
    Release(Vec<bool>),
}

enum Response {
    Logits(Vec<f32>),
    Sampled(SampledStep),
    Ids(Vec<u32>),
}

/// Send-only proxy on the engine scheduler thread. The NCCL communicator and
/// CUDA context never leave their owning GPU thread (cudarc Comm is !Send).
pub struct TpGenerator {
    commands: std::sync::mpsc::Sender<(Command, std::sync::mpsc::Sender<Result<Response, String>>)>,
    vocab: usize,
    max_ctx: usize,
    slots: usize,
    device_sampling: bool,
    poisoned: Option<String>,
}

impl TpGenerator {
    pub fn load(
        stream: TcpStream,
        resolved: Resolved,
        path: &Path,
        pack: &Path,
        gpu: usize,
        max_ctx: usize,
        slots: usize,
    ) -> Result<Self, String> {
        let (commands, rx) = std::sync::mpsc::channel::<(
            Command,
            std::sync::mpsc::Sender<Result<Response, String>>,
        )>();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let path = path.to_owned();
        let pack = pack.to_owned();
        std::thread::Builder::new()
            .name("qwen35-tp-rank0".into())
            .spawn(move || {
                let mut coordinator =
                    match TpCoordinator::load(stream, &resolved, &path, &pack, gpu, max_ctx, slots)
                    {
                        Ok(coordinator) => coordinator,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                if ready_tx
                    .send(Ok((
                        coordinator.model.vocab(),
                        coordinator.model.max_ctx(),
                        coordinator.model.supports_device_sampling(),
                    )))
                    .is_err()
                {
                    return;
                }
                for (cmd, reply) in rx {
                    let result = if coordinator.pipe.is_some()
                        && !matches!(cmd, Command::PipeNext(_) | Command::PipeDrain)
                    {
                        Err("TP pipe must drain before release, reset or forward".into())
                    } else {
                        match cmd {
                            Command::Reset => coordinator
                                .reset_both()
                                .map(|_| Response::Logits(Vec::new())),
                            Command::Step(token) => coordinator.step(token).map(Response::Logits),
                            Command::Prefill(slot, tokens) => {
                                coordinator.prefill(slot, &tokens).map(Response::Logits)
                            }
                            Command::Batch(tokens, positions) => {
                                coordinator.batch(&tokens, &positions).map(Response::Logits)
                            }
                            Command::BatchSampled(tokens, positions, plans) => coordinator
                                .sampled(&tokens, &positions, &plans)
                                .map(Response::Sampled),
                            Command::PipeBegin(tokens, positions, plans) => coordinator
                                .pipe_begin(&tokens, &positions, &plans)
                                .map(|_| Response::Logits(Vec::new())),
                            Command::PipeNext(plans) => {
                                coordinator.pipe_next(&plans).map(Response::Ids)
                            }
                            Command::PipeDrain => coordinator.pipe_drain().map(Response::Ids),
                            Command::Release(occupied) => coordinator
                                .release(&occupied)
                                .map(|_| Response::Logits(Vec::new())),
                        }
                    };
                    if let Err(e) = &result {
                        coordinator.poisoned = Some(e.clone());
                    }
                    let failed = result.is_err();
                    let _ = reply.send(result);
                    if failed {
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        let (vocab, max_ctx, device_sampling) = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            commands,
            vocab,
            max_ctx,
            slots,
            device_sampling,
            poisoned: None,
        })
    }

    fn request(&mut self, command: Command) -> Result<Response, String> {
        if let Some(e) = &self.poisoned {
            return Err(e.clone());
        }
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        let result = self
            .commands
            .send((command, reply_tx))
            .map_err(|e| e.to_string())
            .and_then(|_| reply_rx.recv().map_err(|e| e.to_string()))
            .and_then(|result| result);
        if let Err(ref e) = result {
            self.poisoned = Some(e.clone());
        }
        result
    }

    fn request_logits(&mut self, command: Command) -> Result<Vec<f32>, String> {
        match self.request(command)? {
            Response::Logits(logits) => Ok(logits),
            _ => Err("TP reply kind mismatch".into()),
        }
    }
}

impl Generator for TpGenerator {
    fn serial_only(&self) -> bool {
        self.slots == 1
    }
    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        if max_batch != self.slots {
            return Err(GenError::Config(
                "TP batch width changed after rank bootstrap".into(),
            ));
        }
        Ok(self.slots)
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        // The Generator trait has no error return here. Keep the proxy poisoned
        // and surface the failure; subsequent forwards must fail rather than
        // reuse a slot whose worker-side release was not acknowledged.
        if let Err(e) = self.request(Command::Release(occupied.to_vec())) {
            tracing::error!(error = %e, "TP slot release failed; rank pair poisoned");
        }
    }
    fn supports_device_sampling(&self) -> bool {
        self.device_sampling
    }
    fn supports_decode_pipe(&self) -> bool {
        self.device_sampling
    }
    fn decode_pipe_context_limit(&self) -> Option<usize> {
        Some(self.max_ctx)
    }
    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        self.request_logits(Command::PipeBegin(
            tokens.to_vec(),
            positions.to_vec(),
            plans.to_vec(),
        ))
        .map(|_| ())
        .map_err(GenError::Backend)
    }
    fn decode_pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
        match self
            .request(Command::PipeNext(plans.to_vec()))
            .map_err(GenError::Backend)?
        {
            Response::Ids(ids) => Ok(ids),
            _ => Err(GenError::Backend("TP pipe reply kind mismatch".into())),
        }
    }
    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        match self
            .request(Command::PipeDrain)
            .map_err(GenError::Backend)?
        {
            Response::Ids(ids) => Ok(ids),
            _ => Err(GenError::Backend("TP pipe reply kind mismatch".into())),
        }
    }
    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.request_logits(Command::Prefill(slot, tokens.to_vec()))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        self.request_logits(Command::Batch(tokens.to_vec(), positions.to_vec()))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        match self
            .request(Command::BatchSampled(
                tokens.to_vec(),
                positions.to_vec(),
                plans.to_vec(),
            ))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))?
        {
            Response::Sampled(step) => Ok(step),
            _ => Err(GenError::Backend("TP sampled reply kind mismatch".into())),
        }
    }
    fn reset(&mut self) {
        if let Err(e) = self.request(Command::Reset) {
            tracing::error!(error = %e, "TP reset failed; rank pair poisoned");
        }
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.request_logits(Command::Step(token))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn max_context(&self) -> usize {
        self.max_ctx
    }
}

impl Drop for TpCoordinator {
    fn drop(&mut self) {
        let _ = shutdown_worker(&mut self.stream, self.poisoned.is_none());
    }
}

/// Runs in the rank-1 process, with no HTTP listener, tokenizer or sampler.
pub fn run_worker(
    mut stream: TcpStream,
    resolved: &Resolved,
    model_path: &Path,
    pack: &Path,
    gpu: usize,
) -> Result<(), String> {
    if !resolved.is_worker() {
        return Err("TP worker must have rank 1".into());
    }
    stream
        .set_read_timeout(Some(READY_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(STEP_TIMEOUT))
        .map_err(|e| e.to_string())?;
    let run = (|| -> Result<(), String> {
        let (max_ctx, slots) =
            match ControlMessage::from_stream(&mut stream).map_err(|e| e.to_string())? {
                ControlMessage::TpInit {
                    checkpoint_sha256,
                    pack_blake3,
                    max_ctx,
                    slots,
                } => {
                    let (own_checkpoint, own_pack) = hashes(model_path, pack)?;
                    if own_checkpoint != checkpoint_sha256
                        || own_pack != pack_blake3
                        || max_ctx == 0
                        || !(1..=2).contains(&slots)
                    {
                        return Err(
                            "rank-1 checkpoint, CUDA pack or context disagrees with rank 0".into(),
                        );
                    }
                    (max_ctx, slots)
                }
                ControlMessage::Shutdown { graceful: true } => return Ok(()),
                other => return Err(format!("expected Phase 9 init: {other:?}")),
            };
        ControlMessage::TpReady { sequence: 0 }
            .to_stream(&mut stream)
            .map_err(|e| e.to_string())?;
        let id = receive_nccl_id(&mut stream).map_err(|e| e.to_string())?;
        let exec = Arc::new(GpuExecutor::new(gpu, pack).map_err(|e| e.to_string())?);
        let group = NcclCommunicator::from_resolved(Some(resolved), exec.stream.context(), id)
            .map_err(|e| e.to_string())?
            .ok_or("TP group absent")?;
        let map = MappedGguf::open(model_path).map_err(|e| e.to_string())?;
        let mut model =
            Qwen35TpRank::load_slots(exec.clone(), &map, &group, max_ctx, KvDtype::Fp16, slots)
                .map_err(|e| e.to_string())?;
        let mut logical = logical(max_ctx, slots)?;
        let mut positions = vec![0; slots];
        let mut pipe_slots: Option<Vec<usize>> = None;
        let mut pipe_plane = 0usize;
        let mut pending_pipe: Option<u64> = None;
        let mut sequence = 1;
        ControlMessage::TpReady { sequence }
            .to_stream(&mut stream)
            .map_err(|e| e.to_string())?;
        // Idle serving may last indefinitely; only rank 0's per-step ACK
        // reads have a timeout. A closed control socket still ends this loop.
        stream.set_read_timeout(None).map_err(|e| e.to_string())?;
        loop {
            let msg = ControlMessage::from_stream(&mut stream).map_err(|e| e.to_string())?;
            let got = match &msg {
                ControlMessage::TpReset { sequence, .. }
                | ControlMessage::TpBatch { sequence, .. }
                | ControlMessage::TpPipeBegin { sequence, .. }
                | ControlMessage::TpPipeNext { sequence, .. }
                | ControlMessage::TpPipeDrain { sequence }
                | ControlMessage::TpRelease { sequence, .. } => *sequence,
                ControlMessage::Shutdown { graceful: true } if pipe_slots.is_none() => {
                    return Ok(());
                }
                ControlMessage::Shutdown { graceful: true } => {
                    return Err("TP pipe must drain before shutdown".into());
                }
                ControlMessage::Shutdown { graceful: false } => return Err("rank 0 aborted".into()),
                other => return Err(format!("unexpected TP command: {other:?}")),
            };
            if got != sequence + 1 {
                return Err(format!(
                    "TP command sequence {got}, expected {}",
                    sequence + 1
                ));
            }
            if pipe_slots.is_some()
                && !matches!(
                    msg,
                    ControlMessage::TpPipeNext { .. } | ControlMessage::TpPipeDrain { .. }
                )
            {
                return Err("TP pipe must drain before another command".into());
            }
            match msg {
                ControlMessage::TpPipeBegin {
                    rows, kv_events, ..
                } => {
                    if rows.is_empty()
                        || rows.len() > slots
                        || rows.len() != kv_events.len()
                        || rows.windows(2).any(|w| w[0].0 >= w[1].0)
                    {
                        return Err("invalid TP pipe begin membership".into());
                    }
                    for (&(slot, _, position), value) in rows.iter().zip(kv_events) {
                        if slot >= slots || position >= max_ctx || position != positions[slot] {
                            return Err("TP pipe begin position mismatch".into());
                        }
                        let event: Event =
                            serde_json::from_value(value).map_err(|e| e.to_string())?;
                        if event.operation != (Operation::Ensure { slot, position }) {
                            return Err("TP pipe begin KV event mismatch".into());
                        }
                        logical.mirror(&event).map_err(str::to_owned)?;
                    }
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for &(slot, token, position) in &rows {
                        model
                            .forward_token_worker_slot(&group, &logical, token, position, slot)
                            .map_err(|e| e.to_string())?;
                        positions[slot] += 1;
                    }
                    pipe_slots = Some(rows.iter().map(|r| r.0).collect());
                    pipe_plane = 0;
                    pending_pipe = Some(got);
                }
                ControlMessage::TpPipeNext {
                    rows,
                    source_plane,
                    next_plane,
                    kv_events,
                    ..
                } => {
                    let members = pipe_slots.as_ref().ok_or("TP pipe next without begin")?;
                    if rows.len() != members.len()
                        || rows.len() != kv_events.len()
                        || source_plane != pipe_plane
                        || next_plane != (source_plane ^ 1)
                    {
                        return Err("TP pipe next shape or plane mismatch".into());
                    }
                    for ((&(slot, position), &member), value) in
                        rows.iter().zip(members).zip(kv_events)
                    {
                        if slot != member || position >= max_ctx || position != positions[slot] {
                            return Err("TP pipe next position or membership mismatch".into());
                        }
                        let event: Event =
                            serde_json::from_value(value).map_err(|e| e.to_string())?;
                        if event.operation != (Operation::Ensure { slot, position }) {
                            return Err("TP pipe next KV event mismatch".into());
                        }
                        logical.mirror(&event).map_err(str::to_owned)?;
                    }
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for (slot, position) in rows {
                        model
                            .forward_feedback_to_feedback(
                                &group,
                                &logical,
                                slot,
                                position,
                                source_plane,
                                next_plane,
                                crate::sampler::DevicePlan::Greedy,
                            )
                            .map_err(|e| e.to_string())?;
                        positions[slot] += 1;
                    }
                    let previous = pending_pipe
                        .replace(got)
                        .ok_or("TP pipe missing pending tick")?;
                    ControlMessage::TpReady { sequence: previous }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    pipe_plane = next_plane;
                }
                ControlMessage::TpPipeDrain { .. } => {
                    if pipe_slots.is_none() {
                        return Err("TP pipe drain without begin".into());
                    }
                    exec.synchronize().map_err(|e| e.to_string())?;
                    group.stream().synchronize().map_err(|e| e.to_string())?;
                    pipe_slots = None;
                    let previous = pending_pipe.take().ok_or("TP pipe missing pending tick")?;
                    ControlMessage::TpReady { sequence: previous }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                }
                ControlMessage::TpReset { kv_event, .. } => {
                    let event: Event =
                        serde_json::from_value(kv_event).map_err(|e| e.to_string())?;
                    if event.operation != Operation::Flush {
                        return Err("TP reset is not a flush".into());
                    }
                    logical.mirror(&event).map_err(str::to_owned)?;
                    model.reset().map_err(|e| e.to_string())?;
                    positions.fill(0);
                }
                ControlMessage::TpRelease {
                    slots: freed,
                    kv_events,
                    ..
                } => {
                    if freed.is_empty()
                        || freed.len() != kv_events.len()
                        || freed.windows(2).any(|w| w[0] >= w[1])
                    {
                        return Err("invalid TP release membership".into());
                    }
                    for (&slot, value) in freed.iter().zip(kv_events) {
                        if slot >= slots || positions[slot] == 0 {
                            return Err("TP release slot invalid".into());
                        }
                        let event: Event =
                            serde_json::from_value(value).map_err(|e| e.to_string())?;
                        if event.operation != (Operation::Release { slot }) {
                            return Err("TP release event mismatch".into());
                        }
                        logical.mirror(&event).map_err(str::to_owned)?;
                    }
                    for slot in freed {
                        model.reset_slot(slot).map_err(|e| e.to_string())?;
                        positions[slot] = 0;
                    }
                }
                ControlMessage::TpBatch {
                    rows, kv_events, ..
                } => {
                    if rows.is_empty()
                        || rows.len() > slots
                        || rows.len() != kv_events.len()
                        || rows.windows(2).any(|w| w[0].0 >= w[1].0)
                    {
                        return Err("TP batch membership invalid".into());
                    }
                    for (&(slot, _, position), value) in rows.iter().zip(kv_events) {
                        if slot >= slots || position >= max_ctx || position != positions[slot] {
                            return Err("TP worker position diverged".into());
                        }
                        let event: Event =
                            serde_json::from_value(value).map_err(|e| e.to_string())?;
                        if event.operation != (Operation::Ensure { slot, position }) {
                            return Err("TP batch KV event mismatch".into());
                        }
                        logical.mirror(&event).map_err(str::to_owned)?;
                    }
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for (slot, token, position) in rows {
                        model
                            .forward_token_worker_slot(&group, &logical, token, position, slot)
                            .map_err(|e| e.to_string())?;
                        positions[slot] += 1;
                    }
                }
                _ => unreachable!("message shape checked above"),
            }
            sequence = got;
            if pending_pipe.is_none() {
                ControlMessage::TpReady { sequence }
                    .to_stream(&mut stream)
                    .map_err(|e| e.to_string())?;
            }
        }
    })();
    if let Err(ref e) = run {
        let _ = ControlMessage::TpError { reason: e.clone() }.to_stream(&mut stream);
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn sockets() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let worker = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (head, _) = listener.accept().unwrap();
        (head, worker)
    }

    #[test]
    fn dense_scheduler_rows_preserve_slot_ids_and_skip_holes() {
        assert_eq!(active_rows(&[9, 8], &[0, 12], 2).unwrap(), vec![(1, 8, 12)]);
        assert_eq!(
            active_rows(&[9, 8], &[3, 12], 2).unwrap(),
            vec![(0, 9, 3), (1, 8, 12)]
        );
        assert!(active_rows(&[0, 0], &[0, 0], 2).unwrap().is_empty());
        assert!(active_rows(&[1], &[1, 2], 2).is_err());
        assert!(active_rows(&[1, 2, 3], &[1, 2, 3], 2).is_err());
    }

    #[test]
    fn sampled_dense_rows_keep_holes_and_host_fallback() {
        use crate::sampler::DevicePlan;
        let rows = active_rows(&[11, 22], &[0, 9], 2).unwrap();
        assert!(
            validate_sampled_rows(
                &rows,
                &[RowSample::Hole, RowSample::Device(DevicePlan::Greedy)],
                2
            )
            .is_ok()
        );
        assert!(validate_sampled_rows(&rows, &[RowSample::Hole, RowSample::Host], 2).is_ok());
        assert!(validate_sampled_rows(&rows, &[RowSample::Host, RowSample::Host], 2).is_err());
        assert!(validate_sampled_rows(&rows, &[RowSample::Hole, RowSample::Hole], 2).is_err());
        assert!(validate_sampled_rows(&rows, &[RowSample::Hole], 2).is_err());
        assert!(
            validate_sampled_rows(
                &[(0, 11, 3), (1, 22, 9)],
                &[
                    RowSample::Host,
                    RowSample::Device(DevicePlan::Categorical { inv_t: 2.0, u: 0.5 })
                ],
                2
            )
            .is_ok()
        );
        assert!(
            validate_sampled_rows(
                &rows,
                &[
                    RowSample::Hole,
                    RowSample::Device(DevicePlan::TruncCat {
                        inv_t: 1.0,
                        u: 0.5,
                        k: 10,
                        top_p: 0.9,
                        min_p: 0.0
                    })
                ],
                2
            )
            .is_err()
        );
    }

    #[test]
    fn pipe_plan_preflight_handles_holes_and_rejects_host() {
        use crate::sampler::DevicePlan;
        let d = RowSample::Device(DevicePlan::Greedy);
        assert_eq!(
            TpCoordinator::pipe_plans(&[1], 2, &[RowSample::Hole, d])
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            TpCoordinator::pipe_plans(&[0, 1], 2, &[RowSample::Hole, d])
                .unwrap()
                .len(),
            2
        );
        assert!(TpCoordinator::pipe_plans(&[1], 2, &[d, d]).is_err());
        assert!(TpCoordinator::pipe_plans(&[0], 2, &[RowSample::Host, RowSample::Hole]).is_err());
        assert!(TpCoordinator::pipe_plans(&[0], 2, &[d]).is_err());
        assert!(TpCoordinator::pipe_plans(&[], 2, &[RowSample::Hole, RowSample::Hole]).is_err());
    }

    #[test]
    fn pipe_acks_defer_old_ready_until_next_is_prepared() {
        let (mut head, mut worker) = sockets();
        ControlMessage::TpPrepared { sequence: 2 }
            .to_stream(&mut worker)
            .unwrap();
        prepared(&mut head, 2).unwrap();
        ControlMessage::TpPrepared { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        ControlMessage::TpReady { sequence: 2 }
            .to_stream(&mut worker)
            .unwrap();
        prepared(&mut head, 3).unwrap();
        ready(&mut head, 2).unwrap();
        ControlMessage::TpReady { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        ControlMessage::TpReady { sequence: 4 }
            .to_stream(&mut worker)
            .unwrap();
        ready(&mut head, 3).unwrap();
        ready(&mut head, 4).unwrap();
    }

    #[test]
    fn mirrored_two_slot_release_and_reuse_keeps_survivor() {
        let mut head = logical(32, 2).unwrap();
        let mut worker = logical(32, 2).unwrap();
        for (slot, position) in [(0, 0), (1, 0), (0, 16), (1, 16)] {
            let event = head
                .authorize(Operation::Ensure { slot, position })
                .unwrap();
            worker.mirror(&event).unwrap();
        }
        let survivor = head.snapshot().tables[1].clone();
        let event = head.authorize(Operation::Release { slot: 0 }).unwrap();
        worker.mirror(&event).unwrap();
        let event = head
            .authorize(Operation::Ensure {
                slot: 0,
                position: 0,
            })
            .unwrap();
        worker.mirror(&event).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        assert_eq!(head.snapshot().tables[1], survivor);
        let event = head.authorize(Operation::Flush).unwrap();
        worker.mirror(&event).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        assert!(head.snapshot().tables.iter().all(Vec::is_empty));
    }

    #[test]
    fn step_requires_prepared_before_completed_ack() {
        let (mut head, mut worker) = sockets();
        ControlMessage::TpReady { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        assert!(
            prepared(&mut head, 3)
                .unwrap_err()
                .contains("did not prepare")
        );
        ControlMessage::TpPrepared { sequence: 4 }
            .to_stream(&mut worker)
            .unwrap();
        assert!(
            prepared(&mut head, 3)
                .unwrap_err()
                .contains("did not prepare")
        );
        ControlMessage::TpPrepared { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        prepared(&mut head, 3).unwrap();
        ControlMessage::TpReady { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        ready(&mut head, 3).unwrap();
    }

    #[test]
    fn worker_error_and_closed_connection_fail_closed() {
        let (mut head, mut worker) = sockets();
        ControlMessage::TpError {
            reason: "bad KV event".into(),
        }
        .to_stream(&mut worker)
        .unwrap();
        assert!(prepared(&mut head, 7).unwrap_err().contains("bad KV event"));
        drop(worker);
        assert!(ready(&mut head, 7).is_err());
    }
}
