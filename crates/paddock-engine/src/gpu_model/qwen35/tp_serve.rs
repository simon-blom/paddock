//! Rank-0-authoritative TP=2 serving over the distributed control channel.
//! The production scheduler owns rank 0's dense slot plan; rank 1 replays the
//! ordered live rows and mirrored KV operations. Each row uses eager kernels.
use std::{io::Read, net::TcpStream, path::Path, sync::Arc, time::Duration};

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
    generator::{GenError, Generator},
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

/// Runs on the production engine thread; owns the control stream until drop.
pub struct TpCoordinator {
    stream: TcpStream,
    group: NcclCommunicator,
    model: Qwen35TpRank,
    logical: MirroredKv,
    positions: Vec<usize>,
    occupied: Vec<bool>,
    sequence: u64,
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

    fn run_rows(&mut self, rows: &[(usize, u32, usize)], width: usize) -> Result<Vec<f32>, String> {
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
        let mut logits = vec![0.0; width * self.model.vocab()];
        for &(slot, token, position) in rows {
            let row = self
                .model
                .forward_token_slot(&self.group, &self.logical, token, position, slot)
                .map_err(|e| e.to_string())?;
            if width == 0 {
                logits = row;
            } else {
                logits[slot * self.model.vocab()..(slot + 1) * self.model.vocab()]
                    .copy_from_slice(&row);
            }
        }
        self.sequence += 1;
        ready(&mut self.stream, self.sequence)?;
        for &(slot, _, _) in rows {
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        Ok(logits)
    }
}

enum Command {
    Reset,
    Step(u32),
    Prefill(usize, Vec<u32>),
    Batch(Vec<u32>, Vec<u32>),
    Release(Vec<bool>),
}

/// Send-only proxy on the engine scheduler thread. The NCCL communicator and
/// CUDA context never leave their owning GPU thread (cudarc Comm is !Send).
pub struct TpGenerator {
    commands: std::sync::mpsc::Sender<(Command, std::sync::mpsc::Sender<Result<Vec<f32>, String>>)>,
    vocab: usize,
    max_ctx: usize,
    slots: usize,
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
            std::sync::mpsc::Sender<Result<Vec<f32>, String>>,
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
                    .send(Ok((coordinator.model.vocab(), coordinator.model.max_ctx())))
                    .is_err()
                {
                    return;
                }
                for (cmd, reply) in rx {
                    let result = match cmd {
                        Command::Reset => coordinator.reset_both().map(|_| Vec::new()),
                        Command::Step(token) => coordinator.step(token),
                        Command::Prefill(slot, tokens) => coordinator.prefill(slot, &tokens),
                        Command::Batch(tokens, positions) => coordinator.batch(&tokens, &positions),
                        Command::Release(occupied) => {
                            coordinator.release(&occupied).map(|_| Vec::new())
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
        let (vocab, max_ctx) = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            commands,
            vocab,
            max_ctx,
            slots,
            poisoned: None,
        })
    }

    fn request(&mut self, command: Command) -> Result<Vec<f32>, String> {
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
        let _ = self.request(Command::Release(occupied.to_vec()));
    }
    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.request(Command::Prefill(slot, tokens.to_vec()))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        self.request(Command::Batch(tokens.to_vec(), positions.to_vec()))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn reset(&mut self) {
        let _ = self.request(Command::Reset);
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.request(Command::Step(token))
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
        let mut model = Qwen35TpRank::load_slots(exec, &map, &group, max_ctx, KvDtype::Fp16, slots)
            .map_err(|e| e.to_string())?;
        let mut logical = logical(max_ctx, slots)?;
        let mut positions = vec![0; slots];
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
                | ControlMessage::TpRelease { sequence, .. } => *sequence,
                ControlMessage::Shutdown { graceful: true } => return Ok(()),
                ControlMessage::Shutdown { graceful: false } => return Err("rank 0 aborted".into()),
                other => return Err(format!("unexpected TP command: {other:?}")),
            };
            if got != sequence + 1 {
                return Err(format!(
                    "TP command sequence {got}, expected {}",
                    sequence + 1
                ));
            }
            match msg {
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
            ControlMessage::TpReady { sequence }
                .to_stream(&mut stream)
                .map_err(|e| e.to_string())?;
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
