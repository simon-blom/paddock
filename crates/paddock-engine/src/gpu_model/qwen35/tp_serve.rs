//! Phase 9 serial, rank-0-authoritative TP=2 serving over the Phase 2 control channel.
//! The production engine owns rank 0. Rank 1 only executes the same ordered
//! reset/forward operations; sampling, admission and cancellation stay on rank 0.
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

fn logical(max_ctx: usize) -> Result<MirroredKv, String> {
    let blocks = u32::try_from(max_ctx.div_ceil(crate::kv_pool::BLOCK_TOKENS))
        .map_err(|_| "KV block count overflow")?;
    MirroredKv::new(blocks, 1, max_ctx).map_err(str::to_owned)
}

/// Runs on the production engine thread; owns the control stream until drop.
pub struct TpCoordinator {
    stream: TcpStream,
    group: NcclCommunicator,
    model: Qwen35TpRank,
    logical: MirroredKv,
    position: usize,
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
    ) -> Result<Self, String> {
        if !resolved.is_coordinator() || max_ctx == 0 {
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
        let model = Qwen35TpRank::load(exec, &map, &group, max_ctx, KvDtype::Fp16)
            .map_err(|e| e.to_string())?;
        ready(&mut stream, 1)?;
        stream
            .set_read_timeout(Some(STEP_TIMEOUT))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            group,
            model,
            logical: logical(max_ctx)?,
            position: 0,
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
        self.position = 0;
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        if self.position >= self.model.max_ctx() {
            return Err("TP context exhausted".into());
        }
        let event = self
            .logical
            .authorize(Operation::Ensure {
                slot: 0,
                position: self.position,
            })
            .map_err(str::to_owned)?;
        let kv_event = serde_json::to_value(&event).map_err(|e| e.to_string())?;
        ControlMessage::TpStep {
            sequence: self.sequence + 1,
            token,
            position: self.position,
            kv_event,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, self.sequence + 1)?;
        let logits = self
            .model
            .forward_token(&self.group, &self.logical, token, self.position)
            .map_err(|e| e.to_string())?;
        self.sequence += 1;
        ready(&mut self.stream, self.sequence)?;
        self.position += 1;
        Ok(logits)
    }
}

enum Command {
    Reset,
    Step(u32),
}

/// Send-only proxy on the engine scheduler thread. The NCCL communicator and
/// CUDA context never leave their owning GPU thread (cudarc Comm is !Send).
pub struct TpGenerator {
    commands: std::sync::mpsc::Sender<(Command, std::sync::mpsc::Sender<Result<Vec<f32>, String>>)>,
    vocab: usize,
    max_ctx: usize,
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
                    match TpCoordinator::load(stream, &resolved, &path, &pack, gpu, max_ctx) {
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
        true
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
        let max_ctx = match ControlMessage::from_stream(&mut stream).map_err(|e| e.to_string())? {
            ControlMessage::TpInit {
                checkpoint_sha256,
                pack_blake3,
                max_ctx,
            } => {
                let (own_checkpoint, own_pack) = hashes(model_path, pack)?;
                if own_checkpoint != checkpoint_sha256 || own_pack != pack_blake3 || max_ctx == 0 {
                    return Err(
                        "rank-1 checkpoint, CUDA pack or context disagrees with rank 0".into(),
                    );
                }
                max_ctx
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
        let mut model = Qwen35TpRank::load(exec, &map, &group, max_ctx, KvDtype::Fp16)
            .map_err(|e| e.to_string())?;
        let mut logical = logical(max_ctx)?;
        let mut sequence = 1;
        ControlMessage::TpReady { sequence }
            .to_stream(&mut stream)
            .map_err(|e| e.to_string())?;
        // Idle serving may last indefinitely; only rank 0's per-step ACK
        // reads have a timeout. A closed control socket still ends this loop.
        stream.set_read_timeout(None).map_err(|e| e.to_string())?;
        loop {
            let msg = ControlMessage::from_stream(&mut stream).map_err(|e| e.to_string())?;
            let (got, value, step) = match msg {
                ControlMessage::TpReset { sequence, kv_event } => (sequence, kv_event, None),
                ControlMessage::TpStep {
                    sequence,
                    token,
                    position,
                    kv_event,
                } => (sequence, kv_event, Some((token, position))),
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
            let event: Event = serde_json::from_value(value).map_err(|e| e.to_string())?;
            match step {
                None if event.operation == Operation::Flush => {
                    logical.mirror(&event).map_err(str::to_owned)?;
                    model.reset().map_err(|e| e.to_string())?;
                }
                Some((token, position))
                    if event.operation == (Operation::Ensure { slot: 0, position }) =>
                {
                    logical.mirror(&event).map_err(str::to_owned)?;
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    model
                        .forward_token_worker(&group, &logical, token, position)
                        .map_err(|e| e.to_string())?;
                }
                _ => return Err("TP logical event disagrees with command".into()),
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
