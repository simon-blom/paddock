//! Generic TP control-plane fan-out for coordinator-side serving.
//!
//! Model schedulers send one ordered command to every nonzero rank and wait
//! for the same phase acknowledgement from every worker before entering the
//! corresponding collective sequence.

use std::{net::TcpStream, time::Duration};

use paddock_dist::{
    config::Resolved,
    protocol::{ControlMessage, NCCL_ID_BYTES, send_nccl_id},
    worker::{WorkerControl, shutdown_worker},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckKind {
    Ready,
    Prepared,
}

pub struct WorkerSet {
    controls: Vec<WorkerControl>,
}

impl WorkerSet {
    pub fn new(
        mut controls: Vec<WorkerControl>,
        resolved: &Resolved,
        ready_timeout: Duration,
        step_timeout: Duration,
    ) -> Result<Self, String> {
        controls.sort_by_key(|worker| worker.rank);
        let expected: Vec<usize> = (1..resolved.tp_size).collect();
        let actual: Vec<usize> = controls.iter().map(|worker| worker.rank).collect();
        if actual != expected {
            return Err(format!(
                "TP worker set mismatch: expected ranks {expected:?}, got {actual:?}"
            ));
        }
        for worker in &mut controls {
            worker
                .stream
                .set_read_timeout(Some(ready_timeout))
                .map_err(|e| format!("rank {} read timeout: {e}", worker.rank))?;
            worker
                .stream
                .set_write_timeout(Some(step_timeout))
                .map_err(|e| format!("rank {} write timeout: {e}", worker.rank))?;
        }
        Ok(Self { controls })
    }

    pub fn set_read_timeout(&mut self, timeout: Duration) -> Result<(), String> {
        for worker in &mut self.controls {
            worker
                .stream
                .set_read_timeout(Some(timeout))
                .map_err(|e| format!("rank {} read timeout: {e}", worker.rank))?;
        }
        Ok(())
    }

    pub fn broadcast(&mut self, msg: &ControlMessage) -> Result<(), String> {
        for worker in &mut self.controls {
            msg.to_stream(&mut worker.stream)
                .map_err(|e| format!("rank {} control send: {e}", worker.rank))?;
        }
        Ok(())
    }

    pub fn barrier(&mut self, kind: AckKind, sequence: u64) -> Result<(), String> {
        for worker in &mut self.controls {
            read_ack(&mut worker.stream, worker.rank, kind, sequence)?;
        }
        Ok(())
    }

    pub fn ready(&mut self, sequence: u64) -> Result<(), String> {
        self.barrier(AckKind::Ready, sequence)
    }

    pub fn prepared(&mut self, sequence: u64) -> Result<(), String> {
        self.barrier(AckKind::Prepared, sequence)
    }

    pub fn send_nccl_id(&mut self, id: &[u8; NCCL_ID_BYTES]) -> Result<(), String> {
        for worker in &mut self.controls {
            send_nccl_id(&mut worker.stream, id)
                .map_err(|e| format!("rank {} NCCL id send: {e}", worker.rank))?;
        }
        Ok(())
    }

    pub fn shutdown(&mut self, graceful: bool) -> Result<(), String> {
        let mut first = None;
        for worker in &mut self.controls {
            if let Err(e) = shutdown_worker(&mut worker.stream, graceful) {
                first.get_or_insert_with(|| format!("rank {} shutdown: {e}", worker.rank));
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub fn ranks(&self) -> impl Iterator<Item = usize> + '_ {
        self.controls.iter().map(|worker| worker.rank)
    }
}

fn read_ack(
    stream: &mut TcpStream,
    rank: usize,
    kind: AckKind,
    sequence: u64,
) -> Result<(), String> {
    let msg = ControlMessage::from_stream(stream).map_err(|e| format!("rank {rank}: {e}"))?;
    match (kind, msg) {
        (AckKind::Ready, ControlMessage::TpReady { sequence: got }) if got == sequence => Ok(()),
        (AckKind::Prepared, ControlMessage::TpPrepared { sequence: got }) if got == sequence => {
            Ok(())
        }
        (_, ControlMessage::TpError { reason } | ControlMessage::Reject { reason }) => {
            Err(format!("rank {rank}: {reason}"))
        }
        (_, other) => Err(format!(
            "rank {rank} acknowledgement out of order: expected {kind:?}({sequence}), got {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ack_kind_is_phase_specific() {
        assert_ne!(AckKind::Ready, AckKind::Prepared);
    }
}
