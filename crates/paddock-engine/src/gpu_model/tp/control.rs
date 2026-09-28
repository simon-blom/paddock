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
    use paddock_dist::{
        config::{RankRole, Resolved},
        worker::WorkerControl,
    };
    use std::net::{TcpListener, TcpStream};

    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (head, _) = listener.accept().unwrap();
        (head, peer)
    }

    fn tp4() -> Resolved {
        Resolved {
            tp_size: 4,
            rank: 0,
            role: RankRole::Coordinator,
            master_addr: "127.0.0.1".into(),
            master_port: 11560,
        }
    }

    fn tp4_workers() -> (WorkerSet, Vec<(usize, TcpStream)>) {
        let mut controls = Vec::new();
        let mut peers = Vec::new();
        // Deliberately unsorted: WorkerSet owns canonical rank ordering.
        for rank in [3usize, 1, 2] {
            let (head, peer) = pair();
            controls.push(WorkerControl { rank, stream: head });
            peers.push((rank, peer));
        }
        let workers = WorkerSet::new(
            controls,
            &tp4(),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .unwrap();
        (workers, peers)
    }

    #[test]
    fn ack_kind_is_phase_specific() {
        assert_ne!(AckKind::Ready, AckKind::Prepared);
    }

    #[test]
    fn worker_set_sorts_and_broadcasts_to_every_tp4_rank() {
        let (mut workers, mut peers) = tp4_workers();
        assert_eq!(workers.ranks().collect::<Vec<_>>(), vec![1, 2, 3]);

        workers
            .broadcast(&ControlMessage::TpReset {
                sequence: 9,
                kv_event: serde_json::json!({"epoch": 1}),
            })
            .unwrap();

        peers.sort_by_key(|(rank, _)| *rank);
        for (rank, peer) in &mut peers {
            match ControlMessage::from_stream(peer).unwrap() {
                ControlMessage::TpReset { sequence, .. } => assert_eq!(sequence, 9),
                other => panic!("rank {rank} received wrong command: {other:?}"),
            }
        }
    }

    #[test]
    fn worker_set_barrier_requires_ack_from_every_tp4_rank() {
        let (mut workers, mut peers) = tp4_workers();
        for (_, peer) in &mut peers {
            ControlMessage::TpPrepared { sequence: 11 }
                .to_stream(peer)
                .unwrap();
        }
        workers.prepared(11).unwrap();

        for (_, peer) in &mut peers {
            ControlMessage::TpReady { sequence: 12 }
                .to_stream(peer)
                .unwrap();
        }
        workers.ready(12).unwrap();
    }

    #[test]
    fn worker_set_reports_the_failing_rank() {
        let (mut workers, mut peers) = tp4_workers();
        peers.sort_by_key(|(rank, _)| *rank);
        ControlMessage::TpPrepared { sequence: 4 }
            .to_stream(&mut peers[0].1)
            .unwrap();
        ControlMessage::TpError {
            reason: "mirror diverged".into(),
        }
        .to_stream(&mut peers[1].1)
        .unwrap();
        ControlMessage::TpPrepared { sequence: 4 }
            .to_stream(&mut peers[2].1)
            .unwrap();

        let err = workers.prepared(4).unwrap_err();
        assert!(err.contains("rank 2"), "{err}");
        assert!(err.contains("mirror diverged"), "{err}");
    }

    #[test]
    fn worker_set_rejects_missing_or_duplicate_membership() {
        let (h1, _p1) = pair();
        let (h2, _p2) = pair();
        let err = match WorkerSet::new(
            vec![
                WorkerControl { rank: 1, stream: h1 },
                WorkerControl { rank: 1, stream: h2 },
            ],
            &tp4(),
            Duration::from_secs(1),
            Duration::from_secs(1),
        ) {
            Ok(_) => panic!("duplicate/missing worker ranks must be rejected"),
            Err(err) => err,
        };
        assert!(err.contains("worker set mismatch"), "{err}");
    }
}
