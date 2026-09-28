//! Process bootstrap: the coordinator listens, spawns (or waits for) the
//! worker ranks, and all sides run the bootstrap handshake.
//!
//! Phase 2 shape: the worker process is a full runner binary started with
//! rank-specific env. It performs the handshake, then idles on a control loop with
//! only Shutdown in its vocabulary - the execution vocabulary (Prefill,
//! Decode, ...) arrives with the distributed executor in a later phase. The
//! worker NEVER binds an HTTP port; the coordinator's API is the only API.

use crate::config::{RankRole, Resolved};
use crate::protocol::{ControlMessage, PROTOCOL_VERSION, ProtocolError, handshake_rank};
use std::net::TcpListener;
use std::process::Command;
use std::time::Duration;

/// Everything that can go wrong during bootstrap.
#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    #[error("bind {addr}:{port} for the control plane: {source}")]
    Bind {
        addr: String,
        port: u16,
        source: std::io::Error,
    },
    #[error("accept on the control plane: {0}")]
    Accept(#[from] std::io::Error),
    #[error("connect to the coordinator: {0}")]
    Connect(std::io::Error),
    #[error("handshake: {0}")]
    Handshake(#[from] ProtocolError),
    #[error("spawn TP worker: {0}")]
    Spawn(std::io::Error),
    #[error("coordinator requested a non-graceful shutdown")]
    Aborted,
}

fn spawn_worker_local_with_paths(
    resolved: &Resolved,
    rank: usize,
    model: Option<&std::path::Path>,
    pack: Option<&std::path::Path>,
) -> Result<std::process::Child, BootstrapError> {
    let exe = std::env::current_exe().map_err(BootstrapError::Spawn)?;
    let mut cmd = Command::new(exe);
    for (k, v) in resolved
        .worker_env_for(rank)
        .map_err(|e| BootstrapError::Handshake(ProtocolError::Rejected(e.to_string())))?
    {
        cmd.env(k, v);
    }
    if let Some(model) = model {
        cmd.env("PADDOCK_TP_MODEL", model);
    }
    if let Some(pack) = pack {
        cmd.env("PADDOCK_TP_PACK", pack);
    }
    cmd.stdin(std::process::Stdio::null());
    // Worker logs go to the same stderr as the coordinator (its stdout is
    // the tracing sink per paddock_admin::logging, and operators debugging
    // a two-node start need the worker's lines in the same stream).
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());
    cmd.spawn().map_err(BootstrapError::Spawn)
}

/// One joined worker control connection, identified by TP rank.
pub struct WorkerControl {
    pub rank: usize,
    pub stream: std::net::TcpStream,
}

/// Backward-compatible TP=2 coordinator helper.
pub fn coordinate(
    resolved: &Resolved,
    spawn_worker: bool,
) -> Result<(std::net::TcpStream, u64), BootstrapError> {
    let (mut workers, session) =
        coordinate_workers_with_paths(resolved, spawn_worker, None, None)?;
    if workers.len() != 1 || workers[0].rank != 1 {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(
            "coordinate() is the TP=2 compatibility helper; use coordinate_workers()".into(),
        )));
    }
    Ok((workers.remove(0).stream, session))
}

/// Join all nonzero ranks in one TP world. Returned controls are rank-sorted.
pub fn coordinate_workers(
    resolved: &Resolved,
    spawn_workers: bool,
) -> Result<(Vec<WorkerControl>, u64), BootstrapError> {
    coordinate_workers_with_paths(resolved, spawn_workers, None, None)
}

fn coordinate_workers_with_paths(
    resolved: &Resolved,
    spawn_workers: bool,
    model: Option<&std::path::Path>,
    pack: Option<&std::path::Path>,
) -> Result<(Vec<WorkerControl>, u64), BootstrapError> {
    if !resolved.is_coordinator() || resolved.tp_size < 2 {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(
            "TP coordinator requires rank 0 and tp_size >= 2".into(),
        )));
    }
    let listener = TcpListener::bind((resolved.master_addr.as_str(), resolved.master_port))
        .map_err(|source| BootstrapError::Bind {
            addr: resolved.master_addr.clone(),
            port: resolved.master_port,
            source,
        })?;
    tracing::info!(
        "rank 0 control plane listening on {}:{} (tp_size={})",
        resolved.master_addr,
        resolved.master_port,
        resolved.tp_size
    );

    if spawn_workers {
        for rank in 1..resolved.tp_size {
            match spawn_worker_local_with_paths(resolved, rank, model, pack) {
                Ok(child) => tracing::info!(
                    rank,
                    pid = child.id(),
                    "spawned TP worker child"
                ),
                Err(e) => tracing::warn!(
                    rank,
                    "could not spawn worker locally ({e}); waiting for a manual or SSH start"
                ),
            }
        }
    }

    let session = next_session();
    let mut joined: Vec<Option<std::net::TcpStream>> =
        (0..resolved.tp_size).map(|_| None).collect();
    let mut remaining = resolved.tp_size - 1;
    while remaining > 0 {
        let (mut stream, peer) = listener.accept()?;
        tracing::info!("control-plane connection from {peer}");
        match greet(&mut stream, resolved, session, &joined) {
            Ok(rank) => {
                joined[rank] = Some(stream);
                remaining -= 1;
            }
            Err(BootstrapError::Handshake(ProtocolError::Rejected(reason))) => {
                tracing::warn!("rejected a control-plane connection: {reason}");
                if let Err(e) = (ControlMessage::Reject { reason }).to_stream(&mut stream) {
                    tracing::debug!("reject delivery failed: {e}");
                }
            }
            Err(e) => return Err(e),
        }
    }

    let workers = (1..resolved.tp_size)
        .map(|rank| WorkerControl {
            rank,
            stream: joined[rank]
                .take()
                .expect("all worker ranks joined before bootstrap completed"),
        })
        .collect();
    Ok((workers, session))
}

/// Server side of one worker handshake. Rank identity is validated before
/// Welcome so duplicate/out-of-range sockets never enter the TP world.
fn greet(
    stream: &mut std::net::TcpStream,
    resolved: &Resolved,
    session: u64,
    joined: &[Option<std::net::TcpStream>],
) -> Result<usize, BootstrapError> {
    let msg = ControlMessage::from_stream(stream)?;
    let ControlMessage::Hello {
        version,
        tp_size,
        rank,
        who,
    } = msg
    else {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(format!(
            "expected Hello, got {msg:?}"
        ))));
    };
    if version != PROTOCOL_VERSION {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(format!(
            "protocol version {version}, coordinator speaks {PROTOCOL_VERSION}"
        ))));
    }
    if tp_size != resolved.tp_size {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(format!(
            "worker configured tp_size={tp_size}, coordinator is tp_size={}",
            resolved.tp_size
        ))));
    }
    if rank == 0 || rank >= resolved.tp_size {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(format!(
            "worker rank {rank} is outside 1..{}",
            resolved.tp_size
        ))));
    }
    if joined.get(rank).is_some_and(Option::is_some) {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(format!(
            "worker rank {rank} already joined"
        ))));
    }
    tracing::info!("worker '{who}' joined as rank {rank} (session {session})");
    ControlMessage::Welcome {
        tp_size: resolved.tp_size,
        rank,
        session,
    }
    .to_stream(stream)?;
    Ok(rank)
}

/// Run one nonzero worker-rank side: dial the coordinator, handshake, then serve
/// the control loop until Shutdown (graceful -> Ok, non-graceful ->
/// [`BootstrapError::Aborted`]).
///
/// Test-only harness for the bootstrap plane: the production worker entry is
/// `run_worker` in paddock-engine (shard load + ordered execution). This loop
/// validates coordination end to end without touching the engine.
#[cfg(test)]
pub fn work(resolved: &Resolved) -> Result<(), BootstrapError> {
    let (mut stream, session) = connect_worker(resolved)?;
    tracing::info!("accepted by coordinator (session {session})");

    // Phase 2 expects exactly one Shutdown after handshake. Execution
    // messages (and the benchmark's NcclId) have their own consumers.
    match ControlMessage::from_stream(&mut stream)? {
        ControlMessage::Shutdown { graceful } => {
            tracing::info!(
                "shutdown from coordinator (graceful={graceful}) - worker exiting {}",
                if graceful { "cleanly" } else { "with error" }
            );
            if graceful {
                Ok(())
            } else {
                Err(BootstrapError::Aborted)
            }
        }
        other => Err(BootstrapError::Handshake(ProtocolError::Rejected(format!(
            "worker received unexpected control message {other:?}"
        )))),
    }
}

/// Join as this configured worker rank and return the bootstrap connection after Hello / Welcome. Standalone parity probes exchange the NCCL ID on it; the Phase 9
/// worker hands it to the ordered model execution loop.
pub fn connect_worker(resolved: &Resolved) -> Result<(std::net::TcpStream, u64), BootstrapError> {
    let addr = (resolved.master_addr.as_str(), resolved.master_port);
    let who = std::env::var("HOSTNAME").unwrap_or_else(|_| "worker".to_string());
    tracing::info!(
        "rank {} dialing coordinator at {}:{} (tp_size={})",
        resolved.rank,
        resolved.master_addr,
        resolved.master_port,
        resolved.tp_size
    );
    let mut stream = connect_with_retry(addr, Duration::from_secs(30))?;
    let session = handshake_rank(&mut stream, resolved.tp_size, resolved.rank, &who)?;
    Ok((stream, session))
}

/// Dial with retry so a worker started in parallel with its coordinator
/// (both launched by a supervisor, or the coordinator's spawn racing the
/// bind) connects once the listener exists.
fn connect_with_retry(
    addr: (&str, u16),
    budget: Duration,
) -> Result<std::net::TcpStream, BootstrapError> {
    let deadline = std::time::Instant::now() + budget;
    let mut last_err: Option<std::io::Error> = None;
    while std::time::Instant::now() < deadline {
        match std::net::TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last_err = Some(e);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    Err(BootstrapError::Connect(last_err.unwrap_or_else(|| {
        std::io::Error::other("connect budget exhausted")
    })))
}

fn next_session() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Send a graceful (or not) Shutdown and close the control channel.
pub fn shutdown_worker(
    stream: &mut std::net::TcpStream,
    graceful: bool,
) -> Result<(), ProtocolError> {
    ControlMessage::Shutdown { graceful }.to_stream(stream)?;
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}

/// Coordinator-held worker controls, stored at bootstrap so shutdown and
/// model handoff operate on the complete TP world.
static COORDINATOR_CONTROLS: std::sync::OnceLock<
    std::sync::Mutex<Option<Vec<WorkerControl>>>,
> = std::sync::OnceLock::new();

/// Join the complete worker world and store its controls for model handoff.
pub fn coordinate_and_store(
    resolved: &Resolved,
    spawn_workers: bool,
    model: Option<&std::path::Path>,
    pack: Option<&std::path::Path>,
) -> Result<u64, BootstrapError> {
    let (workers, session) =
        coordinate_workers_with_paths(resolved, spawn_workers, model, pack)?;
    COORDINATOR_CONTROLS
        .set(std::sync::Mutex::new(Some(workers)))
        .map_err(|_| {
            BootstrapError::Handshake(ProtocolError::Rejected(
                "coordinator already running".into(),
            ))
        })?;
    Ok(session)
}

/// Hand ownership of every worker channel to the distributed model.
pub fn take_controls() -> Result<Vec<WorkerControl>, BootstrapError> {
    COORDINATOR_CONTROLS
        .get()
        .and_then(|m| m.lock().ok())
        .and_then(|mut slot| slot.take())
        .ok_or_else(|| {
            BootstrapError::Handshake(ProtocolError::Rejected(
                "no coordinator channels to take".into(),
            ))
        })
}

/// TP=2 compatibility helper used by the current serving implementation.
/// On a larger world it fails without consuming the stored controls.
pub fn take_control() -> Result<std::net::TcpStream, BootstrapError> {
    let mutex = COORDINATOR_CONTROLS.get().ok_or_else(|| {
        BootstrapError::Handshake(ProtocolError::Rejected(
            "no coordinator channels to take".into(),
        ))
    })?;
    let mut guard = mutex.lock().map_err(|_| {
        BootstrapError::Handshake(ProtocolError::Rejected(
            "coordinator channel lock poisoned".into(),
        ))
    })?;
    let workers = guard.as_ref().ok_or_else(|| {
        BootstrapError::Handshake(ProtocolError::Rejected(
            "no coordinator channels to take".into(),
        ))
    })?;
    if workers.len() != 1 || workers[0].rank != 1 {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(
            "take_control() requires exactly rank 1; use take_controls()".into(),
        )));
    }
    let mut workers = guard.take().expect("checked above");
    Ok(workers.remove(0).stream)
}

/// Best-effort Shutdown fan-out to every joined worker.
pub fn broadcast_shutdown(graceful: bool) -> bool {
    match COORDINATOR_CONTROLS.get() {
        Some(mutex) => {
            let Ok(mut guard) = mutex.lock() else {
                return false;
            };
            let Some(workers) = guard.as_mut() else {
                return false;
            };
            let mut any = false;
            let mut all_ok = true;
            for worker in workers {
                any = true;
                all_ok &= shutdown_worker(&mut worker.stream, graceful).is_ok();
            }
            any && all_ok
        }
        None => false,
    }
}

/// Role helpers used by the runner wiring.
impl Resolved {
    /// True when this process should run the worker control loop instead of
    /// the serving stack.
    pub fn is_worker(&self) -> bool {
        self.role == RankRole::Worker
    }
}

#[cfg(test)]
mod tests {
    //! Host coverage for the Phase 2 bootstrap loop. `work` is the loop's
    //! test harness (the production worker entry is `run_worker`, which
    //! needs the engine and a GPU), so it and these tests live together
    //! here rather than in the crate's public API.

    use super::*;
    use crate::config::ParallelConfig;

    fn resolved(rank: usize, port: u16) -> Resolved {
        let cfg = ParallelConfig {
            tp_size: Some(2),
            rank: Some(rank),
            master_addr: Some("127.0.0.1".into()),
            master_port: Some(port),
        };
        cfg.resolved(false).expect("valid").expect("tp2")
    }

    fn free_port() -> u16 {
        std::net::TcpListener::bind(("127.0.0.1", 0))
            .expect("bind probe")
            .local_addr()
            .expect("addr")
            .port()
    }

    #[test]
    fn worker_loop_exits_cleanly_on_graceful_shutdown() {
        let port = free_port();
        let coord = resolved(0, port);
        let work_cfg = resolved(1, port);

        let t = std::thread::spawn(move || coordinate(&coord, false));
        std::thread::sleep(Duration::from_millis(100));
        let w = std::thread::spawn(move || work(&work_cfg));

        let Ok((mut stream, session)) = t.join().unwrap() else {
            panic!("coordinate failed")
        };
        // Session ids are a process-global monotonic counter shared by every
        // test in this binary - assert "was assigned", not a specific value.
        assert!(session >= 1);
        shutdown_worker(&mut stream, true).unwrap();
        assert!(w.join().unwrap().is_ok());
    }

    #[test]
    fn worker_loop_errors_on_non_graceful_shutdown() {
        let port = free_port();
        let coord = resolved(0, port);
        let work_cfg = resolved(1, port);

        let t = std::thread::spawn(move || coordinate(&coord, false));
        std::thread::sleep(Duration::from_millis(100));
        let w = std::thread::spawn(move || work(&work_cfg));

        let Ok((mut stream, _session)) = t.join().unwrap() else {
            panic!("coordinate failed")
        };
        shutdown_worker(&mut stream, false).unwrap();
        assert!(matches!(w.join().unwrap(), Err(BootstrapError::Aborted)));
    }
}
