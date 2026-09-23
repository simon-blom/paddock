//! Process bootstrap: the coordinator listens, spawns (or waits for) the
//! rank-1 worker, and both sides run the bootstrap handshake.
//!
//! Phase 2 shape: the worker process is a full runner binary started with
//! rank-1 env. It performs the handshake, then idles on a control loop with
//! only Shutdown in its vocabulary - the execution vocabulary (Prefill,
//! Decode, ...) arrives with the distributed executor in a later phase. The
//! worker NEVER binds an HTTP port; the coordinator's API is the only API.

use crate::config::{RankRole, Resolved};
use crate::protocol::{ControlMessage, PROTOCOL_VERSION, ProtocolError, handshake};
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
    #[error("spawn rank-1 worker: {0}")]
    Spawn(std::io::Error),
    #[error("coordinator requested a non-graceful shutdown")]
    Aborted,
}

/// Spawn the rank-1 worker as a local child process running the same binary,
/// with the rank-1 env layered over this process's environment.
///
/// The child is started WITHOUT extra args: the runner's startup branch sees
/// [`crate::config::WORKER_CHILD_ENV`] plus the rank env and enters worker
/// mode before any HTTP surface exists. The worker inherits the
/// coordinator's config surface (model, device, kernel pack, ...) through
/// the environment - the same mechanism the manager uses to start runners -
/// so no second config file exists to drift.
pub fn spawn_worker_local(resolved: &Resolved) -> Result<std::process::Child, BootstrapError> {
    spawn_worker_local_with_paths(resolved, None, None)
}

fn spawn_worker_local_with_paths(
    resolved: &Resolved,
    model: Option<&std::path::Path>,
    pack: Option<&std::path::Path>,
) -> Result<std::process::Child, BootstrapError> {
    let exe = std::env::current_exe().map_err(BootstrapError::Spawn)?;
    let mut cmd = Command::new(exe);
    for (k, v) in resolved.worker_env() {
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

/// Run the rank-0 coordinator side of bootstrap: bind the control plane,
/// accept exactly one worker, exchange the handshake, return the connection.
///
/// `spawn_worker` decides whether the coordinator launches the worker itself
/// (local two-process bring-up and tests) or waits for an operator- or
/// SSH-started one (the real two-node case).
pub fn coordinate(
    resolved: &Resolved,
    spawn_worker: bool,
) -> Result<(std::net::TcpStream, u64), BootstrapError> {
    coordinate_with_paths(resolved, spawn_worker, None, None)
}

fn coordinate_with_paths(
    resolved: &Resolved,
    spawn_worker: bool,
    model: Option<&std::path::Path>,
    pack: Option<&std::path::Path>,
) -> Result<(std::net::TcpStream, u64), BootstrapError> {
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

    if spawn_worker {
        match spawn_worker_local_with_paths(resolved, model, pack) {
            Ok(child) => tracing::info!("spawned rank-1 worker child (pid {})", child.id()),
            Err(e) => tracing::warn!(
                "could not spawn worker locally ({e}); waiting for a manual or SSH start"
            ),
        }
    }

    // Exactly one worker for TP=2. Wrong-world-size dials are rejected and
    // the accept loop keeps waiting for the real worker.
    loop {
        let (mut stream, peer) = listener.accept()?;
        tracing::info!("control-plane connection from {peer}");
        let session = next_session();
        match greet(&mut stream, resolved, session) {
            Ok(()) => return Ok((stream, session)),
            Err(BootstrapError::Handshake(ProtocolError::Rejected(reason))) => {
                tracing::warn!("rejected a control-plane connection: {reason}");
                if let Err(e) = (ControlMessage::Reject { reason }).to_stream(&mut stream) {
                    tracing::debug!("reject delivery failed: {e}");
                }
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Server side of the handshake: read Hello, enforce version + world size,
/// send Welcome/Reject. Mismatches are reported as `Rejected` so the accept
/// loop can continue; the Reject frame is sent here before returning.
fn greet(
    stream: &mut std::net::TcpStream,
    resolved: &Resolved,
    session: u64,
) -> Result<(), BootstrapError> {
    let msg = ControlMessage::from_stream(stream)?;
    let ControlMessage::Hello {
        version,
        tp_size,
        who,
    } = msg
    else {
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(format!(
            "expected Hello, got {msg:?}"
        ))));
    };
    if version != PROTOCOL_VERSION {
        let reason = format!("protocol version {version}, coordinator speaks {PROTOCOL_VERSION}");
        ControlMessage::Reject {
            reason: reason.clone(),
        }
        .to_stream(stream)?;
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(reason)));
    }
    if tp_size != resolved.tp_size {
        let reason = format!(
            "worker configured tp_size={tp_size}, coordinator is tp_size={}",
            resolved.tp_size
        );
        ControlMessage::Reject {
            reason: reason.clone(),
        }
        .to_stream(stream)?;
        return Err(BootstrapError::Handshake(ProtocolError::Rejected(reason)));
    }
    tracing::info!("worker '{who}' joined as rank 1 (session {session})");
    ControlMessage::Welcome {
        tp_size: resolved.tp_size,
        session,
    }
    .to_stream(stream)?;
    Ok(())
}

/// Run the rank-1 worker side: dial the coordinator, handshake, then serve
/// the control loop until Shutdown (graceful -> Ok, non-graceful ->
/// [`BootstrapError::Aborted`]).
///
/// This is the whole worker runtime for Phase 2. It deliberately does not
/// touch the engine: the runner wiring decides what else the worker process
/// does around this loop in later phases (shard load, control-driven
/// execution); today it is a bootstrap skeleton that validates the
/// coordination plane end to end.
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

/// Join as rank 1 and return the bootstrap control connection after Hello /
/// Welcome. Standalone parity probes exchange the NCCL ID on it; the Phase 9
/// worker hands it to the ordered model execution loop.
pub fn connect_worker(resolved: &Resolved) -> Result<(std::net::TcpStream, u64), BootstrapError> {
    let addr = (resolved.master_addr.as_str(), resolved.master_port);
    let who = std::env::var("HOSTNAME").unwrap_or_else(|_| "worker".to_string());
    tracing::info!(
        "rank 1 dialing coordinator at {}:{} (tp_size={})",
        resolved.master_addr,
        resolved.master_port,
        resolved.tp_size
    );
    let mut stream = connect_with_retry(addr, Duration::from_secs(30))?;
    let session = handshake(&mut stream, resolved.tp_size, &who)?;
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

/// The coordinator's held control channel, stored at bootstrap so the
/// runner's shutdown path can release the worker without threading the
/// stream through every layer.
static COORDINATOR_CONTROL: std::sync::OnceLock<std::sync::Mutex<Option<std::net::TcpStream>>> =
    std::sync::OnceLock::new();

/// Run [`coordinate`] and store the connection for later
/// [`broadcast_shutdown`]. Called once by the runner startup on rank 0.
pub fn coordinate_and_store(
    resolved: &Resolved,
    spawn_worker: bool,
    model: Option<&std::path::Path>,
    pack: Option<&std::path::Path>,
) -> Result<u64, BootstrapError> {
    let (stream, session) = coordinate_with_paths(resolved, spawn_worker, model, pack)?;
    COORDINATOR_CONTROL
        .set(std::sync::Mutex::new(Some(stream)))
        .map_err(|_| {
            BootstrapError::Handshake(ProtocolError::Rejected(
                "coordinator already running".into(),
            ))
        })?;
    Ok(session)
}

/// Hand ownership of the bootstrap channel to the Phase 9 model. The generic
/// runner shutdown path must not send Shutdown while the model is executing.
pub fn take_control() -> Result<std::net::TcpStream, BootstrapError> {
    COORDINATOR_CONTROL
        .get()
        .and_then(|m| m.lock().ok())
        .and_then(|mut slot| slot.take())
        .ok_or_else(|| {
            BootstrapError::Handshake(ProtocolError::Rejected(
                "no coordinator channel to take".into(),
            ))
        })
}

/// Send Shutdown to the joined worker, if one is. Returns whether a worker
/// was connected and acknowledged the send (best effort - the worker may
/// already be gone; that is not an error for the coordinator's exit).
pub fn broadcast_shutdown(graceful: bool) -> bool {
    match COORDINATOR_CONTROL.get() {
        Some(mutex) => {
            let Ok(mut stream) = mutex.lock() else {
                return false;
            };
            matches!(
                stream.as_mut().map(|s| shutdown_worker(s, graceful)),
                Some(Ok(()))
            )
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
