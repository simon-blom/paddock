//! The rank0<->rank1 control protocol: length-prefixed JSON over TCP.
//!
//! Bootstrap, model identity and rank-0-authoritative serial execution
//! share this channel. Tensor payloads and collectives stay on the GPU.
//!
//! Framing: u32 little-endian length, then JSON. One request, one response,
//! per exchange - no pipelining, no fragmentation. Every buffer is capped
//! ([`MAX_FRAME`]) so a confused peer cannot exhaust memory.

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpStream;

/// Hard cap on a single control frame. Bootstrap messages are tiny; the
/// future execution vocabulary carries batch descriptions, not tensors.
pub const MAX_FRAME: u32 = 1 << 20; // 1 MiB

/// Errors from the control protocol.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("io on the control channel: {0}")]
    Io(#[from] std::io::Error),
    #[error("control frame of {0} bytes exceeds the {MAX_FRAME}-byte cap")]
    FrameTooLarge(u32),
    #[error("peer sent malformed control JSON: {0}")]
    Malformed(#[from] serde_json::Error),
    #[error("control channel closed before a complete frame arrived")]
    Closed,
    #[error("peer rejected this rank: {0}")]
    Rejected(String),
    #[error("NCCL unique ID must be 128 bytes, got {0}")]
    BadNcclId(usize),
}

/// NCCL's fixed-size opaque unique ID, carried only over the bootstrap TCP
/// connection. This control-plane type has no CUDA or NCCL dependency.
pub const NCCL_ID_BYTES: usize = 128;

pub fn send_nccl_id(stream: &mut TcpStream, id: &[u8; NCCL_ID_BYTES]) -> Result<(), ProtocolError> {
    ControlMessage::NcclId { id: id.to_vec() }.to_stream(stream)
}

pub fn receive_nccl_id(stream: &mut TcpStream) -> Result<[u8; NCCL_ID_BYTES], ProtocolError> {
    match ControlMessage::from_stream(stream)? {
        ControlMessage::NcclId { id } => {
            let len = id.len();
            id.try_into().map_err(|_| ProtocolError::BadNcclId(len))
        }
        ControlMessage::Reject { reason } => Err(ProtocolError::Rejected(reason)),
        other => Err(ProtocolError::Rejected(format!(
            "expected NCCL unique ID after handshake, got {other:?}"
        ))),
    }
}

/// The wire form of a span finisher's sampling plan. `paddock-dist` depends
/// on no engine crate (the standing Phase 2 architectural rule), so rank 0
/// converts its engine `DevicePlan` to this enum and the worker converts
/// back. Only plans the TP rank's device sampler can execute appear here;
/// anything else arrives as `Host` (full-logit finish).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TpSpanFinisherPlan {
    /// pure argmax
    Greedy,
    /// temperature-only categorical
    Categorical { inv_t: f32, u: f32 },
}

/// A control-plane message. Bootstrap messages only, per the phase scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    /// Rank 1 -> rank 0, first message after connect.
    Hello {
        /// Protocol version, for future-gating. 1 = this wire format.
        version: u32,
        /// The world size the worker was configured with. Must match the
        /// coordinator's or the handshake fails (mismatched world-size gate).
        tp_size: usize,
        /// Worker identity: hostname, for logs. Not used for routing.
        who: String,
    },
    /// Rank 0 -> rank 1, reply to Hello. Accepts the worker into the group.
    Welcome {
        /// Echoed so the worker can verify it joined the world it configured.
        tp_size: usize,
        /// Coordinator-assigned group/session id, for log correlation.
        session: u64,
    },
    /// Rank 0 -> rank 1, reply to Hello. Refuses the connection.
    Reject {
        /// Human-readable reason (world-size mismatch, version mismatch...).
        reason: String,
    },
    /// Either direction, typically rank 0 -> rank 1 on shutdown.
    Shutdown {
        /// True when the shutdown is because the peer asked for it, false on
        /// error paths, so the worker knows how to exit (0 vs nonzero).
        graceful: bool,
    },
    /// Phase 3 bootstrap ID. Only control messages travel on this socket;
    /// collectives use the NCCL data path.
    NcclId {
        id: Vec<u8>,
    },
    /// Rank 0 starts Phase 9 execution after both checkpoint and pack hashes
    /// are checked against the worker's local files.
    TpInit {
        checkpoint_sha256: String,
        pack_blake3: String,
        max_ctx: usize,
        slots: usize,
    },
    /// Rank-0-authorized ordered active rows; holes are omitted.
    TpBatch {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        kv_events: Vec<serde_json::Value>,
    },
    /// Mixed decode+chunked-prefill tick: like `TpBatch` but rows may exceed
    /// the slot count (a prompt chunk advances multiple rows per tick alongside
    /// the decode rows). Same execution contract: the worker mirrors every KV
    /// event, then runs one eager forward per row in order (no logits, no
    /// sampling - rank 0 owns both).
    TpMixed {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        kv_events: Vec<serde_json::Value>,
    },
    /// Start an ordered device-feedback decode segment with host tokens.
    TpPipeBegin {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        kv_events: Vec<serde_json::Value>,
    },
    /// Next tick consumes rank-0's previous device IDs via NCCL broadcast.
    TpPipeNext {
        sequence: u64,
        rows: Vec<(usize, usize)>,
        source_plane: usize,
        next_plane: usize,
        kv_events: Vec<serde_json::Value>,
    },
    /// Fence the final tick on both ranks before slot release or reuse.
    TpPipeDrain {
        sequence: u64,
    },
    /// Rank-0-authorized chunked prefill span on the prefill lane: rows are
    /// ordered `(slot, token, position)` prompt steps validated against the
    /// coordinator's mirrored KV. Chunks of the SAME slot must be contiguous
    /// and ascending by position; each finishing chunk's last row may device-
    /// sample or read logits back. The wire finisher plan is a plain enum
    /// because paddock-dist depends on no engine crate: rank 0 converts its
    /// `DevicePlan` to this, the worker converts back.
    TpSpanLaunch {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        /// `(slot, plan)` per finishing chunk, in chunk order; `None` = that
        /// chunk reads full logits for the host sampler. Rank 1 promotes
        /// exactly these slots' lane-local state at the span finish.
        finishers: Vec<(usize, Option<TpSpanFinisherPlan>)>,
        kv_events: Vec<serde_json::Value>,
    },
    /// Fence the in-flight span: join both lanes, promote each finished
    /// slot's lane-local state lane->decode (both ranks, own executors), and
    /// read the finisher result on rank 0 before any release/reset/reuse.
    TpSpanFinish {
        sequence: u64,
    },
    /// Release completed/cancelled slots before the next admission.
    TpRelease {
        sequence: u64,
        slots: Vec<usize>,
        kv_events: Vec<serde_json::Value>,
    },
    /// Worker has mirrored and validated a step's logical KV operation. Rank 0
    /// must see this before launching any collective for the step.
    TpPrepared {
        sequence: u64,
    },
    /// Acknowledgement of model load or a completed reset/step.
    TpReady {
        sequence: u64,
    },
    /// Rank 0 owns the logical KV lifecycle. The worker mirrors this event
    /// before executing; it must never allocate blocks independently.
    TpReset {
        sequence: u64,
        kv_event: serde_json::Value,
    },
    TpStep {
        sequence: u64,
        token: u32,
        position: usize,
        kv_event: serde_json::Value,
    },
    TpError {
        reason: String,
    },
}

impl ControlMessage {
    /// Serialize into a length-prefixed frame.
    pub fn to_frame(&self) -> Result<Vec<u8>, ProtocolError> {
        let mut json = serde_json::to_vec(self)?;
        let len = json.len() as u32;
        if len > MAX_FRAME {
            return Err(ProtocolError::FrameTooLarge(len));
        }
        let mut frame = Vec::with_capacity(4 + json.len());
        frame.extend_from_slice(&len.to_le_bytes());
        frame.append(&mut json);
        Ok(frame)
    }

    /// Read exactly one frame and deserialize it.
    pub fn from_stream(stream: &mut TcpStream) -> Result<Self, ProtocolError> {
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf);
        if len > MAX_FRAME {
            return Err(ProtocolError::FrameTooLarge(len));
        }
        if len == 0 {
            return Err(ProtocolError::Closed);
        }
        let mut json = vec![0u8; len as usize];
        stream.read_exact(&mut json)?;
        Ok(serde_json::from_slice(&json)?)
    }

    /// Write one frame and flush.
    pub fn to_stream(&self, stream: &mut TcpStream) -> Result<(), ProtocolError> {
        stream.write_all(&self.to_frame()?)?;
        stream.flush()?;
        Ok(())
    }
}

/// One handshake exchange: worker sends Hello, reads the verdict. Returns
/// the session id on acceptance, or the rejection reason as an error.
pub fn handshake(stream: &mut TcpStream, tp_size: usize, who: &str) -> Result<u64, ProtocolError> {
    ControlMessage::Hello {
        version: PROTOCOL_VERSION,
        tp_size,
        who: who.to_string(),
    }
    .to_stream(stream)?;
    match ControlMessage::from_stream(stream)? {
        ControlMessage::Welcome {
            tp_size: echoed,
            session,
        } => {
            if echoed != tp_size {
                return Err(ProtocolError::Rejected(format!(
                    "coordinator runs tp_size={echoed}, we configured {tp_size}"
                )));
            }
            Ok(session)
        }
        ControlMessage::Reject { reason } => Err(ProtocolError::Rejected(reason)),
        other => Err(ProtocolError::Rejected(format!(
            "unexpected handshake reply {other:?}"
        ))),
    }
}

/// Wire format version. Bumped when the control vocabulary changes shape.
pub const PROTOCOL_VERSION: u32 = 1;
