//! The rank0<->rank1 control protocol: length-prefixed JSON over TCP.
//!
//! Phase 2 scope is bootstrap coordination only - handshake, world-size
//! agreement, shutdown. The command vocabulary for execution control
//! (Prefill/Decode/ResetSlot/...) arrives with the distributed executor in a
//! later phase; the frame format here is chosen so that extension is a new
//! enum variant, not a wire change.
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
