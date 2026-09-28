//! The rank-0 <-> worker control protocol: length-prefixed JSON over TCP.
//!
//! Bootstrap, model identity and TP serving execution share this channel.
//! Tensor payloads and collectives stay on the GPU.
//!
//! Framing: u32 little-endian length, then JSON. Each exchange uses bounded
//! frames; the decode pipe may defer a Ready until the next Prepared.
//! Every buffer is capped ([`MAX_FRAME`]) so a confused peer cannot exhaust
//! memory.

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpStream;

/// Hard cap on a single control frame. Bootstrap messages are tiny;
/// serving commands carry bounded row lists and KV snapshots, not tensors.
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

/// Control-plane messages for bootstrap and rank-0-authoritative TP serving.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    /// Worker -> rank 0, first message after connect.
    Hello {
        /// Protocol version.
        version: u32,
        /// The world size the worker was configured with.
        tp_size: usize,
        /// The worker's configured rank. Must be in 1..tp_size.
        rank: usize,
        /// Worker identity: hostname, for logs only.
        who: String,
    },
    /// Rank 0 -> worker, reply to Hello. Accepts this rank into the group.
    Welcome {
        /// Echoed so the worker can verify the world it joined.
        tp_size: usize,
        /// Echoed worker rank; protects against cross-wired worker sockets.
        rank: usize,
        /// Coordinator-assigned group/session id, shared by the TP world.
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
        /// KV cache dtype for BOTH ranks (Phase 12 per-layer KV types: the
        /// TP=2 path serves fp8-e4m3 KV where the device supports it, like
        /// TP=1). `"fp16"` or `"fp8_e4m3"`; anything else fails rank 1's
        /// identity check closed before NCCL init. Rank 0 resolves the
        /// runner's `PADDOCK_KV_CACHE_DTYPE` gate (including its sm_89
        /// device check) and sends the resolved value, so the ranks can
        /// never disagree.
        kv_dtype: String,
        /// CUDA-graph mode for BOTH ranks (upstream-readiness I4). Rank 0
        /// resolves `PADDOCK_TP_GRAPH` on the coordinator and sends the
        /// resolved value; rank 1 must not read that variable itself, so a
        /// hand-started remote worker cannot disagree with rank 0 about
        /// graphed/eager sequencing (a mispair hangs the collectives).
        use_graphs: bool,
        /// DeltaNet checkpoint-pool slots per layer for the mirrored prefix
        /// cache (v4). Rank 0 resolves the capacity (default 4; 0 disables
        /// resume entirely) and both ranks size their state pools + radix
        /// free-lists from THIS value, so the pools can never disagree.
        ckpt_slots: u32,
        /// The coordinator-resolved TP prefill span row cap (v6): the
        /// maximum contiguous prompt rows one whole-model span advance
        /// processes. Both ranks chunk every run from THIS value (the
        /// worker never reads `PADDOCK_TP_SPAN_CAP` itself), so a
        /// hand-started remote worker cannot pair a different span geometry
        /// and desynchronize the collectives.
        span_cap: usize,
    },
    /// Rank-0-authorized ordered active rows; holes are omitted.
    ///
    /// KV mirror transport (upstream-readiness B2): every mutating command
    /// carries the ordered rows - each row IS its `Ensure { slot, position }`
    /// logical operation - plus ONE end-of-tick `Snapshot` (the coordinator's
    /// logical state after the whole tick). The worker applies the ordered
    /// operations on its mirror, then requires the resulting state to equal
    /// the snapshot; any divergence fails closed before GPU work. One
    /// snapshot per tick bounds every frame at O(context), where the old
    /// one-full-snapshot-per-row design grew O(rows x context) and overflowed
    /// the 1 MiB frame cap for prefill spans beyond ~250 rows at 16k context.
    TpBatch {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        kv_state: serde_json::Value,
    },
    /// Mixed decode+chunked-prefill tick: like `TpBatch` but rows may exceed
    /// the slot count (a prompt chunk advances multiple rows per tick alongside
    /// the decode rows). Same execution contract: the worker applies the
    /// ordered row operations and validates the end-of-tick snapshot, then
    /// executes the rows in order - decode rows as one-token forwards, the
    /// trailing chunk run (the last `chunk_rows` entries, which MUST be the
    /// non-decode rows in wire order) as batched span prefill (no logits, no
    /// sampling - rank 0 owns both).
    TpMixed {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        /// Length of the trailing chunk run inside `rows` (wire order):
        /// rows[..rows.len()-chunk_rows] are decode rows, the rest prompt
        /// rows executed as one or more batched spans. v3.
        chunk_rows: usize,
        /// Checkpoint cuts claimed when a cold prompt first takes Mixed
        /// ownership. Empty on later ticks and every Async-owned prompt.
        reserve_cuts: Vec<usize>,
        /// Prefix-cache checkpoint cuts the chunk run's spans end exactly on,
        /// `(cut position, rank-0 pool index)` in cut order (v4). The worker
        /// snapshots its own DeltaNet state at those span boundaries into its
        /// own mirror-deterministic pool index; the wire indices are for
        /// validation only (the worker attaches ITS index at publish).
        /// Empty on every non-prefill tick.
        ckpts: Vec<(usize, u32)>,
        kv_state: serde_json::Value,
    },
    /// Start an ordered device-feedback decode segment with host tokens.
    TpPipeBegin {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        kv_state: serde_json::Value,
    },
    /// Next tick consumes rank-0's previous device IDs via NCCL broadcast.
    TpPipeNext {
        sequence: u64,
        rows: Vec<(usize, usize)>,
        source_plane: usize,
        next_plane: usize,
        kv_state: serde_json::Value,
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
    /// `DevicePlan` to this, the worker converts back. `kv_state` is the
    /// end-of-tick mirror snapshot (see TpBatch).
    TpSpanLaunch {
        sequence: u64,
        rows: Vec<(usize, u32, usize)>,
        /// `(slot, plan)` per finishing chunk, in chunk order; `None` = that
        /// chunk reads full logits for the host sampler. Rank 1 promotes
        /// exactly these slots' lane-local state at the span finish.
        finishers: Vec<(usize, Option<TpSpanFinisherPlan>)>,
        /// Prefix-cache checkpoint cuts this launch's spans end exactly on,
        /// `(cut position, rank-0 pool index)` (v4). See TpMixed.ckpts.
        ckpts: Vec<(usize, u32)>,
        kv_state: serde_json::Value,
    },
    /// Fence the in-flight span: join both lanes, promote each finished
    /// slot's lane-local state lane->decode (both ranks, own executors), and
    /// read the finisher result on rank 0 before any release/reset/reuse.
    TpSpanFinish {
        sequence: u64,
    },
    /// Release completed/cancelled slots before the next admission. Each
    /// freed slot is its `Release { slot }` logical operation; `kv_state`
    /// is the end-of-tick mirror snapshot (see TpBatch).
    TpRelease {
        sequence: u64,
        slots: Vec<usize>,
        kv_state: serde_json::Value,
    },
    /// Prefix-cache admission (v4): the coordinator's exact logical resume
    /// decision for `slot`. `tokens` is the FULL prompt (sent once, at
    /// admission only); `resume` is the block-aligned position to start
    /// prefill at (0 = cold). The worker mirrors `Operation::Admit` - which
    /// independently validates the cached chain AND its checkpoint at exactly
    /// `resume` on the worker's own radix - BEFORE acknowledging Prepared, so
    /// a rank that cannot satisfy the decision fails closed instead of
    /// resuming alone. No physical page ids travel: each rank resolves the
    /// logical prefix to its own (mirror-identical) physical blocks.
    TpPrefixAdmit {
        sequence: u64,
        slot: usize,
        tokens: Vec<u32>,
        resume: usize,
        /// Ordered cuts reserved by rank 0; rank 1 reserves the same cuts
        /// before comparing the mirrored logical state.
        cuts: Vec<usize>,
        kv_state: serde_json::Value,
    },
    /// Prefix-cache publication (v4): attach the checkpoint indices the
    /// prefill snapshotted (in cut order), publish the prompt's full pages,
    /// then recycle any reservation that was not attached. `kv_state` is the
    /// end-of-tick mirror snapshot. Both ranks run the publication only
    /// after their rank-local GPU snapshot succeeded, so an attached
    /// checkpoint always has real state behind it on every rank.
    TpPrefixPublish {
        sequence: u64,
        slot: usize,
        /// `(cut position, checkpoint index)` in ascending cut order.
        checkpoints: Vec<(usize, u32)>,
        kv_state: serde_json::Value,
    },
    /// Worker has mirrored and validated a tick's logical KV operations
    /// (ordered rows applied, end-of-tick snapshot matched). Rank 0 must see
    /// this before launching any collective for the tick.
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
pub fn handshake(
    stream: &mut TcpStream,
    tp_size: usize,
    who: &str,
) -> Result<u64, ProtocolError> {
    handshake_rank(stream, tp_size, 1, who)
}

/// Rank-aware handshake for worlds larger than two.
pub fn handshake_rank(
    stream: &mut TcpStream,
    tp_size: usize,
    rank: usize,
    who: &str,
) -> Result<u64, ProtocolError> {
    ControlMessage::Hello {
        version: PROTOCOL_VERSION,
        tp_size,
        rank,
        who: who.to_string(),
    }
    .to_stream(stream)?;
    match ControlMessage::from_stream(stream)? {
        ControlMessage::Welcome {
            tp_size: echoed,
            rank: echoed_rank,
            session,
        } => {
            if echoed != tp_size || echoed_rank != rank {
                return Err(ProtocolError::Rejected(format!(
                    "coordinator welcomed tp_size={echoed} rank={echoed_rank}, we configured tp_size={tp_size} rank={rank}"
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
///
/// Version 2: mutating commands carry one end-of-tick KV mirror snapshot
/// instead of one full-snapshot event per row (B2: bounded frames), and
/// `TpInit` carries the coordinator-resolved CUDA-graph mode (I4).
///
/// Version 3: `TpMixed` carries `chunk_rows` (the trailing prompt-run length
/// the worker must execute as batched span prefill), keeping both ranks'
/// span geometry host-derived from one wire value.
///
/// Version 4: prefix-cache resume. `TpInit` carries the coordinator-resolved
/// DeltaNet checkpoint-pool capacity (`ckpt_slots`); `TpPrefixAdmit` carries
/// the full prompt tokens plus the coordinator's block-aligned resume
/// decision (validated independently on both ranks inside `Operation::Admit`
/// before either rank adopts); `TpPrefixPublish` attaches the post-prefill
/// checkpoint indices and publishes the full pages. Logical identity only -
/// no physical page ids ever travel.
///
/// Version 5: `TpMixed.reserve_cuts` claims cold-prompt checkpoints only on
/// first Mixed ownership, before that tick's KV Ensure operations.
///
/// Version 6: `TpInit` carries the coordinator-resolved TP prefill span row
/// cap (`span_cap`).
///
/// Version 7: Hello/Welcome carry the worker rank so one coordinator can
/// identify and validate every rank in worlds larger than two.
pub const PROTOCOL_VERSION: u32 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp_mixed_chunk_rows_roundtrip_and_missing_field_fails_closed() {
        let msg = ControlMessage::TpMixed {
            sequence: 7,
            rows: vec![(0, 11, 0), (0, 12, 1), (1, 22, 5)],
            chunk_rows: 2,
            reserve_cuts: vec![64],
            ckpts: vec![(64, 0)],
            kv_state: serde_json::json!({"tables": []}),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ControlMessage = serde_json::from_str(&json).unwrap();
        match back {
            ControlMessage::TpMixed {
                sequence,
                rows,
                chunk_rows,
                ..
            } => {
                assert_eq!(sequence, 7);
                assert_eq!(rows.len(), 3);
                assert_eq!(chunk_rows, 2);
            }
            other => panic!("wrong message: {other:?}"),
        }
        let mut missing: serde_json::Value = serde_json::from_str(&json).unwrap();
        missing.as_object_mut().unwrap().remove("chunk_rows");
        assert!(serde_json::from_value::<ControlMessage>(missing).is_err());
        let mut missing: serde_json::Value = serde_json::from_str(&json).unwrap();
        missing.as_object_mut().unwrap().remove("reserve_cuts");
        assert!(serde_json::from_value::<ControlMessage>(missing).is_err());
    }

    #[test]
    fn protocol_version_is_six() {
        assert_eq!(PROTOCOL_VERSION, 6);
    }

    #[test]
    fn tp_init_carries_the_span_cap_roundtrip() {
        let msg = ControlMessage::TpInit {
            checkpoint_sha256: "a".repeat(64),
            pack_blake3: "b".repeat(64),
            max_ctx: 65536,
            slots: 2,
            kv_dtype: "fp16".to_owned(),
            use_graphs: false,
            ckpt_slots: 4,
            span_cap: 128,
        };
        let json = serde_json::to_string(&msg).unwrap();
        match serde_json::from_str(&json).unwrap() {
            ControlMessage::TpInit { span_cap, .. } => assert_eq!(span_cap, 128),
            other => panic!("wrong message: {other:?}"),
        }
        // The field is mandatory: dropping it must fail closed (a v5 frame
        // must never silently default the cap on one rank only).
        let mut missing = serde_json::to_value(&msg).unwrap();
        missing.as_object_mut().unwrap().remove("span_cap");
        assert!(serde_json::from_value::<ControlMessage>(missing).is_err());
    }

    #[test]
    fn tp_prefix_admit_and_publish_roundtrip() {
        let admit = ControlMessage::TpPrefixAdmit {
            sequence: 9,
            slot: 1,
            tokens: vec![5, 6, 7],
            resume: 32,
            cuts: vec![],
            kv_state: serde_json::json!({"tables": []}),
        };
        let json = serde_json::to_string(&admit).unwrap();
        match serde_json::from_str(&json).unwrap() {
            ControlMessage::TpPrefixAdmit {
                sequence,
                slot,
                tokens,
                resume,
                ..
            } => {
                assert_eq!((sequence, slot, tokens, resume), (9, 1, vec![5, 6, 7], 32));
            }
            other => panic!("wrong message: {other:?}"),
        }
        let publish = ControlMessage::TpPrefixPublish {
            sequence: 10,
            slot: 1,
            checkpoints: vec![(512, 0), (1024, 1)],
            kv_state: serde_json::json!({"tables": []}),
        };
        let json = serde_json::to_string(&publish).unwrap();
        match serde_json::from_str(&json).unwrap() {
            ControlMessage::TpPrefixPublish {
                sequence,
                slot,
                checkpoints,
                ..
            } => {
                assert_eq!((sequence, slot), (10, 1));
                assert_eq!(checkpoints, vec![(512, 0), (1024, 1)]);
            }
            other => panic!("wrong message: {other:?}"),
        }
    }
}
