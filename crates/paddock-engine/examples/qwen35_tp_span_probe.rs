//! Batched-TP-prefill-span prototype probe (two ranks over the control
//! channel). Worker first, then head:
//! qwen35_tp_span_probe RANK MASTER_IP MODEL PACK [PORT] [SPANS]
//!
//! For each case the probe compares, on identical fresh initial state per arm:
//!   A. the existing serial one-row TP prefill (`forward_token_slot` per row)
//!   B. the new batched span prefill (`forward_prefill_span`)
//! comparing the final-row logits within the two-slot oracle tolerance, plus
//! wall time per arm, speedup, and effective tokens/second. Cases cover
//! 1/16/64-row spans, ~1k prompt rows as repeated 64-row spans, and a span
//! sequence crossing 16-token KV page boundaries, each followed (where the
//! case includes one) by a decode row that cross-checks post-span KV and
//! DeltaNet state through a continuation forward.
//!
//! DeltaNet state and KV state are not read back directly: both arms advance
//! the same slot's recurrent/conv/paged-KV state through the same token
//! stream, so the decode-continuation logits comparison IS the state check
//! (any state divergence propagates into it). Rank 1 additionally replays
//! both arms in order, which exercises the paired-collective discipline of
//! the new span path against the worker protocol.
//!
//! No production serving path is touched; this is a standalone probe.
use paddock_dist::{
    config::{ParallelConfig, Resolved},
    protocol::{ControlMessage, receive_nccl_id, send_nccl_id},
    worker::{connect_worker, coordinate, shutdown_worker},
};
use paddock_engine::{
    gpu::distributed::{NcclCommunicator, create_unique_id},
    gpu::{GpuExecutor, KvDtype},
    gpu_model::qwen35::{
        Qwen35TpRank,
        tp_kv::{Event, MirroredKv, Operation},
    },
};
use paddock_models::mapped::MappedGguf;
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    io::{Read, Write},
    net::TcpStream,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

const MAX_CTX: usize = 256;
const SLOTS: usize = 1;
// The two-slot TP oracle's accepted tolerance (identical numeric classes).
const ABS: f32 = 1e-2;
const REL: f32 = 1e-3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Action {
    /// One span of `rows` fresh prompt tokens starting at the running position.
    Span { rows: usize },
    /// One decode row continuing after the spans.
    Decode { token: u32, position: usize },
}

#[derive(Serialize, Deserialize)]
struct Packet {
    sequence: usize,
    action: Action,
    event: Event,
}

fn cases(spans: usize) -> Vec<(String, Vec<Action>)> {
    let mut out = Vec::new();
    for rows in [1usize, 16, 64] {
        out.push((
            format!("{rows}row"),
            vec![
                Action::Span { rows },
                Action::Decode {
                    token: 42,
                    position: rows,
                },
            ],
        ));
    }
    // ~1k prompt rows as repeated 64-row spans plus one decode row on top.
    let mut many = vec![Action::Span { rows: 64 }; spans];
    many.push(Action::Decode {
        token: 42,
        position: 64 * spans,
    });
    out.push((format!("64x{spans}+decode"), many));
    // Page-boundary crossing: BLOCK_TOKENS is 16, so a span starting at an
    // odd offset lands its tail across a page edge inside ONE span.
    out.push((
        "page-cross+decode".into(),
        vec![
            Action::Span { rows: 16 }, // positions 0..16
            Action::Span { rows: 13 }, // 16..29: tail crosses the 16/32 boundary
            Action::Decode {
                token: 7,
                position: 29,
            },
            Action::Span { rows: 21 }, // 30..51: crosses the 32/48 boundary mid-span
            Action::Decode {
                token: 9,
                position: 51,
            },
        ],
    ));
    out
}

fn operation(action: &Action) -> Operation {
    match action {
        // The coordinator's Ensure covers the span's LAST position; the pool
        // backing it covers every earlier row of the contiguous span.
        Action::Span { rows } => Operation::Ensure {
            slot: 0,
            position: rows.saturating_sub(1),
        },
        Action::Decode { position, .. } => Operation::Ensure {
            slot: 0,
            position: *position,
        },
    }
}

fn send_packet(stream: &mut TcpStream, packet: &Packet) -> Result<(), Box<dyn Error>> {
    let bytes = serde_json::to_vec(packet)?;
    if bytes.len() > 65536 {
        return Err("probe packet exceeds cap".into());
    }
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

fn recv_packet(stream: &mut TcpStream) -> Result<Packet, Box<dyn Error>> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len)?;
    let size = u32::from_le_bytes(len) as usize;
    if size == 0 || size > 65536 {
        return Err("probe packet invalid length".into());
    }
    let mut bytes = vec![0u8; size];
    stream.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn ack(stream: &mut TcpStream, rank: usize) -> Result<(), Box<dyn Error>> {
    if rank == 0 {
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte)?;
        if byte != [0x42] {
            return Err("worker acknowledgement mismatch".into());
        }
    } else {
        stream.write_all(&[0x42])?;
    }
    Ok(())
}

fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as u32
}

fn compare(a: &[f32], b: &[f32], label: &str) -> Result<(f32, bool), Box<dyn Error>> {
    if a.len() != b.len() {
        return Err(format!("{label}: vocab mismatch").into());
    }
    let mut max_abs = 0.0f32;
    let mut ok = true;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(format!("{label}: nonfinite logit {i}").into());
        }
        let delta = (x - y).abs();
        max_abs = max_abs.max(delta);
        if delta > ABS + REL * y.abs() {
            ok = false;
        }
    }
    Ok((max_abs, ok))
}

/// Deterministic pseudo-prompt tokens (the probe compares arms against each
/// other, not against English text).
fn prompt_token(i: usize) -> u32 {
    (i * 7919 % 32000) as u32
}

/// Run `actions` through one arm on a freshly reset model. `batched` selects
/// the whole-model span traversal (true) or the serial one-row prefill
/// (false). Returns the last span's final-row logits, the decode-continuation
/// logits (when the case has a decode row) and the wall time of the span work.
#[allow(clippy::too_many_arguments)]
fn run_arm(
    rank: usize,
    control: &mut TcpStream,
    tp: &mut Qwen35TpRank,
    kv: &mut MirroredKv,
    exec: &GpuExecutor,
    group: &NcclCommunicator,
    actions: &[Action],
    batched: bool,
) -> Result<(Vec<f32>, Option<Vec<f32>>, Duration), Box<dyn Error>> {
    tp.reset()?;
    exec.synchronize()?;
    let mut position = 0usize;
    let mut last_span_logits: Option<Vec<f32>> = None;
    let mut decode_logits: Option<Vec<f32>> = None;
    let started = Instant::now();
    for (sequence, action) in actions.iter().enumerate() {
        let packet = if rank == 0 {
            let event = kv.authorize(operation(action))?;
            let packet = Packet {
                sequence,
                action: action.clone(),
                event,
            };
            send_packet(control, &packet)?;
            packet
        } else {
            let packet = recv_packet(control)?;
            if packet.sequence != sequence || packet.action != *action {
                return Err(format!("sequence={sequence}: command mismatch").into());
            }
            kv.mirror(&packet.event)?;
            packet
        };
        ack(control, rank)?;
        match &packet.action {
            Action::Span { rows } => {
                if batched {
                    let tokens: Vec<u32> = (0..*rows).map(|i| prompt_token(position + i)).collect();
                    let logits = tp.forward_prefill_span(group, kv, 0, &tokens, position)?;
                    if rank == 0 {
                        last_span_logits = Some(logits);
                    }
                } else {
                    let mut logits = Vec::new();
                    for i in 0..*rows {
                        logits = tp.forward_token_slot(
                            group,
                            kv,
                            prompt_token(position + i),
                            position + i,
                            0,
                        )?;
                    }
                    if rank == 0 {
                        last_span_logits = Some(logits);
                    }
                }
                position += rows;
            }
            Action::Decode {
                token,
                position: pos,
            } => {
                // Both arms advance continuation state identically: the same
                // token through the serial decode path on top of the span.
                let logits = tp.forward_token_slot(group, kv, *token, *pos, 0)?;
                if rank == 0 {
                    decode_logits = Some(logits);
                }
                position = pos + 1;
            }
        }
    }
    let wall = started.elapsed();
    let span_out = last_span_logits.ok_or("case produced no span logits")?;
    Ok((span_out, decode_logits, wall))
}

fn run(
    rank: usize,
    control: &mut TcpStream,
    resolved: &Resolved,
    model: &Path,
    pack: &Path,
    spans: usize,
) -> Result<(), Box<dyn Error>> {
    let id = if rank == 0 {
        let id = create_unique_id()?;
        send_nccl_id(control, &id)?;
        id
    } else {
        receive_nccl_id(control)?
    };
    let exec = Arc::new(GpuExecutor::new(0, pack)?);
    let group = NcclCommunicator::from_resolved(Some(resolved), exec.stream.context(), id)?
        .ok_or("TP group absent")?;
    let map = MappedGguf::open(model)?;
    let mut tp =
        Qwen35TpRank::load_slots(exec.clone(), &map, &group, MAX_CTX, KvDtype::Fp16, SLOTS)?;
    let blocks = u32::try_from(MAX_CTX.div_ceil(paddock_engine::kv_pool::BLOCK_TOKENS) * SLOTS)?;
    let mut kv = MirroredKv::new(blocks, SLOTS, MAX_CTX)?;
    if rank == 0 {
        let mut violations = Vec::new();
        for (name, actions) in cases(spans) {
            let (serial_span, serial_dec, serial_wall) = run_arm(
                rank, control, &mut tp, &mut kv, &exec, &group, &actions, false,
            )?;
            let (batch_span, batch_dec, batch_wall) = run_arm(
                rank, control, &mut tp, &mut kv, &exec, &group, &actions, true,
            )?;
            let label = format!("case={name}");
            let (max_abs, ok) = compare(&batch_span, &serial_span, &label)?;
            let rows_total = actions
                .iter()
                .map(|a| match a {
                    Action::Span { rows } => *rows,
                    Action::Decode { .. } => 1,
                })
                .sum::<usize>();
            let dec_note = match (&serial_dec, &batch_dec) {
                (Some(s), Some(b)) => {
                    let (dmax, dok) = compare(b, s, &format!("{label} decode"))?;
                    let (gs, gb) = (greedy(s), greedy(b));
                    format!(
                        " decode_max_abs={dmax:.6} decode_ok={dok} decode_greedy serial={gs} batched={gb}"
                    )
                }
                _ => String::new(),
            };
            let status = if ok { "OK" } else { "VIOLATION" };
            println!(
                "span_probe {status} {label} rows={rows_total} max_abs={max_abs:.8} \
                 serial_wall={serial_wall:.1?} batched_wall={batch_wall:.1?} \
                 speedup={:.2} tok_s_serial={:.0} tok_s_batched={:.0}{dec_note}",
                serial_wall.as_secs_f64() / batch_wall.as_secs_f64(),
                rows_total as f64 / serial_wall.as_secs_f64(),
                rows_total as f64 / batch_wall.as_secs_f64(),
            );
            if !ok {
                violations.push(label);
            }
        }
        if !violations.is_empty() {
            return Err(format!(
                "{} span tolerance violations; first: {}",
                violations.len(),
                violations[0]
            )
            .into());
        }
    } else {
        // Rank 1 replays both arms in the same order; its forwards pair the
        // collectives either way, which is itself part of what the probe
        // exercises (a rank-desynced span forward hangs or errors here).
        for (_, actions) in cases(spans) {
            let _ = run_arm(
                rank, control, &mut tp, &mut kv, &exec, &group, &actions, false,
            )?;
            let _ = run_arm(
                rank, control, &mut tp, &mut kv, &exec, &group, &actions, true,
            )?;
        }
    }
    group.stream().synchronize()?;
    exec.synchronize()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_tp_span_probe RANK MASTER_IP MODEL PACK [PORT] [SPANS]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank must be 0 or 1".into());
    }
    let port: u16 = args.get(5).map_or(Ok(11566), |s| s.parse())?;
    let spans: usize = args.get(6).map_or(Ok(16), |s| s.parse())?;
    let resolved = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some(args[2].clone()),
        master_port: Some(port),
    }
    .resolved(false)?
    .ok_or("expected TP=2")?;
    let (mut control, _) = if rank == 0 {
        coordinate(&resolved, false)?
    } else {
        connect_worker(&resolved)?
    };
    control.set_read_timeout(Some(Duration::from_secs(600)))?;
    control.set_write_timeout(Some(Duration::from_secs(600)))?;
    let result = run(
        rank,
        &mut control,
        &resolved,
        Path::new(&args[3]),
        Path::new(&args[4]),
        spans,
    );
    if rank == 0 {
        shutdown_worker(&mut control, result.is_ok())?;
    } else {
        match ControlMessage::from_stream(&mut control)? {
            ControlMessage::Shutdown { graceful } if graceful == result.is_ok() => {}
            other => return Err(format!("unexpected shutdown {other:?}").into()),
        }
    }
    result
}
