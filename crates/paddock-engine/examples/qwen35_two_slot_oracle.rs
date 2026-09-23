//! Two-slot eager TP=2 correctness oracle. Worker first, then head:
//! qwen35_two_slot_oracle RANK MASTER_IP MODEL PACK [PORT]
//! This exercises rank-0-authorized row order, cancellation at a decode boundary,
//! survivor continuation, release/reuse, page crossing and exact reset replay.
use paddock_dist::{
    config::{ParallelConfig, Resolved},
    protocol::{ControlMessage, receive_nccl_id, send_nccl_id},
    worker::{connect_worker, coordinate, shutdown_worker},
};
use paddock_engine::{
    gpu::distributed::{NcclCommunicator, create_unique_id},
    gpu::{GpuExecutor, KvDtype},
    gpu_model::qwen35::{
        GpuQwen35, Qwen35TpRank,
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
    time::Duration,
};

const MAX_CTX: usize = 48;
const SLOTS: usize = 2;
// The independent single-slot TP=2 run reaches 0.00804 max absolute TP=1
// difference at position 13; exact interleaved-vs-isolated TP comparison below
// detects slot corruption without conflating it with accumulated TP rounding.
const ABS: f32 = 1e-2;
const REL: f32 = 1e-3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Action {
    Row {
        slot: usize,
        segment: usize,
        token: u32,
        position: usize,
    },
    Release {
        slot: usize,
    },
    Flush,
}

#[derive(Serialize, Deserialize)]
struct Packet {
    sequence: usize,
    action: Action,
    event: Event,
}

fn script() -> Vec<Action> {
    let mut actions = vec![Action::Flush];
    for (position, token) in [1, 2, 3].into_iter().enumerate() {
        actions.push(Action::Row {
            slot: 0,
            segment: 0,
            token,
            position,
        });
    }
    for (position, token) in [4, 5].into_iter().enumerate() {
        actions.push(Action::Row {
            slot: 1,
            segment: 1,
            token,
            position,
        });
    }
    for position in 3..=20 {
        actions.push(Action::Row {
            slot: 0,
            segment: 0,
            token: 100 + position as u32,
            position,
        });
        if (3..=8).contains(&position) {
            actions.push(Action::Row {
                slot: 1,
                segment: 1,
                token: 200 + position as u32,
                position: position - 1,
            });
        }
        if position == 9 {
            // The other slot just completed a decode row. Cancel slot 1 at
            // this execution boundary, while slot 0 still has pending work.
            actions.push(Action::Release { slot: 1 });
        }
        if (10..=15).contains(&position) {
            actions.push(Action::Row {
                slot: 1,
                segment: 2,
                token: 300 + position as u32,
                position: position - 10,
            });
        }
    }
    actions
}

fn operation(action: &Action) -> Operation {
    match action {
        Action::Row { slot, position, .. } => Operation::Ensure {
            slot: *slot,
            position: *position,
        },
        Action::Release { slot } => Operation::Release { slot: *slot },
        Action::Flush => Operation::Flush,
    }
}

fn send_packet(stream: &mut TcpStream, packet: &Packet) -> Result<(), Box<dyn Error>> {
    let bytes = serde_json::to_vec(packet)?;
    if bytes.len() > 65536 {
        return Err("oracle packet exceeds cap".into());
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
        return Err("oracle packet invalid length".into());
    }
    let mut bytes = vec![0; size];
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

fn greedy(logits: &[f32]) -> Result<u32, Box<dyn Error>> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite or empty logits".into());
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(u32::try_from(best)?)
}

fn compare(
    tp: &[f32],
    serial: &[f32],
    label: &str,
) -> Result<(f32, Option<String>), Box<dyn Error>> {
    if tp.len() != serial.len() {
        return Err(format!("{label}: vocab mismatch").into());
    }
    let mut max_abs = 0.0f32;
    let mut worst = (0.0f32, String::new());
    for (i, (&a, &b)) in tp.iter().zip(serial).enumerate() {
        let delta = (a - b).abs();
        max_abs = max_abs.max(delta);
        if !a.is_finite() || !b.is_finite() {
            return Err(format!("{label}: nonfinite logit {i} tp={a} serial={b}").into());
        }
        let ratio = delta / (ABS + REL * b.abs());
        if ratio > worst.0 {
            worst = (
                ratio,
                format!("{label}: logit {i} tp={a} serial={b} abs={delta} tolerance_ratio={ratio}"),
            );
        }
    }
    if greedy(tp)? != greedy(serial)? {
        return Err(format!(
            "{label}: greedy token differs: tp={} serial={}",
            greedy(tp)?,
            greedy(serial)?
        )
        .into());
    }
    Ok((max_abs, (worst.0 > 1.0).then_some(worst.1)))
}

fn rank_parity(
    stream: &mut TcpStream,
    rank: usize,
    logits: &[f32],
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if rank == 0 {
        let mut bytes = Vec::with_capacity(4 + logits.len() * 4);
        bytes.extend_from_slice(&u32::try_from(logits.len())?.to_le_bytes());
        for &value in logits {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        stream.write_all(&bytes)?;
        ack(stream, rank)?;
    } else {
        let mut size = [0u8; 4];
        stream.read_exact(&mut size)?;
        let n = u32::from_le_bytes(size) as usize;
        if n != logits.len() {
            return Err(format!("{label}: rank vocab mismatch").into());
        }
        let mut bytes = vec![0u8; n * 4];
        stream.read_exact(&mut bytes)?;
        for (i, (&value, word)) in logits.iter().zip(bytes.chunks_exact(4)).enumerate() {
            if value.to_bits() != u32::from_le_bytes(word.try_into()?) {
                return Err(format!("{label}: rank logit {i} mismatch").into());
            }
        }
        ack(stream, rank)?;
    }
    Ok(())
}

fn run(
    rank: usize,
    control: &mut TcpStream,
    resolved: &Resolved,
    model: &Path,
    pack: &Path,
) -> Result<(), Box<dyn Error>> {
    if std::env::var_os("PADDOCK_NO_SPEC").is_none() {
        return Err("set PADDOCK_NO_SPEC=1".into());
    }
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
    let mut serial = if rank == 0 {
        Some(GpuQwen35::load(exec.clone(), &map, MAX_CTX)?)
    } else {
        None
    };
    let blocks = u32::try_from(MAX_CTX.div_ceil(paddock_engine::kv_pool::BLOCK_TOKENS) * SLOTS)?;
    let mut kv = MirroredKv::new(blocks, SLOTS, MAX_CTX)?;
    let mut actions = script();
    if std::env::var_os("PADDOCK_ORACLE_SINGLE_SLOT").is_some() {
        actions.retain(|action| matches!(action, Action::Flush | Action::Row { slot: 0, .. }));
    }
    let row_count = actions
        .iter()
        .filter(|a| matches!(a, Action::Row { .. }))
        .count();
    let mut replay = Vec::<Vec<f32>>::with_capacity(row_count);
    let mut segments: [Vec<(u32, Vec<f32>)>; 3] = std::array::from_fn(|_| Vec::new());
    let mut positions = [0usize; SLOTS];
    let mut cancelled = false;
    let mut freed = false;
    for pass in 0..2 {
        for (sequence, expected) in actions.iter().enumerate() {
            let packet = if rank == 0 {
                let event = kv.authorize(operation(expected))?;
                let packet = Packet {
                    sequence,
                    action: expected.clone(),
                    event,
                };
                send_packet(control, &packet)?;
                packet
            } else {
                let packet = recv_packet(control)?;
                if packet.sequence != sequence
                    || packet.action != *expected
                    || packet.event.operation != operation(expected)
                {
                    return Err(format!("pass={pass} sequence={sequence}: command mismatch").into());
                }
                kv.mirror(&packet.event)?;
                packet
            };
            ack(control, rank)?; // worker has verified logical KV before NCCL
            match packet.action {
                Action::Flush => {
                    tp.reset()?;
                    positions.fill(0);
                    cancelled = false;
                    freed = false;
                    exec.synchronize()?;
                    ack(control, rank)?;
                }
                Action::Release { slot } => {
                    if slot != 1 || cancelled || positions[slot] == 0 || positions[0] < 10 {
                        return Err("release was not between survivor decode rows".into());
                    }
                    // Both ranks finish the prior row before freeing GPU payload.
                    exec.synchronize()?;
                    tp.reset_slot(slot)?;
                    exec.synchronize()?;
                    positions[slot] = 0;
                    cancelled = true;
                    freed = true;
                    ack(control, rank)?;
                }
                Action::Row {
                    slot,
                    segment,
                    token,
                    position,
                } => {
                    if slot >= SLOTS
                        || position != positions[slot]
                        || (segment == 1 && cancelled)
                        || (segment == 2 && !freed)
                        || (segment == 0 && slot != 0)
                        || (segment != 0 && slot != 1)
                    {
                        return Err(
                            format!("pass={pass} sequence={sequence}: row state diverged").into(),
                        );
                    }
                    let logits = tp.forward_token_slot(&group, &kv, token, position, slot)?;
                    if pass == 0 {
                        replay.push(logits.clone());
                        if rank == 0 {
                            segments[segment].push((token, logits.clone()));
                        }
                    } else {
                        let first = &replay[positions_for_row(&actions, sequence)];
                        if logits
                            .iter()
                            .map(|v| v.to_bits())
                            .ne(first.iter().map(|v| v.to_bits()))
                        {
                            return Err(format!(
                                "rank={rank} pass={pass} sequence={sequence}: replay differs"
                            )
                            .into());
                        }
                    }
                    let label = format!("pass={pass} seq={sequence} slot={slot} pos={position}");
                    rank_parity(control, rank, &logits, &label)?;
                    positions[slot] += 1;
                    if rank == 0 {
                        println!(
                            "{label} greedy={} exact_rank=true exact_replay={} checksum={:.6}",
                            greedy(&logits)?,
                            pass == 1,
                            logits.iter().sum::<f32>()
                        );
                    }
                }
            }
        }
    }
    // A third pass removes all slot-1 operations. Exact slot-0 logits must
    // equal the interleaved pass, including the page boundary and cancellation.
    let baseline: Vec<_> = actions
        .iter()
        .enumerate()
        .filter(|(_, action)| matches!(action, Action::Flush | Action::Row { slot: 0, .. }))
        .collect();
    for (sequence, (original_index, expected)) in baseline.into_iter().enumerate() {
        let packet = if rank == 0 {
            let event = kv.authorize(operation(expected))?;
            let packet = Packet {
                sequence,
                action: expected.clone(),
                event,
            };
            send_packet(control, &packet)?;
            packet
        } else {
            let packet = recv_packet(control)?;
            if packet.sequence != sequence
                || packet.action != *expected
                || packet.event.operation != operation(expected)
            {
                return Err(format!("baseline sequence={sequence}: command mismatch").into());
            }
            kv.mirror(&packet.event)?;
            packet
        };
        ack(control, rank)?;
        match packet.action {
            Action::Flush => {
                tp.reset()?;
                exec.synchronize()?;
                ack(control, rank)?;
            }
            Action::Row {
                slot,
                token,
                position,
                ..
            } => {
                let logits = tp.forward_token_slot(&group, &kv, token, position, slot)?;
                let first = &replay[positions_for_row(&actions, original_index)];
                if logits
                    .iter()
                    .map(|v| v.to_bits())
                    .ne(first.iter().map(|v| v.to_bits()))
                {
                    return Err(format!("rank={rank} baseline slot=0 position={position}: interleaving changed logits").into());
                }
                rank_parity(
                    control,
                    rank,
                    &logits,
                    &format!("baseline slot=0 position={position}"),
                )?;
            }
            Action::Release { .. } => unreachable!(),
        }
    }
    if rank == 0 {
        println!("slot-0 interleaved versus isolated: exact across 21 rows");
    }
    let mut violations = Vec::new();
    if let Some(s) = &mut serial {
        for (segment, trace) in segments.iter().enumerate() {
            s.reset();
            for (position, (token, tp_logits)) in trace.iter().enumerate() {
                let serial_logits = s.forward_one_no_graph(*token)?;
                let label = format!("segment={segment} position={position}");
                let (max_abs, violation) = compare(tp_logits, &serial_logits, &label)?;
                if let Some(message) = violation {
                    println!("tp1_oracle VIOLATION {message}");
                    violations.push(message);
                }
                println!(
                    "tp1_oracle {label} greedy={} max_abs={max_abs:.8}",
                    greedy(tp_logits)?
                );
            }
        }
    }
    if !violations.is_empty() {
        return Err(format!(
            "{} TP1 tolerance violations; first: {}",
            violations.len(),
            violations[0]
        )
        .into());
    }
    group.stream().synchronize()?;
    exec.synchronize()?;
    Ok(())
}

fn positions_for_row(actions: &[Action], end: usize) -> usize {
    actions[..end]
        .iter()
        .filter(|a| matches!(a, Action::Row { .. }))
        .count()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_two_slot_oracle RANK MASTER_IP MODEL PACK [PORT]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank must be 0 or 1".into());
    }
    let port: u16 = args.get(5).map_or(Ok(11564), |s| s.parse())?;
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
    control.set_read_timeout(Some(Duration::from_secs(180)))?;
    control.set_write_timeout(Some(Duration::from_secs(180)))?;
    let result = run(
        rank,
        &mut control,
        &resolved,
        Path::new(&args[3]),
        Path::new(&args[4]),
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
