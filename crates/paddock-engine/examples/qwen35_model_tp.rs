//! Whole-backbone eager batch-one Qwen3.8 TP=2 parity gate.
//! Worker first, then head: qwen35_model_tp RANK MASTER_IP MODEL PACK [PORT]
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
use std::{
    error::Error,
    io::{Read, Write},
    net::TcpStream,
    path::Path,
    sync::Arc,
    time::Duration,
};

const MAX_CTX: usize = 16;
const PROMPT: [u32; 3] = [1, 2, 3];
const GENERATED: usize = 2;
const TOL_ABS: f32 = 1e-3;
const TOL_REL: f32 = 1e-3;

fn kv_event(
    stream: &mut TcpStream,
    rank: usize,
    kv: &mut MirroredKv,
    op: Operation,
) -> Result<(), Box<dyn Error>> {
    if rank == 0 {
        let event = kv.authorize(op)?;
        let data = serde_json::to_vec(&event)?;
        if data.len() > 65536 {
            return Err("KV event too large".into());
        }
        stream.write_all(&(data.len() as u32).to_le_bytes())?;
        stream.write_all(&data)?;
    } else {
        let mut size = [0; 4];
        stream.read_exact(&mut size)?;
        let n = u32::from_le_bytes(size) as usize;
        if n > 65536 {
            return Err("KV event too large".into());
        }
        let mut data = vec![0; n];
        stream.read_exact(&mut data)?;
        let event: Event = serde_json::from_slice(&data)?;
        if event.operation != op {
            return Err("KV operation diverged".into());
        }
        kv.mirror(&event)?;
    }
    Ok(())
}

fn exchange_token(
    stream: &mut TcpStream,
    rank: usize,
    token: &mut u32,
) -> Result<(), Box<dyn Error>> {
    if rank == 0 {
        stream.write_all(&token.to_le_bytes())?;
    } else {
        let mut bytes = [0; 4];
        stream.read_exact(&mut bytes)?;
        *token = u32::from_le_bytes(bytes);
    }
    Ok(())
}

fn argmax(values: &[f32]) -> Result<u32, Box<dyn Error>> {
    let (&first, rest) = values.split_first().ok_or("empty logits")?;
    if !first.is_finite() {
        return Err("non-finite logits".into());
    }
    let mut best = first;
    let mut index = 0usize;
    for (i, &value) in rest.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("non-finite logit at {}", i + 1).into());
        }
        if value > best {
            best = value;
            index = i + 1;
        }
    }
    Ok(u32::try_from(index)?)
}

fn compare_logits(tp: &[f32], serial: &[f32], step: usize) -> Result<f32, Box<dyn Error>> {
    if tp.len() != serial.len() {
        return Err("vocabulary size mismatch".into());
    }
    let mut max_abs = 0.0f32;
    for (i, (&a, &b)) in tp.iter().zip(serial).enumerate() {
        let delta = (a - b).abs();
        max_abs = max_abs.max(delta);
        if !a.is_finite() || !b.is_finite() || delta > TOL_ABS + TOL_REL * b.abs() {
            return Err(format!("step={step} logit={i} tp={a} serial={b} abs={delta}").into());
        }
    }
    if argmax(tp)? != argmax(serial)? {
        return Err(format!("step={step} greedy token mismatch").into());
    }
    Ok(max_abs)
}

fn exchange_and_check_logits(
    stream: &mut TcpStream,
    rank: usize,
    logits: &[f32],
    pass: usize,
    step: usize,
) -> Result<(), Box<dyn Error>> {
    if rank == 0 {
        let mut bytes = Vec::with_capacity(4 + logits.len() * 4);
        bytes.extend_from_slice(&u32::try_from(logits.len())?.to_le_bytes());
        for &value in logits {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        stream.write_all(&bytes)?;
    } else {
        let mut size = [0; 4];
        stream.read_exact(&mut size)?;
        let n = u32::from_le_bytes(size) as usize;
        if n != logits.len() {
            return Err(format!(
                "pass={pass} step={step}: rank logit count {n} != {}",
                logits.len()
            )
            .into());
        }
        let mut bytes = vec![0; n * 4];
        stream.read_exact(&mut bytes)?;
        for (i, (&local, remote)) in logits.iter().zip(bytes.chunks_exact(4)).enumerate() {
            let head = u32::from_le_bytes(remote.try_into()?);
            if local.to_bits() != head {
                return Err(format!(
                    "pass={pass} step={step}: rank logit {i} differs: worker={local} head={}",
                    f32::from_bits(head)
                )
                .into());
            }
        }
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
        return Err("run with PADDOCK_NO_SPEC=1 to keep speculation disabled".into());
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
        .ok_or("expected NCCL TP group")?;
    let map = MappedGguf::open(model)?;
    let dtype = KvDtype::Fp16;
    let mut tp = Qwen35TpRank::load(exec.clone(), &map, &group, MAX_CTX, dtype)?;
    let mut serial = if rank == 0 {
        Some(GpuQwen35::load(exec.clone(), &map, MAX_CTX)?)
    } else {
        None
    };
    let page_tokens = paddock_engine::gpu_model::prefix_cache::BLOCK_TOKENS;
    let blocks = u32::try_from(MAX_CTX.div_ceil(page_tokens))?;
    let mut logical = MirroredKv::new(blocks, 1, MAX_CTX)?;
    let total_steps = PROMPT.len() + GENERATED;
    let mut first_pass = Vec::<Vec<f32>>::with_capacity(total_steps);
    let mut first_serial = Vec::<Vec<f32>>::with_capacity(total_steps);
    let mut first_tokens = Vec::<u32>::with_capacity(total_steps);
    for pass in 0..2 {
        if pass == 1 {
            kv_event(control, rank, &mut logical, Operation::Reset)?;
            tp.reset()?;
            if let Some(s) = &mut serial {
                s.reset();
            }
        }
        let mut token = PROMPT[0];
        for step in 0..total_steps {
            if step < PROMPT.len() {
                token = PROMPT[step];
            }
            exchange_token(control, rank, &mut token)?;
            kv_event(
                control,
                rank,
                &mut logical,
                Operation::Ensure {
                    slot: 0,
                    position: step,
                },
            )?;
            let tp_logits = tp.forward_token(&group, &logical, token, step)?;
            if pass == 0 {
                first_pass.push(tp_logits.clone());
            } else if tp_logits != first_pass[step] {
                return Err(format!("rank={rank} reset replay mismatch at step={step}").into());
            }
            if let Some(s) = &mut serial {
                let serial_logits = s.forward_one_no_graph(token)?;
                let max_abs = compare_logits(&tp_logits, &serial_logits, step)?;
                if pass == 0 {
                    first_serial.push(serial_logits);
                    first_tokens.push(token);
                }
                println!(
                    "rank=0 pass={pass} step={step} max_abs={max_abs:.8} greedy={} checksum={:.6}",
                    argmax(&tp_logits)?,
                    tp_logits.iter().sum::<f32>()
                );
            }
            exchange_and_check_logits(control, rank, &tp_logits, pass, step)?;
            if rank == 1 {
                println!(
                    "rank=1 pass={pass} step={step} exact_rank_parity=true checksum={:.6}",
                    tp_logits.iter().sum::<f32>()
                );
            }
            if step + 1 < total_steps && rank == 0 {
                token = argmax(&tp_logits)?;
            }
        }
    }
    // Exercise the ordinary TP=1 decode path, including first-token graph
    // capture and later graph replays, against the eager oracle from pass 0.
    if let Some(s) = &mut serial {
        s.reset();
        for (step, (&token, eager)) in first_tokens.iter().zip(&first_serial).enumerate() {
            let graph_logits = s.forward_one(token)?;
            let max_abs = compare_logits(&graph_logits, eager, step)?;
            println!(
                "rank=0 tp1_graph step={step} max_abs={max_abs:.8} greedy={}",
                argmax(&graph_logits)?
            );
        }
    }
    group.stream().synchronize()?;
    exec.synchronize()?;
    kv_event(control, rank, &mut logical, Operation::Reset)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_model_tp RANK MASTER_IP MODEL PACK [PORT]".into());
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
