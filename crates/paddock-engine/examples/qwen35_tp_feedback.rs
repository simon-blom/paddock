//! Two-rank proof that rank-0 device-sampled IDs feed the next TP forward.
//! qwen35_tp_feedback RANK MASTER_IP MODEL PACK [PORT]
use paddock_dist::{
    config::ParallelConfig,
    protocol::{receive_nccl_id, send_nccl_id},
    worker::{connect_worker, coordinate, shutdown_worker},
};
use paddock_engine::{
    gpu::{
        GpuExecutor, KvDtype,
        distributed::{NcclCommunicator, create_unique_id},
    },
    gpu_model::qwen35::{
        Qwen35TpRank,
        tp_kv::{MirroredKv, Operation},
    },
    sampler::DevicePlan,
};
use paddock_models::mapped::MappedGguf;
use std::{
    error::Error,
    io::{Read, Write},
    path::Path,
    sync::Arc,
};

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32
}

fn ensure(
    kv: &mut MirroredKv,
    stream: &mut std::net::TcpStream,
    rank: usize,
    position: usize,
) -> Result<(), Box<dyn Error>> {
    if rank == 0 {
        let event = kv.authorize(Operation::Ensure { slot: 0, position })?;
        let bytes = serde_json::to_vec(&event)?;
        stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
        stream.write_all(&bytes)?;
    } else {
        let mut len = [0; 4];
        stream.read_exact(&mut len)?;
        let n = u32::from_le_bytes(len) as usize;
        if n > 65536 {
            return Err("oversize KV event".into());
        }
        let mut bytes = vec![0; n];
        stream.read_exact(&mut bytes)?;
        let event = serde_json::from_slice(&bytes)?;
        kv.mirror(&event)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("PADDOCK_NO_SPEC").is_none() {
        return Err("set PADDOCK_NO_SPEC=1".into());
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_tp_feedback RANK MASTER_IP MODEL PACK [PORT]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank outside TP=2".into());
    }
    let port = args.get(5).map_or(Ok(18566), |p| p.parse())?;
    let resolved = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some(args[2].clone()),
        master_port: Some(port),
    }
    .resolved(false)?
    .ok_or("TP=2 required")?;
    let (mut stream, _) = if rank == 0 {
        coordinate(&resolved, false)?
    } else {
        connect_worker(&resolved)?
    };
    let id = if rank == 0 {
        let id = create_unique_id()?;
        send_nccl_id(&mut stream, &id)?;
        id
    } else {
        receive_nccl_id(&mut stream)?
    };
    let exec = Arc::new(GpuExecutor::new(0, Path::new(&args[4]))?);
    let group = NcclCommunicator::from_resolved(Some(&resolved), exec.stream.context(), id)?
        .ok_or("NCCL missing")?;
    let map = MappedGguf::open(Path::new(&args[3]))?;
    let mut model = Qwen35TpRank::load_slots(exec.clone(), &map, &group, 48, KvDtype::Fp16, 2)?;
    let mut kv = MirroredKv::new(6, 2, 48)?;
    let mut tokens = vec![1u32];
    let mut events = Vec::new();
    for position in 0..6 {
        ensure(&mut kv, &mut stream, rank, position)?;
        let plane = position % 2;
        if position == 0 {
            if rank == 0 {
                events.push(model.forward_host_to_feedback(
                    &group,
                    &kv,
                    1,
                    0,
                    0,
                    plane,
                    DevicePlan::Greedy,
                )?);
            } else {
                model.forward_token_worker_slot(&group, &kv, 1, 0, 0)?;
            }
        } else {
            let old = plane ^ 1;
            let event = model.forward_feedback_to_feedback(
                &group,
                &kv,
                0,
                position,
                old,
                plane,
                DevicePlan::Greedy,
            )?;
            if rank == 0 {
                events.push(event.ok_or("rank 0 missing feedback event")?);
                // Tick N has been enqueued before tick N-1 reaches the host.
                let id = model.feedback_id_after(&events[position - 1], 0, old)?;
                tokens.push(id);
            }
        }
    }
    if rank == 0 {
        tokens.push(model.feedback_id_after(&events[5], 0, 1)?);
        model.reset()?;
        let reset = kv.authorize(Operation::Flush)?;
        let bytes = serde_json::to_vec(&reset)?;
        stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
        stream.write_all(&bytes)?;
        for position in 0..6 {
            ensure(&mut kv, &mut stream, rank, position)?;
            stream.write_all(&tokens[position].to_le_bytes())?;
            let logits = model.forward_token_slot(&group, &kv, tokens[position], position, 0)?;
            assert_eq!(
                argmax(&logits),
                tokens[position + 1],
                "feedback mismatch at {position}"
            );
        }
        println!(
            "TP device feedback: six sampled IDs matched eager greedy replay; device broadcast and alternating planes passed"
        );
    } else {
        let mut len = [0; 4];
        stream.read_exact(&mut len)?;
        let n = u32::from_le_bytes(len) as usize;
        if n > 65536 {
            return Err("oversize reset event".into());
        }
        let mut bytes = vec![0; n];
        stream.read_exact(&mut bytes)?;
        kv.mirror(&serde_json::from_slice(&bytes)?)?;
        model.reset()?;
        for position in 0..6 {
            ensure(&mut kv, &mut stream, rank, position)?;
            let mut token = [0; 4];
            stream.read_exact(&mut token)?;
            model.forward_token_worker_slot(&group, &kv, u32::from_le_bytes(token), position, 0)?;
        }
    }
    group.stream().synchronize()?;
    exec.synchronize()?;
    if rank == 0 {
        shutdown_worker(&mut stream, true)?;
    } else {
        // The probe's final shutdown is explicit rather than an unbounded peer wait.
        let _ = paddock_dist::protocol::ControlMessage::from_stream(&mut stream)?;
    }
    Ok(())
}
