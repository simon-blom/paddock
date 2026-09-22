//! Standalone two-Spark NCCL microbenchmark (no model, pack, or scheduler).
//! Build: cargo build -p paddock-engine --example nccl_bench
//! Run rank 1 on worker: nccl_bench 1 192.168.100.10 [port]
//! Run rank 0 on head:   nccl_bench 0 192.168.100.10 [port]
//! Set LD_LIBRARY_PATH to the installed NCCL lib directory on both nodes;
//! set NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 and
//! NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET to audit RoCE routing.
//! `nccl_bench tp1` verifies the historical no-NCCL path in a fresh process.

use std::error::Error;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaStream};
use paddock_dist::config::ParallelConfig;
use paddock_dist::protocol::{ControlMessage, receive_nccl_id, send_nccl_id};
use paddock_dist::worker::{connect_worker, coordinate, shutdown_worker};
use paddock_engine::gpu::distributed::{Communicator, NcclCommunicator, create_unique_id};

type Fail = Box<dyn Error>;

fn mapped_nccl() -> Result<bool, Fail> {
    Ok(std::fs::read_to_string("/proc/self/maps")?.contains("libnccl.so"))
}

fn main() -> Result<(), Fail> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "tp1") {
        assert!(!mapped_nccl()?, "NCCL was loaded before TP=1 probe");
        let cfg = ParallelConfig::default();
        assert!(cfg.resolved(true)?.is_none());
        let ctx = CudaContext::new(0)?;
        let compute = ctx.new_stream()?;
        let buffer = compute.clone_htod(&[1.0_f32, 2.0])?;
        assert_eq!(compute.clone_dtoh(&buffer)?, vec![1.0, 2.0]);
        assert!(!mapped_nccl()?, "TP=1 loaded NCCL");
        println!("TP=1 CUDA allocation/copy passed; libnccl absent from /proc/self/maps");
        return Ok(());
    }
    if args.len() < 3 {
        return Err("usage: nccl_bench [0|1] MASTER_IP [PORT] | tp1".into());
    }
    let rank: usize = args[1].parse()?;
    let port: u16 = args.get(3).map_or(Ok(11561), |v| v.parse())?;
    let resolved = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some(args[2].clone()),
        master_port: Some(port),
    }
    .resolved(false)?
    .ok_or("expected TP=2")?;
    let (mut control, session) = if rank == 0 {
        coordinate(&resolved, false)?
    } else {
        connect_worker(&resolved)?
    };
    control.set_read_timeout(Some(std::time::Duration::from_secs(120)))?;
    control.set_write_timeout(Some(std::time::Duration::from_secs(120)))?;
    let result = (|| -> Result<(), Fail> {
        let id = if rank == 0 {
            let id = create_unique_id()?;
            send_nccl_id(&mut control, &id)?;
            id
        } else {
            receive_nccl_id(&mut control)?
        };
        let ctx = CudaContext::new(0)?;
        // Mirror GpuExecutor: cudarc's automatic cross-stream event tracking
        // is disabled. Only the explicit compute <-> communication fences can
        // order the benchmark's uploads, collectives and readbacks.
        // SAFETY: every cross-stream buffer use below is enclosed by
        // `after_compute`/`before_compute`, with a stream sync at teardown.
        unsafe { ctx.disable_event_tracking() };
        let compute = ctx.new_stream()?;
        let group = NcclCommunicator::from_resolved(Some(&resolved), &ctx, id)?
            .ok_or("missing TP=2 process group")?;
        println!("rank={rank} session={session} NCCL process group initialized");
        // Both ranks check the same deterministic contents before timing.
        check_collectives(&compute, &group)?;
        println!("rank={rank} all-reduce/all-gather/reduce-scatter/broadcast parity PASS");
        for bytes in [1024, 65536, 1048576, 8388608, 33554432] {
            for op in [Op::AllReduce, Op::AllGather] {
                bench(&compute, &group, rank, bytes, op)?;
            }
        }
        group.stream().synchronize()?;
        compute.synchronize()?;
        drop(group);
        Ok(())
    })();
    if rank == 0 {
        shutdown_worker(&mut control, result.is_ok())?;
    } else {
        match ControlMessage::from_stream(&mut control)? {
            ControlMessage::Shutdown { graceful } if graceful == result.is_ok() => {}
            other => return Err(format!("unexpected shutdown: {other:?}").into()),
        }
    }
    result?;
    println!("rank={rank} communicator and bootstrap shutdown clean");
    Ok(())
}

fn check_collectives(compute: &Arc<CudaStream>, group: &NcclCommunicator) -> Result<(), Fail> {
    let rank = group.rank();
    let a = compute.clone_htod(&vec![(rank + 1) as f32; 256])?;
    let mut out = compute.alloc_zeros::<f32>(256)?;
    group.after_compute(compute)?;
    group.all_reduce(&a, &mut out)?;
    group.before_compute(compute)?;
    assert!(
        compute.clone_dtoh(&out)?.iter().all(|x| *x == 3.0),
        "all-reduce rank {rank}"
    );

    let mut gathered = compute.alloc_zeros::<f32>(512)?;
    group.after_compute(compute)?;
    group.all_gather(&a, &mut gathered)?;
    group.before_compute(compute)?;
    let values = compute.clone_dtoh(&gathered)?;
    assert!(values[..256].iter().all(|x| *x == 1.0));
    assert!(values[256..].iter().all(|x| *x == 2.0));

    let send = compute.clone_htod(&vec![(rank + 1) as f32; 512])?;
    let mut scattered = compute.alloc_zeros::<f32>(256)?;
    group.after_compute(compute)?;
    group.reduce_scatter(&send, &mut scattered)?;
    group.before_compute(compute)?;
    assert!(compute.clone_dtoh(&scattered)?.iter().all(|x| *x == 3.0));

    let mut bcast = compute.clone_htod(&vec![if rank == 0 { 7.0_f32 } else { 0.0 }; 256])?;
    group.after_compute(compute)?;
    group.broadcast(&mut bcast, 0)?;
    group.before_compute(compute)?;
    assert!(compute.clone_dtoh(&bcast)?.iter().all(|x| *x == 7.0));
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Op {
    AllReduce,
    AllGather,
}

fn bench(
    compute: &Arc<CudaStream>,
    group: &NcclCommunicator,
    rank: usize,
    bytes: usize,
    op: Op,
) -> Result<(), Fail> {
    let count = bytes / std::mem::size_of::<f32>();
    let src = compute.clone_htod(&vec![(rank + 1) as f32; count])?;
    let mut dst =
        compute.alloc_zeros::<f32>(count * if matches!(op, Op::AllGather) { 2 } else { 1 })?;
    group.after_compute(compute)?;
    for _ in 0..5 {
        match op {
            Op::AllReduce => group.all_reduce(&src, &mut dst)?,
            Op::AllGather => group.all_gather(&src, &mut dst)?,
        }
    }
    group.stream().synchronize()?; // benchmark warmup only; not a model hot path
    let iters = if bytes >= 8 * 1024 * 1024 { 10 } else { 30 };
    let start = group
        .stream()
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = group
        .stream()
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(group.stream())?;
    for _ in 0..iters {
        match op {
            Op::AllReduce => group.all_reduce(&src, &mut dst)?,
            Op::AllGather => group.all_gather(&src, &mut dst)?,
        }
    }
    end.record(group.stream())?;
    end.synchronize()?; // measurement boundary only
    let ms = start.elapsed_ms(&end)? / iters as f32;
    group.before_compute(compute)?;
    let values = compute.clone_dtoh(&dst)?;
    match op {
        Op::AllReduce => assert!(values.iter().all(|v| *v == 3.0)),
        Op::AllGather => {
            assert!(values[..count].iter().all(|v| *v == 1.0));
            assert!(values[count..].iter().all(|v| *v == 2.0));
        }
    }
    // NCCL 2-rank bus bandwidth factor: all-reduce 1, all-gather 1/2.
    let alg_gbps = bytes as f64 / (ms as f64 * 1e-3) / 1e9;
    let bus_gbps = alg_gbps
        * if matches!(op, Op::AllGather) {
            0.5
        } else {
            1.0
        };
    println!(
        "rank={rank} op={op:?} bytes={bytes} iters={iters} latency_us={:.2} alg_GBps={alg_gbps:.3} bus_GBps={bus_gbps:.3} parity=PASS",
        ms * 1000.0
    );
    Ok(())
}
