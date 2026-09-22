//! Two-Spark, one-layer Qwen3.8 dense FFN parity probe. No scheduler/GQA.
//! Run worker first, then head: qwen35_ffn_tp RANK MASTER_IP MODEL PACK [PORT] [LAYER]
//! Both nodes need the same pinned GGUF and CUDA pack. NCCL uses the
//! environment from `nccl_bench`; only the head uploads whole FFN weights
//! for the independent TP=1 oracle. Neither TP rank loads a whole FFN weight.

use std::error::Error;
use std::path::Path;

use paddock_dist::config::ParallelConfig;
use paddock_dist::protocol::{ControlMessage, receive_nccl_id, send_nccl_id};
use paddock_dist::worker::{connect_worker, coordinate, shutdown_worker};
use paddock_engine::gpu::distributed::{NcclCommunicator, create_unique_id};
use paddock_engine::gpu::{GpuExecutor, QuantW};
use paddock_engine::gpu_model::qwen35::ffn_tp::FfnTpRank;
use paddock_models::mapped::MappedGguf;

fn gemv(
    exec: &GpuExecutor,
    weight: &QuantW,
    x: &cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<(), Box<dyn Error>> {
    match weight {
        QuantW::Q8(w) => exec.q8_0_gemv_repacked(w, None, x, y)?,
        QuantW::Kq(w) => exec.kquant_gemv(w, x, y)?,
    }
    Ok(())
}

fn serial_oracle(
    exec: &GpuExecutor,
    map: &MappedGguf,
    layer: usize,
    x: &cudarc::driver::CudaSlice<f32>,
    ff: usize,
    hidden: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let name = |part| format!("blk.{layer}.ffn_{part}.weight");
    let gate = exec.load_quantw(map, &name("gate"))?;
    let up = exec.load_quantw(map, &name("up"))?;
    let down = exec.load_quantw(map, &name("down"))?;
    let mut g = exec.alloc(ff)?;
    let mut u = exec.alloc(ff)?;
    let mut y = exec.alloc(hidden)?;
    gemv(exec, &gate, x, &mut g)?;
    gemv(exec, &up, x, &mut u)?;
    exec.swiglu(&mut g, &u, ff)?;
    gemv(exec, &down, &g, &mut y)?;
    Ok(exec.to_host(&y)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_ffn_tp RANK MASTER_IP MODEL PACK [PORT] [LAYER]".into());
    }
    let rank: usize = args[1].parse()?;
    let port: u16 = args.get(5).map_or(Ok(11562), |v| v.parse())?;
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
    control.set_read_timeout(Some(std::time::Duration::from_secs(180)))?;
    control.set_write_timeout(Some(std::time::Duration::from_secs(180)))?;
    let result = (|| -> Result<(), Box<dyn Error>> {
        let id = if rank == 0 {
            let id = create_unique_id()?;
            send_nccl_id(&mut control, &id)?;
            id
        } else {
            receive_nccl_id(&mut control)?
        };
        let exec = GpuExecutor::new(0, Path::new(&args[4]))?;
        let group = NcclCommunicator::from_resolved(Some(&resolved), exec.stream.context(), id)?
            .ok_or("expected NCCL communicator")?;
        let map = MappedGguf::open(Path::new(&args[3]))?;
        let layer: usize = args.get(6).map_or(Ok(0), |v| v.parse())?;
        let (gate, _) = map.tensor_bytes(&format!("blk.{layer}.ffn_gate.weight"))?;
        let hidden = usize::try_from(gate.dims[0])?;
        let ff = usize::try_from(gate.dims[1])?;
        let (up, _) = map.tensor_bytes(&format!("blk.{layer}.ffn_up.weight"))?;
        let (down, _) = map.tensor_bytes(&format!("blk.{layer}.ffn_down.weight"))?;
        println!(
            "rank={rank} layer={layer} hidden={hidden} ff={ff} types gate/up/down={:?}/{:?}/{:?}",
            gate.ggml_type, up.ggml_type, down.ggml_type
        );
        let mut tp = FfnTpRank::load(&exec, &map, layer, &group)?;
        // Several nontrivial inputs, including a second forward on reused
        // buffers: tests both block selection and NCCL fence ordering.
        for seed in [3_usize, 17, 53] {
            let host: Vec<f32> = (0..hidden)
                .map(|i| ((i * 31 + seed * 7) % 127) as f32 / 63.0 - 1.0)
                .collect();
            let x = exec.to_device(&host)?;
            let expected = if rank == 0 {
                Some(serial_oracle(&exec, &map, layer, &x, ff, hidden)?)
            } else {
                None
            };
            let got = exec.to_host(tp.forward(&exec, &group, &x)?)?;
            assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
            if let Some(expected) = expected {
                let mut max_abs = 0.0_f32;
                let mut max_rel = 0.0_f32;
                for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
                    let delta = (a - b).abs();
                    max_abs = max_abs.max(delta);
                    max_rel = max_rel.max(delta / b.abs().max(1e-4));
                    assert!(
                        delta <= 1e-3 + 1e-3 * b.abs(),
                        "seed={seed} i={i} tp={a} serial={b}"
                    );
                }
                println!(
                    "seed={seed} FFN TP=2 vs TP=1 PASS max_abs={max_abs:.8} max_rel={max_rel:.8} checksum={:.6}",
                    got.iter().sum::<f32>()
                );
            } else {
                println!(
                    "rank=1 seed={seed} finite output, checksum={:.6}",
                    got.iter().sum::<f32>()
                );
            }
        }
        group.stream().synchronize()?;
        exec.synchronize()?;
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
    println!("rank={rank} FFN probe shut down cleanly");
    Ok(())
}
