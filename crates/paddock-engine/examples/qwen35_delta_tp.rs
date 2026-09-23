//! Isolated two-Spark Qwen3.8 DeltaNet decode/native-span prefill probe.
//! Serial GPU oracle loads whole GGUF weights separately on each rank; the
//! component under test loads rank-local weights/state and reduces only output.
//! Usage: qwen35_delta_tp RANK MASTER_IP MODEL PACK [PORT] [LAYER]
//! Adapted from Erik Bogado / ErikBPF tp-10 (8495d2d) and tp-11 (02ba232).
use cudarc::driver::CudaSlice;
use paddock_dist::{
    config::ParallelConfig,
    protocol::{ControlMessage, receive_nccl_id, send_nccl_id},
    worker::{connect_worker, coordinate, shutdown_worker},
};
use paddock_engine::{
    gpu::distributed::{Communicator, NcclCommunicator, create_unique_id},
    gpu::{GpuExecutor, RepackedKQ, RepackedQ8},
    gpu_model::qwen35::delta_tp::{DeltaGeometry, DeltaTpRank},
};
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::{error::Error, path::Path};

const WIDTH: usize = 5120;
const S: usize = 128;
const NK: usize = 16;
const NV: usize = 48;
const K: usize = 4;
const MIXED: usize = (2 * NK + NV) * S;
const VALUE: usize = NV * S;

struct Serial {
    weights: [RepackedKQ; 3],
    ab: [RepackedQ8; 2],
    conv_weight: CudaSlice<f32>,
    a: CudaSlice<f32>,
    dt: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    recurrent: CudaSlice<f32>,
    conv: CudaSlice<f32>,
    eps: f32,
}
impl Serial {
    fn load(e: &GpuExecutor, map: &MappedGguf, layer: usize) -> Result<Self, Box<dyn Error>> {
        let n = |part: &str| format!("blk.{layer}.{part}");
        let weights = ["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"]
            .map(|x| e.repack_kquant(map, &n(x)))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "wrong projection count")?;
        let ab = ["ssm_alpha.weight", "ssm_beta.weight"]
            .map(|x| e.repack_q8(map, &n(x)))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "wrong gate count")?;
        let mut s = Self {
            weights,
            ab,
            conv_weight: e.upload(map, &n("ssm_conv1d.weight"))?.buf,
            a: e.upload(map, &n("ssm_a"))?.buf,
            dt: e.upload(map, &n("ssm_dt.bias"))?.buf,
            norm: e.upload(map, &n("ssm_norm.weight"))?.buf,
            recurrent: e.alloc(NV * S * S)?,
            conv: e.alloc((K - 1) * MIXED)?,
            eps: map
                .gguf()
                .arch_field("attention.layer_norm_rms_epsilon")
                .and_then(Value::as_f32)
                .unwrap_or(1e-6),
        };
        s.reset(e)?;
        Ok(s)
    }
    fn reset(&mut self, e: &GpuExecutor) -> Result<(), Box<dyn Error>> {
        e.stream.memset_zeros(&mut self.recurrent)?;
        e.stream.memset_zeros(&mut self.conv)?;
        Ok(())
    }
    fn decode(&mut self, e: &GpuExecutor, x: &CudaSlice<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut mixed = e.alloc(MIXED)?;
        let mut conv_out = e.alloc(MIXED)?;
        let mut q = e.alloc(VALUE)?;
        let mut k = e.alloc(VALUE)?;
        let mut v = e.alloc(VALUE)?;
        let mut gate = e.alloc(NV)?;
        let mut beta = e.alloc(NV)?;
        let mut attn = e.alloc(VALUE)?;
        let mut z = e.alloc(VALUE)?;
        let mut core = e.alloc(VALUE)?;
        let mut out = e.alloc(WIDTH)?;
        e.kquant_gemv(&self.weights[0], x, &mut mixed)?;
        e.conv_step(
            &mut self.conv,
            &mixed,
            &self.conv_weight,
            &mut conv_out,
            MIXED,
            K,
        )?;
        e.deltanet_split_gqa_norm(&conv_out, &mut q, &mut k, &mut v, 1, NK, NV, S)?;
        e.deltanet_alpha_beta_gate(
            &self.ab[0],
            &self.ab[1],
            x,
            &self.a,
            &self.dt,
            &mut gate,
            &mut beta,
            NV,
        )?;
        e.gated_delta_recurrent_v2(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            None,
            &mut self.recurrent,
            0,
            None,
            &mut attn,
            1,
            1,
            NV,
            S,
        )?;
        e.kquant_gemv(&self.weights[1], x, &mut z)?;
        e.gated_rmsnorm(&attn, &z, &self.norm, &mut core, NV, S, self.eps)?;
        e.kquant_gemv(&self.weights[2], &core, &mut out)?;
        Ok(e.to_host(&out)?)
    }
}
fn host_input(seed: usize) -> Vec<f32> {
    (0..WIDTH)
        .map(|i| (((i * 17 + (seed + 1) * 13) % 127) as f32 - 63.0) / 41.0)
        .collect()
}
fn check(got: &[f32], want: &[f32], label: &str) -> Result<f32, Box<dyn Error>> {
    if got.len() != want.len() {
        return Err(format!("{label}: length {} != {}", got.len(), want.len()).into());
    }
    let mut max = 0.0f32;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let diff = (a - b).abs();
        if !(a.is_finite() && b.is_finite() && diff <= 1e-3 + 1e-3 * b.abs()) {
            return Err(format!("{label}[{i}]: {a} vs {b}").into());
        }
        max = max.max(diff);
    }
    Ok(max)
}
fn check_state(
    e: &GpuExecutor,
    tp: &DeltaTpRank,
    reference: &Serial,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let g: &DeltaGeometry = &tp.geometry;
    let (state, conv) = tp.state();
    let s = e.to_host(state)?;
    let c = e.to_host(conv)?;
    let ref_s = e.to_host(&reference.recurrent)?;
    let ref_c = e.to_host(&reference.conv)?;
    for (local, &global) in g.value_heads.iter().enumerate() {
        check(
            &s[local * S * S..(local + 1) * S * S],
            &ref_s[global * S * S..(global + 1) * S * S],
            &format!("{label} recurrent head {global}"),
        )?;
    }
    for lag in 0..K - 1 {
        for (local, &global) in g.channels.iter().enumerate() {
            let a = c[lag * g.mixed() + local];
            let b = ref_c[lag * MIXED + global];
            if !(a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-3 + 1e-3 * b.abs()) {
                return Err(format!("{label} conv lag={lag} channel={global}: {a} vs {b}").into());
            }
        }
    }
    Ok(())
}
fn run(
    e: &GpuExecutor,
    group: &NcclCommunicator,
    tp: &mut DeltaTpRank,
    reference: &mut Serial,
) -> Result<(), Box<dyn Error>> {
    let rank = group.rank();
    let short = e.alloc(WIDTH - 1)?;
    let too_long = e.alloc(65 * WIDTH)?;
    let one = e.alloc(WIDTH)?;
    if tp.decode(e, group, &short).is_ok()
        || tp.prefill(e, group, &one, 0).is_ok()
        || tp.prefill(e, group, &too_long, 65).is_ok()
    {
        return Err("invalid DeltaNet TP input was accepted".into());
    }
    check_state(e, tp, reference, "rejected input")?;
    let mut step = 0;
    let mut first = None;
    for phase in 0..5 {
        // Warm-state decode, followed by four native causal-conv/recurrent spans.
        let rows = match phase {
            0 => 8,
            1 => 2,
            2 => 7,
            3 => 31,
            _ => 64,
        };
        if phase == 0 {
            for _ in 0..rows {
                let input = e.to_device(&host_input(step))?;
                let want = reference.decode(e, &input)?;
                let got = e.to_host_len(tp.decode(e, group, &input)?, WIDTH)?;
                let max = check(&got, &want, &format!("decode {step}"))?;
                check_state(e, tp, reference, &format!("decode {step}"))?;
                if step == 0 {
                    first = Some((input, got));
                }
                println!("rank={rank} decode={step} max_abs={max:.6}");
                step += 1;
            }
        } else {
            let mut inputs = Vec::with_capacity(rows * WIDTH);
            let mut want = Vec::with_capacity(rows * WIDTH);
            for t in 0..rows {
                let h = host_input(step + t);
                want.extend(reference.decode(e, &e.to_device(&h)?)?);
                inputs.extend(h);
            }
            let x = e.to_device(&inputs)?;
            let got = e.to_host_len(tp.prefill(e, group, &x, rows)?, rows * WIDTH)?;
            let max = check(&got, &want, &format!("prefill {rows} from {step}"))?;
            check_state(e, tp, reference, &format!("prefill {rows} from {step}"))?;
            println!("rank={rank} prefill={rows} start={step} max_abs={max:.6}");
            step += rows;
            let next = e.to_device(&host_input(step))?;
            let expected = reference.decode(e, &next)?;
            let actual = e.to_host_len(tp.decode(e, group, &next)?, WIDTH)?;
            let max = check(&actual, &expected, &format!("continuation {step}"))?;
            check_state(e, tp, reference, &format!("continuation {step}"))?;
            println!("rank={rank} continuation={step} max_abs={max:.6}");
            step += 1;
        }
    }
    tp.reset(e)?;
    reference.reset(e)?;
    let (x, original) = first.ok_or("missing first step")?;
    let want = reference.decode(e, &x)?;
    let got = e.to_host_len(tp.decode(e, group, &x)?, WIDTH)?;
    check(&got, &want, "reset oracle")?;
    if original != got {
        return Err("reset failed exact replay".into());
    }
    check_state(e, tp, reference, "reset replay")?;
    tp.reset(e)?;
    reference.reset(e)?;
    check_state(e, tp, reference, "second reset zero")?;
    let mut inputs = Vec::with_capacity(9 * WIDTH);
    let mut want = Vec::with_capacity(9 * WIDTH);
    for t in 0..9 {
        let h = host_input(t);
        want.extend(reference.decode(e, &e.to_device(&h)?)?);
        inputs.extend(h);
    }
    let input = e.to_device(&inputs)?;
    let actual = e.to_host_len(tp.prefill(e, group, &input, 9)?, 9 * WIDTH)?;
    let cold_max = check(&actual, &want, "cold prefill 9")?;
    check_state(e, tp, reference, "cold prefill 9")?;
    println!("rank={rank} cold_prefill=9 max_abs={cold_max:.6}");
    println!(
        "rank={rank} PASSED DeltaNet decode/prefill/continuation/reset; local_state_bytes={}",
        tp.local_state_bytes()
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_delta_tp RANK MASTER_IP MODEL PACK [PORT] [LAYER]".into());
    }
    let rank: usize = args[1].parse()?;
    let port: u16 = args.get(5).map_or(Ok(11569), |v| v.parse())?;
    let layer: usize = args.get(6).map_or(Ok(0), |v| v.parse())?;
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
    control.set_read_timeout(Some(std::time::Duration::from_secs(600)))?;
    control.set_write_timeout(Some(std::time::Duration::from_secs(600)))?;
    let result = (|| -> Result<(), Box<dyn Error>> {
        let id = if rank == 0 {
            let id = create_unique_id()?;
            send_nccl_id(&mut control, &id)?;
            id
        } else {
            receive_nccl_id(&mut control)?
        };
        let e = GpuExecutor::new(0, Path::new(&args[4]))?;
        let group = NcclCommunicator::from_resolved(Some(&resolved), e.stream.context(), id)?
            .ok_or("expected NCCL")?;
        let map = MappedGguf::open(Path::new(&args[3]))?;
        let mut tp = DeltaTpRank::load(&e, &map, layer, &group)?;
        let mut oracle = Serial::load(&e, &map, layer)?;
        run(&e, &group, &mut tp, &mut oracle)
    })();
    if rank == 0 {
        shutdown_worker(&mut control, result.is_ok())?;
    } else {
        match ControlMessage::from_stream(&mut control)? {
            ControlMessage::Shutdown { graceful } if graceful == result.is_ok() => {}
            other => return Err(format!("unexpected shutdown: {other:?}").into()),
        }
    }
    result
}
