//! Two-Spark Qwen3.8 one-layer GQA parity, independent full-weight GPU oracle.
//! Worker first, then head: qwen35_gqa_tp RANK MASTER_IP MODEL PACK [PORT] [LAYER]
use cudarc::driver::CudaSlice;
use paddock_dist::{
    config::ParallelConfig,
    protocol::{ControlMessage, receive_nccl_id, send_nccl_id},
    worker::{connect_worker, coordinate, shutdown_worker},
};
use paddock_engine::{
    gpu::distributed::{NcclCommunicator, create_unique_id},
    gpu::{GpuExecutor, KvDtype, QuantW},
    gpu_model::qwen35::gqa_tp::GqaTpRank,
};
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::{error::Error, path::Path};

fn gemv(
    e: &GpuExecutor,
    w: &QuantW,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn Error>> {
    match w {
        QuantW::Q8(w) => e.q8_0_gemv_repacked(w, None, x, y)?,
        QuantW::Kq(w) => e.kquant_gemv(w, x, y)?,
    }
    Ok(())
}
struct Oracle {
    weights: [QuantW; 4],
    qnorm: CudaSlice<f32>,
    knorm: CudaSlice<f32>,
    sinks: CudaSlice<f32>,
    kc: CudaSlice<u8>,
    vc: CudaSlice<u8>,
    nh: usize,
    nk: usize,
    hd: usize,
    width: usize,
    max_ctx: usize,
    pos: usize,
    eps: f32,
    nrot: usize,
    yarn: (f32, f32, f32, f32, f32, f32),
    sections: [u32; 4],
    dtype: KvDtype,
}
impl Oracle {
    fn new(
        e: &GpuExecutor,
        map: &MappedGguf,
        layer: usize,
        max_ctx: usize,
        dtype: KvDtype,
    ) -> Result<Self, Box<dyn Error>> {
        let u = |key| -> Result<usize, Box<dyn Error>> {
            Ok(usize::try_from(
                map.gguf()
                    .arch_field(key)
                    .and_then(Value::as_u64)
                    .ok_or_else(|| format!("missing {key}"))?,
            )?)
        };
        let f = |key| map.gguf().arch_field(key).and_then(Value::as_f32);
        let (width, nh, nk, hd, nrot) = (
            u("embedding_length")?,
            u("attention.head_count")?,
            u("attention.head_count_kv")?,
            u("attention.key_length")?,
            u("rope.dimension_count")?,
        );
        let name = |s: &str| format!("blk.{layer}.{s}.weight");
        let weights = [
            e.load_quantw(map, &name("attn_q"))?,
            e.load_quantw(map, &name("attn_k"))?,
            e.load_quantw(map, &name("attn_v"))?,
            e.load_quantw(map, &name("attn_output"))?,
        ];
        let mut sections = [0; 4];
        if let Some(Value::Array(items)) = map.gguf().arch_field("rope.dimension_sections") {
            for (dst, item) in sections.iter_mut().zip(items) {
                *dst = u32::try_from(item.as_u64().ok_or("bad rope section")?)?;
            }
        }
        Ok(Self {
            weights,
            qnorm: e.upload(map, &name("attn_q_norm"))?.buf,
            knorm: e.upload(map, &name("attn_k_norm"))?.buf,
            sinks: e.alloc_no_sinks(nh)?,
            kc: e.alloc_u8(max_ctx * nk * hd * dtype.bytes())?,
            vc: e.alloc_u8(max_ctx * nk * hd * dtype.bytes())?,
            nh,
            nk,
            hd,
            width,
            max_ctx,
            pos: 0,
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            nrot,
            yarn: YarnRope::new(
                nrot,
                f("rope.freq_base").unwrap_or(1e7),
                1.0,
                u("context_length")?,
                0.0,
                1.0,
                32.0,
                1.0,
            )
            .kernel_params(),
            sections,
            dtype,
        })
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn forward(&mut self, e: &GpuExecutor, x: &CudaSlice<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        let qdim = self.nh * self.hd;
        let kvdim = self.nk * self.hd;
        let mut qg = e.alloc(2 * qdim)?;
        let mut k = e.alloc(kvdim)?;
        let mut v = e.alloc(kvdim)?;
        gemv(e, &self.weights[0], x, &mut qg)?;
        gemv(e, &self.weights[1], x, &mut k)?;
        gemv(e, &self.weights[2], x, &mut v)?;
        let mut q = e.alloc(qdim)?;
        let mut gate = e.alloc(qdim)?;
        e.split_qg(&qg, &mut q, &mut gate, 1, self.nh, self.hd)?;
        let mut qn = e.alloc(qdim)?;
        let mut kn = e.alloc(kvdim)?;
        e.rmsnorm_batch(&q, &self.qnorm, &mut qn, self.hd, self.eps, self.nh)?;
        e.rmsnorm_batch(&k, &self.knorm, &mut kn, self.hd, self.eps, self.nk)?;
        let mut axes = e.alloc_u32(4)?;
        let mut pos = e.alloc_u32(1)?;
        let mut slots = e.alloc_u32(1)?;
        e.stream.memcpy_htod(&[self.pos as u32; 4], &mut axes)?;
        e.stream.memcpy_htod(&[self.pos as u32], &mut pos)?;
        e.stream.memset_zeros(&mut slots)?;
        e.mrope(
            &mut qn,
            &axes,
            1,
            self.nh,
            self.hd,
            self.nrot,
            self.yarn,
            self.sections,
        )?;
        e.mrope(
            &mut kn,
            &axes,
            1,
            self.nk,
            self.hd,
            self.nrot,
            self.yarn,
            self.sections,
        )?;
        e.kv_append_batch(
            &kn,
            &mut self.kc,
            &pos,
            Some(&slots),
            kvdim,
            self.max_ctx,
            1,
            self.dtype,
        )?;
        e.kv_append_batch(
            &v,
            &mut self.vc,
            &pos,
            Some(&slots),
            kvdim,
            self.max_ctx,
            1,
            self.dtype,
        )?;
        let mut attn = e.alloc(qdim)?;
        e.attn_decode_batch(
            &qn,
            &self.kc,
            &self.vc,
            &self.sinks,
            &mut attn,
            &pos,
            Some(&slots),
            self.nh,
            self.nk,
            self.hd,
            self.max_ctx,
            kvdim,
            0,
            1,
            1.0 / (self.hd as f32).sqrt(),
            self.dtype,
        )?;
        e.mul_sigmoid(&mut attn, &gate, qdim)?;
        let mut out = e.alloc(self.width)?;
        gemv(e, &self.weights[3], &attn, &mut out)?;
        self.pos += 1;
        Ok(e.to_host(&out)?)
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_gqa_tp RANK MASTER_IP MODEL PACK [PORT] [LAYER]".into());
    }
    let rank: usize = args[1].parse()?;
    let port: u16 = args.get(5).map_or(Ok(11563), |s| s.parse())?;
    let layer: usize = args.get(6).map_or(Ok(3), |s| s.parse())?;
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
        let e = GpuExecutor::new(0, Path::new(&args[4]))?;
        let group = NcclCommunicator::from_resolved(Some(&resolved), e.stream.context(), id)?
            .ok_or("expected NCCL")?;
        let map = MappedGguf::open(Path::new(&args[3]))?;
        let dtype = KvDtype::Fp16;
        let mut tp = GqaTpRank::load(&e, &map, layer, &group, 16, dtype)?;
        let mut serial = if rank == 0 {
            Some(Oracle::new(&e, &map, layer, 16, dtype)?)
        } else {
            None
        };
        println!(
            "rank={rank} layer={layer} geometry={:?} local_kv_bytes={}",
            tp.geometry,
            tp.local_kv_bytes()
        );
        let mut replay = Vec::new();
        for pass in 0..2 {
            tp.reset();
            if let Some(s) = &mut serial {
                s.reset();
            }
            for step in 0..4 {
                let host: Vec<f32> = (0..tp.geometry.width)
                    .map(|i| ((i * 31 + step * 7 + 3) % 127) as f32 / 63.0 - 1.0)
                    .collect();
                let x = e.to_device(&host)?;
                let expected = if let Some(s) = &mut serial {
                    Some(s.forward(&e, &x)?)
                } else {
                    None
                };
                let got = e.to_host(tp.forward(&e, &group, &x)?)?;
                if pass == 0 {
                    replay.push(got.clone());
                } else if got != replay[step] {
                    return Err(format!("reset replay mismatch step={step}").into());
                }
                if !got.iter().all(|v| v.is_finite()) {
                    return Err("non-finite output".into());
                }
                if let Some(expected) = expected {
                    let mut max_abs = 0.0_f32;
                    for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
                        let delta = (a - b).abs();
                        max_abs = max_abs.max(delta);
                        if delta > 1e-3 + 1e-3 * b.abs() {
                            return Err(
                                format!("pass={pass} step={step} i={i} tp={a} serial={b}").into()
                            );
                        }
                    }
                    println!(
                        "pass={pass} step={step} GQA parity PASS max_abs={max_abs:.8} checksum={:.6}",
                        got.iter().sum::<f32>()
                    );
                } else {
                    println!(
                        "rank=1 pass={pass} step={step} checksum={:.6}",
                        got.iter().sum::<f32>()
                    );
                }
            }
            if tp.position() != 4 {
                return Err("position not advanced".into());
            }
        }
        group.stream().synchronize()?;
        e.synchronize()?;
        Ok(())
    })();
    if rank == 0 {
        shutdown_worker(&mut control, result.is_ok())?;
    } else {
        match ControlMessage::from_stream(&mut control)? {
            ControlMessage::Shutdown { graceful } if graceful == result.is_ok() => {}
            other => return Err(format!("unexpected shutdown {other:?}").into()),
        }
    }
    result?;
    println!("rank={rank} GQA probe shut down cleanly");
    Ok(())
}
