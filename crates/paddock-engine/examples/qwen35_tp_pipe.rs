//! Two-rank production-generator pure-decode probe. Worker first, then head:
//! qwen35_tp_pipe RANK MASTER_IP MODEL PACK [PORT]
use paddock_dist::{
    config::ParallelConfig,
    worker::{connect_worker, coordinate},
};
use paddock_engine::{
    generator::{Generator, RowSample},
    gpu_model::qwen35::tp_serve::{TpGenerator, run_worker},
    sampler::DevicePlan,
};
use std::{error::Error, path::Path};

fn greedy(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .expect("non-empty logit row")
        .0 as u32
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_tp_pipe RANK MASTER_IP MODEL PACK [PORT]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank must be 0 or 1".into());
    }
    let resolved = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some(args[2].clone()),
        master_port: Some(args.get(5).map_or(Ok(18567), |p| p.parse())?),
    }
    .resolved(false)?
    .ok_or("TP=2 required")?;
    let (stream, _) = if rank == 0 {
        coordinate(&resolved, false)?
    } else {
        connect_worker(&resolved)?
    };
    let model = Path::new(&args[3]);
    let pack = Path::new(&args[4]);
    if rank == 1 {
        run_worker(stream, &resolved, model, pack, 0)?;
        return Ok(());
    }
    let mut engine = TpGenerator::load(stream, resolved, model, pack, 0, 48, 2)?;
    engine.enable_batch(2)?;
    if !engine.supports_device_sampling() {
        return Err("sample_rows unavailable".into());
    }
    let vocab = engine.vocab();
    let plans = [RowSample::Device(DevicePlan::Greedy); 2];
    let prompts = [[1, 2], [4, 5]];
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    let mut tokens = [103, 203];
    let mut expected = Vec::new();
    for position in 2..=20 {
        let logits = engine.forward_batch(&tokens, &[position; 2])?;
        tokens = [greedy(&logits[..vocab]), greedy(&logits[vocab..])];
        expected.push(tokens);
    }
    engine.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    engine.decode_pipe_begin(&[103, 203], &[2, 2], &plans)?;
    for (i, want) in expected.iter().enumerate().take(expected.len() - 1) {
        let ids = engine.decode_pipe_next(&plans)?;
        assert_eq!(ids, want, "pipe mismatch at position {}", i + 2);
    }
    assert_eq!(
        engine.decode_pipe_drain()?,
        expected.last().expect("drain produced rows").as_slice()
    );
    engine.release_inactive_slots(&[true, false]);
    engine.forward_prefill(1, &[301, 302])?;
    let after = engine.forward_batch(
        &[expected.last().expect("drain produced rows")[0], 303],
        &[21, 2],
    )?;
    assert_eq!(after.len(), 2 * vocab);
    engine.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    engine.decode_pipe_begin(&[103, 203], &[2, 2], &plans)?;
    assert_eq!(engine.decode_pipe_drain()?, expected[0].as_slice());
    println!(
        "TP production pipe: eager greedy IDs match through position 20 and KV page; two-rank drain, release/reuse and reset/replay passed"
    );
    Ok(())
}
