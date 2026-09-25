//! Two-rank production generator device-sampling probe:
//! qwen35_tp_sampled RANK MASTER_IP MODEL PACK [PORT]
use paddock_dist::{
    config::ParallelConfig,
    worker::{connect_worker, coordinate},
};
use paddock_engine::sampler::DevicePlan;
use paddock_engine::{
    generator::{Generator, RowSample},
    gpu_model::qwen35::tp_serve::{TpGenerator, run_worker},
};
use std::{error::Error, path::Path};

fn greedy(logits: &[f32]) -> u32 {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .expect("non-empty logit row");
    idx as u32
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_tp_sampled RANK MASTER_IP MODEL PACK [PORT]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank must be 0 or 1".into());
    }
    let port: u16 = args.get(5).map_or(Ok(11565), |s| s.parse())?;
    let resolved = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some(args[2].clone()),
        master_port: Some(port),
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
    if !engine.supports_device_sampling() {
        return Err("pack missing sample_rows".into());
    }
    engine.enable_batch(2)?;
    engine.forward_prefill(0, &[1, 2])?;
    engine.forward_prefill(1, &[4, 5])?;
    let baseline = engine.forward_batch(&[103, 203], &[2, 2])?;
    let vocab = engine.vocab();
    let expected = greedy(&baseline[..vocab]);
    let host_expected = baseline[vocab..].to_vec();
    engine.reset();
    engine.forward_prefill(0, &[1, 2])?;
    engine.forward_prefill(1, &[4, 5])?;
    let sampled = engine.forward_batch_sampled(
        &[103, 203],
        &[2, 2],
        &[RowSample::Device(DevicePlan::Greedy), RowSample::Host],
    )?;
    assert_eq!(sampled.ids.len(), 2);
    assert_eq!(sampled.ids[0], expected);
    assert_eq!(sampled.host_rows.len(), 1);
    assert_eq!(sampled.host_rows[0].0, 1);
    assert!(
        sampled.host_rows[0]
            .1
            .iter()
            .zip(&host_expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    engine.release_inactive_slots(&[true, false]);
    engine.forward_prefill(1, &[301, 302])?;
    let mixed = engine.forward_batch_sampled(
        &[104, 303],
        &[3, 2],
        &[RowSample::Host, RowSample::Device(DevicePlan::Greedy)],
    )?;
    assert_eq!(mixed.host_rows.len(), 1);
    assert_eq!(mixed.host_rows[0].0, 0);
    assert!(mixed.ids[1] < vocab as u32);
    // A fixed rank-0 draw must replay exactly after resetting both ranks.
    let categorical = RowSample::Device(DevicePlan::Categorical {
        inv_t: 1.0 / 0.8,
        u: 0.37,
    });
    let first =
        engine.forward_batch_sampled(&[105, 304], &[4, 3], &[categorical, RowSample::Host])?;
    assert!(first.ids[0] < vocab as u32);
    engine.reset();
    engine.forward_prefill(0, &[1, 2])?;
    engine.forward_prefill(1, &[301, 302])?;
    engine.forward_batch_sampled(
        &[103, 303],
        &[2, 2],
        &[RowSample::Device(DevicePlan::Greedy), RowSample::Host],
    )?;
    engine.forward_batch_sampled(
        &[104, 304],
        &[3, 3],
        &[RowSample::Host, RowSample::Device(DevicePlan::Greedy)],
    )?;
    let replay =
        engine.forward_batch_sampled(&[105, 304], &[4, 4], &[categorical, RowSample::Host])?;
    // This separate replay has a different rank-1 history, but slot 0
    // remains independent of that slot's token and sampler selection.
    assert_eq!(first.ids[0], replay.ids[0]);
    engine.release_inactive_slots(&[false, true]);
    let hole = engine.forward_batch_sampled(
        &[0, 305],
        &[0, 5],
        &[RowSample::Hole, RowSample::Device(DevicePlan::Greedy)],
    )?;
    assert_eq!(hole.ids.len(), 2);
    assert_eq!(hole.ids[0], 0);
    assert!(hole.host_rows.is_empty());
    // Keep the surviving slot on the sampled path across the 16-token
    // logical KV page while slot 0 remains empty.
    let mut next = hole.ids[1];
    for position in 6..=20 {
        let step = engine.forward_batch_sampled(
            &[0, next],
            &[0, position],
            &[RowSample::Hole, RowSample::Device(DevicePlan::Greedy)],
        )?;
        assert!(step.host_rows.is_empty());
        assert!(step.ids[1] < vocab as u32);
        next = step.ids[1];
    }
    println!(
        "TP device sampling: greedy={} host_exact=true categorical_replay=true release_reuse=true sparse_hole=true kv_page_crossed=true",
        sampled.ids[0]
    );
    Ok(())
}
