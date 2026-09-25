//! Direct two-rank TP=2 speculative oracle for Qwen3.8-27B.
//! Run rank 1 first, then rank 0:
//! qwen35_tp_spec RANK MASTER_IP MODEL PACK [PORT]
use paddock_dist::{
    config::ParallelConfig,
    worker::{connect_worker, coordinate},
};
use paddock_engine::{
    generator::{Generator, RowSample},
    gpu_model::qwen35::tp_serve::{run_worker, TpGenerator},
    sampler::DevicePlan,
};
use std::{error::Error, path::Path};

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .expect("non-empty logits")
        .0 as u32
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_tp_spec RANK MASTER_IP MODEL PACK [PORT]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank must be 0 or 1".into());
    }
    let port: u16 = args.get(5).map_or(Ok(18569), |p| p.parse())?;
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
    engine.enable_batch(2)?;
    let vocab = engine.vocab();
    let prompts = [[1u32, 2], [4, 5]];
    let pending = [103u32, 203];

    // Establish the accepted non-spec TP=2 oracle for both slots.
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    let baseline_logits = engine.forward_batch(&pending, &[2, 2])?;
    let first = [
        argmax(&baseline_logits[..vocab]),
        argmax(&baseline_logits[vocab..]),
    ];
    let mut greedy_tokens = first;
    let mut greedy_oracle = [[0u32; 4]; 2];
    greedy_oracle[0][0] = first[0];
    greedy_oracle[1][0] = first[1];
    for step in 1usize..4 {
        let logits = engine.forward_batch(&greedy_tokens, &[(2 + step) as u32; 2])?;
        greedy_tokens = [argmax(&logits[..vocab]), argmax(&logits[vocab..])];
        greedy_oracle[0][step] = greedy_tokens[0];
        greedy_oracle[1][step] = greedy_tokens[1];
    }

    // Full acceptance: pending + three exact oracle-derived draft tokens.
    engine.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    let full = engine
        .forward_spec_batch(&[
            (
                0,
                2,
                vec![
                    pending[0],
                    greedy_oracle[0][0],
                    greedy_oracle[0][1],
                    greedy_oracle[0][2],
                ],
            ),
            (
                1,
                2,
                vec![
                    pending[1],
                    greedy_oracle[1][0],
                    greedy_oracle[1][1],
                    greedy_oracle[1][2],
                ],
            ),
        ])?
        .ok_or("TP speculative hook unavailable")?;
    assert_eq!(full.len(), 8);

    // Zero acceptance: the first draft token is deliberately wrong. The
    // returned vector is padded, but no padded target row may execute.
    engine.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    let wrong = [
        greedy_oracle[0][0].wrapping_add(1),
        greedy_oracle[1][0].wrapping_add(1),
    ];
    let zero = engine
        .forward_spec_batch(&[
            (0, 2, vec![pending[0], wrong[0]]),
            (1, 2, vec![pending[1], wrong[1]]),
        ])?
        .ok_or("TP speculative hook unavailable")?;
    assert_eq!(zero.len(), 4);
    assert_eq!(zero[0], greedy_oracle[0][0]);
    assert_eq!(zero[2], greedy_oracle[1][0]);

    // Partial acceptance: the first draft is right and the second is wrong.
    engine.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    let partial = engine
        .forward_spec_batch(&[
            (0, 2, vec![pending[0], greedy_oracle[0][0], wrong[0]]),
            (1, 2, vec![pending[1], greedy_oracle[1][0], wrong[1]]),
        ])?
        .ok_or("TP speculative hook unavailable")?;
    assert_eq!(partial.len(), 6);
    assert_eq!(partial[0], greedy_oracle[0][0]);
    assert_eq!(partial[3], greedy_oracle[1][0]);

    // The next committed token must be the target bonus token at position 4;
    // this catches accidental execution of the rejected suffix and therefore
    // checks the rank-paired KV/state position advance.
    let after = engine.forward_batch(&[partial[1], partial[4]], &[4, 4])?;
    assert_eq!(after.len(), 2 * vocab);

    // Sampled oracle: fixed categorical plans generate the draft sequence
    // first, then the same flattened plans are replayed through verification.
    let categorical = DevicePlan::Categorical {
        inv_t: 1.25,
        u: 0.37,
    };
    engine.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    let mut sampled_tokens = pending;
    let mut sampled_oracle = [[0u32; 4]; 2];
    for step in 0usize..4 {
        let result = engine.forward_batch_sampled(
            &sampled_tokens,
            &[(2 + step) as u32, (2 + step) as u32],
            &[
                RowSample::Device(categorical),
                RowSample::Device(categorical),
            ],
        )?;
        sampled_tokens = [result.ids[0], result.ids[1]];
        sampled_oracle[0][step] = result.ids[0];
        sampled_oracle[1][step] = result.ids[1];
    }
    engine.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        engine.forward_prefill(slot, prompt)?;
    }
    let sampled_plans = vec![categorical; 8];
    let sampled = engine
        .forward_spec_batch_plans(
            &[
                (
                    0,
                    2,
                    vec![
                        pending[0],
                        sampled_oracle[0][0],
                        sampled_oracle[0][1],
                        sampled_oracle[0][2],
                    ],
                ),
                (
                    1,
                    2,
                    vec![
                        pending[1],
                        sampled_oracle[1][0],
                        sampled_oracle[1][1],
                        sampled_oracle[1][2],
                    ],
                ),
            ],
            &sampled_plans,
        )?
        .ok_or("TP sampled speculative hook unavailable")?;
    assert_eq!(sampled.len(), 8);
    assert_eq!(&sampled[0..4], &sampled_oracle[0]);
    assert_eq!(&sampled[4..8], &sampled_oracle[1]);
    println!(
        "TP speculative direct oracle: zero/partial/full acceptance, sampled fixed-plan replay, padded picks, committed-position advance, and both-rank replay passed; vocab={vocab}"
    );
    Ok(())
}
