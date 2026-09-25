//! Two-rank direct Stage-2 span/pipe oracle. Worker first, then head:
//! qwen35_tp_overlap RANK MASTER_IP MODEL PACK [PORT]
use paddock_dist::{
    config::ParallelConfig,
    worker::{connect_worker, coordinate},
};
use paddock_engine::{
    generator::{FinishSample, Generator, RowSample},
    gpu_model::qwen35::tp_serve::{run_worker, TpGenerator},
    sampler::DevicePlan,
};
use std::{error::Error, path::Path};

fn same_logits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "logit {i}: {x} != {y}");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: qwen35_tp_overlap RANK MASTER_IP MODEL PACK [PORT]".into());
    }
    let rank: usize = args[1].parse()?;
    if rank > 1 {
        return Err("rank must be 0 or 1".into());
    }
    let resolved = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some(args[2].clone()),
        master_port: Some(args.get(5).map_or(Ok(18568), |p| p.parse())?),
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
    assert!(
        engine.supports_overlap(),
        "TP=2 device-sampled overlap capability is required"
    );
    let cat = RowSample::Device(DevicePlan::Categorical {
        inv_t: 1.25,
        u: 0.37,
    });
    let hole = RowSample::Hole;
    let prompt: Vec<u32> = (4..24).collect();
    engine.forward_prefill(0, &[1, 2])?;
    engine.forward_prefill(1, &prompt[..prompt.len() - 1])?;
    let decode = engine.forward_batch_sampled(&[103, 0], &[2, 0], &[cat, hole])?;
    let expected_decode = decode.ids[0];
    let final_row =
        engine
        .forward_batch_sampled(&[0, *prompt.last().expect("non-empty prompt")], &[0, 19], &[hole, cat])?;
    let expected_finisher = final_row.ids[1];
    let expected_next = engine.forward_batch(&[expected_decode, expected_finisher], &[3, 20])?;
    engine.release_inactive_slots(&[true, false]);
    engine.forward_prefill(1, &[301, 302])?;
    let expected_reused = engine.forward_batch(&[103, 303], &[4, 2])?;

    engine.reset();
    engine.forward_prefill(0, &[1, 2])?;
    engine.prefill_begin(1, prompt)?;
    assert!(engine.unified_span_launch(32, &[(1, cat)])?);
    assert!(
        !engine.prefill_abort(1),
        "active span cannot abort/reuse its slot"
    );
    engine.decode_pipe_begin_slots(&[0], &[103], &[2], &[cat])?;
    let got_decode = engine.decode_pipe_drain()?;
    assert_eq!(got_decode, [expected_decode], "categorical decode ID");
    let finished = engine.unified_span_finish()?;
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].0, 1);
    assert_eq!(finished[0].2, 20);
    let got_finisher = match finished[0].1 {
        FinishSample::Sampled(id) => id,
        _ => return Err("finisher was not device sampled".into()),
    };
    assert_eq!(got_finisher, expected_finisher, "categorical final-row ID");
    let got_next = engine.forward_batch(&[got_decode[0], got_finisher], &[3, 20])?;
    same_logits(&got_next, &expected_next);
    engine.release_inactive_slots(&[true, false]);
    engine.forward_prefill(1, &[301, 302])?;
    let reused = engine.forward_batch(&[103, 303], &[4, 2])?;
    same_logits(&reused, &expected_reused);

    // A sampled finisher returns only an ID; compare every final-row logit
    // via a separate host-finisher span against a reset eager prefill.
    let host_prompt: Vec<u32> = (40..58).collect();
    engine.reset();
    engine.forward_prefill(0, &[1, 2])?;
    let expected_final_logits = engine.forward_prefill(1, &host_prompt)?;
    engine.reset();
    engine.forward_prefill(0, &[1, 2])?;
    engine.prefill_begin(1, host_prompt)?;
    assert!(engine.unified_span_launch(32, &[(1, RowSample::Host)])?);
    engine.decode_pipe_begin_slots(&[0], &[103], &[2], &[cat])?;
    assert_eq!(engine.decode_pipe_drain()?, [expected_decode]);
    let finished = engine.unified_span_finish()?;
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].0, 1);
    assert_eq!(finished[0].2, 18);
    let FinishSample::Logits(ref final_logits) = finished[0].1 else {
        return Err("host finisher did not return logits".into());
    };
    same_logits(final_logits, &expected_final_logits);

    // The queue admits both chunks but grants one contiguous slot per span.
    // Finishing the first must leave the second queued and its state intact.
    engine.reset();
    let expected_a = engine.forward_prefill(0, &[61, 62, 63])?;
    let expected_b = engine.forward_prefill(1, &[71, 72, 73, 74])?;
    engine.reset();
    engine.prefill_begin(0, vec![61, 62, 63])?;
    engine.prefill_begin(1, vec![71, 72, 73, 74])?;
    for (slot, expected) in [(0, expected_a), (1, expected_b)] {
        assert!(engine.unified_span_launch(32, &[(0, RowSample::Host), (1, RowSample::Host)])?);
        let finished = engine.unified_span_finish()?;
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0, slot);
        let FinishSample::Logits(ref logits) = finished[0].1 else {
            return Err("queued finisher did not return logits".into());
        };
        same_logits(logits, &expected);
    }
    println!(
        "TP Stage-2 direct oracle: categorical decode/finisher IDs, full final-row and next-row logits, page crossing, drain and release/reuse passed"
    );
    Ok(())
}
