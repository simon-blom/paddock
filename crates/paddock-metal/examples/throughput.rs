//! GPU-only native-path diagnostic, not a serving comparison. Fixed token IDs
//! hold tokenization constant. Prefill uses the real mixed scheduler contract.
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("requires an M5 Mac");
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use paddock_engine::generator::Generator;
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(
        args.get(1)
            .ok_or("throughput MODEL C PROMPT_TOKENS DECODE_STEPS")?,
    );
    let c: usize = args.get(2).ok_or("missing concurrency")?.parse()?;
    let p: usize = args.get(3).ok_or("missing prompt length")?.parse()?;
    let n: usize = args.get(4).ok_or("missing decode steps")?.parse()?;
    if c == 0 || p == 0 || n == 0 {
        return Err("positive sizes required".into());
    }
    let context = args
        .get(5)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or((p + n + 8).max(512));
    if context < p + n + 8 {
        return Err("context must cover prompt, warmup and decode".into());
    }
    let mut model = paddock_metal::Granite::load(path, context, c, None)?;
    let vocab = model.vocab();
    let argmax = |logits: &[f32]| -> u32 {
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|x| x.is_finite()));
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("nonempty validated vocabulary")
            .0 as u32
    };
    let mut tokens = vec![0; c];
    let start = Instant::now();
    for slot in 0..c {
        model.prefill_begin(
            slot,
            (0..p)
                .map(|i| 100 + ((i + slot * 7) % 100) as u32)
                .collect(),
        )?;
    }
    let mut finished = 0;
    let mut prefill_gpu = 0.0;
    while finished < c {
        let (_, done) = model.forward_mixed(&[], usize::MAX)?;
        prefill_gpu += model.last_gpu_seconds;
        for (slot, logits, _) in done {
            tokens[slot] = argmax(&logits);
            finished += 1;
        }
    }
    let prefill_seconds = start.elapsed().as_secs_f64();
    let mut position = p as u32;
    for _ in 0..4 {
        let logits = model.forward_batch(&tokens, &vec![position; c])?;
        for (slot, row) in logits.chunks_exact(vocab).enumerate() {
            tokens[slot] = argmax(row);
        }
        position += 1;
    }
    let start = Instant::now();
    let mut gpu_seconds = 0.0;
    for _ in 0..n {
        let logits = model.forward_batch(&tokens, &vec![position; c])?;
        gpu_seconds += model.last_gpu_seconds;
        for (slot, row) in logits.chunks_exact(vocab).enumerate() {
            tokens[slot] = argmax(row);
        }
        position += 1;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"concurrency":c,"prompt_tokens":p,"steps":n,"context":context,
        "prefill_seconds":prefill_seconds,"prefill_gpu_seconds":prefill_gpu,
        "decode_seconds":seconds,"gpu_seconds":gpu_seconds,
        "aggregate_tps":(c*n) as f64/seconds,"per_request_tps":n as f64/seconds,
        "allocated_bytes":model.device_mem_used()})
    );
    Ok(())
}
