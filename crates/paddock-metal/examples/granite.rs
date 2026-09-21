//! Native-GPU bring-up and same-GGUF parity harness. No CPU model execution.
//! cargo run --release -p paddock-metal --example granite -- MODEL PROMPT [N]
#[cfg(target_os = "macos")]
use paddock_engine::generator::Generator;
#[cfg(target_os = "macos")]
use paddock_models::mapped::MappedGguf;
#[cfg(target_os = "macos")]
use paddock_tokenizer::GgufTokenizer;
#[cfg(target_os = "macos")]
use std::{path::Path, time::Instant};

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the Granite Metal example requires an M5 Mac");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(args.get(1).ok_or("usage: granite MODEL PROMPT [N]")?);
    let prompt = args.get(2).ok_or("missing prompt")?;
    let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16);
    let mapped = MappedGguf::open(path)?;
    let tokenizer = GgufTokenizer::from_gguf(mapped.gguf())?;
    let tokens: Vec<u32> = if prompt.starts_with('[') {
        serde_json::from_str(prompt)?
    } else {
        tokenizer.encode(prompt)?
    };
    drop(mapped);
    let started = Instant::now();
    let mut model = paddock_metal::Granite::load(path, (tokens.len() + n + 1).max(512), 2, None)?;
    eprintln!("loaded in {:.2}s", started.elapsed().as_secs_f64());
    let t = Instant::now();
    let mut logits = model.forward_prefill_stream(&tokens)?;
    let prefill_seconds = t.elapsed().as_secs_f64();
    let mut generated = Vec::new();
    let mut steps = Vec::new();
    let t = Instant::now();
    for i in 0..n {
        if logits.len() != tokenizer.vocab_size || logits.iter().any(|x| !x.is_finite()) {
            return Err("invalid or nonfinite GPU logits".into());
        }
        let mut ranked: Vec<usize> = (0..logits.len()).collect();
        ranked.select_nth_unstable_by(9, |&a, &b| logits[b].total_cmp(&logits[a]));
        ranked[..10].sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let id = ranked[0] as u32;
        steps.push(serde_json::json!({"id": id, "top": ranked[..10].iter().map(|&j| (j, logits[j])).collect::<Vec<_>>()}));
        generated.push(id);
        if i + 1 < n {
            logits = model.forward(id)?;
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "prompt_tokens": tokens, "generated_tokens": generated,
            "text": tokenizer.decode(&generated, false)?, "steps": steps,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": t.elapsed().as_secs_f64(),
            "allocated_bytes": model.device_mem_used(),
        })
    );
    Ok(())
}
