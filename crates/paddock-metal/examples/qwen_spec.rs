//! Internal same-model spec/off diagnostic, not a rival serving scoreboard.
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("requires macOS")
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use paddock_engine::generator::Generator;
    use paddock_models::mapped::MappedGguf;
    use paddock_tokenizer::GgufTokenizer;
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 6 {
        return Err("qwen_spec MODEL off|mtp|DFLASH_PATH C N PROMPT [DRAFT_K] [VERIFY_K]".into());
    }
    let path = Path::new(&args[1]);
    let mode = &args[2];
    let c: usize = args[3].parse()?;
    let n: usize = args[4].parse()?;
    let tok = if path.is_dir() {
        GgufTokenizer::from_hf_dir(&if path.join("manifest.json").is_file() {
            path.join("tokenizer")
        } else {
            path.to_path_buf()
        })?
    } else {
        let map = MappedGguf::open(path)?;
        GgufTokenizer::from_gguf(map.gguf())?
    };
    let p = tok.encode(&args[5])?;
    let k: usize = args
        .get(6)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(if mode == "mtp" { 3 } else { 7 });
    let verify_k: usize = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(k);
    let pick = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .expect("nonempty vocabulary")
            .0 as u32
    };
    let mut model = paddock_metal::Qwen35::load(path, 4096, c, None)?;
    if mode == "mtp" {
        model.attach_mtp(path)?;
    } else if mode != "off" {
        model.attach_dflash(Path::new(mode))?;
    }
    let started = Instant::now();
    let mut outputs = vec![Vec::new(); c];
    let mut pos = vec![p.len() as u32; c];
    for (i, out) in outputs.iter_mut().enumerate() {
        out.push(pick(&model.forward_prefill(i, &p)?));
    }
    let prefill = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut rounds = 0;
    let mut accepted = 0;
    let mut proposed = 0;
    let mut draft_seconds = 0.;
    let mut verify_seconds = 0.;
    while outputs.iter().any(|o| o.len() < n) {
        let live: Vec<_> = outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| o.len() < n)
            .map(|(i, o)| (i, *o.last().expect("prefilled slot")))
            .collect();
        if mode == "off" {
            let mut tokens = vec![0; c];
            let mut positions = vec![0; c];
            for &(i, t) in &live {
                tokens[i] = t;
                positions[i] = pos[i];
            }
            let logits = model.forward_batch(&tokens, &positions)?;
            for &(i, _) in &live {
                outputs[i].push(pick(&logits[i * model.vocab()..(i + 1) * model.vocab()]));
                pos[i] += 1;
            }
        } else {
            let t = Instant::now();
            let drafts = model
                .spec_draft_batch(&live, k)?
                .ok_or("drafter declined")?;
            draft_seconds += t.elapsed().as_secs_f64();
            let reqs: Vec<_> = live
                .iter()
                .zip(drafts)
                .map(|(&(i, t), d)| {
                    let chunk = std::iter::once(t)
                        .chain(d.into_iter().take(verify_k.min(n - outputs[i].len() - 1)))
                        .collect();
                    (i, pos[i] as usize, chunk)
                })
                .collect();
            let t = Instant::now();
            let picks = model.forward_spec_batch(&reqs)?.ok_or("verify declined")?;
            verify_seconds += t.elapsed().as_secs_f64();
            let mut base = 0;
            for (i, _, chunk) in reqs {
                let count = 1 + chunk[1..]
                    .iter()
                    .zip(&picks[base..])
                    .take_while(|(a, b)| a == b)
                    .count();
                outputs[i].extend_from_slice(&picks[base..base + count]);
                pos[i] += count as u32;
                accepted += count - 1;
                proposed += chunk.len() - 1;
                base += chunk.len();
            }
        }
        rounds += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"mode":mode,"concurrency":c,"n":n,"k":k,"verify_k":verify_k,"prompt_tokens":p.len(),"prefill_seconds":prefill,"decode_seconds":elapsed,
        "decode_tps":(c*(n-1)) as f64/elapsed,"rounds":rounds,"accepted":accepted,"proposed":proposed,"draft_seconds":draft_seconds,"verify_seconds":verify_seconds,
        "allocated_bytes":model.device_mem_used(),"tokens":outputs,"text":outputs.iter().map(|o|tok.decode(o,false)).collect::<Result<Vec<_>,_>>()?})
    );
    Ok(())
}
