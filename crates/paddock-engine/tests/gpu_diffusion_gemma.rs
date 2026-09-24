//! DiffusionGemma 26B-A4B on the GPU: the gemma4 body with the block-
//! diffusion lane. Two gates on the unsloth Q8_0 file:
//!
//!   - a STRUCTURED READ: the response template `answer: <slot>` seeded
//!     into a narrow canvas after a yes/no question, one forward, the label
//!     distribution read at the slot (the `/v1/systemone` primitive);
//!   - a BLOCK GENERATION: a random 256-canvas denoised at the authors'
//!     schedule until stable and confident (or 48 steps), committed, decoded.
//!
//! Heavy (one 27 GB upload). Inputs, with the env overrides the other gates
//! use: PADDOCK_DGEMMA (the GGUF; default the elected Q8_0 under the models
//! root), PADDOCK_DG_STEPS (cap, default the file's 48), PADDOCK_DG_SEED
//! (default 42), PADDOCK_DG_PROMPT, PADDOCK_DG_TEMP (default: the schedule;
//! 0 = deterministic).

mod common;

use std::sync::Arc;
use std::time::Instant;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const DGEMMA_Q8: &[&str] = &[
    "diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q8_0.gguf",
    "diffusiongemma-26B-A4B-it-Q8_0.gguf",
];

/// `<bos><|turn>user\n{text}<turn|>\n<|turn>model\n` - the Gemma 4 turn
/// markers the file's own chat template renders; the runner does this with
/// the template, the gate spells it out.
fn chat_prompt(tok: &GgufTokenizer, user: &str) -> Vec<u32> {
    let text = format!("<|turn>user\n{user}<turn|>\n<|turn>model\n");
    let mut ids = vec![2u32]; // <bos>
    ids.extend(tok.encode(&text).expect("encode"));
    ids
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn load() -> Option<(
    Arc<paddock_engine::gpu::GpuExecutor>,
    GpuGemma4,
    GgufTokenizer,
)> {
    if !common::heavy() {
        return None;
    }
    let exec = common::gpu_arc()?;
    let path = common::model("PADDOCK_DGEMMA", DGEMMA_Q8)?;
    let map = MappedGguf::open(&path).expect("map gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let free_gb = |e: &paddock_engine::gpu::GpuExecutor| {
        e.device_mem_info()
            .map_or(f64::NAN, |(free, _)| free as f64 / 1e9)
    };
    let before = free_gb(&exec);
    let t0 = Instant::now();
    let model = GpuGemma4::load_with(exec.clone(), &map, 4096, None).expect("load");
    // the device-memory bracket the catalog row is priced from: weights as
    // the family accounts them (body + the diffusion lane) and what the
    // load actually took off the card
    eprintln!(
        "loaded {} in {:.1}s: canvas {} weights {:?} B; free VRAM {before:.2} -> {:.2} GB",
        path.display(),
        t0.elapsed().as_secs_f32(),
        model.canvas_len(),
        model.weights_mem_bytes(),
        free_gb(&exec)
    );
    Some((exec, model, tok))
}

/// A JSON array of integers, read without a JSON dependency: the reference
/// probe writes `[2, 106, ...]` and nothing else.
fn read_id_list(path: &std::path::Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().expect("an id"))
        .collect()
}

/// `PADDOCK_DG_DUMP=<dir>`: the reference probe. The reference script
/// wrote `prompt_ids.json` and `canvas.json` there and ran two decoder passes
/// over that fixed canvas on the bf16 checkpoint (pass 1 unconditioned,
/// pass 2 self-conditioned on pass 1). The same two passes here, at
/// temperature 1, probs written as `paddock_probs{1,2}.bin` (f32 `[w][vocab]`
/// little-endian) for the compare script. No RNG on either side.
#[test]
fn diffusion_gemma_dumps_the_reference_probe() {
    let Ok(dir) = std::env::var("PADDOCK_DG_DUMP") else {
        eprintln!("SKIP: PADDOCK_DG_DUMP unset (the reference probe is opt-in)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let Some((exec, mut model, _tok)) = load() else {
        return;
    };
    let prompt = read_id_list(&dir.join("prompt_ids.json"));
    let canvas = read_id_list(&dir.join("canvas.json"));
    let base = prompt.len();
    model.reset();
    model.forward_prefill_stream(&prompt).expect("prefill");
    let mut st = model.canvas_new(canvas.len()).expect("canvas");
    model.canvas_seed(&mut st, &canvas).expect("seed");
    for pass in 1..=2u32 {
        model.canvas_read(0, base, &mut st, &[]).expect("read");
        let probs = exec.to_host(&st.probs).expect("probs");
        let bytes: Vec<u8> = probs.iter().flat_map(|p| p.to_le_bytes()).collect();
        let path = dir.join(format!("paddock_probs{pass}.bin"));
        std::fs::write(&path, bytes).expect("write probs");
        eprintln!(
            "pass {pass}: argmax {:?} -> {}",
            st.last_argmax,
            path.display()
        );
        // the seeded canvas stays the same for pass 2 (the reference feeds
        // the identical decoder_input_ids); only the self-conditioning changes
        model.canvas_seed(&mut st, &canvas).expect("re-seed");
    }
}

/// The structured read. The template `answer: yes` / `answer: no` must
/// tokenize identically except at ONE position (the example server's
/// `resolve_template` rule); the canvas is that template plus `<turn|>`,
/// padded to 16, the slot left to noise, everything else seeded.
#[test]
fn diffusion_gemma_reads_a_yes_no_slot() {
    let Some((exec, mut model, tok)) = load() else {
        return;
    };
    let prompt = chat_prompt(
        &tok,
        "Answer with the single word yes or no, in the form `answer: <word>`.\n\
         Question: Is water wet?",
    );
    let yes = tok.encode("answer: yes").expect("encode");
    let no = tok.encode("answer: no").expect("encode");
    assert_eq!(
        yes.len(),
        no.len(),
        "labels must tokenize to one token each"
    );
    let diff: Vec<usize> = (0..yes.len()).filter(|&i| yes[i] != no[i]).collect();
    assert_eq!(
        diff.len(),
        1,
        "labels must differ at exactly one position: {diff:?}"
    );
    let slot = diff[0];
    let (id_yes, id_no) = (yes[slot], no[slot]);

    // prefill the prompt (single-stream path), then the read canvas after it
    let base = prompt.len();
    model.reset();
    model.forward_prefill_stream(&prompt).expect("prefill");

    // Two heads for the same template. The one that reads is the EMPTY
    // THOUGHT CHANNEL the model opens every answer with - vLLM's example
    // server seeds exactly `enc("<|channel>thought\n") + enc("<channel|>")`
    // (its SCAFFOLD) ahead of the template when thinking is off. Measured
    // 2026-09-23 on the Q8_0 file: with it the slot reads " yes" 0.78 /
    // " no" 0.02 (80 % of the mass on the two labels); bare, the model wants
    // a newline at the slot (0.84) and puts 0.9 % on the labels - the bare
    // row stays as the diagnostic that shows why the scaffold is not
    // optional.
    let mut scaffold = tok.encode("<|channel>thought\n").expect("encode");
    scaffold.extend(tok.encode("<channel|>").expect("encode"));
    let heads: Vec<(&str, Vec<u32>)> = vec![("bare", Vec::new()), ("scaffold", scaffold)];

    let mut best: Option<(f32, f32)> = None;
    for (name, head) in &heads {
        let mut canvas = head.clone();
        canvas.extend(&yes);
        canvas.push(106); // <turn|>
        let w = canvas.len().next_multiple_of(16);
        canvas.resize(w, 0); // <pad>
        let pos = head.len() + slot;
        canvas[pos] = 12345; // the slot is noise
        let mut st = model.canvas_new(w).expect("canvas");
        model.canvas_seed(&mut st, &canvas).expect("seed");
        let t0 = Instant::now();
        let probs = model
            .canvas_read(0, base, &mut st, &[id_yes, id_no])
            .expect("read");
        let (p_yes, p_no) = (probs[pos * 2], probs[pos * 2 + 1]);
        // the slot's top 5, decoded, so a wrong template reads as words
        let row = exec.to_host(&st.probs).expect("probs");
        let mut top: Vec<(u32, f32)> = row[pos * vocab_of(&model)..(pos + 1) * vocab_of(&model)]
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top5: Vec<String> = top[..5]
            .iter()
            .map(|(id, p)| {
                format!(
                    "{id}={:?}:{p:.3}",
                    tok.decode(&[*id], false).unwrap_or_default()
                )
            })
            .collect();
        eprintln!(
            "[{name}] read in {:.2}s: w {w} slot {pos} p(yes) {p_yes:.4} p(no) {p_no:.4} \
             entropy {:.4} top5 {}",
            t0.elapsed().as_secs_f32(),
            st.last_entropy[pos],
            top5.join(" ")
        );
        assert!(p_yes.is_finite() && p_no.is_finite());
        if best.is_none_or(|(y, n)| p_yes + p_no > y + n) {
            best = Some((p_yes, p_no));
        }
    }
    let (p_yes, p_no) = best.expect("a head ran");
    assert!(
        p_yes + p_no > 0.5,
        "the label set should hold most of the slot's mass: {}",
        p_yes + p_no
    );
    assert!(p_yes > p_no, "water is wet: p(yes) {p_yes} <= p(no) {p_no}");
}

fn vocab_of(model: &GpuGemma4) -> usize {
    model.vocab()
}

/// Block generation at the authors' schedule: random canvas, denoise until
/// stable+confident or the cap, commit, decode. Prints the text and the
/// per-step trace; asserts the loop converged and produced words.
#[test]
fn diffusion_gemma_generates_a_block() {
    let Some((_exec, mut model, tok)) = load() else {
        return;
    };
    let cfg = model.diffusion_config().expect("diffusion file");
    let prompt_text = std::env::var("PADDOCK_DG_PROMPT")
        .unwrap_or_else(|_| "Write one sentence about the sea.".into());
    let prompt = chat_prompt(&tok, &prompt_text);
    let seed: u64 = env_or("PADDOCK_DG_SEED", 42);
    let cap: u32 = env_or("PADDOCK_DG_STEPS", cfg.max_steps);
    let fixed_temp: Option<f32> = std::env::var("PADDOCK_DG_TEMP")
        .ok()
        .and_then(|v| v.parse().ok());

    let base = prompt.len();
    model.reset();
    model.forward_prefill_stream(&prompt).expect("prefill");
    let w = model.canvas_len();
    let mut st = model.canvas_new(w).expect("canvas");
    let init = model.random_canvas(w, seed, 0);
    model.canvas_seed(&mut st, &init).expect("seed");

    let t0 = Instant::now();
    let mut converged = None;
    for step in 0..cap {
        let temp = fixed_temp.unwrap_or_else(|| cfg.temperature(step));
        let s = model
            .canvas_step(0, base, &mut st, temp, seed)
            .expect("step");
        eprintln!(
            "step {step:2} t {temp:.3}: accepted {:3} mean_entropy {:.4} stable {} converged {}",
            s.n_accepted, s.mean_entropy, s.stable, s.converged
        );
        assert!(
            s.mean_entropy.is_finite(),
            "entropy went non-finite at step {step}"
        );
        if s.converged {
            converged = Some(step + 1);
            break;
        }
    }
    let steps = converged.unwrap_or(cap);
    let block = st.last_argmax.clone();
    // the answer ends at the first EOS-class id (generation_config.json:
    // <eos> 1, <turn|> 106, 50); everything after it is the canvas' tail
    let end = block
        .iter()
        .position(|&t| matches!(t, 1 | 106 | 50))
        .unwrap_or(block.len());
    let text = tok.decode(&block[..end], true).expect("decode");
    eprintln!(
        "converged after {steps} of {cap} steps in {:.1}s ({:.2} s/step); {} tokens before EOS:\n{text}",
        t0.elapsed().as_secs_f32(),
        t0.elapsed().as_secs_f32() / steps as f32,
        end
    );
    model.canvas_commit_at(0, base, &block).expect("commit");
    assert!(converged.is_some(), "did not converge within {cap} steps");
    assert!(end > 0 && !text.trim().is_empty(), "an empty answer");
    assert!(
        text.split_whitespace().count() >= 3,
        "expected a sentence, got {text:?}"
    );
}
