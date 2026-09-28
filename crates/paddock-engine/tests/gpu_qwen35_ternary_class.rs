//! The ternary class at model level (Bonsai 2 27B, PrismML PTQ1_0 behind the
//! prism Hadamard rotation): a decode row scores the SAME logits, bit for
//! bit, whether its slot decodes alone or shares the tick with other slots.
//!
//! The kernel gate (`gpu_ternary::ternary_nb_lane_bitmatches_the_single_walk`)
//! holds the NB-row lane to the batch-1 walk plane by plane; this one holds
//! the whole decode step - norms, rotations, attention, the DeltaNet
//! recurrence and every projection - across tick widths. It is what makes a
//! request's output independent of how many other agents are decoding, and
//! what a speculative verify round (a wider tick of one sequence) relies on.
//!
//! Heavy GPU test (a 27B model): --test-threads=1.

mod common;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

const BONSAI_PTQ1: &[&str] = &["Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"];

fn argmax(row: &[f32]) -> u32 {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

#[test]
fn decode_rows_score_the_same_alone_or_sharing_a_tick() {
    let Some(path) = common::model("BONSAI_GGUF", BONSAI_PTQ1) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuQwen35::load(exec.clone(), &map, 4096).expect("load Bonsai 27B");
    let seated = m.enable_batch(64).expect("enable_batch");
    assert!(seated >= 64, "only {seated} slots seated");
    let vocab = m.vocab;

    // 64 different prompts (token ids of the Qwen vocabulary, any valid
    // ones will do: the gate compares the engine with itself)
    let prompts: Vec<Vec<u32>> = (0..64u32)
        .map(|s| {
            (0..12u32 + 3 * s)
                .map(|i| 1000 + ((i * 7919 + s * 104_729) % 50_000))
                .collect()
        })
        .collect();
    let steps = 6usize;

    // slot 0 decodes `steps` ticks at `width`, fed `feed` (or its own
    // greedy picks), the other slots their own picks; slot 0's logits per
    // tick come back as bits, with the tokens it was fed
    let mut run = |width: usize, feed: Option<&[u32]>| -> (Vec<Vec<u32>>, Vec<u32>) {
        let mut last = vec![0u32; width];
        for (s, l) in last.iter_mut().enumerate() {
            let lg = m
                .forward_prefill_slot(s, &prompts[s])
                .expect("prefill slot");
            *l = argmax(&lg);
        }
        let mut pos: Vec<u32> = (0..width).map(|s| prompts[s].len() as u32).collect();
        let mut bits = Vec::with_capacity(steps);
        let mut fed = Vec::with_capacity(steps);
        for t in 0..steps {
            if let Some(f) = feed {
                last[0] = f[t];
            }
            fed.push(last[0]);
            let lg = m.forward_batch(&last, &pos).expect("decode tick");
            bits.push(
                lg[..vocab]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<u32>>(),
            );
            for s in 0..width {
                last[s] = argmax(&lg[s * vocab..(s + 1) * vocab]);
                pos[s] += 1;
            }
        }
        (bits, fed)
    };

    // TERN_CLASS_WIDTHS="3 5 6" narrows a failure to its width (a dev aid)
    let widths: Vec<usize> = std::env::var("TERN_CLASS_WIDTHS")
        .ok()
        .map(|v| {
            v.split_whitespace()
                .filter_map(|w| w.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![2, 3, 8, 16, 25, 32, 63]);
    let (alone, fed) = run(1, None);
    for width in widths {
        let (shared, _) = run(width, Some(&fed));
        for (t, (a, s)) in alone.iter().zip(&shared).enumerate() {
            let diff = a.iter().zip(s).filter(|(x, y)| x != y).count();
            assert_eq!(
                diff, 0,
                "tick {t}: slot 0 at width {width} differs from slot 0 alone in {diff} of {vocab} logits"
            );
        }
        eprintln!("width {width}: slot 0's {steps} ticks bit-identical to decoding alone");
    }
}
