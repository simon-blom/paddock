use super::*;

#[test]
#[ignore = "requires elected Qwen3 Q8 GGUF in PADDOCK_QWEN3_ENCODER_TEST_MODEL"]
fn ragged_enqueue_ahead_and_cancellation() {
    let path = std::env::var("PADDOCK_QWEN3_ENCODER_TEST_MODEL").expect("test model");
    let mut m = Qwen3Encoder::load(std::path::Path::new(&path), 2048, Some(16 << 30)).unwrap();
    let seqs = vec![
        vec![151643],
        vec![9707, 11, 1879, 0, 151643],
        vec![100; 33],
        vec![101; 129],
    ];
    let mut singles = Vec::new();
    for s in &seqs {
        let p = m.embed_submit(std::slice::from_ref(s), 0).unwrap();
        singles.push(m.embed_collect(&p).unwrap().remove(0));
    }
    let p = m.embed_submit(&seqs, 0).unwrap();
    let next = m.embed_submit(&[vec![200; 65], vec![500; 2]], 0).unwrap();
    let batch = m.embed_collect(&p).unwrap();
    for (a, b) in singles.iter().zip(&batch) {
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("ragged singleton/batch max {max}");
        assert!(max < 0.002, "{max}");
        let norm = b.iter().map(|x| x * x).sum::<f32>();
        assert!((norm - 1.).abs() < 1e-5);
    }
    drop(next); // cancellation fences this completion without poisoning later jobs
    let again = m.embed_submit(&seqs, 0).unwrap();
    assert_eq!(batch, m.embed_collect(&again).unwrap());
    assert!(m.embed_submit(&[vec![]], 0).is_err());
    assert!(m.embed_submit(&[vec![0; 2049]], 0).is_err());
    assert!(m.embed_submit(&[vec![m.vocab as u32]], 0).is_err());
    assert!(m.rerank_submit(&seqs, 1, 1, 0).is_err());
    let p = m.rerank_submit(&seqs, 9693, 2152, 0).unwrap();
    let scores = m.rerank_collect(&p, 9693, 2152).unwrap();
    for (seq, score) in seqs.iter().zip(scores) {
        let p = m
            .rerank_submit(std::slice::from_ref(seq), 9693, 2152, 0)
            .unwrap();
        assert!((score - m.rerank_collect(&p, 9693, 2152).unwrap()[0]).abs() < 0.005);
    }
}
