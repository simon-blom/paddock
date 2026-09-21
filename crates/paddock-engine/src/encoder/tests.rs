//! Scheduler-only transport fixtures. No model inference or CPU oracle.
use super::*;
use std::sync::Mutex;

struct Transport {
    submitted: Arc<Mutex<Vec<usize>>>,
}
impl EncoderBackend for Transport {
    type Pending = Vec<Vec<u32>>;
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(7)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(11)
    }
    fn coalesce_row_budget(&self) -> usize {
        4
    }
    fn lanes(&mut self) -> usize {
        1
    }
    fn pool_ready(&self, _: &Self::Pending) -> bool {
        true
    }
    fn embed_submit(&mut self, s: &[Vec<u32>], _: usize) -> Result<Self::Pending, String> {
        let rows = s.iter().map(Vec::len).sum();
        assert!(rows <= 4, "scheduler overfilled backend row budget");
        self.submitted.lock().unwrap().push(rows);
        Ok(s.to_vec())
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> Result<Vec<Vec<f32>>, String> {
        Ok(p.iter().map(|s| vec![s[0] as f32]).collect())
    }
    fn rerank_submit(
        &mut self,
        s: &[Vec<u32>],
        _: u32,
        _: u32,
        l: usize,
    ) -> Result<Self::Pending, String> {
        self.embed_submit(s, l)
    }
    fn rerank_collect(&mut self, p: &Self::Pending, _: u32, _: u32) -> Result<Vec<f32>, String> {
        Ok(p.iter().map(|s| s[0] as f32).collect())
    }
}

#[tokio::test]
async fn bounded_merges_preserve_replies_and_skip_cancelled_jobs() {
    let submitted = Arc::new(Mutex::new(Vec::new()));
    let copy = submitted.clone();
    let metrics = Arc::new(EngineMetrics::default());
    let encoder = Encoder::spawn(
        move || Ok(Transport { submitted: copy }),
        Some(metrics.clone()),
    )
    .unwrap();
    assert!(!encoder.block_scale_calibration());
    let (tx, rx) = oneshot::channel();
    drop(rx);
    encoder
        .tx
        .send(EncodeJob::Embed {
            seqs: vec![vec![999; 99]],
            reply: tx,
        })
        .unwrap();
    let mut receivers = Vec::new();
    for i in 0..12 {
        let (reply, rx) = oneshot::channel();
        encoder
            .tx
            .send(EncodeJob::Embed {
                seqs: vec![vec![i; 3]],
                reply,
            })
            .unwrap();
        receivers.push(rx);
    }
    for (i, rx) in receivers.into_iter().enumerate() {
        assert_eq!(rx.await.unwrap().unwrap(), vec![vec![i as f32]]);
    }
    assert_eq!(
        encoder.rerank(vec![vec![20; 3]], 1, 2).await.unwrap(),
        vec![20.]
    );
    assert_eq!(metrics.weights_mem_bytes.load(Relaxed), 7);
    assert!(
        !encoder
            .apply_profile("cuda-only".into(), None)
            .await
            .unwrap()
    );
    assert!(encoder.calibrate(vec![], 0, vec![]).await.is_err());
    assert!(submitted.lock().unwrap().iter().all(|&n| n <= 4));
}
