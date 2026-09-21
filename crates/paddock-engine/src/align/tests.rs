use super::*;
struct Fake;
impl AlignBackend for Fake {
    fn limits(&self) -> AlignLimits {
        AlignLimits {
            batch: 4,
            rows: 32,
            frames: 400,
        }
    }
    fn validate(&self, r: &AlignReq) -> Result<(), String> {
        if r.ids.is_empty() {
            Err("empty".into())
        } else {
            Ok(())
        }
    }
    fn run_batch(
        &mut self,
        rs: &[&AlignReq],
        _: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<Result<Vec<u32>, String>>, String> {
        Ok(rs
            .iter()
            .map(|r| Ok(vec![r.ids[0]; r.ts_rows.len()]))
            .collect())
    }
}
fn req(id: u32) -> AlignReq {
    AlignReq {
        ids: vec![id],
        mel: MelFeatures {
            data: vec![],
            n_frames: 100,
            n_samples: 16000,
            global_max: 0.,
        },
        splice_at: 0,
        n_audio: 1,
        ts_rows: vec![0, 0],
    }
}
#[tokio::test]
async fn alignment_seam_respects_bounds_and_returns_own_rows() {
    let a = Aligner::spawn(|| Ok(Fake)).unwrap();
    let permits = (0..8).map(|_| a.reserve().unwrap()).collect::<Vec<_>>();
    assert!(a.reserve().is_err());
    drop(permits);
    let (x, y, z, w) = tokio::join!(
        a.align(req(1)),
        a.align(req(2)),
        a.align(req(3)),
        a.align(req(4))
    );
    assert_eq!(x.unwrap(), [1, 1]);
    assert_eq!(y.unwrap(), [2, 2]);
    assert_eq!(z.unwrap(), [3, 3]);
    assert_eq!(w.unwrap(), [4, 4]);
    let mut r = req(5);
    r.ids.clear();
    assert!(a.align(r).await.is_err());
    let mut r = req(5);
    r.mel.n_frames = 401;
    assert!(a.align(r).await.is_err());
    assert_eq!(a.align(req(6)).await.unwrap(), [6, 6]);
}

#[tokio::test]
async fn alignment_coalescing_skips_canceled_and_invalid_without_shifting_results() {
    struct Gate {
        first: bool,
        entered: Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
        batches: Sender<Vec<u32>>,
    }
    impl AlignBackend for Gate {
        fn limits(&self) -> AlignLimits {
            Fake.limits()
        }
        fn validate(&self, r: &AlignReq) -> Result<(), String> {
            Fake.validate(r)
        }
        fn run_batch(
            &mut self,
            rs: &[&AlignReq],
            _: &dyn Fn(usize) -> bool,
        ) -> Result<Vec<Result<Vec<u32>, String>>, String> {
            if self.first {
                self.first = false;
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
            self.batches
                .send(rs.iter().map(|r| r.ids[0]).collect())
                .unwrap();
            Fake.run_batch(rs, &|_| false)
        }
    }
    let (entered_tx, entered_rx) = channel();
    let (release_tx, release_rx) = channel();
    let (batch_tx, batch_rx) = channel();
    let a = Aligner::spawn(move || {
        Ok(Gate {
            first: true,
            entered: entered_tx,
            release: release_rx,
            batches: batch_tx,
        })
    })
    .unwrap();
    let enqueue = |r| {
        let (reply, rx) = oneshot::channel();
        a.tx.send(Job {
            req: r,
            reply,
            _permit: a.reserve().unwrap(),
        })
        .unwrap();
        rx
    };
    let first = enqueue(req(0));
    entered_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let mut bad = req(9);
    bad.ids.clear();
    let invalid = enqueue(bad);
    let canceled = enqueue(req(99));
    drop(canceled);
    let live = (1..=4).map(|i| enqueue(req(i))).collect::<Vec<_>>();
    release_tx.send(()).unwrap();
    assert_eq!(first.await.unwrap().unwrap(), [0, 0]);
    assert!(invalid.await.unwrap().is_err());
    for (i, r) in live.into_iter().enumerate() {
        assert_eq!(r.await.unwrap().unwrap(), [(i + 1) as u32; 2]);
    }
    assert_eq!(batch_rx.recv().unwrap(), [0]);
    assert_eq!(batch_rx.recv().unwrap(), [1, 2, 3, 4]);
    assert!(batch_rx.try_recv().is_err());
}
