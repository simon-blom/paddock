use super::*;
use axum::{
    body::{Body, Bytes},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use std::sync::atomic::{AtomicBool, AtomicUsize};

/// How long a step that only has to terminate is given. These are hang
/// watchdogs, not latency budgets: the suite runs a few hundred tests at once,
/// often with its temp directory on a slow disk, and a one or two second bound
/// written on an idle machine then fails for reasons that have nothing to do
/// with the code under test. A real hang still fails, just later.
const WATCHDOG: Duration = Duration::from_secs(30);

async fn origin(app: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/file", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, task)
}

#[test]
fn retry_policy_is_bounded_and_does_not_retry_contract_errors() {
    assert!(retry_delay(&DlError::Transport("reset".into()), 0, 0).is_some());
    assert_eq!(
        retry_delay(
            &DlError::RetryableStatus {
                status: 429,
                retry_after: Some(3)
            },
            0,
            0
        ),
        Some(Duration::from_secs(3))
    );
    assert!(
        retry_delay(
            &DlError::RetryableStatus {
                status: 429,
                retry_after: Some(300)
            },
            0,
            0
        )
        .is_none()
    );
    for error in [
        DlError::Http("Invalid Content-Range".into()),
        DlError::Status(401),
        DlError::NotFound {
            url: "missing".into(),
        },
        DlError::Size {
            expected: 1,
            got: 2,
        },
        DlError::Cancelled,
    ] {
        assert!(retry_delay(&error, 0, 0).is_none());
    }
}

#[tokio::test]
async fn retry_keeps_worker_alive_and_publishes_verified_bytes() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let count = attempts.clone();
    let app = axum::Router::new().route(
        "/file",
        axum::routing::get(move |headers: HeaderMap| {
            let count = count.clone();
            async move {
                let probe = headers.get("range").unwrap() == "bytes=0-0";
                if !probe && count.fetch_add(1, Ordering::Relaxed) < 2 {
                    return (StatusCode::SERVICE_UNAVAILABLE, "transient").into_response();
                }
                (
                    StatusCode::PARTIAL_CONTENT,
                    [(
                        "content-range",
                        if probe {
                            "bytes 0-0/4096"
                        } else {
                            "bytes 0-4095/4096"
                        },
                    )],
                    vec![7u8; if probe { 1 } else { 4096 }],
                )
                    .into_response()
            }
        }),
    );
    let (url, server) = origin(app).await;
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.gguf");
    let data = vec![7u8; 4096];
    let total = Arc::new(AtomicU64::new(0));
    download_file(
        &reqwest::Client::new(),
        &url,
        &dest,
        &hex(&Sha256::digest(&data)),
        4096,
        total.clone(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(attempts.load(Ordering::Relaxed), 3);
    assert_eq!(total.load(Ordering::Relaxed), 4096);
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    server.abort();
}

#[tokio::test]
async fn permanent_transient_failure_stops_after_four_attempts() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let count = attempts.clone();
    let app = axum::Router::new().route(
        "/file",
        axum::routing::get(move || {
            count.fetch_add(1, Ordering::Relaxed);
            async { StatusCode::SERVICE_UNAVAILABLE }
        }),
    );
    let (url, server) = origin(app).await;
    let total = Arc::new(AtomicU64::new(100));
    let error = fetch_range(&reqwest::Client::new(), &url, 0, 4095, 4096, &total)
        .await
        .err()
        .unwrap();
    assert!(matches!(
        error,
        DlError::RetryableStatus { status: 503, .. }
    ));
    assert_eq!(attempts.load(Ordering::Relaxed), ATTEMPTS);
    assert_eq!(total.load(Ordering::Relaxed), 100);
    server.abort();
}

#[tokio::test]
async fn interrupted_body_retries_without_double_counting() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let count = attempts.clone();
    let app = axum::Router::new().route(
        "/file",
        axum::routing::get(move || {
            let fail = count.fetch_add(1, Ordering::Relaxed) == 0;
            async move {
                let (tx, rx) = tokio::sync::mpsc::channel(2);
                tokio::spawn(async move {
                    let _ = tx
                        .send(Ok::<_, std::io::Error>(Bytes::from(vec![7u8; 2048])))
                        .await;
                    tokio::time::sleep(Duration::from_millis(40)).await;
                    let tail = if fail {
                        Err(std::io::Error::from(std::io::ErrorKind::ConnectionReset))
                    } else {
                        Ok(Bytes::from(vec![7u8; 2048]))
                    };
                    let _ = tx.send(tail).await;
                });
                let stream = futures::stream::unfold(rx, |mut rx| async move {
                    rx.recv().await.map(|item| (item, rx))
                });
                (
                    StatusCode::PARTIAL_CONTENT,
                    [("content-range", "bytes 0-4095/4096")],
                    Body::from_stream(stream),
                )
            }
        }),
    );
    let (url, server) = origin(app).await;
    let total = Arc::new(AtomicU64::new(100));
    let (bytes, progress) = fetch_range(&reqwest::Client::new(), &url, 0, 4095, 4096, &total)
        .await
        .unwrap();
    assert_eq!(bytes, vec![7u8; 4096]);
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
    assert_eq!(total.load(Ordering::Relaxed), 4196);
    progress.commit();
    assert_eq!(total.load(Ordering::Relaxed), 4196);
    server.abort();
}

#[tokio::test]
async fn partial_segment_reports_live_progress_and_cancel_rolls_it_back() {
    let app = axum::Router::new().route(
        "/file",
        axum::routing::get(|| async {
            let stream = futures::stream::once(async {
                Ok::<_, std::io::Error>(Bytes::from(vec![7u8; 2048]))
            })
            .chain(futures::stream::pending());
            (
                StatusCode::PARTIAL_CONTENT,
                [("content-range", "bytes 0-4095/4096")],
                Body::from_stream(stream),
            )
        }),
    );
    let (url, server) = origin(app).await;
    let total = Arc::new(AtomicU64::new(100));
    let progress = total.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    let worker = tokio::spawn(async move {
        cancellable(
            Some(&stop),
            fetch_range(&reqwest::Client::new(), &url, 0, 4095, 4096, &progress),
        )
        .await
    });
    tokio::time::timeout(WATCHDOG, async {
        while total.load(Ordering::Relaxed) == 100 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        total.load(Ordering::Relaxed),
        2148,
        "progress arrives before the range finishes"
    );
    cancel.store(true, Ordering::Relaxed);
    let result = tokio::time::timeout(WATCHDOG, worker)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, Err(DlError::Cancelled)));
    assert_eq!(
        total.load(Ordering::Relaxed),
        100,
        "incomplete bytes are never counted as resumable"
    );
    server.abort();
}

#[tokio::test]
async fn terminal_failure_stops_silent_peers_and_leaves_no_detached_writer() {
    let app = axum::Router::new().route(
        "/file",
        axum::routing::get(|headers: HeaderMap| async move {
            if headers.get("range").unwrap() == "bytes=0-16777215" {
                return StatusCode::NOT_FOUND.into_response();
            }
            let stream = futures::stream::pending::<Result<Bytes, std::io::Error>>();
            (
                StatusCode::PARTIAL_CONTENT,
                [("content-range", "bytes 16777216-33554431/33554432")],
                Body::from_stream(stream),
            )
                .into_response()
        }),
    );
    let (url, server) = origin(app).await;
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.gguf");
    let total = Arc::new(AtomicU64::new(0));
    let result = tokio::time::timeout(
        WATCHDOG,
        download_ranged(
            &reqwest::Client::new(),
            &url,
            &part_path(&dest),
            &dest,
            SEGMENT * 2,
            &total,
            None,
        ),
    )
    .await
    .unwrap();
    assert!(matches!(result, Err(DlError::NotFound { .. })));
    assert_eq!(total.load(Ordering::Relaxed), 0);
    assert!(!dest.exists());
    assert!(load_state(&state_path(&dest), 2).iter().all(|done| !done));
    server.abort();
}

#[tokio::test]
async fn silent_body_has_a_bounded_idle_timeout() {
    let app = axum::Router::new().route(
        "/file",
        axum::routing::get(|| async {
            let stream =
                futures::stream::once(async { Ok::<_, std::io::Error>(Bytes::from_static(b"x")) })
                    .chain(futures::stream::pending());
            (
                StatusCode::PARTIAL_CONTENT,
                [("content-range", "bytes 0-1/2")],
                Body::from_stream(stream),
            )
        }),
    );
    let (url, server) = origin(app).await;
    let total = Arc::new(AtomicU64::new(0));
    let mut progress = RangeProgress::new(&total);
    let result = tokio::time::timeout(
        IDLE_TIMEOUT + WATCHDOG,
        fetch_once(&reqwest::Client::new(), &url, 0, 1, 2, &mut progress),
    )
    .await
    .unwrap();
    assert!(
        matches!(result, Err(DlError::Transport(message)) if message.contains("No download data"))
    );
    drop(progress);
    assert_eq!(total.load(Ordering::Relaxed), 0);
    server.abort();
}

/// Explicit network diagnostic: bounded public R2 ranges, held only in memory.
#[tokio::test]
#[ignore = "downloads 128 MiB of public R2 ranges; run explicitly to measure transport"]
async fn r2_range_transport_probe() {
    let url = "https://models.truespar.io/models/kb-whisper-large/kb-whisper-large-f16.gguf";
    for adaptive in [false, true] {
        let client = reqwest::Client::builder()
            .http2_adaptive_window(adaptive)
            .build()
            .unwrap();
        let start = std::time::Instant::now();
        let total = Arc::new(AtomicU64::new(0));
        let results = futures::future::join_all((0..4).map(|i| {
            fetch_range(
                &client,
                url,
                i * SEGMENT,
                (i + 1) * SEGMENT - 1,
                3223785280,
                &total,
            )
        }))
        .await;
        let mut count = 0;
        for result in results {
            let (bytes, progress) = result.unwrap();
            count += bytes.len();
            progress.commit();
        }
        eprintln!(
            "adaptive={adaptive} bytes={count} seconds={:.3} MB/s={:.2}",
            start.elapsed().as_secs_f64(),
            count as f64 / start.elapsed().as_secs_f64() / 1e6
        );
    }
}
