use super::*;
use std::sync::atomic::AtomicBool;

fn fixture(dir: &Path, url: &str, data: &[u8]) -> Catalog {
    let reg = Registry::new(dir.into()).with_backend("metal");
    let mut model = reg
        .catalog
        .models
        .iter()
        .find(|m| {
            m.artifacts
                .iter()
                .any(|a| a.runtime.supports_backend("metal") && a.kind == ArtifactKind::Weights)
        })
        .unwrap()
        .clone();
    let mut weights = model
        .artifacts
        .iter()
        .find(|a| a.runtime.supports_backend("metal") && a.kind == ArtifactKind::Weights)
        .unwrap()
        .clone();
    model.id = "fixture".into();
    weights.id = "mlx".into();
    weights.files = vec![CatalogFile {
        dest: "fixture/model.safetensors".into(),
        url: url.into(),
        size: data.len() as u64,
        sha256: hex(&Sha256::digest(data)),
    }];
    weights.default = true;
    model.artifacts = vec![weights];
    Catalog {
        schema: 3,
        models: vec![model],
    }
}

async fn settled(reg: &Registry, id: &str) -> PullStatus {
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        loop {
            let status = reg.job(id).unwrap().status.lock().unwrap().clone();
            if !matches!(status, PullStatus::Running) {
                return status;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("bounded download completion")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paused_selection_survives_real_sqlite_reopen_and_completes() {
    let dir = tempfile::tempdir().unwrap();
    let data = vec![9u8; 20 * 1024 * 1024];
    let url = tests::spawn_origin(data.clone()).await;
    let catalog = fixture(dir.path(), &url, &data);
    let db = dir.path().join("product.db");
    let reg = Registry::from_catalog(catalog.clone(), dir.path().join("models"))
        .with_backend("metal")
        .with_store(Arc::new(crate::store::Store::open(&db).unwrap()))
        .unwrap();
    let id = reg.start_pull("fixture", Some(&["mlx".into()])).unwrap();
    assert!(reg.cancel_pull(&id));
    assert!(matches!(settled(&reg, &id).await, PullStatus::Cancelled));
    reg.pause_downloads().await;
    drop(reg);
    let reg = Registry::from_catalog(catalog, dir.path().join("models"))
        .with_backend("metal")
        .with_store(Arc::new(crate::store::Store::open(&db).unwrap()))
        .unwrap();
    assert!(matches!(
        *reg.job(&id).unwrap().status.lock().unwrap(),
        PullStatus::Cancelled
    ));
    let retry = reg.resume_pull(&id).unwrap();
    assert!(matches!(settled(&reg, &retry).await, PullStatus::Done));
    assert_eq!(
        std::fs::read(dir.path().join("models/fixture/model.safetensors")).unwrap(),
        data
    );
    assert!(reg.job(&retry).unwrap().follow.lock().unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlapping_downloads_and_catalog_drift_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let data = vec![3u8; 64 * 1024];
    let url = tests::spawn_origin(data.clone()).await;
    let catalog = fixture(dir.path(), &url, &data);
    let db = Arc::new(crate::store::Store::open(&dir.path().join("product.db")).unwrap());
    let reg = Registry::from_catalog(catalog.clone(), dir.path().join("models"))
        .with_backend("metal")
        .with_store(db.clone())
        .unwrap();
    let id = reg.start_pull("fixture", Some(&["mlx".into()])).unwrap();
    assert!(reg.start_pull("fixture", Some(&["mlx".into()])).is_err());
    reg.cancel_pull(&id);
    settled(&reg, &id).await;
    let mut changed = catalog;
    changed.models[0].artifacts[0].files[0].sha256 = "0".repeat(64);
    let other = Registry::from_catalog(changed, dir.path().join("models"))
        .with_backend("metal")
        .with_store(db)
        .unwrap();
    assert!(
        other
            .resume_pull(&id)
            .unwrap_err()
            .to_string()
            .contains("catalog changed")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completed_segments_resume_and_corrupt_bitmap_does_not_loop() {
    let dir = tempfile::tempdir().unwrap();
    let data = vec![7u8; 20 * 1024 * 1024];
    let url = tests::spawn_origin(data.clone()).await;
    let sha = hex(&Sha256::digest(&data));
    let dest = dir.path().join("model.safetensors");
    let part = part_path(&dest);
    std::fs::write(&part, &data).unwrap();
    bind_resume(&dest, &url, data.len() as u64, &sha).unwrap();
    std::fs::write(state_path(&dest), [1, 0]).unwrap();
    let counter = Arc::new(AtomicU64::new(0));
    download_file(
        &reqwest::Client::new(),
        &url,
        &dest,
        &sha,
        data.len() as u64,
        counter.clone(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(counter.load(Ordering::Relaxed), data.len() as u64);
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    // Retry a corrupt all-done partial file. Failure invalidates the bitmap;
    // a second attempt refetches and publishes, rather than repeating the hash error.
    let bad = dir.path().join("bad.safetensors");
    std::fs::write(part_path(&bad), vec![0; data.len()]).unwrap();
    bind_resume(&bad, &url, data.len() as u64, &sha).unwrap();
    std::fs::write(state_path(&bad), [1, 1]).unwrap();
    assert!(matches!(
        download_file(
            &reqwest::Client::new(),
            &url,
            &bad,
            &sha,
            data.len() as u64,
            Arc::new(AtomicU64::new(0)),
            None
        )
        .await,
        Err(DlError::Checksum { .. })
    ));
    assert!(!bad.exists() && !state_path(&bad).exists());
    download_file(
        &reqwest::Client::new(),
        &url,
        &bad,
        &sha,
        data.len() as u64,
        Arc::new(AtomicU64::new(0)),
        None,
    )
    .await
    .unwrap();
    assert_eq!(std::fs::read(bad).unwrap(), data);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_headers_cancel_promptly_without_publishing() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/never", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let dir = tempfile::tempdir().unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        trigger.store(true, Ordering::Relaxed);
    });
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        download_file(
            &reqwest::Client::new(),
            &url,
            &dir.path().join("stall.gguf"),
            &"0".repeat(64),
            100,
            Arc::new(AtomicU64::new(0)),
            Some(cancel),
        ),
    )
    .await
    .unwrap();
    assert!(matches!(result, Err(DlError::Cancelled)));
    server.abort();
}

#[test]
fn sidecars_do_not_alias_json_and_model_names() {
    assert_ne!(
        part_path(Path::new("tokenizer.json")),
        part_path(Path::new("tokenizer.model"))
    );
}

#[tokio::test]
async fn failed_intent_save_never_starts_network_work() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = fixture(dir.path(), "http://127.0.0.1:1/never", &[1u8]);
    let db = dir.path().join("product.db");
    let store = Arc::new(crate::store::Store::open(&db).unwrap());
    let reg = Registry::from_catalog(catalog, dir.path().join("models"))
        .with_backend("metal")
        .with_store(store)
        .unwrap();
    // Only this disposable test database is altered to inject a write failure.
    rusqlite::Connection::open(db)
        .unwrap()
        .execute("DROP TABLE model_downloads", [])
        .unwrap();
    assert!(reg.start_pull("fixture", Some(&["mlx".into()])).is_err());
    assert!(reg.jobs().is_empty());
    assert!(!dir.path().join("models/fixture").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malformed_content_range_is_rejected_before_body_allocation() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/file", listener.local_addr().unwrap());
    let app = axum::Router::new().route(
        "/file",
        axum::routing::get(|| async {
            (
                axum::http::StatusCode::PARTIAL_CONTENT,
                [("content-range", "bytes 4-7/8")],
                vec![1u8; 4],
            )
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let result = fetch_range(
        &reqwest::Client::new(),
        &url,
        0,
        3,
        8,
        &Arc::new(AtomicU64::new(0)),
    )
    .await;
    assert!(matches!(result, Err(DlError::Http(message)) if message.contains("Content-Range")));
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disk_guard_refuses_before_creating_a_transfer() {
    let dir = tempfile::tempdir().unwrap();
    let mut catalog = fixture(dir.path(), "http://127.0.0.1:1/never", &[1u8]);
    catalog.models[0].artifacts[0].files[0].size = disk_free(dir.path()).unwrap() + 1;
    let reg = Registry::from_catalog(catalog, dir.path().into()).with_backend("metal");
    assert!(matches!(
        reg.start_pull("fixture", Some(&["mlx".into()])),
        Err(DlError::Disk { .. })
    ));
    assert!(reg.jobs().is_empty());
    assert!(!dir.path().join("fixture").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_crashed_running_record_recovers_paused_without_starting_network_work() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = fixture(dir.path(), "http://127.0.0.1:1/never", &[1u8]);
    let store = Arc::new(crate::store::Store::open(&dir.path().join("product.db")).unwrap());
    let reg = Registry::from_catalog(catalog.clone(), dir.path().join("models"))
        .with_backend("metal")
        .with_store(store.clone())
        .unwrap();
    let id = reg.start_pull("fixture", Some(&["mlx".into()])).unwrap();
    reg.cancel_pull(&id);
    settled(&reg, &id).await;
    let mut interrupted = reg.job(&id).unwrap().record();
    interrupted["status"] = serde_json::json!({"state":"running"});
    store.save_download(&interrupted).unwrap();
    drop(reg);
    let reopened = Registry::from_catalog(catalog, dir.path().join("models"))
        .with_backend("metal")
        .with_store(store)
        .unwrap();
    assert!(matches!(
        *reopened.job(&id).unwrap().status.lock().unwrap(),
        PullStatus::Cancelled
    ));
    assert!(!dir.path().join("models/fixture/model.safetensors").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn same_size_corruption_is_not_treated_as_installed_success() {
    let dir = tempfile::tempdir().unwrap();
    let data = vec![2u8; 1024];
    let catalog = fixture(dir.path(), "http://127.0.0.1:1/never", &data);
    std::fs::create_dir_all(dir.path().join("fixture")).unwrap();
    std::fs::write(
        dir.path().join("fixture/model.safetensors"),
        vec![0u8; 1024],
    )
    .unwrap();
    let reg = Registry::from_catalog(catalog, dir.path().into()).with_backend("metal");
    let id = reg.start_pull("fixture", Some(&["mlx".into()])).unwrap();
    assert!(matches!(settled(&reg, &id).await, PullStatus::Error { .. }));
    assert_eq!(
        std::fs::read(dir.path().join("fixture/model.safetensors")).unwrap(),
        vec![0; 1024]
    );
}
