use super::*;
use axum::{
    Router,
    body::Body,
    http::Request,
    routing::{get, post},
};
use tower::ServiceExt;

#[tokio::test]
async fn unlock_is_an_explicit_revision_checked_job_without_key_material() {
    let state = Arc::new(AppState::for_tests());
    state.db.prepare_cloud_credentials().unwrap();
    let sessions = Sessions::default();
    let row = state.db.create_cloud_endpoint(&json!({"name":"Keyless fixture", "kind":"openai-compat", "baseUrl":"https://example.invalid/v1"})).unwrap();
    let endpoint = row["id"].as_str().unwrap();
    for (revision, status) in [(1, "failed"), (0, "unlocked")] {
        let start = sessions
            .execute(
                state.clone(),
                Command::Unlock {
                    id: endpoint.into(),
                    revision,
                },
            )
            .unwrap();
        let result = terminal(&sessions, &state, start["job"]["id"].as_str().unwrap()).await;
        assert_eq!(result["job"]["status"], status);
        assert_eq!(result["job"]["endpoint"], Value::Null);
        assert_eq!(result["job"]["models"], json!([]));
    }
    for command in [
        json!({"kind":"unlock","id":endpoint}),
        json!({"kind":"unlock","id":endpoint,"revision":0,"apiKey":"not-accepted"}),
    ] {
        assert!(serde_json::from_value::<Command>(command).is_err());
    }
}

async fn terminal(sessions: &Sessions, state: &Arc<AppState>, id: &str) -> Value {
    for _ in 0..200 {
        let value = sessions
            .execute(state.clone(), Command::Poll { id: id.into() })
            .unwrap();
        if !matches!(value["job"]["status"].as_str(), Some("checking" | "saving")) {
            return value;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("Synthetic connection operation did not settle")
}
async fn provider() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new()
        .route("/v1/models", get(|| async { axum::Json(json!({"data":[{"id":"fixture-model"}]})) }))
        .route("/v1/chat/completions", post(|headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| async move {
            assert!(headers.get("authorization").is_none());
            assert_eq!(body["model"], "fixture-model");
            let text = concat!(
                "data: {\"id\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Fixture reply\"}}]}\n\n",
                "data: {\"id\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n");
            ([("content-type", "text/event-stream")], text)
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (base, task)
}

#[tokio::test]
async fn native_check_save_dedup_and_shared_inference_use_one_connection() {
    let state = Arc::new(AppState::for_tests());
    let sessions = Sessions::default();
    let (base, server) = provider().await;
    let draft = serde_json::from_value(json!({"name":"Fixture", "kind":"openai-compat", "baseUrl":base, "allowUnauthenticated":true, "apiKey":""})).unwrap();
    let start = sessions
        .execute(state.clone(), Command::Check { draft })
        .unwrap();
    let id = start["job"]["id"].as_str().unwrap();
    assert!(state.db.list_cloud_endpoints().unwrap().is_empty());
    let checked = terminal(&sessions, &state, id).await;
    assert_eq!(checked["job"]["status"], "checked");
    let models: Vec<Pick> = serde_json::from_value(json!([{"id":"fixture-model"}])).unwrap();
    sessions
        .execute(
            state.clone(),
            Command::Save {
                id: id.into(),
                models,
            },
        )
        .unwrap();
    let saved = terminal(&sessions, &state, id).await;
    assert_eq!(saved["job"]["status"], "saved");
    let endpoint = saved["job"]["endpoint"]["id"].as_str().unwrap();
    sessions
        .execute(
            state.clone(),
            Command::Save {
                id: id.into(),
                models: vec![],
            },
        )
        .unwrap();
    let rows = state.db.list_cloud_endpoints().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0]["models"].as_array().unwrap().len(),
        1,
        "receipt replay must not replace picks"
    );
    let response = paddock_manager::routes::router(state.clone()).oneshot(Request::builder()
        .method("POST").uri(format!("/api/cloud/{endpoint}/v1/responses"))
        .header("content-type", "application/json")
        .body(Body::from(json!({"model":format!("cloud:{endpoint}:fixture-model"),"input":"Synthetic question", "stream":true,"max_output_tokens":16}).to_string())).unwrap()).await.unwrap();
    assert!(response.status().is_success());
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("Fixture reply"), "{text}");
    assert!(text.contains("response.completed"), "{text}");
    server.abort();
}

#[tokio::test]
async fn checked_draft_cancellation_prevents_late_save_and_stale_edits_fail() {
    let state = Arc::new(AppState::for_tests());
    let sessions = Sessions::default();
    let (base, server) = provider().await;
    let draft = serde_json::from_value(json!({"name":"Fixture", "kind":"openai-compat", "baseUrl":base, "allowUnauthenticated":true})).unwrap();
    let started = sessions
        .execute(state.clone(), Command::Check { draft })
        .unwrap();
    let id = started["job"]["id"].as_str().unwrap();
    assert_eq!(
        terminal(&sessions, &state, id).await["job"]["status"],
        "checked"
    );
    sessions
        .execute(state.clone(), Command::Cancel { id: id.into() })
        .unwrap();
    assert!(
        sessions
            .execute(
                state.clone(),
                Command::Save {
                    id: id.into(),
                    models: vec![]
                }
            )
            .is_err()
    );
    assert!(state.db.list_cloud_endpoints().unwrap().is_empty());
    assert!(
        serde_json::from_value::<Command>(json!({"kind":"list", "url":"https://other.invalid"}))
            .is_err()
    );
    server.abort();
}
