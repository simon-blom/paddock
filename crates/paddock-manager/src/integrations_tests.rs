use super::*;

fn draft(url: &str) -> Draft {
    serde_json::from_value(json!({"label":"fixture","url":url,"headers":{}})).unwrap()
}

fn isolated_state() -> (tempfile::TempDir, Arc<AppState>) {
    let dir = tempfile::tempdir().unwrap();
    let mut state = AppState::for_tests();
    state.supervisor = Arc::new(crate::supervisor::Supervisor::new(
        crate::supervisor::SpawnDefaults {
            runner_bin: None,
            device: "metal".into(),
            kernel_pack: None,
            models_dirs: vec![],
            logs_dir: dir.path().join("logs"),
            runners_dir: dir.path().join("runners"),
            work_dir: dir.path().into(),
            base_port: 41540,
            health_timeout: Duration::from_millis(100),
        },
        state.registry.clone(),
        None,
        None,
    ));
    (dir, Arc::new(state))
}

#[test]
fn urls_and_headers_reject_credential_escape_routes() {
    for url in [
        "file:///tmp/a",
        "http://example.com/mcp",
        "https://u:secret@example.com/mcp",
        "https://example.com/mcp?key=secret",
        "https://example.com/mcp#fragment",
    ] {
        assert!(safe_url(url).is_err());
    }
    assert!(safe_url("http://[::1]:9999/mcp").is_ok());
    let state = Arc::new(AppState::for_tests());
    for name in ["Host", "Cookie", "Content-Length", "Mcp-Session-Id"] {
        let mut d = draft("https://example.invalid/mcp");
        d.headers = Some(BTreeMap::from([(name.into(), "fixture".into())]));
        assert!(prepare(&state, d).is_err());
    }
}

#[test]
fn native_keys_are_metadata_only_and_stale_edits_fail() {
    let state = Arc::new(AppState::for_tests());
    let mut d = draft("https://example.invalid/mcp");
    d.headers = Some(BTreeMap::from([(
        "Authorization".into(),
        "Bearer synthetic-secret".into(),
    )]));
    let doc = prepare(&state, d.clone()).unwrap();
    let id = state.db.save_native_connector(&doc).unwrap();
    let rows = state.db.native_connectors().unwrap();
    assert!(!rows[0].to_string().contains("synthetic-secret"));
    assert_eq!(rows[0]["hasHeaders"], true);
    assert_eq!(rows[0]["keychain"], true);
    assert_eq!(
        state.db.get_connector(&id).unwrap().unwrap()["headers"]["Authorization"],
        "Bearer synthetic-secret"
    );
    d.id = Some(id.clone());
    d.revision = Some(1);
    d.headers = None;
    let stale = prepare(&state, d.clone()).unwrap();
    state.db.set_connector_scope(&id, false, &[]).unwrap();
    assert!(state.db.save_native_connector(&stale).is_err());
    d.revision = Some(2);
    d.url = "https://different.invalid/mcp".into();
    assert!(prepare(&state, d).is_err());
    assert_eq!(
        state.db.get_connector(&id).unwrap().unwrap()["headers"]["Authorization"],
        "Bearer synthetic-secret"
    );
}

#[tokio::test]
async fn search_save_preserves_other_settings_and_rejects_stale_revision() {
    let (_dir, state) = isolated_state();
    let path = state.supervisor.server_config_path(13491);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let raw = "# user comment\nmodel = 'fixture.gguf'\nmax_ctx = 4096\napi_key = 'runner-secret'\n";
    std::fs::write(&path, raw).unwrap();
    let value = search_settings(&state, 13491).unwrap();
    assert!(!value.to_string().contains("runner-secret"));
    let rev = value["search"]["revision"].as_str().unwrap();
    save_search(&state, 13491, rev, "brave", Some("fixture-key".into()))
        .await
        .unwrap();
    let after = std::fs::read_to_string(&path).unwrap();
    assert!(after.starts_with(raw));
    assert!(after.contains("fixture-key"));
    assert!(
        save_search(&state, 13491, rev, "exa", Some("other-key".into()))
            .await
            .is_err()
    );
    assert!(
        !state
            .supervisor
            .list()
            .await
            .iter()
            .any(|r| r.port == 13491)
    );
    let current = search_settings(&state, 13491).unwrap();
    assert!(!current.to_string().contains("fixture-key"));
    assert!(
        save_search(
            &state,
            13491,
            current["search"]["revision"].as_str().unwrap(),
            "exa",
            None
        )
        .await
        .is_err()
    );
    save_search(
        &state,
        13491,
        current["search"]["revision"].as_str().unwrap(),
        "",
        None,
    )
    .await
    .unwrap();
    assert_eq!(std::fs::read_to_string(path).unwrap(), raw);
}

#[test]
fn catalog_projection_is_bounded_and_never_imports_unsafe_urls() {
    let v = catalog_row(
        &json!({"key":"repo:a/b","name":"Fixture","token":"must-not-return","remoteEndpoints":[{"url":"file:///etc/passwd"},{"url":"https://user:secret@example.com"},{"url":"https://example.com/mcp","transport":"streamable-http"}]}),
    );
    assert_eq!(v["remoteEndpoints"].as_array().unwrap().len(), 1);
    assert!(!v.to_string().contains("must-not-return"));
    assert_eq!(
        catalog_detail(
            &json!({"server":{"key":"repo:a/b","name":"Fixture"},"note":"Untrusted catalog"}),
            "repo:a/b"
        )
        .unwrap()["name"],
        "Fixture"
    );
    assert!(catalog_detail(&json!({"server":{"key":"different"}}), "repo:a/b").is_err());
    let detailed = catalog_detail(&json!({"server":{"key":"repo:a/b",
        "categories":["developer"],"spdxLicense":"MIT","tools":["read","write"],
        "repoUrl":"javascript:alert(1)","homepage":"https://example.com/",
        "connection":{"recommended":{"url":"https://user:secret@example.com/mcp"},"authRequired":true,"note":"Requires an account"}
    }}), "repo:a/b").unwrap();
    assert_eq!(detailed["spdxLicense"], "MIT");
    assert_eq!(detailed["tools"], json!(["read", "write"]));
    assert!(detailed["repoUrl"].is_null());
    assert!(detailed["connection"]["recommendedURL"].is_null());
    assert_eq!(detailed["connection"]["authRequired"], true);
}

#[tokio::test]
async fn auth_gated_check_is_reachable_but_not_authorized_or_saved() {
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", socket.local_addr().unwrap());
    let app = axum::Router::new().route(
        "/mcp",
        axum::routing::post(|| async { axum::http::StatusCode::UNAUTHORIZED }),
    );
    let server = tokio::spawn(async move {
        axum::serve(socket, app).await.unwrap();
    });
    let state = Arc::new(AppState::for_tests());
    let checked = tokio::time::timeout(
        Duration::from_secs(5),
        execute(state.clone(), Operation::Check { draft: draft(&url) }),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(checked["authRequired"], true);
    assert_eq!(checked["tools"], json!([]));
    assert!(state.db.native_connectors().unwrap().is_empty());
    server.abort();
}

#[tokio::test]
async fn native_probe_lists_real_mcp_without_calling_a_tool() {
    use axum::{Json, Router, response::IntoResponse, routing::post};
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", socket.local_addr().unwrap());
    let app=Router::new().route("/mcp",post(|Json(body):Json<Value>|async move{
        let result=match body["method"].as_str().unwrap_or(""){
            "initialize"=>json!({"protocolVersion":body["params"]["protocolVersion"],"capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1"}}),
            "tools/list"=>json!({"tools":[{"name":"fixture_tool","description":"Read only fixture","inputSchema":{"type":"object"}}]}),
            "notifications/initialized"=>return axum::http::StatusCode::ACCEPTED.into_response(),
            other=>panic!("Unexpected MCP method: {other}"),
        };
        Json(json!({"jsonrpc":"2.0","id":body["id"],"result":result})).into_response()
    }));
    let server = tokio::spawn(async move {
        axum::serve(socket, app).await.unwrap();
    });
    let state = Arc::new(AppState::for_tests());
    let value = tokio::time::timeout(
        Duration::from_secs(5),
        execute(state.clone(), Operation::Check { draft: draft(&url) }),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(value["tools"][0]["name"], "fixture_tool");
    assert!(state.db.native_connectors().unwrap().is_empty());
    server.abort();
}

#[tokio::test]
async fn native_scope_and_delete_reject_stale_reviews_and_preserve_handwritten_tools() {
    let (_dir, state) = isolated_state();
    let saved = execute(
        state.clone(),
        Operation::Save {
            draft: draft("https://example.invalid/mcp"),
        },
    )
    .await
    .unwrap();
    let id = saved["savedId"].as_str().unwrap().to_owned();
    let path = state.supervisor.server_config_path(13511);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let original = "# preserve\nmodel='fixture.gguf'\n[[mcp_servers]]\nserver_label='hand'\nserver_url='https://manual.invalid/mcp'\n";
    std::fs::write(&path, original).unwrap();
    assert_eq!(state.db.native_connectors().unwrap()[0]["system"], false);
    execute(
        state.clone(),
        Operation::Scope {
            id: id.clone(),
            revision: 1,
            all: false,
            ports: vec![13511],
        },
    )
    .await
    .unwrap();
    assert!(std::fs::read_to_string(&path).unwrap().contains(&id));
    assert!(
        execute(
            state.clone(),
            Operation::Remove {
                id: id.clone(),
                revision: 1
            }
        )
        .await
        .is_err()
    );
    assert!(
        execute(
            state.clone(),
            Operation::Scope {
                id: id.clone(),
                revision: 1,
                all: true,
                ports: vec![]
            }
        )
        .await
        .is_err()
    );
    execute(state.clone(), Operation::Remove { id, revision: 2 })
        .await
        .unwrap();
    assert!(state.db.native_connectors().unwrap().is_empty());
    let after = std::fs::read_to_string(path).unwrap();
    assert!(
        after.contains("manual.invalid")
            && after.contains("# preserve")
            && !after.contains("example.invalid")
    );
}

#[tokio::test]
async fn oauth_issuer_state_and_revision_checks_precede_token_publication() {
    use axum::{
        Json, Router,
        extract::Query,
        routing::{get, post},
    };
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicUsize, Ordering},
    };
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", socket.local_addr().unwrap());
    let url = format!("{base}/mcp");
    let issuer = base.clone();
    let resource = url.clone();
    let exchanges = Arc::new(AtomicUsize::new(0));
    let count = exchanges.clone();
    let app=Router::new()
        .route("/mcp",post(||async{axum::http::StatusCode::UNAUTHORIZED}))
        .route("/.well-known/oauth-protected-resource/mcp",get(move||{let issuer=issuer.clone();let resource=resource.clone();async move{Json(json!({"resource":resource,"authorization_servers":[issuer]}))}}))
        .route("/.well-known/oauth-authorization-server",get({let base=base.clone();move||{let base=base.clone();async move{Json(json!({"issuer":base,"authorization_endpoint":format!("{base}/authorize"),"token_endpoint":format!("{base}/token"),"code_challenge_methods_supported":["S256"],"authorization_response_iss_parameter_supported":true}))}}}))
        .route("/token",post(move||{let count=count.clone();async move{count.fetch_add(1,Ordering::SeqCst);Json(json!({"access_token":"synthetic-access","refresh_token":"synthetic-refresh","expires_in":3600}))}}));
    let server = tokio::spawn(async move {
        axum::serve(socket, app).await.unwrap();
    });
    let state = Arc::new(AppState::for_tests());
    let id = execute(state.clone(), Operation::Save { draft: draft(&url) })
        .await
        .unwrap()["savedId"]
        .as_str()
        .unwrap()
        .to_owned();
    async fn start(state: Arc<AppState>, id: &str) -> HashMap<String, String> {
        let v = execute_with_origin(
            state,
            Operation::SignIn {
                id: id.into(),
                revision: 1,
                client_id: Some("native-fixture".into()),
            },
            Some("http://127.0.0.1:12400".into()),
        )
        .await
        .unwrap();
        let url = reqwest::Url::parse(v["authorization"]["url"].as_str().unwrap()).unwrap();
        let params = url
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect::<HashMap<_, _>>();
        assert_eq!(params["code_challenge_method"], "S256");
        assert_eq!(params["code_challenge"].len(), 43);
        params
    }
    let first = start(state.clone(), &id).await;
    let bad = HashMap::from([
        ("state".into(), first["state"].clone()),
        ("code".into(), "fixture-code".into()),
        ("iss".into(), "https://wrong.invalid".into()),
    ]);
    crate::oauth::callback(State(state.clone()), Query(bad)).await;
    assert_eq!(exchanges.load(Ordering::SeqCst), 0);
    let second = start(state.clone(), &id).await;
    let good = HashMap::from([
        ("state".into(), second["state"].clone()),
        ("code".into(), "fixture-code".into()),
        ("iss".into(), base),
    ]);
    crate::oauth::callback(State(state.clone()), Query(good.clone())).await;
    assert_eq!(exchanges.load(Ordering::SeqCst), 1);
    assert_eq!(state.db.native_connectors().unwrap()[0]["connected"], true);
    assert!(
        !state.db.native_connectors().unwrap()[0]
            .to_string()
            .contains("synthetic-access")
    );
    crate::oauth::callback(State(state.clone()), Query(good)).await;
    assert_eq!(
        exchanges.load(Ordering::SeqCst),
        1,
        "callback must be one-use"
    );
    assert!(
        state
            .db
            .set_connector_oauth_checked(&id, 1, "{\"access_token\":\"late\"}")
            .is_err()
    );
    server.abort();
}
