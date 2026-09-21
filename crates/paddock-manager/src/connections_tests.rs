use super::*;
use axum::{
    Router,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::get,
};
use serde_json::json;

fn store() -> Store {
    Store::open(&std::path::PathBuf::from(":memory:")).unwrap()
}
fn draft(base: &str) -> Draft {
    serde_json::from_value(json!({"name":"Fixture", "kind":"openai-compat", "baseUrl":base, "apiKey":"", "allowUnauthenticated":true})).unwrap()
}
fn picks() -> Vec<Pick> {
    serde_json::from_value(json!([{"id":"fixture-model", "ctx":8192}])).unwrap()
}
async fn fixture(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", socket.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(socket, app).await.unwrap();
    });
    (base, task)
}

#[test]
fn url_and_pick_policy_is_explicit() {
    for bad in [
        "file:///tmp/a",
        "http://example.com/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?key=x",
        "https://example.com/v1#x",
        "https://example.com/v1/chat/completions",
    ] {
        assert!(normalized_base(bad).is_err(), "{bad}");
    }
    assert_eq!(
        normalized_base(" http://127.0.0.1:9000/v1/ ").unwrap(),
        "http://127.0.0.1:9000/v1"
    );
    assert!(normalized_base("http://[::1]:9000/v1").is_ok());
    assert!(!is_openrouter("https://openrouter.ai.evil.invalid/api/v1"));
    assert!(!is_openrouter("https://other.invalid/openrouter.ai"));
    let mut p = picks();
    p[0].provider = Some("provider/turbo".into());
    assert!(validate_picks(&p, true).is_ok());
    assert!(validate_picks(&p, false).is_err());
    p.push(p[0].clone());
    assert!(validate_picks(&p, true).is_err());
    p[1].provider = Some("provider/region".into());
    assert!(validate_picks(&p, true).is_ok());
}

#[test]
fn stale_edits_never_change_keys_models_or_destinations() {
    let db = store();
    let mut d = draft("https://example.invalid/v1");
    d.api_key = Some("fixture-secret".into());
    d.allow_unauthenticated = false;
    let prepared = prepare(&db, d).unwrap();
    let row = db.save_checked_connection(&prepared, &picks()).unwrap();
    assert!(!row.to_string().contains("fixture-secret"));
    let id = row["id"].as_str().unwrap();
    let mut edit = draft("https://example.invalid/v1");
    edit.id = Some(id.into());
    edit.revision = Some(1);
    edit.api_key = None;
    edit.allow_unauthenticated = false;
    let reviewed = prepare(&db, edit.clone()).unwrap();
    db.update_cloud_endpoint(id, &json!({"name":"Changed elsewhere"}))
        .unwrap();
    assert!(
        db.save_checked_connection(&reviewed, &[])
            .unwrap_err()
            .contains("changed")
    );
    assert!(db.set_connection_models(id, 1, &[]).is_err());
    assert!(db.remove_connection(id, 1).is_err());
    assert_eq!(
        db.cloud_endpoint_secret(id).unwrap().unwrap().2,
        "fixture-secret"
    );
    assert_eq!(
        db.list_cloud_endpoints().unwrap()[0]["models"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    edit.revision = Some(2);
    edit.base_url = "https://different.invalid/v1".into();
    assert!(
        prepare(&db, edit)
            .err()
            .unwrap()
            .contains("entering its key again")
    );
}

#[test]
fn saved_choices_survive_reopen_and_keyless_access_requires_explicit_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paddock.db");
    let db = Store::open(&path).unwrap();
    let mut d = draft("http://localhost:9988/v1");
    d.allow_unauthenticated = false;
    assert!(prepare(&db, d.clone()).is_err());
    d.allow_unauthenticated = true;
    let prepared = prepare(&db, d).unwrap();
    let row = db.save_checked_connection(&prepared, &picks()).unwrap();
    let id = row["id"].as_str().unwrap();
    drop(db);
    let db = Store::open(&path).unwrap();
    let listed = db.list_cloud_endpoints().unwrap();
    assert_eq!(listed[0], row);
    assert!(db.connection_allows_unauthenticated(id));
    db.remove_connection(id, 1).unwrap();
    assert!(db.list_cloud_endpoints().unwrap().is_empty());
}

#[tokio::test]
async fn probe_does_not_persist_or_generate_and_sends_no_empty_auth_header() {
    let app = Router::new().route(
        "/v1/models",
        get(|headers: HeaderMap| async move {
            assert!(headers.get("authorization").is_none());
            axum::Json(json!({"data":[{"id":"fixture-model"},{"id":"fixture-model"}]}))
        }),
    );
    let (base, task) = fixture(app).await;
    let db = store();
    let prepared = prepare(&db, draft(&base)).unwrap();
    let rows = probe(&prepared).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert!(db.list_cloud_endpoints().unwrap().is_empty());
    task.abort();
}

#[tokio::test]
async fn probes_reject_redirects_errors_html_and_oversized_bodies_without_echoing_secrets() {
    let app = Router::new()
        .route(
            "/bad/models",
            get(|| async {
                (StatusCode::UNAUTHORIZED, "fixture-secret must not echo").into_response()
            }),
        )
        .route(
            "/redirect/models",
            get(|| async { axum::response::Redirect::temporary("/target") }),
        )
        .route("/html/models", get(|| async { "<html>ok</html>" }))
        .route(
            "/huge/models",
            get(|| async { "x".repeat(4 * 1024 * 1024 + 1) }),
        );
    let (base, task) = fixture(app).await;
    let origin = base.trim_end_matches("/v1");
    for route in ["bad", "redirect", "html", "huge"] {
        let mut d = draft(&format!("{origin}/{route}"));
        d.api_key = Some("fixture-secret".into());
        d.allow_unauthenticated = false;
        let message = probe(&prepare(&store(), d).unwrap()).await.unwrap_err();
        assert!(!message.contains("fixture-secret"));
        if route == "redirect" {
            assert!(message.contains("redirect"));
        }
    }
    task.abort();
}

#[test]
fn commands_do_not_accept_privileged_or_ambiguous_input() {
    assert!(
        serde_json::from_value::<Draft>(
            json!({"name":"x","kind":"openai-compat","baseUrl":OPENROUTER,"path":"/api/keys"})
        )
        .is_err()
    );
    let db = store();
    let mut d = draft(OPENROUTER);
    assert!(prepare(&db, d.clone()).is_err());
    d.api_key = Some("bad\nheader".into());
    d.allow_unauthenticated = false;
    assert!(prepare(&db, d).is_err());
}

#[test]
fn credential_reference_publication_and_retirement_never_put_native_keys_in_sqlite() {
    // The test vault implements the same reference contract without touching
    // the user's OS Keychain or causing authorization dialogs.
    let db = store();
    let mut d = draft("https://example.invalid/v1");
    d.api_key = Some("fixture-first".into());
    d.allow_unauthenticated = false;
    let row = db
        .save_checked_connection(&prepare(&db, d.clone()).unwrap(), &picks())
        .unwrap();
    let id = row["id"].as_str().unwrap();
    let first = db.connection_for_revision(id, 1).unwrap().2;
    assert!(first.starts_with("paddock-keychain:v1:"));
    assert!(!first.contains("fixture-first"));
    d.id = Some(id.into());
    d.revision = Some(1);
    d.api_key = Some("fixture-second".into());
    db.save_checked_connection(&prepare(&db, d).unwrap(), &picks())
        .unwrap();
    let second = db.connection_for_revision(id, 2).unwrap().2;
    assert_ne!(first, second);
    assert!(crate::credentials::resolve(first).is_err());
    assert_eq!(
        crate::credentials::resolve(second.clone()).unwrap(),
        "fixture-second"
    );
    db.update_cloud_endpoint(id, &json!({"apiKey":"fixture-web-replacement"}))
        .unwrap();
    let third = db.connection_for_revision(id, 3).unwrap().2;
    assert!(third.starts_with("paddock-keychain:v1:"));
    assert_eq!(
        crate::credentials::resolve(third.clone()).unwrap(),
        "fixture-web-replacement"
    );
    assert!(
        db.update_cloud_endpoint(id, &json!({"baseUrl":"https://different.invalid/v1"}))
            .is_err()
    );
    db.remove_connection(id, 3).unwrap();
    assert!(crate::credentials::resolve(third).is_err());
    assert!(crate::credentials::resolve(second).is_err());
}
