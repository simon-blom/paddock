//! OAuth 2.1 for MCP connectors - the spec's auth for HTTP transports. The
//! MANAGER runs the interactive flow:
//!
//!   401 -> protected-resource metadata (RFC 9728) -> authorization-server
//!   metadata (RFC 8414 / OIDC fallback) -> dynamic client registration
//!   (RFC 7591, manual client_id as the fallback) -> authorization code +
//!   PKCE S256 in the user's browser with a loopback redirect
//!   (`/api/connectors/oauth/callback` on this manager) -> tokens stored on
//!   the connector row.
//!
//! Tokens then ride exactly like pasted headers: the API layer merges
//! `Authorization: Bearer ...` into the connector's headers (inline tool specs
//! and materialized TOML entries alike), and a refresh re-materializes scoped
//! rows - the runners' live registry picks the new token up on the next
//! request, no restarts. The runner itself never does interactive auth.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Json, Response};
use base64::Engine as _;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::routes::AppState;

static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(8))
        // 45s total: the check-then-save probe hits real MCP servers that
        // cold-start (IIS app pools take 15-30s to spin up).
        // Dead hosts still fail fast on connect; only a
        // warming server actually spends the budget.
        .timeout(std::time::Duration::from_secs(45))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("TLS backend available")
});

/// In-flight authorization flows, keyed by the `state` parameter. Entries are
/// short-lived (the user is mid-browser-redirect); stale ones are swept on
/// each start.
struct Pending {
    connector_id: String,
    verifier: String,
    token_endpoint: String,
    client_id: String,
    client_secret: Option<String>,
    redirect_uri: String,
    resource: String,
    created: std::time::Instant,
    revision: u64,
    issuer: String,
    require_issuer: bool,
}
static FLOWS: LazyLock<Mutex<HashMap<String, Pending>>> = LazyLock::new(Mutex::default);

fn b64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn err(status: StatusCode, msg: String) -> Response {
    (
        status,
        Json(json!({"error": {"type": "oauth_error", "message": msg}})),
    )
        .into_response()
}

async fn get_json(url: &str, resource: &str) -> Option<Value> {
    let res = crate::oauth_http::client(url, resource)
        .await
        .ok()?
        .get(url)
        .header("Accept", "application/json")
        .send()
        .await
        .ok()?;
    if !res.status().is_success() {
        return None;
    }
    crate::oauth_http::json(res).await.ok()
}

const INIT_BODY: &str = r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"paddock","version":"0.1"}}}"#;

/// Lightweight MCP handshake probe (initialize over streamable HTTP) - the
/// pre-save check for the connector form. 401 = reachable but wants
/// credentials (fine to save); success = handshake (server name when the
/// reply is plain JSON; an SSE reply proves the handshake without one).
pub async fn probe(url: &str, headers: &serde_json::Map<String, Value>) -> Value {
    let mut req = HTTP
        .post(url)
        .header("Accept", "application/json, text/event-stream")
        .header("Content-Type", "application/json")
        .body(INIT_BODY);
    for (k, v) in headers {
        if let Some(s) = v.as_str() {
            req = req.header(k, s);
        }
    }
    match req.send().await {
        Ok(r) if r.status() == StatusCode::UNAUTHORIZED => {
            json!({"ok": false, "auth_required": true})
        }
        Ok(r) if r.status().is_success() => {
            let name = r.json::<Value>().await.ok().and_then(|v| {
                v.pointer("/result/serverInfo/name")
                    .and_then(Value::as_str)
                    .map(String::from)
            });
            json!({"ok": true, "server": name})
        }
        Ok(r) => json!({"ok": false, "error": format!("answered HTTP {}", r.status())}),
        Err(e) => json!({"ok": false, "error": e.to_string()}),
    }
}

struct Discovered {
    authorization_endpoint: String,
    token_endpoint: String,
    registration_endpoint: Option<String>,
    scopes: Option<String>,
    issuer: String,
    require_issuer: bool,
}

/// Tiered discovery, the way real MCP clients do it: the 401's
/// `WWW-Authenticate: ... resource_metadata="..."` pointer first, then the
/// RFC 9728 well-known locations at the server origin. The selected issuer's
/// RFC 8414 / OIDC metadata must match exactly and advertise PKCE S256.
async fn discover(server_url: &str) -> Result<Discovered, String> {
    let parsed = reqwest::Url::parse(server_url).map_err(|e| format!("bad server url: {e}"))?;
    let origin = format!("{}://{}", parsed.scheme(), parsed.authority());
    let path = parsed.path().trim_end_matches('/');

    // 1. provoke a 401 and read its pointer
    let mut prm_urls: Vec<String> = Vec::new();
    let mut challenged_scope = None;
    if let Ok(res) = crate::oauth_http::client(server_url, server_url)
        .await?
        .post(server_url)
        .header("Accept", "application/json, text/event-stream")
        .header("Content-Type", "application/json")
        .body(INIT_BODY)
        .send()
        .await
        && res.status() == StatusCode::UNAUTHORIZED
        && let Some(www) = res
            .headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok())
        && let Some(idx) = www.find("resource_metadata=")
    {
        challenged_scope = www
            .split("scope=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .filter(|s| s.len() <= 4096)
            .map(str::to_owned);
        let rest = &www[idx + "resource_metadata=".len()..];
        let url = rest
            .trim_start_matches('"')
            .split('"')
            .next()
            .unwrap_or("")
            .to_string();
        if !url.is_empty() {
            prm_urls.push(url);
        }
    }
    // 2. well-known fallbacks
    if !path.is_empty() {
        prm_urls.push(format!(
            "{origin}/.well-known/oauth-protected-resource{path}"
        ));
    }
    prm_urls.push(format!("{origin}/.well-known/oauth-protected-resource"));

    let mut issuer: Option<String> = None;
    let mut scopes: Option<String> = None;
    for u in &prm_urls {
        if let Some(prm) = get_json(u, server_url).await {
            if prm.get("resource").and_then(Value::as_str).is_none_or(|r| {
                crate::integrations::safe_url(r).ok()
                    != crate::integrations::safe_url(server_url).ok()
            }) {
                continue;
            }
            if let Some(a) = prm
                .get("authorization_servers")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(Value::as_str)
            {
                issuer = Some(a.to_string());
            }
            scopes = prm
                .get("scopes_supported")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" ")
                });
            if issuer.is_some() {
                break;
            }
        }
    }
    // Never guess that the resource server is also its authorization server.
    let issuer = issuer.ok_or("The server did not provide valid protected-resource metadata.")?;
    let issuer_url = reqwest::Url::parse(&issuer).map_err(|_| "Invalid authorization issuer.")?;

    for meta_url in [
        format!(
            "{}://{}/.well-known/oauth-authorization-server{}",
            issuer_url.scheme(),
            issuer_url.authority(),
            issuer_url.path().trim_end_matches('/')
        ),
        format!(
            "{}://{}/.well-known/openid-configuration{}",
            issuer_url.scheme(),
            issuer_url.authority(),
            issuer_url.path().trim_end_matches('/')
        ),
        format!(
            "{}/.well-known/openid-configuration",
            issuer.trim_end_matches('/')
        ),
    ] {
        if let Some(meta) = get_json(&meta_url, server_url).await
            && let (Some(auth), Some(token)) = (
                meta.get("authorization_endpoint").and_then(Value::as_str),
                meta.get("token_endpoint").and_then(Value::as_str),
            )
        {
            if meta["issuer"].as_str() != Some(issuer.as_str()) {
                continue;
            }
            if !meta["code_challenge_methods_supported"]
                .as_array()
                .is_some_and(|a| a.iter().any(|v| v == "S256"))
            {
                return Err(
                    "The authorization server does not advertise required PKCE S256 support."
                        .into(),
                );
            }
            crate::integrations::safe_url(auth)?;
            crate::integrations::safe_url(token)?;
            return Ok(Discovered {
                authorization_endpoint: auth.to_string(),
                token_endpoint: token.to_string(),
                registration_endpoint: meta
                    .get("registration_endpoint")
                    .and_then(Value::as_str)
                    .map(String::from),
                scopes: challenged_scope.or(scopes),
                issuer: issuer.clone(),
                require_issuer: meta["authorization_response_iss_parameter_supported"] == true,
            });
        }
    }
    Err(format!(
        "no OAuth metadata found for {server_url} - the server may use plain API-key headers \
         instead of sign-in"
    ))
}

/// `POST /api/connectors/{id}/oauth/start` - body may carry `{"client_id"}`
/// for authorization servers without dynamic registration. Returns the
/// authorize URL for the Studio to open in the user's browser.
pub async fn start(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    // Starting/replacing sign-in only needs identity and revision. Do not
    // unlock an old credential merely to let the user reconnect.
    let row = match state.db.native_connectors().map(|rows| {
        rows.into_iter()
            .find(|row| row["id"].as_str() == Some(id.as_str()))
    }) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return err(
                StatusCode::BAD_REQUEST,
                format!("no connector with id {id}"),
            );
        }
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let server_url = row["url"].as_str().unwrap_or("").to_string();
    let d = match discover(&server_url).await {
        Ok(d) => d,
        Err(e) => return err(StatusCode::BAD_GATEWAY, e),
    };
    // the redirect must come back to this manager - the address the Studio
    // reached us on is exactly that
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("127.0.0.1:11500");
    let redirect_uri = format!("http://{host}/api/connectors/oauth/callback");

    // client: caller-supplied, else dynamic registration (RFC 7591)
    let mut client_id = body
        .get("client_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut client_secret: Option<String> = None;
    if client_id.is_empty() {
        let Some(reg) = &d.registration_endpoint else {
            return err(
                StatusCode::BAD_GATEWAY,
                "this authorization server has no dynamic registration - enter a client id \
                 from the provider's app settings"
                    .into(),
            );
        };
        let reg_client = match crate::oauth_http::client(reg, &server_url).await {
            Ok(c) => c,
            Err(e) => return err(StatusCode::BAD_GATEWAY, e),
        };
        let reg_res = reg_client
            .post(reg)
            .json(&json!({
                "client_name": "Paddock",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
                "application_type": "native",
            }))
            .send()
            .await;
        match reg_res {
            Ok(r) if r.status().is_success() => {
                let v = crate::oauth_http::json(r).await.unwrap_or(Value::Null);
                client_id = v
                    .get("client_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                client_secret = v
                    .get("client_secret")
                    .and_then(Value::as_str)
                    .map(String::from);
            }
            Ok(r) => {
                return err(
                    StatusCode::BAD_GATEWAY,
                    format!("client registration refused (HTTP {})", r.status()),
                );
            }
            Err(e) => return err(StatusCode::BAD_GATEWAY, format!("client registration: {e}")),
        }
        if client_id.is_empty() {
            return err(
                StatusCode::BAD_GATEWAY,
                "registration returned no client_id".into(),
            );
        }
    }

    // PKCE S256 + state
    let verifier = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let challenge = {
        use sha2::Digest;
        b64url(&sha2::Sha256::digest(verifier.as_bytes()))
    };
    let flow_state = uuid::Uuid::new_v4().simple().to_string();
    {
        let mut flows = FLOWS.lock().unwrap_or_else(|e| e.into_inner());
        flows.retain(|_, p| p.created.elapsed() < std::time::Duration::from_secs(600));
        flows.retain(|_, p| p.connector_id != id);
        if flows.len() >= 32 {
            return err(
                StatusCode::TOO_MANY_REQUESTS,
                "Too many sign-in flows. Wait for an existing flow to expire.".into(),
            );
        }
        flows.insert(
            flow_state.clone(),
            Pending {
                connector_id: id,
                verifier,
                token_endpoint: d.token_endpoint,
                client_id: client_id.clone(),
                client_secret,
                redirect_uri: redirect_uri.clone(),
                resource: server_url.clone(),
                created: std::time::Instant::now(),
                revision: row["revision"].as_u64().unwrap_or(0),
                issuer: d.issuer.clone(),
                require_issuer: d.require_issuer,
            },
        );
    }
    let mut url = match reqwest::Url::parse(&d.authorization_endpoint) {
        Ok(u) => u,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("authorization_endpoint: {e}"),
            );
        }
    };
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &client_id)
        .append_pair("redirect_uri", &redirect_uri)
        .append_pair("state", &flow_state)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        // RFC 8707: bind the token to this MCP server
        .append_pair("resource", &server_url);
    if let Some(s) = &d.scopes
        && !s.is_empty()
    {
        url.query_pairs_mut().append_pair("scope", s);
    }
    Json(json!({"url": url.as_str()})).into_response()
}

/// `GET /api/connectors/oauth/callback?code=...&state=...` - the browser lands
/// here after the provider's consent screen. Exchanges the code, stores the
/// tokens, re-materializes scoped rows, and tells the human to close the tab.
pub async fn callback(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let page = |title: &str, body: &str| {
        let body = body
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;");
        Html(format!(
            "<!doctype html><meta charset=utf-8><title>{title}</title>\
             <body style=\"font-family:system-ui;display:grid;place-items:center;height:90vh\">\
             <div style=\"text-align:center\"><h2>{title}</h2><p>{body}</p></div>"
        ))
        .into_response()
    };
    let Some(st) = q.get("state") else {
        return page(
            "Sign-in failed",
            "the provider sent no code - close this tab and try again.",
        );
    };
    let Some(p) = FLOWS.lock().unwrap_or_else(|e| e.into_inner()).remove(st) else {
        return page(
            "Sign-in expired",
            "this window is stale - close it and press Connect again.",
        );
    };
    if p.created.elapsed() > std::time::Duration::from_secs(600)
        || q.get("iss").is_some_and(|issuer| issuer != &p.issuer)
        || p.require_issuer && !q.contains_key("iss")
    {
        return page(
            "Sign-in refused",
            "The sign-in expired or the authorization issuer did not match.",
        );
    }
    if q.contains_key("error") {
        return page(
            "Sign-in cancelled",
            "The authorization server did not grant access. You can close this tab.",
        );
    }
    let Some(code) = q.get("code").filter(|s| !s.is_empty() && s.len() <= 8192) else {
        return page(
            "Sign-in failed",
            "The provider returned no valid authorization code.",
        );
    };
    if state
        .db
        .native_connectors()
        .ok()
        .and_then(|rows| rows.into_iter().find(|r| r["id"] == p.connector_id))
        .and_then(|r| r["revision"].as_u64())
        != Some(p.revision)
    {
        return page(
            "Sign-in expired",
            "The connector changed. Start sign-in again.",
        );
    }
    let mut form = vec![
        ("grant_type", "authorization_code".to_string()),
        ("code", code.clone()),
        ("redirect_uri", p.redirect_uri.clone()),
        ("client_id", p.client_id.clone()),
        ("code_verifier", p.verifier.clone()),
        ("resource", p.resource.clone()),
    ];
    if let Some(sec) = &p.client_secret {
        form.push(("client_secret", sec.clone()));
    }
    let client = match crate::oauth_http::client(&p.token_endpoint, &p.resource).await {
        Ok(c) => c,
        Err(_) => return page("Sign-in refused", "Invalid authorization destination."),
    };
    let res = client.post(&p.token_endpoint).form(&form).send().await;
    let tokens = match res {
        Ok(r) if r.status().is_success() => crate::oauth_http::json(r).await.unwrap_or(Value::Null),
        Ok(_) => {
            return page(
                "Sign-in failed",
                "The authorization server refused the token exchange.",
            );
        }
        Err(e) => return page("Sign-in failed", &format!("token exchange: {e}")),
    };
    let Some(access) = tokens.get("access_token").and_then(Value::as_str) else {
        return page("Sign-in failed", "the provider returned no access token.");
    };
    let expires_at = tokens
        .get("expires_in")
        .and_then(Value::as_u64)
        .map(|s| now_ms().saturating_add(s.saturating_sub(60).saturating_mul(1000)));
    let blob = json!({
        "access_token": access,
        "refresh_token": tokens.get("refresh_token"),
        "expires_at": expires_at,
        "token_endpoint": p.token_endpoint,
        "client_id": p.client_id,
        "client_secret": p.client_secret,
        "resource": p.resource,
    });
    let _guard = crate::connectors::MUTATIONS.lock().await;
    let db_state = state.clone();
    let id = p.connector_id.clone();
    if crate::integrations::blocking(move || {
        db_state
            .db
            .set_connector_oauth_checked(&id, p.revision, &blob.to_string())
            .map_err(|_| "Could not save OAuth credentials.".into())
    })
    .await
    .is_err()
    {
        return page(
            "Sign-in failed",
            "The connector changed or Keychain could not save the credentials. Try again.",
        );
    }
    crate::connectors::rematerialize(&state, &p.connector_id);
    page(
        "Connected",
        "Paddock is signed in to this server - you can close this tab.",
    )
}

/// `POST /api/connectors/{id}/oauth/disconnect` - drop the tokens (and the
/// bearer from any materialized entries).
pub async fn disconnect(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let _guard = crate::connectors::MUTATIONS.lock().await;
    cancel(&id);
    match state.db.set_connector_oauth(&id, "") {
        Ok(()) => {
            crate::connectors::rematerialize(&state, &id);
            Json(json!({"ok": true})).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Cancellation is a backend operation: closing native UI cannot leave an
/// abandoned browser callback able to publish credentials later.
pub fn cancel(id: &str) {
    FLOWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|_, p| p.connector_id != id);
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Refresh a row's access token when it is stale; returns the row with a
/// fresh `oauth` blob (and persists + re-materializes it). Failure marks the
/// blob expired-but-kept so the UI can offer Reconnect.
pub async fn ensure_fresh(state: &Arc<AppState>, row: Value) -> Value {
    let oauth = &row["oauth"];
    if oauth.is_null() {
        return row;
    }
    let stale = oauth
        .get("expires_at")
        .and_then(Value::as_u64)
        .is_some_and(|t| t <= now_ms());
    if !stale {
        return row;
    }
    let Some(refresh) = oauth.get("refresh_token").and_then(Value::as_str) else {
        return row; // expired, nothing to refresh with - Reconnect territory
    };
    let (Some(endpoint), Some(client_id)) = (
        oauth.get("token_endpoint").and_then(Value::as_str),
        oauth.get("client_id").and_then(Value::as_str),
    ) else {
        return row;
    };
    let mut form = vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh.to_string()),
        ("client_id", client_id.to_string()),
    ];
    if let Some(r) = oauth.get("resource").and_then(Value::as_str) {
        form.push(("resource", r.to_string()));
    }
    if let Some(sec) = oauth.get("client_secret").and_then(Value::as_str) {
        form.push(("client_secret", sec.to_string()));
    }
    let resource = row["url"].as_str().unwrap_or("");
    let Ok(client) = crate::oauth_http::client(endpoint, resource).await else {
        return row;
    };
    let Ok(res) = client.post(endpoint).form(&form).send().await else {
        return row;
    };
    if !res.status().is_success() {
        tracing::warn!(connector = %row["label"], status = %res.status(), "token refresh refused - reconnect needed");
        return row;
    }
    let tokens = crate::oauth_http::json(res).await.unwrap_or(Value::Null);
    let Some(access) = tokens.get("access_token").and_then(Value::as_str) else {
        return row;
    };
    let mut blob = oauth.clone();
    blob["access_token"] = json!(access);
    if let Some(rt) = tokens.get("refresh_token").and_then(Value::as_str) {
        blob["refresh_token"] = json!(rt);
    }
    blob["expires_at"] = tokens
        .get("expires_in")
        .and_then(Value::as_u64)
        .map(|s| json!(now_ms().saturating_add(s.saturating_sub(60).saturating_mul(1000))))
        .unwrap_or(Value::Null);
    let id = row["id"].as_str().unwrap_or("");
    let _guard = crate::connectors::MUTATIONS.lock().await;
    let saved = id.to_owned();
    let revision = row["revision"].as_u64().unwrap_or(0);
    let db_state = state.clone();
    let text = blob.to_string();
    if crate::integrations::blocking(move || {
        db_state
            .db
            .set_connector_oauth_checked(&saved, revision, &text)
            .map_err(|_| "OAuth refresh could not be saved.".into())
    })
    .await
    .is_err()
    {
        // Another refresh may have won, or the user may have disconnected.
        // Never return newly obtained credentials after a failed publication.
        return state.db.get_connector(id).ok().flatten().unwrap_or(row);
    }
    crate::connectors::rematerialize(state, id);
    let mut out = row;
    out["oauth"] = blob;
    out
}
