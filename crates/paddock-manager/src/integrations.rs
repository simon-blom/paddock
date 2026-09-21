//! Native tools management, sharing the connector library and runner config.
//! This is not a privileged HTTP tunnel: destinations and output fields are
//! allowlisted here. Network operations are read-only unless explicitly saved.
use crate::routes::AppState;
use axum::{
    Json,
    extract::{Path, State},
    response::Response,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock},
    time::Duration,
};

#[cfg(test)]
#[path = "integrations_tests.rs"]
mod tests;

static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("TLS available")
});

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Draft {
    pub id: Option<String>,
    pub revision: Option<u64>,
    pub label: String,
    pub url: String,
    pub headers: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub registry_key: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    List {},
    Search {
        query: String,
    },
    Detail {
        key: String,
    },
    Check {
        draft: Draft,
    },
    Save {
        draft: Draft,
    },
    Remove {
        id: String,
        revision: u64,
    },
    Scope {
        id: String,
        revision: u64,
        all: bool,
        ports: Vec<u16>,
    },
    Tools {
        id: String,
    },
    Unlock {
        id: String,
        revision: u64,
    },
    SearchSettings {
        port: u16,
    },
    SaveSearch {
        port: u16,
        revision: String,
        provider: String,
        key: Option<String>,
    },
    SignIn {
        id: String,
        revision: u64,
        client_id: Option<String>,
    },
    CancelSignIn {
        id: String,
    },
    Disconnect {
        id: String,
        revision: u64,
    },
}
impl Operation {
    pub fn mutation(&self) -> bool {
        matches!(
            self,
            Self::Save { .. }
                | Self::Remove { .. }
                | Self::Scope { .. }
                | Self::SaveSearch { .. }
                | Self::SignIn { .. }
                | Self::CancelSignIn { .. }
                | Self::Disconnect { .. }
                | Self::Unlock { .. }
        )
    }
}

pub fn safe_url(raw: &str) -> Result<String, String> {
    let url = reqwest::Url::parse(raw.trim()).map_err(|_| "Enter a complete server URL.")?;
    let local = url.host_str().is_some_and(|h| {
        h == "localhost"
            || h.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|a| a.is_loopback())
    });
    if raw.len() > 2048
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.query().is_some()
        || !(url.scheme() == "https" || url.scheme() == "http" && local)
    {
        return Err(
            "Use HTTPS, or HTTP on localhost. Put credentials in headers, not in the URL.".into(),
        );
    }
    Ok(url.to_string())
}
fn id_valid(id: &str) -> Result<(), String> {
    uuid::Uuid::parse_str(id)
        .map(|_| ())
        .map_err(|_| "Invalid connector identity.".into())
}
fn prepare(state: &Arc<AppState>, draft: Draft) -> Result<Value, String> {
    let url = safe_url(&draft.url)?;
    if draft.label.is_empty()
        || draft.label.len() > 64
        || !draft
            .label
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        || draft.registry_key.len() > 256
    {
        return Err("Use a unique label of 1-64 letters, digits, hyphens or underscores.".into());
    }
    let old = if let Some(id) = &draft.id {
        id_valid(id)?;
        let row = state
            .db
            .native_connectors()
            .map_err(|_| "The connector library is unavailable.")?
            .into_iter()
            .find(|r| r["id"] == *id)
            .ok_or("This connector was removed.")?;
        if row["revision"].as_u64() != draft.revision {
            return Err("This connector changed. Reload it before continuing.".into());
        }
        Some(row)
    } else {
        if draft.revision.is_some() {
            return Err("Invalid connector revision.".into());
        }
        None
    };
    let headers = if let Some(headers) = draft.headers {
        if headers.len() > 16 {
            return Err("At most 16 credential headers are supported.".into());
        }
        let mut names = std::collections::HashSet::new();
        for (k, v) in &headers {
            let name = k.to_ascii_lowercase();
            if !names.insert(name.clone())
                || matches!(
                    name.as_str(),
                    "host"
                        | "content-length"
                        | "connection"
                        | "transfer-encoding"
                        | "origin"
                        | "cookie"
                        | "accept"
                        | "content-type"
                )
                || name.starts_with("mcp-")
                || k.len() > 128
                || v.len() > 8192
                || axum::http::HeaderName::from_bytes(k.as_bytes()).is_err()
                || axum::http::HeaderValue::from_str(v).is_err()
                || v.chars().any(char::is_control)
            {
                return Err("Review credential headers. Duplicate, reserved or invalid headers are not allowed.".into());
            }
        }
        json!(headers)
    } else if let Some(old) = &old {
        if old["url"]
            .as_str()
            .and_then(|s| safe_url(s).ok())
            .as_deref()
            != Some(url.as_str())
        {
            return Err(
                "Changing the server URL requires explicit replacement or removal of its headers."
                    .into(),
            );
        }
        state.db.get_connector(old["id"].as_str().unwrap_or("")).map_err(|_|"Saved headers are unavailable. Unlock Keychain or explicitly replace the headers.")?
            .ok_or("This connector was removed.")?["headers"].clone()
    } else {
        json!({})
    };
    Ok(
        json!({"id":draft.id,"revision":draft.revision,"label":draft.label,"url":url,"headers":headers,"registryKey":draft.registry_key}),
    )
}

pub async fn execute(state: Arc<AppState>, op: Operation) -> Result<Value, String> {
    execute_with_origin(state, op, None).await
}

pub async fn execute_with_origin(
    state: Arc<AppState>,
    op: Operation,
    origin: Option<String>,
) -> Result<Value, String> {
    match op {
        Operation::List {} => {
            let rows = blocking(move || {
                state
                    .db
                    .native_connectors()
                    .map_err(|_| "Could not load connectors.".into())
            })
            .await?;
            Ok(json!({"connectors":rows}))
        }
        Operation::Search { query } => {
            if query.len() > 256 || query.chars().any(char::is_control) {
                return Err("Search text is too long or invalid.".into());
            }
            let mut url = reqwest::Url::parse("https://registry.truespar.com/v1/servers")
                .map_err(|_| "Registry unavailable.")?;
            url.query_pairs_mut()
                .append_pair("hostable", "true")
                .append_pair("limit", "50")
                .append_pair("q", query.trim());
            let body = fetch(url).await?;
            let rows = body["results"]
                .as_array()
                .ok_or("Invalid connector catalog.")?;
            if rows.len() > 50 {
                return Err("Connector catalog exceeds its limit.".into());
            }
            Ok(json!({"results":rows.iter().map(catalog_row).collect::<Vec<_>>()}))
        }
        Operation::Detail { key } => {
            if key.is_empty() || key.len() > 256 || key.chars().any(char::is_control) {
                return Err("Invalid catalog identity.".into());
            }
            let mut url = reqwest::Url::parse("https://registry.truespar.com/v1/servers/")
                .map_err(|_| "Registry unavailable.")?;
            url.path_segments_mut()
                .map_err(|_| "Registry unavailable.")?
                .pop_if_empty()
                .push(&key);
            let body = fetch(url).await?;
            Ok(json!({"detail":catalog_detail(&body,&key)?}))
        }
        Operation::Check { draft } => {
            let oauth_id = if draft.headers.is_none() {
                draft.id.clone()
            } else {
                None
            };
            let db = state.clone();
            let doc = blocking(move || {
                let mut doc = prepare(&db, draft)?;
                if let Some(id) = oauth_id {
                    let row = db
                        .db
                        .get_connector(&id)
                        .map_err(|_| "Saved credentials are unavailable.")?
                        .ok_or("This connector was removed.")?;
                    if row["revision"] != doc["revision"] {
                        return Err("The connector changed during the check. Reload it.".into());
                    }
                    if let Some(token) = row["oauth"]["access_token"].as_str()
                        && let Some(headers) = doc["headers"].as_object_mut()
                        && !headers
                            .keys()
                            .any(|k| k.eq_ignore_ascii_case("authorization"))
                    {
                        headers.insert("Authorization".into(), json!(format!("Bearer {token}")));
                    }
                }
                Ok(doc)
            })
            .await?;
            let cfg = paddock_mcp::ServerConfig {
                id: "native-review".into(),
                label: "review".into(),
                transport: paddock_mcp::Transport::Http {
                    url: doc["url"].as_str().unwrap_or_default().into(),
                    headers: serde_json::from_value(doc["headers"].clone())
                        .map_err(|_| "Invalid headers.")?,
                },
            };
            let client = match paddock_mcp::McpClient::connect(&cfg).await {
                Ok(client) => client,
                Err(_) => {
                    // Distinguish a reachable, auth-gated server from a failed
                    // handshake. Read headers only; never buffer a probe body.
                    let mut probe = HTTP
                        .post(doc["url"].as_str().unwrap_or_default())
                        .header("Accept", "application/json, text/event-stream")
                        .json(&json!({"jsonrpc":"2.0","id":0,"method":"initialize",
                          "params":{"protocolVersion":"2026-07-28","capabilities":{},
                            "clientInfo":{"name":"paddock","version":env!("CARGO_PKG_VERSION")}}}));
                    if let Some(headers) = doc["headers"].as_object() {
                        for (key, value) in headers {
                            if let Some(value) = value.as_str() {
                                probe = probe.header(key, value);
                            }
                        }
                    }
                    if probe
                        .send()
                        .await
                        .is_ok_and(|r| r.status() == reqwest::StatusCode::UNAUTHORIZED)
                    {
                        return Ok(json!({"authRequired":true,"tools":[],
                          "message":"Server is reachable but requires credentials. Save it to sign in; no tool was executed."}));
                    }
                    return Err("MCP handshake failed. Check the URL and credentials, or explicitly save anyway.".into());
                }
            };
            let tools = client
                .list_tools()
                .await
                .map_err(|_| "The server did not return a valid tool list.")?;
            if tools.len() > 4096 {
                return Err("This server exposes too many tools for a review.".into());
            }
            Ok(
                json!({"tools":tools.iter().map(|t| json!({"name":short(&t.name,256),"description":short(t.description.as_deref().unwrap_or(""),2048)})).collect::<Vec<_>>(),"message":"MCP handshake and tool listing succeeded. No tool was executed."}),
            )
        }
        Operation::Save { draft } => {
            let _guard = crate::connectors::MUTATIONS.lock().await;
            let s = state.clone();
            let id = blocking(move || {
                let doc = prepare(&s, draft)?;
                s.db.save_native_connector(&doc).map_err(store_error)
            })
            .await?;
            // Existing scope is preserved, never broadened by editing credentials.
            let s = state.clone();
            let saved = id.clone();
            let warning =
                blocking(move || Ok(crate::connectors::rematerialize_checked(&s, &saved).err()))
                    .await?;
            Ok(
                json!({"savedId":id,"message":warning.map(|_|"Saved to the library, but endpoint synchronization failed. Reapply the connector scope before using it.").unwrap_or("Connector saved. Enable it in the composer's Tools and connectors picker.")}),
            )
        }
        Operation::Remove { id, revision } => {
            id_valid(&id)?;
            let mut headers = axum::http::HeaderMap::new();
            headers.insert(
                "x-paddock-revision",
                revision
                    .to_string()
                    .parse()
                    .map_err(|_| "Invalid revision.")?,
            );
            response(crate::connectors::remove(State(state), Path(id), headers).await).await?;
            Ok(json!({"message":"Connector removed. Conversations were preserved."}))
        }
        Operation::Scope {
            id,
            revision,
            all,
            mut ports,
        } => {
            id_valid(&id)?;
            ports.sort_unstable();
            ports.dedup();
            if ports.len() > 128
                || ports
                    .iter()
                    .any(|p| *p < 1024 || !state.supervisor.server_config_path(*p).is_file())
                || all && !ports.is_empty()
            {
                return Err("Select existing endpoints or every model, not both.".into());
            }
            response(
                crate::connectors::scope(
                    State(state),
                    Path(id),
                    Json(json!({"all":all,"ports":ports,"revision":revision})),
                )
                .await,
            )
            .await?;
            Ok(json!({"message":"Connector scope applied. Models were not restarted."}))
        }
        Operation::Unlock { id, revision } => {
            id_valid(&id)?;
            blocking(move || {
                state
                    .db
                    .unlock_connector(&id, revision)
                    .map_err(store_error)
            })
            .await?;
            Ok(json!({"message":"Connector access is ready. No tool was executed."}))
        }
        Operation::Tools { id } => {
            id_valid(&id)?;
            let body = response(
                crate::connectors::tools(State(state), Json(json!({"connector_id":id}))).await,
            )
            .await?;
            if body["ok"] != true {
                return Err(
                    "Tool listing failed. Check connectivity and credentials, then retry.".into(),
                );
            }
            Ok(json!({"tools":body["tools"],"message":"Tools listed. No tool was executed."}))
        }
        Operation::SearchSettings { port } => search_settings(&state, port),
        Operation::SaveSearch {
            port,
            revision,
            provider,
            key,
        } => save_search(&state, port, &revision, &provider, key).await,
        Operation::SignIn {
            id,
            revision,
            client_id,
        } => {
            id_valid(&id)?;
            if client_id
                .as_ref()
                .is_some_and(|s| s.len() > 1024 || s.chars().any(char::is_control))
            {
                return Err("Invalid OAuth client ID.".into());
            }
            let rows = state
                .db
                .native_connectors()
                .map_err(|_| "Connector library unavailable.")?;
            let row = rows
                .iter()
                .find(|r| r["id"] == id && r["revision"].as_u64() == Some(revision))
                .ok_or("This connector changed. Reload before signing in.")?;
            safe_url(row["url"].as_str().unwrap_or(""))?;
            let origin = origin.ok_or("Open Studio once before starting browser sign-in.")?;
            let host = reqwest::Url::parse(&origin)
                .map_err(|_| "Invalid callback host.")?
                .authority()
                .to_owned();
            let mut headers = axum::http::HeaderMap::new();
            headers.insert("host", host.parse().map_err(|_| "Invalid callback host.")?);
            // The origin is supplied by the actual app-owned listener, never
            // by Swift JSON, a catalog record or a model-produced URL.
            let started = tokio::time::timeout(
                Duration::from_secs(60),
                crate::oauth::start(
                    State(state.clone()),
                    Path(id.clone()),
                    headers,
                    Json(json!({"client_id":client_id})),
                ),
            )
            .await
            .map_err(|_| "Authorization discovery timed out.")?;
            let result = response(started).await?;
            let url = result["url"]
                .as_str()
                .ok_or("No authorization URL was returned.")?;
            if url.len() > 16384 {
                return Err("Authorization URL exceeds its limit.".into());
            }
            let parsed = reqwest::Url::parse(url).map_err(|_| "Invalid authorization URL.")?;
            if !["https", "http"].contains(&parsed.scheme()) {
                return Err("Invalid authorization URL scheme.".into());
            }
            Ok(
                json!({"authorization":{"connectorId":id,"url":url,"revision":revision},"message":"Review the authorization server before opening your browser."}),
            )
        }
        Operation::CancelSignIn { id } => {
            id_valid(&id)?;
            crate::oauth::cancel(&id);
            state
                .db
                .invalidate_connector_revision(&id)
                .map_err(|_| "Could not cancel sign-in. The connector may have been removed.")?;
            Ok(json!({"message":"Sign-in cancelled. Existing saved credentials were not removed."}))
        }
        Operation::Disconnect { id, revision } => {
            let _guard = crate::connectors::MUTATIONS.lock().await;
            id_valid(&id)?;
            crate::oauth::cancel(&id);
            let s = state.clone();
            let saved = id.clone();
            blocking(move || {
                s.db.set_connector_oauth_checked(&saved, revision, "")
                    .map_err(|_| "This connector changed. Reload before disconnecting.".into())
            })
            .await?;
            crate::connectors::rematerialize_checked(&state,&id).map_err(|_|"Sign-in was removed, but endpoint synchronization failed. Reapply the connector scope.")?;
            Ok(json!({"message":"OAuth sign-in removed. Manually saved headers were not changed."}))
        }
    }
}

fn store_error(e: crate::store::StoreError) -> String {
    match e {
        crate::store::StoreError::Bad(s) => s,
        _ => "Could not save the connector. Nothing was changed.".into(),
    }
}
pub async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|_| "The operation failed unexpectedly.")?
}
async fn response(r: Response) -> Result<Value, String> {
    let status = r.status();
    let bytes = axum::body::to_bytes(r.into_body(), 2 * 1024 * 1024)
        .await
        .map_err(|_| "Response exceeds its size limit.")?;
    if !status.is_success() {
        return Err(if status == axum::http::StatusCode::CONFLICT {
            "This connector changed. Reload it before continuing."
        } else {
            "The connector operation failed. Reload and check its settings before retrying."
        }
        .into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "Invalid connector response.".into())
}
async fn fetch(url: reqwest::Url) -> Result<Value, String> {
    let mut r = HTTP
        .get(url)
        .send()
        .await
        .map_err(|_| "The connector registry could not be reached.")?;
    if !r.status().is_success() {
        return Err("The connector registry refused the request. Retry later.".into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = r.chunk().await.map_err(|_| "Catalog download failed.")? {
        if bytes.len() + chunk.len() > 2 * 1024 * 1024 {
            return Err("Connector catalog exceeds its size limit.".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| "Invalid connector catalog.".into())
}
fn short(s: &str, n: usize) -> String {
    s.chars()
        .filter(|c| !c.is_control() || *c == '\n')
        .take(n)
        .collect()
}
fn catalog_row(v: &Value) -> Value {
    let endpoints=v["remoteEndpoints"].as_array().into_iter().flatten().take(16)
        .filter_map(|e|safe_url(e["url"].as_str()?).ok().map(|url|json!({"url":url,"transport":short(e["transport"].as_str().unwrap_or(""),32)}))).collect::<Vec<_>>();
    json!({"key":short(v["key"].as_str().unwrap_or(""),256),"name":short(v["name"].as_str().unwrap_or(""),256),
        "description":short(v["description"].as_str().unwrap_or(""),4096),"domain":short(v["domain"].as_str().unwrap_or(""),256),
        "authorityTier":short(v["authorityTier"].as_str().unwrap_or(""),32),"liveness":short(v["liveness"].as_str().unwrap_or(""),32),
        "githubStars":v["githubStars"].as_u64(),"toolCount":v["toolCount"].as_u64(),"remoteEndpoints":endpoints})
}

fn catalog_detail(body: &Value, key: &str) -> Result<Value, String> {
    let row = body.get("server").unwrap_or(body);
    if row["key"].as_str() != Some(key) {
        return Err("The catalog returned a different connector. Retry the search.".into());
    }
    let mut detail = catalog_row(row);
    for (field, limit) in [("categories", 24), ("tools", 256)] {
        detail[field] = json!(
            row[field]
                .as_array()
                .into_iter()
                .flatten()
                .take(limit)
                .filter_map(Value::as_str)
                .map(|s| short(s, 256))
                .collect::<Vec<_>>()
        );
    }
    for field in ["spdxLicense", "lastHandshakeAt"] {
        detail[field] = json!(row[field].as_str().map(|s| short(s, 128)));
    }
    for field in ["repoUrl", "homepage"] {
        detail[field] = json!(row[field].as_str().and_then(|s| safe_url(s).ok()));
    }
    detail["connection"] = json!({
        "recommendedURL":row["connection"]["recommended"]["url"].as_str().and_then(|s|safe_url(s).ok()),
        "authRequired":row["connection"]["authRequired"].as_bool(),
        "note":row["connection"]["note"].as_str().map(|s|short(s,2048))
    });
    Ok(detail)
}

fn search_settings(state: &Arc<AppState>, port: u16) -> Result<Value, String> {
    if port < 1024 {
        return Err("Select a saved model endpoint.".into());
    }
    let (raw, revision) = state
        .supervisor
        .read_config_file(port)
        .map_err(|_| "The endpoint configuration is unavailable.")?;
    let doc: toml::Value =
        toml::from_str(&raw).map_err(|_| "The endpoint configuration is invalid.")?;
    Ok(json!({"search":{"port":port,"revision":revision,
        "provider":doc.get("web_search_provider").and_then(toml::Value::as_str).unwrap_or(""),
        "hasKey":doc.get("web_search_api_key").and_then(toml::Value::as_str).is_some_and(|s|!s.is_empty())}}))
}
async fn save_search(
    state: &Arc<AppState>,
    port: u16,
    revision: &str,
    provider: &str,
    key: Option<String>,
) -> Result<Value, String> {
    let _guard = crate::connectors::MUTATIONS.lock().await;
    if port < 1024 || !["", "exa", "tavily", "firecrawl", "brave", "perplexity"].contains(&provider)
    {
        return Err("Select a supported web-search provider and saved endpoint.".into());
    }
    let (raw, current) = state
        .supervisor
        .read_config_file(port)
        .map_err(|_| "The endpoint configuration is unavailable.")?;
    if current != revision {
        return Err("This endpoint changed. Reload it before saving.".into());
    }
    let text = patch_search(&raw, provider, key)?;
    state
        .supervisor
        .write_live_tools_config(port, &text, revision)
        .map_err(
            |_| "The endpoint changed or its config could not be saved. Reload before retrying.",
        )?;
    Ok(
        json!({"message":"Web search saved. Applies on the next request; no model was restarted. No paid search was performed."}),
    )
}

/// Start and Edit share provider validation and key replacement semantics.
pub(crate) fn patch_search(
    raw: &str,
    provider: &str,
    key: Option<String>,
) -> Result<String, String> {
    if !["", "exa", "tavily", "firecrawl", "brave", "perplexity"].contains(&provider) {
        return Err("Select a supported web-search provider.".into());
    }
    let mut doc: toml_edit::DocumentMut =
        raw.parse().map_err(|_| "Invalid endpoint configuration.")?;
    let old_provider = doc
        .get("web_search_provider")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if key.is_none() && provider != old_provider && !provider.is_empty() {
        return Err("Changing search providers requires that provider's key.".into());
    }
    let key = key.unwrap_or_else(|| {
        doc.get("web_search_api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned()
    });
    if !provider.is_empty()
        && (key.trim().is_empty() || key.len() > 8192 || key.chars().any(char::is_control))
    {
        return Err("Enter a valid search-provider key.".into());
    }
    if provider.is_empty() {
        doc.remove("web_search_provider");
        doc.remove("web_search_api_key");
    } else {
        doc["web_search_provider"] = toml_edit::value(provider);
        doc["web_search_api_key"] = toml_edit::value(key.trim());
    }
    Ok(doc.to_string())
}

/// Same-directory replace: runners never read a half-written credential or
/// TOML document. Temporary files are private and flushed before publication.
pub(crate) fn atomic_config(path: &std::path::Path, content: &str) -> Result<(), String> {
    use std::io::Write;
    let parent = path.parent().ok_or("Invalid endpoint location.")?;
    let mut file =
        tempfile::NamedTempFile::new_in(parent).map_err(|_| "Cannot prepare endpoint settings.")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|_| "Cannot protect endpoint settings.")?;
    }
    file.write_all(content.as_bytes())
        .and_then(|_| file.as_file().sync_all())
        .map_err(|_| "Cannot write endpoint settings.")?;
    file.persist(path)
        .map_err(|_| "Cannot publish endpoint settings.")?;
    Ok(())
}
