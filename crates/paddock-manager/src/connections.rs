//! Shared connection policy for native management. Checking a draft never
//! persists it or performs inference. The caller owns cancellation; this module
//! bounds time/body size and never returns provider bodies or credential errors.
use crate::store::Store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashSet, sync::LazyLock, time::Duration};

pub const OPENROUTER: &str = "https://openrouter.ai/api/v1";
static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("TLS available")
});

#[cfg(test)]
#[path = "connections_tests.rs"]
mod tests;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Draft {
    pub id: Option<String>,
    pub revision: Option<u64>,
    pub name: String,
    pub kind: String,
    pub base_url: String,
    /// None preserves a stored credential, Some("") explicitly clears it.
    pub api_key: Option<String>,
    #[serde(default)]
    pub allow_unauthenticated: bool,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Pick {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_out: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

pub fn validate_picks(picks: &[Pick], openrouter: bool) -> Result<(), String> {
    if picks.len() > 256 {
        return Err("At most 256 models can be enabled per connection.".into());
    }
    let mut seen = HashSet::new();
    for pick in picks {
        if !plain(&pick.id, 256)
            || pick.id.contains('@')
            || pick.id.starts_with("cloud:")
            || pick.display.as_ref().is_some_and(|s| !plain(s, 256))
            || pick
                .provider
                .as_ref()
                .is_some_and(|s| !openrouter || !plain(s, 128) || s.contains('@'))
            || pick.ctx.is_some_and(|n| n == 0 || n > 100_000_000)
            || pick.max_out.is_some_and(|n| n == 0 || n > 100_000_000)
            || !seen.insert((&pick.id, &pick.provider))
        {
            return Err("Review the model IDs, provider choices and limits. Duplicate or invalid models cannot be saved.".into());
        }
    }
    Ok(())
}
fn plain(s: &str, max: usize) -> bool {
    !s.trim().is_empty() && s == s.trim() && s.len() <= max && !s.chars().any(char::is_control)
}

pub fn normalized_base(base: &str) -> Result<String, String> {
    let url = reqwest::Url::parse(base.trim()).map_err(|_| "Enter a complete API base URL.")?;
    let local = url.host_str().is_some_and(|h| {
        h == "localhost"
            || h.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    if base.len() > 2048
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.host_str().is_none()
        || !(url.scheme() == "https" || (url.scheme() == "http" && local))
    {
        return Err("Use HTTPS, or HTTP for localhost only. Keep credentials, query parameters and fragments out of the URL.".into());
    }
    let path = url.path().trim_end_matches('/');
    if ["/responses", "/chat/completions", "/models", "/messages"]
        .iter()
        .any(|s| path.ends_with(s))
    {
        return Err(
            "Enter the API base URL (usually ending in /v1), not a models or generation route."
                .into(),
        );
    }
    Ok(url.to_string().trim_end_matches('/').into())
}
pub fn is_openrouter(base: &str) -> bool {
    base.trim_end_matches('/') == OPENROUTER
}

pub struct Prepared {
    pub draft: Draft,
    pub key: String,
}

pub fn prepare(db: &Store, mut draft: Draft) -> Result<Prepared, String> {
    draft.name = draft.name.trim().into();
    if !plain(&draft.name, 128)
        || !matches!(
            draft.kind.as_str(),
            "openai-compat" | "openai" | "anthropic"
        )
    {
        return Err("Enter a name and a supported API format.".into());
    }
    draft.base_url = normalized_base(&draft.base_url)?;
    let saved = if let Some(id) = &draft.id {
        if uuid::Uuid::parse_str(id).is_err() {
            return Err("Invalid connection identity.".into());
        }
        Some(
            db.connection_for_revision(
                id,
                draft
                    .revision
                    .ok_or("Refresh this connection before editing.")?,
            )?,
        )
    } else {
        if draft.revision.is_some() {
            return Err("A new connection cannot have a saved revision.".into());
        }
        None
    };
    let key = match draft.api_key.take() {
        Some(key) => key.trim().to_owned(),
        None => match saved {
            Some((kind, base, stored)) => {
                if kind != draft.kind || base != draft.base_url {
                    return Err("Changing the API address or format requires entering its key again. The old key will not be sent to a new destination.".into());
                }
                db.resolve_cloud_credential(stored)?
            }
            None => String::new(),
        },
    };
    if key.len() > 8192 || key.chars().any(char::is_control) {
        return Err("The API key has an invalid length or contains control characters.".into());
    }
    if key.is_empty() && (!draft.allow_unauthenticated || is_openrouter(&draft.base_url)) {
        return Err("Enter an API key, or explicitly choose a custom endpoint that requires no authentication.".into());
    }
    if !key.is_empty() && draft.allow_unauthenticated {
        return Err("Choose API key authentication or no authentication, not both.".into());
    }
    Ok(Prepared { draft, key })
}

/// OpenRouter's model list is public, so only /key verifies its credential.
/// A custom /models success proves reachability/schema, not generation support
/// or model entitlement; the UI says that explicitly.
pub async fn probe(prepared: &Prepared) -> Result<Vec<Value>, String> {
    let d = &prepared.draft;
    let openrouter = is_openrouter(&d.base_url);
    let url = format!(
        "{}/{}",
        d.base_url,
        if openrouter { "key" } else { "models" }
    );
    let mut request = HTTP.get(url);
    if d.kind == "anthropic" {
        request = request.header("anthropic-version", "2023-06-01");
        if !prepared.key.is_empty() {
            request = request.header("x-api-key", &prepared.key);
        }
    } else if !prepared.key.is_empty() {
        request = request.bearer_auth(&prepared.key);
    }
    let response = request.send().await.map_err(|e| if e.is_timeout() {
        "The connection timed out. Check the address and network, then retry."
    } else { "The connection could not be reached securely. Check its address, certificate and network." })?;
    if !response.status().is_success() {
        return Err(match response.status().as_u16() {
            401 | 403 => "Authentication was refused. Check the API key and account permissions.",
            300..=399 => "The endpoint redirected the check. Use its final API base URL; credentials were not forwarded.",
            404 => "No model-list API exists at this address. Check the API base URL.",
            429 => "The provider rate-limited the check. Wait before retrying.",
            _ => "The provider could not complete the check. Retry later.",
        }.into());
    }
    let body = bounded_json(response).await?;
    if openrouter {
        let data = body
            .get("data")
            .filter(|v| v.is_object())
            .ok_or("The endpoint did not return valid key information.")?;
        if data.get("is_management_key").and_then(Value::as_bool) == Some(true)
            || data.get("is_provisioning_key").and_then(Value::as_bool) == Some(true)
        {
            return Err("Use an inference API key, not an account-management key.".into());
        }
        return Ok(Vec::new());
    }
    let models = body
        .get("data")
        .and_then(Value::as_array)
        .ok_or("The endpoint answered, but not with a compatible model list.")?;
    if models.len() > 4096 {
        return Err("The endpoint returned too many models.".into());
    }
    let mut seen = HashSet::new();
    Ok(models
        .iter()
        .filter_map(|m| crate::cloud::normalize_provider_model(&d.kind, m))
        .filter(|m| {
            m["id"]
                .as_str()
                .is_some_and(|id| plain(id, 256) && seen.insert(id.to_owned()))
        })
        .collect())
}

async fn bounded_json(response: reqwest::Response) -> Result<Value, String> {
    use futures_util::StreamExt;
    const MAX: usize = 4 * 1024 * 1024;
    if response.content_length().is_some_and(|n| n > MAX as u64) {
        return Err("The provider response exceeds the connection-check limit.".into());
    }
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| "The connection check was interrupted.")?;
        if bytes.len() + chunk.len() > MAX {
            return Err("The provider response exceeds the connection-check limit.".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| "The endpoint did not return valid JSON.".into())
}
