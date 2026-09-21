//! Anonymous, read-only catalog requests. No manager handle, credentials or
//! caller-supplied URL: slow internet queries cannot hold up local lifecycle work.
use axum::extract::Query;
use serde::Deserialize;
use std::{collections::HashMap, sync::LazyLock, time::Duration};

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Catalog {},
    Providers { model: String },
}

static RUNTIME: LazyLock<Result<tokio::runtime::Runtime, String>> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("paddock-catalog")
        .enable_all()
        .build()
        .map_err(|_| "The cloud catalog worker could not start".into())
});
static SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);

fn request(bytes: &[u8]) -> Result<Request, String> {
    let request: Request =
        serde_json::from_slice(bytes).map_err(|_| "Invalid cloud catalog request")?;
    if let Request::Providers { model } = &request {
        let parts: Vec<_> = model.split('/').collect();
        if model.len() > 256
            || parts.len() != 2
            || parts.iter().any(|part| {
                part.is_empty()
                    || *part == "."
                    || *part == ".."
                    || !part
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_.:~".contains(&b))
            })
        {
            return Err("Invalid OpenRouter model identifier".into());
        }
    }
    Ok(request)
}

pub fn browse(bytes: &[u8]) -> Result<String, String> {
    let request = request(bytes)?;
    let _slot = SLOTS
        .try_acquire()
        .map_err(|_| "OpenRouter requests are still finishing. Please try again shortly.")?;
    RUNTIME.as_ref().map_err(Clone::clone)?.block_on(async {
        tokio::time::timeout(Duration::from_secs(45), async {
            // Reuse exactly the web host's public catalog and pricing projections.
            let response = match request {
                Request::Catalog {} => paddock_manager::cloud::browse().await,
                Request::Providers { model } => {
                    paddock_manager::cloud::browse_endpoints(Query(HashMap::from([(
                        "model".into(),
                        model,
                    )])))
                    .await
                }
            };
            let status = response.status();
            if !status.is_success() {
                // Never forward arbitrary upstream HTML/error bodies to the app.
                return Err(format!(
                    "OpenRouter could not load this catalog (HTTP {}). Please try again.",
                    status.as_u16()
                ));
            }
            let bytes = axum::body::to_bytes(response.into_body(), super::MAX_SNAPSHOT)
                .await
                .map_err(|_| "OpenRouter's catalog exceeds the supported size")?;
            String::from_utf8(bytes.to_vec()).map_err(|_| "Invalid catalog text".into())
        })
        .await
        .map_err(|_| "OpenRouter took too long to respond. Please try again.".to_string())?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_typed_public_catalog_requests_are_accepted() {
        assert!(request(br#"{"kind":"catalog"}"#).is_ok());
        assert!(request(br#"{"kind":"providers","model":"qwen/qwen3.8-27b:free"}"#).is_ok());
        assert!(request(br#"{"kind":"providers","model":"~openai/gpt-astra-latest"}"#).is_ok());
        for invalid in [
            r#"{"kind":"catalog","api_key":"test-only"}"#,
            r#"{"kind":"catalog","url":"https://example.com"}"#,
            r#"{"kind":"delete"}"#,
            r#"{"kind":"providers","model":"a/../keys"}"#,
            r#"{"kind":"providers","model":"a/.."}"#,
            r#"{"kind":"providers","model":"a/b?key=test-only"}"#,
            r#"{"kind":"providers","model":"a/%2e%2e"}"#,
            r#"{"kind":"providers","model":"a/b#fragment"}"#,
            r#"{"kind":"providers","model":"/"}"#,
        ] {
            assert!(request(invalid.as_bytes()).is_err(), "{invalid}");
        }
    }
}
