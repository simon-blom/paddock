//! One bounded, credential-free error envelope for HTTP and streaming providers.
//! Parse before clipping: clipping JSON made tool-loop errors unreadable in both
//! Studios. No automatic retries (a partially executed tool could be repeated).
use futures::StreamExt;
use serde_json::{Value, json};

pub(crate) fn normalize(value: &Value, status: Option<u16>) -> Value {
    let e = value
        .get("error")
        .filter(|v| v.is_object())
        .unwrap_or(value);
    let meta = &e["metadata"];
    let code = e["code"]
        .as_u64()
        .or_else(|| e["status"].as_u64())
        .or(status.map(u64::from));
    let raw = meta["raw"].as_str().unwrap_or_default();
    let nested: Option<Value> = serde_json::from_str(raw).ok();
    let detail = nested
        .as_ref()
        .and_then(|v| v.pointer("/error/message").or_else(|| v.get("message")))
        .and_then(Value::as_str)
        .or_else(|| (!raw.trim().is_empty() && !raw.trim().starts_with(['{', '['])).then_some(raw));
    let outer = e["message"].as_str().or_else(|| value["error"].as_str());
    let mut message = detail
        .or(outer)
        .unwrap_or("The provider could not complete this request.")
        .trim();
    if message.starts_with(['{', '[', '<']) {
        message =
            "The provider returned an unreadable error. Retry shortly or choose another model.";
    }
    if detail.is_some()
        && let Some(i) = message.find("https://").or_else(|| message.find("http://"))
    {
        let prefix = message[..i].trim_end();
        message = prefix
            .rfind(['.', '!', '?', ','])
            .map(|j| prefix[..=j].trim_end_matches(','))
            .unwrap_or(prefix);
    }
    if message.is_empty() {
        message = outer
            .filter(|s| !s.trim().is_empty() && !s.trim().starts_with(['{', '[', '<']))
            .unwrap_or("The provider could not complete this request.");
    }
    let provider = meta["provider_name"]
        .as_str()
        .filter(|p| p.len() <= 80 && !p.contains(['\n', '\r']));
    let mut out = json!({"message": message.chars().take(600).collect::<String>()});
    if let Some(p) = provider {
        out["message"] = json!(format!(
            "{p}: {}",
            out["message"].as_str().unwrap_or_default()
        ));
    }
    if let Some(code) = code {
        out["code"] = json!(code);
    }
    let mut details = serde_json::Map::new();
    for k in ["provider_name", "provider_error_code", "limit_source"] {
        if let Some(s) = meta[k].as_str() {
            details.insert(k.into(), json!(s.chars().take(120).collect::<String>()));
        }
    }
    if let Some(v) = meta["missing_attestation_types"].as_array() {
        details.insert(
            "missing_attestation_types".into(),
            Value::Array(
                v.iter()
                    .filter_map(|v| v.as_str())
                    .take(8)
                    .map(|s| json!(s.chars().take(80).collect::<String>()))
                    .collect(),
            ),
        );
    }
    if raw.contains("https://openrouter.ai/settings/integrations") {
        details.insert("action".into(), json!("openrouter_integrations"));
    }
    if !details.is_empty() {
        out["metadata"] = Value::Object(details);
    }
    if let Some(t) = e["type"].as_str() {
        out["type"] = json!(t.chars().take(80).collect::<String>());
    }
    out
}

pub(crate) async fn response(response: reqwest::Response) -> Value {
    let status = response.status().as_u16();
    let retry = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u32>().ok());
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else { break };
        if bytes.len() + chunk.len() > 64 * 1024 {
            bytes.clear();
            break;
        }
        bytes.extend_from_slice(&chunk);
    }
    let value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({"message":"The provider could not complete this request."}));
    let mut e = normalize(&value, Some(status));
    if let Some(seconds) = retry {
        e["retry_after_seconds"] = json!(seconds);
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overloaded_openrouter_body_is_parsed_before_bounding_and_allowlisted() {
        let raw = json!({"error":{"message":"Provider returned error","code":429,"metadata":{
            "raw":"moonshotai/kimi-k3 is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations",
            "provider_name":"DeepInfra","provider_error_code":"engine_overloaded","limit_source":"upstream_provider_shared_pool",
            "headers":{"Authorization":"PRIVATE_SECRET"},"remedy_hint":"x".repeat(1000)}},"user_id":"PRIVATE_ACCOUNT"});
        let e = normalize(&raw, Some(429));
        assert!(e["message"].as_str().unwrap().starts_with("DeepInfra: "));
        assert!(e["message"].as_str().unwrap().contains("retry shortly"));
        assert_eq!(e["metadata"]["action"], "openrouter_integrations");
        assert_eq!(e["metadata"]["provider_error_code"], "engine_overloaded");
        assert!(!e.to_string().contains("PRIVATE_"));
        assert!(!e.to_string().contains("http"));
    }
    #[test]
    fn nested_upstream_json_is_not_displayed_as_a_message() {
        let e = normalize(
            &json!({"error":{"metadata":{"raw":"{\"error\":{\"message\":\"Capacity exhausted\"}}"}}}),
            Some(503),
        );
        assert_eq!(e["message"], "Capacity exhausted");
        assert_eq!(e["code"], 503);
    }
}
