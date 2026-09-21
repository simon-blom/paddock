//! Native presentation allowlist. Conversation JSON can include future fields
//! and inline connector settings. Preserve them in Rust/SQLite, but never copy
//! them across FFI and rely on Swift's decoder to ignore sensitive fields.
use serde_json::Value;

fn keep(value: &mut Value, fields: &[&str]) -> Result<(), String> {
    value
        .as_object_mut()
        .ok_or("Invalid chat projection")?
        .retain(|key, _| fields.contains(&key.as_str()));
    Ok(())
}
pub fn project(value: &mut Value) -> Result<(), String> {
    keep(
        value,
        &[
            "conversation",
            "conversations",
            "stream_id",
            "events",
            "done",
            "accepted",
        ],
    )?;
    if let Some(doc) = value.get_mut("conversation") {
        keep(doc, &["id", "title", "model", "messages"])?;
        for message in doc["messages"]
            .as_array_mut()
            .ok_or("Invalid chat messages")?
        {
            keep(
                message,
                &[
                    "id",
                    "role",
                    "content",
                    "reasoning",
                    "streaming",
                    "stopped",
                    "error",
                    "incomplete",
                    "nativeStatus",
                    "model",
                    "usage",
                ],
            )?;
            for part in message["content"]
                .as_array_mut()
                .ok_or("Invalid chat content")?
            {
                keep(part, &["type", "text"])?;
            }
            if let Some(usage) = message.get_mut("usage").filter(|u| !u.is_null()) {
                keep(
                    usage,
                    &["promptTokens", "completionTokens", "reasoningTokens"],
                )?;
            }
        }
    }
    if let Some(rows) = value.get_mut("conversations") {
        for row in rows.as_array_mut().ok_or("Invalid chat history")? {
            keep(row, &["id", "title", "model", "updatedAt"])?;
        }
    }
    if let Some(events) = value.get_mut("events") {
        for event in events.as_array_mut().ok_or("Invalid chat events")? {
            keep(event, &["kind", "output_index", "content_index", "delta"])?;
        }
    }
    if let Some(done) = value.get_mut("done").filter(|d| !d.is_null()) {
        keep(done, &["conversation_id", "status", "error", "save_error"])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn private_and_unknown_fields_never_cross_the_chat_boundary() {
        let mut value = json!({"conversation":{"id":"c","title":"title","model":"m","toolSelection":[{"headers":{"Authorization":"secret"}}],"unknown":"secret","messages":[{"id":"a","role":"assistant","content":[{"type":"image","url":"secret","headers":"secret"},{"type":"text","text":"visible"}],"response":{"future":"secret"},"toolCalls":"secret","usage":{"promptTokens":3,"future":"secret"}}]}});
        project(&mut value).unwrap();
        assert!(!value.to_string().contains("secret"));
        assert_eq!(
            value["conversation"]["messages"][0]["content"][1]["text"],
            "visible"
        );
        assert_eq!(
            value["conversation"]["messages"][0]["usage"]["promptTokens"],
            3
        );
    }
}
