//! Bounded Responses decoding for embedded Studio clients. The network remains
//! the existing manager relay; this module has no URLs, credentials or UI code.
//! EOF is never success. Full terminal output is authoritative, including text
//! a provider delivered only in its terminal event. See OpenAI's semantic-event
//! contract: https://developers.openai.com/api/docs/guides/streaming-responses
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const MAX_EVENT: usize = 1024 * 1024;
pub const MAX_OUTPUT: usize = 4 * 1024 * 1024;

/// UTF-8 is decoded only after a complete line has arrived. CR, LF, CRLF and
/// multiline data work across arbitrary transport chunks, including split UTF-8.
#[derive(Default)]
pub struct Sse {
    line: Vec<u8>,
    data: String,
    cr: bool,
}
impl Sse {
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        let mut frames = Vec::new();
        for &b in bytes {
            if self.cr {
                self.cr = false;
                if b == b'\n' {
                    continue;
                }
            }
            if b == b'\r' || b == b'\n' {
                self.cr = b == b'\r';
                let line = std::str::from_utf8(&self.line).map_err(|_| "Invalid stream UTF-8")?;
                if line.is_empty() {
                    if !self.data.is_empty() {
                        frames.push(std::mem::take(&mut self.data));
                    }
                } else if let Some(value) = line.strip_prefix("data:") {
                    if !self.data.is_empty() {
                        self.data.push('\n');
                    }
                    self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
                }
                self.line.clear();
            } else {
                self.line.push(b);
            }
            if self.line.len() + self.data.len() > MAX_EVENT {
                return Err("Response event exceeds 1 MiB".into());
            }
        }
        Ok(frames)
    }
    pub fn finish(&mut self) -> Result<Vec<String>, String> {
        self.push(b"\n\n")
    }
}

#[derive(Default)]
pub struct ResponseText {
    parts: BTreeMap<(u64, u64), String>,
    reasoning: BTreeMap<(u64, u64), String>,
    pub status: Option<String>,
    pub error: Option<String>,
    pub usage: Option<Value>,
    pub response: Option<Value>,
    pub sequence: Option<u64>,
    bytes: usize,
}
impl ResponseText {
    pub fn text(&self) -> String {
        self.parts.values().cloned().collect::<Vec<_>>().join("\n")
    }
    pub fn reasoning(&self) -> String {
        self.reasoning
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
    /// Deltas are forwarded as semantic JSON, never interpreted by Swift.
    /// Renderer adapters can share the same event vocabulary as browser Studio.
    pub fn apply(&mut self, data: &str) -> Result<Option<Value>, String> {
        if data == "[DONE]" {
            return Ok(None);
        }
        if self.status.is_some() {
            return Err("Response event arrived after its terminal event".into());
        }
        let event: Value = serde_json::from_str(data).map_err(|_| "Malformed Responses event")?;
        let kind = event["type"].as_str().ok_or("Response event has no type")?;
        if let Some(sequence) = event["sequence_number"].as_u64() {
            if self.sequence.is_some_and(|old| sequence <= old) {
                return Err("Response event sequence is not increasing".into());
            }
            self.sequence = Some(sequence);
        }
        let key = (
            event["output_index"].as_u64().unwrap_or(0),
            event["content_index"]
                .as_u64()
                .or(event["summary_index"].as_u64())
                .unwrap_or(0),
        );
        match kind {
            "response.output_text.delta"
            | "response.refusal.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning.delta" => {
                let delta = event["delta"]
                    .as_str()
                    .ok_or("Response delta is not text")?;
                self.bytes = self
                    .bytes
                    .checked_add(delta.len())
                    .ok_or("Response size overflow")?;
                if self.bytes > MAX_OUTPUT {
                    return Err("Response exceeds the native transcript's 4 MiB limit".into());
                }
                let reasoning = kind.contains("reasoning");
                let parts = if reasoning {
                    &mut self.reasoning
                } else {
                    &mut self.parts
                };
                parts.entry(key).or_default().push_str(delta);
                Ok(Some(
                    json!({"kind": if reasoning {"reasoning"} else {"text"}, "output_index":key.0,"content_index":key.1,"delta":delta}),
                ))
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                let response = event
                    .get("response")
                    .filter(|r| r.is_object())
                    .ok_or("Terminal event has no response")?;
                let status = kind.trim_start_matches("response.");
                if response["status"].as_str().is_some_and(|s| s != status) {
                    return Err("Terminal response status disagrees with its event".into());
                }
                self.status = Some(status.into());
                self.error = response["error"]["message"].as_str().map(str::to_owned);
                self.usage = response.get("usage").cloned();
                // Retain the full terminal object in the shared conversation
                // document: future clients must not lose unknown output items.
                self.response = Some(response.clone());
                if response.to_string().len() > MAX_OUTPUT {
                    return Err("Terminal response exceeds 4 MiB".into());
                }
                if let Some(items) = response["output"].as_array() {
                    let mut text = BTreeMap::new();
                    let mut reasoning = BTreeMap::new();
                    for (i, item) in items.iter().enumerate() {
                        if item["type"] != "message" && item["type"] != "reasoning" {
                            return Err("Runner returned an output type not connected in native text chat; its terminal response was retained".into());
                        }
                        for field in ["content", "summary"] {
                            if let Some(parts) = item[field].as_array() {
                                for (j, part) in parts.iter().enumerate() {
                                    let value = part["text"].as_str().or(part["refusal"].as_str());
                                    if let Some(value) = value {
                                        let target = if item["type"] == "reasoning" {
                                            &mut reasoning
                                        } else {
                                            &mut text
                                        };
                                        target.insert((i as u64, j as u64), value.to_owned());
                                    }
                                }
                            }
                        }
                    }
                    // An explicitly empty message is authoritative too. Some
                    // local runners omit reasoning from the final object, so
                    // retain its streamed value when there is no reasoning item.
                    if items.iter().any(|i| i["type"] == "message") {
                        self.parts = text;
                    }
                    if items.iter().any(|i| i["type"] == "reasoning") {
                        self.reasoning = reasoning;
                    }
                }
                Ok(None)
            }
            "error" => Err(event["message"]
                .as_str()
                .unwrap_or("Runner reported a stream error")
                .to_owned()),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_chunk_boundaries_and_newlines() {
        let input = "event: x\r\ndata: {\"type\":\"x\",\r\ndata: \"text\":\"å😀\"}\r\n\r\n: comment\ndata: [DONE]\n\n";
        for n in 1..input.len() {
            let mut s = Sse::default();
            let mut frames = Vec::new();
            for bytes in input.as_bytes().chunks(n) {
                frames.extend(s.push(bytes).unwrap());
            }
            frames.extend(s.finish().unwrap());
            assert_eq!(
                frames,
                vec!["{\"type\":\"x\",\n\"text\":\"å😀\"}", "[DONE]"]
            );
        }
    }
    #[test]
    fn terminal_tail_and_reasoning_are_preserved() {
        let mut r = ResponseText::default();
        r.apply(r#"{"type":"response.output_text.delta","delta":"first","sequence_number":0}"#)
            .unwrap();
        r.apply(
            r#"{"type":"response.reasoning_text.delta","delta":"thought","sequence_number":1}"#,
        )
        .unwrap();
        assert!(r.status.is_none());
        r.apply(r#"{"type":"response.incomplete","sequence_number":2,"response":{"status":"incomplete","output":[{"type":"message","content":[{"type":"output_text","text":"first tail"}]}],"usage":{"output_tokens":8}}}"#).unwrap();
        assert_eq!(r.text(), "first tail");
        assert_eq!(r.reasoning(), "thought");
        assert_eq!(r.status.as_deref(), Some("incomplete"));
        assert_eq!(r.usage.unwrap()["output_tokens"], 8);
    }
    #[test]
    fn malformed_duplicate_and_missing_terminal_do_not_succeed() {
        let mut r = ResponseText::default();
        assert!(r.apply("bad").is_err());
        let e = r#"{"type":"response.output_text.delta","delta":"hi","sequence_number":1}"#;
        r.apply(e).unwrap();
        assert!(r.apply(e).is_err());
        r.apply("[DONE]").unwrap();
        assert!(r.status.is_none());
        assert!(Sse::default().push(&vec![b'x'; MAX_EVENT + 1]).is_err());
    }
}
