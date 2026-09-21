//! Application-owned text chat sessions over the shared runner client/store.
//! No UI-specific database or HTTP listener. Native commands name an observed
//! port/PID, never a URL or key. The browser's existing conversation JSON stays
//! the durable format; native-only status metadata is additive.
use crate::{
    routes::AppState,
    studio_stream::{MAX_OUTPUT, ResponseText, Sse},
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, watch};

const MAX_DOCUMENT: usize = 8 * 1024 * 1024;
const MAX_TEXT: usize = 128 * 1024;
const MAX_DELTA: usize = 16 * 1024;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    List,
    Load {
        conversation_id: String,
    },
    Send {
        conversation_id: Option<String>,
        port: u16,
        pid: u32,
        text: String,
    },
    Poll {
        stream_id: String,
    },
    Cancel {
        stream_id: String,
    },
}
struct Turn {
    conversation_id: String,
    receiver: mpsc::Receiver<Value>,
    cancel: watch::Sender<bool>,
    done: Arc<Mutex<Option<Value>>>,
}
#[derive(Default)]
pub struct Sessions {
    turns: Mutex<HashMap<String, Turn>>,
}

impl Sessions {
    pub fn cancel_all(&self) {
        if let Ok(turns) = self.turns.lock() {
            for turn in turns.values() {
                let _ = turn.cancel.send(true);
            }
        }
    }
    pub fn active(&self) -> bool {
        self.turns.lock().is_ok_and(|turns| {
            turns
                .values()
                .any(|t| t.done.lock().map_or(true, |d| d.is_none()))
        })
    }
    pub async fn command(&self, state: Arc<AppState>, command: Command) -> Result<Value, String> {
        match command {
            Command::List => Ok(
                json!({"conversations":state.db.list_conversations().map_err(|e|e.to_string())?}),
            ),
            Command::Load { conversation_id } => {
                valid_id(&conversation_id)?;
                let mut doc = state
                    .db
                    .get_conversation(&conversation_id)
                    .map_err(|e| e.to_string())?
                    .ok_or("Conversation not found")?;
                if doc.to_string().len() > MAX_DOCUMENT {
                    return Err("Conversation exceeds the native 8 MiB view budget".into());
                }
                let busy = self
                    .turns
                    .lock()
                    .map_err(|_| "Chat state unavailable")?
                    .values()
                    .any(|t| {
                        t.conversation_id == conversation_id
                            && t.done.lock().is_ok_and(|d| d.is_none())
                    });
                if !busy && let Some(messages) = doc["messages"].as_array_mut() {
                    for m in messages {
                        if m["streaming"] == true {
                            m["streaming"] = json!(false);
                            m["error"] = json!(
                                "Generation was interrupted before it was saved as complete."
                            );
                            m["nativeStatus"] = json!("interrupted");
                        }
                    }
                }
                Ok(json!({"conversation":doc}))
            }
            Command::Poll { stream_id } => {
                let mut turns = self.turns.lock().map_err(|_| "Chat state unavailable")?;
                let turn = turns
                    .get_mut(&stream_id)
                    .ok_or("Chat stream is no longer available")?;
                let mut events = Vec::new();
                let mut bytes = 0;
                while bytes < 64 * 1024 {
                    match turn.receiver.try_recv() {
                        Ok(value) => {
                            bytes += value.to_string().len();
                            events.push(value);
                        }
                        Err(_) => break,
                    }
                }
                // Terminal state is reported only after draining every delta.
                let done = if turn.receiver.is_empty() {
                    turn.done
                        .lock()
                        .map_err(|_| "Chat state unavailable")?
                        .clone()
                } else {
                    None
                };
                let result = json!({"events":events,"done":done});
                if done.is_some() {
                    turns.remove(&stream_id);
                }
                Ok(result)
            }
            Command::Cancel { stream_id } => {
                let turns = self.turns.lock().map_err(|_| "Chat state unavailable")?;
                let turn = turns
                    .get(&stream_id)
                    .ok_or("Chat stream is no longer available")?;
                let _ = turn.cancel.send(true);
                Ok(json!({"accepted":true}))
            }
            Command::Send {
                conversation_id,
                port,
                pid,
                text,
            } => {
                if text.trim().is_empty() || text.len() > MAX_TEXT {
                    return Err("Enter a message no larger than 128 KiB".into());
                }
                let runners = state.supervisor.list().await;
                let runner = runners
                    .iter()
                    .find(|r| r.port == port && r.pid == pid && r.status == "ok")
                    .ok_or(
                        "The selected runner changed or is not ready. Refresh and select it again.",
                    )?;
                let model = runner
                    .model
                    .as_deref()
                    .ok_or("This endpoint does not serve chat")?;
                let mut doc = if let Some(id) = conversation_id {
                    valid_id(&id)?;
                    state
                        .db
                        .get_conversation(&id)
                        .map_err(|e| e.to_string())?
                        .ok_or("Conversation not found")?
                } else {
                    new_conversation(model, &text)
                };
                let input = prepare_turn(&mut doc, model, &text)?;
                let id = doc["id"]
                    .as_str()
                    .ok_or("Invalid conversation identity")?
                    .to_owned();
                let stream_id = uuid::Uuid::new_v4().to_string();
                let (sender, receiver) = mpsc::channel(32);
                let (cancel, cancelled) = watch::channel(false);
                let done = Arc::new(Mutex::new(None));
                {
                    let mut turns = self.turns.lock().map_err(|_| "Chat state unavailable")?;
                    // Completed but abandoned receipts are bounded too. Never
                    // evict an active stream or its queued content implicitly.
                    let active = turns
                        .values()
                        .filter(|t| t.done.lock().map_or(true, |d| d.is_none()))
                        .count();
                    if turns.len() >= 32
                        || active >= 4
                        || turns.values().any(|t| t.conversation_id == id)
                    {
                        return Err(
                            "Finish receiving existing chats before starting another response"
                                .into(),
                        );
                    }
                    state.db.put_conversation(&doc).map_err(|e| e.to_string())?;
                    turns.insert(
                        stream_id.clone(),
                        Turn {
                            conversation_id: id.clone(),
                            receiver,
                            cancel,
                            done: done.clone(),
                        },
                    );
                }
                let receipt = json!({"stream_id":stream_id,"conversation":doc});
                tokio::spawn(async move {
                    run(state, port, doc, input, sender, cancelled, done).await;
                });
                Ok(receipt)
            }
        }
    }
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn valid_id(id: &str) -> Result<(), String> {
    if id.is_empty()
        || id.len() > 128
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        Err("Invalid conversation identity".into())
    } else {
        Ok(())
    }
}
fn new_conversation(model: &str, text: &str) -> Value {
    json!({"id":uuid::Uuid::new_v4().to_string(),"title":text.chars().take(72).collect::<String>(),"model":model,"systemPrompt":"","messages":[],"createdAt":now(),"updatedAt":now(),"toolSelection":[],"webSearchEnabled":false,
        "params":{"temperature":null,"topP":null,"topK":null,"minP":null,"maxTokens":8192,"frequencyPenalty":null,"presencePenalty":null,"repeatPenalty":null,"seed":null,"stop":[],"thinking":true,"reasoningEffort":"","preserveThinking":false}})
}
fn prepare_turn(doc: &mut Value, model: &str, text: &str) -> Result<Value, String> {
    if doc.to_string().len()
        > MAX_DOCUMENT - MAX_OUTPUT - crate::studio_stream::MAX_EVENT - MAX_TEXT - 64 * 1024
    {
        return Err("This conversation exceeds the native text-chat context budget. Start a new chat; history was not truncated.".into());
    }
    let messages = doc["messages"]
        .as_array()
        .ok_or("Invalid conversation messages")?;
    // Text-only first slice: never silently drop branches, attachments, tools,
    // compaction or compare lanes from an existing web Studio conversation.
    if doc.get("serverCompaction").is_some_and(|v| !v.is_null())
        || doc
            .get("compareModels")
            .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        || doc["webSearchEnabled"] == true
        || doc["toolSelection"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
        || doc.get("compaction").is_some_and(|v| !v.is_null())
    {
        return Err(
            "This conversation needs a Studio workflow not connected in the native app yet".into(),
        );
    }
    // Capability-aware reasoning controls and stop sequences are not connected
    // yet. Do not silently reinterpret a web conversation's explicit settings.
    let params = &doc["params"];
    if doc["summary"].as_str().is_some_and(|s| !s.is_empty()) {
        return Err("Compacted conversations require web Studio for now".into());
    }
    if params["preserveThinking"] == true
        || params["thinking"] == false
        || params["reasoningEffort"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
        || params["thinkingBudget"].as_u64().is_some_and(|n| n > 0)
        || params["stop"].as_array().is_some_and(|a| !a.is_empty())
    {
        return Err(
            "This conversation uses reasoning or stop controls not connected in native Studio yet"
                .into(),
        );
    }
    let mut input = Vec::new();
    let mut previous: Option<&str> = None;
    for m in messages {
        if m.get("group").is_some_and(|v| !v.is_null())
            || m.get("lane").is_some_and(|v| !v.is_null())
            || m["toolCalls"].as_array().is_some_and(|a| !a.is_empty())
            || m["webSearches"].as_array().is_some_and(|a| !a.is_empty())
        {
            return Err(
                "Compare and tool conversations are not connected in native Studio yet".into(),
            );
        }
        if m.get("parentId").is_some() && m["parentId"].as_str() != previous {
            return Err("Branched conversations require web Studio for now".into());
        }
        previous = m["id"].as_str();
        if m["role"] != "user" && m["role"] != "assistant" && m["role"] != "system" {
            return Err("Unsupported conversation role".into());
        }
        let parts = m["content"].as_array().ok_or("Invalid message content")?;
        if parts.iter().any(|p| p["type"] != "text") {
            return Err("Attachments are not connected in native Studio yet".into());
        }
        let body = parts
            .iter()
            .map(|p| p["text"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        if !body.is_empty() {
            input.push(json!({"role":m["role"],"content":body}));
        }
    }
    if doc.get("leafId").is_some() && doc["leafId"].as_str() != previous {
        return Err("This conversation has an inactive branch; open it in web Studio".into());
    }
    let parent = previous.map(str::to_owned);
    let user_id = uuid::Uuid::new_v4().to_string();
    let assistant_id = uuid::Uuid::new_v4().to_string();
    input.push(json!({"role":"user","content":text}));
    let mut request = json!({"model":model,"input":input,"instructions":doc["systemPrompt"].as_str().unwrap_or(""),"stream":true,"store":false,"tools":[],"max_output_tokens":params["maxTokens"].as_u64().filter(|n|*n>0).unwrap_or(8192),"truncation":"disabled"});
    for (saved, wire) in [
        ("temperature", "temperature"),
        ("topP", "top_p"),
        ("topK", "top_k"),
        ("minP", "min_p"),
        ("frequencyPenalty", "frequency_penalty"),
        ("presencePenalty", "presence_penalty"),
        ("repeatPenalty", "repeat_penalty"),
        ("seed", "seed"),
    ] {
        if let Some(value) = params.get(saved).filter(|v| !v.is_null()) {
            request[wire] = value.clone();
        }
    }
    let messages = doc["messages"]
        .as_array_mut()
        .ok_or("Invalid conversation messages")?;
    messages.push(json!({"id":user_id,"parentId":parent,"role":"user","content":[{"type":"text","text":text}],"createdAt":now()}));
    messages.push(json!({"id":assistant_id,"parentId":user_id,"role":"assistant","model":model,"content":[{"type":"text","text":""}],"reasoning":"","streaming":true,"nativeStatus":"in_progress","createdAt":now()}));
    doc["leafId"] = json!(assistant_id);
    doc["model"] = json!(model);
    doc["updatedAt"] = json!(now());
    Ok(request)
}
async fn run(
    state: Arc<AppState>,
    port: u16,
    mut doc: Value,
    input: Value,
    sender: mpsc::Sender<Value>,
    mut cancel: watch::Receiver<bool>,
    done: Arc<Mutex<Option<Value>>>,
) {
    let mut response = ResponseText::default();
    let work = async {
        let result =
            crate::routes::studio_responses(state.clone(), port, input.to_string().into()).await;
        if !result.status().is_success() {
            return Err(format!(
                "Runner rejected the request (HTTP {})",
                result.status()
            ));
        }
        if !result
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .is_some_and(|s| s.starts_with("text/event-stream"))
        {
            return Err("Runner did not return a Responses event stream".into());
        }
        consume(result.into_body(), &mut response, &sender).await
    };
    let result = tokio::select! {biased; _=cancel.changed()=>Err("Generation cancelled".into()), result=work=>result};
    let cancelled = *cancel.borrow();
    let status = if cancelled {
        "cancelled"
    } else if result.is_err() {
        "failed"
    } else {
        response.status.as_deref().unwrap_or("failed")
    };
    let error = result.err().or(response.error.clone());
    let Some(assistant) = doc["messages"].as_array_mut().and_then(|m| m.last_mut()) else {
        if let Ok(mut done) = done.lock() {
            *done = Some(
                json!({"conversation_id":doc["id"],"status":"failed","save_error":"Conversation lost its response slot"}),
            );
        }
        return;
    };
    assistant["content"] = json!([{"type":"text","text":response.text()}]);
    assistant["reasoning"] = json!(response.reasoning());
    assistant["streaming"] = json!(false);
    assistant["nativeStatus"] = json!(status);
    if cancelled {
        assistant["stopped"] = json!(true);
    } else if let Some(error) = &error {
        assistant["error"] = json!(error);
    }
    if status == "incomplete"
        && response
            .response
            .as_ref()
            .is_some_and(|r| r["incomplete_details"]["reason"] == "max_output_tokens")
    {
        assistant["incomplete"] = json!("length");
    }
    if let Some(usage) = response.usage {
        assistant["usage"] = json!({"promptTokens":usage["input_tokens"],"completionTokens":usage["output_tokens"],"reasoningTokens":usage["output_tokens_details"]["reasoning_tokens"]});
    }
    if let Some(raw) = response.response {
        assistant["response"] = raw;
    }
    doc["updatedAt"] = json!(now());
    let saved = state.db.put_conversation(&doc).map_err(|e| e.to_string());
    if let Ok(mut done) = done.lock() {
        *done = Some(
            json!({"conversation_id":doc["id"],"status":status,"error":error,"save_error":saved.err()}),
        );
    }
}
async fn consume(
    body: axum::body::Body,
    response: &mut ResponseText,
    sender: &mpsc::Sender<Value>,
) -> Result<(), String> {
    let mut body = body.into_data_stream();
    let mut sse = Sse::default();
    loop {
        let chunk = tokio::time::timeout(Duration::from_secs(120), body.next())
            .await
            .map_err(|_| "Runner stream stalled for 120 seconds")?;
        let ended = chunk.is_none();
        let frames = match chunk {
            Some(Ok(bytes)) => sse.push(&bytes)?,
            Some(Err(_)) => return Err("Runner disconnected while streaming".into()),
            None => sse.finish()?,
        };
        for frame in frames {
            if let Some(event) = response.apply(&frame)? {
                let delta = event["delta"].as_str().ok_or("Invalid transcript delta")?;
                let mut start = 0;
                while start < delta.len() {
                    let mut end = (start + MAX_DELTA).min(delta.len());
                    while !delta.is_char_boundary(end) {
                        end -= 1;
                    }
                    let mut part = event.clone();
                    part["delta"] = json!(&delta[start..end]);
                    tokio::time::timeout(Duration::from_secs(30), sender.send(part))
                        .await
                        .map_err(|_| "Transcript stopped accepting updates")?
                        .map_err(|_| "Transcript receiver closed")?;
                    start = end;
                }
            }
        }
        if response.status.is_some() {
            return Ok(());
        }
        if ended {
            return Err("Runner stream ended without a terminal Responses event".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_shape_and_followup_reuse_browser_conversation_contract() {
        let mut doc = new_conversation("test", "hello");
        let request = prepare_turn(&mut doc, "test", "hello").unwrap();
        assert_eq!(request["input"][0]["content"], "hello");
        assert_eq!(request["truncation"], "disabled");
        assert_eq!(doc["messages"][1]["parentId"], doc["messages"][0]["id"]);
        doc["messages"][1]["content"][0]["text"] = json!("world");
        let next = prepare_turn(&mut doc, "test", "again").unwrap();
        assert_eq!(next["input"].as_array().unwrap().len(), 3);
        doc["messages"][0]["content"][0]["type"] = json!("image");
        assert!(prepare_turn(&mut doc, "test", "no").is_err());
    }
    #[tokio::test]
    async fn truncated_stream_is_failure_and_queue_preserves_unicode() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut r = ResponseText::default();
        let body = axum::body::Body::from(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"å😀\"}\n\n",
        );
        assert!(
            consume(body, &mut r, &tx)
                .await
                .unwrap_err()
                .contains("terminal")
        );
        assert_eq!(rx.try_recv().unwrap()["delta"], "å😀");
        assert_eq!(r.text(), "å😀");
    }
    #[test]
    fn commands_reject_keys_urls_and_oversized_identity() {
        assert!(
            serde_json::from_str::<Command>(
                r#"{"kind":"send","port":12345,"pid":42,"text":"hi","api_key":"no"}"#
            )
            .is_err()
        );
        assert!(valid_id("../other").is_err());
        assert!(valid_id(&"a".repeat(129)).is_err());
    }
    #[test]
    fn shared_store_roundtrip_preserves_settings_and_unknown_fields() {
        let store = crate::store::Store::open(&std::path::PathBuf::from(":memory:")).unwrap();
        let mut doc = new_conversation("test", "hello");
        doc["futureField"] = json!({"keep":true});
        doc["params"]["temperature"] = json!(0.25);
        doc["params"]["maxTokens"] = json!(37);
        let request = prepare_turn(&mut doc, "test", "hello").unwrap();
        assert_eq!(request["temperature"], 0.25);
        assert_eq!(request["max_output_tokens"], 37);
        assert!(request.get("top_p").is_none());
        store.put_conversation(&doc).unwrap();
        let mut saved = store
            .get_conversation(doc["id"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(saved, doc);
        assert_eq!(store.list_conversations().unwrap().len(), 1);
        saved["params"]["thinking"] = json!(false);
        assert!(prepare_turn(&mut saved, "test", "again").is_err());
    }
    #[tokio::test]
    async fn terminal_is_delivered_after_all_queued_deltas() {
        let (tx, rx) = mpsc::channel(32);
        let mut response = ResponseText::default();
        let body = axum::body::Body::from(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello tail\"}]}]}}\n\n",
        );
        consume(body, &mut response, &tx).await.unwrap();
        assert_eq!(rx.len(), 1);
        assert_eq!(response.text(), "hello tail");
        assert_eq!(response.status.as_deref(), Some("completed"));
    }
    #[tokio::test]
    async fn closed_receiver_fails_without_losing_collected_text() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let mut response = ResponseText::default();
        let body = axum::body::Body::from(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"held\"}\n\n",
        );
        assert!(
            consume(body, &mut response, &tx)
                .await
                .unwrap_err()
                .contains("receiver closed")
        );
        assert_eq!(response.text(), "held");
    }
}
