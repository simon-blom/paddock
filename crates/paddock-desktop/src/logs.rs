//! Read-only native log subscriptions. File I/O runs on blocking workers; ABI
//! calls only drain bounded queues. The UI cannot supply a path or a URL.
use paddock_manager::{log_tail::Tail, routes::AppState};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Command {
    Open { port: u16 },
    Poll { id: String },
    Close { id: String },
}
struct Feed {
    chunks: Vec<String>,
    bytes: usize,
    state: &'static str,
    touched: Instant,
    task: Option<tokio::task::AbortHandle>,
}
impl Drop for Feed {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}
#[derive(Default, Clone)]
pub(crate) struct Sessions(Arc<Mutex<HashMap<String, Feed>>>);
impl Sessions {
    pub fn close_all(&self) {
        if let Ok(mut feeds) = self.0.lock() {
            feeds.clear();
        }
    }
    pub fn execute(&self, state: Arc<AppState>, command: Command) -> Result<Value, String> {
        let mut feeds = self
            .0
            .lock()
            .map_err(|_| "Log subscriptions unavailable.")?;
        feeds.retain(|_, feed| feed.touched.elapsed() < Duration::from_secs(30));
        match command {
            Command::Close { id } => {
                feeds.remove(&id);
                Ok(json!({"state":"closed","text":""}))
            }
            Command::Poll { id } => {
                let feed = feeds
                    .get_mut(&id)
                    .ok_or("Log subscription expired. Reconnecting is safe.")?;
                feed.touched = Instant::now();
                let text = feed.chunks.drain(..).collect::<String>();
                feed.bytes = 0;
                Ok(json!({"id":id,"state":feed.state,"text":text}))
            }
            Command::Open { port } => {
                if port < 1024 {
                    return Err("Select a model endpoint.".into());
                }
                if feeds.len() >= 2 {
                    return Err("Close an existing log viewer first.".into());
                }
                let id = uuid::Uuid::new_v4().to_string();
                feeds.insert(
                    id.clone(),
                    Feed {
                        chunks: Vec::new(),
                        bytes: 0,
                        state: "connecting",
                        touched: Instant::now(),
                        task: None,
                    },
                );
                let shared = self.clone();
                let ticket = id.clone();
                let task = tokio::spawn(async move {
                    let mut tail = Tail::new(state.supervisor.log_path(port), None);
                    let mut initial = true;
                    loop {
                        let room = if let Ok(mut feeds) = shared.0.lock() {
                            if feeds
                                .get(&ticket)
                                .is_some_and(|f| f.touched.elapsed() >= Duration::from_secs(30))
                            {
                                feeds.remove(&ticket);
                                return;
                            }
                            let Some(feed) = feeds.get(&ticket) else {
                                return;
                            };
                            feed.bytes < 256 * 1024
                        } else {
                            return;
                        };
                        if room {
                            let supervisor = state.supervisor.clone();
                            let live_key = supervisor.runner_key(port).await;
                            let result = tokio::task::spawn_blocking(move || {
                                let text = if initial {
                                    tail.history(300).unwrap_or_default()
                                } else {
                                    tail.advance()
                                };
                                let available = tail.available();
                                let text = paddock_manager::native_endpoints::safe_log_text(
                                    &supervisor,
                                    port,
                                    &text,
                                    live_key.as_deref(),
                                );
                                (tail, text, available)
                            })
                            .await;
                            let Ok((next, text, available)) = result else {
                                if let Ok(mut feeds) = shared.0.lock()
                                    && let Some(feed) = feeds.get_mut(&ticket)
                                {
                                    feed.state = "failed";
                                }
                                return;
                            };
                            tail = next;
                            // A log may not exist until the first launch. Its
                            // first appearance still gets bounded history.
                            if available {
                                initial = false;
                            }
                            if let Ok(mut feeds) = shared.0.lock() {
                                let Some(feed) = feeds.get_mut(&ticket) else {
                                    return;
                                };
                                feed.state = if available { "live" } else { "waiting" };
                                if !text.is_empty() {
                                    feed.bytes += text.len();
                                    feed.chunks.push(text);
                                }
                            } else {
                                return;
                            }
                        }
                        tokio::time::sleep(Duration::from_millis(300)).await;
                    }
                });
                if let Some(feed) = feeds.get_mut(&id) {
                    feed.task = Some(task.abort_handle());
                }
                Ok(json!({"id":id,"state":"connecting","text":""}))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn subscriptions_are_bounded_read_only_and_close_idempotently() {
        let sessions = Sessions::default();
        let state = Arc::new(AppState::for_tests());
        for raw in [
            r#"{"kind":"open","port":13494,"path":"/private"}"#,
            r#"{"kind":"erase","port":13494}"#,
        ] {
            assert!(serde_json::from_str::<Command>(raw).is_err());
        }
        let first = sessions
            .execute(state.clone(), Command::Open { port: 13494 })
            .unwrap();
        sessions
            .execute(state.clone(), Command::Open { port: 13495 })
            .unwrap();
        assert!(
            sessions
                .execute(state.clone(), Command::Open { port: 13496 })
                .is_err()
        );
        let id = first["id"].as_str().unwrap().to_owned();
        assert_eq!(
            sessions
                .execute(state.clone(), Command::Poll { id: id.clone() })
                .unwrap()["text"],
            ""
        );
        sessions
            .execute(state.clone(), Command::Close { id: id.clone() })
            .unwrap();
        sessions
            .execute(state.clone(), Command::Close { id: id.clone() })
            .unwrap();
        assert!(sessions.execute(state, Command::Poll { id }).is_err());
        sessions.close_all();
    }
}
