//! Private native connection commands. Network/Keychain work is admitted here
//! and runs outside the serial ABI queue. A checked draft stays only in Rust;
//! saving names that receipt, never round-trips a stored key through Swift.
use paddock_manager::{
    connections::{Draft, Pick, Prepared},
    routes::AppState,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(test)]
#[path = "connections_tests.rs"]
mod tests;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Command {
    List {},
    Unlock {
        id: String,
        revision: u64,
    },
    Check {
        draft: Draft,
    },
    Poll {
        id: String,
    },
    Cancel {
        id: String,
    },
    Save {
        id: String,
        models: Vec<Pick>,
    },
    Models {
        id: String,
        revision: u64,
        models: Vec<Pick>,
    },
    Remove {
        id: String,
        revision: u64,
    },
}
#[derive(Clone, Serialize)]
pub(crate) struct Job {
    id: String,
    status: &'static str,
    message: String,
    models: Vec<Value>,
    endpoint: Option<Value>,
}
struct Slot {
    job: Job,
    prepared: Option<Arc<Prepared>>,
    created: Instant,
    task: Option<tokio::task::AbortHandle>,
}
#[derive(Clone, Default)]
pub(crate) struct Sessions(Arc<Mutex<HashMap<String, Slot>>>);

impl Sessions {
    fn reserve(&self, status: &'static str) -> Result<Job, String> {
        let mut slots = self
            .0
            .lock()
            .map_err(|_| "Connection checks unavailable.")?;
        slots.retain(|_, s| {
            let keep = s.job.status == "saving" || s.created.elapsed() < Duration::from_secs(300);
            if !keep && let Some(task) = &s.task {
                task.abort();
            }
            keep
        });
        if slots
            .values()
            .filter(|s| matches!(s.job.status, "checking" | "saving"))
            .count()
            >= 2
        {
            return Err("Wait for the current connection operations to finish.".into());
        }
        if slots.len() >= 8 {
            let oldest = slots
                .iter()
                .filter(|(_, s)| !matches!(s.job.status, "checking" | "saving" | "checked"))
                .min_by_key(|(_, s)| s.created)
                .map(|(id, _)| id.clone());
            if let Some(id) = oldest {
                slots.remove(&id);
            } else {
                return Err("Close an existing connection review first.".into());
            }
        }
        let job = Job {
            id: uuid::Uuid::new_v4().to_string(),
            status,
            message: String::new(),
            models: vec![],
            endpoint: None,
        };
        slots.insert(
            job.id.clone(),
            Slot {
                job: job.clone(),
                prepared: None,
                created: Instant::now(),
                task: None,
            },
        );
        // Expiry runs even with no further UI calls, so an abandoned check
        // cannot retain its credential indefinitely.
        let sessions = self.clone();
        let id = job.id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(300)).await;
            if let Ok(mut slots) = sessions.0.lock()
                && slots.get(&id).is_some_and(|s| s.job.status != "saving")
                && let Some(s) = slots.remove(&id)
                && let Some(t) = s.task
            {
                t.abort();
            }
        });
        Ok(job)
    }
    fn set_task(&self, id: &str, task: tokio::task::AbortHandle) {
        if let Ok(mut slots) = self.0.lock()
            && let Some(slot) = slots.get_mut(id)
        {
            slot.task = Some(task);
        }
    }
    fn finish(&self, id: &str, result: Result<Option<Value>, String>) {
        if let Ok(mut slots) = self.0.lock()
            && let Some(slot) = slots.get_mut(id)
        {
            if slot.job.status == "cancelled" {
                return;
            }
            slot.prepared = None;
            match result {
                Ok(endpoint) => {
                    slot.job.status = "saved";
                    slot.job.message = "Saved.".into();
                    slot.job.endpoint = endpoint;
                }
                Err(message) => {
                    slot.job.status = "failed";
                    slot.job.message = message;
                }
            }
        }
    }
    pub fn saving(&self) -> bool {
        self.0
            .lock()
            .is_ok_and(|slots| slots.values().any(|s| s.job.status == "saving"))
    }
    pub fn cancel_checks(&self) {
        if let Ok(mut slots) = self.0.lock() {
            slots.retain(|_, slot| {
                if slot.job.status == "saving" {
                    true
                } else {
                    if let Some(t) = &slot.task {
                        t.abort();
                    }
                    false
                }
            });
        }
    }
    pub fn execute(&self, state: Arc<AppState>, command: Command) -> Result<Value, String> {
        match command {
            Command::Unlock { id, revision } => {
                let job = self.reserve("checking")?;
                let sessions = self.clone();
                let receipt = job.id.clone();
                let task = tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        state.db.unlock_cloud_connection(&id, revision)
                    })
                    .await;
                    if let Ok(mut slots) = sessions.0.lock()
                        && let Some(slot) = slots.get_mut(&receipt)
                    {
                        slot.job.status = if matches!(result, Ok(Ok(()))) {
                            "unlocked"
                        } else {
                            "failed"
                        };
                        slot.job.message = match result {
                            Ok(Ok(())) => "Cloud access is ready for this app session.".into(),
                            Ok(Err(message)) => message,
                            Err(_) => "Cloud access could not be prepared. Try unlocking the account again.".into(),
                        };
                    }
                });
                self.set_task(&job.id, task.abort_handle());
                Ok(json!({"job": job}))
            }
            Command::List {} => Ok(
                json!({"connections": state.db.list_cloud_endpoints().map_err(|_| "Could not load saved connections.")?}),
            ),
            Command::Check { draft } => {
                let job = self.reserve("checking")?;
                let sessions = self.clone();
                let id = job.id.clone();
                let task = tokio::spawn(async move {
                    let prepared = tokio::task::spawn_blocking(move || {
                        paddock_manager::connections::prepare(&state.db, draft)
                    })
                    .await;
                    let result = async {
                        let prepared = Arc::new(prepared.map_err(|_| {
                            "The connection check failed unexpectedly.".to_string()
                        })??);
                        let models = tokio::time::timeout(
                            Duration::from_secs(20),
                            paddock_manager::connections::probe(&prepared),
                        )
                        .await
                        .map_err(|_| "The connection check timed out.".to_string())??;
                        Ok::<_, String>((prepared, models))
                    }
                    .await;
                    if let Ok(mut slots) = sessions.0.lock()
                        && let Some(slot) = slots.get_mut(&id)
                    {
                        if slot.job.status != "checking" {
                            return;
                        }
                        match result {
                            Ok((prepared, models)) => {
                                slot.prepared = Some(prepared);
                                slot.job.models = models;
                                slot.job.status = "checked";
                                slot.job.message = "Connection checked. No generation was sent; model access is confirmed when you chat.".into();
                            }
                            Err(message) => {
                                slot.job.status = "failed";
                                slot.job.message = message;
                            }
                        }
                    }
                });
                self.set_task(&job.id, task.abort_handle());
                Ok(json!({"job":job}))
            }
            Command::Poll { id } => {
                let slots = self
                    .0
                    .lock()
                    .map_err(|_| "Connection checks unavailable.")?;
                let slot = slots
                    .get(&id)
                    .ok_or("This check expired. Check the connection again.")?;
                Ok(json!({"job":slot.job}))
            }
            Command::Cancel { id } => {
                let mut slots = self
                    .0
                    .lock()
                    .map_err(|_| "Connection checks unavailable.")?;
                if slots
                    .get(&id)
                    .is_some_and(|slot| slot.job.status == "saving")
                {
                    return Err("Wait for the save to finish before closing.".into());
                }
                if let Some(slot) = slots.remove(&id)
                    && let Some(task) = slot.task
                {
                    task.abort();
                }
                Ok(json!({}))
            }
            Command::Save { id, models } => {
                let mut slots = self
                    .0
                    .lock()
                    .map_err(|_| "Connection checks unavailable.")?;
                let slot = slots
                    .get_mut(&id)
                    .ok_or("This check expired. Check the connection again.")?;
                // Duplicate saves return the accepted receipt, not another write.
                if matches!(slot.job.status, "saving" | "saved") {
                    return Ok(json!({"job":slot.job}));
                }
                let prepared = slot
                    .prepared
                    .clone()
                    .filter(|_| slot.job.status == "checked")
                    .ok_or("Check this connection successfully before saving.")?;
                paddock_manager::connections::validate_picks(
                    &models,
                    paddock_manager::connections::is_openrouter(&prepared.draft.base_url),
                )?;
                slot.job.status = "saving";
                let job = slot.job.clone();
                drop(slots);
                let sessions = self.clone();
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        state
                            .db
                            .save_checked_connection(&prepared, &models)
                            .map(Some)
                    })
                    .await
                    .unwrap_or_else(|_| {
                        Err(
                            "The connection save failed unexpectedly. Refresh before retrying."
                                .into(),
                        )
                    });
                    sessions.finish(&id, result);
                });
                Ok(json!({"job":job}))
            }
            Command::Models {
                id,
                revision,
                models,
            } => {
                let job = self.reserve("saving")?;
                let receipt = job.id.clone();
                let sessions = self.clone();
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        state
                            .db
                            .set_connection_models(&id, revision, &models)
                            .map(|_| None)
                    })
                    .await
                    .unwrap_or_else(|_| Err("The model selection could not be saved.".into()));
                    sessions.finish(&receipt, result);
                });
                Ok(json!({"job":job}))
            }
            Command::Remove { id, revision } => {
                let job = self.reserve("saving")?;
                let receipt = job.id.clone();
                let sessions = self.clone();
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        state.db.remove_connection(&id, revision).map(|_| None)
                    })
                    .await
                    .unwrap_or_else(|_| Err("The connection could not be removed.".into()));
                    sessions.finish(&receipt, result);
                });
                Ok(json!({"job":job}))
            }
        }
    }
}
