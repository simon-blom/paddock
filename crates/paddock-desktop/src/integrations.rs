//! Bounded asynchronous native tools commands. Query cancellation aborts only
//! read-only work; accepted writes settle even if Swift drops its waiter.
use paddock_manager::{integrations::Operation, routes::AppState};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(test)]
#[path = "integrations_tests.rs"]
mod tests;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Command {
    Run { operation: Operation },
    Poll { id: String },
    Cancel { id: String },
}
struct Job {
    mutation: bool,
    created: Instant,
    result: Value,
    task: Option<tokio::task::AbortHandle>,
}
#[derive(Default, Clone)]
pub(crate) struct Sessions(Arc<Mutex<HashMap<String, Job>>>);
impl Sessions {
    pub fn saving(&self) -> bool {
        self.0.lock().is_ok_and(|s| {
            s.values()
                .any(|j| j.mutation && j.result["status"] == "running")
        })
    }
    pub fn cancel_reads(&self) {
        if let Ok(mut jobs) = self.0.lock() {
            jobs.retain(|_, j| {
                if j.mutation && j.result["status"] == "running" {
                    true
                } else {
                    if let Some(t) = &j.task {
                        t.abort();
                    }
                    false
                }
            });
        }
    }
    pub fn execute(
        &self,
        state: Arc<AppState>,
        command: Command,
        origin: Option<String>,
    ) -> Result<Value, String> {
        let mut jobs = self.0.lock().map_err(|_| "Tools operations unavailable.")?;
        match command {
            Command::Poll { id } => {
                Ok(json!({"job":jobs.get(&id).ok_or("This operation expired. Retry it.")?.result}))
            }
            Command::Cancel { id } => {
                if jobs
                    .get(&id)
                    .is_some_and(|j| j.mutation && j.result["status"] == "running")
                {
                    return Err("Wait for the save to finish.".into());
                }
                if let Some(j) = jobs.remove(&id)
                    && let Some(t) = j.task
                {
                    t.abort();
                }
                Ok(json!({}))
            }
            Command::Run { operation } => {
                jobs.retain(|_, j| {
                    j.result["status"] == "running"
                        || j.created.elapsed() < Duration::from_secs(120)
                });
                let mutation = operation.mutation();
                if jobs
                    .values()
                    .filter(|j| j.result["status"] == "running")
                    .count()
                    >= 4
                    || mutation
                        && jobs
                            .values()
                            .any(|j| j.mutation && j.result["status"] == "running")
                {
                    return Err("Wait for the current tools operation to finish.".into());
                }
                if jobs.len() >= 32
                    && let Some(id) = jobs
                        .iter()
                        .filter(|(_, j)| j.result["status"] != "running")
                        .min_by_key(|(_, j)| j.created)
                        .map(|(id, _)| id.clone())
                {
                    jobs.remove(&id);
                }
                let id = uuid::Uuid::new_v4().to_string();
                let result = json!({"id":id,"status":"running","message":"","value":{}});
                jobs.insert(
                    id.clone(),
                    Job {
                        mutation,
                        created: Instant::now(),
                        result: result.clone(),
                        task: None,
                    },
                );
                let sessions = self.clone();
                let ticket = id.clone();
                let task = tokio::spawn(async move {
                    let worker = tokio::spawn(async move {
                        if mutation {
                            paddock_manager::integrations::execute_with_origin(
                                state, operation, origin,
                            )
                            .await
                        } else {
                            tokio::time::timeout(
                                Duration::from_secs(60),
                                paddock_manager::integrations::execute_with_origin(
                                    state, operation, origin,
                                ),
                            )
                            .await
                            .map_err(|_| {
                                "The tools operation timed out. Retry when the server is reachable."
                                    .to_owned()
                            })?
                        }
                    });
                    // Aborting a read must also abort its network future.
                    struct AbortOnDrop(tokio::task::AbortHandle);
                    impl Drop for AbortOnDrop {
                        fn drop(&mut self) {
                            self.0.abort();
                        }
                    }
                    let guard = AbortOnDrop(worker.abort_handle());
                    let outcome = worker
                        .await
                        .unwrap_or_else(|_| Err("The tools operation failed unexpectedly.".into()));
                    drop(guard);
                    if let Ok(mut jobs) = sessions.0.lock()
                        && let Some(job) = jobs.get_mut(&ticket)
                    {
                        job.result = match outcome {
                            Ok(value) if value.to_string().len() <= 2 * 1024 * 1024 => {
                                json!({"id":ticket,"status":"succeeded","message":"","value":value})
                            }
                            Ok(_) => {
                                json!({"id":ticket,"status":"failed","message":"The tools response exceeded its size limit.","value":{}})
                            }
                            Err(message) => {
                                json!({"id":ticket,"status":"failed","message":message,"value":{}})
                            }
                        };
                    }
                });
                if let Some(j) = jobs.get_mut(&id) {
                    j.task = Some(task.abort_handle());
                }
                Ok(json!({"job":result}))
            }
        }
    }
}
