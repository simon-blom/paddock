//! Native management operations. A closed command set, not an arbitrary HTTP
//! proxy. Requests execute on the runtime; polling never blocks the ABI queue.
use axum::{body::Body, http::Request};
use futures_util::FutureExt;
use paddock_manager::routes::AppState;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tower::ServiceExt;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Command {
    Usage {
        from: i64,
        to: i64,
        port: Option<u16>,
    },
    Activity {
        port: Option<u16>,
        before: Option<i64>,
    },
    Cache {},
    Storage {},
    Benchmark {
        port: u16,
        pid: u32,
        concurrency: usize,
        long: bool,
    },
    BenchmarkHistory {},
    ExportBenchmark {
        id: String,
        path: std::path::PathBuf,
    },
    Profiles {
        model: String,
        artifact: String,
    },
    SaveProfile {
        port: u16,
        revision: String,
        name: String,
    },
    RemoveProfile {
        id: String,
    },
    ClientInfo {
        port: u16,
        pid: u32,
    },
    ExportCredentials {
        port: u16,
        pid: u32,
        path: std::path::PathBuf,
    },
    Backup {
        path: std::path::PathBuf,
    },
    Poll {
        id: String,
    },
    Close {
        id: String,
    },
}

struct Job {
    benchmark: bool,
    touched: Instant,
    result: Option<Result<Value, String>>,
    task: Option<tokio::task::AbortHandle>,
}
impl Drop for Job {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}
#[derive(Default, Clone)]
pub(crate) struct Sessions(Arc<Mutex<HashMap<String, Job>>>);
impl Sessions {
    pub fn close_all(&self) {
        if let Ok(mut jobs) = self.0.lock() {
            jobs.clear();
        }
    }

    pub fn execute(&self, state: Arc<AppState>, command: Command) -> Result<Value, String> {
        let mut jobs = self.0.lock().map_err(|_| "Management jobs unavailable")?;
        jobs.retain(|_, job| job.touched.elapsed() < Duration::from_secs(60));
        match command {
            Command::Close { id } => {
                jobs.remove(&id);
                Ok(json!({"state":"closed"}))
            }
            Command::Poll { id } => {
                let job = jobs
                    .get_mut(&id)
                    .ok_or("Request status expired. Check the result before retrying.")?;
                job.touched = Instant::now();
                Ok(match &job.result {
                    None => json!({"id":id,"state":"running"}),
                    Some(Ok(value)) => {
                        json!({"id":id,"state":"complete","payload":value.to_string()})
                    }
                    Some(Err(message)) => json!({"id":id,"state":"failed","message":message}),
                })
            }
            command => {
                let benchmark = matches!(command, Command::Benchmark { .. });
                if benchmark
                    && jobs
                        .values()
                        .any(|job| job.benchmark && job.result.is_none())
                {
                    return Err("A benchmark is already running.".into());
                }
                if matches!(
                    command,
                    Command::Usage { .. } | Command::Activity { .. } | Command::Cache {}
                ) {
                    read_path(&command)?;
                }
                if jobs.len() >= 8 {
                    return Err("Too many management requests; try again shortly.".into());
                }
                let id = uuid::Uuid::new_v4().to_string();
                jobs.insert(
                    id.clone(),
                    Job {
                        benchmark,
                        touched: Instant::now(),
                        result: None,
                        task: None,
                    },
                );
                let shared = self.clone();
                let ticket = id.clone();
                let task = tokio::spawn(async move {
                    let result = tokio::time::timeout(
                        Duration::from_secs(if benchmark { 900 } else { 300 }),
                        std::panic::AssertUnwindSafe(run(state, command)).catch_unwind(),
                    )
                    .await
                    .map_err(|_| "Management request timed out".to_string())
                    .and_then(|result| result.map_err(|_| "Management worker failed".to_string()))
                    .and_then(|result| result);
                    if let Ok(mut jobs) = shared.0.lock()
                        && let Some(job) = jobs.get_mut(&ticket)
                    {
                        job.result = Some(result);
                    }
                });
                jobs.get_mut(&id).expect("inserted job").task = Some(task.abort_handle());
                Ok(json!({"id":id,"state":"running"}))
            }
        }
    }
}

async fn run(state: Arc<AppState>, command: Command) -> Result<Value, String> {
    match command {
        Command::Storage {} => paddock_manager::native_storage::inventory(state).await,
        Command::Benchmark {
            port,
            pid,
            concurrency,
            long,
        } => paddock_manager::native_bench::run(state, port, pid, concurrency, long).await,
        Command::BenchmarkHistory {} => Ok(
            json!({"reports":state.db.native_benchmarks().map_err(|_| "Could not read benchmarks.")?}),
        ),
        Command::ExportBenchmark { id, path } => tokio::task::spawn_blocking(move || {
            paddock_manager::native_exports::benchmark(&state.db, &id, &path)
        })
        .await
        .map_err(|_| "Benchmark export failed")?,
        Command::Profiles { model, artifact } => Ok(
            json!({"profiles":state.db.model_profiles(&model, &artifact).map_err(|_| "Could not read model profiles.")?}),
        ),
        Command::SaveProfile {
            port,
            revision,
            name,
        } => {
            let (raw, current) = state.supervisor.read_config_file(port)?;
            if current != revision {
                return Err("The instance changed. Reload before saving a profile.".into());
            }
            let spec = state.supervisor.spec_from_config_text(&raw)?;
            let artifact = spec
                .artifact
                .ok_or("This instance does not identify a catalog artifact.")?;
            let projection =
                paddock_manager::native_endpoints::projection(&state.supervisor, port)?;
            if projection["revision"].as_str() != Some(&revision) {
                return Err("The instance changed. Reload before saving a profile.".into());
            }
            let saved = state
                .db
                .save_model_profile(&name, &spec.model, &artifact, &projection["settings"])
                .map_err(|e| e.to_string())?;
            Ok(json!({"profiles":[saved]}))
        }
        Command::RemoveProfile { id } => {
            state
                .db
                .remove_model_profile(&id)
                .map_err(|_| "Could not remove the profile.")?;
            Ok(json!({"profiles":[]}))
        }
        Command::ClientInfo { port, pid } => {
            paddock_manager::native_exports::client_info(state, port, pid).await
        }
        Command::ExportCredentials { port, pid, path } => {
            paddock_manager::native_exports::credential_file(state, port, pid, &path).await
        }
        Command::Backup { path } => tokio::task::spawn_blocking(move || {
            paddock_manager::native_exports::backup(&state.db, &path)
        })
        .await
        .map_err(|_| "Backup worker failed")?,
        command => read(state, read_path(&command)?).await,
    }
}

fn read_path(command: &Command) -> Result<String, String> {
    match command {
        Command::Usage { from, to, port } => {
            if *from < 0 || *from >= *to || to.saturating_sub(*from) > 366 * 86_400_000 {
                return Err("Choose a usage range of up to one year.".into());
            }
            Ok(format!(
                "/api/usage/history?from={from}&to={to}{}",
                port_query(*port)
            ))
        }
        Command::Activity { port, before } => {
            if before.is_some_and(|v| v < 0) {
                return Err("Invalid activity cursor.".into());
            }
            Ok(format!(
                "/api/activity?limit=200{}{}",
                port_query(*port),
                before.map(|t| format!("&before={t}")).unwrap_or_default()
            ))
        }
        Command::Cache {} => Ok("/api/cache".into()),
        _ => Err("Not a read command".into()),
    }
}

fn port_query(port: Option<u16>) -> String {
    port.map(|p| format!("&port={p}")).unwrap_or_default()
}

async fn read(state: Arc<AppState>, path: String) -> Result<Value, String> {
    let response = paddock_manager::routes::router(state)
        .oneshot(
            Request::builder()
                .uri(path)
                .body(Body::empty())
                .map_err(|_| "Invalid request")?,
        )
        .await
        .map_err(|_| "Management service unavailable")?;
    if !response.status().is_success() {
        return Err(format!("Management request failed ({})", response.status()));
    }
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .map_err(|_| "Management response exceeds its size limit")?;
    serde_json::from_slice(&bytes).map_err(|_| "Invalid management response".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn native_reads_use_real_routes_and_release_receipts() {
        let state = Arc::new(AppState::for_tests());
        let sessions = Sessions::default();
        for command in [
            Command::Usage {
                from: 0,
                to: 10_000,
                port: None,
            },
            Command::Activity {
                port: None,
                before: None,
            },
            Command::Cache {},
        ] {
            let started = sessions.execute(state.clone(), command).unwrap();
            let id = started["id"].as_str().unwrap();
            let result = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    let result = sessions
                        .execute(state.clone(), Command::Poll { id: id.into() })
                        .unwrap();
                    if result["state"] != "running" {
                        break result;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            assert_eq!(result["state"], "complete", "{result}");
            let payload: Value = serde_json::from_str(result["payload"].as_str().unwrap()).unwrap();
            assert!(payload.is_object());
            sessions
                .execute(state.clone(), Command::Close { id: id.into() })
                .unwrap();
            assert!(
                sessions
                    .execute(state.clone(), Command::Poll { id: id.into() })
                    .is_err()
            );
        }
    }
    #[test]
    fn closed_protocol_and_bounded_ranges() {
        assert!(
            serde_json::from_value::<Command>(json!({"kind":"get","url":"https://example.com"}))
                .is_err()
        );
        assert!(serde_json::from_value::<Command>(json!({"kind":"cache","key":"secret"})).is_err());
        assert!(
            read_path(&Command::Usage {
                from: 9,
                to: 8,
                port: None
            })
            .is_err()
        );
        assert!(
            read_path(&Command::Usage {
                from: 0,
                to: i64::MAX,
                port: None
            })
            .is_err()
        );
        assert_eq!(
            read_path(&Command::Activity {
                port: Some(11540),
                before: Some(1234)
            })
            .unwrap(),
            "/api/activity?limit=200&port=11540&before=1234"
        );
    }
}
