//! Typed, bounded native commands. Mutations run on Tokio, never on Swift's
//! snapshot worker. Only one lifecycle mutation is admitted at a time, so two
//! starts cannot both spend the same memory grant. No URLs, paths, eviction
//! plans or arbitrary privileged routes are accepted from the UI. API-key
//! replacement is input-only; saved credentials never enter a receipt.
use axum::{body::Body, http::Request};
use paddock_manager::routes::AppState;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

const HISTORY: usize = 32;

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Command {
    Prepare {
        model: String,
        artifact: String,
    },
    Poll {
        id: u64,
    },
    Create {
        model: String,
        artifact: String,
        port: Option<u16>,
        max_ctx: Option<usize>,
        max_batch: Option<usize>,
        #[serde(default)]
        changes: Vec<paddock_manager::native_endpoints::Change>,
        #[serde(default)]
        allow_network: bool,
        tools: Option<paddock_manager::native_endpoints::CreationTools>,
    },
    Start {
        port: u16,
        revision: String,
        #[serde(default)]
        allow_network: bool,
    },
    Stop {
        port: u16,
        pid: u32,
    },
    Edit {
        port: u16,
        revision: String,
        pid: Option<u32>,
        changes: Vec<paddock_manager::native_endpoints::Change>,
        apply: paddock_manager::native_endpoints::Apply,
        #[serde(default)]
        allow_network: bool,
    },
    Remove {
        port: u16,
        revision: String,
    },
}

impl Command {
    fn port(&self) -> Option<u16> {
        match self {
            Self::Poll { .. } | Self::Prepare { .. } => None,
            Self::Create { port, .. } => *port,
            Self::Start { port, .. }
            | Self::Stop { port, .. }
            | Self::Edit { port, .. }
            | Self::Remove { port, .. } => Some(*port),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.port().is_some_and(|port| port < 1024) {
            return Err("Choose an unprivileged model port between 1024 and 65535.".into());
        }
        match self {
            Self::Create {
                model,
                artifact,
                max_ctx,
                max_batch,
                ..
            } => {
                if model.is_empty()
                    || model.len() > 200
                    || artifact.is_empty()
                    || artifact.len() > 100
                {
                    return Err("Select a model and an artifact from the catalog.".into());
                }
                if max_ctx.is_some_and(|n| !(256..=1_048_576).contains(&n))
                    || max_batch.is_some_and(|n| !(1..=256).contains(&n))
                {
                    return Err(
                        "The context or concurrency limit is outside the supported range.".into(),
                    );
                }
            }
            Self::Start { revision, .. }
            | Self::Edit { revision, .. }
            | Self::Remove { revision, .. }
                if revision.is_empty() || revision.len() > 128 =>
            {
                return Err("Refresh this endpoint before changing it.".into());
            }
            Self::Stop { pid: 0, .. } => {
                return Err("Refresh this runner before stopping it.".into());
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Serialize)]
pub(crate) struct Job {
    pub id: u64,
    pub port: Option<u16>,
    pub action: &'static str,
    pub state: &'static str,
    pub message: String,
}

#[derive(Default)]
pub(crate) struct Jobs {
    next_id: u64,
    rows: Vec<Job>,
}

impl Jobs {
    pub fn get(&self, id: u64) -> Result<Job, String> {
        self.rows.iter().find(|j|j.id == id).cloned().ok_or_else(||"The model operation receipt is no longer available. Refresh the endpoint before taking another action.".into())
    }
    pub fn snapshot(&self) -> Vec<Job> {
        self.rows.clone()
    }
    pub fn active(&self) -> bool {
        self.rows.iter().any(|j| j.state == "running")
    }

    fn reserve(&mut self, command: &Command) -> Result<Job, String> {
        command.validate()?;
        if self.active() {
            return Err("A model operation is already in progress. Wait for it to finish.".into());
        }
        self.next_id += 1;
        let (action, message) = match command {
            Command::Poll { .. } | Command::Prepare { .. } => {
                return Err("A read cannot create a model operation.".into());
            }
            Command::Create { .. } => (
                "create",
                "Checking the installed artifact and available memory, then loading the model…",
            ),
            Command::Start { .. } => (
                "start",
                "Checking the saved endpoint and available memory, then loading the model…",
            ),
            Command::Stop { .. } => ("stop", "Draining requests and stopping the model…"),
            Command::Edit {
                apply: paddock_manager::native_endpoints::Apply::Defer,
                ..
            } => (
                "save",
                "Checking the reviewed configuration and saving for the next start…",
            ),
            Command::Edit { .. } => (
                "restart",
                "Checking configuration and memory, then restarting on the same port…",
            ),
            Command::Remove { .. } => (
                "remove",
                "Checking that the endpoint is stopped, then removing its configuration…",
            ),
        };
        let job = Job {
            id: self.next_id,
            port: command.port(),
            action,
            state: "running",
            message: message.into(),
        };
        if self.rows.len() == HISTORY {
            self.rows.remove(0);
        }
        self.rows.push(job.clone());
        Ok(job)
    }

    fn finish(&mut self, id: u64, result: Result<(Option<u16>, String), String>) {
        if let Some(job) = self.rows.iter_mut().find(|j| j.id == id) {
            (job.state, job.message) = match result {
                Ok((port, message)) => {
                    job.port = port;
                    ("succeeded", message)
                }
                Err(message) => ("failed", message),
            };
        }
    }
}

pub(crate) fn submit(
    runtime: &tokio::runtime::Runtime,
    jobs: Arc<Mutex<Jobs>>,
    state: Arc<AppState>,
    command: Command,
) -> Result<Job, String> {
    let job = jobs
        .lock()
        .map_err(|_| "Model job state unavailable")?
        .reserve(&command)?;
    let id = job.id;
    let task = runtime.spawn(execute_with_progress(
        state,
        command,
        Some((id, jobs.clone())),
    ));
    // The monitor also settles panics. Neither cancellation of a Swift Task nor
    // closing a window aborts a start half-way through owning a child process.
    runtime.spawn(async move {
        let result = task.await.unwrap_or_else(|_| Err("The model operation failed unexpectedly. Refresh the endpoint state before retrying.".into()));
        if let Ok(mut jobs) = jobs.lock() { jobs.finish(id, result); }
    });
    Ok(job)
}

#[cfg(test)]
async fn execute(state: Arc<AppState>, command: Command) -> Result<(Option<u16>, String), String> {
    execute_with_progress(state, command, None).await
}

async fn execute_with_progress(
    state: Arc<AppState>,
    command: Command,
    progress: Option<(u64, Arc<Mutex<Jobs>>)>,
) -> Result<(Option<u16>, String), String> {
    let requested_port = command.port();
    // Non-create commands always carry a concrete existing endpoint identity.
    let port = requested_port.unwrap_or_default();
    let (method, path, body) = match command {
        Command::Poll { .. } | Command::Prepare { .. } => {
            return Err("A read cannot execute a model operation.".into());
        }
        Command::Edit {
            revision,
            pid,
            changes,
            apply,
            allow_network,
            ..
        } => {
            return paddock_manager::native_endpoints::edit(
                state,
                port,
                revision,
                pid,
                changes,
                apply,
                allow_network,
            )
            .await
            .map(|message| (Some(port), message));
        }
        Command::Remove { revision, .. } => {
            return paddock_manager::native_endpoints::remove(state, port, &revision)
                .await
                .map(|message| (Some(port), message));
        }
        Command::Create {
            model,
            artifact,
            max_ctx,
            max_batch,
            mut changes,
            allow_network,
            tools,
            ..
        } => {
            // Compatibility for older typed callers. Duplicate fields are still
            // rejected by Edit's allowlist; there is only one authority per value.
            if let Some(value) = max_ctx {
                changes.push(paddock_manager::native_endpoints::Change::MaxCtx(Some(
                    value,
                )));
            }
            if let Some(value) = max_batch {
                changes.push(paddock_manager::native_endpoints::Change::MaxBatch(Some(
                    value,
                )));
            }
            let port = paddock_manager::native_endpoints::create_config(
                &state,
                &model,
                &artifact,
                requested_port,
                &changes,
                allow_network,
                tools,
            )
            .await?;
            if let Some((id, jobs)) = &progress
                && let Ok(mut jobs) = jobs.lock()
                && let Some(job) = jobs.rows.iter_mut().find(|j| j.id == *id)
            {
                job.port = Some(port);
            }
            // Start the final file verbatim. Passing through SpawnSpec here
            // would drop Advanced fields such as sampling and speech settings.
            (
                "POST",
                format!("/api/servers/{port}/start"),
                serde_json::json!({}),
            )
        }
        Command::Start {
            revision,
            allow_network,
            ..
        } => {
            if state.supervisor.config_file_hash(port).as_deref() != Some(&revision) {
                return Err(
                    "This endpoint's configuration changed. Refresh and review it before starting."
                        .into(),
                );
            }
            let spec = state
                .supervisor
                .spec_from_config_file(&state.supervisor.server_config_path(port))
                .map_err(
                    |_| "The saved endpoint configuration cannot be read. Check its config file.",
                )?;
            if !spec.host.is_some_and(|host| host.is_loopback())
                && (!allow_network || !spec.api_key.as_deref().is_some_and(|k| k.len() >= 16))
            {
                return Err("Starting a network endpoint requires explicit confirmation and a saved API key of at least 16 characters. Nothing was started.".into());
            }
            check_local_port(port)?;
            (
                "POST",
                format!("/api/servers/{port}/start"),
                serde_json::json!({}),
            )
        }
        Command::Stop { pid, .. } => {
            if !state
                .supervisor
                .list()
                .await
                .iter()
                .any(|r| r.port == port && r.pid == pid)
            {
                return Err("The runner on this port changed or already stopped. Refresh before trying again.".into());
            }
            (
                "DELETE",
                format!("/api/runners/{port}?timeout_ms=30000"),
                serde_json::Value::Null,
            )
        }
    };
    let response = paddock_manager::routes::router(state.clone())
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .map_err(|e| e.to_string())?,
        )
        .await
        .map_err(|e| e.to_string())?;
    let status = response.status();
    if status.is_success() {
        if method == "DELETE" {
            return Ok((
                requested_port,
                "Model stopped. Its endpoint configuration is saved.".into(),
            ));
        }
        // Decode only the elected port. The privileged RunnerView (including
        // keys) never enters a native receipt, even for automatically assigned ports.
        #[derive(Deserialize)]
        struct Started {
            port: u16,
        }
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await
            .map_err(|_| "The model operation completed, but its status could not be read. Check Models.")?;
        let started: Started = serde_json::from_slice(&bytes).map_err(
            |_| "The model operation completed, but its address could not be read. Check Models.",
        )?;
        return Ok((
            Some(started.port),
            "Model is ready. Its local endpoint is available to your applications.".into(),
        ));
    }
    // Raw runner log tails may contain user tool configuration/credentials.
    // Keep diagnostics on disk until there is a separately audited log view.
    // Admission and validation get explicit safe categories, never raw TOML.
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap_or_default();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
    Err(match status.as_u16() {
        507 => "There is not enough available model memory for this configuration. Stop an endpoint or reduce context/concurrency; no endpoints were automatically evicted.".into(),
        400 | 502 => startup_message(body["error"]["reason"].as_str()).into(),
        409 => "The endpoint conflicts with an existing operation or configuration. Refresh before retrying.".into(),
        _ => format!("The endpoint operation was refused (status {}). Refresh its state and check the local runner log.", status.as_u16()),
    })
}

/// Whitelisted categories only: raw error messages can contain paths, keys or
/// tool configuration. Never claim a log/config exists for a pre-launch failure.
fn startup_message(reason: Option<&str>) -> &'static str {
    match reason {
        Some("unsupported_configuration") => {
            "These settings are not supported by this model. Use the recommended settings under Advanced."
        }
        Some("model_missing" | "download_failed") => {
            "The model or its required companion files are unavailable. Check Downloads before starting again."
        }
        Some("runner_missing") => {
            "The bundled model runner is unavailable. Rebuild or reinstall Paddock."
        }
        Some("port_unavailable" | "already_configured") => {
            "This local address is unavailable. Choose Automatic under Advanced, or start the saved model from Models."
        }
        Some("launch_io") => {
            "Paddock could not prepare or launch the model. Check disk space and file permissions."
        }
        Some("startup_exit") => {
            "The model runner exited while loading. Open this model in Models to see its logs."
        }
        Some("startup_timeout") => {
            "The model took too long to become ready. Open this model in Models to see its logs."
        }
        _ => "The model could not start. Check its status in Models for details.",
    }
}

fn check_local_port(port: u16) -> Result<(), String> {
    // Diagnostic preflight, not a reservation: the runner's bind is still the
    // final authority. Never stop or take over an unrelated process on a port.
    std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
        .map(drop)
        .map_err(|_| {
            format!("Port {port} is unavailable. Choose Automatic under Advanced or a different fixed port.")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_port_is_pending_until_the_supervisor_elects_it() {
        let command: Command = serde_json::from_str(
            r#"{"kind":"create","model":"kb-whisper-large","artifact":"f16"}"#,
        )
        .unwrap();
        command.validate().unwrap();
        let mut jobs = Jobs::default();
        let job = jobs.reserve(&command).unwrap();
        assert_eq!(job.port, None);
        jobs.finish(job.id, Ok((Some(12345), "Ready".into())));
        assert_eq!(jobs.get(job.id).unwrap().port, Some(12345));
        assert_eq!(jobs.get(job.id).unwrap().state, "succeeded");
        for port in [0, 80, 1023] {
            let command: Command = serde_json::from_value(serde_json::json!({"kind":"create","model":"kb-whisper-large","artifact":"f16","port":port})).unwrap();
            assert!(command.validate().is_err());
        }
    }

    #[test]
    fn startup_errors_never_invent_files_or_echo_privileged_details() {
        assert!(startup_message(Some("unsupported_configuration")).contains("settings"));
        for reason in [
            None,
            Some("unsupported_configuration"),
            Some("runner_missing"),
            Some("private-key /private/model/path"),
        ] {
            let message = startup_message(reason);
            assert!(!message.contains("private-key") && !message.contains("/private"));
            assert!(!message.contains("retained") && !message.contains("logs/runner-"));
        }
    }
    #[test]
    fn commands_are_strict_and_job_storage_is_bounded() {
        for raw in [
            r#"{"kind":"create","model":"qwen","artifact":"mlx","port":12}"#,
            r#"{"kind":"stop","port":12345,"pid":0}"#,
        ] {
            assert!(
                serde_json::from_str::<Command>(raw)
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
        assert!(
            serde_json::from_str::<Command>(
                r#"{"kind":"stop","port":12345,"pid":42,"url":"http://evil"}"#
            )
            .is_err()
        );
        let command = Command::Stop {
            port: 12345,
            pid: 42,
        };
        let mut jobs = Jobs::default();
        for _ in 0..100 {
            let job = jobs.reserve(&command).unwrap();
            assert!(jobs.reserve(&command).is_err());
            jobs.finish(job.id, Ok((Some(12345), "Done".into())));
        }
        assert_eq!(jobs.snapshot().len(), HISTORY);
        assert!(!jobs.active());
    }
    #[test]
    fn polling_is_read_only_and_unknown_receipts_fail() {
        let mut jobs = Jobs::default();
        let job = jobs
            .reserve(&Command::Remove {
                port: 12345,
                revision: "reviewed".into(),
            })
            .unwrap();
        for _ in 0..100 {
            assert_eq!(jobs.get(job.id).unwrap().id, job.id);
        }
        assert!(jobs.get(job.id + 1).is_err());
        assert_eq!(jobs.next_id, 1);
        assert_eq!(jobs.snapshot().len(), 1);
        assert!(jobs.active());
        assert!(jobs.reserve(&Command::Poll { id: job.id }).is_err());
        jobs.finish(job.id, Ok((Some(12345), "Saved".into())));
        assert_eq!(jobs.get(job.id).unwrap().state, "succeeded");
    }
    #[tokio::test]
    async fn validation_never_downloads_or_creates_an_endpoint() {
        let state = Arc::new(AppState::for_tests());
        let err = execute(
            state.clone(),
            Command::Create {
                model: "not-a-catalog-model".into(),
                artifact: "mlx".into(),
                port: Some(12345),
                max_ctx: None,
                max_batch: None,
                changes: vec![],
                allow_network: false,
                tools: None,
            },
        )
        .await
        .unwrap_err();
        assert!(err.contains("catalog model"));
        // Speech exports now reach the same installation checks as chat.
        // This never loads a runner or fetches weights from the network.
        for (model, artifact) in [
            ("qwen3-asr-1.7b", "q8"),
            ("granite-speech-4.1-2b", "q8"),
            ("granite-speech-4.1-2b-plus", "q8"),
            ("kb-whisper-large", "f16"),
            ("nb-whisper-large", "f16"),
            ("roest-v3-whisper-1.5b", "f16"),
        ] {
            let err = execute(
                state.clone(),
                Command::Create {
                    model: model.into(),
                    artifact: artifact.into(),
                    port: None,
                    max_ctx: None,
                    max_batch: None,
                    changes: vec![],
                    allow_network: false,
                    tools: None,
                },
            )
            .await
            .unwrap_err();
            assert!(err.contains("downloaded weights"), "{model}: {err}");
        }
        assert!(
            execute(
                state.clone(),
                Command::Start {
                    port: 12345,
                    revision: "stale".into(),
                    allow_network: false,
                }
            )
            .await
            .unwrap_err()
            .contains("changed")
        );
        assert!(
            execute(
                state,
                Command::Stop {
                    port: 12345,
                    pid: 42
                }
            )
            .await
            .unwrap_err()
            .contains("changed")
        );
    }
}
