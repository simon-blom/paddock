//! Log streaming (doc §11.3): llama-swap's proven selector shape - one
//! endpoint that serves the manager's own log, a single runner's, or the
//! merged set, buffered-history-first with an opt-out, optionally following.
//!
//! `GET /api/logs?target=manager|all|<port>&follow=&tail=&history=`
//!
//! The response is a plain chunked text stream (curl-able, CLI-friendly). In
//! `all` mode every line is prefixed with its source (`[manager]`, `[11540]`)
//! - the merged view must never make two runners' lines indistinguishable.
//!   Following is file-tail based: runner logs are plain files the supervisor
//!   opened at spawn (§11.3), and the manager tees its own tracing output to
//!   `logs/manager.log`, so one mechanism covers every source - including a
//!   runner that outlived a manager restart. Rotation/truncation is detected
//!   (length shrank -> restart from 0) and new runner logs appear in `all` mode
//!   as they are created.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};

#[derive(Debug, serde::Deserialize)]
pub struct LogsQuery {
    /// "manager", "all" (default), or a runner port.
    pub target: Option<String>,
    /// Keep the stream open and push new lines as they land.
    pub follow: Option<bool>,
    /// History lines per source before following (default 200).
    pub tail: Option<usize>,
    /// `false` skips the buffered history - new lines only (llama-swap's
    /// no-history opt-out).
    pub history: Option<bool>,
}

enum Target {
    Manager,
    Runner(u16),
    All,
}
use crate::log_tail::Tail;

/// Runner log files currently present: (port, path). The naming contract is
/// the supervisor's `logs/runner-<port>.log` (§11.3).
fn runner_logs(logs_dir: &Path) -> Vec<(u16, PathBuf)> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(logs_dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(port) = name
                .strip_prefix("runner-")
                .and_then(|s| s.strip_suffix(".log"))
                .and_then(|s| s.parse::<u16>().ok())
            {
                out.push((port, e.path()));
            }
        }
    }
    out.sort_by_key(|(p, _)| *p);
    out
}

pub async fn handle(
    State(state): State<Arc<crate::routes::AppState>>,
    Query(q): Query<LogsQuery>,
) -> Response {
    let target = match q.target.as_deref() {
        None | Some("all") => Target::All,
        Some("manager") => Target::Manager,
        Some(s) => match s.parse::<u16>() {
            Ok(p) => Target::Runner(p),
            Err(_) => {
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    axum::Json(serde_json::json!({"error": {"type": "invalid_request_error",
                        "message": format!("target {s:?}: expected \"manager\", \"all\", or a runner port")}})),
                )
                    .into_response();
            }
        },
    };
    let follow = q.follow.unwrap_or(false);
    let tail = q.tail.unwrap_or(200);
    let history = q.history.unwrap_or(true);
    let logs_dir = state.supervisor.logs_dir().to_path_buf();

    // In single-target mode lines go out raw; only the merged view prefixes.
    let mut tails: HashMap<PathBuf, Tail> = HashMap::new();
    let add = |tails: &mut HashMap<PathBuf, Tail>, path: PathBuf, prefix: Option<String>| {
        tails
            .entry(path.clone())
            .or_insert_with(|| Tail::new(path, prefix));
    };
    let merged = matches!(target, Target::All);
    match &target {
        Target::Manager => add(&mut tails, logs_dir.join("manager.log"), None),
        Target::Runner(p) => add(&mut tails, logs_dir.join(format!("runner-{p}.log")), None),
        Target::All => {
            add(
                &mut tails,
                logs_dir.join("manager.log"),
                Some("[manager] ".into()),
            );
            for (port, path) in runner_logs(&logs_dir) {
                add(&mut tails, path, Some(format!("[{port}] ")));
            }
        }
    }
    if !follow && tails.values().all(|t| !t.path().exists()) {
        return (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(serde_json::json!({"error": {"type": "not_found_error",
                "message": "no matching log files yet"}})),
        )
            .into_response();
    }

    let (tx, rx) =
        tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);
    tokio::spawn(async move {
        // History first (unless opted out) - stable order: manager, then ports.
        let mut ordered: Vec<Tail> = tails.into_values().collect();
        tails = HashMap::new();
        ordered.sort_by(|a, b| a.path().cmp(b.path()));
        for mut t in ordered {
            // history=false still calls history(0): it positions the offset
            // at EOF so the follow phase emits only new lines.
            let Ok((t, chunk)) = tokio::task::spawn_blocking(move || {
                let chunk = t.history(if history { tail } else { 0 });
                (t, chunk)
            })
            .await
            else {
                return;
            };
            tails.insert(t.path().to_path_buf(), t);
            if let Some(c) = chunk
                && tx.send(Ok(c.into())).await.is_err()
            {
                return;
            }
        }
        if !follow {
            return;
        }
        // Follow: poll for growth; in merged mode also pick up new runner
        // logs as they appear (a spawn after the stream opened).
        let mut rescan = tokio::time::Instant::now();
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            if merged && rescan.elapsed() >= std::time::Duration::from_secs(2) {
                rescan = tokio::time::Instant::now();
                let root = logs_dir.clone();
                let Ok(found) = tokio::task::spawn_blocking(move || runner_logs(&root)).await
                else {
                    return;
                };
                for (port, path) in found {
                    tails
                        .entry(path.clone())
                        .or_insert_with(|| Tail::new(path, Some(format!("[{port}] "))));
                }
            }
            let Ok((next, chunk)) = tokio::task::spawn_blocking(move || {
                let mut chunk = String::new();
                for t in tails.values_mut() {
                    chunk.push_str(&t.advance());
                }
                (tails, chunk)
            })
            .await
            else {
                return;
            };
            tails = next;
            if !chunk.is_empty() && tx.send(Ok(chunk.into())).await.is_err() {
                return; // client hung up
            }
            if tx.is_closed() {
                return;
            }
        }
    });

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    Response::builder()
        .header(
            axum::http::header::CONTENT_TYPE,
            "text/plain; charset=utf-8",
        )
        .header("x-accel-buffering", "no")
        .body(axum::body::Body::from_stream(stream))
        .unwrap_or_else(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_emits_history_then_only_new_complete_lines() {
        let dir = std::env::temp_dir().join(format!("paddock-logs-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("runner-11540.log");
        std::fs::write(&path, "one\ntwo\nthree\n").unwrap();

        let mut t = Tail::new(path.clone(), Some("[11540] ".into()));
        let h = t.history(2).unwrap();
        assert_eq!(h, "[11540] two\n[11540] three\n");
        assert_eq!(t.advance(), "");

        // a partial line is carried, not emitted
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        write!(f, "fo").unwrap();
        assert_eq!(t.advance(), "");
        writeln!(f, "ur").unwrap();
        assert_eq!(t.advance(), "[11540] four\n");

        // truncation restarts from the top instead of misreading
        std::fs::write(&path, "fresh\n").unwrap();
        assert_eq!(
            t.advance(),
            "[11540] [Log rotated or truncated]\n[11540] fresh\n"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn runner_log_discovery_parses_ports_and_ignores_strangers() {
        let dir = std::env::temp_dir().join(format!("paddock-logs-scan-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("runner-11540.log"), "x").unwrap();
        std::fs::write(dir.join("runner-11541.log"), "x").unwrap();
        std::fs::write(dir.join("manager.log"), "x").unwrap();
        std::fs::write(dir.join("runner-notaport.log"), "x").unwrap();
        let found = runner_logs(&dir);
        assert_eq!(
            found.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            vec![11540, 11541]
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
