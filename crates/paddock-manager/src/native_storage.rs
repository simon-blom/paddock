//! Read-only artifact inventory. Enumerates the compiled registry, not arbitrary
//! user directories, and stats files off the native management queue.
use crate::routes::AppState;
use serde_json::{Value, json};
use std::sync::Arc;

pub async fn inventory(state: Arc<AppState>) -> Result<Value, String> {
    let configured = state.supervisor.configured().await;
    tokio::task::spawn_blocking(move || {
        let registry = &state.registry;
        let mut artifacts = Vec::new();
        for model in &registry.catalog().models {
            for artifact in &model.artifacts {
                let mut bytes = 0_u64;
                let mut present = 0_usize;
                let mut unreadable = false;
                for file in &artifact.files {
                    match std::fs::metadata(registry.models_dir().join(&file.dest)) {
                        Ok(metadata) if metadata.is_file() => {
                            bytes = bytes.saturating_add(metadata.len());
                            present += 1;
                        }
                        Err(error) if error.kind() != std::io::ErrorKind::NotFound => { unreadable = true; }
                        _ => {}
                    }
                }
                if present == 0 && !unreadable { continue; }
                let references: Vec<_> = configured.iter().filter(|endpoint| endpoint.model.as_deref() == Some(&model.id)
                    && endpoint.artifact.as_deref() == Some(&artifact.id)).map(|endpoint| endpoint.port).collect();
                artifacts.push(json!({"id":format!("{}:{}",model.id,artifact.id),"model":model.display,"artifact":artifact.label,
                    "bytes": if unreadable { None } else { Some(bytes) },"present_files":present,"total_files":artifact.files.len(),
                    "path":artifact.entry_path(registry.models_dir()),"configured_ports":references}));
            }
        }
        Ok(json!({"artifacts":artifacts}))
    }).await.map_err(|_| "Could not inspect installed model files.")?
}
