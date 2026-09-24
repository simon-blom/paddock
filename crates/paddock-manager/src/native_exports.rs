//! Explicit native exports. No secrets in replies or debug formatting. Files
//! are owner-only, atomically published, and never overwrite existing paths.
use crate::{routes::AppState, store::Store};
use serde_json::{Value, json};
use std::{io::Write, path::Path, sync::Arc};

fn destination(path: &Path) -> Result<&Path, String> {
    if !path.is_absolute() || path.file_name().is_none() {
        return Err("Choose an absolute export filename.".into());
    }
    let parent = path.parent().ok_or("Choose an export folder.")?;
    if !parent.is_dir() {
        return Err("The export folder no longer exists.".into());
    }
    if path.symlink_metadata().is_ok() {
        return Err("A file already exists there. Choose a new filename.".into());
    }
    Ok(parent)
}

fn publish(
    path: &Path,
    write: impl FnOnce(&mut std::fs::File) -> Result<(), String>,
) -> Result<u64, String> {
    let mut file = tempfile::NamedTempFile::new_in(destination(path)?)
        .map_err(|_| "Could not create the private export file.")?;
    write(file.as_file_mut())?;
    file.as_file()
        .sync_all()
        .map_err(|_| "Could not finish the export.")?;
    let bytes = file
        .as_file()
        .metadata()
        .map_err(|_| "Could not read export size.")?
        .len();
    file.persist_noclobber(path)
        .map_err(|_| "Could not publish the export; an existing file was not replaced.")?;
    Ok(bytes)
}

pub fn backup(store: &Store, path: &Path) -> Result<Value, String> {
    destination(path)?;
    let snapshot = store
        .snapshot_to_temp()
        .map_err(|_| "Could not snapshot conversations.")?;
    // Guard removes the credential-bearing intermediate on every exit path.
    let snapshot = tempfile::TempPath::try_from_path(snapshot)
        .map_err(|_| "Could not secure the backup snapshot.")?;
    crate::api::sanitize_export_file(&snapshot)?;
    let bytes = publish(path, |output| {
        let mut input = std::fs::File::open(&snapshot).map_err(|_| "Could not read the backup.")?;
        std::io::copy(&mut input, output).map_err(|_| "Could not write the backup.")?;
        Ok(())
    })?;
    Ok(json!({"bytes":bytes}))
}

async fn live_client(
    state: &AppState,
    port: u16,
    pid: u32,
) -> Result<(String, Option<String>), String> {
    let runner = state
        .supervisor
        .list()
        .await
        .into_iter()
        .find(|r| r.port == port && r.pid == pid && r.status != "unreachable")
        .ok_or("The selected instance changed or stopped. Select its running instance again.")?;
    let model = runner
        .model
        .ok_or("Select a running chat model for client setup.")?;
    let key = state.supervisor.runner_key_checked(port, pid).await?;
    Ok((model, key))
}

pub async fn client_info(state: Arc<AppState>, port: u16, pid: u32) -> Result<Value, String> {
    let (model, key) = live_client(&state, port, pid).await?;
    Ok(
        json!({"base_url":format!("http://127.0.0.1:{port}/v1"),"model":model,"has_key":key.is_some()}),
    )
}

pub async fn credential_file(
    state: Arc<AppState>,
    port: u16,
    pid: u32,
    path: &Path,
) -> Result<Value, String> {
    let _guard = crate::connectors::MUTATIONS.lock().await;
    let (model, key) = live_client(&state, port, pid).await?;
    let key = key.ok_or("This instance does not require an API key.")?;
    let content = format!(
        "export PADDOCK_BASE_URL={}\nexport PADDOCK_MODEL={}\nexport PADDOCK_API_KEY={}\n",
        shell_quote(&format!("http://127.0.0.1:{port}/v1")),
        shell_quote(&model),
        shell_quote(&key)
    );
    let bytes = publish(path, |file| {
        file.write_all(content.as_bytes())
            .map_err(|_| "Could not write credentials.".into())
    })?;
    Ok(json!({"bytes":bytes}))
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

pub fn benchmark(store: &Store, id: &str, path: &Path) -> Result<Value, String> {
    let report = store
        .native_benchmark(id)
        .map_err(|_| "Could not load benchmark history.")?
        .ok_or("This benchmark is no longer in history.")?;
    let bytes = publish(path, |file| {
        serde_json::to_writer_pretty(file, &report)
            .map_err(|_| "Could not write the benchmark report.".into())
    })?;
    Ok(json!({"bytes":bytes}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exports_are_private_exclusive_and_credential_free() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(&root.path().join("source.db")).unwrap();
        store
            .put_conversation(&json!({"id":"chat","title":"Keep","model":"local","messages":[]}))
            .unwrap();
        store
            .put_attachment(
                "image",
                Some("chat"),
                "image/png",
                "image.png",
                None,
                None,
                b"image-bytes",
            )
            .unwrap();
        store.create_cloud_endpoint(&json!({"name":"Cloud","kind":"openai-compat","baseUrl":"https://example.invalid/v1","apiKey":"DO-NOT-EXPORT-SECRET"})).unwrap();
        let connector = store.create_connector(&json!({"label":"MCP","url":"https://example.invalid/mcp","headers":{"Authorization":"CONNECTOR-SECRET"}})).unwrap();
        store
            .set_connector_oauth(
                connector["id"].as_str().unwrap(),
                "{\"access_token\":\"OAUTH-SECRET\"}",
            )
            .unwrap();
        let dest = root.path().join("backup.db");
        backup(&store, &dest).unwrap();
        let bytes = std::fs::read(&dest).unwrap();
        assert!(!bytes.windows(20).any(|v| v == b"DO-NOT-EXPORT-SECRET"));
        for secret in [b"CONNECTOR-SECRET".as_slice(), b"OAUTH-SECRET"] {
            assert!(!bytes.windows(secret.len()).any(|v| v == secret));
        }
        let exported = Store::open(&dest).unwrap();
        assert!(exported.get_conversation("chat").unwrap().is_some());
        assert_eq!(
            exported.get_attachment("image").unwrap().unwrap().1,
            b"image-bytes"
        );
        assert!(backup(&store, &dest).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&dest).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
    #[test]
    fn no_shell_injection_or_destination_overwrite() {
        assert_eq!(shell_quote("a'b$(touch nope)"), "'a'\\''b$(touch nope)'");
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("key.env");
        publish(&path, |f| f.write_all(b"secret").map_err(|e| e.to_string())).unwrap();
        assert!(publish(&path, |_| Ok(())).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"secret");
    }
}
