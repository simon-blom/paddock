//! Durable download intent and bounded admission shared by both frontends.
use super::*;

pub(super) fn digest(files: &[CatalogFile]) -> String {
    hex(&Sha256::digest(
        serde_json::to_vec(files).expect("catalog files serialize"),
    ))
}

/// Sparse logical length is not disk allocation. Count blocks already allocated
/// to the partial file, so resuming does not reserve the same disk space twice.
pub(super) fn additional_bytes(root: &Path, file: &CatalogFile) -> u64 {
    let dest = root.join(&file.dest);
    if std::fs::metadata(&dest).is_ok_and(|m| m.is_file() && m.len() == file.size) {
        return 0;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(md) = std::fs::metadata(part_path(&dest))
            && md.is_file()
            && md.len() == file.size
        {
            return file.size.saturating_sub(md.blocks().saturating_mul(512));
        }
    }
    file.size
}

impl PullJob {
    pub(super) fn record(&self) -> serde_json::Value {
        let mut record = self.snapshot();
        record["selection_digest"] = self.selection_digest.clone().into();
        // A process restart must not replay privileged queued spawn/eviction
        // plans. Recovery offers the bytes, then requires a fresh Start click.
        record
            .as_object_mut()
            .expect("a download snapshot is a JSON object")
            .remove("start");
        record
    }
}

impl Registry {
    /// Shared preflight used by native review and authoritative admission.
    pub fn download_disk_need(&self, model: &str, artifacts: &[String]) -> Result<u64, DlError> {
        Ok(self
            .pull_files(model, Some(artifacts))?
            .iter()
            .map(|f| additional_bytes(&self.models_dir, f))
            .sum())
    }
    pub fn with_store(mut self, store: Arc<crate::store::Store>) -> Result<Self, DlError> {
        for record in store
            .load_downloads()
            .map_err(|e| DlError::Http(e.to_string()))?
        {
            let Some(id) = record["id"].as_str() else {
                continue;
            };
            let Some(model) = record["model"].as_str() else {
                continue;
            };
            let artifacts: Option<Vec<String>> =
                serde_json::from_value(record["artifacts"].clone()).ok();
            let files = self
                .pull_files(model, artifacts.as_deref())
                .unwrap_or_default();
            let mut status: PullStatus =
                serde_json::from_value(record["status"].clone()).unwrap_or(PullStatus::Cancelled);
            if matches!(status, PullStatus::Running) {
                status = PullStatus::Cancelled;
            }
            if matches!(status, PullStatus::Done)
                && (files.is_empty()
                    || files.iter().any(|f| {
                        !std::fs::metadata(self.models_dir.join(&f.dest))
                            .is_ok_and(|md| md.is_file() && md.len() == f.size)
                    }))
            {
                status = PullStatus::Cancelled;
            }
            let downloaded = files
                .iter()
                .map(|f| {
                    let dest = self.models_dir.join(&f.dest);
                    if std::fs::metadata(&dest).is_ok_and(|m| m.len() == f.size) {
                        f.size
                    } else if std::fs::metadata(part_path(&dest)).is_ok_and(|m| m.len() == f.size) {
                        load_state(&state_path(&dest), f.size.div_ceil(SEGMENT) as usize)
                            .iter()
                            .enumerate()
                            .filter(|(_, done)| **done)
                            .map(|(i, _)| seg_len(i, f.size))
                            .sum()
                    } else {
                        0
                    }
                })
                .sum();
            let job = Arc::new(PullJob {
                id: id.into(),
                model_id: model.into(),
                display: record["display"].as_str().unwrap_or(model).into(),
                artifacts,
                downloaded: Arc::new(AtomicU64::new(downloaded)),
                total: record["total"].as_u64().unwrap_or(0),
                created_ms: record["created_ms"].as_u64().unwrap_or(0),
                status: std::sync::Mutex::new(status),
                cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                follow: std::sync::Mutex::new(None),
                follow_state: std::sync::Mutex::new(None),
                files,
                selection_digest: record["selection_digest"].as_str().unwrap_or("").into(),
                phase: std::sync::Mutex::new(
                    "Recovered after restart; resume to verify and finish".into(),
                ),
                stage: std::sync::Mutex::new("paused"),
            });
            self.jobs
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(id.into(), job);
        }
        self.store = Some(store);
        Ok(self)
    }

    pub(super) fn pull_files(
        &self,
        model_id: &str,
        artifacts: Option<&[String]>,
    ) -> Result<Vec<CatalogFile>, DlError> {
        let model = self
            .catalog_of(model_id)
            .ok_or_else(|| DlError::Http("Unknown model".into()))?;
        let selected = match artifacts {
            Some(ids) => ids
                .iter()
                .map(|id| {
                    model
                        .artifact(id)
                        .ok_or_else(|| DlError::Http("Unknown artifact".into()))
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => model.default_bundle_for_backend(&self.backend, self.cc),
        };
        if selected.is_empty()
            || selected
                .iter()
                .any(|a| !a.runtime.supports_backend(&self.backend))
        {
            return Err(DlError::Http("Select compatible model files".into()));
        }
        let mut seen = std::collections::HashMap::new();
        let mut files = Vec::new();
        for f in selected.iter().flat_map(|a| &a.files) {
            if f.size == 0
                || f.sha256.len() != 64
                || !f.sha256.bytes().all(|b| b.is_ascii_hexdigit())
                || Path::new(&f.dest)
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(DlError::Http("Invalid catalog file contract".into()));
            }
            if let Some(prior) = seen.insert(f.dest.clone(), (f.size, f.sha256.clone())) {
                if prior != (f.size, f.sha256.clone()) {
                    return Err(DlError::Http("Conflicting catalog files".into()));
                }
            } else {
                files.push(f.clone());
            }
        }
        if files.is_empty() {
            return Err(DlError::Http("The selected artifact has no files".into()));
        }
        Ok(files)
    }

    /// Pending writers settle before the runtime can release directory ownership.
    /// No runner is involved. The on-disk intent remains resumable after a crash.
    pub async fn pause_downloads(&self) {
        for job in self.jobs() {
            self.cancel_pull(&job.id);
        }
        while self.jobs().iter().any(|j| {
            matches!(
                *j.status
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                PullStatus::Running
            )
        }) {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}
