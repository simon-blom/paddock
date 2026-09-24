//! Per-user, per-device cross-process load serialization. Model loaders still
//! admit against fresh physical memory: a lock cannot reserve other apps' RAM.
pub(crate) fn load_lock(device: &str, gpu: usize) -> Result<std::fs::File, String> {
    if !matches!(device, "metal" | "cuda") {
        return Err("Device load admission requires CUDA or Metal".into());
    }
    #[cfg(unix)]
    let root = paddock_admin::runtime_dir();
    #[cfg(windows)]
    let root = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .ok_or("model_load_failed: LOCALAPPDATA is unavailable")?
        .join("Paddock")
        .join("runtime");
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    lock_path(
        &root.join(format!("residency-{device}-{gpu}.lock")),
        std::time::Duration::from_secs(120),
    )
}

fn lock_path(
    path: &std::path::Path,
    timeout: std::time::Duration,
) -> Result<std::fs::File, String> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path).map_err(|e| e.to_string())?;
    let start = std::time::Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) if start.elapsed() < timeout => {
                std::thread::sleep(std::time::Duration::from_millis(25))
            }
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(
                    "model_busy: another runner is loading on this device; retry shortly".into(),
                );
            }
            Err(error) => {
                return Err(format!(
                    "model_load_failed: cannot coordinate device loading: {error}"
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_handles_serialize_and_release_without_stale_locks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("device.lock");
        let owner = lock_path(&path, std::time::Duration::ZERO).unwrap();
        assert!(
            lock_path(&path, std::time::Duration::ZERO)
                .unwrap_err()
                .starts_with("model_busy:")
        );
        // File presence is never interpreted as ownership (crashed owners
        // release the OS lock without having to remove this persistent inode).
        drop(owner);
        let next = lock_path(&path, std::time::Duration::ZERO).unwrap();
        let other_device =
            lock_path(&dir.path().join("other.lock"), std::time::Duration::ZERO).unwrap();
        drop((next, other_device));
    }
}
