//! One control-plane owner per data directory, across native and web hosts.
//! An OS lock is released on crash; a persistent lock file is not a stale lock.
use std::{
    fs::{File, OpenOptions},
    io,
    path::Path,
};

pub struct ManagerOwnership(File);

impl ManagerOwnership {
    pub fn acquire(data: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(data)?;
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(data.join("manager.lock"))?;
        file.try_lock().map_err(|e| {
            io::Error::other(format!(
                "Paddock's data directory is already managed or cannot be locked ({}): {e}. Close the other Paddock manager before retrying.", data.display()))
        })?;
        Ok(Self(file))
    }
}

impl Drop for ManagerOwnership {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ownership_is_exclusive_and_released_without_deleting_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let owner = ManagerOwnership::acquire(dir.path()).unwrap();
        assert!(ManagerOwnership::acquire(dir.path()).is_err());
        drop(owner);
        assert!(dir.path().join("manager.lock").exists());
        assert!(ManagerOwnership::acquire(dir.path()).is_ok());
    }
}
