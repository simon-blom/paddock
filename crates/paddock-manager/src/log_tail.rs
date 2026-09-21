//! Bounded file tail shared by web Manager and native diagnostics. Never scan
//! the entire log to show its last lines; carry bytes, not partial UTF-8 strings.
use std::{
    fs::{File, Metadata},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

pub const READ_BYTES: usize = 64 * 1024;
const HISTORY_BYTES: usize = 256 * 1024;
const LINE_BYTES: usize = 16 * 1024;

#[cfg(unix)]
fn identity(md: &Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (md.dev(), md.ino())
}
#[cfg(not(unix))]
fn identity(md: &Metadata) -> (u64, u64) {
    let created = md
        .created()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    (
        created.map_or(0, |t| t.as_secs()),
        created.map_or(0, |t| u64::from(t.subsec_nanos())),
    )
}

pub struct Tail {
    path: PathBuf,
    prefix: Option<String>,
    offset: u64,
    identity: Option<(u64, u64)>,
    anchor: Vec<u8>,
    carry: Vec<u8>,
    dropping: bool,
    available: bool,
}
impl Tail {
    pub fn new(path: PathBuf, prefix: Option<String>) -> Self {
        Self {
            path,
            prefix,
            offset: 0,
            identity: None,
            anchor: Vec::new(),
            carry: Vec::new(),
            dropping: false,
            available: false,
        }
    }
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
    pub fn available(&self) -> bool {
        self.available
    }

    fn open(&mut self) -> Option<(File, Metadata)> {
        self.available = false;
        let file = File::open(&self.path).ok()?;
        let md = file.metadata().ok()?;
        if !md.is_file() {
            return None;
        }
        self.available = true;
        Some((file, md))
    }
    pub fn history(&mut self, n: usize) -> Option<String> {
        let (mut file, md) = self.open()?;
        self.identity = Some(identity(&md));
        self.offset = md.len();
        self.carry.clear();
        self.dropping = false;
        if n == 0 {
            let start = self.offset.saturating_sub(64);
            file.seek(SeekFrom::Start(start)).ok()?;
            self.anchor.clear();
            file.take(self.offset - start)
                .read_to_end(&mut self.anchor)
                .ok()?;
            self.dropping = self.anchor.last().is_some_and(|b| *b != b'\n');
            return None;
        }
        let start = md.len().saturating_sub(HISTORY_BYTES as u64);
        file.seek(SeekFrom::Start(start)).ok()?;
        let mut bytes = Vec::new();
        file.take(md.len() - start).read_to_end(&mut bytes).ok()?;
        self.anchor = bytes[bytes.len().saturating_sub(64)..].to_vec();
        // A bounded history slice can begin halfway through a line. Never
        // present that fragment as a complete event (or a complete credential).
        let cut = if start > 0 {
            bytes
                .iter()
                .position(|b| *b == b'\n')
                .map_or(bytes.len(), |p| p + 1)
        } else {
            0
        };
        self.dropping = start > 0 && cut == bytes.len() && bytes.last() != Some(&b'\n');
        let text = self.consume(&bytes[cut..]);
        let lines: Vec<&str> = text.lines().collect();
        let mut out = String::new();
        if start > 0 {
            self.push(&mut out, "[Earlier history omitted: 256 KiB read limit]");
        }
        for line in &lines[lines.len().saturating_sub(n.min(4000))..] {
            out.push_str(line);
            out.push('\n');
        }
        (!out.is_empty()).then_some(out)
    }
    pub fn advance(&mut self) -> String {
        let Some((mut file, md)) = self.open() else {
            return String::new();
        };
        let id = identity(&md);
        let mut rotated = self.identity.is_some_and(|old| old != id) || md.len() < self.offset;
        // Copy-truncate can regrow between polls without changing the inode.
        // Check a tiny overlap, not the full file or merely its final length.
        if !rotated && !self.anchor.is_empty() {
            let mut overlap = vec![0; self.anchor.len()];
            if file
                .seek(SeekFrom::Start(self.offset - overlap.len() as u64))
                .is_err()
                || file.read_exact(&mut overlap).is_err()
            {
                self.available = false;
                return String::new();
            }
            rotated = overlap != self.anchor;
        }
        let mut out = String::new();
        if rotated {
            self.offset = 0;
            self.carry.clear();
            self.dropping = false;
            self.anchor.clear();
            self.push(&mut out, "[Log rotated or truncated]");
        }
        self.identity = Some(id);
        if md.len() == self.offset {
            return out;
        }
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            self.available = false;
            return out;
        }
        let mut bytes = Vec::new();
        if file
            .take(READ_BYTES as u64)
            .read_to_end(&mut bytes)
            .is_err()
        {
            self.available = false;
            return out;
        }
        self.offset += bytes.len() as u64;
        self.anchor.extend_from_slice(&bytes);
        if self.anchor.len() > 64 {
            self.anchor.drain(..self.anchor.len() - 64);
        }
        out.push_str(&self.consume(&bytes));
        out
    }
    fn consume(&mut self, bytes: &[u8]) -> String {
        let mut out = String::new();
        for &byte in bytes {
            if byte == b'\n' {
                if self.dropping {
                    self.push(
                        &mut out,
                        "[Incomplete or oversized log line omitted: 16 KiB limit]",
                    );
                } else {
                    self.push(
                        &mut out,
                        String::from_utf8_lossy(&self.carry).trim_end_matches('\r'),
                    );
                }
                self.carry.clear();
                self.dropping = false;
            } else if !self.dropping {
                if self.carry.len() == LINE_BYTES {
                    self.carry.clear();
                    self.dropping = true;
                } else {
                    self.carry.push(byte);
                }
            }
        }
        out
    }
    fn push(&self, out: &mut String, line: &str) {
        if let Some(prefix) = &self.prefix {
            out.push_str(prefix);
        }
        out.push_str(line);
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn partial_utf8_history_and_follow_are_not_duplicated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");
        std::fs::write(&path, b"one\ntwo\npartial \xe2").unwrap();
        let mut tail = Tail::new(path.clone(), None);
        assert_eq!(tail.history(1).unwrap(), "two\n");
        assert_eq!(tail.advance(), "");
        File::options()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"\x82\xac\nnext\n")
            .unwrap();
        assert_eq!(tail.advance(), "partial €\nnext\n");
        assert_eq!(tail.advance(), "");
    }
    #[test]
    fn larger_replacement_is_a_new_generation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");
        std::fs::write(&path, "old\n").unwrap();
        let mut tail = Tail::new(path.clone(), None);
        tail.history(300);
        std::fs::rename(&path, dir.path().join("previous")).unwrap();
        std::fs::write(&path, "new and larger\n").unwrap();
        assert_eq!(
            tail.advance(),
            "[Log rotated or truncated]\nnew and larger\n"
        );
    }
    #[test]
    fn copy_truncate_regrown_between_polls_does_not_skip_the_new_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");
        std::fs::write(&path, "old\n").unwrap();
        let mut tail = Tail::new(path.clone(), None);
        tail.history(300);
        std::fs::write(&path, "larger replacement in place\n").unwrap();
        assert_eq!(
            tail.advance(),
            "[Log rotated or truncated]\nlarger replacement in place\n"
        );
        assert_eq!(tail.advance(), "");
    }
    #[test]
    fn sparse_history_and_unterminated_lines_are_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");
        let mut file = File::create(&path).unwrap();
        file.set_len(4 * 1024 * 1024 * 1024).unwrap();
        file.seek(SeekFrom::End(0)).unwrap();
        file.write_all(b"\nlast\n").unwrap();
        let mut tail = Tail::new(path.clone(), None);
        let out = tail.history(300).unwrap();
        assert!(out.ends_with("last\n") && out.len() < 200);
        File::options()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&vec![b'x'; READ_BYTES * 3])
            .unwrap();
        assert!(tail.advance().is_empty());
        assert!(tail.carry.len() <= LINE_BYTES);
        assert!(tail.advance().is_empty());
        assert!(tail.advance().is_empty());
        File::options()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"\nvalid\n")
            .unwrap();
        assert!(tail.advance().ends_with("valid\n"));
    }
}
