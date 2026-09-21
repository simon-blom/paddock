//! Metal checkpoint transport primitives. Frozen buffers cannot be mutated or
//! rebound as GPU destinations after publication to a disk worker.
use crate::device::{Buffer, MetalDevice, MetalError, Result};
use objc2_metal::MTLBuffer;
use paddock_engine::kv_tier::{
    cold::{Payload, Reservation},
    digest::{CacheNamespace, IdentityDigest, PrivacyScope},
};
use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

pub(crate) type FileVersion = (PathBuf, u64, u64, u64, i64, i64, i64, i64);
fn files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.join("manifest.json").is_file() && path.join("target").is_dir() {
        return paddock_models::splash::files(path).map_err(|e| MetalError::Model(e.to_string()));
    }
    let mut files = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| MetalError::Model(e.to_string()))?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(|e| MetalError::Model(e.to_string()))?
            .into_iter()
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("json" | "safetensors")
                )
            })
            .collect()
    } else if let Some(split) = paddock_models::split::parse_split_name(path) {
        (1..=split.count).map(|i| split.sibling(i)).collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(MetalError::Model("empty checkpoint identity".into()));
    }
    Ok(files)
}
pub(crate) fn versions(path: &Path) -> Result<Vec<FileVersion>> {
    use std::os::unix::fs::MetadataExt;
    files(path)?
        .into_iter()
        .map(|p| {
            let m = std::fs::metadata(&p).map_err(|e| MetalError::Model(e.to_string()))?;
            Ok((
                p,
                m.dev(),
                m.ino(),
                m.size(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            ))
        })
        .collect()
}
pub(crate) fn require_unchanged(loaded: &[FileVersion]) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    for v in loaded {
        let m = std::fs::metadata(&v.0).map_err(|e| MetalError::Model(e.to_string()))?;
        if (
            m.dev(),
            m.ino(),
            m.size(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        ) != (v.1, v.2, v.3, v.4, v.5, v.6, v.7)
        {
            return Err(MetalError::Model("checkpoint changed during load/identity hashing; reload before enabling persistent KV".into()));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct KvOffloadConfig {
    pub ram_bytes: u64,
    pub disk: Option<(PathBuf, u64)>,
    /// Runner trust domain. No credentials are written to cache metadata.
    pub scope: Vec<u8>,
}
impl KvOffloadConfig {
    /// Rotate durable-cache ownership when a runner's credential changes.
    /// Only this domain-separated digest enters diagnostics or directory keys.
    pub fn endpoint_scope(port: u16, api_key: Option<&str>) -> Vec<u8> {
        let mut h = blake3::Hasher::new_derive_key("paddock metal endpoint trust domain v1");
        h.update(&port.to_le_bytes());
        h.update(api_key.unwrap_or_default().as_bytes());
        h.finalize().as_bytes().to_vec()
    }
}
static CONFIG: OnceLock<Option<KvOffloadConfig>> = OnceLock::new();
pub fn configure_kv_offload(config: Option<KvOffloadConfig>) {
    let _ = CONFIG.set(config);
}
pub fn kv_offload_config() -> Option<KvOffloadConfig> {
    CONFIG.get().cloned().flatten()
}

struct Frozen(Buffer);
// SAFETY: Frozen is constructed only after the last GPU writer completed. The
// wrapped Buffer is private and never exposed again; shared users can only read.
unsafe impl Sync for Frozen {}
impl AsRef<[u8]> for Frozen {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: completed, immutable shared-storage allocation, held by self.
        unsafe { std::slice::from_raw_parts(self.0.raw.contents().as_ptr().cast(), self.0.len()) }
    }
}
pub(crate) type Span<'a> = (&'a Buffer, usize, usize);
/// Preserve logical page order while coalescing adjacent physical runs. A
/// contiguous prefix needs one blit per plane, not thousands of tiny encodes.
pub(crate) fn paged_spans<'a>(buffer: &'a Buffer, blocks: &[u32], bytes: usize) -> Vec<Span<'a>> {
    let mut spans: Vec<Span<'a>> = Vec::new();
    for &block in blocks {
        let offset = block as usize * bytes;
        if let Some((_, start, len)) = spans.last_mut()
            && *start + *len == offset
        {
            *len += bytes;
        } else {
            spans.push((buffer, offset, bytes));
        }
    }
    spans
}
pub(crate) fn capture(
    device: &MetalDevice,
    spans: &[Span<'_>],
    reservation: Reservation,
) -> Result<Payload> {
    let size = spans
        .iter()
        .try_fold(0usize, |n, (_, _, len)| n.checked_add(*len))
        .ok_or_else(|| MetalError::Memory("checkpoint size overflow".into()))?;
    let frozen = device.alloc(size)?;
    let mut offset = 0;
    let copies: Vec<_> = spans
        .iter()
        .map(|&(buffer, from, len)| {
            let to = offset;
            offset += len;
            (buffer, from, &frozen, to, len)
        })
        .collect();
    device.copy_regions(&copies)?;
    Ok(Payload::new(Frozen(frozen), reservation))
}
pub(crate) fn restore(spans: &[Span<'_>], bytes: &[u8]) -> Result<()> {
    let size = spans.iter().try_fold(0usize, |n, (buffer, offset, len)| {
        if offset
            .checked_add(*len)
            .is_none_or(|end| end > buffer.len())
        {
            None
        } else {
            n.checked_add(*len)
        }
    });
    if size != Some(bytes.len()) {
        return Err(MetalError::Model(
            "KV checkpoint schema/length mismatch".into(),
        ));
    }
    let mut cursor = 0;
    for &(buffer, offset, len) in spans {
        // SAFETY: family owns submission, has required a committed boundary,
        // and reserved every destination page. All ranges validated above; no
        // partial publication on a malformed checkpoint. Source cannot alias.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes[cursor..].as_ptr(),
                buffer.raw.contents().as_ptr().cast::<u8>().add(offset),
                len,
            );
        }
        cursor += len;
    }
    Ok(())
}

struct OwnedSpan {
    buffer: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    len: usize,
}
// SAFETY: task spans exclusively own their reserved destination ranges. Metal
// buffer handles have no thread affinity; retention pins their allocations.
unsafe impl Send for OwnedSpan {}
struct ScatterTask {
    spans: Vec<OwnedSpan>,
    payload: std::sync::Arc<Payload>,
    done: std::sync::mpsc::Sender<()>,
}
/// One bounded copy lane per model, independent of disk latency. The family
/// reserves pages/state until its completion is acquired, including cancellation.
pub(crate) struct ScatterWorker {
    tx: Option<std::sync::mpsc::SyncSender<ScatterTask>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl ScatterWorker {
    pub(crate) fn new() -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<ScatterTask>(8);
        let worker = std::thread::Builder::new()
            .name("paddock-kv-restore".into())
            .spawn(move || {
                while let Ok(task) = rx.recv() {
                    let bytes = task.payload.bytes();
                    let mut cursor = 0;
                    for span in task.spans {
                        // SAFETY: submission validated the full payload and every
                        // range. The engine reserves these disjoint destinations
                        // until receiving done; no GPU/host reader observes them.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes[cursor..].as_ptr(),
                                span.buffer
                                    .contents()
                                    .as_ptr()
                                    .cast::<u8>()
                                    .add(span.offset),
                                span.len,
                            );
                        }
                        cursor += span.len;
                    }
                    // mpsc release/acquire publishes CPU writes before GPU submit.
                    let _ = task.done.send(());
                }
            })
            .map_err(|e| MetalError::Device(e.to_string()))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
        })
    }
    /// Caller must reserve every destination and hold it until completion.
    pub(crate) fn submit(
        &self,
        spans: &[Span<'_>],
        payload: std::sync::Arc<Payload>,
    ) -> Result<std::sync::mpsc::Receiver<()>> {
        let size = spans.iter().try_fold(0usize, |total, (b, start, len)| {
            if start.checked_add(*len).is_none_or(|end| end > b.len()) {
                None
            } else {
                total.checked_add(*len)
            }
        });
        if size != Some(payload.bytes().len()) {
            return Err(MetalError::Model(
                "async KV checkpoint schema/length mismatch".into(),
            ));
        }
        let spans = spans
            .iter()
            .map(|&(b, offset, len)| OwnedSpan {
                buffer: b.raw.clone(),
                offset,
                len,
            })
            .collect();
        let (done, rx) = std::sync::mpsc::channel();
        self.tx
            .as_ref()
            .expect("scatter sender is retained until worker teardown")
            .try_send(ScatterTask {
                spans,
                payload,
                done,
            })
            .map_err(|_| {
                MetalError::Memory("KV restore copy lane is full or unavailable".into())
            })?;
        Ok(rx)
    }
}
impl Drop for ScatterWorker {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Full content commitment, not a path/mtime/sampled-weight guess. Done once
/// during opt-in load; token and layout namespaces survive runner restarts.
pub(crate) fn namespace(paths: &[&Path], layout: &[u8], scope: Vec<u8>) -> Result<CacheNamespace> {
    use std::io::Read;
    let mut h = blake3::Hasher::new_derive_key("paddock metal checkpoint identity v1");
    h.update(layout);
    // Commit both the kernels and the host-side state ABI. A changed shader,
    // quantized loader, row planner or companion invalidates old recurrent KV.
    for source in [
        crate::device::SHADER_SOURCE,
        include_str!("device.rs"),
        include_str!("offload.rs"),
        include_str!("weights.rs"),
        include_str!("affine.rs"),
        include_str!("splash.rs"),
        include_str!("../../paddock-models/src/splash.rs"),
        include_str!("../../paddock-models/src/splash/vision.rs"),
        include_str!("projection.rs"),
        include_str!("schedule.rs"),
        include_str!("paged_offload.rs"),
        include_str!("granite/mod.rs"),
        include_str!("granite/multimodal.rs"),
        include_str!("gpt_oss/mod.rs"),
        include_str!("gpt_oss/load.rs"),
        include_str!("gpt_oss/forward.rs"),
        include_str!("gpt_oss/serving.rs"),
        include_str!("laguna/mod.rs"),
        include_str!("laguna/load.rs"),
        include_str!("laguna/forward.rs"),
        include_str!("laguna/serving.rs"),
        include_str!("qwen35/attention.rs"),
        include_str!("qwen35/checkpoint.rs"),
        include_str!("qwen35/geometry.rs"),
        include_str!("qwen35/mod.rs"),
        include_str!("qwen35/moe.rs"),
        include_str!("qwen35/load.rs"),
        include_str!("qwen35/forward.rs"),
        include_str!("qwen35/projection.rs"),
        include_str!("qwen35/spec.rs"),
        include_str!("qwen35/workspace.rs"),
        include_str!("qwen35/serving.rs"),
        include_str!("qwen35/offload.rs"),
        include_str!("qwen35/mtp.rs"),
        include_str!("qwen35/dflash.rs"),
        include_str!("qwen35/multimodal.rs"),
        include_str!("qwen35/vision.rs"),
    ] {
        h.update(&(source.len() as u64).to_le_bytes());
        h.update(source.as_bytes());
    }
    let mut scratch = vec![0u8; 4 << 20];
    for path in paths {
        let files = files(path)?;
        h.update(&(files.len() as u64).to_le_bytes());
        for file in files {
            let mut f = std::fs::File::open(file).map_err(|e| MetalError::Model(e.to_string()))?;
            h.update(
                &f.metadata()
                    .map_err(|e| MetalError::Model(e.to_string()))?
                    .len()
                    .to_le_bytes(),
            );
            loop {
                let n = f
                    .read(&mut scratch)
                    .map_err(|e| MetalError::Model(e.to_string()))?;
                if n == 0 {
                    break;
                }
                h.update(&scratch[..n]);
            }
        }
    }
    Ok(CacheNamespace {
        identity: IdentityDigest(*h.finalize().as_bytes()),
        scope: PrivacyScope::PerUser(scope),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_engine::kv_tier::cold::{ColdCache, ColdConfig};
    #[test]
    fn checkpoint_gpu_gather_and_checked_scatter_are_byte_exact() {
        let d = MetalDevice::new(Some(64 << 20)).unwrap();
        let a = d
            .upload(&(0u8..=255).cycle().take(16384).collect::<Vec<_>>())
            .unwrap();
        let b = d.upload(&vec![17; 8192]).unwrap();
        let runs = paged_spans(&a, &[5, 6, 1, 2, 3, 9], 256);
        assert_eq!(
            runs.iter()
                .map(|(_, offset, len)| (*offset, *len))
                .collect::<Vec<_>>(),
            vec![(1280, 512), (256, 768), (2304, 256)]
        );
        let mut c = ColdCache::open(
            ColdConfig {
                ram_bytes: 1 << 20,
                disk: None,
            },
            65536,
        )
        .unwrap();
        let p = capture(
            &d,
            &[(&a, 512, 8192), (&b, 1024, 4096)],
            c.reserve(12288).unwrap(),
        )
        .unwrap();
        let dst = d.upload(&vec![99u8; 16384]).unwrap();
        assert!(restore(&[(&dst, 0, 4)], p.bytes()).is_err());
        assert_eq!(unsafe { dst.read_u32(1) }, [0x63636363]);
        restore(&[(&dst, 256, 8192), (&dst, 9216, 4096)], p.bytes()).unwrap();
        let got = unsafe { dst.read_u32(4096) }
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(&got[256..8448], &p.bytes()[..8192]);
        assert_eq!(&got[9216..13312], &p.bytes()[8192..]);
        assert!(got[..256].iter().all(|b| *b == 99));
        assert!(got[8448..9216].iter().all(|b| *b == 99));
        assert!(got[13312..].iter().all(|b| *b == 99));
        let worker = ScatterWorker::new().unwrap();
        let p = std::sync::Arc::new(p);
        let async_dst = d.upload(&vec![0u8; 12288]).unwrap();
        worker
            .submit(&[(&async_dst, 0, 12288)], p.clone())
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let got = unsafe { async_dst.read_u32(3072) }
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(got, p.bytes());
        assert!(worker.submit(&[(&async_dst, 0, 12289)], p.clone()).is_err());
        // Cancellation drops a receipt, not its page/byte ownership. Joining
        // at model shutdown still fences the final writer.
        drop(worker.submit(&[(&async_dst, 0, 12288)], p).unwrap());
        drop(worker);
    }
}
