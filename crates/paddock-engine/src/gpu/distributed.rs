//! Engine-owned NCCL process group. `paddock-dist` transports the opaque ID
//! during bootstrap; no model/scheduler data is sent over TCP here.
//!
//! Collectives enqueue on a dedicated communication stream. The caller must
//! fence producer compute with `after_compute` and fence consumers with
//! `before_compute`; neither fence synchronizes the host. Buffers must remain
//! alive until the consumer stream passes the completion event (or, for a
//! standalone benchmark, until the communication stream is synchronized).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::nccl::{self, Comm, Id, NcclType, ReduceOp};
use paddock_dist::config::Resolved;
use paddock_dist::protocol::NCCL_ID_BYTES;

#[derive(Debug, thiserror::Error)]
pub enum CollectiveError {
    #[error(
        "NCCL library (libnccl.so) is unavailable; install NCCL 2.30 and add its lib directory to LD_LIBRARY_PATH"
    )]
    MissingNccl,
    #[error("invalid tensor-parallel group: {0}")]
    InvalidGroup(String),
    #[error("invalid collective buffer: {0}")]
    Shape(String),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("NCCL {operation} failed: {status:?}")]
    Nccl {
        operation: &'static str,
        status: nccl::sys::ncclResult_t,
    },
    #[error("NCCL {0} returned an unfinished status; asynchronous initialization is not supported")]
    InProgress(&'static str),
}

fn nccl_result<T>(
    operation: &'static str,
    result: Result<T, nccl::result::NcclError>,
) -> Result<T, CollectiveError> {
    result.map_err(|e| CollectiveError::Nccl {
        operation,
        status: e.0,
    })
}

fn enqueue(
    operation: &'static str,
    result: Result<nccl::result::NcclStatus, nccl::result::NcclError>,
) -> Result<(), CollectiveError> {
    match nccl_result(operation, result)? {
        nccl::result::NcclStatus::Success => Ok(()),
        _ => Err(CollectiveError::InProgress(operation)),
    }
}

/// Check before the first NCCL call: cudarc's lazy loader panics when the
/// library is absent. This probe is not called on TP=1.
fn require_nccl() -> Result<(), CollectiveError> {
    // SAFETY: probes whether dlopen can locate the library; does not call NCCL.
    if unsafe { nccl::sys::is_culib_present() } {
        Ok(())
    } else {
        Err(CollectiveError::MissingNccl)
    }
}

/// Rank 0 creates this opaque 128-byte ID; send it over the existing
/// length-prefixed control channel using `paddock_dist::protocol::send_nccl_id`.
/// Never called for TP=1.
pub fn create_unique_id() -> Result<[u8; NCCL_ID_BYTES], CollectiveError> {
    require_nccl()?;
    let id = nccl_result("get unique ID", Id::new())?;
    Ok(id.internal().map(|byte| byte.to_ne_bytes()[0]))
}

/// The interface model code will consume in later phases. All calls enqueue
/// asynchronously on the communicator's dedicated stream. No socket or NCCL
/// bootstrap detail leaks into model layers.
pub trait Communicator {
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    fn after_compute(&self, compute: &CudaStream) -> Result<(), CollectiveError>;
    fn before_compute(&self, compute: &CudaStream) -> Result<(), CollectiveError>;
    fn all_reduce<T: NcclType, S: DevicePtr<T>, R: DevicePtrMut<T>>(
        &self,
        src: &S,
        dst: &mut R,
    ) -> Result<(), CollectiveError>;
    fn all_gather<T: NcclType, S: DevicePtr<T>, R: DevicePtrMut<T>>(
        &self,
        src: &S,
        dst: &mut R,
    ) -> Result<(), CollectiveError>;
    fn reduce_scatter<T: NcclType, S: DevicePtr<T>, R: DevicePtrMut<T>>(
        &self,
        src: &S,
        dst: &mut R,
    ) -> Result<(), CollectiveError>;
    fn broadcast<T: NcclType, R: DevicePtrMut<T>>(
        &self,
        dst: &mut R,
        root: usize,
    ) -> Result<(), CollectiveError>;
}

/// One NCCL communicator per rank, bound to one dedicated CUDA stream.
/// `Comm` keeps its stream alive; the events remain valid across enqueues.
pub struct NcclCommunicator {
    comm: Comm,
    stream: Arc<CudaStream>,
}

impl NcclCommunicator {
    /// Called by both ranks after handshake and ID exchange. For the TP=1
    /// historical path, callers receive `None` without touching NCCL or CUDA.
    pub fn from_resolved(
        resolved: Option<&Resolved>,
        context: &Arc<CudaContext>,
        unique_id: [u8; NCCL_ID_BYTES],
    ) -> Result<Option<Self>, CollectiveError> {
        let Some(resolved) = resolved else {
            return Ok(None);
        };
        if resolved.tp_size != 2 || resolved.rank >= 2 {
            return Err(CollectiveError::InvalidGroup(format!(
                "expected TP=2 rank 0 or 1, got TP={} rank {}",
                resolved.tp_size, resolved.rank
            )));
        }
        require_nccl()?;
        context.bind_to_thread()?;
        let stream = context.new_stream()?;
        let internal = unique_id.map(|byte| std::ffi::c_char::from_ne_bytes([byte]));
        let comm = nccl_result(
            "init rank",
            Comm::from_rank(stream.clone(), resolved.rank, 2, Id::uninit(internal)),
        )?;
        Ok(Some(Self { comm, stream }))
    }

    /// Expose the stream for GPU timing and standalone probes. Normal model
    /// code uses the `Communicator` fences instead.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl Communicator for NcclCommunicator {
    fn rank(&self) -> usize {
        self.comm.rank()
    }
    fn world_size(&self) -> usize {
        self.comm.world_size()
    }

    fn after_compute(&self, compute: &CudaStream) -> Result<(), CollectiveError> {
        let ready = compute.record_event(None)?;
        self.stream.wait(&ready)?;
        Ok(())
    }

    fn before_compute(&self, compute: &CudaStream) -> Result<(), CollectiveError> {
        let done: CudaEvent = self.stream.record_event(None)?;
        compute.wait(&done)?;
        Ok(())
    }

    fn all_reduce<T: NcclType, S: DevicePtr<T>, R: DevicePtrMut<T>>(
        &self,
        src: &S,
        dst: &mut R,
    ) -> Result<(), CollectiveError> {
        if src.len() != dst.len() {
            return Err(CollectiveError::Shape(format!(
                "all-reduce: input {} != output {}",
                src.len(),
                dst.len()
            )));
        }
        enqueue("all-reduce", self.comm.all_reduce(src, dst, &ReduceOp::Sum))
    }

    fn all_gather<T: NcclType, S: DevicePtr<T>, R: DevicePtrMut<T>>(
        &self,
        src: &S,
        dst: &mut R,
    ) -> Result<(), CollectiveError> {
        if src.len().checked_mul(self.world_size()) != Some(dst.len()) {
            return Err(CollectiveError::Shape(format!(
                "all-gather: output {} != input {} * {}",
                dst.len(),
                src.len(),
                self.world_size()
            )));
        }
        enqueue("all-gather", self.comm.all_gather(src, dst))
    }

    fn reduce_scatter<T: NcclType, S: DevicePtr<T>, R: DevicePtrMut<T>>(
        &self,
        src: &S,
        dst: &mut R,
    ) -> Result<(), CollectiveError> {
        if dst.len().checked_mul(self.world_size()) != Some(src.len()) {
            return Err(CollectiveError::Shape(format!(
                "reduce-scatter: input {} != output {} * {}",
                src.len(),
                dst.len(),
                self.world_size()
            )));
        }
        enqueue(
            "reduce-scatter",
            self.comm.reduce_scatter(src, dst, &ReduceOp::Sum),
        )
    }

    fn broadcast<T: NcclType, R: DevicePtrMut<T>>(
        &self,
        dst: &mut R,
        root: usize,
    ) -> Result<(), CollectiveError> {
        if root >= self.world_size() {
            return Err(CollectiveError::Shape(format!(
                "broadcast root {root} outside world size {}",
                self.world_size()
            )));
        }
        enqueue("broadcast", self.comm.broadcast_in_place(dst, root as i32))
    }
}
