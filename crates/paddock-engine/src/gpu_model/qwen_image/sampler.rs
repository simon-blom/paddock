//! The flow-matching schedule, the timestep embedding, guidance and the
//! Euler update - `FlowMatchEulerDiscreteScheduler` as the 2.1 pipeline
//! configures it, in f32 the way numpy/torch evaluate it.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor};

pub use crate::image::schedule::{Schedule, mu_for_tokens, timestep_embedding};

/// Classifier-free guidance in place: `u = u + s * (c - u)`. Three
/// launches over the existing fused scale-add; `c` is left holding `c - u`
/// afterwards (it is dead by then).
pub fn guidance_combine(
    exec: &GpuExecutor,
    uncond: &mut CudaSlice<f32>,
    cond: &CudaSlice<f32>,
    scale: f32,
    n: usize,
) -> Result<(), GpuError> {
    // diff = c - u, computed into a scratch copy of c
    let mut diff = exec.alloc(n)?;
    exec.copy_region(cond, 0, &mut diff, 0, n)?;
    exec.scale_add(&mut diff, uncond, -1.0, n)?;
    exec.scale_add(uncond, &diff, scale, n)
}
