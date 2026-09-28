//! Paged address modes of the attention kernels Flash-Next (qwen4exp) still
//! reads off a dense slot-major strip (pack slots 688-691). Each wrapper is
//! its dense twin in `attention.rs` with the strip replaced by a block pool
//! `[n_blocks, 16, kv_dim]` and per-slot tables (`block_tables`, stride
//! `blocks_per_slot`, u32 block ids): key `p` of slot `s` is pool row
//! `block_tables[s*bps + p/16]*16 + p%16`. Nothing else changes, so a paged
//! launch is bit-identical to the dense one over the same keys - the gate is
//! `tests/gpu_paged_attn_modes.rs`, on scrambled interleaved tables.
//!
//! A table entry past a slot's live keys is never read, so the caller only
//! has to back the blocks a launch's rows actually reach.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::types::KvDtype;
use super::{GpuError, GpuExecutor};

impl GpuExecutor {
    /// True when the pack carries every paged twin of Flash-Next's dense
    /// attention arms (slots 688-691).
    pub fn has_attn_paged_modes(&self) -> bool {
        self.kernels.attn_prefill_batch_paged.is_some()
            && self.kernels.attn_decode_fmha_paged.is_some()
            && self.kernels.attn_decode_fmha_sp_paged.is_some()
            && self.kernels.attn_decode_batch_ps_paged.is_some()
    }

    /// Paged twin of [`Self::attn_prefill_batch`] (slot 688): the per-tile
    /// multi-slot prefill walk over the pool.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_batch_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        tile_row0: &CudaSlice<u32>,
        tile_slot: &CudaSlice<u32>,
        n_qtiles: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        n_rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_batch_paged
            .ok_or(GpuError::MissingOp("attn_prefill_batch_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        let (trp, _g9) = tile_row0.device_ptr(&self.stream);
        let (tsp, _g10) = tile_slot.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 688); pools [n_blocks, 16, kv_dim] of kv_dtype
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                trp as *const _,
                tsp as *const _,
                n_qtiles as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                n_rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged twin of [`Self::attn_decode_fmha`] (slot 689).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_fmha_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_fmha_paged
            .ok_or(GpuError::MissingOp("attn_decode_fmha_paged"))?;
        self.decode_paged_call(
            f,
            q,
            pool_k,
            pool_v,
            sinks,
            out,
            positions,
            slots,
            block_tables,
            blocks_per_slot,
            [n_heads, n_kv_heads, head_dim, kv_dim, swa_window, batch],
            scale,
            kv_dtype,
        )
    }

    /// Paged twin of [`Self::attn_decode_batch_ps`] (slot 691).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch_ps_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_ps_paged
            .ok_or(GpuError::MissingOp("attn_decode_batch_ps_paged"))?;
        self.decode_paged_call(
            f,
            q,
            pool_k,
            pool_v,
            sinks,
            out,
            positions,
            slots,
            block_tables,
            blocks_per_slot,
            [n_heads, n_kv_heads, head_dim, kv_dim, swa_window, batch],
            scale,
            kv_dtype,
        )
    }

    /// One launch of an `attn_decode_batch_paged`-shaped entry; `dims` is
    /// `[n_heads, n_kv_heads, head_dim, kv_dim, swa_window, batch]`.
    #[allow(clippy::too_many_arguments)]
    fn decode_paged_call(
        &self,
        f: paddock_kernels::abi::AttnDecodeBatchPagedFn,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        dims: [usize; 6],
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let [n_heads, n_kv_heads, head_dim, kv_dim, swa_window, batch] = dims;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools [n_blocks, 16, kv_dim] of kv_dtype
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged twin of [`Self::attn_decode_fmha_sp`] (slot 690): `part` is the
    /// same caller-owned scratch of batch*n_heads*split*(head_dim+2) f32.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_fmha_sp_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        split: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_fmha_sp_paged
            .ok_or(GpuError::MissingOp("attn_decode_fmha_sp_paged"))?;
        debug_assert!(part.len() >= batch * n_heads * split * (head_dim + 2));
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (ptp, _g6) = part.device_ptr_mut(&self.stream);
        let (pp, _g7) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 690); scratch size checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                ptp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                split as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }
}
