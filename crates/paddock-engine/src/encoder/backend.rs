//! Backend-owned completion handles keep CUDA events and Metal command buffers
//! out of the request scheduler. Construction and all calls stay on its thread;
//! neither the backend nor its pending handles need to be Send/Sync.

pub trait EncoderBackend {
    type Pending;
    fn weights_mem_bytes(&self) -> Option<u64>;
    fn device_mem_used(&self) -> Option<u64>;
    fn coalesce_row_budget(&self) -> usize;
    /// Validate each request before merging, so one malformed caller cannot
    /// fail unrelated callers sharing its GPU batch. No device work here.
    fn validate(&self, _seqs: &[Vec<u32>]) -> Result<(), String> {
        Ok(())
    }
    fn lanes(&mut self) -> usize;
    fn inflight_capacity(&mut self) -> usize {
        2 * self.lanes()
    }
    fn pool_ready(&self, pending: &Self::Pending) -> bool;
    fn embed_submit(&mut self, seqs: &[Vec<u32>], lane: usize) -> Result<Self::Pending, String>;
    fn embed_collect(&mut self, pending: &Self::Pending) -> Result<Vec<Vec<f32>>, String>;
    fn rerank_submit(
        &mut self,
        seqs: &[Vec<u32>],
        yes: u32,
        no: u32,
        lane: usize,
    ) -> Result<Self::Pending, String>;
    fn rerank_collect(
        &mut self,
        pending: &Self::Pending,
        yes: u32,
        no: u32,
    ) -> Result<Vec<f32>, String>;
    fn block_scale_calibration(&self) -> bool {
        false
    }
    fn calibrate_bs(
        &mut self,
        _seqs: &[Vec<u32>],
        _n_docs: usize,
        _rel: &[usize],
    ) -> Result<&'static str, String> {
        Err("this encoder has no block-scale calibration".into())
    }
    fn calibrate_bs_rerank(
        &mut self,
        _seqs: &[Vec<u32>],
        _yes: u32,
        _no: u32,
        _group: usize,
        _rel: &[usize],
    ) -> Result<&'static str, String> {
        Err("this encoder has no block-scale calibration".into())
    }
    fn import_smooth(&mut self, _bytes: &[u8]) -> Result<bool, String> {
        Ok(false)
    }
    fn apply_bs_profile(&mut self, _profile: &str) -> Result<bool, String> {
        Ok(false)
    }
    fn export_smooth(&self) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
impl EncoderBackend for crate::gpu_model::qwen3::GpuQwen3 {
    type Pending = crate::gpu_model::qwen3::PendingPooled;
    fn weights_mem_bytes(&self) -> Option<u64> {
        self.weights_mem_bytes()
    }
    fn device_mem_used(&self) -> Option<u64> {
        self.device_mem_used()
    }
    fn coalesce_row_budget(&self) -> usize {
        self.coalesce_row_budget()
    }
    fn lanes(&mut self) -> usize {
        self.lanes()
    }
    fn pool_ready(&self, p: &Self::Pending) -> bool {
        self.pool_ready(p)
    }
    fn embed_submit(&mut self, s: &[Vec<u32>], lane: usize) -> Result<Self::Pending, String> {
        self.embed_submit(s, lane).map_err(|e| e.to_string())
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> Result<Vec<Vec<f32>>, String> {
        self.embed_collect(p).map_err(|e| e.to_string())
    }
    fn rerank_submit(
        &mut self,
        s: &[Vec<u32>],
        y: u32,
        n: u32,
        lane: usize,
    ) -> Result<Self::Pending, String> {
        self.rerank_submit(s, y, n, lane).map_err(|e| e.to_string())
    }
    fn rerank_collect(&mut self, p: &Self::Pending, y: u32, n: u32) -> Result<Vec<f32>, String> {
        self.rerank_collect(p, y, n).map_err(|e| e.to_string())
    }
    fn block_scale_calibration(&self) -> bool {
        true
    }
    fn calibrate_bs(
        &mut self,
        s: &[Vec<u32>],
        n: usize,
        r: &[usize],
    ) -> Result<&'static str, String> {
        self.calibrate_bs(s, n, r).map_err(|e| e.to_string())
    }
    fn calibrate_bs_rerank(
        &mut self,
        s: &[Vec<u32>],
        y: u32,
        n: u32,
        g: usize,
        r: &[usize],
    ) -> Result<&'static str, String> {
        self.calibrate_bs_rerank(s, y, n, g, r)
            .map_err(|e| e.to_string())
    }
    fn import_smooth(&mut self, b: &[u8]) -> Result<bool, String> {
        self.import_smooth(b).map_err(|e| e.to_string())
    }
    fn apply_bs_profile(&mut self, p: &str) -> Result<bool, String> {
        self.apply_bs_profile(p).map_err(|e| e.to_string())
    }
    fn export_smooth(&self) -> Result<Option<Vec<u8>>, String> {
        self.export_smooth().map_err(|e| e.to_string())
    }
}
