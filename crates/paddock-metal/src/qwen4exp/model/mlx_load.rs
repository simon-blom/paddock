use super::super::{affine, mlx};
use super::*;

impl FlashNext {
    pub(super) fn load_mlx(
        path: &Path,
        context: usize,
        batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        Self::memory(context, batch)?;
        let source = mlx::Source::open(path)?;
        let weight_bytes = source.plan.resident_weight_bytes;
        let (device, chunk, cache_bytes, scratch_bytes) =
            Self::mlx_device(context, batch, weight_bytes, budget)?;
        let required = weight_bytes + cache_bytes + scratch_bytes;
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Flash Next MLX text needs {required} bytes ({weight_bytes} compressed weights/small GPU parameters + {cache_bytes} cache + {scratch_bytes} scratch); grant {}. No CPU/offload fallback or hidden memory-limit increase",
                device.budget_bytes()
            )));
        }
        let pages = context.div_ceil(BLOCK_TOKENS);
        let embedding = source.weight(&device, &format!("{}.embed_tokens", mlx::ROOT))?;
        let head = source.weight(&device, "language_model.lm_head")?;
        let output_hc = HyperConnection::load_mlx(
            &device,
            &source,
            &format!("{}.hyper_connection_mixer", mlx::ROOT),
            false,
        )?;
        let ple_weights = ple::Weights::load_mlx(&device, &source)?;
        // Materialize the largest single allocation while the remaining
        // budget is free. Admitting it after 79 GB of decoder weights creates
        // a large residency transition at the tightest point of the load.
        let ple_table = source.table(&device)?;
        #[cfg(test)]
        eprintln!(
            "MLX_NATIVE PLE loaded allocated_bytes={}",
            device.allocated_bytes()
        );
        // All expert planes stream straight into their final allocation.
        let mut layers = Vec::with_capacity(48);
        for li in 0..48 {
            let hc = HyperConnection::load_mlx(
                &device,
                &source,
                &format!("{}.layers.{li}.attn_hyper_connection", mlx::ROOT),
                true,
            )?;
            let mixer = if li % 4 == 3 {
                Mixer::Qsa(
                    qsa::Weights::load_mlx(&device, &source, li)?,
                    qsa::Cache::new(&device, batch, pages)?,
                )
            } else {
                Mixer::Delta(
                    deltanet::Weights::load_mlx(&device, &source, li)?,
                    deltanet::Cache::new(&device, batch)?,
                )
            };
            layers.push(Layer {
                hc,
                mixer,
                ffn: moe::Weights::load_mlx(&device, &source, li)?,
            });
            if li % 8 == 7 {
                #[cfg(test)]
                eprintln!(
                    "MLX_NATIVE loading layers={} allocated_bytes={}",
                    li + 1,
                    device.allocated_bytes()
                );
                tracing::info!(
                    layers = li + 1,
                    "loading native Metal Flash Next affine checkpoint"
                );
            }
        }
        let scratch = Scratch::new(&device, context, batch, chunk)?;
        let affine_scratch = Some(device.alloc(affine::WORKSPACE_BYTES)?);
        if device.allocated_bytes() != required {
            return Err(MetalError::Memory(format!(
                "Flash Next MLX ledger differs: {} vs {required}",
                device.allocated_bytes()
            )));
        }
        tracing::warn!(
            weight_bytes,
            cache_bytes,
            scratch_bytes,
            context,
            prefill_rows = chunk,
            batch,
            "EXPERIMENTAL native Flash Next MLX text graph; BF16 KV/activations, F32 recurrent state; no vision/MTP or registry election; generation/performance qualification pending"
        );
        Ok(Self {
            device,
            embedding,
            head,
            output_hc,
            ple_weights,
            ple_table,
            layers,
            scratch,
            affine_scratch,
            slots: (0..batch).map(|_| Slot::default()).collect(),
            pool: KvPool::with_blocks((pages * batch) as u32),
            pending: VecDeque::new(),
            context,
            chunk,
            pages,
            weight_bytes,
            cache_bytes,
            poisoned: false,
            last_gpu_seconds: 0.,
        })
    }

    /// Prefer wider prefill without stealing a previously admissible cache
    /// budget. A rejected new_planned memory check precedes queue/compiler/
    /// weight allocation. Never retry other errors or raise an explicit grant.
    pub(super) fn mlx_device(
        context: usize,
        batch: usize,
        weights: u64,
        budget: Option<u64>,
    ) -> Result<(MetalDevice, usize, u64, u64)> {
        for chunk in [MLX_CHUNK, 512, CHUNK] {
            let (cache, scratch) = Self::memory_rows(context, batch, chunk)?;
            let scratch = scratch + affine::WORKSPACE_BYTES as u64;
            let required = weights
                .checked_add(cache)
                .and_then(|v| v.checked_add(scratch))
                .ok_or_else(|| MetalError::Memory("Flash Next MLX reservation overflow".into()))?;
            match MetalDevice::new_planned(budget, required) {
                Ok(device) => return Ok((device, chunk, cache, scratch)),
                Err(MetalError::Memory(_)) if chunk != CHUNK => continue,
                Err(error) => return Err(error),
            }
        }
        unreachable!("last bounded capacity returns its exact admission result")
    }

    pub(super) fn is_mlx(&self) -> bool {
        affine::is_affine(self.embedding.ty)
    }
}
