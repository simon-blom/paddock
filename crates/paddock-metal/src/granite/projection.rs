use super::*;

impl Granite {
    pub(super) fn direct_prefill(&self) -> bool {
        let enabled = !self.mlx
            && self.device.tensor_accelerated()
            && self.width == 2048
            && self.ff == 6144
            && self.heads == 16
            && self.kv_heads == 2
            && self.head_dim == 128
            && self.attention_scale.to_bits() == (1.0 / 128f32.sqrt()).to_bits();
        #[cfg(test)]
        let enabled = enabled && !minicpm_tests::BASELINE_PREFILL.get();
        enabled
    }

    pub(super) fn project(
        &self,
        cmd: &Commands<'_>,
        planes: &[(&Weight, &Buffer)],
        input: &Buffer,
        rows: usize,
        scale: f32,
    ) {
        if self.mlx {
            assert_eq!(scale, 1.0);
            crate::affine::project_llama(cmd, planes, input, rows, &self.scratch.gemm_input);
            return;
        }
        let paired = self.width == 2048
            && self.heads == 16
            && self.kv_heads == 2
            && self.ff == 6144
            && scale == 1.0
            && (3..=4).contains(&rows)
            && planes.iter().all(|(w, _)| matches!(w.ty, 12..=14));
        #[cfg(test)]
        let paired = paired && !minicpm_tests::BASELINE_PROJECTION.get();
        if paired {
            crate::weights::paired_projections(cmd, planes, input, rows);
        } else if planes.len() == 1 {
            planes[0].0.linear(
                cmd,
                input,
                planes[0].1,
                rows,
                scale,
                &self.scratch.gemm_input,
            );
        } else {
            assert_eq!(scale, 1.0);
            projections(cmd, planes, input, rows, &self.scratch.gemm_input);
        }
    }
}
