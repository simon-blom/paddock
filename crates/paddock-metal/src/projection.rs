//! Preserve F32 operands through the backbone of the discontinuous MoE graph.
//! A live decode cannot acquire a new F16 rounding boundary merely because
//! an unrelated prefill joins its tick. Use original bounded TensorOps tiles
//! for prompt rows and register-reusing SIMD for narrow rows, not CPU replay.
use crate::device::{Buffer, Commands};
use crate::weights::Weight;

pub(crate) fn project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
) {
    for &(w, out) in planes {
        let (kernel, columns, tile) = match w.ty {
            12..=14 => match rows {
                1 => ("linear_kquant1", 4, 1),
                2 => ("linear_kquant2", 16, 2),
                3 => ("linear_kquant3", 16, 3),
                4 => ("linear_kquant4", 16, 4),
                5 => ("gemma_full5", 16, 5),
                6 => ("gemma_full6", 16, 6),
                7 => ("gemma_full7", 16, 7),
                8 => ("gemma_full8", 16, 8),
                _ => ("laguna_dense_f32", 16, 32),
            },
            8 => match rows {
                1 => ("linear_q8_r1", 4, 1),
                2..=4 => ("linear_q8_r4", 4, 4),
                5..=8 => ("linear_q8_r8", 4, 8),
                9..=15 => ("linear_q8_r16", 4, 16),
                _ => ("laguna_dense_f32", 16, 32),
            },
            _ => unreachable!("validated precise projection type"),
        };
        cmd.dispatch(
            kernel,
            &[&w.buffer, input, out],
            &[w.k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(columns), rows.div_ceil(tile), 1],
            128,
        );
    }
}
