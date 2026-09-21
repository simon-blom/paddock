//! Exact storage-format primitives for Flash Next's missing low-bit planes.
//! This is not qwen4exp model qualification: the graph, PLE lifetime, sparse
//! cache and full-generation gates still have to be connected and measured.
use crate::device::{Buffer, Commands, MetalError, Result};

pub(crate) fn is_iq(ty: u32) -> bool {
    matches!(ty, 20..=22)
}

/// A GGUF superblock can straddle rows (notably K=640). Validate the whole
/// tensor, not rounded-up per-row storage; all addressing uses global element
/// indices. A 32-element row boundary also makes every vector/tile load safe.
pub(crate) fn validate(ty: u32, dims: &[usize], bytes: usize) -> Result<()> {
    let (block, size) = match ty {
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        _ => return Err(MetalError::Model("unsupported Metal i-quant type".into())),
    };
    if !(2..=3).contains(&dims.len())
        || dims.iter().any(|&d| d == 0 || d > u32::MAX as usize)
        || !dims[0].is_multiple_of(32)
    {
        return Err(MetalError::Model(
            "Metal i-quant needs positive bounded matrix dimensions and a 32-aligned K".into(),
        ));
    }
    let elements = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    let expected = elements
        .filter(|n| n.is_multiple_of(block))
        .and_then(|n| (n / block).checked_mul(size));
    if expected != Some(bytes) {
        return Err(MetalError::Model(
            "Metal i-quant tensor byte count / whole-superblock mismatch".into(),
        ));
    }
    Ok(())
}

pub(crate) fn linear(
    cmd: &Commands<'_>,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    params: &[u32; 5],
    prepared: bool,
) {
    let [k, n, rows, ty, _] = *params;
    debug_assert!(k.is_multiple_of(32) && rows > 0 && is_iq(ty));
    let format = (ty - 20) as usize;
    let (kernel, columns, tile) = if prepared {
        (
            ["iq_prepared20", "iq_prepared21", "iq_prepared22"][format],
            32,
            32,
        )
    } else if rows > 8 {
        (["iq_mm20", "iq_mm21", "iq_mm22"][format], 32, 32)
    } else {
        let (index, tile) = match rows {
            1 => (0, 1),
            2..=4 => (1, 4),
            _ => (2, 8),
        };
        (
            [
                ["iq_mv20_1", "iq_mv20_4", "iq_mv20_8"],
                ["iq_mv21_1", "iq_mv21_4", "iq_mv21_8"],
                ["iq_mv22_1", "iq_mv22_4", "iq_mv22_8"],
            ][format][index],
            4,
            tile,
        )
    };
    cmd.dispatch(
        kernel,
        &[weight, input, output],
        params,
        [
            n.div_ceil(columns) as usize,
            rows.div_ceil(tile) as usize,
            1,
        ],
        128,
    );
}

#[cfg(test)]
#[path = "iquant/tests.rs"]
mod tests;
