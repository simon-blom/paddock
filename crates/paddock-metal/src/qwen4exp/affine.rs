//! Flash Next's affine storage. This is deliberately separate from the
//! qualified dense-Qwen group-64 kernels: changing that arithmetic would
//! invalidate an unrelated model's parity gate. No host dequantization.
use crate::{
    device::{Buffer, Commands, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

pub(super) const A4G32: u32 = 0x101;
pub(super) const A8G64: u32 = 0x102;
// Split-K targets at most 512 output tiles, each 32x32 BF16 values.
// Partial storage is bounded independently of the model's weight sizes.
pub(super) const WORKSPACE_BYTES: usize = 512 * 32 * 32 * 2;
pub(super) fn is_affine(ty: u32) -> bool {
    matches!(ty, A4G32 | A8G64)
}

pub(super) fn format(ty: u32) -> (usize, usize) {
    match ty {
        A4G32 => (4, 32),
        A8G64 => (8, 64),
        _ => panic!("not a Flash Next affine tensor"),
    }
}

/// Validate all three planes before allocating. Shape is logical row-major,
/// including the leading expert dimension, not the packed U32 shape.
pub(super) fn parts<'a>(
    source: &'a ShardedSafetensors,
    base: &str,
    shape: &[usize],
    ty: u32,
) -> Result<Vec<&'a [u8]>> {
    let (bits, group) = format(ty);
    if shape.len() < 2
        || shape.contains(&0)
        || !shape.last().is_some_and(|k| k.is_multiple_of(group))
    {
        return Err(MetalError::Model(format!(
            "{base}: invalid affine shape {shape:?}"
        )));
    }
    let mut result = Vec::with_capacity(3);
    for (suffix, dtype, divisor) in [
        ("weight", StDtype::U32, 32 / bits),
        ("scales", StDtype::Bf16, group),
        ("biases", StDtype::Bf16, group),
    ] {
        let name = format!("{base}.{suffix}");
        let mut expected = shape.to_vec();
        *expected
            .last_mut()
            .expect("affine shape has at least two validated dimensions") /= divisor;
        let (info, bytes) = source
            .bytes(&name)
            .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
        if info.dtype != dtype || info.shape != expected {
            return Err(MetalError::Model(format!(
                "{name}: expected {dtype:?} {expected:?}, got {:?} {:?}",
                info.dtype, info.shape
            )));
        }
        result.push(bytes);
    }
    Ok(result)
}

pub(super) fn load(
    d: &MetalDevice,
    source: &ShardedSafetensors,
    base: &str,
    shape: &[usize],
    ty: u32,
) -> Result<Weight> {
    let parts = parts(source, base, shape, ty)?;
    let size = parts.iter().map(|p| p.len()).sum();
    let buffer = d.upload_with(size, |out| {
        let mut offset = 0;
        for (part, suffix) in parts.iter().zip(["weight", "scales", "biases"]) {
            source
                .read_into(
                    &format!("{base}.{suffix}"),
                    &mut out[offset..offset + part.len()],
                )
                .map_err(|e| MetalError::Model(e.to_string()))?;
            offset += part.len();
        }
        Ok(())
    })?;
    Ok(Weight {
        buffer,
        ty,
        k: *shape
            .last()
            .expect("parts validated the affine shape before upload"),
        n: shape[..shape.len() - 1].iter().product(),
    })
}

/// Bounded tiled prefill and direct compressed-vector decode. The narrow
/// HC injection / recurrent gates use a row-invariant vector contraction,
/// as required by the upstream model's explicit singleton projection.
pub(super) fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    if !cmd.independent_rows()
        && let Some(spans) = cmd.projection_rows()
    {
        let mut end = 0;
        for &(start, count, logical_count) in spans {
            assert!(start == end && count > 0 && start + count <= rows);
            assert!(count <= logical_count && logical_count <= 1024);
            project_span(cmd, w, x, y, count, start, logical_count);
            end += count;
        }
        assert_eq!(end, rows);
        return;
    }
    project_span(cmd, w, x, y, rows, 0, rows);
}

fn project_span(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    y: &Buffer,
    rows: usize,
    start: usize,
    logical_rows: usize,
) {
    let (bits, group) = format(w.ty);
    assert!(rows > 0 && x.len() >= (start + rows) * w.k * 4 && y.len() >= (start + rows) * w.n * 4);
    let tile = !cmd.independent_rows() && logical_rows >= 13 && w.n > 48;
    let wide = !cmd.independent_rows() && !tile && logical_rows > 1 && w.n > 48;
    // Test-binary-only arithmetic bisect. No runner setting or production
    // branch: isolate compiler specialization from the prefill graph.
    #[cfg(test)]
    let audit = *{
        static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        MODE.get_or_init(
            || match std::env::var("PADDOCK_FLASH_NEXT_MLX_MV_AUDIT").as_deref() {
                Ok("generic") => 1,
                Ok("entries") => 2,
                Err(_) => 0,
                _ => panic!("invalid test-only affine audit mode"),
            },
        )
    };
    #[cfg(not(test))]
    let audit = 0;
    if audit == 1 && !tile && !wide {
        cmd.dispatch(
            "q4a_mv",
            &[&w.buffer, x, y],
            &[
                w.k as u32,
                w.n as u32,
                rows as u32,
                bits as u32,
                group as u32,
                1,
                start as u32,
            ],
            [w.n.div_ceil(16), rows, 1],
            128,
        );
        return;
    }
    let narrow = audit != 2 && !tile && !wide && bits == 4 && w.n == 4;
    let half_group = !tile
        && !wide
        && audit != 2
        && bits == 4
        && w.k >= 8192
        && w.k.is_multiple_of(512)
        && w.n >= 8
        && w.n <= 512
        && w.n.is_multiple_of(8);
    let align = group.max(32);
    let mut parts = if tile {
        (512 / (w.n.div_ceil(32) * logical_rows.div_ceil(32)))
            .min(w.k / align)
            .max(1)
    } else {
        1
    };
    while !w.k.is_multiple_of(parts * align) {
        parts -= 1;
    }
    if tile
        && parts > 1
        && let Some(scratch) = cmd.projection_workspace()
    {
        assert!(scratch.len() >= rows * w.n * parts * 2);
        let p = [
            w.k as u32,
            w.n as u32,
            rows as u32,
            bits as u32,
            group as u32,
            parts as u32,
            start as u32,
        ];
        cmd.dispatch(
            "q4a_mm_split",
            &[&w.buffer, x, scratch],
            &p,
            [w.n.div_ceil(32), rows.div_ceil(32), parts],
            128,
        );
        cmd.dispatch(
            "q4a_mm_join",
            &[scratch, y],
            &p,
            [(rows * w.n).div_ceil(256), 1, 1],
            256,
        );
        return;
    }
    cmd.dispatch(
        if tile && parts == 1 {
            if bits == 4 {
                "q4a_mm4_packed"
            } else {
                "q4a_mm8_packed"
            }
        } else if tile {
            "q4a_mm"
        } else if wide {
            "q4a_wide"
        } else if narrow {
            "q4a_mv4_narrow"
        } else if half_group {
            "q4a_mv4_fast2"
        } else {
            let fast = w.k.is_multiple_of((32 / bits) * 64) && w.n >= 8 && w.n.is_multiple_of(8);
            match (bits, fast) {
                (4, true) => "q4a_mv4_fast",
                (4, false) => "q4a_mv4",
                (8, true) => "q4a_mv8_fast",
                (8, false) => "q4a_mv8",
                _ => unreachable!(),
            }
        },
        &[&w.buffer, x, y],
        &[
            w.k as u32,
            w.n as u32,
            rows as u32,
            bits as u32,
            group as u32,
            parts as u32,
            start as u32,
        ],
        [
            w.n.div_ceil(if tile {
                32
            } else if wide {
                8
            } else if narrow {
                4
            } else if half_group {
                8
            } else {
                16
            }),
            rows.div_ceil(if tile { 32 } else { 1 }),
            1,
        ],
        if wide || half_group { 64 } else { 128 },
    );
}

pub(super) fn gather(cmd: &Commands<'_>, w: &Weight, ids: &Buffer, y: &Buffer, rows: usize) {
    let (bits, group) = format(w.ty);
    assert!(ids.len() >= rows * 4 && y.len() >= rows * w.k * 4);
    cmd.dispatch(
        "q4a_gather",
        &[&w.buffer, ids, y],
        &[
            w.k as u32,
            w.n as u32,
            rows as u32,
            bits as u32,
            group as u32,
        ],
        [(rows * w.k).div_ceil(256), 1, 1],
        256,
    );
}

/// Expert entries stay GPU-resident. `input_per_entry` distinguishes the
/// repeated input rows of gate/up from the routed activation rows of down.
#[cfg(test)]
pub(super) fn experts(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    ids: &Buffer,
    y: &Buffer,
    entries: usize,
    input_per_entry: bool,
) {
    experts_ordered(cmd, w, x, ids, y, entries, input_per_entry, None);
}

pub(super) fn experts_ordered(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    ids: &Buffer,
    y: &Buffer,
    entries: usize,
    input_per_entry: bool,
    order: Option<&Buffer>,
) {
    assert_eq!(w.ty, A4G32);
    assert!(w.n.is_multiple_of(512));
    let n = w.n / 512;
    assert!(entries > 0 && entries <= 10240 && entries.is_multiple_of(10));
    assert!(ids.len() >= entries * 4 && y.len() >= entries * n * 4);
    assert!(
        x.len()
            >= (if input_per_entry {
                entries
            } else {
                entries / 10
            }) * w.k
                * 4
    );
    let buffers = [&w.buffer, x, ids, y, order.unwrap_or(ids)];
    if let Some(order) = order {
        assert!(order.len() >= entries * 4);
    }
    // Same arithmetic, shared compressed loads. Keep sparse/decode and the
    // narrower down contraction on their established vector programs.
    let paired = order.is_some() && entries >= 640 && w.k == 2560 && n == 640;
    cmd.dispatch(
        if paired {
            "q4a_expert4_fast_pair"
        } else {
            match (
                w.k.is_multiple_of(512) && n >= 8 && n.is_multiple_of(8),
                order.is_some(),
            ) {
                (true, true) => "q4a_expert4_fast_ordered",
                (false, true) => "q4a_expert4_ordered",
                (true, false) => "q4a_expert4_fast",
                (false, false) => "q4a_expert4",
            }
        },
        &buffers,
        &[
            w.k as u32,
            n as u32,
            entries as u32,
            u32::from(input_per_entry),
            512,
        ],
        if paired {
            [n.div_ceil(16) * 4, entries.div_ceil(8), 1]
        } else if order.is_some() {
            [n.div_ceil(16) * 8, entries.div_ceil(8), 1]
        } else {
            [n.div_ceil(16), entries, 1]
        },
        128,
    );
}
