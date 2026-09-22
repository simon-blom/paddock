//! Rank-local, pre-upload matrix slices for GGUF and plain Safetensors.
//!
//! Both formats use the logical `[input, output]` convention here: GGUF
//! stores that order, while Safetensors stores `[output, input]`. This module
//! handles bytes and geometry only; model-specific choices of which weights
//! to split belong to the later TP layer work.
//!
//! Adapted from the checked row/column block-selection ideas in Erik Bogado's
//! ErikBPF/paddock `contrib/tp-05-shard-loaders` (895f8db), with per-tensor
//! layout selection informed by `contrib/tp-12-mixed-types` (d856695).
//! The fork's in-process GPU transfer and model execution are not used.

use std::borrow::Cow;

use crate::ggml_type::GgmlType;
use crate::mapped::{MapError, MappedGguf};
use crate::safetensors::{ShardedSafetensors, StDtype};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardKind {
    Replicated,
    /// Contiguous output rows; a column-parallel projection.
    OutputRows,
    /// Input columns gathered block-by-block from every output row;
    /// a row-parallel projection.
    InputColumns,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorSliceRequest {
    pub kind: ShardKind,
    pub rank: usize,
    pub world_size: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum TensorSliceError {
    #[error(transparent)]
    Gguf(#[from] MapError),
    #[error("tensor {0} is absent from Safetensors")]
    Missing(String),
    #[error("tensor {name}: {reason}")]
    Invalid { name: String, reason: String },
}

/// Local dimensions are *always* `[input, output]`, regardless of source
/// format. Whole output rows borrow the mmap; input columns own only the
/// selected blocks (never an uploaded full tensor).
pub struct TensorShard<'a> {
    pub bytes: Cow<'a, [u8]>,
    pub dims: [usize; 2],
}

fn invalid(name: &str, reason: impl Into<String>) -> TensorSliceError {
    TensorSliceError::Invalid {
        name: name.to_owned(),
        reason: reason.into(),
    }
}

fn slice_matrix<'a>(
    name: &str,
    bytes: &'a [u8],
    dims: [usize; 2],
    (block_elems, block_bytes): (usize, usize),
    request: TensorSliceRequest,
) -> Result<TensorShard<'a>, TensorSliceError> {
    let [input, output] = dims;
    if !matches!(request.world_size, 1 | 2) || request.rank >= request.world_size {
        return Err(invalid(name, "expected TP=1/2 and a rank inside the group"));
    }
    if input == 0 || output == 0 || block_elems == 0 || block_bytes == 0 {
        return Err(invalid(name, "empty matrix or unknown block layout"));
    }
    // Blocks may not wrap across rows: a whole tensor's byte size alone does
    // not establish that each row is independently splittable.
    if !input.is_multiple_of(block_elems) {
        return Err(invalid(
            name,
            format!("input width {input} is not aligned to {block_elems}"),
        ));
    }
    let stride = (input / block_elems)
        .checked_mul(block_bytes)
        .ok_or_else(|| invalid(name, "row byte stride overflow"))?;
    if stride.checked_mul(output) != Some(bytes.len()) {
        return Err(invalid(
            name,
            "source byte length does not match matrix layout",
        ));
    }
    let axis = match request.kind {
        ShardKind::Replicated => {
            return Ok(TensorShard {
                bytes: Cow::Borrowed(bytes),
                dims,
            });
        }
        ShardKind::OutputRows => output,
        ShardKind::InputColumns => input,
    };
    if !axis.is_multiple_of(request.world_size) {
        return Err(invalid(
            name,
            format!(
                "axis {axis} cannot split evenly over {} ranks",
                request.world_size
            ),
        ));
    }
    let local = axis / request.world_size;
    if local == 0 {
        return Err(invalid(name, "empty rank-local slice"));
    }
    // rank < world_size and local * world_size == axis: no overflow here.
    let start = request.rank * local;
    let end = start + local;
    match request.kind {
        ShardKind::OutputRows => Ok(TensorShard {
            bytes: Cow::Borrowed(&bytes[start * stride..end * stride]),
            dims: [input, local],
        }),
        ShardKind::InputColumns => {
            if !start.is_multiple_of(block_elems) || !end.is_multiple_of(block_elems) {
                return Err(invalid(
                    name,
                    format!("input shard {start}..{end} cuts {block_elems}-element blocks"),
                ));
            }
            let begin = (start / block_elems) * block_bytes;
            let finish = (end / block_elems) * block_bytes;
            let per_row = finish - begin;
            let total = per_row
                .checked_mul(output)
                .ok_or_else(|| invalid(name, "local byte length overflow"))?;
            let mut selected = Vec::with_capacity(total);
            for row in bytes.chunks_exact(stride) {
                selected.extend_from_slice(&row[begin..finish]);
            }
            Ok(TensorShard {
                bytes: Cow::Owned(selected),
                dims: [local, output],
            })
        }
        ShardKind::Replicated => unreachable!(),
    }
}

/// GGUF matrix adapter: pick each tensor's own block layout, never a model-
/// wide quantization assumption. No decoded or GPU-resident full copy is made.
pub fn gguf_shard<'a>(
    map: &'a MappedGguf,
    name: &str,
    request: TensorSliceRequest,
) -> Result<(GgmlType, TensorShard<'a>), TensorSliceError> {
    let (info, bytes) = map.tensor_bytes(name)?;
    if info.dims.len() != 2 {
        return Err(invalid(name, "expected a 2-D GGUF matrix"));
    }
    let dims = [
        usize::try_from(info.dims[0]).map_err(|_| invalid(name, "input dimension overflow"))?,
        usize::try_from(info.dims[1]).map_err(|_| invalid(name, "output dimension overflow"))?,
    ];
    let layout = info.ggml_type.block_layout().ok_or_else(|| {
        invalid(
            name,
            format!("unsupported GGUF layout {:?}", info.ggml_type),
        )
    })?;
    Ok((
        info.ggml_type,
        slice_matrix(name, bytes, dims, layout, request)?,
    ))
}

/// Plain Safetensors matrix adapter. F8 with companion scales, NVFP4/U8,
/// MXFP4 and other composite/packed planes require a format-specific adapter
/// that slices payload and scales together; never treat them as plain bytes.
pub fn safetensors_shard<'a>(
    st: &'a ShardedSafetensors,
    name: &str,
    request: TensorSliceRequest,
) -> Result<(StDtype, TensorShard<'a>), TensorSliceError> {
    let (info, bytes) = st
        .bytes(name)
        .ok_or_else(|| TensorSliceError::Missing(name.to_owned()))?;
    if info.shape.len() != 2 {
        return Err(invalid(name, "expected a 2-D Safetensors matrix"));
    }
    if !matches!(info.dtype, StDtype::Bf16 | StDtype::F16 | StDtype::F32) {
        return Err(invalid(
            name,
            format!(
                "Safetensors {:?} needs a format-specific slice adapter",
                info.dtype
            ),
        ));
    }
    let dims = [info.shape[1], info.shape[0]];
    Ok((
        info.dtype,
        slice_matrix(name, bytes, dims, (1, info.dtype.bytes()), request)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(kind: ShardKind, rank: usize) -> TensorSliceRequest {
        TensorSliceRequest {
            kind,
            rank,
            world_size: 2,
        }
    }

    #[test]
    fn fork_kquant_rows_and_columns_reconstruct() {
        // ErikBPF tp-05 row/column reconstruction oracle, extended to the
        // mixed types the pinned UD-Q4_K_M checkpoint actually contains.
        for ty in [
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q3K,
            GgmlType::Iq4Xs,
            GgmlType::Q8_0,
            GgmlType::Iq4Nl,
        ] {
            let (block, unit) = ty.block_layout().unwrap();
            let input = block * 4;
            let output = 6;
            let row_bytes = input / block * unit;
            let bytes: Vec<u8> = (0..output * row_bytes).map(|n| (n % 251) as u8).collect();
            let a = slice_matrix(
                "w",
                &bytes,
                [input, output],
                (block, unit),
                req(ShardKind::OutputRows, 0),
            )
            .unwrap();
            let b = slice_matrix(
                "w",
                &bytes,
                [input, output],
                (block, unit),
                req(ShardKind::OutputRows, 1),
            )
            .unwrap();
            assert_eq!([a.bytes.as_ref(), b.bytes.as_ref()].concat(), bytes);
            assert!(matches!(a.bytes, Cow::Borrowed(_)));
            let a = slice_matrix(
                "w",
                &bytes,
                [input, output],
                (block, unit),
                req(ShardKind::InputColumns, 0),
            )
            .unwrap();
            let b = slice_matrix(
                "w",
                &bytes,
                [input, output],
                (block, unit),
                req(ShardKind::InputColumns, 1),
            )
            .unwrap();
            assert_eq!(a.dims, [input / 2, output]);
            assert!(matches!(a.bytes, Cow::Owned(_)));
            let rejoined: Vec<u8> = (0..output)
                .flat_map(|r| {
                    a.bytes[r * row_bytes / 2..(r + 1) * row_bytes / 2]
                        .iter()
                        .chain(&b.bytes[r * row_bytes / 2..(r + 1) * row_bytes / 2])
                        .copied()
                })
                .collect();
            assert_eq!(rejoined, bytes, "{ty:?}");
        }
    }

    #[test]
    fn rejects_bad_geometry_before_any_upload() {
        let source = vec![0u8; 4 * 144];
        let layout = (256, 144);
        let odd_rows = vec![0u8; 3 * 2 * 144];
        for bad in [
            slice_matrix(
                "w",
                &source,
                [256, 4],
                layout,
                req(ShardKind::InputColumns, 0),
            ),
            slice_matrix(
                "w",
                &source[..575],
                [512, 2],
                layout,
                req(ShardKind::OutputRows, 0),
            ),
            slice_matrix(
                "w",
                &source,
                [511, 2],
                layout,
                req(ShardKind::OutputRows, 0),
            ),
            slice_matrix(
                "w",
                &source,
                [512, 3],
                layout,
                req(ShardKind::OutputRows, 0),
            ),
            slice_matrix(
                "w",
                &odd_rows,
                [512, 3],
                layout,
                req(ShardKind::OutputRows, 0),
            ),
            slice_matrix(
                "w",
                &source,
                [usize::MAX - 255, 2],
                layout,
                req(ShardKind::OutputRows, 0),
            ),
            slice_matrix(
                "w",
                &source,
                [512, 2],
                layout,
                TensorSliceRequest {
                    kind: ShardKind::Replicated,
                    rank: 2,
                    world_size: 2,
                },
            ),
        ] {
            assert!(bad.is_err());
        }
    }

    #[test]
    fn plain_matrix_byte_reconstruction_and_tp1() {
        for unit in [2, 4] {
            let bytes: Vec<u8> = (0..4 * 6 * unit).map(|n| n as u8).collect();
            let a = slice_matrix(
                "st",
                &bytes,
                [6, 4],
                (1, unit),
                req(ShardKind::InputColumns, 0),
            )
            .unwrap();
            let b = slice_matrix(
                "st",
                &bytes,
                [6, 4],
                (1, unit),
                req(ShardKind::InputColumns, 1),
            )
            .unwrap();
            let joined: Vec<u8> = (0..4)
                .flat_map(|r| {
                    a.bytes[r * 3 * unit..(r + 1) * 3 * unit]
                        .iter()
                        .chain(&b.bytes[r * 3 * unit..(r + 1) * 3 * unit])
                        .copied()
                })
                .collect();
            assert_eq!(joined, bytes);
            let full = slice_matrix(
                "st",
                &bytes,
                [6, 4],
                (1, unit),
                TensorSliceRequest {
                    kind: ShardKind::Replicated,
                    rank: 0,
                    world_size: 1,
                },
            )
            .unwrap();
            assert_eq!(full.bytes.as_ref(), bytes);
            assert!(matches!(full.bytes, Cow::Borrowed(_)));
        }
    }

    #[test]
    fn safetensors_file_adapter_slices_before_upload() {
        let dir = tempfile::tempdir().unwrap();
        let payload: Vec<u8> = (0..32).collect(); // BF16, [out=4, in=4]
        let header = serde_json::json!({
            "weight": {"dtype":"BF16", "shape":[4,4], "data_offsets":[0,32]},
            "packed": {"dtype":"U8", "shape":[4,4], "data_offsets":[32,48]}
        })
        .to_string();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&payload);
        file.extend_from_slice(&[0; 16]);
        std::fs::write(dir.path().join("model.safetensors"), file).unwrap();
        let st = ShardedSafetensors::open_dir(dir.path()).unwrap();
        let (ty, rows) = safetensors_shard(&st, "weight", req(ShardKind::OutputRows, 0)).unwrap();
        assert_eq!(ty, StDtype::Bf16);
        assert_eq!(rows.dims, [4, 2]);
        assert_eq!(rows.bytes.as_ref(), &payload[..16]);
        assert!(matches!(rows.bytes, Cow::Borrowed(_)));
        let (_, columns) =
            safetensors_shard(&st, "weight", req(ShardKind::InputColumns, 1)).unwrap();
        assert_eq!(columns.dims, [2, 4]);
        assert_eq!(
            columns.bytes.as_ref(),
            &[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
        );
        assert!(safetensors_shard(&st, "packed", req(ShardKind::InputColumns, 0)).is_err());
        assert!(matches!(
            safetensors_shard(&st, "absent", req(ShardKind::Replicated, 0)),
            Err(TensorSliceError::Missing(_))
        ));
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.8-27B UD-Q4_K_M GGUF (PADDOCK_TP_TEST_GGUF)"]
    fn pinned_checkpoint_mixed_quant_shards() {
        let path = std::env::var("PADDOCK_TP_TEST_GGUF").expect("set PADDOCK_TP_TEST_GGUF");
        let map = MappedGguf::open(std::path::Path::new(&path)).unwrap();
        let mut covered = std::collections::HashSet::new();
        for info in map.tensor_infos() {
            if info.dims.len() != 2 || !info.name.starts_with("blk.0.") {
                continue;
            }
            let ty = info.ggml_type;
            if !matches!(
                ty,
                GgmlType::Q4K
                    | GgmlType::Q5K
                    | GgmlType::Q6K
                    | GgmlType::Q3K
                    | GgmlType::Iq4Xs
                    | GgmlType::Q8_0
            ) {
                continue;
            }
            if !covered.insert(ty.raw()) {
                continue;
            }
            let (_, original) = map.tensor_bytes(&info.name).unwrap();
            let (_, a) = gguf_shard(&map, &info.name, req(ShardKind::OutputRows, 0)).unwrap();
            let (_, b) = gguf_shard(&map, &info.name, req(ShardKind::OutputRows, 1)).unwrap();
            assert_eq!(
                [a.bytes.as_ref(), b.bytes.as_ref()].concat(),
                original,
                "{}",
                info.name
            );
            let (_, a) = gguf_shard(&map, &info.name, req(ShardKind::InputColumns, 0)).unwrap();
            let (_, b) = gguf_shard(&map, &info.name, req(ShardKind::InputColumns, 1)).unwrap();
            let stride = a.bytes.len() / a.dims[1];
            let source_stride = original.len() / a.dims[1];
            for row in 0..a.dims[1] {
                assert_eq!(
                    &a.bytes[row * stride..(row + 1) * stride],
                    &original[row * source_stride..row * source_stride + stride]
                );
                assert_eq!(
                    &b.bytes[row * stride..(row + 1) * stride],
                    &original[row * source_stride + stride..(row + 1) * source_stride]
                );
            }
        }
        assert!(covered.contains(&GgmlType::Q5K.raw()));
        assert!(covered.contains(&GgmlType::Q3K.raw()));
        assert!(covered.contains(&GgmlType::Iq4Xs.raw()));
        assert!(covered.len() >= 4, "observed mixed types: {covered:?}");
    }
}
