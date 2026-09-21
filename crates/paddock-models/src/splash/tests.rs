use super::*;

#[test]
fn headers_fail_closed() {
    let mut bytes = vec![0; ALIGN];
    bytes[..8].copy_from_slice(b"MDFL0006");
    bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
    validate_header(&bytes, b"MDFL0006", 3, 1).unwrap();
    for n in [0, 8, 15, 16, ALIGN - 1] {
        assert!(validate_header(&bytes[..n], b"MDFL0006", 3, 1).is_err());
    }
    assert!(validate_header(&bytes, b"MDFL0005", 3, 1).is_err());
    assert!(validate_header(&bytes, b"MDFL0006", 2, 1).is_err());
    assert!(validate_header(&bytes, b"MDFL0006", 3, 0).is_err());
    assert!(align(usize::MAX).is_err());
}

#[test]
fn tiled_affine_slices_preserve_all_codes_scales_and_biases() {
    // Cross both storage tiles, two quant groups, and the unaligned 48-row
    // alpha/beta slices of a fused projection. Values are not dequantized.
    let (k, n) = (128, 512);
    let mut mmap = memmap2::MmapMut::map_anon(k * n / 16 * 9).unwrap();
    for i in 0..mmap.len() {
        mmap[i] = (i.wrapping_mul(131).wrapping_add(i / 7) % 256) as u8;
    }
    let t = Tensor {
        map: Arc::new(mmap.make_read_only().unwrap()),
        dims: vec![k, n],
        layout: Layout::Affine {
            codes: 0..k * n / 2,
            scales: k * n / 2..k * n / 2 + k * n / 32,
            biases: k * n / 2 + k * n / 32..k * n / 16 * 9,
            stored_n: n,
            first: 0,
            tiled: true,
        },
    };
    for (first, count) in [(0, 512), (240, 48), (464, 48)] {
        let slice = rows(&t, first, count).unwrap();
        let mut out = vec![0; slice.output_bytes()];
        slice.copy_native(&mut out).unwrap();
        for row in 0..count {
            for g in 0..2 {
                let r = first + row;
                let src = (r / 256 * 2 + g) * 256 + r % 256;
                let dst = row * 2 + g;
                assert_eq!(
                    &out[dst * 32..dst * 32 + 32],
                    &t.map[src * 32..src * 32 + 32]
                );
                for plane in 0..2 {
                    assert_eq!(
                        &out[k * count / 2 + plane * k * count / 32 + dst * 2
                            ..k * count / 2 + plane * k * count / 32 + dst * 2 + 2],
                        &t.map[k * n / 2 + plane * k * n / 32 + src * 2
                            ..k * n / 2 + plane * k * n / 32 + src * 2 + 2]
                    );
                }
            }
        }
        assert!(slice.copy_native(&mut []).is_err());
        let padded = count.div_ceil(256) * 256;
        let mut tiled = memmap2::MmapMut::map_anon(slice.tiled_bytes()).unwrap();
        slice.copy_tiled(&mut tiled).unwrap();
        assert!(slice.copy_tiled(&mut []).is_err());
        let reconstructed = Tensor {
            map: Arc::new(tiled.make_read_only().unwrap()),
            dims: vec![k, count],
            layout: Layout::Affine {
                codes: 0..k * padded / 2,
                scales: k * padded / 2..k * padded / 2 + k * padded / 32,
                biases: k * padded / 2 + k * padded / 32..k * padded / 16 * 9,
                stored_n: padded,
                first: 0,
                tiled: true,
            },
        };
        let mut roundtrip = vec![0; out.len()];
        reconstructed.copy_native(&mut roundtrip).unwrap();
        assert_eq!(
            out, roundtrip,
            "tiled kernel storage changed the quantized values"
        );
        for row in count..padded {
            for g in 0..2 {
                let dst = (row / 256 * 2 + g) * 256 + row % 256;
                assert!(
                    reconstructed.map[dst * 32..dst * 32 + 32]
                        .iter()
                        .all(|&v| v == 0)
                );
            }
        }
    }
    assert!(rows(&t, 500, 48).is_err());
    assert!(rows(&t, usize::MAX, 2).is_err());
}

#[test]
fn sections_reject_truncation_trailing_bytes_and_zero() {
    let map = Arc::new(
        memmap2::MmapMut::map_anon(3 * ALIGN)
            .unwrap()
            .make_read_only()
            .unwrap(),
    );
    let mut file = Packed {
        map: map.clone(),
        offset: 16,
    };
    assert_eq!(file.section(2).unwrap(), ALIGN..ALIGN + 2);
    assert!(file.section(0).is_err());
    assert_eq!(file.section(2).unwrap(), 2 * ALIGN..2 * ALIGN + 2);
    assert!(file.section(1).is_err());
    file.finish().unwrap();
    assert!(Packed { map, offset: 16 }.finish().is_err());
}

#[test]
fn unknown_manifest_is_not_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("manifest.json"), b"{\"artifacts\":[]}").unwrap();
    assert!(inventory(dir.path()).is_err());
}

#[test]
fn artifact_hash_and_size_are_checked_before_sections_are_exposed() {
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = vec![0; ALIGN];
    bytes[..8].copy_from_slice(b"MDFL0006");
    let a = Artifact {
        path: "layer.bin".into(),
        size: ALIGN as u64,
        sha256: sha(&bytes),
    };
    std::fs::write(dir.path().join(&a.path), &bytes).unwrap();
    Packed::open(dir.path(), &a, b"MDFL0006", 0, 0).unwrap();
    bytes[128] = 1;
    std::fs::write(dir.path().join(&a.path), &bytes).unwrap();
    assert!(Packed::open(dir.path(), &a, b"MDFL0006", 0, 0).is_err());
    std::fs::write(dir.path().join(&a.path), &bytes[..16]).unwrap();
    assert!(Packed::open(dir.path(), &a, b"MDFL0006", 0, 0).is_err());
}

#[test]
#[ignore = "requires PADDOCK_SPLASH_MODEL and PADDOCK_METAL_MLX_MODEL; full target byte audit"]
fn packed_target_matches_upstream_checkpoint() {
    use crate::safetensors::{ShardedSafetensors, qwen35_hf_name};
    let packed = Target::open(Path::new(&std::env::var("PADDOCK_SPLASH_MODEL").unwrap())).unwrap();
    let mlx = ShardedSafetensors::open_dir(Path::new(
        &std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap(),
    ))
    .unwrap();
    for (name, t) in &packed.tensors {
        let hf = match name.as_str() {
            "token_embd.weight" => "language_model.model.embed_tokens.weight".into(),
            "output_norm.weight" => "language_model.model.norm.weight".into(),
            "output.weight" => "language_model.lm_head.weight".into(),
            _ => qwen35_hf_name(name).unwrap().replacen(
                "model.language_model.",
                "language_model.model.",
                1,
            ),
        };
        let mut native = vec![0; t.output_bytes()];
        t.copy_native(&mut native).unwrap();
        if t.affine() {
            let base = hf.strip_suffix(".weight").unwrap();
            let mut offset = 0;
            for key in [
                hf.clone(),
                format!("{base}.scales"),
                format!("{base}.biases"),
            ] {
                let (_, data) = mlx.bytes(&key).unwrap();
                assert!(
                    native[offset..offset + data.len()] == *data,
                    "packed tensor differs: {key}"
                );
                offset += data.len();
            }
            assert_eq!(offset, native.len());
        } else {
            let (_, data) = mlx.bytes(&hf).unwrap();
            let (halves, _) = data.as_chunks::<2>();
            let (words, _) = native.as_chunks::<4>();
            for (s, d) in halves.iter().zip(words) {
                let mut want = f32::from_bits((u16::from_le_bytes(*s) as u32) << 16);
                let got = f32::from_le_bytes(*d);
                if name.ends_with("ssm_a") {
                    want = -want.exp();
                }
                assert!(
                    (want - got).abs() <= 1e-5 * want.abs().max(1.),
                    "{name}: {want} != {got}"
                );
            }
        }
    }
}
