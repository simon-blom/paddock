//! Shared exact-GGUF projection dispatch for the native Metal model graphs.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_models::mapped::MappedGguf;

pub(crate) struct Weight {
    pub(crate) buffer: Buffer,
    pub(crate) ty: u32,
    pub(crate) k: usize,
    pub(crate) n: usize,
}
impl Weight {
    fn can_expand(&self, workspace: &Buffer, rows: usize) -> bool {
        rows >= 1024
            && self.n >= 1024
            && matches!(self.ty, 12 | 13 | 14 | 23)
            && workspace.len() / 2
                >= self.k.div_ceil(128) * 128 * rows.div_ceil(128) * 128 + self.k * self.n
    }
    fn can_split(&self, workspace: &Buffer, rows: usize) -> bool {
        matches!(self.ty, 12 | 13 | 14 | 23)
            && (5..=64).contains(&rows)
            && self.k >= 4096
            && self.k.is_multiple_of(1024)
            && (1024..=8192).contains(&self.n)
            && workspace.len() >= self.k * rows.div_ceil(128) * 128 * 2 + 4 * rows * self.n * 4
    }
    // The prepared activation prefix is immutable until every independent
    // consumer has finished. Q/K/V and gate/up share one padded conversion.
    // Long-K projections may reuse the disjoint workspace tail for F32 partials.
    pub(crate) fn linear_prepared(
        &self,
        cmd: &Commands<'_>,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        scale: f32,
    ) {
        if crate::iquant::is_iq(self.ty) {
            crate::iquant::linear(
                cmd,
                &self.buffer,
                input,
                output,
                &[
                    self.k as u32,
                    self.n as u32,
                    rows as u32,
                    self.ty,
                    scale.to_bits(),
                ],
                true,
            );
            return;
        }
        if self.can_split(input, rows) {
            // Four independent K ranges shorten the long contraction loop.
            // The existing allocation must cover both the padded activation
            // prefix and every F32 partial. Otherwise use the ordinary tile;
            // never grow the model's memory grant for this election.
            let tile = if rows <= 8 {
                8
            } else if rows <= 32 {
                32
            } else {
                64
            };
            let kernel = match tile {
                8 => "linear_ktile_split8",
                32 => "linear_ktile_split32",
                _ => "linear_ktile_split64",
            };
            let p = [
                self.k as u32,
                self.n as u32,
                rows as u32,
                self.ty,
                scale.to_bits(),
            ];
            cmd.dispatch(
                kernel,
                &[&self.buffer, input, input],
                &p,
                [
                    self.n.div_ceil(if tile <= 32 { 16 } else { 32 }),
                    rows.div_ceil(tile),
                    4,
                ],
                128,
            );
            cmd.dispatch(
                "linear_ktile_join",
                &[input, output],
                &p,
                [(rows * self.n).div_ceil(256), 1, 1],
                256,
            );
            return;
        }
        if self.can_expand(input, rows) {
            let params = [
                self.k as u32,
                self.n as u32,
                rows as u32,
                self.ty,
                scale.to_bits(),
            ];
            cmd.dispatch(
                "linear_kexpand",
                &[&self.buffer, input],
                &params,
                [(self.k * self.n).div_ceil(4 * 256), 1, 1],
                256,
            );
            cmd.dispatch(
                "linear_kexpanded128",
                &[input, output],
                &params,
                [self.n.div_ceil(64), rows.div_ceil(128), 1],
                128,
            );
            return;
        }
        let (kernel, tile) = if rows <= 32 || (rows <= 128 && self.n <= 1024) {
            ("linear_quant_tile32", 32)
        } else if rows <= 64 || (rows <= 128 && self.n <= 4096) {
            ("linear_quant_tile64", 64)
        } else {
            ("linear_quant_tile128", 128)
        };
        let kquant = matches!(self.ty, 12 | 13 | 14 | 23);
        let tile = if kquant && rows <= 8 {
            8
        } else if kquant && (65..=96).contains(&rows) {
            96
        } else {
            tile
        };
        let kernel = if kquant {
            match tile {
                8 => "linear_ktile8",
                32 => "linear_ktile32",
                64 => "linear_ktile64",
                96 => "linear_ktile96",
                _ => "linear_ktile128",
            }
        } else if self.ty == 8
            && self.k.is_multiple_of(128)
            && self.n.is_multiple_of(64)
            && rows.is_multiple_of(tile)
        {
            match tile {
                32 => "linear_quant_full32",
                64 => "linear_quant_full64",
                _ => "linear_quant_full128",
            }
        } else {
            kernel
        };
        cmd.dispatch(
            kernel,
            &[&self.buffer, input, output],
            &[
                self.k as u32,
                self.n as u32,
                rows as u32,
                self.ty,
                scale.to_bits(),
            ],
            [
                self.n.div_ceil(if kquant {
                    if tile <= 32 { 16 } else { 32 }
                } else {
                    64
                }),
                rows.div_ceil(tile),
                1,
            ],
            128,
        );
    }

    pub(crate) fn load(
        device: &MetalDevice,
        map: &MappedGguf,
        name: &str,
        dims: &[usize],
    ) -> Result<Self> {
        Self::load_inner(device, map, name, dims, false)
    }

    /// Explicit projection/gather opt-in for the new i-quant primitives.
    /// Generic model loaders also hand Weight buffers to format-specific
    /// embedding/fused kernels; allowing new types there would falsely admit
    /// unimplemented graphs. Keep this separate until a consumer is audited.
    #[allow(dead_code)] // intentionally exercised only by prerequisite gates until the graph lands
    pub(crate) fn load_iq(
        device: &MetalDevice,
        map: &MappedGguf,
        name: &str,
        dims: &[usize],
    ) -> Result<Self> {
        Self::load_inner(device, map, name, dims, true)
    }

    fn load_inner(
        device: &MetalDevice,
        map: &MappedGguf,
        name: &str,
        dims: &[usize],
        iq_only: bool,
    ) -> Result<Self> {
        let (t, bytes) = map
            .tensor_bytes(name)
            .map_err(|e| MetalError::Model(e.to_string()))?;
        if t.dims.iter().map(|x| *x as usize).collect::<Vec<_>>() != dims {
            return Err(MetalError::Model(format!(
                "{name}: shape {:?}, expected {dims:?}",
                t.dims
            )));
        }
        let supported = if iq_only {
            crate::iquant::is_iq(t.raw_type)
        } else {
            matches!(t.raw_type, 0 | 1 | 8 | 12 | 13 | 14 | 23 | 30)
        };
        if !supported {
            return Err(MetalError::Model(format!(
                "{name}: unsupported weight type {:?}",
                t.ggml_type
            )));
        }
        if crate::iquant::is_iq(t.raw_type) {
            crate::iquant::validate(t.raw_type, dims, bytes.len())?;
        }
        if t.raw_type == 8 && !dims[0].is_multiple_of(32) {
            return Err(MetalError::Model(format!(
                "{name}: Q8 row is not block aligned"
            )));
        }
        if matches!(t.raw_type, 12 | 13 | 14 | 23) && !dims[0].is_multiple_of(256) {
            return Err(MetalError::Model(format!(
                "{name}: K-quant row is not block aligned"
            )));
        }
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| MetalError::Model(format!("{name}: shape overflow")))?;
        let expected = match t.raw_type {
            8 => (elements / 32).checked_mul(34),
            12 => (elements / 256).checked_mul(144),
            13 => (elements / 256).checked_mul(176),
            14 => (elements / 256).checked_mul(210),
            20 => (elements / 32).checked_mul(18),
            21 => (elements / 256).checked_mul(110),
            22 => (elements / 256).checked_mul(82),
            23 => (elements / 256).checked_mul(136),
            0 => elements.checked_mul(4),
            _ => elements.checked_mul(2),
        };
        if expected != Some(bytes.len()) {
            return Err(MetalError::Model(format!(
                "{name}: tensor byte count mismatch"
            )));
        }
        Ok(Self {
            buffer: device.upload(bytes)?,
            ty: t.raw_type,
            k: dims[0],
            n: *dims.get(1).unwrap_or(&1),
        })
    }
    pub(crate) fn linear(
        &self,
        cmd: &Commands<'_>,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        scale: f32,
        gemm_input: &Buffer,
    ) {
        if self.ty == crate::splash::PACKED4 {
            assert_eq!(scale, 1.0);
            return crate::splash::project(cmd, &[(self, output)], input, rows, gemm_input);
        }
        if self.ty == crate::affine::AFFINE4 {
            assert_eq!(scale, 1.0, "affine projection requires unit scale");
            return crate::affine::project(cmd, &[(self, output)], input, rows, gemm_input);
        }
        let p = [
            self.k as u32,
            self.n as u32,
            rows as u32,
            self.ty,
            scale.to_bits(),
        ];
        if crate::iquant::is_iq(self.ty) {
            crate::iquant::linear(cmd, &self.buffer, input, output, &p, false);
            return;
        }
        // A ragged second SIMD tile reloads the whole matrix. The BM8 tensor
        // tile avoids the 5--7-row verification cliff as well as full blocks.
        if rows >= 16 || (rows >= 5 && matches!(self.ty, 12 | 13 | 14 | 23)) {
            cmd.dispatch(
                "linear_input_padded",
                &[input, gemm_input],
                &p,
                [
                    (self.k.div_ceil(128) * 128 * rows.div_ceil(128) * 128).div_ceil(256),
                    1,
                    1,
                ],
                256,
            );
            self.linear_prepared(cmd, gemm_input, output, rows, scale);
        } else {
            if matches!(self.ty, 12 | 13 | 14 | 23) {
                let (kernel, tile) = match rows {
                    1 => ("linear_kquant1", 1),
                    2 => ("linear_kquant2", 2),
                    3 => ("linear_kquant3", 3),
                    4 => ("linear_kquant4", 4),
                    _ => ("linear_kquant_tail4", 4),
                };
                cmd.dispatch(
                    kernel,
                    &[&self.buffer, input, output],
                    &p,
                    [
                        self.n.div_ceil(if rows == 1 { 4 } else { 16 }),
                        rows.div_ceil(tile),
                        1,
                    ],
                    128,
                );
                return;
            }
            let tile = match rows {
                1..=4 => rows,
                5..=8 => 8,
                _ => 16,
            };
            let input = if self.ty == 8 && (2..=4).contains(&rows) {
                cmd.dispatch(
                    "linear_input",
                    &[input, gemm_input],
                    &p,
                    [(self.k * rows).div_ceil(256), 1, 1],
                    256,
                );
                gemm_input
            } else {
                input
            };
            let kernel = match (self.ty, tile) {
                (8, 1) => "linear_q8_r1",
                (8, 2) => "linear_q8_h2",
                (8, 3) => "linear_q8_h3",
                (8, 4) => "linear_q8_h4",
                (8, 8) => "linear_q8_r8",
                (8, _) => "linear_q8_r16",
                _ => "linear",
            };
            cmd.dispatch(
                kernel,
                &[&self.buffer, input, output],
                &p,
                [
                    self.n.div_ceil(4),
                    if self.ty == 8 {
                        rows.div_ceil(tile)
                    } else {
                        rows
                    },
                    1,
                ],
                128,
            );
        }
    }
}

// Model graphs opt in after shape-specific qualification; the default ladder
// remains unchanged.
// Both ordinary narrow decode and target verification consume the same F32
// operands here, including odd output-column tails and mixed Q4/Q5/Q6 planes.
pub(crate) fn paired_projections(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    m: usize,
) {
    debug_assert!((3..=4).contains(&m) && planes.iter().all(|(w, _)| matches!(w.ty, 12..=14)));
    let k = planes[0].0.k;
    assert!(planes.iter().all(|(w, _)| w.k == k));
    if planes.len() == 1 {
        let (w, out) = planes[0];
        cmd.dispatch(
            if m == 3 { "gemma_pair3" } else { "gemma_pair4" },
            &[&w.buffer, input, out],
            &[k as u32, w.n as u32, m as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(32), 1, 1],
            128,
        );
    } else {
        assert!(matches!(planes.len(), 2 | 3));
        let third = planes.get(2).unwrap_or(&planes[1]);
        cmd.dispatch(
            if m == 3 {
                "gemma_multi_pair3"
            } else {
                "gemma_multi_pair4"
            },
            &[
                &planes[0].0.buffer,
                &planes[1].0.buffer,
                &third.0.buffer,
                input,
                planes[0].1,
                planes[1].1,
                third.1,
            ],
            &[
                k as u32,
                planes[0].0.n as u32,
                planes[1].0.n as u32,
                if planes.len() == 3 {
                    third.0.n as u32
                } else {
                    0
                },
                m as u32,
                planes[0].0.ty,
                planes[1].0.ty,
                third.0.ty,
            ],
            [planes.iter().map(|(w, _)| w.n.div_ceil(32)).sum(), 1, 1],
            128,
        );
    }
}

/// Independent projections can share a launch without introducing a dependency
/// or moving the original GGUF matrices. Full decode tiles have no row-tail
/// checks in their contraction loop; other shapes retain the ragged path.
pub(crate) fn projections(
    cmd: &Commands<'_>,
    weights: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    gemm_input: &Buffer,
) {
    assert!(!weights.is_empty());
    if weights.iter().all(|(w, _)| w.ty == crate::splash::PACKED4) {
        return crate::splash::project(cmd, weights, input, rows, gemm_input);
    }
    if weights.iter().all(|(w, _)| w.ty == crate::affine::AFFINE4) {
        return crate::affine::project(cmd, weights, input, rows, gemm_input);
    }
    if weights
        .iter()
        .any(|(w, _)| matches!(w.ty, crate::splash::PACKED4 | crate::affine::AFFINE4))
    {
        for &(weight, output) in weights {
            weight.linear(cmd, input, output, rows, 1.0, gemm_input);
        }
        return;
    }
    // I-quant direct tiles retain F32 operands across the narrow/wide-row
    // boundary. Do not introduce the shared helper's F16 conversion here.
    if weights.iter().any(|(w, _)| crate::iquant::is_iq(w.ty)) {
        for &(weight, output) in weights {
            weight.linear(cmd, input, output, rows, 1.0, gemm_input);
        }
        return;
    }
    if rows >= 16
        || (rows >= 5
            && weights
                .iter()
                .all(|(w, _)| matches!(w.ty, 12 | 13 | 14 | 23)))
    {
        let k = weights[0].0.k;
        assert!(weights.iter().all(|(w, _)| w.k == k));
        cmd.dispatch(
            "linear_input_padded",
            &[input, gemm_input],
            &[k as u32, 0, rows as u32],
            [
                (k.div_ceil(128) * 128 * rows.div_ceil(128) * 128).div_ceil(256),
                1,
                1,
            ],
            256,
        );
        if weights.iter().all(|(w, _)| w.can_expand(gemm_input, rows)) {
            for &(weight, output) in weights {
                weight.linear_prepared(cmd, gemm_input, output, rows, 1.0);
            }
        } else if weights
            .iter()
            .all(|(w, _)| matches!(w.ty, 8 | 12 | 13 | 14 | 23))
        {
            assert!(matches!(weights.len(), 2 | 3));
            let third = weights.get(2).unwrap_or(&weights[1]);
            let n3 = if weights.len() == 3 { third.0.n } else { 0 };
            let (kernel, tile) = if rows <= 32 {
                ("linear_multi_quant32", 32)
            } else if rows <= 64 {
                ("linear_multi_quant64", 64)
            } else {
                ("linear_multi_quant128", 128)
            };
            let kquant = weights
                .iter()
                .all(|(w, _)| matches!(w.ty, 12 | 13 | 14 | 23));
            let tile = if kquant && rows <= 8 {
                8
            } else if kquant && (65..=96).contains(&rows) {
                96
            } else {
                tile
            };
            let kernel = if kquant {
                match tile {
                    8 => "linear_multi_ktile8",
                    32 => "linear_multi_ktile32",
                    64 => "linear_multi_ktile64",
                    96 => "linear_multi_ktile96",
                    _ => "linear_multi_ktile128",
                }
            } else {
                kernel
            };
            cmd.dispatch(
                kernel,
                &[
                    &weights[0].0.buffer,
                    &weights[1].0.buffer,
                    &third.0.buffer,
                    gemm_input,
                    weights[0].1,
                    weights[1].1,
                    third.1,
                ],
                &[
                    k as u32,
                    weights[0].0.n as u32,
                    weights[1].0.n as u32,
                    n3 as u32,
                    rows as u32,
                    weights[0].0.ty,
                    weights[1].0.ty,
                    third.0.ty,
                ],
                [
                    weights
                        .iter()
                        .map(|(w, _)| {
                            w.n.div_ceil(if kquant {
                                if tile <= 32 { 16 } else { 32 }
                            } else {
                                64
                            })
                        })
                        .sum(),
                    rows.div_ceil(tile),
                    1,
                ],
                128,
            );
        } else {
            for &(weight, output) in weights {
                weight.linear_prepared(cmd, gemm_input, output, rows, 1.0);
            }
        }
    } else if (1..=4).contains(&rows)
        && weights
            .iter()
            .all(|(w, _)| matches!(w.ty, 12 | 13 | 14 | 23))
    {
        assert!(matches!(weights.len(), 2 | 3));
        let k = weights[0].0.k;
        assert!(weights.iter().all(|(w, _)| w.k == k));
        let third = weights.get(2).unwrap_or(&weights[1]);
        let n3 = if weights.len() == 3 { third.0.n } else { 0 };
        let kernel = match rows {
            1 => "linear_multi_kquant1",
            2 => "linear_multi_kquant2",
            3 => "linear_multi_kquant3",
            _ => "linear_multi_kquant4",
        };
        cmd.dispatch(
            kernel,
            &[
                &weights[0].0.buffer,
                &weights[1].0.buffer,
                &third.0.buffer,
                input,
                weights[0].1,
                weights[1].1,
                third.1,
            ],
            &[
                k as u32,
                weights[0].0.n as u32,
                weights[1].0.n as u32,
                n3 as u32,
                rows as u32,
                weights[0].0.ty,
                weights[1].0.ty,
                third.0.ty,
            ],
            [
                weights
                    .iter()
                    .map(|(w, _)| w.n.div_ceil(if rows == 1 { 4 } else { 16 }))
                    .sum(),
                1,
                1,
            ],
            128,
        );
    } else if (1..=4).contains(&rows) && weights.iter().all(|(w, _)| w.ty == 8) {
        assert!(matches!(weights.len(), 2 | 3));
        let k = weights[0].0.k;
        assert!(weights.iter().all(|(w, _)| w.k == k));
        let third = weights.get(2).unwrap_or(&weights[1]);
        let n3 = if weights.len() == 3 { third.0.n } else { 0 };
        let p = [
            k as u32,
            weights[0].0.n as u32,
            weights[1].0.n as u32,
            n3 as u32,
            rows as u32,
        ];
        let input = if rows > 1 {
            cmd.dispatch(
                "linear_input",
                &[input, gemm_input],
                &[k as u32, 0, rows as u32],
                [(k * rows).div_ceil(256), 1, 1],
                256,
            );
            gemm_input
        } else {
            input
        };
        cmd.dispatch(
            match rows {
                1 => "linear_multi_q8_r1",
                2 => "linear_multi_q8_r2",
                3 => "linear_multi_q8_r3",
                _ => "linear_multi_q8_r4",
            },
            &[
                &weights[0].0.buffer,
                &weights[1].0.buffer,
                &third.0.buffer,
                input,
                weights[0].1,
                weights[1].1,
                third.1,
            ],
            &p,
            [(weights[0].0.n + weights[1].0.n + n3).div_ceil(4), 1, 1],
            128,
        );
    } else {
        for &(weight, output) in weights {
            weight.linear(cmd, input, output, rows, 1.0, gemm_input);
        }
    }
}
