//! Full native Flash Next FFN: HC mix -> 512/top10 routed + shared experts
//! -> HC scatter. Scratch is reusable across all 48 layers; encoding never
//! allocates, reads routing to the host or submits a command buffer.
use super::residual::{self, HyperConnection, WIDTH, load_weight};
use super::{affine, mlx};
use crate::{
    device::{Buffer, Commands, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::mapped::MappedGguf;

const EXPERTS: usize = 512;
const ACTIVE: usize = 10;
const FF: usize = 640;

pub(super) struct Weights {
    router: Weight,
    shared_router: Option<Weight>,
    gate: Weight,
    up: Weight,
    down: Weight,
    shared_gate: Weight,
    shared_up: Weight,
    shared_down: Weight,
    hc: HyperConnection,
}
impl Weights {
    pub(super) fn load_mlx(d: &MetalDevice, s: &mlx::Source, layer: usize) -> Result<Self> {
        if layer >= 48 {
            return Err(MetalError::Model("invalid Flash Next MoE layer".into()));
        }
        let root = format!("{}.layers.{layer}", mlx::ROOT);
        let w = |suffix: &str| s.weight(d, &format!("{root}.mlp.{suffix}"));
        Ok(Self {
            router: w("gate")?,
            shared_router: Some(w("shared_expert_gate")?),
            gate: w("switch_mlp.gate_proj")?,
            up: w("switch_mlp.up_proj")?,
            down: w("switch_mlp.down_proj")?,
            shared_gate: w("shared_expert.gate_proj")?,
            shared_up: w("shared_expert.up_proj")?,
            shared_down: w("shared_expert.down_proj")?,
            hc: HyperConnection::load_mlx(d, s, &format!("{root}.mlp_hyper_connection"), true)?,
        })
    }
    pub(super) fn load(d: &MetalDevice, map: &MappedGguf, layer: usize) -> Result<Self> {
        if layer >= 48 {
            return Err(MetalError::Model("invalid Flash Next MoE layer".into()));
        }
        let w = |name: &str, dims: &[usize], types: &[u32]| {
            let name = format!("blk.{layer}.{name}.weight");
            let (info, bytes) = map
                .tensor_bytes(&name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            if !types.contains(&info.raw_type)
                || (info.raw_type == 0
                    && !bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|b| f32::from_le_bytes(*b).is_finite()))
            {
                return Err(MetalError::Model(format!("invalid Flash Next MoE {name}")));
            }
            load_weight(d, map, &name, dims, info.raw_type)
        };
        // Byte concatenation only: no cast/requantization or host model math.
        // One F32 projection covers 512 routed logits and the shared gate.
        let mut bytes = Vec::with_capacity((EXPERTS + 1) * WIDTH * 4);
        for (suffix, dims) in [
            ("ffn_gate_inp", vec![WIDTH as u64, EXPERTS as u64]),
            ("ffn_gate_inp_shexp", vec![WIDTH as u64]),
        ] {
            let name = format!("blk.{layer}.{suffix}.weight");
            let (info, raw) = map
                .tensor_bytes(&name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            if info.raw_type != 0
                || info.dims != dims
                || raw.len() != dims.iter().product::<u64>() as usize * 4
                || !raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
            {
                return Err(MetalError::Model(format!(
                    "invalid Flash Next router {name}"
                )));
            }
            bytes.extend_from_slice(raw);
        }
        let router = Weight {
            buffer: d.upload(&bytes)?,
            ty: 0,
            k: WIDTH,
            n: EXPERTS + 1,
        };
        let ty = if layer == 2 { 21 } else { 22 };
        Ok(Self {
            router,
            shared_router: None,
            gate: w("ffn_gate_exps", &[WIDTH, FF, EXPERTS], &[ty])?,
            up: w("ffn_up_exps", &[WIDTH, FF, EXPERTS], &[ty])?,
            down: w("ffn_down_exps", &[FF, WIDTH, EXPERTS], &[20])?,
            shared_gate: w("ffn_gate_shexp", &[WIDTH, FF], &[8, 14])?,
            shared_up: w("ffn_up_shexp", &[WIDTH, FF], &[8, 14])?,
            shared_down: w("ffn_down_shexp", &[FF, WIDTH], &[8])?,
            hc: HyperConnection::load(d, map, &format!("blk.{layer}.hc_ffn"), true)?,
        })
    }

    /// Caller must validate the whole batch before any stateful layer encodes
    /// and check invalid routing after completion before publishing results.
    pub(super) fn encode_ffn(
        &self,
        cmd: &Commands<'_>,
        h: &Buffer,
        hc: &residual::Workspace,
        s: &Workspace,
        rows: usize,
    ) -> Result<()> {
        s.validate(h, rows, true)?;
        s.validate(&hc.mixed, rows, false)?;
        self.hc.encode(cmd, h, hc, rows);
        self.encode(cmd, &hc.mixed, s, rows)?;
        self.hc.combine(cmd, h, &s.output, hc, rows);
        Ok(())
    }

    pub(super) fn encode(
        &self,
        cmd: &Commands<'_>,
        x: &Buffer,
        s: &Workspace,
        rows: usize,
    ) -> Result<()> {
        s.validate(x, rows, false)?;
        if let Some(shared_router) = &self.shared_router {
            self.encode_mlx(cmd, x, s, rows, shared_router);
            return Ok(());
        }
        if rows <= 8 {
            residual::project(cmd, &self.router, x, &s.logits, rows);
        } else {
            cmd.dispatch(
                "q4m_router_mm",
                &[&self.router.buffer, x, &s.logits],
                &[rows as u32],
                [513usize.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
        }
        cmd.dispatch(
            "iq_route512",
            &[&s.logits, &s.ids, &s.weights, &s.shared_scale, &s.invalid],
            &[rows as u32],
            [rows, 1, 1],
            32,
        );
        self.encode_experts(cmd, x, s, rows, rows > 8, false);
        for (w, y) in [
            (&self.shared_gate, &s.shared_gate),
            (&self.shared_up, &s.shared_up),
        ] {
            if w.ty == 14 {
                crate::projection::project(cmd, &[(w, y)], x, rows);
            } else {
                residual::project(cmd, w, x, y, rows);
            }
        }
        cmd.dispatch(
            "q4m_silu",
            &[&s.shared_gate, &s.shared_up, &s.shared_gate],
            &[(rows * FF) as u32],
            [(rows * FF).div_ceil(256), 1, 1],
            256,
        );
        residual::project(
            cmd,
            &self.shared_down,
            &s.shared_gate,
            &s.shared_output,
            rows,
        );
        cmd.dispatch(
            "q4m_fold",
            &[
                &s.down,
                &s.weights,
                &s.shared_output,
                &s.shared_scale,
                &s.invalid,
                &s.output,
            ],
            &[rows as u32],
            [(rows * WIDTH / 4).div_ceil(256), 1, 1],
            256,
        );
        Ok(())
    }

    fn encode_mlx(
        &self,
        cmd: &Commands<'_>,
        x: &Buffer,
        s: &Workspace,
        rows: usize,
        shared_router: &Weight,
    ) {
        affine::project(cmd, &self.router, x, &s.logits, rows);
        affine::project(cmd, shared_router, x, &s.shared_scale, rows);
        cmd.dispatch(
            "q4b_route",
            &[&s.logits, &s.ids, &s.weights, &s.shared_scale, &s.invalid],
            &[rows as u32],
            [rows, 1, 1],
            32,
        );
        let entries = rows * ACTIVE;
        let mut matrix_mask = [0u32; 32];
        if let Some(spans) = cmd.projection_rows() {
            for &(start, count, logical) in spans {
                // Match the checkpoint's sorted RHS matrix contraction at
                // B/E >= 4, anchored to the logical prompt chunk. A neighbour
                // must never switch a decoding row's arithmetic.
                if logical * ACTIVE >= EXPERTS * 4 {
                    for row in start..start + count {
                        matrix_mask[row / 32] |= 1 << (row % 32);
                    }
                }
            }
        }
        let matrix = matrix_mask.iter().any(|&v| v != 0);
        let vectors = (0..rows).any(|r| matrix_mask[r / 32] & (1 << (r % 32)) == 0);
        if matrix {
            cmd.dispatch(
                "moe_align",
                &[&s.ids, &s.lists, &s.counts],
                &[entries as u32],
                [EXPERTS, 1, 1],
                256,
            );
            cmd.dispatch(
                "iq_tiles512",
                &[&s.counts, &s.tiles],
                &[EXPERTS as u32],
                [1, 1, 1],
                512,
            );
        }
        let order = if !matrix && (64..=1280).contains(&entries) {
            cmd.dispatch(
                "q4a_expert_order",
                &[&s.ids, &s.lists],
                &[entries as u32],
                [1, 1, 1],
                256,
            );
            Some(&s.lists)
        } else {
            None
        };
        let project_experts = |w: &Weight, x: &Buffer, y: &Buffer, per_entry: bool| {
            if matrix {
                let n = w.n / EXPERTS;
                let mut p = [0u32; 37];
                p[..5].copy_from_slice(&[
                    w.k as u32,
                    n as u32,
                    entries as u32,
                    u32::from(per_entry),
                    512,
                ]);
                p[5..].copy_from_slice(&matrix_mask);
                cmd.dispatch(
                    "q4a_expert_mm_wide",
                    &[&w.buffer, x, &s.lists, &s.counts, &s.tiles, y],
                    &p,
                    [n.div_ceil(64), Workspace::tiles(rows), 1],
                    128,
                );
                if vectors {
                    cmd.dispatch(
                        "q4a_expert_vector_masked",
                        &[&w.buffer, x, &s.ids, y],
                        &p,
                        [n.div_ceil(16), entries, 1],
                        128,
                    );
                }
            } else {
                affine::experts_ordered(cmd, w, x, &s.ids, y, entries, per_entry, order);
            }
        };
        project_experts(&self.gate, x, &s.act, false);
        // The down-output allocation is dead until gate/up activation is
        // complete, and is larger than the routed up plane. Reuse its prefix.
        project_experts(&self.up, x, &s.down, false);
        cmd.dispatch(
            "mlx_swiglu",
            &[&s.act, &s.down],
            &[(entries * FF) as u32],
            [(entries * FF).div_ceil(256), 1, 1],
            256,
        );
        project_experts(&self.down, &s.act, &s.down, true);
        affine::project(cmd, &self.shared_gate, x, &s.shared_gate, rows);
        affine::project(cmd, &self.shared_up, x, &s.shared_up, rows);
        cmd.dispatch(
            "mlx_swiglu",
            &[&s.shared_gate, &s.shared_up],
            &[(rows * FF) as u32],
            [(rows * FF).div_ceil(256), 1, 1],
            256,
        );
        affine::project(
            cmd,
            &self.shared_down,
            &s.shared_gate,
            &s.shared_output,
            rows,
        );
        cmd.dispatch(
            "q4b_fold",
            &[
                &s.down,
                &s.weights,
                &s.shared_output,
                &s.shared_scale,
                &s.output,
            ],
            &[rows as u32],
            [(rows * WIDTH).div_ceil(256), 1, 1],
            256,
        );
    }

    // Internal test seam compares sparse SIMD / TensorOps / adaptive routes
    // on identical weights and activations. There is no serving env switch.
    fn encode_experts(
        &self,
        cmd: &Commands<'_>,
        x: &Buffer,
        s: &Workspace,
        rows: usize,
        grouped: bool,
        force_tensor: bool,
    ) {
        let entries = rows * ACTIVE;
        let p = [
            FF as u32,
            WIDTH as u32,
            entries as u32,
            1,
            EXPERTS as u32,
            u32::from(force_tensor),
        ];
        if grouped {
            cmd.dispatch(
                "moe_align",
                &[&s.ids, &s.lists, &s.counts],
                &[entries as u32],
                [EXPERTS, 1, 1],
                256,
            );
            cmd.dispatch(
                "iq_tiles512",
                &[&s.counts, &s.tiles],
                &[EXPERTS as u32],
                [1, 1, 1],
                512,
            );
            cmd.dispatch(
                if self.gate.ty == 21 {
                    "q4m_gu_mm21"
                } else {
                    "q4m_gu_mm22"
                },
                &[
                    &self.gate.buffer,
                    &self.up.buffer,
                    x,
                    &s.lists,
                    &s.counts,
                    &s.tiles,
                    &s.act,
                ],
                &[rows as u32, u32::from(force_tensor)],
                [FF / 32, Workspace::tiles(rows), 1],
                128,
            );
            cmd.dispatch(
                "q4m_down_grouped",
                &[
                    &self.down.buffer,
                    &s.act,
                    &s.lists,
                    &s.counts,
                    &s.tiles,
                    &s.down,
                ],
                &p,
                [WIDTH / 32, Workspace::tiles(rows), 1],
                128,
            );
        } else {
            cmd.dispatch(
                if self.gate.ty == 21 {
                    "q4m_gu_mv21"
                } else {
                    "q4m_gu_mv22"
                },
                &[&self.gate.buffer, &self.up.buffer, x, &s.ids, &s.act],
                &[rows as u32],
                [FF / 4, entries, 1],
                128,
            );
            cmd.dispatch(
                "iq_expert_mv20",
                &[&self.down.buffer, &s.act, &s.ids, &s.down],
                &p[..5],
                [WIDTH / 4, entries, 1],
                128,
            );
        }
    }
}

pub(super) struct Workspace {
    capacity: usize,
    pub(super) logits: Buffer,
    pub(super) ids: Buffer,
    pub(super) weights: Buffer,
    pub(super) shared_scale: Buffer,
    pub(super) invalid: Buffer,
    lists: Buffer,
    counts: Buffer,
    tiles: Buffer,
    act: Buffer,
    pub(super) down: Buffer,
    shared_gate: Buffer,
    shared_up: Buffer,
    pub(super) shared_output: Buffer,
    pub(super) output: Buffer,
}
impl Workspace {
    fn tiles(rows: usize) -> usize {
        (rows * ACTIVE).div_ceil(32) + EXPERTS
    }
    pub(super) fn bytes(rows: usize) -> Result<usize> {
        if !(1..=1024).contains(&rows) {
            return Err(MetalError::Model(
                "Flash Next MoE rows must be 1..=1024".into(),
            ));
        }
        Ok(4 * (rows
            * (513
                + 2 * ACTIVE
                + 2
                + EXPERTS * ACTIVE
                + ACTIVE * (FF + WIDTH)
                + 2 * FF
                + 2 * WIDTH)
            + EXPERTS
            + 1
            + 2 * Self::tiles(rows)))
    }
    pub(super) fn new(d: &MetalDevice, rows: usize) -> Result<Self> {
        Self::bytes(rows)?;
        let b = |n| d.alloc(n * 4);
        Ok(Self {
            capacity: rows,
            logits: b(rows * 513)?,
            ids: b(rows * ACTIVE)?,
            weights: b(rows * ACTIVE)?,
            shared_scale: b(rows)?,
            invalid: b(rows)?,
            lists: b(EXPERTS * rows * ACTIVE)?,
            counts: b(EXPERTS)?,
            tiles: b(1 + 2 * Self::tiles(rows))?,
            act: b(rows * ACTIVE * FF)?,
            down: b(rows * ACTIVE * WIDTH)?,
            shared_gate: b(rows * FF)?,
            shared_up: b(rows * FF)?,
            shared_output: b(rows * WIDTH)?,
            output: b(rows * WIDTH)?,
        })
    }
    fn validate(&self, x: &Buffer, rows: usize, hyper: bool) -> Result<()> {
        if rows == 0
            || rows > self.capacity
            || x.len() < rows * WIDTH * 4 * if hyper { 4 } else { 1 }
        {
            return Err(MetalError::Model(
                "Flash Next MoE invalid rows/input buffer".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
