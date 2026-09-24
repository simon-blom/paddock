//! Gemma's hybrid GEGLU FFN, not Qwen's sigmoid-gated shared expert.
//! Raw fused Q8 gate/up bytes remain compressed and resident. Reuse our
//! GPU compaction and bounded TensorOps contraction, never host routing.
use super::*;
use paddock_models::mapped::MappedGguf;

const WIDTH: usize = 2816;
const FF: usize = 704;
const EXPERTS: usize = 128;
const ACTIVE: usize = 8;

pub(super) fn validate(map: &MappedGguf, count: usize) -> Result<()> {
    for (key, value) in [
        ("expert_count", EXPERTS),
        ("expert_used_count", ACTIVE),
        ("expert_feed_forward_length", FF),
    ] {
        if map.gguf().arch_field(key).and_then(|v| v.as_u64()) != Some(value as u64) {
            return Err(MetalError::Model(format!(
                "Gemma MoE requires {key}={value}"
            )));
        }
    }
    // Reject unsupported exports before bulk uploads. This graph's supported
    // artifact is Q8, not every quant the shared dense ladder can decode.
    let diffusion = map.gguf().architecture() == Some("diffusion-gemma");
    for tensor in &map.gguf().tensors {
        if !matches!(tensor.raw_type, 0 | 8)
            && !(diffusion && matches!(tensor.raw_type, 6 | 12 | 14))
        {
            return Err(MetalError::Model(format!(
                "{}: Gemma 26B-A4B Metal requires Q8/F32 tensors",
                tensor.name
            )));
        }
    }
    for i in 0..count {
        for (name, dims, ty) in [
            ("ffn_gate_up_exps.weight", vec![WIDTH, FF * 2, EXPERTS], 8),
            ("ffn_down_exps.weight", vec![FF, WIDTH, EXPERTS], 8),
            ("ffn_gate_inp.weight", vec![WIDTH, EXPERTS], 0),
            ("ffn_gate_inp.scale", vec![WIDTH], 0),
            ("ffn_down_exps.scale", vec![EXPERTS], 0),
            ("pre_ffw_norm_2.weight", vec![WIDTH], 0),
            ("post_ffw_norm_1.weight", vec![WIDTH], 0),
            ("post_ffw_norm_2.weight", vec![WIDTH], 0),
        ] {
            let name = format!("blk.{i}.{name}");
            let tensor = map
                .tensor_info(&name)
                .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
            let compact = diffusion
                && ((name.ends_with("ffn_gate_up_exps.weight") && tensor.raw_type == 12)
                    || (name.ends_with("ffn_down_exps.weight") && tensor.raw_type == 6));
            if (tensor.raw_type != ty && !compact)
                || tensor.dims.iter().map(|&n| n as usize).collect::<Vec<_>>() != dims
            {
                return Err(MetalError::Model(format!(
                    "{name}: expected type {ty}, shape {dims:?}"
                )));
            }
        }
    }
    Ok(())
}

pub(super) struct Experts {
    router: Weight,
    gamma: Weight,
    pre: Weight,
    shared_post: Weight,
    routed_post: Weight,
    gu: Weight,
    down: Weight,
    scale: Weight,
}
impl Experts {
    pub(super) fn load_mlx(
        d: &MetalDevice,
        source: &mlx::Source,
        cfg: &paddock_models::mlx::DiffusionConfig,
        p: &str,
    ) -> Result<Self> {
        let raw = |name: &str, n| source.raw(d, &format!("{p}.{name}"), &[n], true);
        let packed = |name: &str, k, shape: &[usize]| {
            let name = format!("{p}.{name}");
            source.packed(
                d,
                &name,
                k,
                shape,
                cfg.bits(&name).map_err(MetalError::Model)?,
            )
        };
        Ok(Self {
            router: packed("router.proj", WIDTH, &[EXPERTS])?,
            gamma: raw("router.scale", WIDTH)?,
            pre: raw("pre_feedforward_layernorm_2.weight", WIDTH)?,
            shared_post: raw("post_feedforward_layernorm_1.weight", WIDTH)?,
            routed_post: raw("post_feedforward_layernorm_2.weight", WIDTH)?,
            scale: raw("router.per_expert_scale", EXPERTS)?,
            gu: packed("experts.gate_up_proj", WIDTH, &[EXPERTS, FF * 2])?,
            down: packed("experts.down_proj", FF, &[EXPERTS, WIDTH])?,
        })
    }
    pub(super) fn load(device: &MetalDevice, map: &MappedGguf, i: usize) -> Result<Self> {
        let w = |name: &str, dims: &[usize]| {
            Weight::load(device, map, &format!("blk.{i}.{name}"), dims)
        };
        Ok(Self {
            router: w("ffn_gate_inp.weight", &[WIDTH, EXPERTS])?,
            gamma: w("ffn_gate_inp.scale", &[WIDTH])?,
            pre: w("pre_ffw_norm_2.weight", &[WIDTH])?,
            shared_post: w("post_ffw_norm_1.weight", &[WIDTH])?,
            routed_post: w("post_ffw_norm_2.weight", &[WIDTH])?,
            gu: w("ffn_gate_up_exps.weight", &[WIDTH, 2 * FF, EXPERTS])?,
            down: w("ffn_down_exps.weight", &[FF, WIDTH, EXPERTS])?,
            scale: w("ffn_down_exps.scale", &[EXPERTS])?,
        })
    }
}

pub(super) struct Workspace {
    rows: usize,
    router_input: Buffer,
    input: Buffer,
    logits: Buffer,
    ids: Buffer,
    weights: Buffer,
    lists: Buffer,
    counts: Buffer,
    tiles: Buffer,
    gu: Buffer,
    out: Buffer,
    routed: Buffer,
}
impl Workspace {
    #[cfg(test)]
    pub(super) fn diagnostic(&self, rows: usize) -> [(&str, &Buffer, usize); 7] {
        [
            ("stage.router_logits", &self.logits, rows * EXPERTS),
            ("stage.ids", &self.ids, rows * ACTIVE),
            ("stage.weights", &self.weights, rows * ACTIVE),
            ("stage.routed", &self.routed, rows * WIDTH),
            ("stage.expert_act", &self.gu, rows * ACTIVE * FF * 2),
            ("stage.expert_out", &self.out, rows * ACTIVE * WIDTH),
            ("stage.expert_norm", &self.input, rows * WIDTH),
        ]
    }
    fn sizes(rows: usize) -> [usize; 11] {
        [
            rows * WIDTH,
            rows * WIDTH,
            rows * EXPERTS,
            rows * ACTIVE,
            rows * ACTIVE,
            EXPERTS * rows * ACTIVE,
            EXPERTS,
            1 + 2 * ((rows * ACTIVE).div_ceil(16) + EXPERTS),
            rows * ACTIVE * FF * 2,
            rows * ACTIVE * WIDTH,
            rows * WIDTH,
        ]
    }
    pub(super) fn bytes(rows: usize) -> u64 {
        Self::sizes(rows).into_iter().map(|n| n as u64 * 4).sum()
    }
    pub(super) fn new(device: &MetalDevice, rows: usize) -> Result<Self> {
        let mut sizes = Self::sizes(rows).into_iter();
        let mut a = || device.alloc(sizes.next().expect("eleven workspace planes") * 4);
        Ok(Self {
            rows,
            router_input: a()?,
            input: a()?,
            logits: a()?,
            ids: a()?,
            weights: a()?,
            lists: a()?,
            counts: a()?,
            tiles: a()?,
            gu: a()?,
            out: a()?,
            routed: a()?,
        })
    }
    fn experts(&self, cmd: &Commands<'_>, w: &Experts, rows: usize, grouped: bool) {
        if w.gu.ty != 8 || w.down.ty != 8 {
            self.packed_experts(cmd, w, rows, false);
            return;
        }
        let tile = if rows * ACTIVE >= EXPERTS * 32 {
            32
        } else {
            16
        };
        if grouped {
            cmd.dispatch(
                "moe_align",
                &[&self.ids, &self.lists, &self.counts],
                &[(rows * ACTIVE) as u32],
                [EXPERTS, 1, 1],
                256,
            );
            cmd.dispatch(
                "moe_tiles",
                &[&self.counts, &self.tiles],
                &[EXPERTS as u32, tile as u32],
                [1, 1, 1],
                256,
            );
        }
        for down in [false, true] {
            let (k, n) = if down { (FF, WIDTH) } else { (WIDTH, FF) };
            let source = if down { &w.down.buffer } else { &w.gu.buffer };
            let input = if down { &self.gu } else { &self.input };
            let out = if down { &self.out } else { &self.gu };
            let p = [k as u32, n as u32, rows as u32];
            if grouped {
                let kernel = match (down, tile) {
                    (false, 16) => "gmoe_gu_strict16",
                    (false, _) => "gmoe_gu_strict32",
                    (true, 16) => "qmoe_down_strict16",
                    (true, _) => "qmoe_down_strict32",
                };
                cmd.dispatch(
                    kernel,
                    &[
                        source,
                        source,
                        input,
                        &self.lists,
                        &self.counts,
                        &self.tiles,
                        out,
                    ],
                    &p,
                    [
                        n.div_ceil(32) * if down { 1 } else { 2 },
                        (rows * ACTIVE).div_ceil(tile) + EXPERTS,
                        1,
                    ],
                    128,
                );
            } else if down {
                cmd.dispatch(
                    "qmoe_down_decode",
                    &[source, input, &self.ids, out],
                    &p,
                    [n.div_ceil(4), rows * ACTIVE, 1],
                    128,
                );
            } else {
                cmd.dispatch(
                    "gmoe_gu_decode",
                    &[source, input, &self.ids, out],
                    &p,
                    [n.div_ceil(4), rows * ACTIVE, 1],
                    128,
                );
            }
            if !down {
                cmd.dispatch(
                    "gmoe_geglu",
                    &[&self.gu],
                    &[FF as u32, (rows * ACTIVE) as u32],
                    [(rows * ACTIVE * FF).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }
    fn packed_experts(&self, cmd: &Commands<'_>, w: &Experts, rows: usize, mlx: bool) {
        cmd.dispatch(
            "moe_align",
            &[&self.ids, &self.lists, &self.counts],
            &[(rows * ACTIVE) as u32],
            [EXPERTS, 1, 1],
            256,
        );
        cmd.dispatch(
            "moe_tiles",
            &[&self.counts, &self.tiles],
            &[EXPERTS as u32, 16],
            [1, 1, 1],
            256,
        );
        for down in [false, true] {
            let (weight, input, out, k, n) = if down {
                (&w.down, &self.gu, &self.out, FF, WIDTH)
            } else {
                (&w.gu, &self.input, &self.gu, WIDTH, FF)
            };
            if mlx && rows < 64 {
                let n = if down { n } else { n * 2 };
                cmd.dispatch(
                    "dg_expert_vector",
                    &[&weight.buffer, input, &self.ids, out],
                    &[k as u32, n as u32, rows as u32, weight.ty, u32::from(down)],
                    [n.div_ceil(4), rows * ACTIVE, 1],
                    128,
                );
            } else {
                cmd.dispatch(
                    match weight.ty {
                        6 => "dg_experts_q5",
                        8 => "dg_experts_q8",
                        12 => "dg_experts_q4",
                        0x100 => "dg_experts_a4",
                        0x108 => "dg_experts_a8",
                        _ => "dg_experts",
                    },
                    &[
                        &weight.buffer,
                        input,
                        &self.lists,
                        &self.counts,
                        &self.tiles,
                        out,
                    ],
                    &[
                        k as u32,
                        n as u32,
                        rows as u32,
                        weight.ty,
                        u32::from(down),
                        u32::from(mlx),
                    ],
                    [
                        n.div_ceil(32) * if down { 1 } else { 2 },
                        (rows * ACTIVE).div_ceil(16) + EXPERTS,
                        1,
                    ],
                    128,
                );
            }
            if !down {
                cmd.dispatch(
                    if mlx { "dg_moe_geglu" } else { "gmoe_geglu" },
                    &[&self.gu],
                    &[FF as u32, (rows * ACTIVE) as u32],
                    [(rows * ACTIVE * FF).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }
    pub(super) fn execute(
        &self,
        cmd: &Commands<'_>,
        w: &Experts,
        x: &Buffer,
        shared: &Buffer,
        rows: usize,
        eps: f32,
    ) {
        assert!(rows <= self.rows);
        if w.router.ty == 0x108 || w.router.ty == 0x100 {
            self.execute_mlx(cmd, w, x, shared, rows, eps);
            return;
        }
        cmd.dispatch(
            "gmoe_head",
            &[
                x,
                &w.gamma.buffer,
                &w.pre.buffer,
                &self.router_input,
                &self.input,
            ],
            &[WIDTH as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
        cmd.dispatch(
            "linear",
            &[&w.router.buffer, &self.router_input, &self.logits],
            &[WIDTH as u32, EXPERTS as u32, rows as u32, 0, 1f32.to_bits()],
            [EXPERTS / 4, rows, 1],
            128,
        );
        cmd.dispatch(
            "gmoe_route",
            &[&self.logits, &w.scale.buffer, &self.ids, &self.weights],
            &[],
            [rows, 1, 1],
            32,
        );
        self.experts(cmd, w, rows, rows >= 16);
        cmd.dispatch(
            "gmoe_fold",
            &[&self.out, &self.weights, &self.routed],
            &[WIDTH as u32, rows as u32],
            [(rows * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "gmoe_branches",
            &[
                shared,
                &self.routed,
                &w.shared_post.buffer,
                &w.routed_post.buffer,
            ],
            &[WIDTH as u32, eps.to_bits()],
            [rows, 1, 1],
            256,
        );
    }
    fn execute_mlx(
        &self,
        cmd: &Commands<'_>,
        w: &Experts,
        x: &Buffer,
        shared: &Buffer,
        rows: usize,
        eps: f32,
    ) {
        cmd.dispatch(
            "dg_moe_head",
            &[
                x,
                &w.gamma.buffer,
                &w.pre.buffer,
                &self.router_input,
                &self.input,
            ],
            &[WIDTH as u32, eps.to_bits()],
            [rows, 1, 1],
            WIDTH.div_ceil(128) * 32,
        );
        diffusion::project(cmd, &w.router, &self.router_input, &self.logits, rows, true);
        cmd.dispatch(
            "dg_moe_route",
            &[&self.logits, &w.scale.buffer, &self.ids, &self.weights],
            &[],
            [rows, 1, 1],
            32,
        );
        self.packed_experts(cmd, w, rows, true);
        cmd.dispatch(
            "dg_moe_fold",
            &[&self.out, &self.weights, &self.routed],
            &[WIDTH as u32, rows as u32],
            [(rows * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "dg_moe_branches",
            &[
                shared,
                &self.routed,
                &w.shared_post.buffer,
                &w.routed_post.buffer,
            ],
            &[WIDTH as u32, eps.to_bits()],
            [rows, 1, 1],
            WIDTH.div_ceil(128) * 32,
        );
    }
}

// Routing magnifies row-dependent projection rounding. Preserve F32 operands
// for both prefill and decode, including K=2112 (a partial BK128 tile).
pub(super) fn project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
) {
    for &(w, out) in planes {
        if w.ty != 8 || rows == 1 {
            w.linear(cmd, input, out, rows, 1., workspace);
            continue;
        }
        let (kernel, bm, bn) = match rows {
            2..=4 => ("linear_q8_r4", 4, 4),
            5..=8 => ("linear_q8_r8", 8, 4),
            9..=15 => ("linear_q8_r16", 16, 4),
            16 => ("muse_q8_f32_16", 16, 16),
            17..=32 => ("muse_q8_f32_32", 32, 16),
            _ => ("muse_q8_f32_64", 64, 16),
        };
        cmd.dispatch(
            kernel,
            &[&w.buffer, input, out],
            &[w.k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(bn), rows.div_ceil(bm), 1],
            128,
        );
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
