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
    for tensor in &map.gguf().tensors {
        if !matches!(tensor.raw_type, 0 | 8) {
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
            if tensor.raw_type != ty
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
