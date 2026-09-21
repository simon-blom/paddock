//! Top-six/full-softmax routing, ungated shared experts, bounded grouped Q8.
//! Selection stays on GPU; integer compaction avoids floating-point atomics.
use super::*;
use paddock_models::mapped::MappedGguf;
const E: usize = 64;
const A: usize = 6;
const F: usize = 896;
pub(super) struct Experts {
    router: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}
pub(super) fn schema(s: &mut Vec<(String, Vec<usize>, u32)>, i: usize) {
    for (name, dims, ty) in [
        ("ffn_gate_inp.weight", vec![WIDTH, E], 0),
        ("ffn_gate_exps.weight", vec![WIDTH, F, E], 8),
        ("ffn_up_exps.weight", vec![WIDTH, F, E], 8),
        ("ffn_down_exps.weight", vec![F, WIDTH, E], 8),
    ] {
        s.push((format!("blk.{i}.{name}"), dims, ty));
    }
}
impl Experts {
    pub(super) fn load(d: &MetalDevice, map: &MappedGguf, i: usize) -> Result<Self> {
        let w = |n: &str, s: &[usize]| Weight::load(d, map, &format!("blk.{i}.{n}"), s);
        Ok(Self {
            router: w("ffn_gate_inp.weight", &[WIDTH, E])?,
            gate: w("ffn_gate_exps.weight", &[WIDTH, F, E])?,
            up: w("ffn_up_exps.weight", &[WIDTH, F, E])?,
            down: w("ffn_down_exps.weight", &[F, WIDTH, E])?,
        })
    }
}
pub(super) struct Workspace {
    logits: Buffer,
    ids: Buffer,
    weights: Buffer,
    lists: Buffer,
    counts: Buffer,
    tiles: Buffer,
    gu: Buffer,
    out: Buffer,
}
impl Workspace {
    fn sizes() -> [usize; 8] {
        [
            CHUNK * E,
            CHUNK * A,
            CHUNK * A,
            E * CHUNK * A,
            E,
            1 + 2 * ((CHUNK * A).div_ceil(16) + E),
            CHUNK * A * F * 2,
            CHUNK * A * WIDTH,
        ]
    }
    pub(super) fn bytes() -> usize {
        Self::sizes().iter().sum::<usize>() * 4
    }
    pub(super) fn new(d: &MetalDevice) -> Result<Self> {
        let mut sizes = Self::sizes().into_iter();
        let mut a = || d.alloc(sizes.next().expect("eight buffers") * 4);
        Ok(Self {
            logits: a()?,
            ids: a()?,
            weights: a()?,
            lists: a()?,
            counts: a()?,
            tiles: a()?,
            gu: a()?,
            out: a()?,
        })
    }
    pub(super) fn run(
        &self,
        c: &Commands<'_>,
        w: &Experts,
        x: &Buffer,
        delta: &Buffer,
        rows: usize,
    ) {
        c.dispatch(
            "linear",
            &[&w.router.buffer, x, &self.logits],
            &[WIDTH as u32, E as u32, rows as u32, 0, 1f32.to_bits()],
            [E.div_ceil(4), rows, 1],
            128,
        );
        c.dispatch(
            "uocr_route",
            &[&self.logits, &self.ids, &self.weights],
            &[],
            [rows, 1, 1],
            32,
        );
        let grouped = rows >= 16;
        if grouped {
            c.dispatch(
                "moe_align",
                &[&self.ids, &self.lists, &self.counts],
                &[(rows * A) as u32],
                [E, 1, 1],
                256,
            );
            c.dispatch(
                "moe_tiles",
                &[&self.counts, &self.tiles],
                &[E as u32, 16],
                [1, 1, 1],
                256,
            );
        }
        for down in [false, true] {
            let (k, n) = if down { (F, WIDTH) } else { (WIDTH, F) };
            let source = if down { &w.down.buffer } else { &w.gate.buffer };
            let input = if down { &self.gu } else { x };
            let out = if down { &self.out } else { &self.gu };
            let p = [k as u32, n as u32, rows as u32];
            if grouped {
                c.dispatch(
                    if down {
                        "uocr_down_grouped"
                    } else {
                        "uocr_gu_grouped"
                    },
                    &[
                        source,
                        &w.up.buffer,
                        input,
                        &self.lists,
                        &self.counts,
                        &self.tiles,
                        out,
                    ],
                    &p,
                    [
                        n.div_ceil(32) * if down { 1 } else { 2 },
                        (rows * A).div_ceil(16) + E,
                        1,
                    ],
                    128,
                );
            } else if down {
                c.dispatch(
                    "qmoe_down_decode",
                    &[source, input, &self.ids, out],
                    &p,
                    [n.div_ceil(4), rows * A, 1],
                    128,
                );
            } else {
                c.dispatch(
                    "uocr_gu_decode",
                    &[source, &w.up.buffer, input, &self.ids, out],
                    &p,
                    [n.div_ceil(4), rows * A, 1],
                    128,
                );
            }
            if !down {
                c.dispatch(
                    "qmoe_swiglu",
                    &[&self.gu],
                    &[F as u32, (rows * A) as u32],
                    [(rows * A * F).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
        c.dispatch(
            "uocr_fold",
            &[&self.out, &self.weights, delta],
            &[WIDTH as u32, rows as u32],
            [(rows * WIDTH).div_ceil(256), 1, 1],
            256,
        );
    }
}
