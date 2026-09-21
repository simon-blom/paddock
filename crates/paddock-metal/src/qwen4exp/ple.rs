//! GPU n-gram hash, resident IQ4_NL gather and causal dilated convolution.
//! A read/compute/commit split keeps row-parallel prefill free of ring races.
//! Token history and convolution history are independent: EOS resets only
//! n-gram segmentation, not the convolution. Slot reset clears both.
use super::residual::{self, WIDE, WIDTH, load_weight};
use super::{affine, mlx};
use crate::{
    device::{Buffer, Commands, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::mapped::MappedGguf;

pub(super) const TABLE_ROWS: usize = 320001536;
pub(super) struct Weights {
    key: Weight,
    value: Weight,
    key_norm: Weight,
    query_norm: Weight,
    conv_norm: Weight,
    conv: Weight,
}
impl Weights {
    pub(super) fn load_mlx(d: &MetalDevice, s: &mlx::Source) -> Result<Self> {
        let w = |suffix: &str| s.weight(d, &format!("{}.{suffix}", mlx::PLE));
        Ok(Self {
            key: w("key_proj")?,
            value: w("value_proj")?,
            key_norm: w("norm_key.weight")?,
            query_norm: w("norm_query.weight")?,
            conv_norm: w("norm_conv.weight")?,
            conv: w("conv1d.weight")?,
        })
    }
    pub(super) fn load(d: &MetalDevice, map: &MappedGguf) -> Result<Self> {
        let w = |name: &str, dims: &[usize], ty| {
            load_weight(d, map, &format!("blk.1.ple_{name}.weight"), dims, ty)
        };
        Ok(Self {
            key: w("key", &[WIDTH, WIDE], 8)?,
            value: w("value", &[WIDTH, WIDTH], 8)?,
            key_norm: w("norm_key", &[WIDE], 0)?,
            query_norm: w("norm_query", &[WIDE], 0)?,
            conv_norm: w("norm_conv", &[WIDE], 0)?,
            conv: w("conv1d", &[4, WIDE], 0)?,
        })
    }
}

pub(super) fn load_table(d: &MetalDevice, map: &MappedGguf) -> Result<Weight> {
    load_weight(
        d,
        map,
        "per_layer_token_embd.weight",
        &[160, TABLE_ROWS],
        20,
    )
}

// An immutable host control plan, not host model math. All bounds and slot
// contiguity are checked before touching the first GPU-owned history byte.
#[derive(Debug)]
pub(super) struct Plan {
    tokens: Vec<u32>,
    meta: Vec<u32>,
    spans: Vec<u32>,
    lengths: Vec<usize>,
}
impl Plan {
    pub(super) fn new(
        rows: &[(usize, usize, u32)],
        lengths: &[usize],
        capacity: usize,
        context: usize,
    ) -> Result<Self> {
        if rows.is_empty() || rows.len() > capacity {
            return Err(MetalError::Model(
                "Flash Next PLE row capacity exceeded/empty batch".into(),
            ));
        }
        let mut plan = Self {
            tokens: Vec::new(),
            meta: Vec::new(),
            spans: Vec::new(),
            lengths: lengths.to_vec(),
        };
        let mut seen = vec![false; lengths.len()];
        let mut first = 0;
        while first < rows.len() {
            let (slot, start, _) = rows[first];
            if slot >= lengths.len() || seen[slot] {
                return Err(MetalError::Model(
                    "Flash Next PLE invalid/repeated slot span".into(),
                ));
            }
            seen[slot] = true;
            let end = rows[first..]
                .iter()
                .position(|r| r.0 != slot)
                .map_or(rows.len(), |i| first + i);
            for &(s, pos, token) in &rows[first..end] {
                if pos != plan.lengths[s] || pos >= context || token >= 248320 {
                    return Err(MetalError::Model(
                        "Flash Next PLE invalid token/position/context".into(),
                    ));
                }
                plan.lengths[s] += 1;
                plan.tokens.push(token);
                plan.meta
                    .extend([s as u32, pos as u32, first as u32, end as u32]);
            }
            plan.spans.extend([
                first as u32,
                (end - first) as u32,
                slot as u32,
                start as u32,
            ]);
            first = end;
        }
        Ok(plan)
    }
}

/// One PLE site owns its persistent histories and bounded workspace. The
/// model calls stage/encode without per-layer submission or length publication;
/// test-only synchronous methods exercise this identical encoder in isolation.
pub(super) struct State {
    #[cfg(test)]
    capacity: usize,
    #[cfg(test)]
    context: usize,
    #[cfg(test)]
    lengths: Vec<usize>,
    #[cfg(test)]
    poisoned: bool,
    tokens: Buffer,
    meta: Buffer,
    spans: Buffer,
    pub(super) history: Buffer,
    pub(super) ring: Buffer,
    pub(super) ids: Buffer,
    pub(super) embedding: Buffer,
    pub(super) key: Buffer,
    pub(super) query: Buffer,
    pub(super) value: Buffer,
    pub(super) gated: Buffer,
    pub(super) norm: Buffer,
}
impl State {
    pub(super) fn bytes(capacity: usize, slots: usize, context: usize) -> Result<usize> {
        if !(1..=1024).contains(&capacity)
            || !(1..=64).contains(&slots)
            || !(1..=262144).contains(&context)
        {
            return Err(MetalError::Model(
                "Flash Next PLE bounds: rows 1..=1024, slots 1..=64, context 1..=262144".into(),
            ));
        }
        Ok(4 * (capacity * (1 + 4 + 16 + WIDTH * 2 + WIDE * 4) + slots * (4 + 2 + 9 * WIDE)))
    }
    pub(super) fn new(
        d: &MetalDevice,
        capacity: usize,
        slots: usize,
        context: usize,
    ) -> Result<Self> {
        Self::bytes(capacity, slots, context)?;
        let state = Self {
            #[cfg(test)]
            capacity,
            #[cfg(test)]
            context,
            #[cfg(test)]
            lengths: vec![0; slots],
            #[cfg(test)]
            poisoned: false,
            tokens: d.alloc(capacity * 4)?,
            meta: d.alloc(capacity * 4 * 4)?,
            spans: d.alloc(slots * 4 * 4)?,
            history: d.alloc(slots * 2 * 4)?,
            ring: d.alloc(slots * 9 * WIDE * 4)?,
            ids: d.alloc(capacity * 16 * 4)?,
            embedding: d.alloc(capacity * WIDTH * 4)?,
            key: d.alloc(capacity * WIDE * 4)?,
            query: d.alloc(capacity * WIDE * 4)?,
            value: d.alloc(capacity * WIDTH * 4)?,
            gated: d.alloc(capacity * WIDE * 4)?,
            norm: d.alloc(capacity * WIDE * 4)?,
        };
        let cmd = d.begin()?;
        for slot in 0..slots {
            state.encode_reset(&cmd, slot);
        }
        cmd.finish()?;
        Ok(state)
    }
    pub(super) fn cache_bytes(slots: usize) -> usize {
        slots * (2 + 9 * WIDE) * 4
    }
    pub(super) fn encode_reset(&self, cmd: &Commands<'_>, slot: usize) {
        cmd.dispatch(
            "q4x_ple_reset",
            &[&self.ring, &self.history],
            &[WIDE as u32, slot as u32],
            [(9 * WIDE).div_ceil(256), 1, 1],
            256,
        );
    }
    pub(super) fn stage(&self, plan: &Plan) {
        unsafe {
            self.tokens.write_u32(&plan.tokens);
            self.meta.write_u32(&plan.meta);
            self.spans.write_u32(&plan.spans);
        }
    }
    pub(super) fn encode(
        &self,
        cmd: &Commands<'_>,
        weights: &Weights,
        table: &Weight,
        plan: &Plan,
        hidden: &Buffer,
    ) {
        let n = plan.tokens.len();
        let mlx = affine::is_affine(table.ty);
        cmd.dispatch(
            "q4x_ple_hash",
            &[&self.tokens, &self.meta, &self.history, &self.ids],
            &[n as u32, (self.history.len() / 8) as u32],
            [n, 1, 1],
            32,
        );
        if mlx {
            cmd.dispatch(
                "q4a_ple_gather",
                &[&table.buffer, &self.ids, &self.embedding],
                &[(n * 16) as u32, mlx::SHARD_ROWS as u32, 128],
                [(n * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        } else {
            cmd.dispatch(
                "iq_gather",
                &[&table.buffer, &self.ids, &self.embedding],
                &[20, 160, (n * 16) as u32, TABLE_ROWS as u32],
                [(n * WIDTH / 4).div_ceil(256), 1, 1],
                256,
            );
        }
        residual::project(cmd, &weights.key, &self.embedding, &self.key, n);
        residual::project(cmd, &weights.value, &self.embedding, &self.value, n);
        residual::norm(cmd, &self.key, &weights.key_norm, &self.key, n);
        residual::norm(cmd, hidden, &weights.query_norm, &self.query, n);
        cmd.dispatch(
            if mlx { "q4b_ple_gate" } else { "q4x_ple_gate" },
            &[&self.key, &self.query, &self.value, &self.gated],
            &[WIDTH as u32],
            [4, n, 1],
            if mlx { 640 } else { 256 },
        );
        residual::norm(cmd, &self.gated, &weights.conv_norm, &self.norm, n);
        cmd.dispatch(
            if mlx { "q4b_ple_conv" } else { "q4x_ple_conv" },
            &[
                &self.norm,
                &weights.conv.buffer,
                &self.ring,
                &self.meta,
                &self.gated,
                hidden,
            ],
            &[WIDE as u32, n as u32],
            [(n * WIDE).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "q4x_ple_commit",
            &[&self.norm, &self.ring, &self.spans],
            &[WIDE as u32, (plan.spans.len() / 4) as u32],
            [((plan.spans.len() / 4) * 9 * WIDE).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "q4x_ple_tokens_commit",
            &[&self.tokens, &self.history, &self.spans],
            &[(plan.spans.len() / 4) as u32],
            [(plan.spans.len() / 4).div_ceil(32), 1, 1],
            32,
        );
    }
    #[cfg(test)]
    fn healthy(&self) -> Result<()> {
        if self.poisoned {
            Err(MetalError::Model(
                "Flash Next PLE state poisoned by failed GPU work; reload required".into(),
            ))
        } else {
            Ok(())
        }
    }
    #[cfg(test)]
    pub(super) fn reset(&mut self, d: &MetalDevice, slot: usize) -> Result<()> {
        self.healthy()?;
        if slot >= self.lengths.len() {
            return Err(MetalError::Model("invalid PLE reset slot".into()));
        }
        self.poisoned = true;
        let cmd = d.begin()?;
        self.encode_reset(&cmd, slot);
        cmd.submit()?.wait()?;
        self.lengths[slot] = 0;
        self.poisoned = false;
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn run(
        &mut self,
        d: &MetalDevice,
        weights: &Weights,
        table: &Weight,
        rows: &[(usize, usize, u32)],
        hidden: &Buffer,
    ) -> Result<()> {
        self.healthy()?;
        let plan = Plan::new(rows, &self.lengths, self.capacity, self.context)?;
        let n = rows.len();
        if hidden.len() < n * WIDE * 4
            || !matches!(table.ty, 20 | affine::A4G32)
            || table.k != 160
            || table.n != TABLE_ROWS
            || table.buffer.len() != TABLE_ROWS * if table.ty == 20 { 90 } else { 100 }
        {
            return Err(MetalError::Model(
                "invalid Flash Next PLE input/table storage".into(),
            ));
        }
        self.stage(&plan);
        self.poisoned = true;
        let cmd = d.begin()?;
        self.encode(&cmd, weights, table, &plan, hidden);
        cmd.finish()?;
        self.lengths = plan.lengths;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_plan_validates_the_entire_batch_without_publishing_state() {
        let lengths = [1, 3, 0, 2];
        let p = Plan::new(&[(3, 2, 1), (3, 3, 248044), (1, 3, 17)], &lengths, 32, 8).unwrap();
        assert_eq!(p.lengths, [1, 4, 0, 4]);
        assert_eq!(p.spans, [0, 2, 3, 2, 2, 1, 1, 3]);
        for rows in [
            vec![],
            vec![(4, 0, 1)],
            vec![(0, 0, 1)],
            vec![(0, 1, 248320)],
            vec![(0, 1, 1), (0, 3, 1)],
            vec![(0, 1, 1), (2, 0, 1), (0, 2, 1)],
            vec![(0, 1, 1), (1, usize::MAX, 2)],
            vec![(0, 1, 1), (0, 2, 1), (0, 3, 1)],
        ] {
            assert!(Plan::new(&rows, &lengths, 2, 8).is_err());
            assert_eq!(lengths, [1, 3, 0, 2]);
        }
        assert!(Plan::new(&[(0, 1, 1)], &lengths, 8, 1).is_err());
        for args in [
            (0, 1, 1),
            (1025, 1, 1),
            (usize::MAX, 1, 1),
            (1, 0, 1),
            (1, 65, 1),
            (1, 1, 0),
            (1, 1, 262145),
        ] {
            assert!(State::bytes(args.0, args.1, args.2).is_err());
        }
        for rows in [128, 512, 513, 1024] {
            assert!(State::bytes(rows, 1, 1024).is_ok());
        }
    }
}
