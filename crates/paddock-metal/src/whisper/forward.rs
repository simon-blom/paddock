use super::*;
impl Scratch {
    pub(super) fn sizes(cap: usize) -> [usize; 17] {
        let r = T * cap.min(ENC_BATCH);
        [
            r * D * 4,
            2 * r * D * 4,
            r * 3 * D * 4,
            (r + 64) * D * 4,
            (r + 64) * D * 4,
            (r + 64) * D * 4,
            r * D * 4,
            r * FF * 4,
            r * 3 * D * 4,
            cap * 20 * 6 * 66 * 4,
            cap * V * 4,
            cap * 4,
            cap * 4,
            cap * 4,
            cap * 8,
            cap * 12,
            cap * 12,
        ]
    }
    fn new(d: &MetalDevice, cap: usize) -> Result<Self> {
        let sz = Self::sizes(cap);
        let mut a = sz
            .iter()
            .map(|&n| d.alloc(n))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut next = || a.next().expect("fixed scratch layout");
        Ok(Self {
            x: next(),
            norm: next(),
            qkv: next(),
            q: next(),
            k: next(),
            v: next(),
            attn: next(),
            up: next(),
            conv: next(),
            partial: next(),
            logits: next(),
            slots: next(),
            positions: next(),
            tokens: next(),
            rules: next(),
            pick: next(),
            stats: next(),
        })
    }
}
pub(super) fn norm(c: &Commands<'_>, n: &Norm, x: &Buffer, out: &Buffer, rows: usize) {
    c.dispatch(
        "vis_ln",
        &[x, &n.w, &n.b, out],
        &[D as u32, 1e-5f32.to_bits()],
        [rows, 1, 1],
        256,
    );
}
#[allow(clippy::too_many_arguments)]
pub(super) fn plain(
    c: &Commands<'_>,
    w: &Buffer,
    b: &Buffer,
    x: &Buffer,
    out: &Buffer,
    k: usize,
    n: usize,
    rows: usize,
    ep: u32,
) {
    c.dispatch(
        match rows {
            0..=1 => "wh_mv1",
            2 => "wh_mv2",
            3..=4 => "wh_mv4",
            5..=8 => "wh_mv8",
            9..=16 => "wh_mv16",
            _ => "wh_mm",
        },
        &[w, x, out, b],
        &[k as u32, n as u32, rows as u32, ep],
        if rows <= 16 {
            [n.div_ceil(4), 1, 1]
        } else {
            [n.div_ceil(64), rows.div_ceil(32), 1]
        },
        128,
    );
}
pub(super) fn linear(c: &Commands<'_>, w: &Linear, x: &Buffer, out: &Buffer, rows: usize, ep: u32) {
    plain(c, &w.w, &w.b, x, out, w.k, w.n, rows, ep);
}

pub(super) fn mlp(c: &Commands<'_>, m: &Mlp, s: &Scratch, rows: usize) {
    norm(c, &m.norm, &s.x, &s.norm, rows);
    linear(c, &m.up, &s.norm, &s.up, rows, 3);
    linear(c, &m.down, &s.up, &s.x, rows, 2);
}
impl Whisper {
    pub(super) fn prepare(&mut self, cap: usize) -> Result<()> {
        if cap == 0 || cap > MAX_BATCH {
            return Err(error("decode slots must be 1..16"));
        }
        if self.capacity != 0 {
            return if self.capacity == cap {
                Ok(())
            } else {
                Err(error("cannot resize live slot pool"))
            };
        }
        let kv = (L * cap * (T + self.ctx) * D * 4) as u64;
        let scratch = Scratch::sizes(cap).iter().sum::<usize>() as u64;
        // Admission's input and tile uploads are additional live bytes.
        let admission = (cap.min(ENC_BATCH) * (3000 * 128 * 4 + T.div_ceil(32) * 16)) as u64;
        if self.weights_bytes + kv + scratch + admission > self.device.budget_bytes() {
            return Err(MetalError::Memory(
                "Whisper weights + slot self/cross KV + encoder scratch exceed grant".into(),
            ));
        }
        let mut cache = Vec::new();
        for _ in 0..L {
            cache.push(Cache {
                k: self.device.alloc(cap * self.ctx * D * 2)?,
                v: self.device.alloc(cap * self.ctx * D * 2)?,
                ck: self.device.alloc(cap * T * D * 2)?,
                cv: self.device.alloc(cap * T * D * 2)?,
            });
        }
        let scratch = Scratch::new(&self.device, cap)?;
        self.cache = cache;
        self.scratch = Some(scratch);
        self.capacity = cap;
        self.lengths = vec![None; cap];
        Ok(())
    }
    pub(super) fn encode_wave(&mut self, slots: &[usize], mels: &[&MelFeatures]) -> Result<()> {
        if slots.is_empty()
            || slots.len() != mels.len()
            || slots.len() > self.capacity.min(ENC_BATCH)
        {
            return Err(error("invalid encoder admission width"));
        }
        for (i, &slot) in slots.iter().enumerate() {
            if slot >= self.capacity || slots[..i].contains(&slot) {
                return Err(error("invalid/duplicate encoder slot"));
            }
            let m = mels[i];
            if m.n_frames != 3000
                || m.data.len() != 3000 * 128
                || m.n_samples > 480000
                || !m.data.iter().all(|x| x.is_finite())
            {
                return Err(error("requires finite 3000x128 Whisper mel, at most 30 s"));
            }
        }
        let s = self
            .scratch
            .as_ref()
            .ok_or_else(|| error("prepare slots first"))?;
        let rows = slots.len() * T;
        let input = upload(
            &self.device,
            &mels
                .iter()
                .flat_map(|m| m.data.iter().map(|v| v.to_bits()))
                .collect::<Vec<_>>(),
        )?;
        let mut tiles = Vec::new();
        for b in 0..slots.len() {
            for q in (0..T).step_by(32) {
                tiles.extend([
                    (b * T + q) as u32,
                    (T - q).min(32) as u32,
                    (b * T) as u32,
                    T as u32,
                ]);
            }
        }
        let tiles = upload(&self.device, &tiles)?;
        // SAFETY: all previous command buffers are complete at this boundary.
        unsafe {
            s.slots
                .write_u32(&slots.iter().map(|&s| s as u32).collect::<Vec<_>>());
        }
        let c = self.device.begin()?;
        c.dispatch(
            "wh_conv_rows",
            &[&input, &s.conv],
            &[128, 3000, 1, slots.len() as u32],
            [(rows * 2 * 3 * 128).div_ceil(256), 1, 1],
            256,
        );
        linear(&c, &self.conv1, &s.conv, &s.norm, rows * 2, 3);
        c.dispatch(
            "wh_conv_rows",
            &[&s.norm, &s.conv],
            &[D as u32, 3000, 2, slots.len() as u32],
            [(rows * 3 * D).div_ceil(256), 1, 1],
            256,
        );
        linear(&c, &self.conv2, &s.conv, &s.x, rows, 3);
        c.dispatch(
            "wh_position",
            &[&s.x, &self.enc_pos],
            &[rows as u32],
            [(rows * D).div_ceil(256), 1, 1],
            256,
        );
        for layer in &self.enc {
            let a = &layer.attn;
            norm(&c, &a.norm, &s.x, &s.norm, rows);
            plain(&c, &a.qkv, &a.bias, &s.norm, &s.qkv, D, 3 * D, rows, 0);
            c.dispatch(
                "wh_split",
                &[&s.qkv, &a.bias, &s.q, &s.k, &s.v],
                &[rows as u32],
                [((rows + 64) * D).div_ceil(256), 1, 1],
                256,
            );
            c.dispatch(
                "wh_attention",
                &[&s.q, &s.k, &s.v, &s.attn, &tiles],
                &[0],
                [20, slots.len() * T.div_ceil(32), 1],
                64,
            );
            linear(&c, &a.out, &s.attn, &s.x, rows, 2);
            mlp(&c, &layer.mlp, s, rows);
        }
        norm(&c, &self.enc_ln, &s.x, &s.x, rows);
        for (layer, cache) in self.dec.iter().zip(&self.cache) {
            plain(&c, &layer.kv, &layer.vb, &s.x, &s.qkv, D, 2 * D, rows, 0);
            c.dispatch(
                "wh_cross_store",
                &[&s.qkv, &layer.vb, &cache.ck, &cache.cv, &s.slots],
                &[slots.len() as u32],
                [(rows * D).div_ceil(256), 1, 1],
                256,
            );
        }
        c.finish()?;
        for &slot in slots {
            self.lengths[slot] = Some(0);
        }
        self.last_rows = 0;
        Ok(())
    }
    pub(super) fn step(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        pos: &[u32],
        rules: Option<&[u32]>,
    ) -> Result<StepOut> {
        let rows = slots.len();
        if rows == 0 {
            if !tokens.is_empty() || !pos.is_empty() || rules.is_some_and(|r| !r.is_empty()) {
                return Err(error("invalid empty decode"));
            }
            return Ok(StepOut::default());
        }
        if rows > self.capacity
            || tokens.len() != rows
            || pos.len() != rows
            || rules.is_some_and(|r| r.len() != rows * 2)
        {
            return Err(error("invalid decode arrays"));
        }
        for (i, &slot) in slots.iter().enumerate() {
            if slot as usize >= self.capacity
                || slots[..i].contains(&slot)
                || tokens[i] as usize >= V
                || pos[i] as usize >= self.ctx
                || self.lengths[slot as usize] != Some(pos[i] as usize)
            {
                return Err(error(
                    "unadmitted, duplicate, stale or out-of-range decode row",
                ));
            }
        }
        if rules.is_some_and(|r| {
            r.chunks_exact(2)
                .any(|r| r[0] & !31 != 0 || (r[0] & 16 != 0 && !(50365..=V as u32).contains(&r[1])))
        }) {
            return Err(error("invalid timestamp grammar state"));
        }
        let s = self
            .scratch
            .as_ref()
            .ok_or_else(|| error("prepare slots first"))?;
        // SAFETY: no command is pending; arrays were bounded above.
        unsafe {
            s.slots.write_u32(slots);
            s.tokens.write_u32(tokens);
            s.positions.write_u32(pos);
            if let Some(r) = rules {
                s.rules.write_u32(r);
            }
        }
        let c = self.device.begin()?;
        c.dispatch(
            "wh_embed",
            &[
                &self.embedding,
                &self.dec_pos,
                &s.tokens,
                &s.positions,
                &s.x,
            ],
            &[rows as u32],
            [(rows * D).div_ceil(256), 1, 1],
            256,
        );
        for (layer, cache) in self.dec.iter().zip(&self.cache) {
            norm(&c, &layer.attn.norm, &s.x, &s.norm, rows);
            plain(
                &c,
                &layer.attn.qkv,
                &layer.attn.bias,
                &s.norm,
                &s.qkv,
                D,
                3 * D,
                rows,
                0,
            );
            c.dispatch(
                "wh_append",
                &[
                    &s.qkv,
                    &layer.attn.bias,
                    &s.q,
                    &cache.k,
                    &cache.v,
                    &s.slots,
                    &s.positions,
                ],
                &[rows as u32, self.ctx as u32],
                [(rows * D).div_ceil(256), 1, 1],
                256,
            );
            let splits = (pos
                .iter()
                .copied()
                .max()
                .expect("nonempty validated decode") as usize
                + 1)
            .div_ceil(256);
            attention(&c, s, &cache.k, &cache.v, self.ctx, rows, splits, false);
            linear(&c, &layer.attn.out, &s.attn, &s.x, rows, 2);
            norm(&c, &layer.cross_norm, &s.x, &s.norm, rows);
            linear(&c, &layer.q, &s.norm, &s.q, rows, 1);
            attention(&c, s, &cache.ck, &cache.cv, T, rows, 6, true);
            linear(&c, &layer.out, &s.attn, &s.x, rows, 2);
            mlp(&c, &layer.mlp, s, rows);
        }
        norm(&c, &self.dec_ln, &s.x, &s.norm, rows);
        plain(
            &c,
            self.head.as_ref().unwrap_or(&self.embedding),
            &s.norm,
            &s.norm,
            &s.logits,
            D,
            V,
            rows,
            0,
        );
        if rules.is_some() {
            c.dispatch(
                "wh_rules",
                &[&s.logits, &s.rules],
                &[V as u32, 50257, 50364, 50365],
                [rows, 1, 1],
                256,
            );
        }
        c.dispatch(
            "wh_pick",
            &[&s.logits, &s.pick, &s.stats],
            &[V as u32, 50363],
            [rows, 1, 1],
            256,
        );
        c.finish()?;
        for (i, &slot) in slots.iter().enumerate() {
            self.lengths[slot as usize] = Some(pos[i] as usize + 1);
        }
        self.last_rows = rows;
        // SAFETY: finish waited for both logits and the reduction outputs.
        let (ids, stats) = unsafe { (s.pick.read_u32(rows * 3), s.stats.read_f32(0, rows * 3)) };
        if ids.chunks_exact(3).any(|r| r[2] != 0) {
            return Err(error("nonfinite/fully masked logits"));
        }
        let mut out = StepOut::default();
        for (id, st) in ids.chunks_exact(3).zip(stats.chunks_exact(3)) {
            out.next.push(id[0]);
            out.logprob.push(st[0]);
            out.nospeech.push(st[2]);
            out.runner_up
                .push((id[1] < V as u32).then_some((id[1], st[1])));
        }
        Ok(out)
    }
}
pub(super) fn attention(
    c: &Commands<'_>,
    s: &Scratch,
    k: &Buffer,
    v: &Buffer,
    stride: usize,
    rows: usize,
    splits: usize,
    cross: bool,
) {
    c.dispatch(
        "wh_decode",
        &[&s.q, k, v, &s.slots, &s.positions, &s.partial],
        &[stride as u32, cross as u32, splits as u32],
        [20, rows, splits],
        128,
    );
    c.dispatch(
        "wh_merge",
        &[&s.partial, &s.attn],
        &[splits as u32],
        [20, rows, 1],
        32,
    );
}
