//! Grow row scratch on demand, never by maximum image resolution at load. Every
//! allocation is charged to the existing Metal grant; failures are explicit.
use super::*;

impl Qwen35 {
    pub(super) fn reserve_rows(&mut self, rows: usize) -> Result<()> {
        let g = self.geometry;
        if rows <= self.row_capacity {
            return Ok(());
        }
        let n = rows.div_ceil(128) * 128;
        let chunks = n.div_ceil(32) + self.slots.len() * 3;
        // Only large row batches elect transient unpacking. One largest
        // FFN matrix is enough; normal bounded grants reserve no slab.
        let unpack = if n >= 1024 { self.width * self.ff } else { 0 };
        let a = |width| self.device.alloc(n * width * 4);
        let replacement = Scratch {
            gemm: self.device.alloc(if self.splash {
                crate::splash::workspace_bytes(n, g.projection_width(), self.ff)
            } else {
                ((n + 128) * g.projection_width() + unpack) * 2
            })?,
            ids: a(1)?,
            outputs: a(1)?,
            meta: a(2)?,
            mrope: a(4)?,
            limits: a(1)?,
            pages: self.device.alloc(self.slots.len() * self.page_stride * 4)?,
            spans: a(4)?,
            chunks: self.device.alloc(chunks * 16)?,
            bounds: a(2)?,
            attn_tiles: a(2)?,
            decode_rows: a(1)?,
            long_decode_tiles: a(2)?,
            checkpoint_rows: a(1)?,
            checkpoint_spans: a(4)?,
            x: a(self.width)?,
            norm: a(self.width)?,
            delta: a(self.width)?,
            gate: a(self.ff)?,
            up: a(self.ff)?,
            logits: self.device.alloc(self.slots.len() * self.vocab * 4)?,
            qraw: a(g.heads * 512)?,
            q: a(g.heads * 256)?,
            k: a(g.kv_heads * 256)?,
            v: a(g.kv_heads * 256)?,
            attn: a(g.heads * 256)?,
            // Only short (<16-row) spans use split decode. The parts plane is
            // densely indexed by selected decode row, never total prompt size.
            attn_parts: self.device.alloc(
                (self.slots.len() * 15 * g.heads * MAX_SPLITS * 258 * 4).max(if self.splash {
                    n * g.heads * 4 * 258 * 4
                } else {
                    0
                }),
            )?,
            qkv: a(g.conv())?,
            convolved: a(g.conv())?,
            z: a(g.value_heads * 128)?,
            alpha: a(g.value_heads)?,
            beta: a(g.value_heads)?,
            gates: a(g.value_heads * 2)?,
            prepared: self.device.alloc(
                (chunks * g.value_heads * 17408 * g.prepared_element_bytes()).max(if self.splash {
                    crate::splash::attention_scratch_bytes(chunks, g.heads)
                } else {
                    0
                }),
            )?,
        };
        self.reserve_mtp_rows(n)?;
        self.reserve_dflash_rows(n)?;
        let moe_scratch = if g.moe() {
            Some(moe::Workspace::new(&self.device, n)?)
        } else {
            None
        };
        if let Some(ternary) = &mut self.ternary {
            ternary.reserve_rows(&self.device, n)?;
        }
        self.scratch = replacement;
        self.moe_scratch = moe_scratch;
        self.row_capacity = n;
        Ok(())
    }
}
