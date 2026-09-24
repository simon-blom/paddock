//! BF16 checkpoint attention, independent of Granite's F16 cache/rotary layout.
use super::*;
impl Granite {
    pub(super) fn mlx_attention(
        &self,
        cmd: &Commands<'_>,
        layer: &Layer,
        rows: usize,
        tiles: usize,
        decode: usize,
        splits: usize,
    ) {
        let s = &self.scratch;
        cmd.dispatch(
            "llama_mlx_rope",
            &[
                &s.q,
                &s.k,
                &s.v,
                &layer.keys,
                &layer.values,
                &s.meta,
                &s.pages,
            ],
            &[
                self.width as u32,
                self.kv_width as u32,
                128,
                rows as u32,
                self.page_stride as u32,
                self.rope.to_bits(),
            ],
            [
                (rows * (self.width + self.kv_width) / 2).div_ceil(256),
                1,
                1,
            ],
            256,
        );
        let p = [
            self.heads as u32,
            self.kv_heads as u32,
            self.page_stride as u32,
            0,
            0,
            splits as u32,
        ];
        if tiles > 0 {
            cmd.dispatch(
                "llama_mlx_prefill",
                &[
                    &s.q,
                    &layer.keys,
                    &layer.values,
                    &s.meta,
                    &s.pages,
                    &s.attn,
                    &s.attention_tiles,
                ],
                &p,
                [self.heads, tiles, 1],
                128,
            );
        }
        if decode > 0 {
            cmd.dispatch(
                "llama_mlx_decode",
                &[
                    &s.q,
                    &layer.keys,
                    &layer.values,
                    &s.meta,
                    &s.pages,
                    &s.attention_rows,
                    &s.attn_parts,
                ],
                &p,
                [self.kv_heads, decode, splits],
                128,
            );
            cmd.dispatch(
                "muse_merge",
                &[&s.attn_parts, &s.attn, &s.attention_rows],
                &[self.heads as u32, splits as u32, 128],
                [self.heads * decode, 1, 1],
                32,
            );
        }
        cmd.dispatch(
            "gmlx_round",
            &[&s.attn],
            &[(rows * self.width) as u32],
            [(rows * self.width).div_ceil(256), 1, 1],
            256,
        );
    }
}
