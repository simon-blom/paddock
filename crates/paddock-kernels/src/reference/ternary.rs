//! CPU test reference for PrismML's two ternary packings, PTQ1_0 and PQ2_0 -
//! the tensor types of the Bonsai GGUF files. Written from the block layouts
//! (the fork's `ggml-common.h`; the byte order and the base-3 codec are the
//! format itself), not from anyone's decoder.
//!
//! Both hold 128 weights from {-1, 0, +1} under one f16 scale:
//!
//!   PQ2_0  (34 B): f16 d | 32 B of 2-bit slots, element j in bits 2*(j%4) of
//!          byte j/4; weight = (slot - 1) * d. Slot 3 decodes as +2: the codec
//!          allows it, a ternary checkpoint never writes it.
//!   PTQ1_0 (28 B): qs[24] five trits a byte | qh[2] four trits a byte | f16 d.
//!
//! A PTQ1_0 byte is a base-3 fraction of 256 - ggml TQ1_0's codec at half the
//! block. For the 5-trit value `v` (first trit most significant) the byte is
//! `ceil(v * 256 / 243)`, and trit n reads back as the high byte of
//! `((b * 3^n) mod 256) * 3`. qh's four trits sit in the top four positions.
//! Element order is not positional, it follows a 4-wide walk:
//!
//!   element n*16 + m       is trit n of qs[m]        (m in 0..16)
//!   element 80 + n*8 + m   is trit n of qs[16 + m]   (m in 0..8)
//!   element 120 + n*2 + h  is trit n of qh[h]        (n in 0..4)
//!
//! so four neighbouring bytes at one trit position are four neighbouring
//! weights. Numerics: a trit times an f16 scale is exact in f32, which is why
//! the pack's gate against this module is bit identity and not a tolerance.

use super::DequantError;

/// GGUF raw type ids.
pub const PQ2_0: u32 = 142;
pub const PTQ1_0: u32 = 143;

pub const TERNARY_BLOCK_ELEMS: usize = 128;
pub const PQ2_0_BLOCK_BYTES: usize = 34;
pub const PTQ1_0_BLOCK_BYTES: usize = 28;

const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Raw bytes per 128-weight block, or None when `raw_type` is not ternary.
pub fn ternary_block_bytes(raw_type: u32) -> Option<usize> {
    match raw_type {
        PQ2_0 => Some(PQ2_0_BLOCK_BYTES),
        PTQ1_0 => Some(PTQ1_0_BLOCK_BYTES),
        _ => None,
    }
}

/// Trit `n` of a packed byte, as its stored code 0..=2 (weight + 1).
pub fn trit_code(byte: u8, n: usize) -> u8 {
    ((byte.wrapping_mul(POW3[n]) as u16 * 3) >> 8) as u8
}

/// The (byte index within the 26 packed bytes, trit position) holding element
/// `e` of a PTQ1_0 block. Bytes 0..24 are `qs`, 24..26 are `qh`.
pub fn ptq1_0_site(e: usize) -> (usize, usize) {
    match e {
        0..80 => (e % 16, e / 16),
        80..120 => (16 + (e - 80) % 8, (e - 80) / 8),
        _ => (24 + (e - 120) % 2, (e - 120) / 2),
    }
}

/// Weight `e` of a PTQ1_0 block as -1, 0 or +1.
pub fn ptq1_0_weight(block: &[u8], e: usize) -> i8 {
    let (byte, n) = ptq1_0_site(e);
    trit_code(block[byte], n) as i8 - 1
}

/// PTQ1_0 / PQ2_0 -> f32.
pub fn dequant_ternary(raw_type: u32, data: &[u8], out: &mut [f32]) -> Result<(), DequantError> {
    let (kind, bb) = match raw_type {
        PQ2_0 => ("pq2_0", PQ2_0_BLOCK_BYTES),
        _ => ("ptq1_0", PTQ1_0_BLOCK_BYTES),
    };
    if !data.len().is_multiple_of(bb) {
        return Err(DequantError::BadInputLen {
            kind,
            input: data.len(),
            block: bb,
        });
    }
    let expected = data.len() / bb * TERNARY_BLOCK_ELEMS;
    if out.len() != expected {
        return Err(DequantError::BadOutputLen {
            out: out.len(),
            expected,
        });
    }
    let rows = out.as_chunks_mut::<TERNARY_BLOCK_ELEMS>().0;
    for (block, y) in data.chunks_exact(bb).zip(rows) {
        if raw_type == PQ2_0 {
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for (j, w) in y.iter_mut().enumerate() {
                let slot = (block[2 + j / 4] >> (2 * (j % 4))) & 3;
                *w = (slot as i32 - 1) as f32 * d;
            }
        } else {
            let d = half::f16::from_le_bytes([block[26], block[27]]).to_f32();
            for (e, w) in y.iter_mut().enumerate() {
                *w = ptq1_0_weight(block, e) as f32 * d;
            }
        }
    }
    Ok(())
}

/// Pack 128 weights from {-1, 0, +1} into a PTQ1_0 block: the writer half of
/// the codec, here so GPU gates can build planes without a model file.
pub fn encode_ptq1_0(weights: &[i8], d: half::f16) -> [u8; PTQ1_0_BLOCK_BYTES] {
    assert_eq!(weights.len(), TERNARY_BLOCK_ELEMS);
    let mut digits = [[1u8; 5]; 26];
    for (e, &w) in weights.iter().enumerate() {
        assert!((-1..=1).contains(&w), "weight {w} is not a trit");
        let (byte, n) = ptq1_0_site(e);
        digits[byte][n] = (w + 1) as u8;
    }
    let mut block = [0u8; PTQ1_0_BLOCK_BYTES];
    for (b, ds) in digits.iter().enumerate() {
        // qh holds four trits in the top four positions; its fifth digit is 0
        let width = if b < 24 { 5 } else { 4 };
        let mut v = 0u32;
        for (n, &digit) in ds.iter().enumerate() {
            v = v * 3 + if n < width { digit as u32 } else { 0 };
        }
        block[b] = (v * 256).div_ceil(243) as u8;
    }
    block[26..].copy_from_slice(&d.to_le_bytes());
    block
}

/// Pack 128 weights from {-1, 0, +1} into a PQ2_0 block.
pub fn encode_pq2_0(weights: &[i8], d: half::f16) -> [u8; PQ2_0_BLOCK_BYTES] {
    assert_eq!(weights.len(), TERNARY_BLOCK_ELEMS);
    let mut block = [0u8; PQ2_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&d.to_le_bytes());
    for (j, &w) in weights.iter().enumerate() {
        assert!((-1..=1).contains(&w), "weight {w} is not a trit");
        block[2 + j / 4] |= ((w + 1) as u8) << (2 * (j % 4));
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(seed: u32) -> Vec<i8> {
        let mut s = seed;
        (0..TERNARY_BLOCK_ELEMS)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 24) % 3) as i8 - 1
            })
            .collect()
    }

    #[test]
    fn every_byte_value_the_codec_writes_reads_back() {
        for v in 0..243u32 {
            let byte = (v * 256).div_ceil(243) as u8;
            let mut rest = v;
            for n in (0..5).rev() {
                assert_eq!(trit_code(byte, n) as u32, rest % 3, "value {v} trit {n}");
                rest /= 3;
            }
        }
    }

    // Bytes worked out by hand from the layout, so the element order is pinned
    // to the format and not just to this module's own encoder.
    #[test]
    fn ptq1_0_hand_packed_block() {
        let mut block = [128u8; PTQ1_0_BLOCK_BYTES]; // trits 1,1,1,1,1: five zeros
        block[24] = 127; // qh trits 1,1,1,1
        block[25] = 155; // qh trits 1,2,1,1: element 120 + 1*2 + 1 is +1
        block[0] = 213; // trits 2,1,1,1,1: element 0 is +1
        block[19] = 118; // trits 1,1,0,1,1: element 80 + 2*8 + 3 is -1
        block[26..].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        let mut out = [9f32; 128];
        dequant_ternary(PTQ1_0, &block, &mut out).expect("dequants");
        for (e, &w) in out.iter().enumerate() {
            let want = match e {
                0 | 123 => 0.25,
                99 => -0.25,
                _ => 0.0,
            };
            assert_eq!(w, want, "element {e}");
        }
    }

    #[test]
    fn pq2_0_hand_packed_block() {
        let mut block = [0x55u8; PQ2_0_BLOCK_BYTES]; // every slot 1: zeros
        block[..2].copy_from_slice(&half::f16::from_f32(-0.5).to_le_bytes());
        block[2] = 0b01_01_10_00; // element 0 is -1, element 1 is +1
        block[2 + 31] = 0b00_01_01_01; // element 127 is -1
        let mut out = [9f32; 128];
        dequant_ternary(PQ2_0, &block, &mut out).expect("dequants");
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], -0.5);
        assert_eq!(out[127], 0.5);
        assert!(out[2..127].iter().all(|&w| w == 0.0));
    }

    #[test]
    fn both_packings_round_trip_and_agree() {
        for seed in 0..32 {
            let w = pattern(seed);
            let d = half::f16::from_f32(0.0078125 * (seed + 1) as f32);
            let (mut a, mut b) = ([0f32; 128], [0f32; 128]);
            dequant_ternary(PTQ1_0, &encode_ptq1_0(&w, d), &mut a).expect("dequants");
            dequant_ternary(PQ2_0, &encode_pq2_0(&w, d), &mut b).expect("dequants");
            for e in 0..128 {
                assert_eq!(a[e], w[e] as f32 * d.to_f32(), "seed {seed} element {e}");
                assert_eq!(a[e].to_bits(), b[e].to_bits());
            }
        }
    }

    #[test]
    fn length_mismatches_are_errors() {
        let mut out = [0f32; 128];
        assert!(dequant_ternary(PTQ1_0, &[0u8; 27], &mut out).is_err());
        assert!(dequant_ternary(PQ2_0, &[0u8; 34], &mut out[..127]).is_err());
        assert_eq!(ternary_block_bytes(PTQ1_0), Some(28));
        assert_eq!(ternary_block_bytes(12), None);
    }
}
