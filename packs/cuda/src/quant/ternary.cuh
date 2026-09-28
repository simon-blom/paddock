// quant/ternary.cuh - the batch-1 decode lane of PrismML's dense ternary
// packing PTQ1_0 (GGUF raw id 143, the Bonsai files), and the activation
// quantizer it eats.
//
// Why a lane of its own. PTQ1_0 packs five trits into a byte as a base-3
// fraction of 256, so reading trit n back is ((q * 3^n) & 0xFF) * 3 >> 8 -
// arithmetic, where every other format on these streams unpacks with shifts
// and byte permutes. The generic 16-weight window lane pays that per window:
// ~10 register ops per 4 weights before the dp4a, and a 64-weight chunk reads
// its block's 16 head bytes twice. Measured on the A6000 (sm_86, 768 GB/s):
// 266-282 GB/s of weights on the ffn-gate shape, where the 2-bit-slot twin
// PQ2_0 ran 528 on the same lane. This lane runs the same shape at ~450 and
// the wide-input ffn-down at ~500; what was tried and lost on the way is
// noted below where it bears on the code.
//
// The walk turns the digit arithmetic into ONE table read per weight byte,
// and it can because of how the format orders its elements: the five trits
// of head byte m are elements m, 16+m, 32+m, 48+m, 64+m of the block - not
// five neighbours. So instead of regrouping trits to meet the activations,
// the ACTIVATIONS are staged once per launch in the weights' own byte order:
// the operand of byte j is the four int8 activations its first four trits
// multiply, packed as one dp4a word - a byte TRANSPOSE of four activation
// words, since trit n of byte m is element 16n + m - and the fifth trits of
// four neighbouring bytes meet elements 64+4g..64+4g+3, one contiguous word
// as it lies. A weight byte is then
//     si = dp4a(LO[byte], XA[j], si)       LO: 256 x 4 B in shared memory
// and per four bytes one arithmetic digit extraction (pd_trit_digit4 with
// 3^4, registers only) and one more dp4a for their fifth trits. Every weight
// byte is read exactly once - a lane owns whole 256-weight super-blocks - and
// nothing is rebuilt into a wider form: the plane stays 1.75 bpw resident.
//
// The lane is bound by dependent GATHERS from shared memory, not by register
// ops: a second table for the fifth trit lost 17% to the arithmetic, and
// halving the table's bytes moved nothing. Price changes here in gathers
// per weight.
//
// The price is the activation scale. A byte's trits straddle three of the
// per-32 activation groups the int8 lanes quantize in, and an integer dot
// cannot carry three scales, so this lane takes ONE int8 scale per 128
// activations - the weight block. That is a coarser class than llama.cpp's
// Q8_1 (per 32) and it is only sound because every input of this model is
// Hadamard-rotated first: a rotated row is outlier-poor by construction
// (E|max| over 128 Gaussians is 3.3 sigma against 2.9 over 32, a 15% wider
// step). It is still finer than BitNet's own per-token int8 activations. The
// same-weights greedy parity against PrismML's fork is the judge, per the
// sub-4-bit rule - and it did not move.
//
// Shape: NT = 256, a warp per row, lane q the class's leaf q (below). The
// A6000 walk this replaced carried 8 rows a warp over 4 stripes with one
// accumulator a lane, and "fewer rows a warp" measured dead THERE; on GB10's
// LPDDR5X the same walk ran at half the die's read roof and the warp-row walk
// is what streams (numbers at pd_trn_row). The class fixes the fold, not the
// geometry, so a die may take any rows-a-warp without moving a bit.
// Study reference for the byte-owning idea: PrismML's Metal kernel
// (kernel_mul_mv_ptq1_0, MIT). Nothing here is theirs.

#define PD_TRN_NT 256u
// staged words per super-block: per 128-weight block 26 byte operands (16
// head, 8 tail, qh[2]) padded to 28 and 6 fifth-trit words padded to 8, so
// every group of four is one aligned 16-byte load; then the two scales
#define PD_TRN_XA 56u
#define PD_TRN_XB 16u

// Where the staged words of super-block s live - PSB, a template parameter:
//   true   per super-block: the four operands of a weight word contiguous,
//          one 16-byte load a word (what serving runs)
//   false  column-major, word (j, s) at j * ns + s: the lanes of one step read
//          consecutive words, each operand its own load
// The two tie when served; the per-super-block form wins the wide-input plane
// in isolation (ffn down 39.2 vs 43.0 us) and never loses, and it has a
// quarter of the operand loads. The column-major form stays as the probe's
// comparand for the next die (bit 20 of `dtype` asks for it for one call).
template <bool PSB>
__device__ __forceinline__ uint32_t pd_trn_ia(uint32_t ns, uint32_t s, uint32_t bl, uint32_t k) {
    return PSB ? s * PD_TRN_XA + bl * 28u + k : (bl * 26u + k) * ns + s;
}
template <bool PSB>
__device__ __forceinline__ uint32_t pd_trn_ib(uint32_t ns, uint32_t s, uint32_t bl, uint32_t q) {
    return PSB ? s * PD_TRN_XB + bl * 8u + q : (bl * 6u + q) * ns + s;
}

// LO[byte]: trits 0..3 as s8 (digit - 1). The digit formula is the format's
// own, so a byte that is not a canonical encoding decodes exactly as the
// window unpack decodes it.
__host__ __device__ constexpr uint32_t pd_trn_lut_entry(uint32_t b) {
    uint32_t e = 0u, p = 1u;
    for (uint32_t n = 0u; n < 4u; ++n) {
        const uint32_t digit = (((b * p) & 0xFFu) * 3u) >> 8u;
        e |= ((digit - 1u) & 0xFFu) << (8u * n);
        p *= 3u;
    }
    return e;
}

// shared bytes a block needs: the table, then per super-block the staged
// operands and the two per-128 activation scales
__host__ __device__ __forceinline__ uint32_t pd_trn_smem_bytes(uint32_t in_dim) {
    return 1024u + (in_dim >> 8u) * (PD_TRN_XA + PD_TRN_XB + 2u) * 4u;
}

// an f16 held in the low half of a register
__device__ __forceinline__ float pd_trn_f16(uint32_t bits) {
    __half h;
    const uint16_t u = (uint16_t)bits;
    memcpy(&h, &u, 2u);
    return __half2float(h);
}

// 4x4 byte transpose: out[i * stride] = (a[i], b[i], c[i], d[i])
__device__ __forceinline__ void pd_trn_tr4(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                                           int* __restrict__ out, uint32_t stride) {
    const uint32_t t0 = __byte_perm(a, b, 0x5140u), t1 = __byte_perm(a, b, 0x7362u);
    const uint32_t t2 = __byte_perm(c, d, 0x5140u), t3 = __byte_perm(c, d, 0x7362u);
    out[0] = (int)__byte_perm(t0, t2, 0x5410u);
    out[stride] = (int)__byte_perm(t0, t2, 0x7632u);
    out[2u * stride] = (int)__byte_perm(t1, t3, 0x5410u);
    out[3u * stride] = (int)__byte_perm(t1, t3, 0x7632u);
}

// Stage activation block blk = 2s + bl of the row: its 128 int8 come in as
// eight 16-byte loads, and every staged operand is a byte transpose of four
// of those words - the operands of bytes 4g..4g+3 are the transpose of word g
// of chunks 0..3, and their fifth trits meet word g of chunk 4 as it lies.
// (The first cut assembled each operand from four single-byte loads, ~130
// scattered loads a block in EVERY resident block, and the launch was
// staging-bound on planes with few rows: ffn down 69 us, 52 with this.)
template <bool PSB>
__device__ __forceinline__ void pd_trn_stage_block(
        int* __restrict__ sxa, int* __restrict__ sxb, float* __restrict__ sxs,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        uint32_t ns, uint32_t blk) {
    const uint32_t s = blk >> 1u, bl = blk & 1u;
    const uint32_t st = PSB ? 1u : ns;  // stride between a word's four operands
    const uint8_t* x = reinterpret_cast<const uint8_t*>(xq) + (size_t)blk * 128u;
    uint32_t c[8][4];
    #pragma unroll
    for (uint32_t n = 0; n < 8u; ++n) {
        const uint4 v = pd_iq_ld16(x + 16u * n);
        c[n][0] = v.x; c[n][1] = v.y; c[n][2] = v.z; c[n][3] = v.w;
    }
    // head bytes: trits 0..3 of byte m hit m, 16+m, 32+m, 48+m; trit 4 hits 64+m
    #pragma unroll
    for (uint32_t g = 0; g < 4u; ++g) {
        pd_trn_tr4(c[0][g], c[1][g], c[2][g], c[3][g], sxa + pd_trn_ia<PSB>(ns, s, bl, 4u * g), st);
        sxb[pd_trn_ib<PSB>(ns, s, bl, g)] = (int)c[4][g];
    }
    // tail bytes: trit n of qs[16 + m] is element 80 + 8n + m
    #pragma unroll
    for (uint32_t g = 0; g < 2u; ++g) {
        pd_trn_tr4(c[5][g], c[5][2u + g], c[6][g], c[6][2u + g],
                   sxa + pd_trn_ia<PSB>(ns, s, bl, 16u + 4u * g), st);
        sxb[pd_trn_ib<PSB>(ns, s, bl, 4u + g)] = (int)c[7][g];
    }
    // qh[h]: trit n is element 120 + 2n + h (four trits, no fifth)
    sxa[pd_trn_ia<PSB>(ns, s, bl, 24u)] = (int)__byte_perm(c[7][2], c[7][3], 0x6420u);
    sxa[pd_trn_ia<PSB>(ns, s, bl, 25u)] = (int)__byte_perm(c[7][2], c[7][3], 0x7531u);
    sxs[blk] = xs[blk];
}

// The table and the row's operands into a block's shared memory (every
// thread of the block calls it; the caller syncs).
template <bool PSB>
__device__ __forceinline__ void pd_trn_stage(
        uint8_t* __restrict__ smem, const int8_t* __restrict__ xq,
        const float* __restrict__ xs, uint32_t ns, uint32_t nt,
        uint32_t** lut, int** sxa, int** sxb, float** sxs) {
    *lut = reinterpret_cast<uint32_t*>(smem);
    *sxa = reinterpret_cast<int*>(smem + 1024u);
    *sxb = *sxa + (size_t)PD_TRN_XA * ns;
    *sxs = reinterpret_cast<float*>(*sxb + (size_t)PD_TRN_XB * ns);
    const uint32_t tid = threadIdx.x;
    (*lut)[tid] = pd_trn_lut_entry(tid);  // nt = 256: one entry a thread
    for (uint32_t blk = tid; blk < 2u * ns; blk += nt)
        pd_trn_stage_block<PSB>(*sxa, *sxb, *sxs, xq, xs, ns, blk);
}

// four weight bytes against their four operands, then their fifth trits -
// by arithmetic, in registers - against one more
__device__ __forceinline__ int pd_trn_word(const uint32_t* __restrict__ lut, uint32_t w,
                                           int x0, int x1, int x2, int x3, int xb, int si) {
    si = __dp4a((int)lut[w & 0xFFu], x0, si);
    si = __dp4a((int)lut[(w >> 8u) & 0xFFu], x1, si);
    si = __dp4a((int)lut[(w >> 16u) & 0xFFu], x2, si);
    si = __dp4a((int)lut[w >> 24u], x3, si);
    return __dp4a(pd_trit_digit4(w, 81u), xb, si);
}

// one 128-weight block of a super-block: 16 head bytes, 8 tail bytes, qh[2]
template <bool PSB>
__device__ __forceinline__ int pd_trn_block(const uint32_t* __restrict__ lut,
                                            const uint32_t hw[4], uint32_t t0, uint32_t t1,
                                            uint32_t qh, const int* __restrict__ sxa,
                                            const int* __restrict__ sxb, uint32_t ns,
                                            uint32_t s, uint32_t bl) {
    int si = 0;
    if (PSB) {
        const int* xa = sxa + pd_trn_ia<PSB>(ns, s, bl, 0u);
        const int* xb = sxb + pd_trn_ib<PSB>(ns, s, bl, 0u);
        const int4 f0 = *reinterpret_cast<const int4*>(xb);
        const int4 f1 = *reinterpret_cast<const int4*>(xb + 4);
        const int f[6] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y};
        const uint32_t w[6] = {hw[0], hw[1], hw[2], hw[3], t0, t1};
        #pragma unroll
        for (uint32_t g = 0; g < 6u; ++g) {
            const int4 a = *reinterpret_cast<const int4*>(xa + 4u * g);
            si = pd_trn_word(lut, w[g], a.x, a.y, a.z, a.w, f[g], si);
        }
        const int4 q4 = *reinterpret_cast<const int4*>(xa + 24);
        si = __dp4a((int)lut[qh & 0xFFu], q4.x, si);
        return __dp4a((int)lut[(qh >> 8u) & 0xFFu], q4.y, si);
    }
    const uint32_t w[6] = {hw[0], hw[1], hw[2], hw[3], t0, t1};
    #pragma unroll
    for (uint32_t g = 0; g < 6u; ++g) {
        const int* xa = sxa + pd_trn_ia<PSB>(ns, s, bl, 4u * g);
        si = pd_trn_word(lut, w[g], xa[0], xa[ns], xa[2u * ns], xa[3u * ns],
                         sxb[pd_trn_ib<PSB>(ns, s, bl, g)], si);
    }
    si = __dp4a((int)lut[qh & 0xFFu], sxa[pd_trn_ia<PSB>(ns, s, bl, 24u)], si);
    return __dp4a((int)lut[(qh >> 8u) & 0xFFu], sxa[pd_trn_ia<PSB>(ns, s, bl, 25u)], si);
}

// ---- the ternary class ------------------------------------------------------
// Every walk that serves a PTQ1_0 plane - the batch-1 walk below, the
// tensor-core NB walk after it - produces the SAME bits for a row, whatever
// the column count or the walk's geometry. What makes that possible: a
// 128-block's dot is an exact integer (|sum| <= 128 * 127, any order), so
// only the f32 fold is order-sensitive, and it is fixed here:
//   - slices of 32 super-blocks, s in [32k, 32k + 32);
//   - leaf q of slice k = the two block terms of super-block 32k + q, block
//     0 then 1, each leaf = fma(f16(d) * xs, (float)isum, leaf) from 0.0,
//     the multiply and the fma explicit (no contraction left to the compiler);
//   - F_k = the down-shuffle tree over the 32 leaves (offsets 16, 8, 4, 2, 1);
//   - y = F_0 + F_1 + ... in order.
// Every plane of the 27B but ffn down (68 super-blocks: three slices) is one
// slice. Slices fold separately so a walk may split K across CTAs at slice
// boundaries and still land on these bits.
// Why it matters: a verify row (speculative decoding), a decode row sharing a
// tick with seven others and the row decoding alone all score one token the
// same way - the same class the Nemotron W16 lanes hold.
__device__ __forceinline__ float pd_trn_term(uint32_t dbits, float xs, int isum, float leaf) {
    return __fmaf_rn(__fmul_rn(pd_trn_f16(dbits), xs), (float)isum, leaf);
}
// the gate | up launches' SwiGLU, one expression for every walk
__device__ __forceinline__ float pd_trn_swiglu(float g, float u) {
    return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u);
}

// streaming (evict-first) loads of the plane - always global memory, so no
// generic-address select (pd_iq_ld16's)
__device__ __forceinline__ uint4 pd_trn_ld16(const uint8_t* p) {
    return __ldcs(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ uint2 pd_trn_ld8(const uint8_t* p) {
    return __ldcs(reinterpret_cast<const uint2*>(p));
}

// One row by the class, a warp per row: lane q is leaf q, so slice k's lane q
// takes super-block 32k + q over the repacked streams (data: b0.qs[0..16],
// b1.qs[0..16], b0.qs[16..24], b1.qs[16..24]; record: d0, d1, qh0[2],
// qh1[2]). The result lands in lane 0.
//
// GB10 (48 SMs, LPDDR5X ~273 GB/s), bench/ternary_gb10_bench.cu, DRAM-cold:
// the A6000-tuned walk this replaced (8 rows a warp over 4 stripes, one
// accumulator a lane) ran 118-133 GB/s on the 27B's planes - behind even the
// generic window lane (184-203). The warp-row walk puts a row's whole slice
// in flight at once (20 lanes x 56 B on a 5120-wide plane) and sends the
// CTA's first rows to L2 as bulk prefetches ahead of the staging prologue
// (pd_trn_pf_row): 203-221 GB/s as a bench-local kernel, 155-166 as compiled
// here - ptxas sinks each 16-byte load next to its first use (SASS: one LDG,
// a dp4a chain, the next LDG), every one exposing an LPDDR5X latency, and
// neither hoisting the loads out of the decode's branch, asm volatile loads
// nor a plain __ldcs moved that schedule. It stays the class's reference;
// the NB lane below out-streams it even at one column and serves every
// width.
template <bool PSB>
__device__ __forceinline__ float pd_trn_row(
        const uint8_t* __restrict__ row, const uint8_t* __restrict__ rec,
        const uint32_t* __restrict__ lut, const int* __restrict__ sxa,
        const int* __restrict__ sxb, const float* __restrict__ sxs, uint32_t ns, uint32_t lane) {
    float y = 0.0f;
    for (uint32_t base = 0; base < ns; base += 32u) {
        const uint32_t s = base + lane;
        const bool on = s < ns;
        // the super-block's loads, written ahead of the decode (a lane past
        // the row's end re-reads super-block `base` - the lines lane 0 of
        // the warp asks for); ptxas still schedules them next to their uses
        // (see above)
        const uint8_t* sb = row + (size_t)(on ? s : base) * 48u;
        const uint4 h0 = pd_trn_ld16(sb), h1 = pd_trn_ld16(sb + 16u), tl = pd_trn_ld16(sb + 32u);
        const uint2 rc = pd_trn_ld8(rec + (size_t)(on ? s : base) * 8u);
        float leaf = 0.0f;
        if (on) {
            const uint32_t hw0[4] = {h0.x, h0.y, h0.z, h0.w};
            const uint32_t hw1[4] = {h1.x, h1.y, h1.z, h1.w};
            const int s0 = pd_trn_block<PSB>(lut, hw0, tl.x, tl.y, rc.y & 0xFFFFu, sxa, sxb, ns, s, 0u);
            const int s1 = pd_trn_block<PSB>(lut, hw1, tl.z, tl.w, rc.y >> 16u, sxa, sxb, ns, s, 1u);
            leaf = pd_trn_term(rc.x & 0xFFFFu, sxs[2u * s], s0, leaf);
            leaf = pd_trn_term(rc.x >> 16u, sxs[2u * s + 1u], s1, leaf);
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            leaf = __fadd_rn(leaf, __shfl_down_sync(0xffffffffu, leaf, off));
        y = base == 0u ? leaf : __fadd_rn(y, leaf);
    }
    return y;
}

// a warp's first row to L2 before the staging (a hint - no numerics, and
// safe ahead of the PDL wait: the weights are no predecessor's output)
__device__ __forceinline__ void pd_trn_pf_row(const uint8_t* data, const uint8_t* rec,
                                              size_t rdb, size_t rsb, uint32_t o) {
    pd_l2_prefetch_bulk(data + (size_t)o * rdb, (uint32_t)rdb);
    pd_l2_prefetch_bulk(rec + (size_t)o * rsb, (uint32_t)rsb);
}

// y[o] = row_o . x; xq int8 [in_dim], xs f32 [in_dim / 128]. A warp per row.
template <bool PSB>
__global__ void __launch_bounds__(PD_TRN_NT) pd_trn_gemv_b128_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t NT = PD_TRN_NT, WARPS = NT / 32u;
    extern __shared__ uint8_t pd_trn_smem[];
    const uint32_t ns = in_dim >> 8u;
    const size_t rdb = (size_t)ns * 48u, rsb = (size_t)ns * 8u;
    const uint32_t lane = threadIdx.x & 31u, o0 = blockIdx.x * WARPS + (threadIdx.x >> 5u);
    if (lane == 0u && o0 < out_dim) pd_trn_pf_row(data, scales, rdb, rsb, o0);
    PD_PDL_ARM();
    uint32_t* lut; int* sxa; int* sxb; float* sxs;
    pd_trn_stage<PSB>(pd_trn_smem, xq, xs, ns, NT, &lut, &sxa, &sxb, &sxs);
    __syncthreads();
    for (uint32_t o = o0; o < out_dim; o += gridDim.x * WARPS) {
        const float v = pd_trn_row<PSB>(data + (size_t)o * rdb, scales + (size_t)o * rsb, lut, sxa,
                                        sxb, sxs, ns, lane);
        if (lane == 0u) y[o] = v;
    }
}

// Several planes that read the SAME staged row, in one launch: the rows of up
// to three planes laid end to end (q | k | v; in_qkv | gate), or - GLU - a
// gate | up pair walked together with y[o] = silu(gate_o . x) * (up_o . x),
// which also takes the SwiGLU launch with it. One staging instead of one per
// plane, one ramp/drain toll instead of three (wk / wv are 1024 rows: their
// own launches are nearly all toll). Per row this is the single-plane walk -
// the class, bit for bit.
template <bool GLU>
__global__ void __launch_bounds__(PD_TRN_NT) pd_trn_multi_b128_kernel(
        const uint8_t* __restrict__ d0, const uint8_t* __restrict__ r0,
        const uint8_t* __restrict__ d1, const uint8_t* __restrict__ r1,
        const uint8_t* __restrict__ d2, const uint8_t* __restrict__ r2,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        uint32_t in_dim, uint32_t o0, uint32_t o1, uint32_t o2) {
    constexpr uint32_t NT = PD_TRN_NT, WARPS = NT / 32u;
    extern __shared__ uint8_t pd_trn_smem[];
    const uint32_t ns = in_dim >> 8u;
    const size_t rdb = (size_t)ns * 48u, rsb = (size_t)ns * 8u;
    const uint32_t lane = threadIdx.x & 31u, first = blockIdx.x * WARPS + (threadIdx.x >> 5u);
    const uint32_t total = GLU ? o0 : o0 + o1 + o2;
    auto plane = [&](uint32_t o, const uint8_t** dd, const uint8_t** rr, float** yy) {
        const uint32_t pl = GLU ? 0u : (o < o0 ? 0u : (o < o0 + o1 ? 1u : 2u));
        *dd = pl == 0u ? d0 : (pl == 1u ? d1 : d2);
        *rr = pl == 0u ? r0 : (pl == 1u ? r1 : r2);
        *yy = pl == 0u ? y0 : (pl == 1u ? y1 : y2);
        return pl == 0u ? o : (pl == 1u ? o - o0 : o - o0 - o1);
    };
    if (lane == 0u && first < total) {
        const uint8_t* dd; const uint8_t* rr; float* yy;
        const uint32_t po = plane(first, &dd, &rr, &yy);
        pd_trn_pf_row(dd, rr, rdb, rsb, po);
        if (GLU) pd_trn_pf_row(d1, r1, rdb, rsb, po);
    }
    PD_PDL_ARM();
    uint32_t* lut; int* sxa; int* sxb; float* sxs;
    pd_trn_stage<true>(pd_trn_smem, xq, xs, ns, NT, &lut, &sxa, &sxb, &sxs);
    __syncthreads();
    for (uint32_t o = first; o < total; o += gridDim.x * WARPS) {
        const uint8_t* dd; const uint8_t* rr; float* yy;
        const uint32_t po = plane(o, &dd, &rr, &yy);
        const float a = pd_trn_row<true>(dd + (size_t)po * rdb, rr + (size_t)po * rsb, lut, sxa,
                                         sxb, sxs, ns, lane);
        if (GLU) {
            const float u = pd_trn_row<true>(d1 + (size_t)po * rdb, r1 + (size_t)po * rsb, lut,
                                             sxa, sxb, sxs, ns, lane);
            if (lane == 0u) y0[po] = pd_trn_swiglu(a, u);
        } else if (lane == 0u) {
            yy[po] = a;
        }
    }
}

// persistent grid for a warp-per-row walk: every row group resident at once
// where the die holds it, occupancy asked of the kernel itself (its register
// count decides it on GB10, not threads or shared memory)
template <typename K>
static inline uint32_t pd_trn_grid(K kern, uint32_t rows, uint32_t smem) {
    int dev = 0, nsm = 0, occ = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, PD_TRN_NT, smem) != cudaSuccess ||
        occ <= 0)
        occ = 1;
    const uint32_t need = (rows + PD_TRN_NT / 32u - 1u) / (PD_TRN_NT / 32u);
    const uint32_t res = (uint32_t)(nsm > 0 ? nsm : 1) * (uint32_t)occ;
    return need < res ? need : res;
}

// slot 628: batch-1 PTQ1_0 GEMV off per-128 int8 activations (slot 627's).
// Bit 20 of `dtype` asks for the column-major staged comparand for this one
// call - a probe instrument; serving never sets it. Both layouts are the class.
PD_EXPORT
int pd_ternary_gemv_b128(const void* data, const void* scales, const void* xq, const void* xs,
                         void* y, uint32_t in_dim, uint32_t out_dim, uint32_t dtype,
                         void* stream) {
    if (out_dim == 0u) return 0;
    const bool colmajor = (dtype & (1u << 20u)) != 0u;
    dtype &= 0xFFFFu;
    if (dtype != PD_KQ_PTQ1_ID || in_dim == 0u || (in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    const uint32_t smem = pd_trn_smem_bytes(in_dim);
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
#define PD_TRN_GO(PSB)                                                                      \
    pd_pdl_go(pd_trn_gemv_b128_kernel<PSB>, pd_trn_grid(pd_trn_gemv_b128_kernel<PSB>, out_dim, \
        smem), PD_TRN_NT, smem, st, (const uint8_t*)data, (const uint8_t*)scales,           \
        (const int8_t*)xq, (const float*)xs, (float*)y, in_dim, out_dim)
    if (colmajor) PD_TRN_GO(false);
    else PD_TRN_GO(true);
#undef PD_TRN_GO
    return pd_launch_status();
}

// slot 630: up to three PTQ1_0 planes off one staged input (n_planes 1..3,
// outputs y0 / y1 / y2), or with `glu` a gate | up pair folded with SwiGLU
// into y0. All planes share in_dim.
PD_EXPORT
int pd_ternary_gemv_b128_multi(const void* d0, const void* r0, const void* d1, const void* r1,
                               const void* d2, const void* r2, const void* xq, const void* xs,
                               void* y0, void* y1, void* y2, uint32_t in_dim, uint32_t o0,
                               uint32_t o1, uint32_t o2, uint32_t n_planes, uint32_t glu,
                               void* stream) {
    if (n_planes == 0u || n_planes > 3u || in_dim == 0u || (in_dim & 255u) != 0u)
        return cudaErrorInvalidValue;
    if (n_planes < 3u) o2 = 0u;
    if (n_planes < 2u) o1 = 0u;
    if (o0 == 0u) return cudaErrorInvalidValue;
    if (glu != 0u && (n_planes != 2u || o1 != o0)) return cudaErrorInvalidValue;
    const uint32_t smem = pd_trn_smem_bytes(in_dim);
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    const uint32_t rows = glu != 0u ? o0 : o0 + o1 + o2;
    auto st = (cudaStream_t)stream;
#define PD_TRN_MGO(G)                                                                       \
    pd_pdl_go(pd_trn_multi_b128_kernel<G>, pd_trn_grid(pd_trn_multi_b128_kernel<G>, rows,    \
        smem), PD_TRN_NT, smem, st, (const uint8_t*)d0, (const uint8_t*)r0,                 \
        (const uint8_t*)d1, (const uint8_t*)r1, (const uint8_t*)d2, (const uint8_t*)r2,     \
        (const int8_t*)xq, (const float*)xs, (float*)y0, (float*)y1, (float*)y2, in_dim,    \
        o0, o1, o2)
    if (glu != 0u) PD_TRN_MGO(true);
    else PD_TRN_MGO(false);
#undef PD_TRN_MGO
    return pd_launch_status();
}

// ---- NB columns on the int8 tensor cores (slot 686) --------------------------
// The small-batch lane: 1..N activation rows ("columns" here - decode rows
// sharing a tick, a speculative verify round, a short prefill) against one
// weight read, per column the class above bit for bit.
//
// The exact block dot is order-free, so the K order inside a 128-block is
// ours: a table word (a byte's four trits, LO[byte]) is an A k-word of
// mma.m16n8k32.s8, and the staged operand the batch-1 walk dots it with is
// the matching B k-word. Thread (g, tq) of a warp supplies k-words tq + 4i
// (i = 0..7) of every block, for rows g and g + 8 and for column g:
//   i 0..3  head byte 4tq + i            | its staged operand
//   i 4     fifth trits of head word tq  | the fifth-trit operand of the word
//   i 5, 6  tail bytes 2tq, 2tq + 1      | their operands
//   i 7     tq even: fifth trits of tail word tq/2; odd: qh byte tq/2
// and the four k32 steps of a block take i = 2m (a0/a1) and 2m + 1 (a2/a3).
// The block's s32 is then exactly the batch-1 walk's sum, and the fold is the
// class's: warp w of the CTA's 8 owns super-blocks s == w (mod 8) of each
// slice - leaves w, w+8, w+16, w+24 - so the tree's levels 16 and 8 run in the
// thread and 4, 2, 1 across the warps through shared memory.
//
// A CTA owns 16 weight rows at a time (persistent over row chunks) and up to
// 8 columns (grid.y tiles wider batches). Its activations are staged ONCE per
// CTA when the plane is one slice; a wider plane restages per slice.
// GB10, bench/ternary_gb10_bench.cu, 5120-wide planes DRAM-cold: 1 column
// 97 us / 8 columns 102 us on ffn gate (200 / 191 GB/s), 16 columns 106 -
// flat, where a dp4a walk over staged operands ran 179 us at 8 (shared-memory
// operand bound) and the generic lanes re-read the plane per row.
__device__ __forceinline__ void pd_trn_mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                              uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// the 8 k-words thread tq supplies for one row's block
__device__ __forceinline__ void pd_trn_kwords(const uint32_t* __restrict__ lut, uint32_t head,
                                              uint32_t tail, uint32_t qh, uint32_t tq,
                                              uint32_t (&kw)[8]) {
    kw[0] = lut[head & 0xFFu];
    kw[1] = lut[(head >> 8u) & 0xFFu];
    kw[2] = lut[(head >> 16u) & 0xFFu];
    kw[3] = lut[head >> 24u];
    kw[4] = (uint32_t)pd_trit_digit4(head, 81u);
    const uint32_t tb = (tq & 1u) * 16u;
    kw[5] = lut[(tail >> tb) & 0xFFu];
    kw[6] = lut[(tail >> (tb + 8u)) & 0xFFu];
    kw[7] = (tq & 1u) ? lut[(qh >> ((tq >> 1u) * 8u)) & 0xFFu] : (uint32_t)pd_trit_digit4(tail, 81u);
}

// one 128-activation block of a column in the B layout: [tq][8 words], the
// operands computed exactly as pd_trn_stage_block computes them
__device__ __forceinline__ void pd_trn_nb_stage_block(int* __restrict__ dst,
                                                      const int8_t* __restrict__ xq) {
    const uint8_t* x = reinterpret_cast<const uint8_t*>(xq);
    uint32_t c[8][4];
    #pragma unroll
    for (uint32_t n = 0; n < 8u; ++n) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + 16u * n));
        c[n][0] = v.x; c[n][1] = v.y; c[n][2] = v.z; c[n][3] = v.w;
    }
    int op[26], f5[6];
    #pragma unroll
    for (uint32_t g = 0; g < 4u; ++g) {
        pd_trn_tr4(c[0][g], c[1][g], c[2][g], c[3][g], op + 4u * g, 1u);
        f5[g] = (int)c[4][g];
    }
    #pragma unroll
    for (uint32_t g = 0; g < 2u; ++g) {
        pd_trn_tr4(c[5][g], c[5][2u + g], c[6][g], c[6][2u + g], op + 16u + 4u * g, 1u);
        f5[4u + g] = (int)c[7][g];
    }
    op[24] = (int)__byte_perm(c[7][2], c[7][3], 0x6420u);
    op[25] = (int)__byte_perm(c[7][2], c[7][3], 0x7531u);
    #pragma unroll
    for (uint32_t tq = 0; tq < 4u; ++tq) {
        *reinterpret_cast<int4*>(dst + tq * 8u) =
            make_int4(op[4u * tq], op[4u * tq + 1u], op[4u * tq + 2u], op[4u * tq + 3u]);
        *reinterpret_cast<int4*>(dst + tq * 8u + 4u) =
            make_int4(f5[tq], op[16u + 2u * tq], op[17u + 2u * tq],
                      (tq & 1u) ? op[24u + tq / 2u] : f5[4u + tq / 2u]);
    }
}

// one row's A inputs for a super-block: the head words of both blocks, the
// tail words, the record
struct PdTrnNbIn { uint32_t h0, h1, t0, t1; uint2 rc; };
__device__ __forceinline__ PdTrnNbIn pd_trn_nb_load(const uint8_t* __restrict__ data,
                                                     const uint8_t* __restrict__ rec, size_t rdb,
                                                     size_t rsb, uint32_t row, bool ok, uint32_t s,
                                                     uint32_t tq) {
    PdTrnNbIn v{0u, 0u, 0u, 0u, make_uint2(0u, 0u)};
    if (ok) {
        const uint8_t* rb = data + (size_t)row * rdb + (size_t)s * 48u;
        v.h0 = __ldg(reinterpret_cast<const uint32_t*>(rb + 4u * tq));
        v.h1 = __ldg(reinterpret_cast<const uint32_t*>(rb + 16u + 4u * tq));
        v.t0 = __ldg(reinterpret_cast<const uint32_t*>(rb + 32u + 4u * (tq >> 1u)));
        v.t1 = __ldg(reinterpret_cast<const uint32_t*>(rb + 40u + 4u * (tq >> 1u)));
        v.rc = __ldg(reinterpret_cast<const uint2*>(rec + (size_t)row * rsb + (size_t)s * 8u));
    }
    return v;
}

#define PD_TRN_NB_SB 32u   // super-blocks a slice (the class's)
#define PD_TRN_NB_ROWS 16u // weight rows a CTA chunk

__host__ __device__ __forceinline__ uint32_t pd_trn_nb_smem(uint32_t in_dim, uint32_t ncmax) {
    const uint32_t ns = in_dim >> 8u, sbs = ns < PD_TRN_NB_SB ? ns : PD_TRN_NB_SB;
    return 1024u + ncmax * (sbs * 64u + 4u) * 4u + ncmax * 2u * sbs * 4u +
           8u * PD_TRN_NB_ROWS * 8u * 4u;
}

// Planes laid end to end (row counts multiples of 16, so a chunk never
// straddles two), or GLU: gate | up walked in two passes over the same chunk
// and folded with SwiGLU. y_p is [cols][o_p].
template <bool GLU>
__global__ void __launch_bounds__(256) pd_trn_nb_kernel(
        const uint8_t* __restrict__ d0, const uint8_t* __restrict__ r0,
        const uint8_t* __restrict__ d1, const uint8_t* __restrict__ r1,
        const uint8_t* __restrict__ d2, const uint8_t* __restrict__ r2,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        uint32_t in_dim, uint32_t o0, uint32_t o1, uint32_t o2, uint32_t cols, uint32_t ncmax) {
    constexpr uint32_t NT = 256u, SB = PD_TRN_NB_SB, ROWS = PD_TRN_NB_ROWS;
    extern __shared__ uint8_t pd_trn_smem[];
    const uint32_t ns = in_dim >> 8u, nsl = (ns + SB - 1u) / SB;
    const size_t rdb = (size_t)ns * 48u, rsb = (size_t)ns * 8u;
    const uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5u, g = lane >> 2u, tq = lane & 3u;
    const uint32_t c0 = blockIdx.y * 8u, nc = min(8u, cols - c0);
    const uint32_t sbs = ns < SB ? ns : SB, cstride = sbs * 64u + 4u;  // +4: bank pad
    uint32_t* lut = reinterpret_cast<uint32_t*>(pd_trn_smem);
    int* bst = reinterpret_cast<int*>(pd_trn_smem + 1024u);
    float* xst = reinterpret_cast<float*>(bst + ncmax * cstride);
    float* fold = xst + ncmax * 2u * sbs;  // [8 warps][16 rows][8 cols]
    const uint32_t total = GLU ? o0 : o0 + o1 + o2, nchunks = total / ROWS;
    auto plane = [&](uint32_t o, uint32_t pass, const uint8_t** dd, const uint8_t** rr,
                     float** yy, uint32_t* od) {
        const uint32_t pl = GLU ? pass : (o < o0 ? 0u : (o < o0 + o1 ? 1u : 2u));
        *dd = pl == 0u ? d0 : (pl == 1u ? d1 : d2);
        *rr = pl == 0u ? r0 : (pl == 1u ? r1 : r2);
        *yy = GLU ? y0 : (pl == 0u ? y0 : (pl == 1u ? y1 : y2));
        *od = GLU ? o0 : (pl == 0u ? o0 : (pl == 1u ? o1 : o2));
        return GLU ? o : (pl == 0u ? o : (pl == 1u ? o - o0 : o - o0 - o1));
    };
    auto pf_chunk = [&](uint32_t ch) {
        for (uint32_t pass = 0; pass < (GLU ? 2u : 1u); ++pass) {
            const uint8_t* dd; const uint8_t* rr; float* yy; uint32_t od;
            const uint32_t po = plane(ch * ROWS, pass, &dd, &rr, &yy, &od);
            pd_l2_prefetch_bulk(dd + (size_t)po * rdb, (uint32_t)(ROWS * rdb));
            pd_l2_prefetch_bulk(rr + (size_t)po * rsb, (uint32_t)(ROWS * rsb));
        }
    };
    if (threadIdx.x == 0u && blockIdx.x < nchunks) pf_chunk(blockIdx.x);
    PD_PDL_ARM();
    lut[threadIdx.x] = pd_trn_lut_entry(threadIdx.x);  // NT = 256: one entry a thread
    bool staged = false;
    const bool gcol = g < nc;
    // fold threads: thread t < 128 owns (row t >> 3, column t & 7) of the chunk
    const uint32_t frow = threadIdx.x >> 3u, fcol = threadIdx.x & 7u;
    for (uint32_t chunk = blockIdx.x; chunk < nchunks; chunk += gridDim.x) {
        if (threadIdx.x == 0u && chunk + gridDim.x < nchunks) pf_chunk(chunk + gridDim.x);
        float ypass[2] = {0.0f, 0.0f};
        #pragma unroll
        for (uint32_t pass = 0; pass < (GLU ? 2u : 1u); ++pass) {
            const uint8_t* dd; const uint8_t* rr; float* yy; uint32_t od;
            const uint32_t pr = plane(chunk * ROWS, pass, &dd, &rr, &yy, &od);
            const uint32_t ra = pr + g, rb = ra + 8u;
            float ysl = 0.0f;
            for (uint32_t k = 0; k < nsl; ++k) {
                const uint32_t sb0 = k * SB, nsb = min(SB, ns - sb0);
                if (nsl > 1u || !staged) {
                    __syncthreads();
                    for (uint32_t t = threadIdx.x; t < nc * 2u * nsb; t += NT) {
                        const uint32_t c = t / (2u * nsb), bb = t % (2u * nsb);
                        pd_trn_nb_stage_block(bst + c * cstride + bb * 32u,
                            xq + (size_t)(c0 + c) * in_dim + (size_t)(2u * sb0 + bb) * 128u);
                        xst[c * 2u * sbs + bb] = xs[(size_t)(c0 + c) * (in_dim >> 7u) + 2u * sb0 + bb];
                    }
                    __syncthreads();
                    staged = true;
                }
                float leaf[4][4];
                #pragma unroll
                for (uint32_t p = 0; p < 4u; ++p)
                    #pragma unroll
                    for (uint32_t l = 0; l < 4u; ++l) leaf[p][l] = 0.0f;
                uint32_t sl = w;
                const bool any = sl < nsb;
                PdTrnNbIn A0 = pd_trn_nb_load(dd, rr, rdb, rsb, ra, any, sb0 + sl, tq);
                PdTrnNbIn A1 = pd_trn_nb_load(dd, rr, rdb, rsb, rb, any, sb0 + sl, tq);
                for (; sl < nsb; sl += 8u) {
                    const uint32_t li = sl >> 3u;  // leaf w + 8 li
                    // the next super-block's inputs go out before this one decodes
                    const bool more = sl + 8u < nsb;
                    const PdTrnNbIn N0 = pd_trn_nb_load(dd, rr, rdb, rsb, ra, more, sb0 + sl + 8u, tq);
                    const PdTrnNbIn N1 = pd_trn_nb_load(dd, rr, rdb, rsb, rb, more, sb0 + sl + 8u, tq);
                    #pragma unroll
                    for (uint32_t bl = 0; bl < 2u; ++bl) {
                        uint32_t bw[8];
                        if (gcol) {
                            const int* bsrc = bst + g * cstride + (2u * sl + bl) * 32u + tq * 8u;
                            const int4 blo = *reinterpret_cast<const int4*>(bsrc);
                            const int4 bhi = *reinterpret_cast<const int4*>(bsrc + 4);
                            bw[0] = blo.x; bw[1] = blo.y; bw[2] = blo.z; bw[3] = blo.w;
                            bw[4] = bhi.x; bw[5] = bhi.y; bw[6] = bhi.z; bw[7] = bhi.w;
                        } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < 8u; ++i) bw[i] = 0u;
                        }
                        const uint32_t ca = 2u * tq, cb = ca + 1u;
                        const float xa = ca < nc ? xst[ca * 2u * sbs + 2u * sl + bl] : 0.0f;
                        const float xb = cb < nc ? xst[cb * 2u * sbs + 2u * sl + bl] : 0.0f;
                        uint32_t kw0[8], kw1[8];
                        pd_trn_kwords(lut, bl ? A0.h1 : A0.h0, bl ? A0.t1 : A0.t0,
                                      bl ? (A0.rc.y >> 16u) : (A0.rc.y & 0xFFFFu), tq, kw0);
                        pd_trn_kwords(lut, bl ? A1.h1 : A1.h0, bl ? A1.t1 : A1.t0,
                                      bl ? (A1.rc.y >> 16u) : (A1.rc.y & 0xFFFFu), tq, kw1);
                        int acc[4] = {0, 0, 0, 0};
                        #pragma unroll
                        for (uint32_t st = 0; st < 4u; ++st)
                            pd_trn_mma_s8(acc, kw0[2u * st], kw1[2u * st], kw0[2u * st + 1u],
                                          kw1[2u * st + 1u], bw[2u * st], bw[2u * st + 1u]);
                        // C: acc 0, 1 = row g, columns 2tq, 2tq+1; acc 2, 3 = row g + 8
                        const uint32_t da = bl ? (A0.rc.x >> 16u) : (A0.rc.x & 0xFFFFu);
                        const uint32_t db = bl ? (A1.rc.x >> 16u) : (A1.rc.x & 0xFFFFu);
                        const uint32_t dd4[4] = {da, da, db, db};
                        const float xx4[4] = {xa, xb, xa, xb};
                        #pragma unroll
                        for (uint32_t p = 0; p < 4u; ++p)
                            #pragma unroll
                            for (uint32_t l = 0; l < 4u; ++l)
                                if (l == li) leaf[p][l] = pd_trn_term(dd4[p], xx4[p], acc[p], leaf[p][l]);
                    }
                    A0 = N0; A1 = N1;
                }
                // the tree: levels 16 and 8 in the thread (leaves w + 8l), then
                // 4, 2, 1 across the warps
                __syncthreads();
                #pragma unroll
                for (uint32_t p = 0; p < 4u; ++p) {
                    const float v = __fadd_rn(__fadd_rn(leaf[p][0], leaf[p][2]),
                                              __fadd_rn(leaf[p][1], leaf[p][3]));
                    fold[(w * ROWS + g + (p >> 1u) * 8u) * 8u + 2u * tq + (p & 1u)] = v;
                }
                __syncthreads();
                if (threadIdx.x < ROWS * 8u) {
                    float v[8];
                    #pragma unroll
                    for (uint32_t q = 0; q < 8u; ++q) v[q] = fold[(q * ROWS + frow) * 8u + fcol];
                    const float f = __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[4]), __fadd_rn(v[2], v[6])),
                                              __fadd_rn(__fadd_rn(v[1], v[5]), __fadd_rn(v[3], v[7])));
                    ysl = k == 0u ? f : __fadd_rn(ysl, f);
                }
            }
            ypass[pass] = ysl;
            if (!GLU && threadIdx.x < ROWS * 8u && fcol < nc)
                yy[(size_t)(c0 + fcol) * od + pr + frow] = ysl;
        }
        if (GLU && threadIdx.x < ROWS * 8u && fcol < nc)
            y0[(size_t)(c0 + fcol) * o0 + chunk * ROWS + frow] = pd_trn_swiglu(ypass[0], ypass[1]);
    }
}

// slot 686: NB activation columns (xq int8 [cols][in_dim], xs f32 [cols][in_dim / 128],
// slot 627's per-128 class) against up to three PTQ1_0 planes laid end to end
// (y_p [cols][o_p]), or with `glu` a gate | up pair folded with SwiGLU into
// y0. Every o_p a multiple of 16. Per column bit-identical to slot 628 / 630.
PD_EXPORT
int pd_ternary_gemm_nb(const void* d0, const void* r0, const void* d1, const void* r1,
                       const void* d2, const void* r2, const void* xq, const void* xs, void* y0,
                       void* y1, void* y2, uint32_t in_dim, uint32_t o0, uint32_t o1, uint32_t o2,
                       uint32_t n_planes, uint32_t glu, uint32_t cols, void* stream) {
    if (n_planes == 0u || n_planes > 3u || in_dim == 0u || (in_dim & 255u) != 0u || cols == 0u)
        return cudaErrorInvalidValue;
    if (n_planes < 3u) o2 = 0u;
    if (n_planes < 2u) o1 = 0u;
    if (o0 == 0u || ((o0 | o1 | o2) & (PD_TRN_NB_ROWS - 1u)) != 0u) return cudaErrorInvalidValue;
    if (glu != 0u && (n_planes != 2u || o1 != o0)) return cudaErrorInvalidValue;
    const uint32_t ncmax = cols < 8u ? cols : 8u;
    const uint32_t smem = pd_trn_nb_smem(in_dim, ncmax);
    if (smem > 99u * 1024u) return cudaErrorInvalidValue;
    const uint32_t ty = (cols + 7u) / 8u;
    const uint32_t nchunks = (glu != 0u ? o0 : o0 + o1 + o2) / PD_TRN_NB_ROWS;
    auto st = (cudaStream_t)stream;
#define PD_TRN_NBGO(G)                                                                      \
    do {                                                                                    \
        auto kern = pd_trn_nb_kernel<G>;                                                    \
        static bool attr = false;                                                           \
        if (!attr) {                                                                        \
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                                 99 * 1024);                                                \
            attr = true;                                                                    \
        }                                                                                   \
        int dev = 0, nsm = 0, occ = 0;                                                      \
        cudaGetDevice(&dev);                                                                \
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);                  \
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, 256, smem) !=         \
                cudaSuccess || occ <= 0)                                                    \
            occ = 1;                                                                        \
        uint32_t bx = (uint32_t)(nsm > 0 ? nsm : 1) * (uint32_t)occ / ty;                   \
        if (bx == 0u) bx = 1u;                                                              \
        if (bx > nchunks) bx = nchunks;                                                     \
        pd_pdl_go(kern, dim3(bx, ty), 256, smem, st, (const uint8_t*)d0, (const uint8_t*)r0, \
            (const uint8_t*)d1, (const uint8_t*)r1, (const uint8_t*)d2, (const uint8_t*)r2, \
            (const int8_t*)xq, (const float*)xs, (float*)y0, (float*)y1, (float*)y2, in_dim, \
            o0, o1, o2, cols, ncmax);                                                       \
    } while (0)
    if (glu != 0u) PD_TRN_NBGO(true);
    else PD_TRN_NBGO(false);
#undef PD_TRN_NBGO
    return pd_launch_status();
}

// One int8 scale per 128 activations: a 128-thread block per quant block, the
// absmax reduced inside each warp by xor-shuffles and across the four warps
// through shared memory. Same rounding and clamp as pd_quantize_q8.
__global__ void __launch_bounds__(128) pd_quantize_q8_b128_kernel(
        const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ scale,
        uint32_t n_blocks) {
    PD_PDL_ARM();
    __shared__ float wmax[4];
    const uint32_t tid = threadIdx.x, warp = tid >> 5u;
    // grid-stride, every thread running the same trips so the syncs line up
    for (uint32_t b = blockIdx.x; b < n_blocks; b += gridDim.x) {
        const float v = x[(size_t)b * 128u + tid];
        float a = fabsf(v);
        for (uint32_t s = 16; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
        if ((tid & 31u) == 0u) wmax[warp] = a;
        __syncthreads();
        const float m = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
        __syncthreads();
        const float scl = m * (1.0f / 127.0f);
        if (tid == 0u) scale[b] = scl;
        const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int qi = __float2int_rn(v * inv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        q[(size_t)b * 128u + tid] = (signed char)qi;
    }
}

// slot 627: x f32 [n] -> q int8 [n], scale f32 [n / 128]; n a multiple of 128.
PD_EXPORT
int pd_quantize_q8_b128(const void* x, void* q, void* scale, uint32_t n, void* stream) {
    if (n == 0u) return 0;
    if ((n & 127u) != 0u) return cudaErrorInvalidValue;
    const uint32_t n_blocks = n >> 7u;
    const uint32_t grid = n_blocks < 65535u ? n_blocks : 65535u;
    pd_pdl_go(pd_quantize_q8_b128_kernel, grid, 128, 0u, (cudaStream_t)stream,
        (const float*)x, (signed char*)q, (float*)scale, n_blocks);
    return pd_launch_status();
}
