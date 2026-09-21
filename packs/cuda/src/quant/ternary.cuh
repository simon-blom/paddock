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
// Shape: NT = 256, a warp carries 8 rows over 4 stripes - lane l owns row
// l & 7 and the super-blocks s == (l >> 3) mod 4, so every lane keeps ONE
// scalar accumulator and the row sums fall out of two down-shuffles (16,
// then 8). The three input widths of the 27B (20 / 24 / 68 super-blocks) all
// divide by 4, so the lanes are balanced; any other width still computes,
// with a ragged last step. Fewer rows a warp (to raise the block count of a
// plane with few rows) was measured and is DEAD - every resident block pays
// its own staging.
// Study reference for the byte-owning idea: PrismML's Metal kernel
// (kernel_mul_mv_ptq1_0, MIT). Nothing here is theirs.

#define PD_TRN_NT 256u
#define PD_TRN_R 8u
#define PD_TRN_STRIPES 4u
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

// One lane's share of a row: super-blocks q, q + 4, .. of it, over the
// repacked streams (data: b0.qs[0..16], b1.qs[0..16], b0.qs[16..24],
// b1.qs[16..24]; record: d0, d1, qh0[2], qh1[2]).
template <bool PSB>
__device__ __forceinline__ float pd_trn_lane_acc(
        const uint8_t* __restrict__ row, const uint8_t* __restrict__ rec,
        const uint32_t* __restrict__ lut, const int* __restrict__ sxa,
        const int* __restrict__ sxb, const float* __restrict__ sxs,
        uint32_t ns, uint32_t q) {
    float acc = 0.0f;
    for (uint32_t s = q; s < ns; s += PD_TRN_STRIPES) {
        const uint8_t* sb = row + (size_t)s * 48u;
        const uint8_t* rc = rec + (size_t)s * 8u;
        const uint4 h0 = pd_iq_ld16(sb), h1 = pd_iq_ld16(sb + 16u), tl = pd_iq_ld16(sb + 32u);
        const uint32_t hw0[4] = {h0.x, h0.y, h0.z, h0.w};
        const uint32_t hw1[4] = {h1.x, h1.y, h1.z, h1.w};
        // record words: {d0, d1} then {qh0[2], qh1[2]}
        const uint32_t dw = reinterpret_cast<const uint32_t*>(rc)[0];
        const uint32_t qhw = reinterpret_cast<const uint32_t*>(rc)[1];
        const int s0 = pd_trn_block<PSB>(lut, hw0, tl.x, tl.y, qhw & 0xFFFFu, sxa, sxb, ns, s, 0u);
        const int s1 = pd_trn_block<PSB>(lut, hw1, tl.z, tl.w, qhw >> 16u, sxa, sxb, ns, s, 1u);
        acc += (pd_trn_f16(dw & 0xFFFFu) * sxs[2u * s]) * (float)s0;
        acc += (pd_trn_f16(dw >> 16u) * sxs[2u * s + 1u]) * (float)s1;
    }
    return acc;
}

// y[o] = row_o . x; xq int8 [in_dim], xs f32 [in_dim / 128].
template <bool PSB>
__global__ void __launch_bounds__(PD_TRN_NT) pd_trn_gemv_b128_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim) {
    PD_PDL_ARM();
    constexpr uint32_t NT = PD_TRN_NT, WARPS = NT / 32u, R = PD_TRN_R;
    extern __shared__ uint8_t pd_trn_smem[];
    const uint32_t ns = in_dim >> 8u;
    uint32_t* lut; int* sxa; int* sxb; float* sxs;
    pd_trn_stage<PSB>(pd_trn_smem, xq, xs, ns, NT, &lut, &sxa, &sxb, &sxs);
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u, lane = tid & 31u, r = lane & 7u, q = lane >> 3u;
    const size_t rdb = (size_t)ns * 48u, rsb = (size_t)ns * 8u;
    const uint32_t groups = (out_dim + R - 1u) / R;
    for (uint32_t grp = blockIdx.x * WARPS + warp; grp < groups; grp += gridDim.x * WARPS) {
        const uint32_t o = grp * R + r;
        float acc = 0.0f;
        if (o < out_dim)
            acc = pd_trn_lane_acc<PSB>(data + (size_t)o * rdb, scales + (size_t)o * rsb, lut, sxa,
                                       sxb, sxs, ns, q);
        // lanes r, 8 + r, 16 + r, 24 + r hold row r's four stripes
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        if (lane < R && o < out_dim) y[o] = acc;
    }
}

// Several planes that read the SAME staged row, in one launch: the rows of up
// to three planes laid end to end (q | k | v; in_qkv | gate), or - GLU - a
// gate | up pair walked together with y[o] = silu(gate_o . x) * (up_o . x),
// which also takes the SwiGLU launch with it. One staging instead of one per
// plane, one ramp/drain toll instead of three (wk / wv are 1024 rows: their
// own launches are nearly all toll). Row counts are multiples of 8, so a warp
// never straddles two planes. Per row this is the single-plane walk, in its
// order - bit-identical to it.
template <bool GLU>
__global__ void __launch_bounds__(PD_TRN_NT) pd_trn_multi_b128_kernel(
        const uint8_t* __restrict__ d0, const uint8_t* __restrict__ r0,
        const uint8_t* __restrict__ d1, const uint8_t* __restrict__ r1,
        const uint8_t* __restrict__ d2, const uint8_t* __restrict__ r2,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        uint32_t in_dim, uint32_t o0, uint32_t o1, uint32_t o2) {
    PD_PDL_ARM();
    constexpr uint32_t NT = PD_TRN_NT, WARPS = NT / 32u, R = PD_TRN_R;
    extern __shared__ uint8_t pd_trn_smem[];
    const uint32_t ns = in_dim >> 8u;
    uint32_t* lut; int* sxa; int* sxb; float* sxs;
    pd_trn_stage<true>(pd_trn_smem, xq, xs, ns, NT, &lut, &sxa, &sxb, &sxs);
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u, lane = tid & 31u, r = lane & 7u, q = lane >> 3u;
    const size_t rdb = (size_t)ns * 48u, rsb = (size_t)ns * 8u;
    const uint32_t total = GLU ? o0 : o0 + o1 + o2;
    const uint32_t groups = total / R;
    for (uint32_t grp = blockIdx.x * WARPS + warp; grp < groups; grp += gridDim.x * WARPS) {
        const uint32_t o = grp * R + r;
        if (GLU) {
            float ag = pd_trn_lane_acc<true>(d0 + (size_t)o * rdb, r0 + (size_t)o * rsb, lut, sxa,
                                             sxb, sxs, ns, q);
            float au = pd_trn_lane_acc<true>(d1 + (size_t)o * rdb, r1 + (size_t)o * rsb, lut, sxa,
                                             sxb, sxs, ns, q);
            ag += __shfl_down_sync(0xffffffffu, ag, 16);
            au += __shfl_down_sync(0xffffffffu, au, 16);
            ag += __shfl_down_sync(0xffffffffu, ag, 8);
            au += __shfl_down_sync(0xffffffffu, au, 8);
            if (lane < R) y0[o] = (ag / (1.0f + expf(-ag))) * au;
        } else {
            // the whole warp sits in one plane (row counts are multiples of 8)
            const uint32_t pl = o < o0 ? 0u : (o < o0 + o1 ? 1u : 2u);
            const uint32_t po = pl == 0u ? o : (pl == 1u ? o - o0 : o - o0 - o1);
            const uint8_t* dd = pl == 0u ? d0 : (pl == 1u ? d1 : d2);
            const uint8_t* rr = pl == 0u ? r0 : (pl == 1u ? r1 : r2);
            float acc = pd_trn_lane_acc<true>(dd + (size_t)po * rdb, rr + (size_t)po * rsb, lut,
                                              sxa, sxb, sxs, ns, q);
            acc += __shfl_down_sync(0xffffffffu, acc, 16);
            acc += __shfl_down_sync(0xffffffffu, acc, 8);
            if (lane < R) (pl == 0u ? y0 : (pl == 1u ? y1 : y2))[po] = acc;
        }
    }
}

// slot 628: batch-1 PTQ1_0 GEMV off per-128 int8 activations (slot 627's).
// Bit 20 of `dtype` asks for the column-major comparand for this one call - a
// probe instrument, so a bench can interleave the two inside one process
// (numbers from separate runs on a card that also drives a desktop differ by
// more than the variants do). Serving never sets it.
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
    const uint32_t groups = (out_dim + PD_TRN_R - 1u) / PD_TRN_R;
    const uint32_t blocks = pd_iqd_grid(groups, PD_TRN_NT / 32u, smem, PD_TRN_NT);
    auto st = (cudaStream_t)stream;
#define PD_TRN_GO(PSB)                                                                      \
    pd_pdl_go(pd_trn_gemv_b128_kernel<PSB>, blocks, PD_TRN_NT, smem, st,                     \
        (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq, (const float*)xs,  \
        (float*)y, in_dim, out_dim)
    if (colmajor) PD_TRN_GO(false);
    else PD_TRN_GO(true);
#undef PD_TRN_GO
    return pd_launch_status();
}

// slot 630: up to three PTQ1_0 planes off one staged input (n_planes 1..3,
// outputs y0 / y1 / y2), or with `glu` a gate | up pair folded with SwiGLU
// into y0. All planes share in_dim; every out dim a multiple of 8.
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
    if (o0 == 0u || ((o0 | o1 | o2) & 7u) != 0u) return cudaErrorInvalidValue;
    if (glu != 0u && (n_planes != 2u || o1 != o0)) return cudaErrorInvalidValue;
    const uint32_t smem = pd_trn_smem_bytes(in_dim);
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    const uint32_t groups = (glu != 0u ? o0 : o0 + o1 + o2) / PD_TRN_R;
    const uint32_t blocks = pd_iqd_grid(groups, PD_TRN_NT / 32u, smem, PD_TRN_NT);
    auto st = (cudaStream_t)stream;
#define PD_TRN_MGO(G)                                                                       \
    pd_pdl_go(pd_trn_multi_b128_kernel<G>, blocks, PD_TRN_NT, smem, st,                      \
        (const uint8_t*)d0, (const uint8_t*)r0, (const uint8_t*)d1, (const uint8_t*)r1,     \
        (const uint8_t*)d2, (const uint8_t*)r2, (const int8_t*)xq, (const float*)xs,        \
        (float*)y0, (float*)y1, (float*)y2, in_dim, o0, o1, o2)
    if (glu != 0u) PD_TRN_MGO(true);
    else PD_TRN_MGO(false);
#undef PD_TRN_MGO
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
