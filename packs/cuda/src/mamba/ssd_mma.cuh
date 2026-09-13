// The chunked SSD's three GEMM pieces on the tf32 tensor pipe (GB10,
// 2026-09-11): K2 (the per-group Gram
// M = C . B^T), K3 (the per-head chunk state dS = sum_t w_t B_t x_t^T) and
// K5 (the per-head output pair: the masked intra W . x and the inter
// C . S_in). The scalar twins in ssd.cuh are 4x8 / 2x8 register micro-tiles
// fed from shared with 10 scalar LDS per 16 FMA - LDS-bound at 2.5-3.3 TF/s
// on this die (y 356 us, dS 163 us per 512-token pass, 21 ms of a 1k
// request against vLLM's 6.8 for its bf16 tl.dot chain). These run the
// same contractions as 3xTF32 mma.sync m16n8k8 (cvt.rna big + small split
// of both operands, three mma per k8 in the shipped order big.big,
// big.small, small.big, the chain drained into an RN f32 accumulator every
// 32 k - the attention projections' class, gemm/f32_qkv.cuh) - STRICTLY
// finer than vLLM's bf16 operands, and inside the SSD gates (the f64 host
// reference at 1e-4, the f32/f16 twin identity, run-to-run determinism).
// K is walked in 32-wide slices staged into STATIC shared (this die has
// 100 KB per SM / 99 KB per block, so whole-K staging of the 128 x 128
// operands does not fit two CTAs) in the layout each operand arrives in:
// [token][k] planes read k-contiguous (row stride 36: bank = 4 row + k),
// [token][m] planes read m-contiguous (stride M + 8: bank = 8 k + m) - no
// transpose scatter, every fragment load conflict-free. One CTA per
// (chunk, head) or (chunk, group, tile), 27-29 KB each, two per SM.
// `PADDOCK_NO_SSD_MMA=1` pins the scalar twins.
#ifndef PD_SSD_MMA_CUH
#define PD_SSD_MMA_CUH

// hi = tf32(v) (cvt.rna), lo = tf32(v - hi): the two-term split ("big" and
// "small" in the notes above). Not named `small`: the Windows SDK's rpcndr.h
// has `#define small char`, which turns `uint32_t& small` into a syntax error
// in the MSVC-hosted build and nowhere else.
static __device__ __forceinline__ void pd_ssd_split(float v, uint32_t& hi,
                                                    uint32_t& lo) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    hi = r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(lo) : "f"(v - __uint_as_float(r)));
}
static __device__ __forceinline__ void pd_ssd_mma(float* acc, const uint32_t* a,
                                                  const uint32_t* b) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// one k8 step of a 32 x 32 warp tile (2 m16 x 4 n8), 3xTF32. A element
// (m, k) sits at a[m * lda + k] (A_MK) or a[k * lda + m]; B element (k, n)
// at b[n * ldb + k] (B_NK) or b[k * ldb + n]. a/b are already offset to the
// warp's m0 / n0 (in the m/n index, whichever layout); k8 is slice-local.
template <bool A_MK, bool B_NK>
static __device__ __forceinline__ void pd_ssd_wstep(
        float acc[2][4][4], const float* __restrict__ a, uint32_t lda,
        const float* __restrict__ b, uint32_t ldb, uint32_t k8, uint32_t gr,
        uint32_t t4) {
    uint32_t ab[2][4], as[2][4];
    #pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt) {
        const uint32_t mr = mt * 16u + gr;
        float a0, a1, a2, a3;
        if (A_MK) {
            const float* p = a + mr * lda + k8 + t4;
            a0 = p[0]; a1 = p[8u * lda]; a2 = p[4u]; a3 = p[8u * lda + 4u];
        } else {
            const float* p = a + (k8 + t4) * lda + mr;
            a0 = p[0]; a1 = p[8u]; a2 = p[4u * lda]; a3 = p[4u * lda + 8u];
        }
        pd_ssd_split(a0, ab[mt][0], as[mt][0]);
        pd_ssd_split(a1, ab[mt][1], as[mt][1]);
        pd_ssd_split(a2, ab[mt][2], as[mt][2]);
        pd_ssd_split(a3, ab[mt][3], as[mt][3]);
    }
    #pragma unroll
    for (uint32_t nt = 0; nt < 4u; ++nt) {
        const uint32_t nr = nt * 8u + gr;
        float b0, b1;
        if (B_NK) {
            const float* p = b + nr * ldb + k8 + t4;
            b0 = p[0]; b1 = p[4u];
        } else {
            const float* p = b + (k8 + t4) * ldb + nr;
            b0 = p[0]; b1 = p[4u * ldb];
        }
        uint32_t bb[2], bs[2];
        pd_ssd_split(b0, bb[0], bs[0]);
        pd_ssd_split(b1, bb[1], bs[1]);
        #pragma unroll
        for (uint32_t mt = 0; mt < 2u; ++mt) {
            pd_ssd_mma(acc[mt][nt], ab[mt], bb);
            pd_ssd_mma(acc[mt][nt], ab[mt], bs);
            pd_ssd_mma(acc[mt][nt], as[mt], bb);
        }
    }
}
static __device__ __forceinline__ void pd_ssd_zero(float acc[2][4][4]) {
    #pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) acc[mt][nt][e] = 0.0f;
}
// drain the mma chain into the RN accumulator (the tiles' two-level
// accumulation: the DPU chains C with truncation, so at most 12 chained
// mma - one 32-wide k slice - before an RN add)
static __device__ __forceinline__ void pd_ssd_drain(float fac[2][4][4],
                                                    float acc[2][4][4]) {
    #pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                fac[mt][nt][e] += acc[mt][nt][e];
                acc[mt][nt][e] = 0.0f;
            }
}

#define PD_SSD_KS 32u                // k slice
#define PD_SSD_LDK (PD_SSD_KS + 4u)  // [row][k] slice planes: stride 36
#define PD_SSD_LDM (PD_SSD_L + 8u)   // [k][m] planes, M = 128: stride 136
#define PD_SSD_LDN (64u + 8u)        // [k][n] planes, N = HD = 64: stride 72

// K2: M[t][s] = C_t . B_s over the S = 128 dot, per (chunk, group), 64 x 64
// output tiles (grid.x = 4), 128 threads = 4 warps as 2 (t) x 2 (s) of
// 32 x 32. C and B rows arrive k-contiguous: [row][k] slices, both.
template <uint32_t S_>
__global__ void __launch_bounds__(128, 2) pd_ssd_gram_mma_kernel(
        const float* __restrict__ xbc, uint32_t t0, uint32_t conv_dim,
        uint32_t d_inner, uint32_t n_groups, uint32_t n_tok,
        float* __restrict__ m) {
    constexpr uint32_t LD = PD_SSD_LDK, KS = PD_SSD_KS;
    __shared__ float sc[64u * LD];   // [64 t][LD]
    __shared__ float sb[64u * LD];   // [64 s][LD]
    const uint32_t c = blockIdx.y / n_groups, g = blockIdx.y % n_groups;
    const uint32_t tile_t = (blockIdx.x >> 1) * 64u, tile_s = (blockIdx.x & 1u) * 64u;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t m0 = (warp >> 1) * 32u, n0 = (warp & 1u) * 32u;
    const uint32_t base = c * PD_SSD_L;
    float acc[2][4][4], fac[2][4][4];
    pd_ssd_zero(acc); pd_ssd_zero(fac);
    for (uint32_t k0 = 0; k0 < S_; k0 += KS) {
        for (uint32_t e = tid; e < 64u * KS; e += 128u) {
            const uint32_t r = e / KS, k = e % KS;
            const uint32_t trow = base + tile_t + r, srow = base + tile_s + r;
            sc[r * LD + k] = (trow < n_tok)
                ? xbc[(size_t)(t0 + trow) * conv_dim + d_inner + (size_t)(n_groups + g) * S_ + k0 + k]
                : 0.0f;
            sb[r * LD + k] = (srow < n_tok)
                ? xbc[(size_t)(t0 + srow) * conv_dim + d_inner + (size_t)g * S_ + k0 + k]
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t k8 = 0; k8 < KS; k8 += 8u)
            pd_ssd_wstep<true, true>(acc, sc + m0 * LD, LD, sb + n0 * LD, LD, k8, gr, t4);
        pd_ssd_drain(fac, acc);
        __syncthreads();
    }
    float* mo = m + ((size_t)c * n_groups + g) * PD_SSD_L * PD_SSD_L;
    #pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; ++nt) {
            const uint32_t t = tile_t + m0 + mt * 16u + gr;
            const uint32_t s = tile_s + n0 + nt * 8u + 2u * t4;
            *(float2*)(mo + (size_t)t * PD_SSD_L + s) = make_float2(fac[mt][nt][0], fac[mt][nt][1]);
            *(float2*)(mo + (size_t)(t + 8u) * PD_SSD_L + s) = make_float2(fac[mt][nt][2], fac[mt][nt][3]);
        }
}

// K3: dS[j][i] = sum_t (w_t B_t[j]) x_t[i], w_t = exp(cum_L - cum_t) dt_t,
// per (chunk, head): M = j (128), N = i (64), K = t (128) in 32-token
// slices. Both operands arrive token-major ([t][j], [t][i]) = [k][m] / [k][n]
// staging, the w_t weight applied on the B store. 256 threads = 4 (j) x
// 2 (i) warps.
template <uint32_t S_, uint32_t HD_>
__global__ void __launch_bounds__(256, 2) pd_ssd_dstate_mma_kernel(
        const float* __restrict__ xbc, uint32_t t0, uint32_t conv_dim,
        uint32_t d_inner, uint32_t n_groups, uint32_t n_heads,
        uint32_t n_tok, const float* __restrict__ cum,
        const float* __restrict__ dtv, float* __restrict__ ds) {
    constexpr uint32_t L = PD_SSD_L, KS = PD_SSD_KS, LDB = PD_SSD_LDM, LDX = PD_SSD_LDN;
    __shared__ float sb[KS * LDB];   // [t][j] weighted B, one slice
    __shared__ float sx[KS * LDX];   // [t][i]
    __shared__ float swt[L];         // w_t for the whole chunk
    const uint32_t h = blockIdx.x, c = blockIdx.y;
    const uint32_t g = h / (n_heads / n_groups);
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t j0 = (warp >> 1) * 32u, i0 = (warp & 1u) * 32u;
    const float* cumr = cum + ((size_t)c * n_heads + h) * L;
    const float* dtr = dtv + ((size_t)c * n_heads + h) * L;
    const float cum_l = cumr[L - 1u];
    const uint32_t base = c * L;
    if (tid < L)
        swt[tid] = (base + tid < n_tok) ? expf(cum_l - cumr[tid]) * dtr[tid] : 0.0f;
    float acc[2][4][4], fac[2][4][4];
    pd_ssd_zero(acc); pd_ssd_zero(fac);
    for (uint32_t k0 = 0; k0 < L; k0 += KS) {
        __syncthreads();   // swt on the first pass; the previous slice's readers after
        for (uint32_t e = tid; e < KS * S_; e += 256u) {
            const uint32_t t = e / S_, j = e % S_, row = base + k0 + t;
            sb[t * LDB + j] = (row < n_tok)
                ? swt[k0 + t] * xbc[(size_t)(t0 + row) * conv_dim + d_inner + (size_t)g * S_ + j]
                : 0.0f;
        }
        for (uint32_t e = tid; e < KS * HD_; e += 256u) {
            const uint32_t t = e / HD_, i = e % HD_, row = base + k0 + t;
            sx[t * LDX + i] = (row < n_tok)
                ? xbc[(size_t)(t0 + row) * conv_dim + (size_t)h * HD_ + i]
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t k8 = 0; k8 < KS; k8 += 8u)
            pd_ssd_wstep<false, false>(acc, sb + j0, LDB, sx + i0, LDX, k8, gr, t4);
        pd_ssd_drain(fac, acc);
    }
    float* dso = ds + ((size_t)c * n_heads + h) * (S_ * HD_);
    #pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; ++nt) {
            const uint32_t j = j0 + mt * 16u + gr;
            const uint32_t i = i0 + nt * 8u + 2u * t4;
            *(float2*)(dso + (size_t)j * HD_ + i) = make_float2(fac[mt][nt][0], fac[mt][nt][1]);
            *(float2*)(dso + (size_t)(j + 8u) * HD_ + i) = make_float2(fac[mt][nt][2], fac[mt][nt][3]);
        }
}

// K5: y[t][i] = sum_{s<=t} W[t][s] x[s][i] + exp(cum_t) sum_j C_t[j]
// S_in[j][i] + D_h x_t[i], per (chunk, head), 256 threads = 4 (t) x 2 (i)
// warps of 32 x 32. Inter first: A = C [t][j] ([m][k] slices), B = S_in
// [j][i] ([k][n] slices); its drained sum is scaled by exp(cum_t) per row
// into the result registers. Intra then: A = W [t][s] (built cooperatively
// per slice, one expf per entry), B = x [s][i]; a warp's 32 rows t in
// [m0, m0 + 32) see s <= t < m0 + 32, so it skips the slices past that (the
// skipped weights are zero); each slice's drain adds into the same result.
// D_h x_t joins at the store. Three f32 terms, summed in a fixed order -
// the SSD value class (the f64 gate), not the scalar twin's bit pattern.
template <uint32_t S_, uint32_t HD_>
__global__ void __launch_bounds__(256, 2) pd_ssd_y_mma_kernel(
        const float* __restrict__ xbc, uint32_t t0, uint32_t conv_dim,
        uint32_t d_inner, uint32_t n_groups, uint32_t n_heads,
        const float* __restrict__ D, float* __restrict__ y, uint32_t n_tok,
        const float* __restrict__ cum, const float* __restrict__ dtv,
        const float* __restrict__ m, const float* __restrict__ ds) {
    constexpr uint32_t L = PD_SSD_L, KS = PD_SSD_KS, LDA = PD_SSD_LDK, LDB = PD_SSD_LDN;
    __shared__ float sa[L * LDA];    // [t][k]: C slice, then W slice
    __shared__ float sbm[KS * LDB];  // [k][i]: S_in slice, then x slice
    __shared__ float scum[L], sdt[L];
    const uint32_t h = blockIdx.x, c = blockIdx.y;
    const uint32_t g = h / (n_heads / n_groups);
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t m0 = (warp >> 1) * 32u, n0 = (warp & 1u) * 32u;
    const uint32_t base = c * L;
    const float* cumr = cum + ((size_t)c * n_heads + h) * L;
    const float* dtr = dtv + ((size_t)c * n_heads + h) * L;
    if (tid < L) { scum[tid] = cumr[tid]; sdt[tid] = dtr[tid]; }
    float acc[2][4][4], res[2][4][4];
    pd_ssd_zero(acc); pd_ssd_zero(res);
    // ---- inter: C . S_in over j ------------------------------------------
    const float* sin = ds + ((size_t)c * n_heads + h) * (S_ * HD_);
    for (uint32_t k0 = 0; k0 < S_; k0 += KS) {
        __syncthreads();
        for (uint32_t e = tid; e < L * KS; e += 256u) {
            const uint32_t t = e / KS, j = e % KS, row = base + t;
            sa[t * LDA + j] = (row < n_tok)
                ? xbc[(size_t)(t0 + row) * conv_dim + d_inner + (size_t)(n_groups + g) * S_ + k0 + j]
                : 0.0f;
        }
        for (uint32_t e = tid; e < KS * HD_; e += 256u)
            sbm[(e / HD_) * LDB + (e % HD_)] = sin[(size_t)k0 * HD_ + e];
        __syncthreads();
        #pragma unroll
        for (uint32_t k8 = 0; k8 < KS; k8 += 8u)
            pd_ssd_wstep<true, false>(acc, sa + m0 * LDA, LDA, sbm + n0, LDB, k8, gr, t4);
        pd_ssd_drain(res, acc);
    }
    {   // scale the inter sum by exp(cum_t), per row
        const float dec0 = expf(scum[m0 + gr]), dec1 = expf(scum[m0 + gr + 8u]);
        const float dec2 = expf(scum[m0 + 16u + gr]), dec3 = expf(scum[m0 + 16u + gr + 8u]);
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; ++nt) {
            res[0][nt][0] *= dec0; res[0][nt][1] *= dec0; res[0][nt][2] *= dec1; res[0][nt][3] *= dec1;
            res[1][nt][0] *= dec2; res[1][nt][1] *= dec2; res[1][nt][2] *= dec3; res[1][nt][3] *= dec3;
        }
    }
    // ---- intra: W . x over s, W built per slice ---------------------------
    const float* mg = m + ((size_t)c * n_groups + g) * L * L;
    for (uint32_t k0 = 0; k0 < L; k0 += KS) {
        __syncthreads();
        for (uint32_t e = tid; e < L * KS; e += 256u) {
            const uint32_t t = e / KS, s = k0 + (e % KS);
            sa[t * LDA + (s - k0)] = (s <= t)
                ? mg[(size_t)t * L + s] * expf(scum[t] - scum[s]) * sdt[s]
                : 0.0f;
        }
        for (uint32_t e = tid; e < KS * HD_; e += 256u) {
            const uint32_t s = e / HD_, i = e % HD_, row = base + k0 + s;
            sbm[s * LDB + i] = (row < n_tok)
                ? xbc[(size_t)(t0 + row) * conv_dim + (size_t)h * HD_ + i]
                : 0.0f;
        }
        __syncthreads();
        if (k0 < m0 + 32u) {
            #pragma unroll
            for (uint32_t k8 = 0; k8 < KS; k8 += 8u)
                pd_ssd_wstep<true, false>(acc, sa + m0 * LDA, LDA, sbm + n0, LDB, k8, gr, t4);
            pd_ssd_drain(res, acc);
        }
    }
    const float d_h = D[h];
    #pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt)
        #pragma unroll
        for (uint32_t hf = 0; hf < 2u; ++hf) {
            const uint32_t t = m0 + mt * 16u + gr + hf * 8u;
            const uint32_t row = base + t;
            if (row >= n_tok) continue;
            const float* xr = xbc + (size_t)(t0 + row) * conv_dim + (size_t)h * HD_;
            float* yr = y + (size_t)(t0 + row) * d_inner + (size_t)h * HD_;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; ++nt) {
                const uint32_t i = n0 + nt * 8u + 2u * t4;
                const float2 xv = *(const float2*)(xr + i);
                *(float2*)(yr + i) = make_float2(res[mt][nt][hf * 2u] + d_h * xv.x,
                                                 res[mt][nt][hf * 2u + 1u] + d_h * xv.y);
            }
        }
}
#endif
