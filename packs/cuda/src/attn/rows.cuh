// ── rows: split-K decode attention over runs of one slot's query rows ──────
//
// A speculative round attends with several query rows of the SAME slot at
// once: a verify chunk (row i of a run sits at pos0 + i and sees keys up to
// its own position - the causal tail) and a block drafter's mask rows (every
// row at the block end). Both had only the wrong classes to ride. The decode
// partial (lagd) reads the context once per ROW, so an 8-row verify
// re-streams it 8 times; the prefill tile reads it once but runs one CTA per
// (kv head, 4-token tile), so an 8-row nemotron verify was 4 CTAs walking a
// 24K context each - 2.0 ms a layer, ~12 GB/s, and the DFlash drafter's
// generic decode kernel 6.3 ms a layer on the same context (GB10
// 2026-09-26: at 24K the round cost 82 ms against 33 at 1.5K).
//
// FlashAttention-2's forward structure on lagd's staging. One CTA per (kv
// head, group of <= 8 rows of one slot, context split); warp w owns row w of
// the group with its G <= 16 heads as the m16 tile rows, and keeps q, the
// scores, the online softmax state and P in registers - a warp's rows are
// its own, so there is no cross-warp exchange and one barrier per tile
// fences the K/V ring. Every staged K/V tile serves all of the group's rows:
// the context streams once per group, split across CTAs like the decode
// partial, and the partials land in its [head][row][split] layout so the
// production combine finishes them.
//
// `window` > 0 bounds every row to its last `window` keys (a sliding-window
// drafter's layers - DSpark's 1024) the way the decode kernels do: first key
// pos + 1 - window, the group's span from its lowest first key.
//
// Numerics: lagd's default class - q as f16 big + residual (NXQ=2), P as f16
// big + mid + small (NXP=3) - so verify rows attend in the decode rows'
// class. K/V: the f16 pool as is, e4m3 raw-staged and expanded in place
// (exact, lagd's arm). A row whose split holds no key at or before its own
// position (the causal tail can empty a short last split) writes m = -inf,
// l = 0, o = 0, which the combine folds as nothing.
template <uint32_t TILE, uint32_t NXQ, uint32_t NXP, bool PAGED,
          typename KVT = __half>
__global__ void __launch_bounds__(256) pd_attn_rows_kernel(
    const float* __restrict__ q, const KVT* __restrict__ kc,
    const KVT* __restrict__ vc, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots, const uint32_t* __restrict__ groups,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t max_ctx, uint32_t n_heads, uint32_t n_kv_heads, uint32_t kv_dim,
    uint32_t n_rows, uint32_t n_splits, uint32_t window, float scale,
    uint32_t split_keys) {
#if PD_FA_OK
    constexpr uint32_t HD = 128u;
    constexpr bool F8 = sizeof(KVT) == 1u;
    constexpr uint32_t NT = TILE / 8u;        // score n8 tiles per K/V tile
    constexpr uint32_t row_e = HD + 8u;       // +8-half row pad (bank law)
    static_assert(TILE % 16u == 0u, "PV k-steps pair the score tiles");
    const uint32_t kvh = blockIdx.x, gi = blockIdx.y, sp = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t G = n_heads / n_kv_heads;

    const uint32_t row0 = groups[gi * 2u], nr = groups[gi * 2u + 1u];
    const uint32_t slot = slots ? slots[row0] : row0;
    // a row at p sees keys [first(p), p]: first 0, or p + 1 - window
    auto first_key = [&](uint32_t p) {
        return (window > 0u && p + 1u > window) ? p + 1u - window : 0u;
    };
    // the group's span covers every row's keys: [lowest first, max pos + 1)
    uint32_t pmax = 0, pmin_first = 0xffffffffu;
    for (uint32_t i = 0; i < nr; ++i) {
        const uint32_t p = positions[row0 + i];
        pmax = max(pmax, p);
        pmin_first = min(pmin_first, first_key(p));
    }
    const uint32_t end = pmax + 1u;
    // split law: the group's span shared evenly (split_keys 0), or FIXED key
    // ranges [sp * split_keys, (sp + 1) * split_keys) - a row's splits, tile
    // boundaries and fold then depend on its own keys alone, never on the
    // rows it shares a launch with, so a decode tick's row and a verify
    // round's attend bit for bit alike (the W16 decode class)
    uint32_t lo, hi;
    if (split_keys == 0u) {
        const uint32_t chunk = (end - pmin_first + n_splits - 1u) / n_splits;
        lo = min(pmin_first + sp * chunk, end);
        hi = min(lo + chunk, end);
    } else {
        lo = min(max(sp * split_keys, pmin_first), end);
        hi = min(sp * split_keys + split_keys, end);
        lo = min(lo, hi);
    }
    const bool live = warp < nr;
    const uint32_t row = row0 + (live ? warp : 0u);
    const uint32_t mypos = positions[row];
    const uint32_t myfirst = first_key(mypos);
    // a fixed split past the group's keys holds nothing for any of its rows:
    // m = -inf (the combine skips the o read on it) and out
    if (split_keys != 0u && lo >= hi) {
        if (live && t == 0) {
            const uint32_t h0 = kvh * G + g, h1 = kvh * G + g + 8u;
            if (g < G) {
                const size_t p0 = ((size_t)h0 * n_rows + row) * n_splits + sp;
                out_ml[p0 * 2u] = -INFINITY;
                out_ml[p0 * 2u + 1u] = 0.0f;
            }
            if (g + 8u < G) {
                const size_t p1 = ((size_t)h1 * n_rows + row) * n_splits + sp;
                out_ml[p1 * 2u] = -INFINITY;
                out_ml[p1 * 2u + 1u] = 0.0f;
            }
        }
        return;
    }

    extern __shared__ __align__(16) unsigned char rows_smraw[];
    __half* s_kv = (__half*)rows_smraw;   // [2][K,V][TILE][row_e]

    const uint32_t* bt =
        PAGED ? block_tables + (size_t)slot * blocks_per_slot : nullptr;
    const KVT* kcb = PAGED ? nullptr : kc + (size_t)slot * max_ctx * kv_dim;
    const KVT* vcb = PAGED ? nullptr : vc + (size_t)slot * max_ctx * kv_dim;

    // q A-fragments, resident for the whole walk: a0 (g, 2t), a1 (g+8, 2t),
    // a2 (g, 2t+8), a3 (g+8, 2t+8) of each 16-dim k-step; heads past G are
    // zero rows. NXQ=2 adds the residual plane (q - f16(q) is exact).
    uint32_t qa[NXQ][HD / 16u][4];
    {
        const float* qb = q + (size_t)row * n_heads * HD;
        #pragma unroll
        for (uint32_t ks = 0; ks < HD / 16u; ++ks) {
            #pragma unroll
            for (uint32_t part = 0; part < 4u; ++part) {
                const uint32_t hr = (part & 1u) ? g + 8u : g;
                const uint32_t col = ks * 16u + 2u * t + ((part & 2u) ? 8u : 0u);
                float v0 = 0.f, v1 = 0.f;
                if (live && hr < G) {
                    const float* src = qb + (size_t)(kvh * G + hr) * HD + col;
                    v0 = src[0];
                    v1 = src[1];
                }
                const __half b0 = __float2half(v0), b1 = __float2half(v1);
                __half2 h2 = __halves2half2(b0, b1);
                qa[0][ks][part] = *reinterpret_cast<uint32_t*>(&h2);
                if constexpr (NXQ == 2u) {
                    __half2 r2 = __halves2half2(__float2half(v0 - __half2float(b0)),
                                                __float2half(v1 - __half2float(b1)));
                    qa[1][ks][part] = *reinterpret_cast<uint32_t*>(&r2);
                }
            }
        }
    }

    // 16-byte cp.async lines per row: 16 for the f16 pool, 8 raw for e4m3
    // (landing at the FRONT of the destination f16 row slot - lagd's arm)
    constexpr uint32_t lines = (HD * sizeof(KVT)) >> 4;
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        if (n_t < TILE) {
            // zero the stale tail rows: PV multiplies them by exact-0
            // weights, but uninitialized smem can be NaN and 0*NaN = NaN
            for (uint32_t i = d; i < 2u * (TILE - n_t) * lines; i += nth) {
                const uint32_t kvsel = i / ((TILE - n_t) * lines);
                const uint32_t j = i - kvsel * (TILE - n_t) * lines;
                const uint32_t p = n_t + j / lines, l = j % lines;
                *(uint4*)((char*)(s_kv
                    + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e) + l * 16u)
                    = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (uint32_t i = d; i < 2u * n_t * lines; i += nth) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t j = i - kvsel * n_t * lines;
            const uint32_t p = j / lines, l = j - p * lines;
            const uint32_t gpos = t0 + p;
            const KVT* src;
            if (PAGED) {
                const uint32_t blk = bt[gpos >> 4];
                src = (kvsel ? vc : kc)
                    + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                    + (size_t)kvh * HD;
            } else {
                src = (kvsel ? vcb : kcb) + (size_t)gpos * kv_dim + (size_t)kvh * HD;
            }
            __half* dst = s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e;
            pd_attn_cpa16((char*)dst + l * 16u, (const char*)src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    // o frags: 16 n8 tiles over the head dim, rows g / g+8 per lane
    float o_acc[HD / 8u][4];
    #pragma unroll
    for (uint32_t n = 0; n < HD / 8u; ++n)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[n][j] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    if (lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += TILE, bf ^= 1u) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        const bool more = t0 + TILE < hi;
        if (more) stage(bf ^ 1u, t0 + TILE);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        if constexpr (F8) {
            // expand buffer bf's raw e4m3 rows to f16 in place (lagd's arm:
            // chunks to registers, one barrier, write the widened halves)
            constexpr uint32_t NCH = 2u * TILE * (HD / 16u);
            uint4 rv[(NCH + 255u) / 256u];
            uint32_t nc = 0;
            for (uint32_t c = d; c < NCH; c += nth, ++nc) {
                const uint32_t kvsel = c / (TILE * (HD / 16u));
                const uint32_t rem = c - kvsel * TILE * (HD / 16u);
                const uint32_t p = rem / (HD / 16u), j = rem % (HD / 16u);
                const char* rb = (const char*)(s_kv
                    + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e);
                rv[nc] = *(const uint4*)(rb + (size_t)j * 16u);
            }
            __syncthreads();
            nc = 0;
            for (uint32_t c = d; c < NCH; c += nth, ++nc) {
                const uint32_t kvsel = c / (TILE * (HD / 16u));
                const uint32_t rem = c - kvsel * TILE * (HD / 16u);
                const uint32_t p = rem / (HD / 16u), j = rem % (HD / 16u);
                __half* wb = s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e
                           + (size_t)j * 16u;
                const unsigned char* by = (const unsigned char*)&rv[nc];
                #pragma unroll
                for (uint32_t e = 0; e < 16u; ++e) {
                    __nv_fp8_e4m3 f8;
                    f8.__x = by[e];
                    wb[e] = __float2half(float(f8));   // both hops exact
                }
            }
            __syncthreads();
        }
        if (live) {
            const __half* kbuf = s_kv + (size_t)(bf * 2u) * TILE * row_e;
            const __half* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * TILE * row_e;
            // scores [16 heads x TILE keys]: K rows as B (no trans)
            float s[NT][4];
            #pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
                #pragma unroll
                for (uint32_t ks = 0; ks < HD / 16u; ++ks) {
                    uint32_t bfr[2];
                    const __half* bp = kbuf + (size_t)(j * 8u + (lane & 7u)) * row_e
                                     + ks * 16u + (((lane >> 3) & 1u) ? 8u : 0u);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    #pragma unroll
                    for (uint32_t x = 0; x < NXQ; ++x)
                        pd_fa_mma16(s[j], qa[x][ks][0], qa[x][ks][1], qa[x][ks][2],
                                    qa[x][ks][3], bfr[0], bfr[1]);
                }
            }
            // mask (the tile tail and the row's own bounds - causal and the
            // window; heads g and g+8 are the same query row) + scale; row
            // max over the quad
            float mx0 = -INFINITY, mx1 = -INFINITY;
            #pragma unroll
            for (uint32_t j = 0; j < NT; ++j)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t p = j * 8u + 2u * t + cc;
                    const bool ok = p < n_t && t0 + p <= mypos && t0 + p >= myfirst;
                    s[j][cc] = ok ? s[j][cc] * scale : -INFINITY;
                    s[j][2u + cc] = ok ? s[j][2u + cc] * scale : -INFINITY;
                    mx0 = fmaxf(mx0, s[j][cc]);
                    mx1 = fmaxf(mx1, s[j][2u + cc]);
                }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
            }
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            // a row with nothing live yet keeps a finite reference so no
            // -inf - -inf NaN enters the exponentials
            const float rf0 = mn0 == -INFINITY ? 0.0f : mn0;
            const float rf1 = mn1 == -INFINITY ? 0.0f : mn1;
            const float c0 = __expf(m0 - rf0), c1 = __expf(m1 - rf1);
            m0 = mn0;
            m1 = mn1;
            float ps0 = 0.0f, ps1 = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < NT; ++j)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    s[j][cc] = s[j][cc] > -INFINITY ? __expf(s[j][cc] - rf0) : 0.0f;
                    s[j][2u + cc] =
                        s[j][2u + cc] > -INFINITY ? __expf(s[j][2u + cc] - rf1) : 0.0f;
                    ps0 += s[j][cc];
                    ps1 += s[j][2u + cc];
                }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1) {
                ps0 += __shfl_xor_sync(0xffffffffu, ps0, off);
                ps1 += __shfl_xor_sync(0xffffffffu, ps1, off);
            }
            l0 = l0 * c0 + ps0;
            l1 = l1 * c1 + ps1;
            #pragma unroll
            for (uint32_t n = 0; n < HD / 8u; ++n) {
                o_acc[n][0] *= c0;
                o_acc[n][1] *= c0;
                o_acc[n][2] *= c1;
                o_acc[n][3] *= c1;
            }
            // PV: P straight from the score frags (the m16n8 C layout of two
            // adjacent n8 tiles IS the m16k16 A layout), V rows transposed
            #pragma unroll
            for (uint32_t kk = 0; kk < TILE / 16u; ++kk) {
                uint32_t pa[NXP][4];
                const float* src[4] = {&s[2u * kk][0], &s[2u * kk][2],
                                       &s[2u * kk + 1u][0], &s[2u * kk + 1u][2]};
                #pragma unroll
                for (uint32_t part = 0; part < 4u; ++part) {
                    const float w0 = src[part][0], w1 = src[part][1];
                    const __half h0 = __float2half(w0), h1 = __float2half(w1);
                    __half2 b2 = __halves2half2(h0, h1);
                    pa[0][part] = *reinterpret_cast<uint32_t*>(&b2);
                    if constexpr (NXP == 3u) {
                        // big + mid + small carries ~33 bits (> f32's 24)
                        const float r0 = w0 - __half2float(h0);
                        const float r1 = w1 - __half2float(h1);
                        const __half q0 = __float2half(r0), q1 = __float2half(r1);
                        __half2 mid = __halves2half2(q0, q1);
                        __half2 sml = __halves2half2(__float2half(r0 - __half2float(q0)),
                                                     __float2half(r1 - __half2float(q1)));
                        pa[1][part] = *reinterpret_cast<uint32_t*>(&mid);
                        pa[2][part] = *reinterpret_cast<uint32_t*>(&sml);
                    }
                }
                #pragma unroll
                for (uint32_t n = 0; n < HD / 8u; ++n) {
                    uint32_t bfr[2];
                    const __half* bp = vbuf + (size_t)(kk * 16u + (lane & 15u)) * row_e + n * 8u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    #pragma unroll
                    for (uint32_t x = 0; x < NXP; ++x)
                        pd_fa_mma16(o_acc[n], pa[x][0], pa[x][1], pa[x][2], pa[x][3],
                                    bfr[0], bfr[1]);
                }
            }
        }
        // every warp is past buffer bf before the next stage() overwrites it
        __syncthreads();
    }
    // partials in the decode partial's layout: [head][row][split], o raw
    // (relative to m), then (m, l) - every lane of a quad holds the same m/l
    if (live) {
        const uint32_t h0 = kvh * G + g, h1 = kvh * G + g + 8u;
        const size_t p0 = ((size_t)h0 * n_rows + row) * n_splits + sp;
        const size_t p1 = ((size_t)h1 * n_rows + row) * n_splits + sp;
        #pragma unroll
        for (uint32_t n = 0; n < HD / 8u; ++n) {
            const uint32_t col = n * 8u + 2u * t;
            if (g < G) {
                out_o[p0 * HD + col] = o_acc[n][0];
                out_o[p0 * HD + col + 1u] = o_acc[n][1];
            }
            if (g + 8u < G) {
                out_o[p1 * HD + col] = o_acc[n][2];
                out_o[p1 * HD + col + 1u] = o_acc[n][3];
            }
        }
        if (t == 0) {
            if (g < G) {
                out_ml[p0 * 2u] = m0;
                out_ml[p0 * 2u + 1u] = l0;
            }
            if (g + 8u < G) {
                out_ml[p1 * 2u] = m1;
                out_ml[p1 * 2u + 1u] = l1;
            }
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)out_o; (void)out_ml; (void)positions;
    (void)slots; (void)groups; (void)block_tables; (void)blocks_per_slot;
    (void)max_ctx; (void)n_heads; (void)n_kv_heads; (void)kv_dim; (void)n_rows;
    (void)n_splits; (void)window; (void)scale; (void)split_keys;
#endif
}

// The rows partial over `n_groups` groups (`groups` = [n_groups][2] u32 of
// (first row, rows <= 8), every group's rows one slot's and consecutive in
// the q plane). hd128, G <= 16. `paged` = the block-table pool (f16 or
// e4m3); otherwise a dense f16 cache [slots, max_ctx, kv_dim] (the drafter's).
// Partials: out_o [n_heads, n_rows, n_splits, 128], out_ml [..., 2], for the
// batch combine at `batch = n_rows`. `window` 0 = full attention.
static int pd_attn_rows_partial_go(const void* q, const void* kc, const void* vc,
                                   void* out_o, void* out_ml, const void* positions,
                                   const void* slots, const void* groups,
                                   uint32_t n_groups, const void* block_tables,
                                   uint32_t blocks_per_slot, uint32_t max_ctx,
                                   uint32_t n_heads, uint32_t n_kv_heads,
                                   uint32_t head_dim, uint32_t kv_dim, uint32_t n_rows,
                                   uint32_t n_splits, uint32_t window, float scale,
                                   uint32_t kv_dtype, uint32_t paged, uint32_t split_keys,
                                   void* stream) {
    if (n_groups == 0 || n_rows == 0) return 0;
    if (head_dim != 128u || n_kv_heads == 0 || n_heads % n_kv_heads != 0
        || n_heads / n_kv_heads > 16u || n_splits == 0)
        return cudaErrorInvalidValue;
    if (!paged && kv_dtype != PD_KV_FP16) return cudaErrorInvalidValue;
    constexpr uint32_t TILE = 32u;
    if (split_keys % TILE != 0u) return cudaErrorInvalidValue;
    const uint32_t smem = 2u * 2u * TILE * (128u + 8u) * 2u;   // 34,816 B
    const dim3 grid(n_kv_heads, n_groups, n_splits);
    const cudaStream_t st = (cudaStream_t)stream;
#define PD_ROWS_GO(PG, KT)                                                         \
    pd_attn_rows_kernel<TILE, 2u, 3u, PG, KT><<<grid, 256, smem, st>>>(            \
        (const float*)q, (const KT*)kc, (const KT*)vc, (float*)out_o,             \
        (float*)out_ml, (const unsigned int*)positions,                           \
        (const unsigned int*)slots, (const uint32_t*)groups,                      \
        (const uint32_t*)block_tables, blocks_per_slot, max_ctx, n_heads,         \
        n_kv_heads, kv_dim, n_rows, n_splits, window, scale, split_keys)
    if (!paged)
        PD_ROWS_GO(false, __half);
    else if (kv_dtype == PD_KV_FP8_E4M3)
        PD_ROWS_GO(true, __nv_fp8_e4m3);
    else
        PD_ROWS_GO(true, __half);
#undef PD_ROWS_GO
    return pd_launch_status();
}

// The rows partial over `n_groups` groups (`groups` = [n_groups][2] u32 of
// (first row, rows <= 8), every group's rows one slot's and consecutive in
// the q plane). hd128, G <= 16. `paged` = the block-table pool (f16 or
// e4m3); otherwise a dense f16 cache [slots, max_ctx, kv_dim] (the drafter's).
// Partials: out_o [n_heads, n_rows, n_splits, 128], out_ml [..., 2], for the
// batch combine at `batch = n_rows`. `window` 0 = full attention.
PD_EXPORT
int pd_attn_rows_partial(const void* q, const void* kc, const void* vc,
                         void* out_o, void* out_ml, const void* positions,
                         const void* slots, const void* groups,
                         uint32_t n_groups, const void* block_tables,
                         uint32_t blocks_per_slot, uint32_t max_ctx,
                         uint32_t n_heads, uint32_t n_kv_heads,
                         uint32_t head_dim, uint32_t kv_dim, uint32_t n_rows,
                         uint32_t n_splits, uint32_t window, float scale,
                         uint32_t kv_dtype, uint32_t paged, void* stream) {
    return pd_attn_rows_partial_go(q, kc, vc, out_o, out_ml, positions, slots, groups,
                                   n_groups, block_tables, blocks_per_slot, max_ctx,
                                   n_heads, n_kv_heads, head_dim, kv_dim, n_rows, n_splits,
                                   window, scale, kv_dtype, paged, 0u, stream);
}

// ABI 683: the rows partial on FIXED key splits - split s holds keys [s *
// split_keys, (s + 1) * split_keys) (a multiple of 32), so every row's
// partials depend on its own keys alone: the W16 decode class runs decode
// ticks (a group per row) and verify rounds through it on one law and one
// split count (n_splits = the context's worth), and the batch combine folds
// the same splits for both. Splits past a group's keys write m = -inf only.
PD_EXPORT
int pd_attn_rows_partial_fixed(const void* q, const void* kc, const void* vc,
                               void* out_o, void* out_ml, const void* positions,
                               const void* slots, const void* groups,
                               uint32_t n_groups, const void* block_tables,
                               uint32_t blocks_per_slot, uint32_t max_ctx,
                               uint32_t n_heads, uint32_t n_kv_heads,
                               uint32_t head_dim, uint32_t kv_dim, uint32_t n_rows,
                               uint32_t n_splits, uint32_t window, float scale,
                               uint32_t kv_dtype, uint32_t paged, uint32_t split_keys,
                               void* stream) {
    if (split_keys == 0u) return cudaErrorInvalidValue;
    return pd_attn_rows_partial_go(q, kc, vc, out_o, out_ml, positions, slots, groups,
                                   n_groups, block_tables, blocks_per_slot, max_ctx,
                                   n_heads, n_kv_heads, head_dim, kv_dim, n_rows, n_splits,
                                   window, scale, kv_dtype, paged, split_keys, stream);
}
