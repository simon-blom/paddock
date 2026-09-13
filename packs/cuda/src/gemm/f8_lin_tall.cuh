// f8_lin_tall.cuh - kt3t: the kt3 frame on a TALL tile (256 out rows x 128
// batch cols per CTA). Two W row boxes per K-128 stage share one staged Y
// box, so a CTA streams 1.5x the bytes of a kt3 tile for 2x the math: the
// L2-to-smem stream per FLOP - the thing kt3's mainloop is bound by on GB10
// (loads-only 633 us of the 1031, the tensor pipe
// alone 272) - drops by a quarter. The price is the ring: 2 x (2 x 16896 +
// 16384) = 100352 B leaves room for two stages under the 101376 opt-in, not
// kt3's three, and the accumulator doubles: 8 warps at 64 x 64 each,
// acc[32][4] = 128 registers plus a half-stage of fragments (am 32 + bm
// 32) - about 200 live. That does not fit a 288-thread block: registers
// are budgeted per 4-warp granule, so kt3's 9 warps are charged as 12 and
// ptxas caps the block at 168 registers (kt3 itself sits at 167; every
// "168 + spill" in the record - the 2026-07-27 cross-stage rotation, the
// 2026-07-29 N-fusion prototype - was this cap, never a 224 budget). So
// this kernel is 256 threads with no producer warp: lane 0 of warp 0
// issues the TMAs (the slot-release named barrier becomes warp 0's
// bar.sync against the other seven warps' bar.arrive), and 8 warps are
// budgeted as 8 - a 255-register cap. Same mma, same operands, same
// per-element K order (stage, half, k-block) as kt3 -> bit-exact vs kt3
// for every output element; the bench byte-compares them.
//
// NBOX = W boxes per stage (2 = the tall tile; 1 = kt3's tile on this
// frame, for the ring-depth twin), S = ring depth (2 or 3; NBOX=2 only
// fits S=2). Bar ids: 0 init, 1..S ring slots, 5 epilogue; immediate ids
// only (kt64 lesson). Late consumer arrive. Row tail: a tall tile whose
// second box lies past the plane (out_dim % 256 == 128) runs one box and
// idles its t=1 warps; the last row tile's partial rows take the guarded
// scalar store, as kt3's do.
// PD_KT3T_PROBE (bench-only): 1 = no mma, 2 = no TMA, 3 = no landing,
// 4 = no TMA and no landing (kt3's probe numbering).
#ifndef PD_KT3T_PROBE
#define PD_KT3T_PROBE 0
#endif
#if PD_KT3T_PROBE == 6
// probe 6: clock64 phase accounting per warp (cycles, atomicAdd'd by lane
// 0 at exit): [0] warp 0's refill bar.sync + issue, [1] mbarrier phase
// wait, [2] ldmatrix + mma, [3] slot arrive, [4] epilogue, [5] whole
// kernel per warp, [6] warps counted, [7] the stage-0 (prologue) share of
// [1]. The bench prints the split.
__device__ unsigned long long pd_kt3t_cnt[8];
#endif
// ring + mbarriers, never below the staged landing (128 x 132 f32 = 67584 B:
// the kt3-shaped 2-deep twin's ring is smaller than that)
#define PD_KT3T_RING(NBOX, S) ((S) * (16384u + (NBOX) * PD_LIN_BOX) + 16u)
#define PD_KT3T_SMEM(NBOX, S) (PD_KT3T_RING(NBOX, S) > 67584u ? PD_KT3T_RING(NBOX, S) : 67584u)

#if PD_F8W8_TMA_OK && PD_BS_OK
// one half-stage (K-64) of fragments for a 64-row x (8*JN)-col warp:
// kt3's addressing verbatim, JN n-tiles instead of four
template <uint32_t JN>
static __device__ __forceinline__ void pd_kt3t_ldh(
    const unsigned char* wp, const unsigned char* yp, uint32_t h,
    uint32_t lane, uint32_t i0, uint32_t c0w,
    uint32_t (&am)[4][2][4], uint32_t (&bm)[JN][4], uint32_t (&sa)[4]) {
    const uint32_t g = lane >> 2, tq = lane & 3u;
    #pragma unroll
    for (uint32_t s = 0; s < 4u; ++s) {
        const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
        #pragma unroll
        for (uint32_t kb = 0; kb < 2u; ++kb) {
            const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
            pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
        }
        const uint32_t r0 = i0 + s * 16u + g;
        const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
        sa[s] = *(const unsigned short*)(wp + PD_LIN_DATA + rs * 4u + h * 2u);
    }
    #pragma unroll
    for (uint32_t j = 0; j < JN; ++j) {
        const uint32_t col = c0w + j * 8u + (lane & 7u);
        const uint32_t c = h * 4u + (lane >> 3);
        pd_ldm_x4(bm[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
    }
}

// kt3's mma order (k-block outer, then n-tile, then m-tile): per acc
// element the K sequence is kb0 then kb1 - the bit-exact gate rides on it
template <uint32_t JN>
static __device__ __forceinline__ void pd_kt3t_mma(
    float (&acc)[4u * JN][4], const uint32_t (&am)[4][2][4],
    const uint32_t (&bm)[JN][4], const uint32_t (&sa)[4],
    const uint32_t (&sb)[JN]) {
    #pragma unroll
    for (uint32_t j = 0; j < JN; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_bs_mma_w8_kb<0>(acc[s * JN + j], am[s][0][0], am[s][0][1],
                               am[s][0][2], am[s][0][3], bm[j][0], bm[j][1],
                               sa[s], sb[j]);
    #pragma unroll
    for (uint32_t j = 0; j < JN; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_bs_mma_w8_kb<1>(acc[s * JN + j], am[s][1][0], am[s][1][1],
                               am[s][1][2], am[s][1][3], bm[j][2], bm[j][3],
                               sa[s], sb[j]);
}
#endif

template <bool O16, uint32_t NBOX = 2u, uint32_t S = 2u>
__global__ void __launch_bounds__(256, 1) pd_f8_gemm_lin_kt3t(
    const unsigned char* __restrict__ wlin, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t pf) {
    static_assert(NBOX == 1u || NBOX == 2u, "one or two W boxes per stage");
    static_assert(S == 2u || S == 3u, "ring depth 2 or 3");
    static_assert(PD_KT3T_SMEM(NBOX, S) <= 101376u, "ring exceeds the opt-in");
#if PD_F8W8_TMA_OK && PD_BS_OK
    constexpr uint32_t BOX = PD_LIN_BOX;
    constexpr uint32_t PAIR16 = 16384u;
    constexpr uint32_t STAGE = PAIR16 + NBOX * BOX;
    constexpr uint32_t RG = 2u * NBOX;     // 64-row warp groups per CTA
    constexpr uint32_t CG = 8u / RG;       // col groups
    constexpr uint32_t WC = 128u / CG;     // cols per warp (64 tall, 32 kt3-shaped)
    constexpr uint32_t JN = WC / 8u;       // n-tiles per warp
    extern __shared__ __align__(128) unsigned char pd_lin_sht[];
    unsigned char* ydat = pd_lin_sht;                    // [S][16 KB] (1024-aligned: SW128 image)
    unsigned char* wdat = pd_lin_sht + S * PAIR16;       // [S][NBOX][BOX]
    unsigned long long* mb = (unsigned long long*)(pd_lin_sht + S * STAGE);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nsp = in_dim >> 7;                    // K-128 stages (host rejects in_dim % 128)
    const uint32_t nrt = (out_dim + 127u) >> 7;          // plane row tiles
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * (128u * NBOX);
    const uint32_t col_base = (tile % nct) * 128u;
    const uint32_t rt0 = row_base >> 7;
    const uint32_t nbox_live = (rt0 + NBOX <= nrt) ? NBOX : (nrt - rt0);
    const unsigned char* wboxes = wlin + (size_t)rt0 * nsp * BOX;
    const uint32_t lane = tid & 31u, warp = tid >> 5;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        #pragma unroll
        for (uint32_t i = 0; i < S; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0 + i * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 256;");

    // stage issue: thread 0 only. expect_tx for the live boxes + the Y box,
    // then the bulk copies; under the no-TMA probes a plain arrive
    // completes the phase on stale smem.
    auto issue = [&](uint32_t sp, uint32_t b) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
#if PD_KT3T_PROBE == 2 || PD_KT3T_PROBE == 4
        (void)sp;
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
#else
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"(nbox_live * BOX + PAIR16));
        for (uint32_t t = 0; t < nbox_live; ++t) {
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + (b * NBOX + t) * BOX);
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];" ::"r"(wd),
                "l"(wboxes + ((size_t)t * nsp + sp) * BOX), "r"(BOX), "r"(m)
                : "memory");
        }
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
        const int ck = (int)(sp * 128u);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
            "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
            : "memory");
        // L2 prefetch of the W boxes pf stages ahead: the ring's round trip
        // is DRAM latency for the first of the eight col-tile sharers and
        // an L2 miss-in-flight for the rest; with the boxes already in L2
        // the same bytes in flight turn over faster (Little's law - the
        // 2-deep ring holds one stage in flight).
        if (pf != 0u && sp + pf < nsp) {
            for (uint32_t t = 0; t < nbox_live; ++t)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                             ::"l"(wboxes + ((size_t)t * nsp + sp + pf) * BOX), "r"(BOX)
                             : "memory");
        }
#endif
    };
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t sp = 0; sp < S; ++sp)
            if (sp < nsp) issue(sp, sp);
    }

    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t rg = warp % RG, cg = warp / RG;
    const uint32_t t = rg >> 1;              // this warp's W box
    const uint32_t i0 = (rg & 1u) * 64u;     // row half inside the box
    const uint32_t c0w = cg * WC;
    // M-tail: a whole col group past the batch skips its loads and mma
    // (warp-uniform); the ring protocol stays on all 256 threads (kt3's
    // rule). A missing second box idles its warps the same way.
    const bool warp_live = (col_base + c0w < batch) && (t < nbox_live);

    float acc[4u * JN][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;
#if PD_KT3T_PROBE == 6
    long long c_sync = 0, c_wait = 0, c_math = 0, c_arr = 0, c_epi = 0, c_w0 = 0;
    const long long ck0 = clock64();
#define PD_KT3T_T(v) const long long v = clock64();
#else
#define PD_KT3T_T(v)
#endif

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % S;
        PD_KT3T_T(t0)
        // refill: stage sp-1's slot, released by the other seven warps at
        // the end of their iteration sp-1 (bar.arrive), takes stage sp-1+S.
        // warp 0's bar.sync is the eighth party of that barrier generation.
        if (warp == 0u && sp >= 1u && sp - 1u + S < nsp) {
            const uint32_t br = (sp - 1u) % S;
            if (br == 0u)      asm volatile("bar.sync 1, 256;");
            else if (br == 1u) asm volatile("bar.sync 2, 256;");
            else               asm volatile("bar.sync 3, 256;");
            if (lane == 0u) issue(sp - 1u + S, br);
        }
        PD_KT3T_T(t1)
        // x-scales straight from L2, issued before the phase wait (kt3)
        uint32_t sbj[2][JN];
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h)
                #pragma unroll
                for (uint32_t j = 0; j < JN; ++j) {
                    const uint32_t col = col_base + c0w + j * 8u + g;
                    const uint32_t ccol = col < batch ? col : (batch - 1u);
                    sbj[h][j] = *(const unsigned short*)(
                        xs + (size_t)ccol * n_kb + (sp * 2u + h) * 2u);
                }
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT3T_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT3T_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;
        PD_KT3T_T(t2)

        const unsigned char* wp = wdat + (b * NBOX + t) * BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                uint32_t am[4][2][4], bm[JN][4], sa[4];
                pd_kt3t_ldh<JN>(wp, yp, h, lane, i0, c0w, am, bm, sa);
#if PD_KT3T_PROBE == 1
                acc[0][0] += __uint_as_float(am[0][0][0] & 1u) + __uint_as_float(bm[0][0] & 1u) + __uint_as_float(sa[0] & 1u);
#else
                pd_kt3t_mma<JN>(acc, am, bm, sa, sbj[h]);
#endif
            }
        }
        PD_KT3T_T(t3)
        // free slot b (warps 1-7; warp 0 releases through its bar.sync at
        // the next refill): late arrive - the h=1 mma consumed its reads
        if (warp != 0u) {
            if (b == 0u)      asm volatile("bar.arrive 1, 256;");
            else if (b == 1u) asm volatile("bar.arrive 2, 256;");
            else              asm volatile("bar.arrive 3, 256;");
        }
#if PD_KT3T_PROBE == 6
        const long long t4 = clock64();
        c_sync += t1 - t0; c_wait += t2 - t1; c_math += t3 - t2; c_arr += t4 - t3;
        if (sp == 0u) c_w0 = t2 - t1;
#endif
    }
#if PD_KT3T_PROBE == 6
    const long long ce0 = clock64();
#endif

#if PD_KT3T_PROBE == 3 || PD_KT3T_PROBE == 4
    if (acc[0][0] == 1.2345e30f && acc[4u * JN - 1u][3] == 2.3456e30f) y[tid] = acc[1][2];
    return;
#endif
    // Coalesced landing through the idle ring, one 128-row box at a time
    // (kt3's staged epilogue: fragments parked as [batch col][out row] at a
    // 132-float pitch, then 512-byte f32 / 256-byte bf16 row runs). Every
    // warp is past the last phase wait and no TMA is in flight, so the ring
    // is free once the warps meet at barrier 5.
    float* otile = (float*)pd_lin_sht;   // 128 x 132 f32 = 67.6 KB
    __nv_bfloat16* yh = (__nv_bfloat16*)y;
#if PD_KT3T_PROBE == 7
    // landing into an L2-resident window: every CTA stores its tile at the
    // origin (same instructions, same bytes per CTA, no DRAM write stream)
    const uint32_t row_base_st = 0u, col_base_st = 0u;
#else
    const uint32_t row_base_st = row_base, col_base_st = col_base;
#endif
    for (uint32_t rh = 0; rh < nbox_live; ++rh) {
        asm volatile("bar.sync 5, 256;");   // ring free / previous box streamed out
        if (t == rh) {
            #pragma unroll
            for (uint32_t j = 0; j < JN; ++j) {
                const uint32_t cl = c0w + j * 8u + 2u * tq;
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const uint32_t rl = i0 + s * 16u + g;
                    otile[cl * 132u + rl] = acc[s * JN + j][0];
                    otile[(cl + 1u) * 132u + rl] = acc[s * JN + j][1];
                    otile[cl * 132u + rl + 8u] = acc[s * JN + j][2];
                    otile[(cl + 1u) * 132u + rl + 8u] = acc[s * JN + j][3];
                }
            }
        }
        asm volatile("bar.sync 5, 256;");
        const uint32_t rb = row_base_st + rh * 128u;
        if (rb + 128u <= out_dim) {
            for (uint32_t it = warp; it < 128u; it += 8u) {
                const uint32_t c = col_base_st + it;
                if (c >= batch) continue;
                const float4 v = *(const float4*)(otile + it * 132u + lane * 4u);
                if (O16) {
                    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
                    __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
                    uint2 pk; pk.x = *(const uint32_t*)&lo; pk.y = *(const uint32_t*)&hi;
                    *(uint2*)(yh + (size_t)c * out_dim + rb + lane * 4u) = pk;
                } else {
                    *(float4*)(y + (size_t)c * out_dim + rb + lane * 4u) = v;
                }
            }
        } else {
            // row tail (out_dim % 128 != 0 in the plane's last row tile)
            for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
                const uint32_t r = i & 127u, c = i >> 7;
                if (rb + r < out_dim && col_base + c < batch) {
                    const float v = otile[c * 132u + r];
                    if (O16) yh[(size_t)(col_base + c) * out_dim + rb + r] = __float2bfloat16(v);
                    else y[(size_t)(col_base + c) * out_dim + rb + r] = v;
                }
            }
        }
    }
#if PD_KT3T_PROBE == 6
    c_epi = clock64() - ce0;
    if (lane == 0u) {
        atomicAdd(&pd_kt3t_cnt[0], (unsigned long long)c_sync);
        atomicAdd(&pd_kt3t_cnt[1], (unsigned long long)c_wait);
        atomicAdd(&pd_kt3t_cnt[2], (unsigned long long)c_math);
        atomicAdd(&pd_kt3t_cnt[3], (unsigned long long)c_arr);
        atomicAdd(&pd_kt3t_cnt[4], (unsigned long long)c_epi);
        atomicAdd(&pd_kt3t_cnt[5], (unsigned long long)(clock64() - ck0));
        atomicAdd(&pd_kt3t_cnt[6], 1ull);
        atomicAdd(&pd_kt3t_cnt[7], (unsigned long long)c_w0);   // stage-0 (prologue) wait
    }
#endif
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)pf;
#endif
}

// Election + launch, called from pd_f8_gemm_lin_kt's kt3 branch (forward-
// declared there; this segment follows f8_lin.cuh in pack.cu). Returns
// false when off or the shape is outside the tile's band: the caller keeps
// its kt3 route. PADDOCK_LIN_KT3T=1 turns the arm on (default off - see the
// `on` note); PADDOCK_LIN_KT3T_MIN overrides the row floor (default 2048,
// the only band the tile beats kt3). PD_KT3T_SHAPE (bench-only) picks the
// twin: 0 = tall 2-deep, 1 = kt3's tile on this frame at 3-deep, 2 = kt3's
// tile at 2-deep (the ring-depth twin - not bit-exact, a numeric-order
// check only). PD_KT3T_PF sets an L2 prefetch distance (falsified: pf>0
// only slowed it).
static bool pd_f8_lin_kt3t_try(const void* wlin, const CUtensorMap& ym, const void* xs,
                               void* y, uint32_t in_dim, uint32_t out_dim,
                               uint32_t batch, uint32_t o16, cudaStream_t stream,
                               int* status) {
#if !(defined(PD_BS_HOST))
    (void)wlin; (void)ym; (void)xs; (void)y; (void)in_dim; (void)out_dim;
    (void)batch; (void)o16; (void)stream; (void)status;
    return false;
#else
    static const bool on = [] {
        // OPT-IN, default off: a measured non-win at the batch GB10 serving
        // reaches (<= ~1024 prefill rows per tick), so the shipped election
        // keeps kt3. Kept as a bit-exact iteration vehicle (the ktd/ktw/kt5
        // precedent) and the tall-tile frame; PADDOCK_LIN_KT3T=1 enables it.
        const char* e = pd_env("PADDOCK_LIN_KT3T");
        return e != nullptr && atoi(e) != 0 && pd_dev_bs_sass();
    }();
    static const uint32_t min_rows = [] {
        const char* e = pd_env("PADDOCK_LIN_KT3T_MIN");
        return e ? (uint32_t)atoi(e) : 2048u;   // the only band it wins
    }();
    static const int shape = [] {
        const char* e = pd_env("PD_KT3T_SHAPE");
        return e ? atoi(e) : 0;
    }();
    static const uint32_t pf = [] {
        const char* e = pd_env("PD_KT3T_PF");
        return e ? (uint32_t)atoi(e) : 0u;
    }();
    if (!on || batch < min_rows) return false;
    const uint32_t nct = ((batch + 127u) & ~127u) >> 7;
    const uint32_t nbox = shape == 0 ? 2u : 1u;
    const uint32_t nt = ((out_dim + 128u * nbox - 1u) / (128u * nbox)) * nct;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3t<true, 2u, 2u>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_KT3T_SMEM(2u, 2u));
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3t<false, 2u, 2u>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_KT3T_SMEM(2u, 2u));
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3t<true, 1u, 3u>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_KT3T_SMEM(1u, 3u));
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3t<false, 1u, 3u>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_KT3T_SMEM(1u, 3u));
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3t<true, 1u, 2u>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_KT3T_SMEM(1u, 2u));
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3t<false, 1u, 2u>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_KT3T_SMEM(1u, 2u));
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3t<true, 2u, 2u>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3t<false, 2u, 2u>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3t<true, 1u, 3u>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3t<false, 1u, 3u>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3t<true, 1u, 2u>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3t<false, 1u, 2u>);
        attr = true;
    }
    const unsigned char* w = (const unsigned char*)wlin;
    const unsigned char* x = (const unsigned char*)xs;
    float* yo = (float*)y;
    if (shape == 1) {
        if (o16) pd_f8_gemm_lin_kt3t<true, 1u, 3u><<<nt, 256, PD_KT3T_SMEM(1u, 3u), stream>>>(w, ym, x, yo, in_dim, out_dim, batch, pf);
        else     pd_f8_gemm_lin_kt3t<false, 1u, 3u><<<nt, 256, PD_KT3T_SMEM(1u, 3u), stream>>>(w, ym, x, yo, in_dim, out_dim, batch, pf);
    } else if (shape == 2) {
        if (o16) pd_f8_gemm_lin_kt3t<true, 1u, 2u><<<nt, 256, PD_KT3T_SMEM(1u, 2u), stream>>>(w, ym, x, yo, in_dim, out_dim, batch, pf);
        else     pd_f8_gemm_lin_kt3t<false, 1u, 2u><<<nt, 256, PD_KT3T_SMEM(1u, 2u), stream>>>(w, ym, x, yo, in_dim, out_dim, batch, pf);
    } else {
        if (o16) pd_f8_gemm_lin_kt3t<true, 2u, 2u><<<nt, 256, PD_KT3T_SMEM(2u, 2u), stream>>>(w, ym, x, yo, in_dim, out_dim, batch, pf);
        else     pd_f8_gemm_lin_kt3t<false, 2u, 2u><<<nt, 256, PD_KT3T_SMEM(2u, 2u), stream>>>(w, ym, x, yo, in_dim, out_dim, batch, pf);
    }
    *status = pd_launch_status();
    return true;
#endif
}
