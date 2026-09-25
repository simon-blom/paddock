// Bidirectional attention over PACKED variable-length sequences, on the half
// interface - the text encoders' attention: ModernBERT's 28 layers under the
// Laya decision model, and the decision head's two plain encoder layers.
//
// The workload that shaped it. A Laya request is one sequence per question -
// [CLS] question [SEP] options [SEP] state [SEP] - each 15 to 1024 tokens,
// several per request, and a pass packs every queued request's sequences. So
// the unit is a RAGGED batch, and padding it to the longest sequence (what a
// [batch, max_len] SDPA does) would spend most of a pass on pad rows: the
// reference battery's lengths run 15..512 in one request. Instead the rows
// are packed back to back (`cu[s]..cu[s+1]` is sequence s - FlashAttention's
// varlen / ModernBERT's own "unpadded" layout) and each block owns one 64-row
// query tile of one sequence, named by a host-built tile list. Nothing is
// padded but the last tile of each sequence.
//
// What rides the operand staging instead of a pass of its own:
//   - the q/k/v split: q, k and v are read straight out of the fused
//     projection's landing ([rows][3][heads][hd], row stride 3d), so no
//     split kernel and no three planes;
//   - RoPE (rotate_half form, cos/sin tables [pos][hd/2] f32, positions
//     relative to the sequence start): q is rotated in registers as its A
//     fragment loads - a lane's fragment holds dims c and c + hd/2 of the
//     same row, so the pair meets without a shuffle - and k is rotated as its
//     tile stages, eight dims and their eight partners per thread-step;
//   - the q/k/v bias of torch's nn.MultiheadAttention (in_proj_bias), in f32
//     before the round;
//   - 1/sqrt(hd), folded into q before its one round to f16 (as v3c and the
//     vision kernel do).
// So every value enters the tensor cores rounded once, from f32.
//
// The window: ModernBERT's local layers attend |i - j| <= window (64 - HF's
// `sliding_window`, local_attention / 2, inclusive both sides). A tile's key
// span is [q0 - w, q_last + w] clipped to the sequence, so a local layer
// walks ~3 key tiles whatever the sequence length; every score outside a
// row's own window is masked. Full layers pass window 0.
//
// Determinism: every block's work is a function of (sequence, query tile,
// head) and the key tiles it walks are laid from the sequence start, so a
// sequence's output does not depend on what else was packed with it - the
// batch invariance the decision endpoint promises rests on this and on the
// GEMMs' fixed K order.
//
// The tile loop is pd_vision_attn_mma_kernel's (vision.cuh), which is also
// where its one measured fact comes from: at this key tile a double-buffered
// cp.async stage bought nothing - the loop is issue/latency-bound, not
// stage-bound - so the stage here is the same synchronous one, now doing the
// rotation too. The mask and the key span are the only other changes.

#define PD_VL_QW 4u                         // warps a block, 16 query rows each
#define PD_VL_QT (PD_VL_QW * 16u)           // query rows a block
#define PD_VL_KT 64u                        // keys staged per tile
#define PD_VL_SMEM(DP) (2u * PD_VL_KT * ((DP) + 8u) * 2u)
// a tile descriptor is (sequence << PD_VL_TILE_SHIFT) | query-tile index:
// 4096 tiles of 64 rows covers any sequence to 256k tokens
#define PD_VL_TILE_SHIFT 12u

// eight halves at `p` (16 B aligned) as floats, plus an optional f32 bias
__device__ __forceinline__ void pd_vl_ld8(const __half* p, const float* b, float (&o)[8]) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const float2 f = __half22float2(h2[i]);
        o[2u * i] = f.x;
        o[2u * i + 1u] = f.y;
    }
    if (b != nullptr) {
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) o[i] += b[i];
    }
}
__device__ __forceinline__ void pd_vl_st8(__half* p, const float (&v)[8]) {
    uint4 raw;
    __half2* h2 = reinterpret_cast<__half2*>(&raw);
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) h2[i] = __floats2half2_rn(v[2u * i], v[2u * i + 1u]);
    *reinterpret_cast<uint4*>(p) = raw;
}

template <uint32_t DP, bool ROPE, bool BIAS>
__global__ void __launch_bounds__(32u * PD_VL_QW) pd_enc_attn_kernel(
    const __half* __restrict__ qkv, const uint32_t* __restrict__ cu,
    const uint32_t* __restrict__ tiles, const float* __restrict__ cosb,
    const float* __restrict__ sinb, const float* __restrict__ bias,
    __half* __restrict__ out, uint32_t n_heads, uint32_t hd, uint32_t window,
    float scale) {
#if PD_FA_OK
    constexpr uint32_t KT = PD_VL_KT, DPD = DP + 8u, NT = 32u * PD_VL_QW;
    constexpr uint32_t HALF = DP / 2u, HB = HALF / 16u;   // rope pairs: d0 and d0 + HB
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t h = blockIdx.y;
    const uint32_t td = tiles[blockIdx.x];
    const uint32_t s = td >> PD_VL_TILE_SHIFT;
    const uint32_t q0 = (td & ((1u << PD_VL_TILE_SHIFT) - 1u)) * PD_VL_QT;
    const uint32_t base = cu[s], L = cu[s + 1u] - base;
    const uint32_t d = n_heads * hd;
    const size_t rs = 3u * (size_t)d;      // the fused landing's row stride
    const uint32_t wq0 = q0 + warp * 16u;  // this warp's first query row
    const float* bq = BIAS ? bias + (size_t)h * hd : nullptr;
    const float* bk = BIAS ? bias + d + (size_t)h * hd : nullptr;
    const float* bv = BIAS ? bias + 2u * (size_t)d + (size_t)h * hd : nullptr;

    extern __shared__ unsigned char vlsh[];
    __half* sh_k = reinterpret_cast<__half*>(vlsh);
    __half* sh_v = sh_k + (size_t)KT * DPD;

    // ---- q: load (+ bias), rotate, scale, round - one round from f32 ----
    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t qa[DP / 16u][4];
    {
        float qf[DP / 16u][2][4];
#pragma unroll
        for (uint32_t d0 = 0; d0 < DP / 16u; ++d0) {
#pragma unroll
            for (uint32_t e = 0; e < 2u; ++e) {
                const uint32_t qi = wq0 + jr[e];
                const bool ok = qi < L;
                const __half* qp = qkv + (size_t)(base + (ok ? qi : 0u)) * rs + (size_t)h * hd;
                const uint32_t c0 = d0 * 16u + 2u * t4, c1 = c0 + 8u;
                const float2 a = (ok && c0 < hd)
                    ? __half22float2(*reinterpret_cast<const __half2*>(qp + c0)) : make_float2(0.f, 0.f);
                const float2 b = (ok && c1 < hd)
                    ? __half22float2(*reinterpret_cast<const __half2*>(qp + c1)) : make_float2(0.f, 0.f);
                qf[d0][e][0] = a.x;
                qf[d0][e][1] = a.y;
                qf[d0][e][2] = b.x;
                qf[d0][e][3] = b.y;
                if (BIAS && ok) {
                    if (c0 < hd) { qf[d0][e][0] += bq[c0]; qf[d0][e][1] += bq[c0 + 1u]; }
                    if (c1 < hd) { qf[d0][e][2] += bq[c1]; qf[d0][e][3] += bq[c1 + 1u]; }
                }
            }
        }
        if (ROPE) {
#pragma unroll
            for (uint32_t d0 = 0; d0 < HB; ++d0) {
#pragma unroll
                for (uint32_t e = 0; e < 2u; ++e) {
                    const uint32_t qi = wq0 + jr[e];
                    const uint32_t pos = qi < L ? qi : 0u;
                    const uint32_t c0 = d0 * 16u + 2u * t4;
                    const uint32_t cs[4] = {c0, c0 + 1u, c0 + 8u, c0 + 9u};
#pragma unroll
                    for (uint32_t i = 0; i < 4u; ++i) {
                        const float co = cosb[(size_t)pos * HALF + cs[i]];
                        const float si = sinb[(size_t)pos * HALF + cs[i]];
                        const float x1 = qf[d0][e][i], x2 = qf[d0 + HB][e][i];
                        qf[d0][e][i] = x1 * co - x2 * si;
                        qf[d0 + HB][e][i] = x2 * co + x1 * si;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t d0 = 0; d0 < DP / 16u; ++d0) {
#pragma unroll
            for (uint32_t e = 0; e < 2u; ++e) {
                const __half2 p0 = __floats2half2_rn(qf[d0][e][0] * scale, qf[d0][e][1] * scale);
                const __half2 p1 = __floats2half2_rn(qf[d0][e][2] * scale, qf[d0][e][3] * scale);
                qa[d0][e] = *reinterpret_cast<const uint32_t*>(&p0);
                qa[d0][e + 2u] = *reinterpret_cast<const uint32_t*>(&p1);
            }
        }
    }

    // ---- the key span this tile needs ----
    const uint32_t qlast = (q0 + PD_VL_QT < L ? q0 + PD_VL_QT : L) - 1u;
    const uint32_t lo = window ? (q0 > window ? q0 - window : 0u) : 0u;
    const uint32_t hi = window ? (qlast + window + 1u < L ? qlast + window + 1u : L) : L;

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[DP / 8u][4];
#pragma unroll
    for (uint32_t nt = 0; nt < DP / 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    for (uint32_t t0 = lo; t0 < hi; t0 += KT) {
        // K: rotated in pairs (8 dims + their 8 partners a step), or plain
        if (ROPE) {
            constexpr uint32_t KU = KT * (HALF / 8u);
            for (uint32_t u = tid; u < KU; u += NT) {
                const uint32_t kk = u / (HALF / 8u), d8 = (u % (HALF / 8u)) * 8u;
                const uint32_t ks = t0 + kk;
                __half* dst = sh_k + (size_t)kk * DPD + d8;
                if (ks < hi) {
                    const __half* src = qkv + (size_t)(base + ks) * rs + d + (size_t)h * hd + d8;
                    float x1[8], x2[8];
                    pd_vl_ld8(src, BIAS ? bk + d8 : nullptr, x1);
                    pd_vl_ld8(src + HALF, BIAS ? bk + d8 + HALF : nullptr, x2);
                    const float* cr = cosb + (size_t)ks * HALF + d8;
                    const float* sr = sinb + (size_t)ks * HALF + d8;
                    float r1[8], r2[8];
#pragma unroll
                    for (uint32_t i = 0; i < 8u; ++i) {
                        r1[i] = x1[i] * cr[i] - x2[i] * sr[i];
                        r2[i] = x2[i] * cr[i] + x1[i] * sr[i];
                    }
                    pd_vl_st8(dst, r1);
                    pd_vl_st8(dst + HALF, r2);
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                    *reinterpret_cast<uint4*>(dst + HALF) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        } else {
            constexpr uint32_t KU = KT * (DP / 8u);
            for (uint32_t u = tid; u < KU; u += NT) {
                const uint32_t kk = u / (DP / 8u), d8 = (u % (DP / 8u)) * 8u;
                const uint32_t ks = t0 + kk;
                __half* dst = sh_k + (size_t)kk * DPD + d8;
                if (ks < hi && d8 < hd) {
                    const __half* src = qkv + (size_t)(base + ks) * rs + d + (size_t)h * hd + d8;
                    if (BIAS) {
                        float x[8];
                        pd_vl_ld8(src, bk + d8, x);
                        pd_vl_st8(dst, x);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                    }
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        // V: plain (+ bias)
        {
            constexpr uint32_t VU = KT * (DP / 8u);
            for (uint32_t u = tid; u < VU; u += NT) {
                const uint32_t kk = u / (DP / 8u), d8 = (u % (DP / 8u)) * 8u;
                const uint32_t ks = t0 + kk;
                __half* dst = sh_v + (size_t)kk * DPD + d8;
                if (ks < hi && d8 < hd) {
                    const __half* src = qkv + (size_t)(base + ks) * rs + 2u * (size_t)d
                                        + (size_t)h * hd + d8;
                    if (BIAS) {
                        float x[8];
                        pd_vl_ld8(src, bv + d8, x);
                        pd_vl_st8(dst, x);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                    }
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        float s_acc[KT / 8u][4];
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t d0 = 0; d0 < DP / 16u; ++d0) {
#pragma unroll
            for (uint32_t np = 0; np < KT / 16u; ++np) {
                const __half* kp = sh_k
                    + (size_t)(np * 16u + (lg >> 1) * 8u + (lane & 7u)) * DPD
                    + d0 * 16u + (lg & 1u) * 8u;
                uint32_t kb4[4];
                const uint32_t ka = (uint32_t)__cvta_generic_to_shared(kp);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(kb4[0]), "=r"(kb4[1]), "=r"(kb4[2]), "=r"(kb4[3]) : "r"(ka));
                pd_fa_mma16(s_acc[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[0], kb4[1]);
                pd_fa_mma16(s_acc[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[2], kb4[3]);
            }
        }
        // mask: past the span, and (local layers) outside the row's window
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
            const uint32_t kb = t0 + nt * 8u + 2u * t4;
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t kj = kb + (e & 1u);
                const uint32_t qi = wq0 + jr[e >> 1];
                const bool in_win = window == 0u || (kj + window >= qi && kj <= qi + window);
                if (kj >= hi || !in_win) s_acc[nt][e] = -1e30f;
                mn[e >> 1] = fmaxf(mn[e >> 1], s_acc[nt][e]);
            }
        }
#pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float ws[2] = {0.f, 0.f};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const float dd = s_acc[nt][e] - mn[e >> 1];
                const float w = dd >= -20.f ? __expf(dd) : 0.f;
                s_acc[nt][e] = w;
                ws[e >> 1] += w;
            }
        }
#pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], o);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], o);
        }
        float corr[2];
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const float dc = m_st[r] - mn[r];
            corr[r] = dc >= -20.f ? __expf(dc) : 0.f;
            l_st[r] = l_st[r] * corr[r] + ws[r];
            m_st[r] = mn[r];
        }
#pragma unroll
        for (uint32_t nt = 0; nt < DP / 8u; ++nt) {
            o_acc[nt][0] *= corr[0];
            o_acc[nt][1] *= corr[0];
            o_acc[nt][2] *= corr[1];
            o_acc[nt][3] *= corr[1];
        }
#pragma unroll
        for (uint32_t kf = 0; kf < KT / 16u; ++kf) {
            const uint32_t c0 = 2u * kf, c1 = c0 + 1u;
            const __half2 a0 = __floats2half2_rn(s_acc[c0][0], s_acc[c0][1]);
            const __half2 a1 = __floats2half2_rn(s_acc[c0][2], s_acc[c0][3]);
            const __half2 a2 = __floats2half2_rn(s_acc[c1][0], s_acc[c1][1]);
            const __half2 a3 = __floats2half2_rn(s_acc[c1][2], s_acc[c1][3]);
            const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&a0);
            const uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&a1);
            const uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&a2);
            const uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&a3);
            const uint32_t vr = kf * 16u + (lg & 1u) * 8u + (lane & 7u);
            const __half* vp = sh_v + (size_t)vr * DPD + (lg >> 1) * 8u;
#pragma unroll
            for (uint32_t nt = 0; nt < DP / 8u; nt += 2u) {
                uint32_t vb4[4];
                const uint32_t va = (uint32_t)__cvta_generic_to_shared(vp + nt * 8u);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(vb4[0]), "=r"(vb4[1]), "=r"(vb4[2]), "=r"(vb4[3]) : "r"(va));
                pd_fa_mma16(o_acc[nt], pa0, pa1, pa2, pa3, vb4[0], vb4[1]);
                pd_fa_mma16(o_acc[nt + 1u], pa0, pa1, pa2, pa3, vb4[2], vb4[3]);
            }
        }
        __syncthreads();   // tiles read before the next stage overwrites them
    }

    const float nrm[2] = {l_st[0] > 0.f ? 1.f / l_st[0] : 0.f,
                          l_st[1] > 0.f ? 1.f / l_st[1] : 0.f};
#pragma unroll
    for (uint32_t nt = 0; nt < DP / 8u; ++nt) {
        const uint32_t dcol = nt * 8u + 2u * t4;
        if (dcol >= hd) continue;
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t qi = wq0 + jr[r];
            if (qi >= L) continue;
            __half* op = out + (size_t)(base + qi) * d + (size_t)h * hd + dcol;
            *reinterpret_cast<__half2*>(op) = __floats2half2_rn(o_acc[nt][2u * r] * nrm[r],
                                                                o_acc[nt][2u * r + 1u] * nrm[r]);
        }
    }
#else
    (void)qkv; (void)cu; (void)tiles; (void)cosb; (void)sinb; (void)bias; (void)out;
    (void)n_heads; (void)hd; (void)window; (void)scale;
#endif
}

// 673: packed variable-length bidirectional attention off a fused qkv landing.
//   qkv    [rows][3][heads][hd] f16 - q | k | v, each [heads][hd]
//   cu     [n_seq + 1] u32 row offsets (cu[0] = 0)
//   tiles  [n_tiles] u32, (seq << 12) | query tile, one per 64 query rows
//   cosb/sinb  [max_pos][hd/2] f32 rope tables, both null for no rope
//   bias   [3 * heads * hd] f32 (q | k | v) or null
//   out    [rows][heads][hd] f16
//   window 0 = full attention, else |i - j| <= window
// head_dim in {32, 64, 96, 128}; sm_80+. The caller guarantees every position
// it asks rope for is inside the tables.
PD_EXPORT
int pd_enc_attn_h(const void* qkv, const void* cu, const void* tiles, uint32_t n_tiles,
                  const void* cosb, const void* sinb, const void* bias, void* out,
                  uint32_t n_heads, uint32_t head_dim, uint32_t window, void* stream) {
    if (n_tiles == 0 || n_heads == 0) return 0;
    const bool rope = cosb != nullptr;
    if (rope != (sinb != nullptr)) return cudaErrorInvalidValue;
    if (head_dim != 32u && head_dim != 64u && head_dim != 96u && head_dim != 128u)
        return cudaErrorInvalidValue;
    int dev = 0, cc = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
    if (cc < 8) return cudaErrorInvalidValue;
    const bool has_b = bias != nullptr;
    dim3 grid(n_tiles, n_heads);
    const float scale = 1.0f / sqrtf((float)head_dim);
    static bool attr = false;
    if (!attr) {
#define PD_VL_PREF(DP)                                               \
        pd_prefer_max_shared(pd_enc_attn_kernel<DP, false, false>);  \
        pd_prefer_max_shared(pd_enc_attn_kernel<DP, false, true>);   \
        pd_prefer_max_shared(pd_enc_attn_kernel<DP, true, false>);   \
        pd_prefer_max_shared(pd_enc_attn_kernel<DP, true, true>);
        PD_VL_PREF(32u)
        PD_VL_PREF(64u)
        PD_VL_PREF(96u)
        PD_VL_PREF(128u)
#undef PD_VL_PREF
        attr = true;
    }
    cudaStream_t st = (cudaStream_t)stream;
    const __half* q = (const __half*)qkv;
    const uint32_t* c = (const uint32_t*)cu;
    const uint32_t* t = (const uint32_t*)tiles;
    const float* co = (const float*)cosb;
    const float* si = (const float*)sinb;
    const float* b = (const float*)bias;
    __half* o = (__half*)out;
#define PD_VL_GO(DP, R, B)                                                            \
    pd_enc_attn_kernel<DP, R, B><<<grid, 32u * PD_VL_QW, PD_VL_SMEM(DP), st>>>(         \
        q, c, t, co, si, b, o, n_heads, head_dim, window, scale)
#define PD_VL_CASE(DP)                                                                \
    case DP:                                                                          \
        if (rope) { if (has_b) PD_VL_GO(DP, true, true); else PD_VL_GO(DP, true, false); } \
        else { if (has_b) PD_VL_GO(DP, false, true); else PD_VL_GO(DP, false, false); }  \
        break;
    switch (head_dim) {
        PD_VL_CASE(32u)
        PD_VL_CASE(64u)
        PD_VL_CASE(96u)
        PD_VL_CASE(128u)
        default: return cudaErrorInvalidValue;
    }
#undef PD_VL_CASE
#undef PD_VL_GO
    return pd_launch_status();
}
