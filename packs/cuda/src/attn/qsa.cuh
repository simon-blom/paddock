// QSA - the sparse attention of Qwen3.8-Flash-Next (qwen4_exp), part 1: the
// indexer's query and its compressed key cache.
//
// Per attention layer an indexer projects the layer's input to 4 query heads
// and 1 raw key of 128 each. Keys are pooled 4 tokens at a time: when the
// 4th token of block b = [4b, 4b+3] lands,
//     k_b = RoPE(k_norm(bf16(mean_f32(raw_k[4b..4b+3]))), pos = 4b)
// and a query at position i scores its floor((i+1)/4) complete blocks with
// sum_h relu(q_h . k_b), keeps the top 512 plus the tail (i+1) mod 4 tokens,
// and attends to those only. Semantics: HF transformers
// `modular_qwen4_exp.py` L368-475 (4b28d51d0d); the cache shape - compressed
// rows only, on the main page geometry, with a small ring of raw keys - is
// the one vLLM and SGLang serve (studied, not copied). The plan is the
// Flash-Next QSA design note.
//
// Rounding follows the reference where it is cheap to: raw keys and the
// pooled mean are bf16 values (the reference projects in bf16), the norm runs
// in f32 and its output is rounded to bf16, the stored key is bf16. The RoPE
// between runs in f32 through pd_mrope - the same launch the main attention
// rotates with, so both sides of every score share one rotary.
//
// Everything here takes a FIXED grid of all rows (CUDA-graph capture: which
// rows complete a block changes every tick, the launch shape must not).
// Rows of one slot inside a launch are contiguous and consecutive in position
// (a prompt run, a verify chunk, or a lone decode row) - the contract every
// walk of this lane already keeps.

static __device__ __forceinline__ float pd_qsa_bf16r(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// Block-wide sum over blockDim.x (a multiple of 32, <= 1024) threads.
static __device__ __forceinline__ float pd_qsa_block_sum(float v, float* ws) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (uint32_t s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    float t = 0.0f;
    if (threadIdx.x == 0) {
        const uint32_t nw = blockDim.x >> 5;
        for (uint32_t i = 0; i < nw; ++i) t += ws[i];
        ws[32] = t;
    }
    __syncthreads();
    return ws[32];
}

// ---------------------------------------------------------------- query
//
// The indexer query, normalized: row r's q heads sit at src[r*ld + h*hd ..]
// (the fused q|k projection row), dst is [rows, heads, hd] contiguous for the
// pd_mrope that follows. Per head: f32 RMS, (1+w) in the FMA form every
// (1+w) norm of this family uses, rounded to bf16 (the reference's q is a
// bf16 tensor from here on). One CTA per (row, head), one thread per dim.
__global__ void pd_q4x_idx_q_kernel(const float* __restrict__ src,
                                    const float* __restrict__ w,
                                    float* __restrict__ dst, uint32_t heads,
                                    uint32_t hd, uint32_t ld, float eps) {
    __shared__ float ws[33];
    const uint32_t r = blockIdx.x / heads, h = blockIdx.x % heads, i = threadIdx.x;
    const float v = src[(size_t)r * ld + (size_t)h * hd + i];
    const float ss = pd_qsa_block_sum(v * v, ws);
    const float inv = 1.0f / sqrtf(ss / (float)hd + eps);
    const float xn = v * inv;
    dst[((size_t)r * heads + h) * hd + i] = pd_qsa_bf16r(xn + xn * w[i]);
}

PD_EXPORT
int pd_q4x_idx_q(const void* src, const void* w, void* dst, uint32_t rows,
                 uint32_t heads, uint32_t hd, uint32_t ld, float eps, void* stream) {
    if (rows == 0 || heads == 0) return 0;
    if (hd == 0 || hd > 1024 || (hd & 31u) != 0) return -1;
    pd_q4x_idx_q_kernel<<<rows * heads, hd, 0, (cudaStream_t)stream>>>(
        (const float*)src, (const float*)w, (float*)dst, heads, hd, ld, eps);
    return pd_launch_status();
}

// ---------------------------------------------------------------- pool
//
// For every row whose position p closes a block ((p+1) % cr == 0): gather the
// block's cr raw keys, mean them in f32, round to bf16, RMS-normalize with
// (1+w), round to bf16, and stage the result at stage[r] with its block's
// first position in spos ([4, rows] axis-major - the pd_mrope layout; text
// rotates every axis by the same position). Other rows write nothing.
//
// A block's raw keys come from THIS launch when their rows are in it (the
// run's own earlier rows: same slot, consecutive positions) and from the
// slot's ring otherwise - positions before this launch, written by an earlier
// pd_q4x_idx_store. The ring is read here and written only by the store,
// which runs after: a long run's late rows would otherwise overwrite ring
// entries its first rows are still reading.
__global__ void pd_q4x_idx_pool_kernel(const float* __restrict__ raw,
                                       const float* __restrict__ ring,
                                       const uint32_t* __restrict__ pos,
                                       const uint32_t* __restrict__ slots,
                                       const float* __restrict__ w,
                                       float* __restrict__ stage,
                                       uint32_t* __restrict__ spos, uint32_t rows,
                                       uint32_t hd, uint32_t ld, uint32_t koff,
                                       uint32_t ring_len, uint32_t cr, float eps) {
    __shared__ float ws[33];
    const uint32_t r = blockIdx.x, i = threadIdx.x;
    const uint32_t p = pos[r], s = slots[r];
    if ((p + 1u) % cr != 0u) return;   // uniform per CTA
    float sum = 0.0f;
    for (uint32_t j = 0; j < cr; ++j) {
        const uint32_t q = p + 1u - cr + j;   // this block's j-th token
        const uint32_t d = p - q;             // rows back, if it is in this launch
        const float* src;
        if (d <= r && slots[r - d] == s && pos[r - d] == q) {
            src = raw + (size_t)(r - d) * ld + koff;
        } else {
            src = ring + ((size_t)s * ring_len + q % ring_len) * hd;
        }
        sum += pd_qsa_bf16r(src[i]);
    }
    const float m = pd_qsa_bf16r(sum / (float)cr);
    const float ss = pd_qsa_block_sum(m * m, ws);
    const float inv = 1.0f / sqrtf(ss / (float)hd + eps);
    const float xn = m * inv;
    stage[(size_t)r * hd + i] = pd_qsa_bf16r(xn + xn * w[i]);
    if (i < 4u) spos[(size_t)i * rows + r] = p + 1u - cr;
}

PD_EXPORT
int pd_q4x_idx_pool(const void* raw, const void* ring, const void* pos, const void* slots,
                    const void* w, void* stage, void* spos, uint32_t rows, uint32_t hd,
                    uint32_t ld, uint32_t koff, uint32_t ring_len, uint32_t cr, float eps,
                    void* stream) {
    if (rows == 0) return 0;
    if (hd == 0 || hd > 1024 || (hd & 31u) != 0 || cr == 0 || ring_len < cr) return -1;
    pd_q4x_idx_pool_kernel<<<rows, hd, 0, (cudaStream_t)stream>>>(
        (const float*)raw, (const float*)ring, (const uint32_t*)pos, (const uint32_t*)slots,
        (const float*)w, (float*)stage, (uint32_t*)spos, rows, hd, ld, koff, ring_len, cr,
        eps);
    return pd_launch_status();
}

// ---------------------------------------------------------------- store
//
// After pd_mrope rotated the staged keys: a row that closed a block writes it
// (bf16) to the slot's compressed cache at block p / cr - `cap` blocks per
// slot, slot-major like the lane's KV. Then every row files its raw key in the
// slot's ring at p % ring_len, except a row a LATER row of this launch
// (ring_len rows on, same slot, position p + ring_len) would overwrite: only
// the last ring_len rows of a run land, so no two writers race for one entry.
// ring_len >= cr + the deepest verify chunk keeps a rejected draft's entry
// from ever aliasing a committed position the next pool still reads.
__global__ void pd_q4x_idx_store_kernel(const float* __restrict__ raw,
                                        const float* __restrict__ stage,
                                        const uint32_t* __restrict__ pos,
                                        const uint32_t* __restrict__ slots,
                                        __nv_bfloat16* __restrict__ cache,
                                        float* __restrict__ ring, uint32_t rows,
                                        uint32_t hd, uint32_t ld, uint32_t koff,
                                        uint32_t ring_len, uint32_t cr, uint32_t cap) {
    const uint32_t r = blockIdx.x, i = threadIdx.x;
    const uint32_t p = pos[r], s = slots[r];
    if ((p + 1u) % cr == 0u) {
        cache[((size_t)s * cap + p / cr) * hd + i] =
            __float2bfloat16_rn(stage[(size_t)r * hd + i]);
    }
    const uint32_t later = r + ring_len;
    const bool shadowed = later < rows && slots[later] == s && pos[later] == p + ring_len;
    if (!shadowed) {
        ring[((size_t)s * ring_len + p % ring_len) * hd + i] = raw[(size_t)r * ld + koff + i];
    }
}

PD_EXPORT
int pd_q4x_idx_store(const void* raw, const void* stage, const void* pos, const void* slots,
                     void* cache, void* ring, uint32_t rows, uint32_t hd, uint32_t ld,
                     uint32_t koff, uint32_t ring_len, uint32_t cr, uint32_t cap,
                     void* stream) {
    if (rows == 0) return 0;
    if (hd == 0 || hd > 1024 || cr == 0 || ring_len < cr) return -1;
    pd_q4x_idx_store_kernel<<<rows, hd, 0, (cudaStream_t)stream>>>(
        (const float*)raw, (const float*)stage, (const uint32_t*)pos, (const uint32_t*)slots,
        (__nv_bfloat16*)cache, (float*)ring, rows, hd, ld, koff, ring_len, cr, cap);
    return pd_launch_status();
}

// =====================================================================
// Part 2: the selection - score every visible block, keep the top 512.
//
// Query i sees nb = floor((i+1)/cr) complete blocks. score_b = sum over the
// indexer heads of relu(q_h . k_b) (the reference's 1/sqrt(hd) is monotone
// and dropped); the top min(512, nb) blocks are selected, the tail tokens
// [cr*nb, i] always attend and are not listed. nb <= 512 selects everything
// (and needs no scores). Ties: the reference's topk leaves them unspecified;
// here the LOWEST block ids win, deterministically.

// Order-preserving float -> uint32 (larger float, larger key).
static __device__ __forceinline__ uint32_t pd_qsa_key(float s) {
    const uint32_t u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------- logits
//
// scores[b_row * cap + b] for b < nb of every row b_row of this batch
// (rows row0 .. row0+gridDim.x of the launch arrays). grid = (rows, cap/256)
// FIXED by the slot capacity - a decode graph replays across positions - and
// tiles past a row's nb exit at once; rows that select everything (nb <= k)
// write nothing. One thread per block: the key's 128 bf16 in 16-byte loads,
// the row's `heads` query vectors from shared memory. Each K row is read once
// per query row - at 128K a layer's compressed keys (8 MB) sit in GB10's L2,
// which is what makes that affordable for prefill rows; the tensor-core
// shared-tile form is the perf rung after this one.
__global__ void pd_q4x_qsa_logits_kernel(const float* __restrict__ q,
                                         const __nv_bfloat16* __restrict__ cache,
                                         const uint32_t* __restrict__ pos,
                                         const uint32_t* __restrict__ slots,
                                         float* __restrict__ scores, uint32_t row0,
                                         uint32_t heads, uint32_t hd, uint32_t cap,
                                         uint32_t cr, uint32_t k) {
    extern __shared__ float qs[];   // [heads][hd]
    const uint32_t br = blockIdx.x, r = row0 + br;
    const uint32_t nb = (pos[r] + 1u) / cr;
    const uint32_t tile0 = blockIdx.y * blockDim.x;
    if (nb <= k || tile0 >= nb) return;   // uniform per CTA
    for (uint32_t i = threadIdx.x; i < heads * hd; i += blockDim.x) {
        qs[i] = q[(size_t)r * heads * hd + i];
    }
    __syncthreads();
    const uint32_t b = tile0 + threadIdx.x;
    if (b >= nb) return;
    const uint4* kr = reinterpret_cast<const uint4*>(cache + ((size_t)slots[r] * cap + b) * hd);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // heads <= 4
    for (uint32_t c = 0; c < hd / 8u; ++c) {
        const uint4 v = kr[c];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
        float kf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            kf[2 * j] = f.x;
            kf[2 * j + 1] = f.y;
        }
        for (uint32_t h = 0; h < heads; ++h) {
            const float* qh = qs + h * hd + c * 8u;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[h] = fmaf(qh[j], kf[j], acc[h]);
        }
    }
    float s = 0.0f;
    for (uint32_t h = 0; h < heads; ++h) s += fmaxf(acc[h], 0.0f);
    scores[(size_t)br * cap + b] = s;
}

PD_EXPORT
int pd_q4x_qsa_logits(const void* q, const void* cache, const void* pos, const void* slots,
                      void* scores, uint32_t row0, uint32_t rows, uint32_t heads,
                      uint32_t hd, uint32_t cap, uint32_t cr, uint32_t k, void* stream) {
    if (rows == 0 || cap == 0) return 0;
    if (heads == 0 || heads > 4 || hd == 0 || (hd & 7u) != 0 || cr == 0) return -1;
    dim3 grid(rows, (cap + 255u) / 256u);
    pd_q4x_qsa_logits_kernel<<<grid, 256, heads * hd * sizeof(float), (cudaStream_t)stream>>>(
        (const float*)q, (const __nv_bfloat16*)cache, (const uint32_t*)pos,
        (const uint32_t*)slots, (float*)scores, row0, heads, hd, cap, cr, k);
    return pd_launch_status();
}

// ---------------------------------------------------------------- top-k
//
// One CTA per row: exact radix select of the k largest of scores[0, nb) -
// four 8-bit passes over the order-preserving keys find the k-th key T and
// how many keys equal to it are still wanted, then one ordered pass writes
// every key > T and the first wanted keys == T, in ascending block order
// (block-wide scans give each its slot) - so the list comes out sorted, the
// order the attention gather walks the cache in. sel[r*k ..], cnt[r].
#define PD_QSA_TOPK_THREADS 256u

static __device__ __forceinline__ uint32_t pd_qsa_scan_excl(uint32_t v, uint32_t* ws,
                                                            uint32_t* total) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    uint32_t x = v;
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) ws[warp] = x;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t run = 0;
        for (uint32_t w = 0; w < PD_QSA_TOPK_THREADS / 32u; ++w) {
            const uint32_t t = ws[w];
            ws[w] = run;
            run += t;
        }
        ws[PD_QSA_TOPK_THREADS / 32u] = run;
    }
    __syncthreads();
    const uint32_t r = ws[warp] + x - v;
    *total = ws[PD_QSA_TOPK_THREADS / 32u];
    __syncthreads();
    return r;
}

__global__ __launch_bounds__(PD_QSA_TOPK_THREADS) void pd_q4x_qsa_topk_kernel(
    const float* __restrict__ scores, const uint32_t* __restrict__ pos,
    uint32_t* __restrict__ sel, uint32_t* __restrict__ cnt, uint32_t row0, uint32_t cap,
    uint32_t cr, uint32_t k) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t ws[PD_QSA_TOPK_THREADS / 32u + 1u];
    __shared__ uint32_t s_prefix, s_want;
    const uint32_t br = blockIdx.x, r = row0 + br, tid = threadIdx.x;
    const uint32_t nb = (pos[r] + 1u) / cr;
    uint32_t* out = sel + (size_t)r * k;
    if (nb <= k) {   // everything is selected
        for (uint32_t i = tid; i < nb; i += PD_QSA_TOPK_THREADS) out[i] = i;
        if (tid == 0) cnt[r] = nb;
        return;
    }
    const float* sc = scores + (size_t)br * cap;
    uint32_t prefix = 0, mask = 0, want = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t i = tid; i < 256u; i += PD_QSA_TOPK_THREADS) hist[i] = 0;
        __syncthreads();
        for (uint32_t i = tid; i < nb; i += PD_QSA_TOPK_THREADS) {
            const uint32_t u = pd_qsa_key(sc[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t above = 0, d = 0;
            for (int bin = 255; bin >= 0; --bin) {
                const uint32_t h = hist[bin];
                if (above + h >= want) {
                    d = (uint32_t)bin;
                    break;
                }
                above += h;
            }
            s_prefix = prefix | (d << shift);
            s_want = want - above;
        }
        __syncthreads();
        prefix = s_prefix;
        want = s_want;
        mask |= 0xffu << shift;
        __syncthreads();
    }
    // prefix = the k-th largest key; `want` of the keys equal to it are taken
    uint32_t base = 0, eq_seen = 0;
    for (uint32_t c0 = 0; c0 < nb; c0 += PD_QSA_TOPK_THREADS) {
        const uint32_t i = c0 + tid;
        const uint32_t u = i < nb ? pd_qsa_key(sc[i]) : 0u;
        const uint32_t gt = (i < nb && u > prefix) ? 1u : 0u;
        const uint32_t eq = (i < nb && u == prefix) ? 1u : 0u;
        uint32_t eq_total, take_total;
        const uint32_t eq_rank = eq_seen + pd_qsa_scan_excl(eq, ws, &eq_total);
        const uint32_t take = gt | ((eq && eq_rank < want) ? 1u : 0u);
        const uint32_t slot = base + pd_qsa_scan_excl(take, ws, &take_total);
        if (take) out[slot] = i;
        base += take_total;
        eq_seen += eq_total;
    }
    if (tid == 0) cnt[r] = k;
}

PD_EXPORT
int pd_q4x_qsa_topk(const void* scores, const void* pos, void* sel, void* cnt, uint32_t row0,
                    uint32_t rows, uint32_t cap, uint32_t cr, uint32_t k, void* stream) {
    if (rows == 0) return 0;
    if (cr == 0 || k == 0) return -1;
    pd_q4x_qsa_topk_kernel<<<rows, PD_QSA_TOPK_THREADS, 0, (cudaStream_t)stream>>>(
        (const float*)scores, (const uint32_t*)pos, (uint32_t*)sel, (uint32_t*)cnt, row0, cap,
        cr, k);
    return pd_launch_status();
}

// =====================================================================
// Part 3: attention over the selection.
//
// Row r (a query at position p, slot s) attends to the tokens of its
// selected blocks - sel[r*k .. r*k+cnt[r]], ascending, 4 tokens each - then
// its tail [cr*nb, p], nb = (p+1)/cr. With nb <= k the selection is every
// block and this is exactly causal dense attention. All heads of a kv group
// share the selection (one per token per layer, as the reference).
//
// SIMT gather, correctness-first: grid (rows, kv heads, splits), one CTA per
// kv head's G query heads, a split's share of the row's tokens walked in
// tiles of 16 through shared memory (K rows padded one float against bank
// conflicts), f32 online softmax per head, each thread one output dim for
// all G heads. Split partials (acc, m, l) merge in pd_q4x_qsa_combine. The
// KV is the lane's slot-major cache (row = slot*max_ctx + position), f16 or
// unscaled e4m3 read through pd_kv_load. The tensor-core form is its own
// perf rung; this one is the parity anchor.
#define PD_QSA_TT 16u
#define PD_QSA_G_MAX 16u

template <typename KV>
__global__ __launch_bounds__(256) void pd_q4x_qsa_attn_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc, const KV* __restrict__ vc,
    const uint32_t* __restrict__ pos, const uint32_t* __restrict__ slots,
    const uint32_t* __restrict__ sel, const uint32_t* __restrict__ cnt,
    float* __restrict__ part_o, float* __restrict__ part_ml, uint32_t nh, uint32_t nkv,
    uint32_t hd, uint32_t max_ctx, uint32_t k, uint32_t cr, float scale) {
    extern __shared__ float sm[];
    const uint32_t G = nh / nkv;
    const uint32_t r = blockIdx.x, kvh = blockIdx.y, sp = blockIdx.z, ns = gridDim.z;
    const uint32_t tid = threadIdx.x;
    float* qs = sm;                                // [G][hd]
    float* ks = qs + G * hd;                       // [TT][hd+1]
    float* vs = ks + PD_QSA_TT * (hd + 1u);        // [TT][hd]
    float* ps = vs + PD_QSA_TT * hd;               // [G][TT]
    float* hs = ps + G * PD_QSA_TT;                // [G]: this tile's rescale
    uint32_t* ts = reinterpret_cast<uint32_t*>(hs + PD_QSA_G_MAX);   // [TT] kv rows

    const uint32_t p = pos[r], s = slots[r];
    const uint32_t nb = (p + 1u) / cr;
    const uint32_t c = cnt[r];
    const uint32_t ntok = c * cr + (p + 1u - nb * cr);
    const uint32_t per = (ntok + ns - 1u) / ns;
    const uint32_t t0 = sp * per, t1 = min(ntok, t0 + per);
    const uint32_t* rs = sel + (size_t)r * k;
    const size_t kvw = (size_t)nkv * hd;           // one cache row, elements

    for (uint32_t i = tid; i < G * hd; i += blockDim.x) {
        qs[i] = q[((size_t)r * nh + kvh * G) * hd + i] * scale;
    }
    float m_run = -3.0e38f, l_run = 0.0f;          // per head, in thread h < G
    float acc[PD_QSA_G_MAX];
#pragma unroll
    for (uint32_t g = 0; g < PD_QSA_G_MAX; ++g) acc[g] = 0.0f;
    __syncthreads();

    for (uint32_t tb = t0; tb < t1; tb += PD_QSA_TT) {
        const uint32_t nt = min(PD_QSA_TT, t1 - tb);
        if (tid < nt) {
            const uint32_t t = tb + tid;           // this row's t-th token
            const uint32_t tp = t < c * cr ? rs[t / cr] * cr + t % cr : nb * cr + (t - c * cr);
            ts[tid] = s * max_ctx + tp;
        }
        __syncthreads();
        for (uint32_t i = tid; i < nt * hd; i += blockDim.x) {
            const uint32_t j = i / hd, d = i % hd;
            const size_t off = (size_t)ts[j] * kvw + (size_t)kvh * hd + d;
            ks[j * (hd + 1u) + d] = pd_kv_load(kc[off]);
            vs[j * hd + d] = pd_kv_load(vc[off]);
        }
        __syncthreads();
        // scores: one thread per (head, token)
        for (uint32_t i = tid; i < G * PD_QSA_TT; i += blockDim.x) {
            const uint32_t g = i / PD_QSA_TT, j = i % PD_QSA_TT;
            float d0 = -3.0e38f;
            if (j < nt) {
                const float* qg = qs + g * hd;
                const float* kj = ks + j * (hd + 1u);
                float a = 0.0f;
                for (uint32_t d = 0; d < hd; ++d) a = fmaf(qg[d], kj[d], a);
                d0 = a;
            }
            ps[g * PD_QSA_TT + j] = d0;
        }
        __syncthreads();
        // online softmax: thread g owns head g's running (m, l)
        if (tid < G) {
            float mx = m_run;
            for (uint32_t j = 0; j < nt; ++j) mx = fmaxf(mx, ps[tid * PD_QSA_TT + j]);
            const float alpha = __expf(m_run - mx);
            float sum = 0.0f;
            for (uint32_t j = 0; j < PD_QSA_TT; ++j) {
                const float e = j < nt ? __expf(ps[tid * PD_QSA_TT + j] - mx) : 0.0f;
                ps[tid * PD_QSA_TT + j] = e;
                sum += e;
            }
            l_run = l_run * alpha + sum;
            m_run = mx;
            hs[tid] = alpha;
        }
        __syncthreads();
        // P.V: thread d owns output dim d of every head
        for (uint32_t d = tid; d < hd; d += blockDim.x) {
            for (uint32_t g = 0; g < G; ++g) {
                float a = acc[g] * hs[g];
                for (uint32_t j = 0; j < nt; ++j) a = fmaf(ps[g * PD_QSA_TT + j], vs[j * hd + d], a);
                acc[g] = a;
            }
        }
        __syncthreads();
    }
    const size_t pbase = (((size_t)r * nkv + kvh) * ns + sp) * G;
    for (uint32_t d = tid; d < hd; d += blockDim.x) {
        for (uint32_t g = 0; g < G; ++g) part_o[(pbase + g) * hd + d] = acc[g];
    }
    if (tid < G) {
        part_ml[(pbase + tid) * 2u] = m_run;
        part_ml[(pbase + tid) * 2u + 1u] = l_run;
    }
}

PD_EXPORT
int pd_q4x_qsa_attn(const void* q, const void* kc, const void* vc, const void* pos,
                    const void* slots, const void* sel, const void* cnt, void* part_o,
                    void* part_ml, uint32_t rows, uint32_t nh, uint32_t nkv, uint32_t hd,
                    uint32_t max_ctx, uint32_t k, uint32_t cr, uint32_t splits, float scale,
                    uint32_t kv_dtype, void* stream) {
    if (rows == 0) return 0;
    if (nkv == 0 || nh % nkv != 0 || nh / nkv > PD_QSA_G_MAX || hd == 0 || hd > 256 ||
        cr == 0 || splits == 0)
        return -1;
    const uint32_t G = nh / nkv;
    // an f32 accumulator per head per thread covers hd <= blockDim
    const size_t smem = sizeof(float) * ((size_t)G * hd + PD_QSA_TT * (hd + 1u) +
                                         PD_QSA_TT * hd + G * PD_QSA_TT + PD_QSA_G_MAX) +
                        sizeof(uint32_t) * PD_QSA_TT;
    dim3 grid(rows, nkv, splits);
    if (kv_dtype == PD_KV_FP8_E4M3) {
        cudaFuncSetAttribute(pd_q4x_qsa_attn_kernel<__nv_fp8_e4m3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        pd_q4x_qsa_attn_kernel<__nv_fp8_e4m3><<<grid, 256, smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
            (const uint32_t*)pos, (const uint32_t*)slots, (const uint32_t*)sel,
            (const uint32_t*)cnt, (float*)part_o, (float*)part_ml, nh, nkv, hd, max_ctx, k, cr,
            scale);
    } else {
        cudaFuncSetAttribute(pd_q4x_qsa_attn_kernel<__half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        pd_q4x_qsa_attn_kernel<__half><<<grid, 256, smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (const uint32_t*)pos,
            (const uint32_t*)slots, (const uint32_t*)sel, (const uint32_t*)cnt,
            (float*)part_o, (float*)part_ml, nh, nkv, hd, max_ctx, k, cr, scale);
    }
    return pd_launch_status();
}

// Merge the split partials: per (row, head), out = sum_s acc_s e^(m_s-M) /
// sum_s l_s e^(m_s-M). grid (rows, nh), one thread per dim.
__global__ void pd_q4x_qsa_combine_kernel(const float* __restrict__ part_o,
                                          const float* __restrict__ part_ml,
                                          float* __restrict__ out, uint32_t nh, uint32_t nkv,
                                          uint32_t hd, uint32_t ns) {
    const uint32_t r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const uint32_t G = nh / nkv, kvh = h / G, g = h % G;
    const size_t base = ((size_t)r * nkv + kvh) * ns;
    float M = -3.0e38f;
    for (uint32_t s = 0; s < ns; ++s) M = fmaxf(M, part_ml[((base + s) * G + g) * 2u]);
    float L = 0.0f, a = 0.0f;
    for (uint32_t s = 0; s < ns; ++s) {
        const size_t e = (base + s) * G + g;
        const float w = __expf(part_ml[e * 2u] - M);
        L += part_ml[e * 2u + 1u] * w;
        a += part_o[e * hd + d] * w;
    }
    out[((size_t)r * nh + h) * hd + d] = a / L;
}

PD_EXPORT
int pd_q4x_qsa_combine(const void* part_o, const void* part_ml, void* out, uint32_t rows,
                       uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t splits, void* stream) {
    if (rows == 0) return 0;
    if (nkv == 0 || nh % nkv != 0 || hd == 0 || hd > 1024 || splits == 0) return -1;
    pd_q4x_qsa_combine_kernel<<<dim3(rows, nh), hd, 0, (cudaStream_t)stream>>>(
        (const float*)part_o, (const float*)part_ml, (float*)out, nh, nkv, hd, splits);
    return pd_launch_status();
}
