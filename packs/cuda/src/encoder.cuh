// Text-encoder seams: ModernBERT (the encoder under the Laya decision model)
// and the Laya decision head around it.
//
// The pass these serve, rows packed back to back across every question of
// every request in it (see attn/varlen.cuh for the layout):
//
//   ids -> pd_enc_embed_ln -> x (f32 residual) + x16      [rows, d]
//   28 x { qkv GEMM (618) -> pd_enc_attn_h (rope, window) -> o GEMM
//          -> seam (622: x += o; n16 = mlp_norm(x))
//          -> Wi GEMM with the GEGLU landing (672) -> Wo GEMM
//          -> seam (622: x += y; n16 = next attn_norm(x)) }
//   pd_laya_head_entry: X = final_norm(x) + type_emb[q type]; n16 = norm1(X)
//   2 x { in_proj GEMM -> pd_enc_attn_h (in_proj_bias, no rope, full)
//         -> out_proj GEMM -> seam (622 + bias: X += o + b; n16 = norm2(X))
//         -> linear1 GEMM with the bias+ReLU landing (671) -> linear2 GEMM
//         -> seam (622 + bias: X += y + b; n16 = next norm1 / scorer norm) }
//      the LAST layer runs its attention over every row (keys need them all)
//      and everything after it over only the rows anything reads: the option
//      markers and each question's [CLS] (pd_gather_rows)
//   scorer: norm (the last seam) -> GEMM with bias+GELU (624) -> pd_laya_rowdot
//   pd_laya_act_head: the act/escalate head on [CLS] + the option distribution
//
// Everything here is row-local or question-local and walks a fixed order, so a
// row's result does not depend on what else shares its pass.
//
// Needs asr/whisper.cuh (PD_WLN_RU, pd_wln_block_sum) and gemm/f32_qkv.cuh
// (pd_launch_status); f16_dense.cuh's exact GELU.

// Two-pass mean / inverse deviation of a staged row, as pd_whisper_ln_staged
// computes them (same terms, same tree, the dead-slot guard on the
// difference), into caller-owned warp slots so a kernel can take two norms
// back to back without a barrier race on shared partials.
__device__ __forceinline__ void pd_enc_row_stats(const float (&xs)[PD_WLN_RU], uint32_t n,
                                                 float eps, float* wm, float* wv,
                                                 float& mean, float& inv) {
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) acc += xs[u];
    mean = pd_wln_block_sum(acc, wm) / (float)n;
    float vacc = 0.0f;
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        const float dd = (tid + u * nth) < n ? (xs[u] - mean) : 0.0f;
        vacc += dd * dd;
    }
    inv = rsqrtf(pd_wln_block_sum(vacc, wv) / (float)n + eps);
}

// 674: token embedding gather + the embedding LayerNorm (ModernBERT's
// `embeddings.norm`, no bias): x = LN(E[id]) at f32 - the residual stream -
// and the same row at f16, which layer 0's qkv GEMM reads directly (its
// attn_norm is Identity). `b` may be null. d <= 2048.
__global__ void pd_enc_embed_ln_kernel(const __half* __restrict__ emb,
                                       const uint32_t* __restrict__ ids,
                                       const float* __restrict__ w,
                                       const float* __restrict__ b, float* __restrict__ x,
                                       __half* __restrict__ x16, uint32_t n, float eps) {
    __shared__ float wm[32], wv[32];
    const uint32_t row = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const __half* er = emb + (size_t)ids[row] * n;
    float xs[PD_WLN_RU];
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        const uint32_t i = tid + u * nth;
        xs[u] = i < n ? __half2float(er[i]) : 0.0f;
    }
    float mean, inv;
    pd_enc_row_stats(xs, n, eps, wm, wv, mean, inv);
    float* xr = x + (size_t)row * n;
    __half* hr = x16 + (size_t)row * n;
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        const uint32_t i = tid + u * nth;
        if (i < n) {
            const float y = (xs[u] - mean) * inv * w[i] + (b != nullptr ? b[i] : 0.0f);
            xr[i] = y;
            hr[i] = __float2half(y);
        }
    }
}

PD_EXPORT
int pd_enc_embed_ln(const void* emb, const void* ids, const void* w, const void* b, void* x,
                    void* x16, uint32_t rows, uint32_t n, float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    if (n > 256u * PD_WLN_RU) return cudaErrorInvalidValue;
    pd_enc_embed_ln_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const __half*)emb, (const uint32_t*)ids, (const float*)w, (const float*)b, (float*)x,
        (__half*)x16, n, eps);
    return pd_launch_status();
}

// 675: the decision head's entry - the encoder's final norm, the question-type
// embedding, and the head's first pre-norm, in one pass over the residual:
//   X = LN(x; fw) + temb[type[row]]   (in place: x becomes the head's stream)
//   n16 = LN(X; w1, b1)
// `fw` has no bias (ModernBERT's final_norm); the head norms do. d <= 2048.
__global__ void pd_laya_head_entry_kernel(float* __restrict__ x, const float* __restrict__ fw,
                                          const float* __restrict__ temb,
                                          const uint32_t* __restrict__ rtype,
                                          const float* __restrict__ w1,
                                          const float* __restrict__ b1,
                                          __half* __restrict__ n16, uint32_t n, float eps) {
    __shared__ float wm0[32], wv0[32], wm1[32], wv1[32];
    const uint32_t row = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    float* xr = x + (size_t)row * n;
    const float* te = temb + (size_t)rtype[row] * n;
    float xs[PD_WLN_RU];
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        const uint32_t i = tid + u * nth;
        xs[u] = i < n ? xr[i] : 0.0f;
    }
    float mean, inv;
    pd_enc_row_stats(xs, n, eps, wm0, wv0, mean, inv);
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        const uint32_t i = tid + u * nth;
        xs[u] = i < n ? (xs[u] - mean) * inv * fw[i] + te[i] : 0.0f;
        if (i < n) xr[i] = xs[u];
    }
    pd_enc_row_stats(xs, n, eps, wm1, wv1, mean, inv);
    __half* hr = n16 + (size_t)row * n;
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        const uint32_t i = tid + u * nth;
        if (i < n) hr[i] = __float2half((xs[u] - mean) * inv * w1[i] + b1[i]);
    }
}

PD_EXPORT
int pd_laya_head_entry(void* x, const void* fw, const void* temb, const void* rtype,
                       const void* w1, const void* b1, void* n16, uint32_t rows, uint32_t n,
                       float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    if (n > 256u * PD_WLN_RU) return cudaErrorInvalidValue;
    pd_laya_head_entry_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)fw, (const float*)temb, (const uint32_t*)rtype,
        (const float*)w1, (const float*)b1, (__half*)n16, n, eps);
    return pd_launch_status();
}

// 676: dst[i] = src[idx[i]], rows of `row_bytes` (a 16-multiple) - the head's
// last layer continuing on only the rows anything reads.
__global__ void pd_gather_rows_kernel(const uint4* __restrict__ src,
                                      const uint32_t* __restrict__ idx,
                                      uint4* __restrict__ dst, uint32_t row_u4) {
    const uint4* s = src + (size_t)idx[blockIdx.x] * row_u4;
    uint4* o = dst + (size_t)blockIdx.x * row_u4;
    for (uint32_t i = threadIdx.x; i < row_u4; i += blockDim.x) o[i] = s[i];
}

PD_EXPORT
int pd_gather_rows(const void* src, const void* idx, void* dst, uint32_t n_idx,
                   uint32_t row_bytes, void* stream) {
    if (n_idx == 0 || row_bytes == 0) return 0;
    if ((row_bytes & 15u) != 0u) return cudaErrorInvalidValue;
    pd_gather_rows_kernel<<<n_idx, 128, 0, (cudaStream_t)stream>>>(
        (const uint4*)src, (const uint32_t*)idx, (uint4*)dst, row_bytes >> 4);
    return pd_launch_status();
}

// 677: the scorer's last Linear(d, 1): out[i] = sum_j g[i][j] * w[j] + b, one
// warp a row, lanes strided and a fixed shuffle tree - the option logit.
__global__ void pd_laya_rowdot_kernel(const __half* __restrict__ g, const float* __restrict__ w,
                                      float b, float* __restrict__ out, uint32_t m, uint32_t n) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= m) return;
    const __half* gr = g + (size_t)row * n;
    float acc = 0.0f;
    for (uint32_t j = lane; j < n; j += 32u) acc += __half2float(gr[j]) * w[j];
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) out[row] = acc + b;
}

PD_EXPORT
int pd_laya_rowdot(const void* g, const void* w, float b, void* out, uint32_t m, uint32_t n,
                   void* stream) {
    if (m == 0 || n == 0) return 0;
    pd_laya_rowdot_kernel<<<(m + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
        (const __half*)g, (const float*)w, b, (float*)out, m, n);
    return pd_launch_status();
}

// 678: the act/escalate head, one block a question. Its input is the head's
// output at the question's [CLS] row beside four features of the question's
// own option distribution (softmax of its logits, NO temperature - the
// forward's view, not the calibrated answer's):
//   [top1, top1 - top2, H(p) / ln(k), k / 255],  k = max(options, 2)
// then Linear(d + 4, hid) -> exact GELU -> Linear(hid, n_act) -> softmax.
// Single-option questions have no second probability; it is 0, as the
// reference's padded top-2 gives. d + 4 <= 4100 floats of shared memory,
// hid <= 1024, n_act <= 8.
#define PD_LAYA_ACT_MAXD 4096u
#define PD_LAYA_ACT_MAXH 1024u
__global__ void __launch_bounds__(256) pd_laya_act_head_kernel(
    const float* __restrict__ logits, const uint32_t* __restrict__ qoff,
    const float* __restrict__ xcls, const __half* __restrict__ w0,
    const float* __restrict__ b0, const __half* __restrict__ w2,
    const float* __restrict__ b2, float* __restrict__ out, uint32_t d, uint32_t hid,
    uint32_t n_act) {
    __shared__ float inb[PD_LAYA_ACT_MAXD + 4u];
    __shared__ float hs[PD_LAYA_ACT_MAXH];
    __shared__ float z[8];
    const uint32_t q = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t nw = blockDim.x >> 5;
    const uint32_t din = d + 4u;
    for (uint32_t i = tid; i < d; i += blockDim.x) inb[i] = xcls[(size_t)q * d + i];
    if (warp == 0) {
        const float* lg = logits + qoff[q];
        const uint32_t k = qoff[q + 1u] - qoff[q];
        float mx = -INFINITY;
        for (uint32_t j = lane; j < k; j += 32u) mx = fmaxf(mx, lg[j]);
        for (uint32_t o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float se = 0.0f;
        for (uint32_t j = lane; j < k; j += 32u) se += expf(lg[j] - mx);
        for (uint32_t o = 16; o > 0; o >>= 1) se += __shfl_xor_sync(0xffffffffu, se, o);
        // top-2 of the multiset and the entropy; probabilities are >= 0, so a
        // 0 start is the padded reference's missing second entry
        float a1 = 0.0f, a2 = 0.0f, ent = 0.0f;
        for (uint32_t j = lane; j < k; j += 32u) {
            const float p = expf(lg[j] - mx) / se;
            if (p > a1) { a2 = a1; a1 = p; } else if (p > a2) { a2 = p; }
            ent -= p * logf(fmaxf(p, 1e-9f));
        }
        for (uint32_t o = 16; o > 0; o >>= 1) {
            const float o1 = __shfl_xor_sync(0xffffffffu, a1, o);
            const float o2 = __shfl_xor_sync(0xffffffffu, a2, o);
            const float n1 = fmaxf(a1, o1);
            a2 = fmaxf(fminf(a1, o1), fmaxf(a2, o2));
            a1 = n1;
            ent += __shfl_xor_sync(0xffffffffu, ent, o);
        }
        if (lane == 0) {
            const float kc = (float)(k < 2u ? 2u : k);
            inb[d] = a1;
            inb[d + 1u] = a1 - a2;
            inb[d + 2u] = ent / logf(kc);
            inb[d + 3u] = kc / 255.0f;
        }
    }
    __syncthreads();
    for (uint32_t j = warp; j < hid; j += nw) {
        const __half* wr = w0 + (size_t)j * din;
        float acc = 0.0f;
        for (uint32_t c = lane; c < din; c += 32u) acc += __half2float(wr[c]) * inb[c];
        for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) hs[j] = pd_f16_gelu_erf(acc + b0[j]);
    }
    __syncthreads();
    for (uint32_t a = warp; a < n_act; a += nw) {
        const __half* wr = w2 + (size_t)a * hid;
        float acc = 0.0f;
        for (uint32_t j = lane; j < hid; j += 32u) acc += __half2float(wr[j]) * hs[j];
        for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) z[a] = acc + b2[a];
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint32_t a = 0; a < n_act; ++a) mx = fmaxf(mx, z[a]);
        float se = 0.0f;
        for (uint32_t a = 0; a < n_act; ++a) se += expf(z[a] - mx);
        for (uint32_t a = 0; a < n_act; ++a) out[(size_t)q * n_act + a] = expf(z[a] - mx) / se;
    }
}

PD_EXPORT
int pd_laya_act_head(const void* logits, const void* qoff, const void* xcls, const void* w0,
                     const void* b0, const void* w2, const void* b2, void* out, uint32_t nq,
                     uint32_t d, uint32_t hid, uint32_t n_act, void* stream) {
    if (nq == 0) return 0;
    if (d > PD_LAYA_ACT_MAXD || hid > PD_LAYA_ACT_MAXH || n_act == 0 || n_act > 8u)
        return cudaErrorInvalidValue;
    pd_laya_act_head_kernel<<<nq, 256, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const uint32_t*)qoff, (const float*)xcls, (const __half*)w0,
        (const float*)b0, (const __half*)w2, (const float*)b2, (float*)out, d, hid, n_act);
    return pd_launch_status();
}
