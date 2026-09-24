// diffusion/canvas.cuh - block-diffusion canvas ops (DiffusionGemma): the
// transposed bf16 embedding plane the self-conditioning matmul runs on, the
// per-row canvas sampler (softmax stats + entropy + Gumbel-max + argmax over
// the vocab, probs left in place), the per-canvas entropy-bounded accept /
// re-noise / stability step, and a label-id column gather for structured
// reads.
// Textually-included segment of the single pack translation unit; needs
// dit.cuh's `pd_dit_philox_round` and quant/kquant_w4a8.cuh's
// `pd_kq_win_unpack` (both included before this file).
//
// Reference for the algorithm: transformers' generation_diffusion_gemma.py
// (EntropyBoundSampler, LinearTemperatureScheduleLogitsProcessor,
// StableAndConfidentStoppingCriteria) - read, not copied. Everything here is
// plain CUDA.

// ---- E^T as a bf16 [embd][vocab] plane, straight from the Q8_0 rows -----------
//
// The self-conditioning signal is `softmax(logits) @ E` - an NN product with
// the embedding as the K-major operand, which none of the NT weight kernels
// express. `bf16_gemm` wants a `[out][in]` row-major weight, i.e. E^T with
// out = embd, in = vocab. This kernel dequants the Q8_0 rows (34-byte blocks
// of 32 along embd) and writes them transposed, through a 32x32 smem tile so
// both the block reads (along embd) and the plane writes (along vocab) are
// coalesced. One-time load-side pass over the 738 MB plane.
__global__ void __launch_bounds__(256) pd_q8_embed_transpose_bf16_kernel(
    const unsigned char* __restrict__ q8, __nv_bfloat16* __restrict__ dst,
    uint32_t vocab, uint32_t embd) {
    __shared__ float tile[32][33];
    const uint32_t v0 = blockIdx.y * 32u, e0 = blockIdx.x * 32u;  // e0 % 32 == 0
    const uint32_t tx = threadIdx.x & 31u, ty = threadIdx.x >> 5;  // 8 warps
    // read: 32 vocab rows x one 32-wide Q8_0 block each (the block at e0)
    for (uint32_t r = ty; r < 32u; r += 8u) {
        const uint32_t v = v0 + r;
        float val = 0.f;
        if (v < vocab) {
            const unsigned char* blk = q8 + ((size_t)v * (embd / 32u) + (e0 / 32u)) * 34u;
            __half h;
            memcpy(&h, blk, sizeof(h));
            val = (float)((int8_t)blk[2 + tx]) * __half2float(h);
        }
        tile[r][tx] = val;
    }
    __syncthreads();
    // write: 32 embd rows x 32 consecutive vocab columns
    for (uint32_t r = ty; r < 32u; r += 8u) {
        const uint32_t e = e0 + r, v = v0 + tx;
        if (v < vocab) dst[(size_t)e * vocab + v] = __float2bfloat16(tile[tx][r]);
    }
}

PD_EXPORT
int pd_q8_embed_transpose_bf16(const void* q8, void* dst, uint32_t vocab, uint32_t embd,
                               void* stream) {
    if (vocab == 0u || embd == 0u || (embd % 32u) != 0u) return (int)cudaErrorInvalidValue;
    dim3 grid(embd / 32u, (vocab + 31u) / 32u);
    pd_q8_embed_transpose_bf16_kernel<<<grid, 256u, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)q8, (__nv_bfloat16*)dst, vocab, embd);
    return pd_launch_status();
}

// The same E^T plane off a REPACKED k-quant embedding (the Q4_K_M file's Q6_K
// token_embd, or any type the k-quant streams hold): the family keeps no raw
// copy of a k-quant table - its row gathers and the tied head read the
// repacked plane - so the transpose reads that plane too, through the one
// 16-weight window unpack every serving lane decodes with (a weight is
// f * q + g there, the k-quant family's mu term included). A block owns 32
// vocab rows x one 256-wide super-block column: 16 rows per pass, a warp
// decoding two rows' 16 windows into a 32 x 257 f32 tile, then the 256
// embd rows of the tile go out as 32-wide bf16 runs along vocab. Whole
// super-blocks per row (embd % 256), which is what the gather kernel
// requires of the plane as well.
__global__ void __launch_bounds__(256) pd_kq_embed_transpose_bf16_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ dst, uint32_t vocab, uint32_t embd, uint32_t dtype) {
    __shared__ float tile[32][257];
    const uint32_t v0 = blockIdx.y * 32u, s = blockIdx.x;  // super-block column s
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;  // 8 warps
    const uint32_t n_super = embd >> 8u;
    const uint32_t datab = pd_kq_datab(dtype), scb = pd_kq_scb(dtype);
    // read: lane = (row within the warp's pair, window); two passes of 16 rows
    for (uint32_t pass = 0; pass < 2u; ++pass) {
        const uint32_t r = pass * 16u + warp * 2u + (lane >> 4u);
        const uint32_t w = lane & 15u;
        const uint32_t v = v0 + r;
        float vals[16];
        if (v < vocab) {
            const size_t sbi = (size_t)v * n_super + s;
            int wq[4];
            float f, g;
            pd_kq_win_unpack(dtype, data + sbi * datab, scales + sbi * scb, w, wq, &f, &g);
            #pragma unroll
            for (uint32_t j = 0; j < 16u; ++j)
                vals[j] = f * (float)((int8_t)((wq[j >> 2u] >> (8u * (j & 3u))) & 0xffu)) + g;
        } else {
            #pragma unroll
            for (uint32_t j = 0; j < 16u; ++j) vals[j] = 0.f;
        }
        #pragma unroll
        for (uint32_t j = 0; j < 16u; ++j) tile[r][w * 16u + j] = vals[j];
    }
    __syncthreads();
    // write: 256 embd rows x 32 consecutive vocab columns
    const uint32_t e0 = s * 256u;
    for (uint32_t e = warp; e < 256u; e += 8u) {
        const uint32_t v = v0 + lane;
        if (v < vocab) dst[(size_t)(e0 + e) * vocab + v] = __float2bfloat16(tile[lane][e]);
    }
}

PD_EXPORT
int pd_kq_embed_transpose_bf16(const void* data, const void* scales, void* dst,
                               uint32_t vocab, uint32_t embd, uint32_t dtype, void* stream) {
    if (vocab == 0u || embd == 0u || (embd % 256u) != 0u) return (int)cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype) && !pd_kq_valid_iq(dtype)) return (int)cudaErrorInvalidValue;
    dim3 grid(embd / 256u, (vocab + 31u) / 32u);
    pd_kq_embed_transpose_bf16_kernel<<<grid, 256u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (__nv_bfloat16*)dst, vocab, embd, dtype);
    return pd_launch_status();
}

// ---- Philox uniform: one draw per (offset, index) counter -------------------
//
// Same Philox4x32-10 layout as dit.cuh's randn (key = seed lo/hi, counter =
// {offset, 0, index, 0}), taking the first 32-bit word as u = c0 * 2^-32 +
// 2^-33 in (0, 1). Seeded canvases are therefore reproducible across engines
// that adopt the layout, and a (slot, step) offset keeps rows independent.
__device__ __forceinline__ float pd_canvas_uniform(uint32_t seed_lo, uint32_t seed_hi,
                                                   uint32_t offset, uint32_t index) {
    uint32_t c0 = offset, c1 = 0u, c2 = index, c3 = 0u;
    uint32_t k0 = seed_lo, k1 = seed_hi;
#pragma unroll
    for (int r = 0; r < 9; ++r) {
        pd_dit_philox_round(c0, c1, c2, c3, k0, k1);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    pd_dit_philox_round(c0, c1, c2, c3, k0, k1);
    const float inv32 = 2.3283064e-10f;
    return (float)c0 * inv32 + inv32 / 2.0f;
}

// ---- per-row canvas sampler ------------------------------------------------
//
// One block per canvas row over the whole vocab. In: softcapped logits l
// (f32, [rows][vocab]) and the row's inverse temperature. Let z = l * inv_t.
//   sample  = argmax_i (z_i + g_i), g_i = -log(-log u_i)   (Gumbel-max, i.e.
//             one categorical draw from softmax(z) - the same distribution
//             torch.multinomial(softmax(z)) draws from)
//   argmax  = argmax_i l_i  (temperature-invariant, ties -> lowest index)
//   entropy = log S1 - S2 / S1  with S1 = sum e^{z-m}, S2 = sum (z-m) e^{z-m}
//             - Categorical(logits=z).entropy(), computed in f32
//   probs   = e^{z-m} / S1 written back IN PLACE of the logits: this is the
//             `softmax(processed_logits, dtype=float32)` the next step's
//             self-conditioning multiplies with the embedding.
// inv_t = 0 is the one-hot limit (temperature -> 0): sample = argmax,
// entropy = 0, probs = one-hot - the limit of the formulas, not a division.
// A NEGATIVE inv_t is the GREEDY DRAW at |inv_t|: the sample is the argmax
// while the entropy and the probs are those of the |inv_t|-scaled
// distribution. That is what "temperature 0" means for a canvas - the
// entropy-bounded accept step must keep seeing the schedule's entropies, or
// every position reads as certain on the first step and the whole noise
// canvas is accepted as the answer (measured: garbage in one step).
#define PD_CV_TPB 1024u
__global__ void __launch_bounds__(PD_CV_TPB) pd_canvas_sample_kernel(
    float* __restrict__ logits, const float* __restrict__ inv_t, uint32_t seed_lo,
    uint32_t seed_hi, uint32_t offset, unsigned int* __restrict__ out_sample,
    unsigned int* __restrict__ out_argmax, float* __restrict__ out_entropy, uint32_t vocab) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    constexpr uint32_t NW = PD_CV_TPB / 32u;
    float* x = logits + (size_t)row * vocab;
    const float it_raw = inv_t[row];
    const bool greedy = it_raw < 0.f;
    const float it = fabsf(it_raw);
    __shared__ float s_f[NW];
    __shared__ unsigned int s_i[NW];
    __shared__ float s_g[NW];
    __shared__ unsigned int s_gi[NW];
    __shared__ float s_bc[2];

    // pass A: raw argmax, max z, and the Gumbel-max sample (greedy: skipped)
    float bv = -3.402823466e+38f, gv = -3.402823466e+38f;
    uint32_t bi = 0u, gi = 0u;
    const uint32_t base = row * vocab;  // <= 256 * 262144 fits u32 by contract
    for (uint32_t i = tid; i < vocab; i += PD_CV_TPB) {
        const float v = x[i];
        if (v > bv) { bv = v; bi = i; }
        if (it != 0.f && !greedy) {
            const float u = pd_canvas_uniform(seed_lo, seed_hi, offset, base + i);
            const float g = -logf(-logf(u));
            const float s = v * it + g;
            if (s > gv) { gv = s; gi = i; }
        }
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        const float og = __shfl_down_sync(0xffffffffu, gv, off);
        const uint32_t ogi = __shfl_down_sync(0xffffffffu, gi, off);
        if (og > gv || (og == gv && ogi < gi)) { gv = og; gi = ogi; }
    }
    if (lane == 0) { s_f[warp] = bv; s_i[warp] = bi; s_g[warp] = gv; s_gi[warp] = gi; }
    __syncthreads();
    if (tid == 0) {
        for (uint32_t w = 1; w < NW; ++w) {
            if (s_f[w] > bv || (s_f[w] == bv && s_i[w] < bi)) { bv = s_f[w]; bi = s_i[w]; }
            if (s_g[w] > gv || (s_g[w] == gv && s_gi[w] < gi)) { gv = s_g[w]; gi = s_gi[w]; }
        }
        out_argmax[row] = bi;
        out_sample[row] = (it != 0.f && !greedy) ? gi : bi;
        s_bc[0] = bv;  // max logit; max z = bv * it (it >= 0)
    }
    __syncthreads();
    const float maxl = s_bc[0];
    if (it == 0.f) {
        // deterministic limit: one-hot at the argmax, entropy 0
        for (uint32_t i = tid; i < vocab; i += PD_CV_TPB) x[i] = (i == bi) ? 1.f : 0.f;
        if (tid == 0) out_entropy[row] = 0.f;
        return;
    }
    const float m = maxl * it;
    // pass B: S1 = sum e^{z-m}, S2 = sum (z-m) e^{z-m}
    float s1 = 0.f, s2 = 0.f;
    for (uint32_t i = tid; i < vocab; i += PD_CV_TPB) {
        const float d = x[i] * it - m;
        const float e = expf(d);
        s1 += e;
        s2 += d * e;
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        s1 += __shfl_down_sync(0xffffffffu, s1, off);
        s2 += __shfl_down_sync(0xffffffffu, s2, off);
    }
    if (lane == 0) { s_f[warp] = s1; s_g[warp] = s2; }
    __syncthreads();
    if (tid == 0) {
        for (uint32_t w = 1; w < NW; ++w) { s1 += s_f[w]; s2 += s_g[w]; }
        out_entropy[row] = logf(s1) - s2 / s1;
        s_bc[1] = 1.f / s1;
    }
    __syncthreads();
    const float inv_s1 = s_bc[1];
    // pass C: probs in place
    for (uint32_t i = tid; i < vocab; i += PD_CV_TPB) x[i] = expf(x[i] * it - m) * inv_s1;
}

PD_EXPORT
int pd_canvas_sample(void* logits, const void* inv_t, uint32_t seed_lo, uint32_t seed_hi,
                     uint32_t offset, void* out_sample, void* out_argmax, void* out_entropy,
                     uint32_t rows, uint32_t vocab, void* stream) {
    if (rows == 0u || vocab == 0u) return 0;
    // the flat Philox index rides a u32: rows * vocab must fit
    if ((uint64_t)rows * (uint64_t)vocab > 0xFFFFFFFFull) return (int)cudaErrorInvalidValue;
    pd_canvas_sample_kernel<<<rows, PD_CV_TPB, 0, (cudaStream_t)stream>>>(
        (float*)logits, (const float*)inv_t, seed_lo, seed_hi, offset,
        (unsigned int*)out_sample, (unsigned int*)out_argmax, (float*)out_entropy, vocab);
    return pd_launch_status();
}

// ---- per-canvas accept / re-noise / stability ------------------------------
//
// One block per canvas of w <= 1024 positions (the model's 256; narrower
// reads too). Inputs are this step's per-position entropy, sampled token and
// argmax token. The step, exactly as the reference does it:
//   accepted = the entropy-sorted prefix where cumsum - e <= bound
//              (the joint mutual-information bound: the accepted tokens are
//              approximately independent), RECOMPUTED every step - nothing
//              accumulates across steps;
//   canvas[p] = accepted ? sampled[p] : a fresh uniform random id
//              (the next step's input; pinned positions are the caller's
//              business - it overwrites them after this kernel);
//   stable    = every history entry equals this step's argmax canvas
//              (history holds the last `stab` argmax canvases, 0xFFFFFFFF
//              before they exist; stab = 0 means always stable), then the
//              oldest entry is replaced by this one;
//   mean_ent  = mean over positions of the entropy;
//   converged = stable && mean_ent < conf.
// status[4] = {converged, n_accepted, f32 bits of mean_ent, stable}.
#define PD_CA_MAX 1024u
__global__ void __launch_bounds__(PD_CA_MAX) pd_canvas_accept_kernel(
    const float* __restrict__ entropy, const unsigned int* __restrict__ sampled,
    const unsigned int* __restrict__ argmax, unsigned int* __restrict__ canvas,
    unsigned int* __restrict__ hist, unsigned int* __restrict__ status, uint32_t w,
    uint32_t vocab, uint32_t stab, uint32_t step, float bound, float conf, uint32_t seed_lo,
    uint32_t seed_hi, uint32_t offset) {
    const uint32_t tid = threadIdx.x;
    const uint32_t p2 = blockDim.x;  // next power of two >= w, launched by the host
    __shared__ float s_e[PD_CA_MAX];
    __shared__ unsigned int s_ix[PD_CA_MAX];
    __shared__ float s_c[PD_CA_MAX];
    __shared__ unsigned int s_acc[PD_CA_MAX];
    __shared__ float s_red[PD_CA_MAX / 32u];
    __shared__ unsigned int s_and[PD_CA_MAX / 32u];

    const bool live = tid < w;
    const float e = live ? entropy[tid] : 3.402823466e+38f;  // padding sorts last
    s_e[tid] = e;
    s_ix[tid] = tid;
    __syncthreads();
    // bitonic sort ascending on (entropy, index) - p2 elements, p2 threads
    for (uint32_t k = 2; k <= p2; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            const uint32_t ixj = tid ^ j;
            if (ixj > tid) {
                const bool up = (tid & k) == 0u;
                const float a = s_e[tid], b = s_e[ixj];
                const unsigned int ia = s_ix[tid], ib = s_ix[ixj];
                const bool a_gt_b = a > b || (a == b && ia > ib);
                if (a_gt_b == up) {
                    s_e[tid] = b; s_e[ixj] = a;
                    s_ix[tid] = ib; s_ix[ixj] = ia;
                }
            }
            __syncthreads();
        }
    }
    // inclusive prefix sum of the sorted entropies (Hillis-Steele in smem)
    s_c[tid] = s_e[tid];
    __syncthreads();
    for (uint32_t d = 1; d < p2; d <<= 1) {
        const float v = tid >= d ? s_c[tid - d] : 0.f;
        __syncthreads();
        s_c[tid] += v;
        __syncthreads();
    }
    // rank tid accepted if cumsum - e <= bound (padding never is: +inf)
    const bool acc_rank = (tid < w) && (s_c[tid] - s_e[tid] <= bound);
    s_acc[s_ix[tid]] = acc_rank ? 1u : 0u;  // scatter back to positions
    __syncthreads();
    // per position: the new canvas, and the reductions
    uint32_t n_acc = 0u, all_eq = 1u;
    float esum = 0.f;
    if (live) {
        const uint32_t a = s_acc[tid];
        n_acc = a;
        esum = entropy[tid];
        if (a) {
            canvas[tid] = sampled[tid];
        } else {
            const float u = pd_canvas_uniform(seed_lo, seed_hi, offset, tid);
            uint32_t id = (uint32_t)(u * (float)vocab);
            canvas[tid] = id < vocab ? id : vocab - 1u;
        }
        const uint32_t am = argmax[tid];
        for (uint32_t h = 0; h < stab; ++h)
            if (hist[(size_t)h * w + tid] != am) all_eq = 0u;
    }
    __syncthreads();  // history reads done before the overwrite below
    if (live && stab > 0u) hist[(size_t)(step % stab) * w + tid] = argmax[tid];
    // block reductions: sum of entropy, count accepted, AND of stability
    const uint32_t lane = tid & 31u, warp = tid >> 5;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        esum += __shfl_down_sync(0xffffffffu, esum, off);
        n_acc += __shfl_down_sync(0xffffffffu, n_acc, off);
        all_eq &= __shfl_down_sync(0xffffffffu, all_eq, off);
    }
    if (lane == 0) { s_red[warp] = esum; s_and[warp] = all_eq; s_acc[warp] = n_acc; }
    __syncthreads();
    if (tid == 0) {
        const uint32_t nw = p2 / 32u;
        for (uint32_t x = 1; x < nw; ++x) { esum += s_red[x]; all_eq &= s_and[x]; n_acc += s_acc[x]; }
        const float mean = esum / (float)w;
        const uint32_t stable = all_eq;
        const uint32_t conv = (stable && mean < conf) ? 1u : 0u;
        status[0] = conv;
        status[1] = n_acc;
        status[2] = __float_as_uint(mean);
        status[3] = stable;
    }
}

PD_EXPORT
int pd_canvas_accept(const void* entropy, const void* sampled, const void* argmax, void* canvas,
                     void* hist, void* status, uint32_t w, uint32_t vocab, uint32_t stab,
                     uint32_t step, float bound, float conf, uint32_t seed_lo, uint32_t seed_hi,
                     uint32_t offset, void* stream) {
    if (w == 0u || w > PD_CA_MAX || vocab == 0u) return (int)cudaErrorInvalidValue;
    uint32_t p2 = 32u;
    while (p2 < w) p2 <<= 1;
    pd_canvas_accept_kernel<<<1, p2, 0, (cudaStream_t)stream>>>(
        (const float*)entropy, (const unsigned int*)sampled, (const unsigned int*)argmax,
        (unsigned int*)canvas, (unsigned int*)hist, (unsigned int*)status, w, vocab, stab, step,
        bound, conf, seed_lo, seed_hi, offset);
    return pd_launch_status();
}

// ---- label-id column gather: out[r][j] = src[r][ids[j]] ---------------------
//
// The structured read: k label ids (<= a few hundred) out of each row's
// normalized plane. Plain gather, one thread per (row, j).
__global__ void pd_gather_cols_kernel(const float* __restrict__ src,
                                      const unsigned int* __restrict__ ids,
                                      float* __restrict__ out, uint32_t rows, uint32_t n,
                                      uint32_t k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * k) return;
    const uint32_t r = i / k, j = i - r * k;
    const uint32_t id = ids[j];
    out[i] = id < n ? src[(size_t)r * n + id] : 0.f;
}

PD_EXPORT
int pd_gather_cols(const void* src, const void* ids, void* out, uint32_t rows, uint32_t n,
                   uint32_t k, void* stream) {
    const uint32_t total = rows * k;
    if (total == 0u) return 0;
    pd_gather_cols_kernel<<<(total + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)src, (const unsigned int*)ids, (float*)out, rows, n, k);
    return pd_launch_status();
}
