// Row-exact multi-row twins of the batch-1 Q8_0 GEMVs (slot 598), and the
// K-split Q8_0 GEMV (slot 588) with its multi-token sibling.
//
// Why the twins exist: a speculative verify row must land exactly where a
// decode tick lands on that token, bit for bit - a near-tie greedy token does
// not survive a last-ulp difference, and the batched Q8_0 GEMMs (mt tiles,
// mmq) reduce in their own order. The qwen4_exp planes the decode tick runs
// through pd_q8_0_gemv_repacked (the hyper-connection up plane [320 -> 10240]
// and the shared-expert down plane) had no batch form, so the verify walked
// them one row per launch.
//
// Each twin reads a weight row ONCE and dots it against up to NB tokens, and
// per token runs the batch-1 kernel it mirrors verbatim - the same scale
// reads, the same 16-element chunks, the same shuffle tree and fold - so each
// token's output is the batch-1 call's bit for bit. Two bodies, as the
// batch-1 launcher has two: the warp-per-row form for rows of at most 64
// chunks (deltanet/core.cuh's pd_q8_0_gemv_repacked_warp_kernel) and the
// row-per-block form above that. The first twin was the block body per
// (row, token), one weight read per token: in a 4-row Flash-Next verify the
// hc up plane took 89 us against the decode tick's 17.5.

// token-block width for a batch: the kernels hold NB accumulators a lane, so
// NB is the batch itself up to 8 (a wider NB's idle slots cost registers -
// 5 tokens in an 8-wide block took 42 us against 27 for 4 in a 4-wide one)
__host__ static inline uint32_t pd_q8_nb(uint32_t batch) {
    return batch < 2u ? 2u : batch > 8u ? 8u : batch;
}
// dispatch a launch macro M(NB) over the NB pd_q8_nb elects
#define PD_Q8_NB_SWITCH(nb, M)          \
    switch (nb) {                       \
        case 2u: M(2u); break;          \
        case 3u: M(3u); break;          \
        case 4u: M(4u); break;          \
        case 5u: M(5u); break;          \
        case 6u: M(6u); break;          \
        case 7u: M(7u); break;          \
        default: M(8u); break;          \
    }

// the row-per-block body over NB tokens (grid (out_dim, ceil(batch / NB)))
template <uint32_t NB>
__global__ void pd_q8_0_gemv_rows_block_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    const uint32_t o = blockIdx.x, t0 = blockIdx.y * NB;
    if (o >= out_dim) return;
    const uint32_t nt = batch - t0 < NB ? batch - t0 : NB;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t bl = tid; bl < n_blocks; bl += nth) ssc[bl] = __half2float(srow[bl]);
    PD_PDL_ARM();
    __shared__ float wsum[NB][32];
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;
    float acc[NB];
    #pragma unroll
    for (uint32_t t = 0; t < NB; ++t) acc[t] = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
        #pragma unroll
        for (uint32_t t = 0; t < NB; ++t) {
            if (t < nt) {
                const float* xr = x + (size_t)(t0 + t) * in_dim;
                float4 x0 = *reinterpret_cast<const float4*>(xr + base);
                float4 x1 = *reinterpret_cast<const float4*>(xr + base + 4);
                float4 x2 = *reinterpret_cast<const float4*>(xr + base + 8);
                float4 x3 = *reinterpret_cast<const float4*>(xr + base + 12);
                float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                        + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                        + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                        + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
                acc[t] += ssc[base >> 5] * s;
            }
        }
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    #pragma unroll
    for (uint32_t t = 0; t < NB; ++t) {
        float a = acc[t];
        for (uint32_t s = 16; s > 0; s >>= 1) a += __shfl_down_sync(0xffffffffu, a, s);
        if (lane == 0) wsum[t][warp] = a;
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t t = 0; t < nt; ++t) {
            float v = 0.0f;
            for (uint32_t w = 0; w < nwarps; ++w) v += wsum[t][w];
            if (bias) v += bias[o];
            y[(size_t)(t0 + t) * out_dim + o] = v;
        }
    }
}

// the warp-per-row body over NB tokens (grid (ceil(out_dim / 4), ceil(batch /
// NB)), 128 threads): warp w owns row 4*blockIdx.x + w for every token
template <uint32_t NB>
__global__ void pd_q8_0_gemv_rows_warp_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t o = blockIdx.x * 4u + warp, t0 = blockIdx.y * NB;
    const uint32_t nt = batch - t0 < NB ? batch - t0 : NB;
    const uint32_t nchunks = in_dim >> 4;
    const bool live = o < out_dim;
    const int8_t* row = data + (size_t)(live ? o : 0u) * in_dim;
    const __half* srow = scale + (size_t)(live ? o : 0u) * (in_dim >> 5);
    float sc[4];
    #pragma unroll
    for (uint32_t g = 0; g < 4u; ++g) {
        const uint32_t c = g * 32u + lane;
        sc[g] = live && c < nchunks ? __half2float(srow[(c * 16u) >> 5]) : 0.0f;
    }
    PD_PDL_ARM();
    if (!live) return;   // warp-uniform
    float v[NB];
    #pragma unroll
    for (uint32_t t = 0; t < NB; ++t) v[t] = 0.0f;
    #pragma unroll
    for (uint32_t g = 0; g < 4u; ++g) {
        const uint32_t c = g * 32u + lane;
        float acc[NB];
        #pragma unroll
        for (uint32_t t = 0; t < NB; ++t) acc[t] = 0.0f;
        if (c < nchunks) {
            const uint32_t base = c * 16u;
            int4 wv = *reinterpret_cast<const int4*>(row + base);
            const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
            #pragma unroll
            for (uint32_t t = 0; t < NB; ++t) {
                if (t < nt) {
                    const float* xr = x + (size_t)(t0 + t) * in_dim;
                    float4 x0 = *reinterpret_cast<const float4*>(xr + base);
                    float4 x1 = *reinterpret_cast<const float4*>(xr + base + 4);
                    float4 x2 = *reinterpret_cast<const float4*>(xr + base + 8);
                    float4 x3 = *reinterpret_cast<const float4*>(xr + base + 12);
                    float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                            + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                            + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                            + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
                    acc[t] += sc[g] * s;
                }
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NB; ++t) {
            float a = acc[t];
            for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) a += __shfl_down_sync(0xffffffffu, a, s2);
            v[t] += a;   // lane 0's is the group's fold
        }
    }
    if (lane == 0) {
        for (uint32_t t = 0; t < nt; ++t) {
            float r = v[t];
            if (bias) r += bias[o];
            y[(size_t)(t0 + t) * out_dim + o] = r;
        }
    }
}

PD_EXPORT
int pd_q8_0_gemv_repacked_rows(const void* data, const void* scale, const void* bias,
                               const void* x, void* y, uint32_t in_dim, uint32_t out_dim,
                               uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // the batch-1 launcher's own geometry and election (128 threads; the
    // warp-per-row body for rows of at most 64 chunks unless
    // PADDOCK_Q8_NO_WARPROW) - a different thread count would regroup the fold
    const uint32_t threads = 128;
    const uint32_t nb = pd_q8_nb(batch);
    const uint32_t ny = (batch + nb - 1u) / nb;
    static const bool no_warp = pd_env("PADDOCK_Q8_NO_WARPROW") != nullptr;
    cudaStream_t st = (cudaStream_t)stream;
    if (!no_warp && (in_dim >> 4) <= 64u) {
#define PD_Q8_ROWS_W(NB)                                                                    \
        pd_pdl_go(pd_q8_0_gemv_rows_warp_kernel<NB>, dim3((out_dim + 3u) / 4u, ny), threads, \
                  0u, st, (const int8_t*)data, (const __half*)scale, (const float*)bias,     \
                  (const float*)x, (float*)y, in_dim, out_dim, batch)
        PD_Q8_NB_SWITCH(nb, PD_Q8_ROWS_W)
#undef PD_Q8_ROWS_W
        return pd_launch_status();
    }
    const uint32_t shmem = (in_dim >> 5) * sizeof(float);
#define PD_Q8_ROWS_B(NB)                                                                    \
    pd_pdl_go(pd_q8_0_gemv_rows_block_kernel<NB>, dim3(out_dim, ny), threads, shmem, st,    \
              (const int8_t*)data, (const __half*)scale, (const float*)bias,                \
              (const float*)x, (float*)y, in_dim, out_dim, batch)
    PD_Q8_NB_SWITCH(nb, PD_Q8_ROWS_B)
#undef PD_Q8_ROWS_B
    return pd_launch_status();
}

// ---- K-SPLIT Q8_0 GEMV/GEMM for NARROW-OUT planes (slot 588) --------------
// One block per output row is the right shape until the row count stops
// filling the die: the hyper-connection down plane is [in 10240, out 320], so
// the plain GEMV launches 320 blocks on a 148-SM machine (2 per SM against
// the 12 it can hold) and the batched mt kernel is worse - its 16-row tile
// makes TWENTY blocks. The plane is 3.5 MB and the walk runs 96 of these a
// tick, which is how a bandwidth-shaped kernel ended up at ~175 GB/s.
//
// Here the row's dot is split over `split` blocks of K, each accumulating its
// own chunk; the last block to finish a row folds the partials in ASCENDING
// split order and writes. Deterministic (fixed fold order, counters reset to
// zero so a captured graph replays identically), and the per-chunk math is
// the plain kernel's - only the outer sum is regrouped, the same class the
// f32 split-K matvec already carries.
__global__ void pd_q8_0_gemv_sk_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ partials,
    unsigned int* __restrict__ counters, uint32_t in_dim, uint32_t out_dim,
    uint32_t split) {
    const uint32_t o = blockIdx.x, sp = blockIdx.y, b = blockIdx.z;
    if (o >= out_dim) return;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t n_blocks = in_dim >> 5;
    // 32-aligned chunks: a 16-element thread chunk then lies wholly inside one
    // Q8_0 block, exactly as in the plain kernel, so the scale lookup is the
    // same single shared read.
    const uint32_t cblocks = (n_blocks + split - 1u) / split;
    const uint32_t k0 = sp * cblocks * 32u;
    const uint32_t k1 = min(k0 + cblocks * 32u, in_dim);
    extern __shared__ float ssc[];
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t i = tid; k0 + i * 32u < k1; i += nth)
        ssc[i] = __half2float(srow[(k0 >> 5) + i]);
    PD_PDL_ARM();
    __shared__ float wsum[32];
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;
    const float* xr = x + (size_t)b * in_dim;
    float acc = 0.0f;
    for (uint32_t base = k0 + tid * 16u; base < k1; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
        float4 x0 = *reinterpret_cast<const float4*>(xr + base);
        float4 x1 = *reinterpret_cast<const float4*>(xr + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(xr + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(xr + base + 12);
        float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
        acc += ssc[(base - k0) >> 5] * s;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s2);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        const size_t oi = (size_t)b * out_dim + o;
        partials[oi * split + sp] = v;
        __threadfence();
        const unsigned int prev = atomicAdd(&counters[oi], 1u);
        if (prev == split - 1u) {
            // acquire, then the other blocks' partials from L2 (abi.cuh,
            // PD_LAST_BLOCK_FOLD)
            __threadfence();
            float sum = 0.0f;
            for (uint32_t t = 0; t < split; ++t) sum += __ldcg(&partials[oi * split + t]);
            if (bias) sum += bias[o];
            y[oi] = sum;
            counters[oi] = 0u;   // graph-replay safe: back to the initial state
        }
    }
}

// Multi-token sibling of pd_q8_0_gemv_sk_kernel (batch >= 2: a speculative
// verify's rows, a small prefill chunk). The (row, K chunk) block reads its
// weight span ONCE and dots it against up to NB tokens; the batch-1 grid's
// token axis re-read the span per token, which in a 4-row verify put the hc
// down plane at 36 us against its 19 at batch 1. Per token every operation
// is the batch-1 block's - the chunk walk, the scale lookup, the shuffle
// tree, the serial warp fold, the ascending partial fold by the row's last
// block - so each token lands bit for bit where its own launch lands.
template <uint32_t NB>
__global__ void pd_q8_0_gemv_sk_nb_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ partials,
    unsigned int* __restrict__ counters, uint32_t in_dim, uint32_t out_dim,
    uint32_t split, uint32_t batch) {
    const uint32_t o = blockIdx.x, sp = blockIdx.y, t0 = blockIdx.z * NB;
    if (o >= out_dim) return;
    const uint32_t nt = batch - t0 < NB ? batch - t0 : NB;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t cblocks = (n_blocks + split - 1u) / split;
    const uint32_t k0 = sp * cblocks * 32u;
    const uint32_t k1 = min(k0 + cblocks * 32u, in_dim);
    extern __shared__ float ssc[];
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t i = tid; k0 + i * 32u < k1; i += nth)
        ssc[i] = __half2float(srow[(k0 >> 5) + i]);
    PD_PDL_ARM();
    __shared__ float wsum[NB][32];
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;
    float acc[NB];
    #pragma unroll
    for (uint32_t t = 0; t < NB; ++t) acc[t] = 0.0f;
    for (uint32_t base = k0 + tid * 16u; base < k1; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
        #pragma unroll
        for (uint32_t t = 0; t < NB; ++t) {
            if (t < nt) {
                const float* xr = x + (size_t)(t0 + t) * in_dim;
                float4 x0 = *reinterpret_cast<const float4*>(xr + base);
                float4 x1 = *reinterpret_cast<const float4*>(xr + base + 4);
                float4 x2 = *reinterpret_cast<const float4*>(xr + base + 8);
                float4 x3 = *reinterpret_cast<const float4*>(xr + base + 12);
                float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                        + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                        + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                        + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
                acc[t] += ssc[(base - k0) >> 5] * s;
            }
        }
    }
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    #pragma unroll
    for (uint32_t t = 0; t < NB; ++t) {
        float a = acc[t];
        for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) a += __shfl_down_sync(0xffffffffu, a, s2);
        if (lane == 0) wsum[t][warp] = a;
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t t = 0; t < nt; ++t) {
            float v = 0.0f;
            for (uint32_t w = 0; w < nwarps; ++w) v += wsum[t][w];
            partials[((size_t)(t0 + t) * out_dim + o) * split + sp] = v;
        }
        __threadfence();
        for (uint32_t t = 0; t < nt; ++t) {
            const size_t oi = (size_t)(t0 + t) * out_dim + o;
            const unsigned int prev = atomicAdd(&counters[oi], 1u);
            if (prev == split - 1u) {
                __threadfence();   // acquire (PD_LAST_BLOCK_FOLD)
                float sum = 0.0f;
                for (uint32_t q = 0; q < split; ++q) sum += __ldcg(&partials[oi * split + q]);
                if (bias) sum += bias[o];
                y[oi] = sum;
                counters[oi] = 0u;   // graph-replay safe: back to the initial state
            }
        }
    }
}

PD_EXPORT
int pd_q8_0_gemv_sk(const void* data, const void* scale, const void* bias,
                    const void* x, void* y, void* partials, void* counters,
                    uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                    uint32_t split, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (split < 2u || split > 32u) return cudaErrorInvalidValue;
    if (in_dim & 31u) return cudaErrorInvalidValue;
    const uint32_t n_blocks = in_dim >> 5;
    if (split > n_blocks) return cudaErrorInvalidValue;
    const uint32_t threads = 128;
    const uint32_t cblocks = (n_blocks + split - 1u) / split;
    const uint32_t shmem = cblocks * (uint32_t)sizeof(float);
    if (batch == 1u) {
        pd_pdl_go(pd_q8_0_gemv_sk_kernel, dim3(out_dim, split, 1u), threads, shmem,
            (cudaStream_t)stream, (const int8_t*)data, (const __half*)scale,
            (const float*)bias, (const float*)x, (float*)y, (float*)partials,
            (unsigned int*)counters, in_dim, out_dim, split);
        return pd_launch_status();
    }
    // two or more tokens: the multi-token sibling, one weight read per span
    const uint32_t nb = pd_q8_nb(batch);
    const dim3 grid(out_dim, split, (batch + nb - 1u) / nb);
#define PD_Q8_SK_NB(NB)                                                                  \
    pd_pdl_go(pd_q8_0_gemv_sk_nb_kernel<NB>, grid, threads, shmem, (cudaStream_t)stream,  \
        (const int8_t*)data, (const __half*)scale, (const float*)bias, (const float*)x,  \
        (float*)y, (float*)partials, (unsigned int*)counters, in_dim, out_dim, split, batch)
    PD_Q8_NB_SWITCH(nb, PD_Q8_SK_NB)
#undef PD_Q8_SK_NB
    return pd_launch_status();
}

#undef PD_Q8_NB_SWITCH
