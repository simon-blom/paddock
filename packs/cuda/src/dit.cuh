// dit.cuh - diffusion-transformer glue for the image-generation lane
// (Qwen-Image-2.1 first). Textually-included segment of the single pack
// translation unit. Not standalone-compilable: include order is defined by
// ../pack.cu.
//
// Everything heavy in a DiT step is an existing lane: the int8 / W4A8 prefill
// GEMMs eat the block GEMMs, the vision tower's dense f16-mma attention eats
// the target-block attention, layernorm carries the (1 + scale) modulation as
// its weight vector. What is left is the model's own small glue, which is
// what lives here - plain CUDA, f32 planes, abi.cuh helpers only.
//
// The pack-independent contract every kernel here keeps: run-to-run bit
// stability (fixed-order reductions, no atomics into f32), and no
// materialised score matrix anywhere - the one place a DiT wants one (the
// VAE mid block's single-head attention over every latent pixel) goes
// through the f16 GEMM twice with a row softmax in between, which is what
// pd_dit_softmax_rows is for.

// ---- 3-axis rotary embedding, complex form ---------------------------------
//
// diffusers `QwenImage21Rope` + `apply_rotary_emb_qwen(use_real=False)`: the
// head is split into three axis blocks of (d0, d1, d2) dims - frame, height,
// width for this model, 16 + 56 + 56 = 128 - and every ADJACENT pair (2i,
// 2i+1) is one complex number rotated by its axis's position. Pair k of an
// axis of dim d spins at pos * theta^(-2k/d), theta 10000. Positions are
// per-token int32 triples: text tokens advance a shared counter on all three
// axes, an image block freezes the frame axis at that counter and lays its
// tokens out on a height/width grid CENTRED ON ZERO, so the values are
// signed and the frequency table is built the way torch builds it
// (`1 / theta^(arange(0, d, 2) / d)` in f32, then `outer(index, inv)`, then
// polar -> cos/sin): pos is converted to f32 and multiplied, cos/sin in f32.
//
// In place over [rows][n_heads][hd] f32. One thread per (row, head, pair).
__global__ void pd_dit_rope_kernel(float* __restrict__ x, const int32_t* __restrict__ pos,
                                   uint32_t rows, uint32_t n_heads, uint32_t hd,
                                   uint32_t d0, uint32_t d1, uint32_t d2, float theta) {
    const uint32_t pairs = hd >> 1;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)rows * n_heads * pairs;
    if (i >= total) return;
    const uint32_t p = (uint32_t)(i % pairs);
    const size_t rh = i / pairs;                 // row * n_heads + head
    const uint32_t row = (uint32_t)(rh / n_heads);
    // which axis this pair belongs to, and its index inside the axis
    uint32_t axis, k, d;
    const uint32_t p0 = d0 >> 1, p1 = d1 >> 1;
    if (p < p0)            { axis = 0u; k = p;           d = d0; }
    else if (p < p0 + p1)  { axis = 1u; k = p - p0;      d = d1; }
    else                   { axis = 2u; k = p - p0 - p1; d = d2; }
    const float inv = 1.0f / powf(theta, (float)(2u * k) / (float)d);
    const float ang = (float)pos[(size_t)row * 3u + axis] * inv;
    float s, c;
    sincosf(ang, &s, &c);
    float* xp = x + rh * hd + (size_t)p * 2u;
    const float a = xp[0], b = xp[1];
    // (a + ib) * (c + is)
    xp[0] = a * c - b * s;
    xp[1] = a * s + b * c;
}

PD_EXPORT
int pd_dit_rope(void* x, const void* pos, uint32_t rows, uint32_t n_heads, uint32_t hd,
                uint32_t d0, uint32_t d1, uint32_t d2, float theta, void* stream) {
    if (rows == 0u) return 0;
    if ((hd & 1u) || d0 + d1 + d2 != hd || (d0 & 1u) || (d1 & 1u) || (d2 & 1u))
        return cudaErrorInvalidValue;
    const size_t total = (size_t)rows * n_heads * (hd >> 1);
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_dit_rope_kernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>(
        (float*)x, (const int32_t*)pos, rows, n_heads, hd, d0, d1, d2, theta);
    return pd_launch_status();
}

// ---- initial latent noise: torch's CUDA randn, one element per Philox
// counter -------------------------------------------------------------------
//
// torch.randn on a CUDA generator is Philox4x32-10 (key = seed lo/hi,
// counter = {offset, 0, subsequence, 0}) with each element drawn as the FIRST
// Box-Muller output of one counter's 128 bits - when the tensor is small
// enough that every thread of torch's grid yields one element. stable-
// diffusion.cpp's `--rng cuda` is exactly that layout (rng_philox.hpp, the
// sd-webui port), with counter[0] counting randn() calls and counter[2] the
// flat element index, so seed 42 there and seed 42 here draw the same noise.
// (torch itself caps its grid at 8 blocks per SM and folds the tail onto
// rand.y/z/w once a tensor outgrows one element per thread - a 1024^2 latent
// already does on an A6000 - so torch's own noise is not portable across
// GPUs; the reference layout is the one every engine can reproduce.)
//
// Box-Muller as cuRAND writes it: u = x * 2^-32 + 2^-33, v = y * (2pi *
// 2^-32) + (2pi * 2^-33), out = sqrt(-2 ln u) * sin(v). Precise sinf/logf, not
// the fast intrinsics: sd.cpp evaluates this on the host with libm, and the
// precise device forms are the ones that agree with it to an ulp.
//
// The latent is packed [token][channel] (one token per latent pixel, row-
// major over the grid) while torch draws it as [channel][y][x]: element
// (t, c) is flat index c * n_tokens + t of the draw.
__device__ __forceinline__ void pd_dit_philox_round(uint32_t& c0, uint32_t& c1, uint32_t& c2,
                                                    uint32_t& c3, uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0, n1 = lo1, n2 = hi0 ^ c3 ^ k1, n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}

__global__ void pd_dit_philox_randn_kernel(float* __restrict__ out, uint32_t seed_lo,
                                           uint32_t seed_hi, uint32_t offset,
                                           uint32_t n_tokens, uint32_t channels) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)n_tokens * channels;
    if (i >= n) return;
    const uint32_t t = (uint32_t)(i / channels), c = (uint32_t)(i - (size_t)t * channels);
    const uint32_t flat = c * n_tokens + t;
    uint32_t c0 = offset, c1 = 0u, c2 = flat, c3 = 0u;
    uint32_t k0 = seed_lo, k1 = seed_hi;
#pragma unroll
    for (int r = 0; r < 9; ++r) {
        pd_dit_philox_round(c0, c1, c2, c3, k0, k1);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    pd_dit_philox_round(c0, c1, c2, c3, k0, k1);
    const float inv32 = 2.3283064e-10f;
    const float inv32_2pi = 2.3283064e-10f * 6.2831855f;
    const float u = (float)c0 * inv32 + inv32 / 2.0f;
    const float v = (float)c1 * inv32_2pi + inv32_2pi / 2.0f;
    const float s = sqrtf(-2.0f * logf(u));
    out[i] = s * sinf(v);
}

PD_EXPORT
int pd_dit_philox_randn(void* out, uint32_t seed_lo, uint32_t seed_hi, uint32_t offset,
                        uint32_t n_tokens, uint32_t channels, void* stream) {
    const size_t n = (size_t)n_tokens * channels;
    if (n == 0u) return 0;
    const uint32_t blocks = (uint32_t)((n + 255u) / 256u);
    pd_dit_philox_randn_kernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>(
        (float*)out, seed_lo, seed_hi, offset, n_tokens, channels);
    return pd_launch_status();
}

// ---- the block's gated residual: x += g[c] * y ------------------------------
//
// `hidden + tanh(gate) * sublayer_out` with the gate a per-CHANNEL vector
// shared by every token of the forward (one modulation row per forward, see
// the engine: the prefix modulates from t = 0, the target from the sampled
// t, and they are separate forwards). The DINOv3 LayerScale fusion is the
// same product but writes f16 for its own next GEMM; the DiT's next consumer
// is the int8 quantizer, which reads f32, so this stays f32 and unfused.
__global__ void pd_dit_gated_add_kernel(float* __restrict__ x, const float* __restrict__ y,
                                        const float* __restrict__ g, uint32_t n, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t c = (uint32_t)(i % n);
    x[i] = fmaf(g[c], y[i], x[i]);
}

PD_EXPORT
int pd_dit_gated_add(void* x, const void* y, const void* g, uint32_t rows, uint32_t n,
                     void* stream) {
    const size_t total = (size_t)rows * n;
    if (total == 0u) return 0;
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_dit_gated_add_kernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)y, (const float*)g, n, total);
    return pd_launch_status();
}

// ---- SiLU in place ----------------------------------------------------------
// The timestep MLP and the VAE want a bare silu; every existing silu in the
// pack is fused into a GLU. x * sigmoid(x) as torch computes it.
__global__ void pd_dit_silu_kernel(float* __restrict__ x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}

PD_EXPORT
int pd_dit_silu(void* x, uint32_t n, void* stream) {
    if (n == 0u) return 0;
    pd_dit_silu_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>((float*)x, n);
    return pd_launch_status();
}

// ---- row softmax in place, f32 [rows][n] ------------------------------------
//
// The VAE mid block's attention is one head of width C over every latent
// pixel (4096 at 1024^2, 16384 at 2K): scores come out of the f16 GEMM as an
// f32 [rows][n] plane, this normalises each row, and the P.V GEMM follows.
// One block per row; each thread walks a fixed stride and the tree fold is
// fixed, so the sum is bit-stable. `scale` is applied before the max, which
// is where torch's sdpa applies 1/sqrt(E) (the engine folds it into q and
// passes 1 here; the parameter exists so a caller can do it either way).
__global__ void __launch_bounds__(256) pd_dit_softmax_rows_kernel(float* __restrict__ x,
                                                                  uint32_t n, float scale) {
    __shared__ float red[256];
    float* row = x + (size_t)blockIdx.x * n;
    const uint32_t tid = threadIdx.x;
    float m = -3.402823466e+38f;
    for (uint32_t i = tid; i < n; i += 256u) m = fmaxf(m, row[i] * scale);
    red[tid] = m;
    __syncthreads();
    for (uint32_t s = 128u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += 256u) {
        const float e = expf(row[i] * scale - m);
        row[i] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (uint32_t s = 128u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (uint32_t i = tid; i < n; i += 256u) row[i] *= inv;
}

PD_EXPORT
int pd_dit_softmax_rows(void* x, uint32_t rows, uint32_t n, float scale, void* stream) {
    if (rows == 0u || n == 0u) return 0;
    pd_dit_softmax_rows_kernel<<<rows, 256u, 0, (cudaStream_t)stream>>>((float*)x, n, scale);
    return pd_launch_status();
}

// ---- f16 transpose [rows][cols] -> [cols][rows] -----------------------------
// The P.V GEMM wants V as its weight operand, which the f16 GEMM reads as
// [out][in] = [C][pixels]: V comes out of the qkv projection as [pixels][C].
__global__ void pd_dit_transpose_f16_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                                            uint32_t rows, uint32_t cols) {
    __shared__ __half tile[32][33];
    const uint32_t c0 = blockIdx.x * 32u, r0 = blockIdx.y * 32u;
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;   // 32 x 8
    for (uint32_t j = ty; j < 32u; j += 8u) {
        const uint32_t r = r0 + j, c = c0 + tx;
        if (r < rows && c < cols) tile[j][tx] = src[(size_t)r * cols + c];
    }
    __syncthreads();
    for (uint32_t j = ty; j < 32u; j += 8u) {
        const uint32_t c = c0 + j, r = r0 + tx;       // dst[c][r]
        if (r < rows && c < cols) dst[(size_t)c * rows + r] = tile[tx][j];
    }
}

PD_EXPORT
int pd_dit_transpose_f16(const void* src, void* dst, uint32_t rows, uint32_t cols, void* stream) {
    if (rows == 0u || cols == 0u) return 0;
    dim3 grid((cols + 31u) / 32u, (rows + 31u) / 32u);
    pd_dit_transpose_f16_kernel<<<grid, dim3(32u, 8u), 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)dst, rows, cols);
    return pd_launch_status();
}

// ---- qkv split to f16 with the query pre-scaled ----------------------------
// [rows][3C] f32 (the VAE attention's 1x1 qkv conv, bias already added) ->
// three [rows][C] f16 planes, q multiplied by qscale (1/sqrt(C)) before its
// round so the softmax runs at scale 1 - the same fold the dinov3 split does.
__global__ void pd_dit_split3_f16_kernel(const float* __restrict__ src, __half* __restrict__ q,
                                         __half* __restrict__ k, __half* __restrict__ v,
                                         uint32_t C, float qscale, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / C;
    const uint32_t c = (uint32_t)(i - r * C);
    const float* s = src + r * (size_t)(3u * C);
    q[i] = __float2half(s[c] * qscale);
    k[i] = __float2half(s[C + c]);
    v[i] = __float2half(s[2u * C + c]);
}

PD_EXPORT
int pd_dit_split3_f16(const void* src, void* q, void* k, void* v, uint32_t rows, uint32_t C,
                      float qscale, void* stream) {
    const size_t total = (size_t)rows * C;
    if (total == 0u) return 0;
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_dit_split3_f16_kernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)src, (__half*)q, (__half*)k, (__half*)v, C, qscale, total);
    return pd_launch_status();
}

// ---- per-column affine: x[r][c] = x[r][c] * a[c] + b[c] --------------------
// The VAE's latent de-normalisation (`latents * std + mean`, both per
// channel) on the packed [token][channel] plane.
__global__ void pd_dit_affine_cols_kernel(float* __restrict__ x, const float* __restrict__ a,
                                          const float* __restrict__ b, uint32_t n, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t c = (uint32_t)(i % n);
    x[i] = fmaf(x[i], a[c], b[c]);
}

PD_EXPORT
int pd_dit_affine_cols(void* x, const void* a, const void* b, uint32_t rows, uint32_t n,
                       void* stream) {
    const size_t total = (size_t)rows * n;
    if (total == 0u) return 0;
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_dit_affine_cols_kernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)a, (const float*)b, n, total);
    return pd_launch_status();
}

// ---- decoded pixels to 8-bit ------------------------------------------------
// The decoder's clamp(-1, 1), then diffusers' postprocess: (x / 2 + 0.5)
// clamped to [0, 1], times 255, rounded. NHWC in, the same interleaved
// channel order out (RGBA for this VAE), so the host hands the bytes straight
// to the PNG encoder.
__global__ void pd_dit_to_u8_kernel(const float* __restrict__ x, uint8_t* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = fminf(fmaxf(x[i], -1.0f), 1.0f);
    v = fminf(fmaxf(v * 0.5f + 0.5f, 0.0f), 1.0f);
    out[i] = (uint8_t)rintf(v * 255.0f);
}

PD_EXPORT
int pd_dit_to_u8(const void* x, void* out, uint32_t n, void* stream) {
    if (n == 0u) return 0;
    pd_dit_to_u8_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (uint8_t*)out, n);
    return pd_launch_status();
}
