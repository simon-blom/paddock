// dense_pred.cuh - dense-prediction towers: a LayerScale ViT backbone (DINOv3) and the convolutional decoder that turns its patch grid back into a raster
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ dense prediction
// First consumer: tic-forestry-v1 (DINOv3 ViT-L/16 + a four-stage multi-scale
// decoder; reference = transformers' modeling_dinov3_vit.py and the training
// graph's own Fuse module). Nothing here is a token op: the unit of work is a
// chip, every plane is [chips][rows][channels] with the channel innermost
// (NHWC for the decoder), and a batch of chips is pure row count.
//
// Precision class is the vision towers': f32 residual and f32 accumulate, f16
// GEMM operands. Each kernel below that feeds a GEMM writes the f16 staging
// plane itself, so no plane is streamed twice just to change its dtype - the
// pass-count law the PaddleOCR tower fusions established.
//
// Needs asr/whisper.cuh (pd_whisper_ln_staged / pd_whisper_ln_body: the
// register-staged LayerNorm, reused so this tower's norm reduces in exactly
// the order every other f16 tower's does).

// 610: the patch stem. u8 HWC chips of `ch` bands -> normalized f16 patch rows
// in the conv stem's im2row order (col = c*p*p + ky*p + kx, which is the
// [out][c][ky][kx] weight flattened - a relabel, not a permute). Same idea as
// pd_ocr_patches_u8 with two differences: the band count is an argument (the
// forestry chips carry near-infrared as a fourth), and each chip owns
// `chip_rows` >= g*g output rows, the tail written as zeros. The tail is where
// the class and register tokens go: a zero row through the biasless patch GEMM
// is exactly zero, so one broadcast row add afterwards lands the bias on the
// patches and the learned tokens on the tail, and the hidden plane never needs
// a scatter.
//
// (u8/255 - mean)/std in IEEE f32, in that order - the statistics are in 0-1
// units, and scaling them to 0-255 instead is the silent 0.002-mIoU failure
// the bring-up notes warn about. No fast-math in this pack, so both divisions
// round to nearest, which is what the training graph's div_/sub_/div_ did.
__global__ void pd_dp_u8_patch_rows_kernel(const uint8_t* __restrict__ px8,
                                           __half* __restrict__ out,
                                           float m0, float m1, float m2, float m3,
                                           float s0, float s1, float s2, float s3,
                                           uint32_t g, uint32_t p, uint32_t px,
                                           uint32_t ch, uint32_t chip_rows, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t pp = p * p, k = ch * pp;
    const uint64_t row = i / k;
    const uint32_t col = (uint32_t)(i - row * k);
    const uint64_t b = row / chip_rows;
    const uint32_t r2 = (uint32_t)(row - b * chip_rows);
    if (r2 >= g * g) {
        out[i] = __float2half(0.0f);
        return;
    }
    const uint32_t c = col / pp, rem = col - c * pp;
    const uint32_t ky = rem / p, kx = rem - ky * p;
    const uint32_t gy = r2 / g, gx = r2 - gy * g;
    const uint64_t src = ((b * px + gy * p + ky) * px + (gx * p + kx)) * ch + c;
    const float mean = c == 0u ? m0 : (c == 1u ? m1 : (c == 2u ? m2 : m3));
    const float sd = c == 0u ? s0 : (c == 1u ? s1 : (c == 2u ? s2 : s3));
    out[i] = __float2half(((float)px8[src] / 255.0f - mean) / sd);
}

// The band statistics ride the launch as scalars (unused bands are ignored),
// so the kernel does no table read per pixel - the pd_ocr_patches_u8 shape.
PD_EXPORT
int pd_dp_u8_patch_rows(const void* pixels, void* out, float m0, float m1, float m2, float m3,
                        float s0, float s1, float s2, float s3, uint32_t chips, uint32_t px,
                        uint32_t patch, uint32_t ch, uint32_t chip_rows, void* stream) {
    if (chips == 0) return 0;
    if (patch == 0 || px % patch != 0 || ch == 0 || ch > 4u) return cudaErrorInvalidValue;
    const uint32_t g = px / patch;
    if (chip_rows < g * g) return cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)chips * chip_rows * ch * patch * patch;
    const uint64_t blocks = (n + 255ull) / 256ull;
    pd_dp_u8_patch_rows_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)pixels, (__half*)out, m0, m1, m2, m3, s0, s1, s2, s3, g, patch, px,
        ch, chip_rows, n);
    return pd_launch_status();
}

// 611: the fused q|k|v landing -> the three planes vision_attn eats, with the
// q/v biases folded in the load (k_proj carries none - a DINOv3 checkpoint
// fact, not an export slip) and the rope applied in the same pass.
//
// DINOv3's position signal is only this rope, and it is not an integer-position
// rope: patch centres are normalized to (-1, 1), the angle is 2*pi*coord*
// base^(-j/(hd/4)) with base 100, the first hd/4 frequencies read the row
// coordinate and the next hd/4 the column, and the pair is (j, j + hd/2)
// (rotate_half). None of that is reachable from pd_rope2d's theta = pos*ts^i,
// so the angles arrive as cos/sin tables [n_rope][hd/2], built once per
// geometry on the host in the f32 order the reference uses.
//
// Only the first n_rope rows of each chip_rows-row chip are roped - the patch
// tokens. The class and register tokens sit after them and pass through: with
// no mask, attention is permutation-equivariant, so parking the prefix tokens
// at the tail costs nothing and makes "the patch grid" rows [0, n_rope) of a
// chip with no gather.
//
// One thread per pair, same reasoning as pd_rope2d_kernel: consecutive threads
// walk consecutive dims of one head, so a warp's loads and stores coalesce.
__global__ void pd_dp_qkv_split_rope_kernel(const float* __restrict__ qkv,
                                            const float* __restrict__ bq,
                                            const float* __restrict__ bv,
                                            const float* __restrict__ cs,
                                            const float* __restrict__ sn,
                                            float* __restrict__ q, float* __restrict__ k,
                                            float* __restrict__ v, uint32_t d, uint32_t hd,
                                            uint32_t chip_rows, uint32_t n_rope) {
    const uint32_t r = blockIdx.x;
    const size_t src = (size_t)r * 3u * d, dst = (size_t)r * d;
    const uint32_t half = hd / 2u, pairs = d / 2u;
    const uint32_t t = r % chip_rows;
    if (t < n_rope) {
        const float* cr = cs + (size_t)t * half;
        const float* sr = sn + (size_t)t * half;
        for (uint32_t pp = threadIdx.x; pp < pairs; pp += blockDim.x) {
            const uint32_t h = pp / half, j = pp - h * half;
            const uint32_t e0 = h * hd + j, e1 = e0 + half;
            const float c = cr[j], s = sr[j];
            const float q0 = qkv[src + e0] + bq[e0], q1 = qkv[src + e1] + bq[e1];
            const float k0 = qkv[src + d + e0], k1 = qkv[src + d + e1];
            q[dst + e0] = q0 * c - q1 * s;
            q[dst + e1] = q1 * c + q0 * s;
            k[dst + e0] = k0 * c - k1 * s;
            k[dst + e1] = k1 * c + k0 * s;
            v[dst + e0] = qkv[src + 2u * d + e0] + bv[e0];
            v[dst + e1] = qkv[src + 2u * d + e1] + bv[e1];
        }
    } else {
        for (uint32_t pp = threadIdx.x; pp < pairs; pp += blockDim.x) {
            const uint32_t h = pp / half, j = pp - h * half;
            const uint32_t e0 = h * hd + j, e1 = e0 + half;
            q[dst + e0] = qkv[src + e0] + bq[e0];
            q[dst + e1] = qkv[src + e1] + bq[e1];
            k[dst + e0] = qkv[src + d + e0];
            k[dst + e1] = qkv[src + d + e1];
            v[dst + e0] = qkv[src + 2u * d + e0] + bv[e0];
            v[dst + e1] = qkv[src + 2u * d + e1] + bv[e1];
        }
    }
}

PD_EXPORT
int pd_dp_qkv_split_rope(const void* qkv, const void* bq, const void* bv, const void* cs,
                         const void* sn, void* q, void* k, void* v, uint32_t d, uint32_t hd,
                         uint32_t rows, uint32_t chip_rows, uint32_t n_rope, void* stream) {
    if (rows == 0 || d == 0) return 0;
    if (hd == 0 || (hd & 1u) != 0 || d % hd != 0 || chip_rows == 0 || n_rope > chip_rows)
        return cudaErrorInvalidValue;
    pd_dp_qkv_split_rope_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)qkv, (const float*)bq, (const float*)bv, (const float*)cs,
        (const float*)sn, (float*)q, (float*)k, (float*)v, d, hd, chip_rows, n_rope);
    return pd_launch_status();
}

// ---- the half interface (621-623) --------------------------------------------
// The three seams between a ViT block's GEMMs all measured at 620-650 GB/s of
// the A6000's 768: they are as fast as their bytes allow, so the only lever
// left was the bytes. With the GEMMs landing halves (pd_f16_gemm_h) each seam
// reads 2 B an element instead of 4, and attention takes halves too
// (pd_vision_attn_h), so this one also writes 2. Measured at the ViT-L shapes
// of a 16-chip pass (bench/dp_h16_bench.cu): the split 624 -> 317 us, the GELU
// 602 -> 413 us. The arithmetic between the load and the store is unchanged
// f32; what changes numerically is that a GEMM output is rounded to f16 before
// the seam reads it - 11 significant bits where the reference's own bf16
// landing keeps 8. The tower's largest GEMM output on the golden chips is 646
// (qkv, block 3), two orders inside f16's range.
//
// 621: pd_dp_qkv_split_rope on halves, with 1/sqrt(hd) folded into q before
// the round. The f32 attention kernel scales then rounds into its fragments;
// folding the scale here makes q's single round land on the same half, for any
// head dim, and lets pd_vision_attn_h load the fragment as a word.
__global__ void pd_dp_qkv_split_rope_h_kernel(const __half* __restrict__ qkv,
                                              const float* __restrict__ bq,
                                              const float* __restrict__ bv,
                                              const float* __restrict__ cs,
                                              const float* __restrict__ sn,
                                              __half* __restrict__ q, __half* __restrict__ k,
                                              __half* __restrict__ v, uint32_t d, uint32_t hd,
                                              uint32_t chip_rows, uint32_t n_rope, float qs) {
    const uint32_t r = blockIdx.x;
    const size_t src = (size_t)r * 3u * d, dst = (size_t)r * d;
    const uint32_t half = hd / 2u, pairs = d / 2u;
    const uint32_t t = r % chip_rows;
    if (t < n_rope) {
        const float* cr = cs + (size_t)t * half;
        const float* sr = sn + (size_t)t * half;
        for (uint32_t pp = threadIdx.x; pp < pairs; pp += blockDim.x) {
            const uint32_t h = pp / half, j = pp - h * half;
            const uint32_t e0 = h * hd + j, e1 = e0 + half;
            const float c = cr[j], s = sr[j];
            const float q0 = __half2float(qkv[src + e0]) + bq[e0];
            const float q1 = __half2float(qkv[src + e1]) + bq[e1];
            const float k0 = __half2float(qkv[src + d + e0]);
            const float k1 = __half2float(qkv[src + d + e1]);
            q[dst + e0] = __float2half((q0 * c - q1 * s) * qs);
            q[dst + e1] = __float2half((q1 * c + q0 * s) * qs);
            k[dst + e0] = __float2half(k0 * c - k1 * s);
            k[dst + e1] = __float2half(k1 * c + k0 * s);
            v[dst + e0] = __float2half(__half2float(qkv[src + 2u * d + e0]) + bv[e0]);
            v[dst + e1] = __float2half(__half2float(qkv[src + 2u * d + e1]) + bv[e1]);
        }
    } else {
        for (uint32_t pp = threadIdx.x; pp < pairs; pp += blockDim.x) {
            const uint32_t h = pp / half, j = pp - h * half;
            const uint32_t e0 = h * hd + j, e1 = e0 + half;
            q[dst + e0] = __float2half((__half2float(qkv[src + e0]) + bq[e0]) * qs);
            q[dst + e1] = __float2half((__half2float(qkv[src + e1]) + bq[e1]) * qs);
            k[dst + e0] = qkv[src + d + e0];
            k[dst + e1] = qkv[src + d + e1];
            v[dst + e0] = __float2half(__half2float(qkv[src + 2u * d + e0]) + bv[e0]);
            v[dst + e1] = __float2half(__half2float(qkv[src + 2u * d + e1]) + bv[e1]);
        }
    }
}

PD_EXPORT
int pd_dp_qkv_split_rope_h(const void* qkv, const void* bq, const void* bv, const void* cs,
                           const void* sn, void* q, void* k, void* v, uint32_t d, uint32_t hd,
                           uint32_t rows, uint32_t chip_rows, uint32_t n_rope, float qscale,
                           void* stream) {
    if (rows == 0 || d == 0) return 0;
    if (hd == 0 || (hd & 1u) != 0 || d % hd != 0 || chip_rows == 0 || n_rope > chip_rows)
        return cudaErrorInvalidValue;
    pd_dp_qkv_split_rope_h_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const __half*)qkv, (const float*)bq, (const float*)bv, (const float*)cs,
        (const float*)sn, (__half*)q, (__half*)k, (__half*)v, d, hd, chip_rows, n_rope,
        qscale);
    return pd_launch_status();
}

// 612: the residual seam of a LayerScale block in one launch -
//     x += ls * (proj + bias);  out = f16(LayerNorm(x))
// LayerScale multiplies the sublayer output per channel before the residual
// add; leave it out and the checkpoint still loads and the tower returns
// noise. It rides the seam because the seam already touches every element of
// x, proj and the norm output: folded here the multiply is free, as its own
// pass it would be a fourth walk over a plane this kernel walks once. Folding
// ls into the projection weights instead was considered and refused - the
// lambdas are small, the products would land in f16's subnormal tail, and the
// tower would lose exactly the precision the f32 residual exists to keep.
//
// Body is pd_whisper_res_ln_f16_kernel's with the multiply added, down to the
// register staging: the row is held in registers from the residual update
// through the norm, so the seam costs one DRAM round trip instead of four.
//
// PT is what the projection landed as: float (612) or __half (622, behind
// pd_f16_gemm_h). Same body either way - the load widens and nothing else
// moves, so the two agree exactly wherever the f32 landing was representable.
__device__ __forceinline__ float pd_dp_ld(const float* p, uint32_t i) { return p[i]; }
__device__ __forceinline__ float pd_dp_ld(const __half* p, uint32_t i) {
    return __half2float(p[i]);
}

template <typename PT>
__global__ void pd_dp_res_ls_ln_f16_kernel(float* __restrict__ x,
                                           const PT* __restrict__ proj,
                                           const float* __restrict__ bias,
                                           const float* __restrict__ ls,
                                           const float* __restrict__ w,
                                           const float* __restrict__ b,
                                           __half* __restrict__ out, uint32_t n, float eps) {
    const uint32_t row = blockIdx.x;
    float* xr = x + (size_t)row * n;
    const PT* pr = proj + (size_t)row * n;
    __half* orow = out + (size_t)row * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    if (n <= nth * PD_WLN_RU) {
        float xs[PD_WLN_RU];
#pragma unroll
        for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
            const uint32_t i = tid + u * nth;
            xs[u] = i < n ? xr[i] + ls[i] * (pd_dp_ld(pr, i) + bias[i]) : 0.0f;
        }
#pragma unroll
        for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
            const uint32_t i = tid + u * nth;
            if (i < n) xr[i] = xs[u];
        }
        pd_whisper_ln_staged(xs, w, b, orow, n, eps);
        return;
    }
    for (uint32_t i = tid; i < n; i += nth) xr[i] += ls[i] * (pd_dp_ld(pr, i) + bias[i]);
    __syncthreads();
    pd_whisper_ln_body(xr, w, b, orow, n, eps);
}

PD_EXPORT
int pd_dp_res_ls_ln_f16(void* x, const void* proj, const void* bias, const void* ls,
                        const void* w, const void* b, void* out, uint32_t rows, uint32_t n,
                        float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_dp_res_ls_ln_f16_kernel<float><<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)bias, (const float*)ls, (const float*)w,
        (const float*)b, (__half*)out, n, eps);
    return pd_launch_status();
}

// 622: the same seam off a half projection landing.
PD_EXPORT
int pd_dp_res_ls_ln_h(void* x, const void* proj, const void* bias, const void* ls,
                      const void* w, const void* b, void* out, uint32_t rows, uint32_t n,
                      float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_dp_res_ls_ln_f16_kernel<__half><<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const __half*)proj, (const float*)bias, (const float*)ls, (const float*)w,
        (const float*)b, (__half*)out, n, eps);
    return pd_launch_status();
}

// 623: bias + exact GELU on halves, in place: x[r][i] = f16(gelu(x[r][i] + b[i])).
// The up projection lands halves, the down projection eats halves, so the
// plane between them never needs a second copy - the f32 form read 4 B and
// wrote 2 B somewhere else, this reads and writes the same 2. Two elements a
// thread: a scalar 2 B transaction fills half a sector, and the pair is one
// 4 B load, one pack and one 4 B store (measured: 1 a thread 455 us, 2/4/8 all
// 412-416 us on the 16464 x 4096 plane - the byte roof, reached at 2). An odd
// last element takes the scalar form; `total` is even on every tower so far.
__global__ void pd_dp_gelu_bias_h_kernel(__half* __restrict__ x, const float* __restrict__ bias,
                                         uint32_t n, uint64_t total) {
    const uint64_t i0 = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2ull;
    if (i0 >= total) return;
    const uint32_t b0 = (uint32_t)(i0 % n);
    __half* p = x + i0;
    if (i0 + 1ull < total) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(p));
        // the pair may straddle a row end when n is odd: wrap the bias index
        const float va = f.x + bias[b0];
        const float vb = f.y + bias[b0 + 1u < n ? b0 + 1u : 0u];
        *reinterpret_cast<__half2*>(p) = __floats2half2_rn(
            0.5f * va * (1.0f + erff(va * 0.70710678118654752440084436210484f)),
            0.5f * vb * (1.0f + erff(vb * 0.70710678118654752440084436210484f)));
    } else {
        const float va = __half2float(p[0]) + bias[b0];
        p[0] = __float2half(0.5f * va * (1.0f + erff(va * 0.70710678118654752440084436210484f)));
    }
}

PD_EXPORT
int pd_dp_gelu_bias_h(void* x, const void* bias, uint32_t rows, uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    const uint64_t total = (uint64_t)rows * n;
    const uint64_t blocks = ((total + 1ull) / 2ull + 255ull) / 256ull;
    pd_dp_gelu_bias_h_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (__half*)x, (const float*)bias, n, total);
    return pd_launch_status();
}

// ---------------------------------------------------------------- group norm
// PyTorch GroupNorm over an NHWC plane, then exact-erf GELU, then the f16 store
// the next GEMM eats. A group's statistics run over all pixels of a chip and
// the group's channels - a reduction across rows, which is why no row-wise norm
// in this pack can be bent into it.
//
// Shape of the reduction. NHWC puts one pixel's channels side by side, so the
// mapping that coalesces is lane = channel: a warp's 32 lanes read 32 adjacent
// floats of one pixel and step to the next pixel together. Each thread folds
// its channel over a chunk of pixels and drops one partial; a second, tiny
// launch folds a chip's partials per group in fixed (chunk, channel) order.
// Block-per-group with pixel-strided threads is the obvious alternative and is
// wrong here: at the last decoder stage a group is one channel, lanes would sit
// a whole pixel apart, and every 4-byte load becomes its own sector.
//
// Two passes, like every norm in this pack: the mean first, then the centred
// squares. E[x^2] - mean^2 in one pass is cheaper by a walk and cancels
// catastrophically whenever a group's mean dwarfs its spread; a conv output is
// allowed to do that. Fixed order throughout, so the result is run-to-run
// bit-stable. The group fold accumulates in f64 - a few hundred adds per
// group, where the f32 partials' own rounding is the only error left.
#define PD_DP_GN_Q 64u   // pixels folded per partial

//
// `xb` is the bias of the convolution that produced x. A conv bias is a per-
// channel constant and a group spans channels, so it cannot be dropped as a
// shift of the group mean - but it can be added at the load, in all three
// walks, instead of costing the plane a bias pass of its own first. x + xb
// associates exactly as the separate bias_add would have.
template <bool SQ>
__global__ void pd_dp_gn_reduce_kernel(const float* __restrict__ x,
                                       const float* __restrict__ xb,
                                       const float* __restrict__ mean,
                                       float* __restrict__ part, uint32_t P, uint32_t C,
                                       uint32_t cg, uint32_t G, uint32_t nchunk) {
    const uint32_t blk = blockIdx.x, c = threadIdx.x;
    if (c >= C) return;
    const uint32_t chip = blk / nchunk, chunk = blk - chip * nchunk;
    const uint32_t p0 = chunk * PD_DP_GN_Q;
    const uint32_t p1 = (p0 + PD_DP_GN_Q < P) ? (p0 + PD_DP_GN_Q) : P;
    const float* xp = x + ((size_t)chip * P + p0) * C + c;
    const float bb = xb[c];
    const float m = SQ ? mean[(size_t)chip * G + c / cg] : 0.0f;
    float acc = 0.0f;
    for (uint32_t p = p0; p < p1; ++p, xp += C) {
        const float dlt = (*xp + bb) - m;
        acc += SQ ? dlt * dlt : dlt;
    }
    part[(size_t)blk * C + c] = acc;
}

// mode 0: stat = mean; mode 1: stat = 1/sqrt(var + eps), var the biased
// variance (divide by n), which is what torch.nn.GroupNorm normalizes by.
//
// One block per (chip, group). The first form of this fold was one thread per
// (chip, group) walking nchunk * cg partials serially - 1024 of them at the
// last decoder stage (P = 65536, cg = 1) - and it measured 115 us a launch on
// the RTX PRO 6000, sixteen launches a pass, 6% of the pass, for a few
// hundred threads of work. Thread t owns entries t, t + 256, ..., then a
// fixed-order shared tree; f64 throughout, so a (chip, group) folds the same
// way whatever the pass width. Golden vectors unchanged to the pixel.
__global__ void pd_dp_gn_fold_kernel(const float* __restrict__ part, float* __restrict__ stat,
                                     uint32_t nchunk, uint32_t C, uint32_t cg, uint32_t G,
                                     double inv_n, float eps, uint32_t mode) {
    __shared__ double red[256];
    const uint32_t i = blockIdx.x;           // chip * G + g
    const uint32_t chip = i / G, g = i - chip * G;
    const uint32_t n = nchunk * cg;
    const float* base = part + (size_t)chip * nchunk * C + (size_t)g * cg;
    double acc = 0.0;
    for (uint32_t e = threadIdx.x; e < n; e += 256u) {
        const uint32_t ck = e / cg, j = e - ck * cg;
        acc += (double)base[(size_t)ck * C + j];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    #pragma unroll
    for (uint32_t s = 128u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        const double m = red[0] * inv_n;
        stat[i] = mode == 0u ? (float)m : (float)(1.0 / sqrt(m + (double)eps));
    }
}

__global__ void pd_dp_gn_gelu_f16_kernel(const float* __restrict__ x,
                                         const float* __restrict__ xb,
                                         const float* __restrict__ mean,
                                         const float* __restrict__ inv,
                                         const float* __restrict__ w,
                                         const float* __restrict__ b,
                                         __half* __restrict__ out, uint32_t P, uint32_t C,
                                         uint32_t cg, uint32_t G, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pix = i / C;
    const uint32_t c = (uint32_t)(i - pix * C);
    const size_t s = (size_t)(pix / P) * G + c / cg;
    const float y = ((x[i] + xb[c]) - mean[s]) * inv[s] * w[c] + b[c];
    out[i] = __float2half(0.5f * y * (1.0f + erff(y * 0.70710678118654752440084436210484f)));
}

// 613: x f32 [chips][P][C] (+ the producing conv's bias xb [C]) -> out f16,
// same shape. `part` is engine scratch of chips * ceil(P/64) * C floats,
// `stat` of 2 * chips * G (packs never allocate). Five launches on one stream,
// in order; the folds are a block per (chip, group).
PD_EXPORT
int pd_dp_group_norm_gelu_f16(const void* x, const void* xb, const void* w, const void* b,
                              void* out, void* part, void* stat, uint32_t chips, uint32_t P,
                              uint32_t C, uint32_t G, float eps, void* stream) {
    if (chips == 0 || P == 0 || C == 0) return 0;
    // lane = channel needs the channels to fit one block
    if (G == 0 || C % G != 0 || C > 1024u) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t cg = C / G;
    const uint32_t nchunk = (P + PD_DP_GN_Q - 1u) / PD_DP_GN_Q;
    const uint32_t nth = (C + 31u) & ~31u;
    const uint32_t rblocks = chips * nchunk;
    const double inv_n = 1.0 / ((double)P * (double)cg);
    float* mean = (float*)stat;
    float* inv = (float*)stat + (size_t)chips * G;

    pd_dp_gn_reduce_kernel<false><<<rblocks, nth, 0, st>>>(
        (const float*)x, (const float*)xb, nullptr, (float*)part, P, C, cg, G, nchunk);
    pd_dp_gn_fold_kernel<<<chips * G, 256, 0, st>>>((const float*)part, mean, nchunk, C, cg, G,
                                                    inv_n, eps, 0u);
    pd_dp_gn_reduce_kernel<true><<<rblocks, nth, 0, st>>>(
        (const float*)x, (const float*)xb, mean, (float*)part, P, C, cg, G, nchunk);
    pd_dp_gn_fold_kernel<<<chips * G, 256, 0, st>>>((const float*)part, inv, nchunk, C, cg, G,
                                                    inv_n, eps, 1u);
    const uint64_t n = (uint64_t)chips * P * C;
    const uint64_t blocks = (n + 255ull) / 256ull;
    pd_dp_gn_gelu_f16_kernel<<<(uint32_t)blocks, 256, 0, st>>>(
        (const float*)x, (const float*)xb, mean, inv, (const float*)w, (const float*)b,
        (__half*)out, P, C, cg, G, n);
    return pd_launch_status();
}

// ---------------------------------------------------------------- 3x3 conv
// 614/615: im2row for a 3x3 / stride 1 / zero-pad 1 convolution over an NHWC
// plane, straight into the f16 staging its GEMM eats. TAP-outer columns
// (col = (ky*3+kx)*C + c): consecutive threads then walk consecutive channels
// of one source pixel, so loads and stores are both contiguous runs. The
// channel-outer order would make the [out][c][ky][kx] weight a free relabel,
// but it strides every load by a pixel; the loader permutes the weight once
// instead, which costs nothing per chip.
//
// Out-of-range taps are zeros, computed - torch's padding=1 zero-pads the
// tensor the conv is handed, so there is no pad row to keep armed.
//
// `src_chip_rows` is the row stride between chips in the source, which is how
// the first decoder stage reads the patch grid out of a [chips][tokens] plane
// whose chips carry prefix-token rows after their H*W patches.
//
// SOTA note. This is explicit im2row + the tensor-core GEMM, a 9x staging
// plane the implicit-GEMM form (tap gather inside the GEMM's own A-tile
// staging) never materializes. The decoder is ~1.3% of this model's multiply-
// accumulates and its staging is bounded by one stage at a time, so it is the
// interim; the implicit form is the target if a conv-heavy family ever lands.
__device__ __forceinline__ __half pd_dp_to_half(float v) { return __float2half(v); }
__device__ __forceinline__ __half pd_dp_to_half(__half v) { return v; }

template <typename S>
__global__ void pd_dp_im2row3_kernel(const S* __restrict__ src, __half* __restrict__ out,
                                     uint32_t H, uint32_t W, uint32_t C,
                                     uint32_t src_chip_rows) {
    const uint32_t o = blockIdx.x;
    const uint32_t P = H * W, k = 9u * C;
    const uint32_t chip = o / P, pix = o - chip * P;
    const int32_t y = (int32_t)(pix / W), x = (int32_t)(pix - (uint32_t)y * W);
    const S* sb = src + (size_t)chip * src_chip_rows * C;
    __half* orow = out + (size_t)o * k;
    for (uint32_t col = threadIdx.x; col < k; col += blockDim.x) {
        const uint32_t t = col / C, c = col - t * C;
        const int32_t yy = y + (int32_t)(t / 3u) - 1, xx = x + (int32_t)(t % 3u) - 1;
        const bool in = yy >= 0 && yy < (int32_t)H && xx >= 0 && xx < (int32_t)W;
        // clamp the address so the load is always legal, then select
        const size_t a = in ? ((size_t)((uint32_t)yy * W + (uint32_t)xx) * C + c) : (size_t)c;
        orow[col] = in ? pd_dp_to_half(sb[a]) : __float2half(0.0f);
    }
}

static int pd_dp_im2row3_go(const void* src, void* out, uint32_t chips, uint32_t H,
                            uint32_t W, uint32_t C, uint32_t src_chip_rows, bool src_f16,
                            void* stream) {
    if (chips == 0 || H == 0 || W == 0 || C == 0) return 0;
    if (src_chip_rows < H * W) return cudaErrorInvalidValue;
    const uint32_t k = 9u * C;
    const uint32_t nth = k < 256u ? ((k + 31u) & ~31u) : 256u;
    const uint32_t rows = chips * H * W;
    if (src_f16) {
        pd_dp_im2row3_kernel<__half><<<rows, nth, 0, (cudaStream_t)stream>>>(
            (const __half*)src, (__half*)out, H, W, C, src_chip_rows);
    } else {
        pd_dp_im2row3_kernel<float><<<rows, nth, 0, (cudaStream_t)stream>>>(
            (const float*)src, (__half*)out, H, W, C, src_chip_rows);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_dp_im2row3_f32(const void* src, void* out, uint32_t chips, uint32_t H, uint32_t W,
                      uint32_t C, uint32_t src_chip_rows, void* stream) {
    return pd_dp_im2row3_go(src, out, chips, H, W, C, src_chip_rows, false, stream);
}

// f16 source: the plane a group-norm just wrote - a pure gather, no rounding.
PD_EXPORT
int pd_dp_im2row3_f16(const void* src, void* out, uint32_t chips, uint32_t H, uint32_t W,
                      uint32_t C, uint32_t src_chip_rows, void* stream) {
    return pd_dp_im2row3_go(src, out, chips, H, W, C, src_chip_rows, true, stream);
}

// ------------------------------------------------- transposed conv + skip
// 616: the decoder's stage seam in one pass -
//     out = convT2x2s2(x) + bias + bilinear(skip)
// A 2x2 / stride-2 transposed convolution never overlaps: input pixel (i, j)
// writes output pixels (2i+ky, 2j+kx) and nothing else does, so it is one GEMM
// C_in -> 4*C_out per input pixel (the loader lays the weight out tap-major:
// row (ky*2+kx)*C_out + co) and a depth-to-space. `g` is that GEMM's landing
// [chips][h*w][4*C]; this kernel is the depth-to-space, with the two adds that
// follow it folded in.
//
// The skip is the part the architecture notes leave out: the projected token
// grid is hs x hs (the patch grid) at every stage, and the training graph
// bilinearly resizes it (align_corners=False) to the stage's 2h x 2w before
// adding. It is sampled here, per output pixel, so the upsampled skip is never
// a plane. Source index and weights follow torch's upsample_bilinear2d term
// for term: src = scale*(dst+0.5)-0.5 clamped at 0, i1 = min(i0+1, n-1).
// `bias` is convT.bias + project.bias, summed by the loader - a bilinear
// resize of a constant is that constant, so the project bias can ride along.
//
// skip rows are strided by `skip_chip_rows` per chip for the same reason
// im2row's source is: it is a [chips][tokens] plane read in place.
__global__ void pd_dp_convt2_skip_kernel(const float* __restrict__ g,
                                         const float* __restrict__ bias,
                                         const float* __restrict__ skip,
                                         float* __restrict__ out, uint32_t h, uint32_t w,
                                         uint32_t C, uint32_t hs, uint32_t skip_chip_rows,
                                         float ry, float rx) {
    const uint32_t o = blockIdx.x;
    const uint32_t W2 = 2u * w, Pout = 4u * h * w;
    const uint32_t chip = o / Pout, pix = o - chip * Pout;
    const uint32_t Y = pix / W2, X = pix - Y * W2;
    const uint32_t tap = (Y & 1u) * 2u + (X & 1u);
    const float* gr = g + (((size_t)chip * h + (Y >> 1)) * w + (X >> 1)) * 4u * C
                      + (size_t)tap * C;

    float sy = ry * ((float)Y + 0.5f) - 0.5f;
    float sx = rx * ((float)X + 0.5f) - 0.5f;
    sy = sy < 0.0f ? 0.0f : sy;
    sx = sx < 0.0f ? 0.0f : sx;
    const uint32_t y0 = (uint32_t)sy, x0 = (uint32_t)sx;
    const uint32_t y1 = y0 + 1u < hs ? y0 + 1u : hs - 1u;
    const uint32_t x1 = x0 + 1u < hs ? x0 + 1u : hs - 1u;
    const float ly1 = sy - (float)y0, ly0 = 1.0f - ly1;
    const float lx1 = sx - (float)x0, lx0 = 1.0f - lx1;
    const float* sb = skip + (size_t)chip * skip_chip_rows * C;
    const float* s00 = sb + ((size_t)y0 * hs + x0) * C;
    const float* s01 = sb + ((size_t)y0 * hs + x1) * C;
    const float* s10 = sb + ((size_t)y1 * hs + x0) * C;
    const float* s11 = sb + ((size_t)y1 * hs + x1) * C;

    float* orow = out + (size_t)o * C;
    for (uint32_t c = threadIdx.x; c < C; c += blockDim.x) {
        const float up = gr[c] + bias[c];
        const float sk = ly0 * (lx0 * s00[c] + lx1 * s01[c])
                         + ly1 * (lx0 * s10[c] + lx1 * s11[c]);
        orow[c] = up + sk;
    }
}

PD_EXPORT
int pd_dp_convt2_skip(const void* g, const void* bias, const void* skip, void* out,
                      uint32_t chips, uint32_t h, uint32_t w, uint32_t C, uint32_t hs,
                      uint32_t skip_chip_rows, void* stream) {
    if (chips == 0 || h == 0 || w == 0 || C == 0) return 0;
    if (hs == 0 || skip_chip_rows < hs * hs) return cudaErrorInvalidValue;
    const uint32_t nth = C < 256u ? ((C + 31u) & ~31u) : 256u;
    const uint32_t rows = chips * 4u * h * w;
    // torch computes the scale as in/out in f32 when it is handed a size
    const float ry = (float)hs / (float)(2u * h), rx = (float)hs / (float)(2u * w);
    pd_dp_convt2_skip_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
        (const float*)g, (const float*)bias, (const float*)skip, (float*)out, h, w, C, hs,
        skip_chip_rows, ry, rx);
    return pd_launch_status();
}

// ------------------------------------------------------------------ heads
// 617: both 1x1 output heads off one GEMM landing. The class head (C -> ncls)
// and the height head (C -> 1) read the same final plane, so the loader stacks
// them into one (ncls + 1)-row weight; `o` is that landing [rows][ncls+1] and
// this pass adds the biases, takes the argmax to a u8 class raster, and writes
// the regression value - one walk over the widest raster in the model where
// bias, argmax and a split would have been four.
//
// Lowest index wins a tie (strict greater, ascending), the same rule as
// pd_argmax_rows. `logits` is optional [rows][ncls] f16, biased - the parity
// gate's view; NULL skips it. A NaN logit never wins, so a poisoned chip shows
// up as a non-finite height (same plane, same NaN) which the engine checks.
__global__ void pd_dp_seg_heads_kernel(const float* __restrict__ o,
                                       const float* __restrict__ bias,
                                       uint8_t* __restrict__ cls, float* __restrict__ height,
                                       __half* __restrict__ logits, uint32_t ncls,
                                       uint32_t rows) {
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* p = o + (size_t)r * (ncls + 1u);
    float best = p[0] + bias[0];
    uint32_t bi = 0;
    if (logits) logits[(size_t)r * ncls] = __float2half(best);
    for (uint32_t c = 1; c < ncls; ++c) {
        const float val = p[c] + bias[c];
        if (logits) logits[(size_t)r * ncls + c] = __float2half(val);
        if (val > best) {
            best = val;
            bi = c;
        }
    }
    cls[r] = (uint8_t)bi;
    height[r] = p[ncls] + bias[ncls];
}

PD_EXPORT
int pd_dp_seg_heads(const void* o, const void* bias, void* cls, void* height, void* logits,
                    uint32_t rows, uint32_t ncls, void* stream) {
    if (rows == 0) return 0;
    if (ncls == 0 || ncls > 255u) return cudaErrorInvalidValue;
    const uint32_t blocks = (rows + 255u) / 256u;
    pd_dp_seg_heads_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)o, (const float*)bias, (uint8_t*)cls, (float*)height, (__half*)logits,
        ncls, rows);
    return pd_launch_status();
}
