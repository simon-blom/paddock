// quant/hadamard.cuh - the blockwise Walsh-Hadamard rotation of rotated-basis
// checkpoints (PrismML's Bonsai GGUFs, `prism.hadamard.*` version 1; the same
// primitive QuaRot / SpinQuant's online R3/R4 and QuIP#'s RHT need - the one
// kernel that family of formats asks of a runtime).
//
// Such a file stores W' = W R^T with R = H diag(s) per BLK-wide strip of the
// input dimension, so every rotated matmul is fed x' = H(s * x). H is the
// normalized Sylvester matrix, H[r][c] = (-1)^popcount(r & c) / sqrt(BLK),
// its own inverse; s is a fixed +-1 vector over the input width. A table read
// by row lookup (the token embedding) holds rotated rows and comes back
// through the inverse, h = s * H(z).
//
// Shape of the kernel. One thread block of NT = 256 per (row, strip): thread
// t owns the NE = BLK / NT elements whose index is t mod NT and keeps them in
// registers, so the ten butterfly stages of a 1024 strip split three ways by
// WHICH index bit a stage pairs on:
//   bits 0..4  partner is another lane of the same warp  -> a warp shuffle
//   bits 5..7  partner is a thread of another warp       -> through shared
//   bits 8..   partner is another register of the thread -> register ops
// Only the middle class synchronizes the block. Signs (and the GDN head
// permutation, below) fold into the load, the inverse's signs into the store,
// so the transform is one pass with no staging plane of its own.
//
// Numerics: stages run in ascending h, low element a + b, high element a - b,
// the normalization multiplied in on the way in (exact for BLK = 1024, a
// power of two). Every output is therefore one fixed tree of f32 additions
// and the kernel agrees BIT FOR BIT with the CPU test reference
// (paddock-kernels/src/reference/hadamard.rs) however the work is spread -
// the gate in tests/gpu_ternary.rs is identity, not a tolerance.
//
// The fused form (slot 629, Q = true below): the same transform, and then
// the per-128 int8 quantize of quant/ternary.cuh straight out of the
// registers - the decode step of a rotated model runs ~257 rotate-then-
// quantize pairs a token, each a launch-bound epilogue, and this halves
// them. Element i * 256 + t of a strip sits in register i of
// thread t, so quant block 2i + (t >> 7) is one register across one HALF of
// the thread block: its absmax is a warp reduce plus one shared-memory
// exchange between the half's four warps. The rotated values it quantizes
// are bit-identical to the standalone op's, and the bytes and scales to
// pd_quantize_q8_b128 over them - the standalone pair stays the oracle.
// Still open: the norm in front of it (norm -> signs -> H -> quantize).

#define PD_HAD_NT 256u

// mode bits of pd_hadamard_rows
#define PD_HAD_INVERSE 1u  // y = s * H(x): a looked-up row of a rotated table

// Source element of grouped-order element e: `prism.hadamard.gdn_v_grouped`
// files expect the gated-delta-net output projection's input with the value
// heads of one key group adjacent (head = r + rep * k) where the engine lays
// them out tiled (head = k + nk * r). Whole heads move.
__device__ __forceinline__ uint32_t pd_had_tiled(uint32_t e, uint32_t hd, uint32_t nk, uint32_t rep) {
    const uint32_t head = e / hd, off = e - head * hd;
    const uint32_t k = head / rep, r = head - k * rep;
    return (k + nk * r) * hd + off;
}

// Q: also quantize the result (xq int8 [rows, width], xs f32 [rows, width /
// 128]); y may then be null when nobody reads the f32 rows.
template <uint32_t BLK, bool INV, bool PERM, bool Q>
__global__ void __launch_bounds__(PD_HAD_NT) pd_hadamard_kernel(
        const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ signs,
        signed char* __restrict__ xq, float* __restrict__ xs,
        uint32_t rows, uint32_t width, uint32_t hd, uint32_t nk, uint32_t rep) {
    PD_PDL_ARM();
    constexpr uint32_t NT = PD_HAD_NT, NE = BLK / NT;
    __shared__ float sh[BLK];
    const uint32_t tid = threadIdx.x, lane = tid & 31u;
    const uint32_t strips = width / BLK;
    const uint64_t tasks = (uint64_t)rows * strips;
    const float norm = 1.0f / sqrtf((float)BLK);
    // grid-stride over (row, strip): every thread of a block runs the same
    // trips, so the syncs below line up whatever the grid was capped at
    for (uint64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const uint64_t row = task / strips;
        const uint32_t e0 = (uint32_t)(task - row * strips) * BLK;
        const float* xr = x + row * width;
        float* yr = y + row * width;
        float reg[NE];
        #pragma unroll
        for (uint32_t i = 0; i < NE; ++i) {
            const uint32_t e = e0 + i * NT + tid;
            const float v = xr[PERM ? pd_had_tiled(e, hd, nk, rep) : e] * norm;
            reg[i] = INV ? v : v * signs[e];
        }
        // index bits 0..4: the partner lane of the same warp
        #pragma unroll
        for (uint32_t h = 1u; h < 32u; h <<= 1u) {
            #pragma unroll
            for (uint32_t i = 0; i < NE; ++i) {
                const float a = reg[i];
                const float b = __shfl_xor_sync(0xffffffffu, a, h, 32);
                reg[i] = (lane & h) == 0u ? a + b : b - a;
            }
        }
        // index bits 5..7: the partner thread sits in another warp
        #pragma unroll
        for (uint32_t h = 32u; h < NT; h <<= 1u) {
            #pragma unroll
            for (uint32_t i = 0; i < NE; ++i) sh[i * NT + tid] = reg[i];
            __syncthreads();
            #pragma unroll
            for (uint32_t i = 0; i < NE; ++i) {
                const float a = reg[i], b = sh[i * NT + (tid ^ h)];
                reg[i] = (tid & h) == 0u ? a + b : b - a;
            }
            __syncthreads();
        }
        // index bits 8..: both elements are this thread's own
        #pragma unroll
        for (uint32_t step = 1u; step < NE; step <<= 1u) {
            #pragma unroll
            for (uint32_t i = 0; i < NE; i += 2u * step) {
                #pragma unroll
                for (uint32_t k = 0; k < step; ++k) {
                    const float a = reg[i + k], b = reg[i + k + step];
                    reg[i + k] = a + b;
                    reg[i + k + step] = a - b;
                }
            }
        }
        if (!Q || y != nullptr) {
            #pragma unroll
            for (uint32_t i = 0; i < NE; ++i) {
                const uint32_t e = e0 + i * NT + tid;
                yr[e] = INV ? reg[i] * signs[e] : reg[i];
            }
        }
        if (Q) {
            // absmax of quant block 2i + half: reduce inside the warp, then
            // across the half's four warps through shared memory
            const uint32_t warp = tid >> 5u, half = tid >> 7u;
            float am[NE];
            #pragma unroll
            for (uint32_t i = 0; i < NE; ++i) {
                float a = fabsf(reg[i]);
                #pragma unroll
                for (uint32_t s = 16u; s > 0u; s >>= 1u)
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
                am[i] = a;
            }
            if (lane == 0u) {
                #pragma unroll
                for (uint32_t i = 0; i < NE; ++i) sh[i * 8u + warp] = am[i];
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t i = 0; i < NE; ++i) {
                const float* w4 = sh + i * 8u + half * 4u;
                am[i] = fmaxf(fmaxf(w4[0], w4[1]), fmaxf(w4[2], w4[3]));
            }
            __syncthreads();
            signed char* qr = xq + row * width;
            float* sr = xs + row * (width >> 7u);
            #pragma unroll
            for (uint32_t i = 0; i < NE; ++i) {
                const uint32_t e = e0 + i * NT + tid;
                const float scl = am[i] * (1.0f / 127.0f);
                if ((tid & 127u) == 0u) sr[e >> 7u] = scl;
                const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
                int qi = __float2int_rn(reg[i] * inv);
                qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                qr[e] = (signed char)qi;
            }
        }
    }
}

// slot 626: y = H(s * x) over each `block`-wide strip of every `width`-wide
// row (mode 0), or y = s * H(x) (PD_HAD_INVERSE). `signs` is `width` f32 of
// +-1. `hd` != 0 applies the tiled -> grouped GDN head permutation on the way
// in (forward mode only; `width` = hd * nk * rep). x == y is fine without the
// permutation - a thread reads all it owns before it writes - and refused
// with it, where a strip's sources lie in other strips.
PD_EXPORT
int pd_hadamard_rows(const void* x, void* y, const void* signs, uint32_t rows, uint32_t width,
                     uint32_t block, uint32_t mode, uint32_t hd, uint32_t nk, uint32_t rep,
                     void* stream) {
    if (rows == 0u) return 0;
    if (width == 0u || block == 0u || width % block != 0u) return cudaErrorInvalidValue;
    const bool inv = (mode & PD_HAD_INVERSE) != 0u, perm = hd != 0u;
    if (perm && (inv || x == y || (uint64_t)hd * nk * rep != width)) return cudaErrorInvalidValue;
    const uint64_t tasks = (uint64_t)rows * (width / block);
    const uint32_t grid = tasks < 65535u ? (uint32_t)tasks : 65535u;
    auto st = (cudaStream_t)stream;
#define PD_HAD_GO(BLK, INV, PERM)                                                          \
    pd_hadamard_kernel<BLK, INV, PERM, false><<<grid, PD_HAD_NT, 0, st>>>(                  \
        (const float*)x, (float*)y, (const float*)signs, (signed char*)nullptr,            \
        (float*)nullptr, rows, width, hd, nk, rep)
#define PD_HAD_BLK(BLK)                                                                    \
    do {                                                                                   \
        if (inv) PD_HAD_GO(BLK, true, false);                                              \
        else if (perm) PD_HAD_GO(BLK, false, true);                                        \
        else PD_HAD_GO(BLK, false, false);                                                 \
    } while (0)
    switch (block) {
        case 256u: PD_HAD_BLK(256u); break;
        case 512u: PD_HAD_BLK(512u); break;
        case 1024u: PD_HAD_BLK(1024u); break;
        case 2048u: PD_HAD_BLK(2048u); break;
        case 4096u: PD_HAD_BLK(4096u); break;
        default: return cudaErrorInvalidValue;
    }
#undef PD_HAD_BLK
#undef PD_HAD_GO
    return pd_launch_status();
}

// slot 629: the rotation and the per-128 int8 quantize in one launch:
// xq / xs = Q128(H(s * x)), forward only. `y` (nullable) also takes the f32
// rotated rows - x == y is fine without the permutation - for the consumers
// that are not int8 lanes (a gated-delta-net layer's alpha / beta plane).
PD_EXPORT
int pd_hadamard_rows_q8_b128(const void* x, void* y, const void* signs, void* xq, void* xs,
                             uint32_t rows, uint32_t width, uint32_t block, uint32_t hd,
                             uint32_t nk, uint32_t rep, void* stream) {
    if (rows == 0u) return 0;
    if (width == 0u || block < 256u || width % block != 0u) return cudaErrorInvalidValue;
    const bool perm = hd != 0u;
    if (perm && (x == y || (uint64_t)hd * nk * rep != width)) return cudaErrorInvalidValue;
    const uint64_t tasks = (uint64_t)rows * (width / block);
    const uint32_t grid = tasks < 65535u ? (uint32_t)tasks : 65535u;
    auto st = (cudaStream_t)stream;
#define PD_HADQ_GO(BLK, PERM)                                                              \
    pd_hadamard_kernel<BLK, false, PERM, true><<<grid, PD_HAD_NT, 0, st>>>(                 \
        (const float*)x, (float*)y, (const float*)signs, (signed char*)xq, (float*)xs,     \
        rows, width, hd, nk, rep)
#define PD_HADQ_BLK(BLK)                                                                   \
    do {                                                                                   \
        if (perm) PD_HADQ_GO(BLK, true);                                                   \
        else PD_HADQ_GO(BLK, false);                                                       \
    } while (0)
    switch (block) {
        case 256u: PD_HADQ_BLK(256u); break;
        case 512u: PD_HADQ_BLK(512u); break;
        case 1024u: PD_HADQ_BLK(1024u); break;
        case 2048u: PD_HADQ_BLK(2048u); break;
        case 4096u: PD_HADQ_BLK(4096u); break;
        default: return cudaErrorInvalidValue;
    }
#undef PD_HADQ_BLK
#undef PD_HADQ_GO
    return pd_launch_status();
}

// slot 625: capability marker - the i-quant lanes serve PrismML's ternary
// packings PTQ1_0 (GGUF raw id 143) and PQ2_0 (142). The dtypes ride existing
// entry points, so their slot presence cannot answer; this one can.
PD_EXPORT
int pd_kquant_ternary(void) { return 0; }
