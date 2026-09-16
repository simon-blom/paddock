// Row-exact multi-row twin of the batch-1 Q8_0 GEMV (slot 598).
//
// Why it exists: a speculative verify row must land exactly where a decode
// tick lands on that token, bit for bit - a near-tie greedy token does not
// survive a last-ulp difference, and the batched Q8_0 GEMMs (mt tiles, mmq)
// reduce in their own order. The qwen4_exp planes the decode tick runs through
// pd_q8_0_gemv_repacked (the hyper-connection up plane [320 -> 10240] and the
// shared-expert down plane) had no batch form, so the verify walked them one
// row per launch.
//
// Here the grid gains a token axis - grid (out_dim, batch), 128 threads - and
// every (row, token) block runs the batch-1 block verbatim: the same scale
// preload into shared, the same 16-element chunks at tid*16 stride, the same
// shuffle tree and serial cross-warp fold. A block never sees `batch`, so each
// token's output is the batch-1 call's bit for bit, at one launch.
__global__ void pd_q8_0_gemv_repacked_rows_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    const uint32_t b = blockIdx.y;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t bl = tid; bl < n_blocks; bl += nth) ssc[bl] = __half2float(srow[bl]);
    PD_PDL_ARM();
    __shared__ float wsum[32];
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;
    const float* xr = x + (size_t)b * in_dim;
    float acc = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
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
        acc += ssc[base >> 5] * s;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[o];
        y[(size_t)b * out_dim + o] = v;
    }
}

PD_EXPORT
int pd_q8_0_gemv_repacked_rows(const void* data, const void* scale, const void* bias,
                               const void* x, void* y, uint32_t in_dim, uint32_t out_dim,
                               uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // the batch-1 launcher's own geometry (128 threads, one scale row of
    // shared) - a different thread count would regroup the fold
    const uint32_t threads = 128;
    const uint32_t shmem = (in_dim >> 5) * sizeof(float);
    pd_pdl_go(pd_q8_0_gemv_repacked_rows_kernel, dim3(out_dim, batch), threads, shmem,
              (cudaStream_t)stream, (const int8_t*)data, (const __half*)scale,
              (const float*)bias, (const float*)x, (float*)y, in_dim, out_dim);
    return pd_launch_status();
}
