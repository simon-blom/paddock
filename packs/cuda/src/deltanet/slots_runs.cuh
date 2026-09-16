// Row-exact GDN walk for a speculative verify (slot 599).
//
// A verify row's recurrence has to land exactly where the decode tick lands on
// that token - that tick is pd_gated_delta_recurrent_slots_gn (slot 564) at
// n = 1. The runs walk plus the separate gated norm agree with it only to the
// last ulp (12 of 6144 elements at 1.5e-8 on the Flash-Next layer-0 probe,
// 2026-09-14), and a near-tie greedy token does not survive that; the verify
// then fell back to one slots_gn launch per row per layer.
//
// This is that decode body, token for token, walked over a run's rows inside
// one block: grid (n_heads, n_runs), D threads. Each token loads its state
// column from `states` and stores it back exactly as the decode kernel does
// (so the next token reads what the tick would have read), stages and
// L2-normalizes q/k through the same tree, and folds the gated norm through
// the same epilogue. Every statement is the decode kernel's in its order; the
// only addition is a barrier closing each token so all D threads stay in step.
// gn_w == nullptr writes the plain recurrence output - the rollback replay's
// form, where only the state matters.
//
// f32 state and the compile-time-D arms only, like the fold it mirrors: every
// other geometry declines with -1 and the caller keeps its per-row launches.
template <uint32_t D>
__global__ __launch_bounds__(D) void pd_gated_delta_recurrent_runs_slots_kernel_t(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* __restrict__ states,
        float* __restrict__ out, const unsigned int* __restrict__ run_off,
        const unsigned int* __restrict__ run_len,
        const unsigned int* __restrict__ run_slot, uint32_t n_heads,
        const float* __restrict__ gn_z, const float* __restrict__ gn_w, float gn_eps) {
    const uint32_t h = blockIdx.x;
    const uint32_t r = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (h >= n_heads) return;
    const uint32_t off = run_off[r];
    const uint32_t len = run_len[r];
    if (len == 0) return;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    float* red  = smem + 2 * D;
    const float scale = rsqrtf((float)D);
    float* s_head = states + ((size_t)run_slot[r] * n_heads + h) * (size_t)D * D;

    for (uint32_t t = 0; t < len; ++t) {
        const uint32_t b = off + t;
        const size_t base = ((size_t)b * n_heads + h) * (size_t)D;
        const float qj = q[base + j];
        const float kj = k[base + j];
        const float vj = v[base + j];

        q_sh[j] = qj * qj;
        k_sh[j] = kj * kj;
        __syncthreads();
        #pragma unroll
        for (uint32_t s = D >> 1; s > 0; s >>= 1) {
            if (j < s) { q_sh[j] += q_sh[j + s]; k_sh[j] += k_sh[j + s]; }
            __syncthreads();
        }
        if (j == 0) { red[0] = rsqrtf(q_sh[0] + 1e-6f); red[1] = rsqrtf(k_sh[0] + 1e-6f); }
        __syncthreads();
        q_sh[j] = qj * red[0] * scale;
        k_sh[j] = kj * red[1];
        __syncthreads();

        const float g_t = expf(g[(size_t)b * n_heads + h]);
        const float beta_t = beta[(size_t)b * n_heads + h];

        float u = 0.0f;
        float col[D];
        #pragma unroll
        for (uint32_t i = 0; i < D; ++i) {
            col[i] = pd_dns_ld(s_head + (size_t)i * D + j) * g_t;
            u += col[i] * k_sh[i];
        }
        const float delta = beta_t * (vj - u);
        float o = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < D; ++i) {
            col[i] += k_sh[i] * delta;
            o += col[i] * q_sh[i];
            pd_dns_st(s_head + (size_t)i * D + j, col[i]);
        }
        if (gn_w == nullptr) {
            out[base + j] = o;
            __syncthreads();
            continue;
        }
        __syncthreads();
        q_sh[j] = o * o;
        __syncthreads();
        #pragma unroll
        for (uint32_t s = D >> 1; s > 0; s >>= 1) {
            if (j < s) q_sh[j] += q_sh[j + s];
            __syncthreads();
        }
        const float gn_inv = 1.0f / sqrtf(q_sh[0] / (float)D + gn_eps);
        const float y = gn_w[j] * (o * gn_inv) *
                        (1.0f / (1.0f + expf(-gn_z[base + j])));
        out[base + j] = y;
        __syncthreads();
    }
}

PD_EXPORT
int pd_gated_delta_recurrent_runs_slots(const void* q, const void* k, const void* v,
                                        const void* g, const void* beta, void* states,
                                        void* out, const void* run_off,
                                        const void* run_len, const void* run_slot,
                                        const void* gn_z, const void* gn_w, float gn_eps,
                                        uint32_t n_runs, uint32_t n_heads,
                                        uint32_t head_dim, void* stream) {
    if (n_runs == 0 || n_heads == 0 || head_dim == 0) return 0;
    // decline wherever the decode tick would not have run the compile-time-D
    // f32 body this mirrors: a narrow state class, a runtime-D geometry, the
    // generic-kernel pin, or (with the norm) slot 564's own kill switch -
    // in each of those the tick took another route, so this must not stand in
    static const bool gn_off = [] {
        const char* e = pd_env("PADDOCK_Q38FN_GDN_GN");
        return e && e[0] == '0';
    }();
    const int cls = pd_dns_state_class();
    if (cls == 1 || cls == 2 || cls == 3 || pd_dn_slots_generic_env() ||
        !(head_dim == 128u || head_dim == 64u) || (gn_w != nullptr && gn_off)) {
        return -1;
    }
    const size_t shmem = ((size_t)2 * head_dim + 2) * sizeof(float);
    const dim3 grid(n_heads, n_runs);
#define PD_DN_RUNS_SLOTS_T(DD)                                                 \
    do {                                                                       \
        pd_gated_delta_recurrent_runs_slots_kernel_t<DD>                       \
            <<<grid, (DD), shmem, (cudaStream_t)stream>>>(                     \
                (const float*)q, (const float*)k, (const float*)v,             \
                (const float*)g, (const float*)beta, (float*)states,           \
                (float*)out, (const unsigned int*)run_off,                     \
                (const unsigned int*)run_len, (const unsigned int*)run_slot,   \
                n_heads, (const float*)gn_z, (const float*)gn_w, gn_eps);      \
        return pd_launch_status();                                             \
    } while (0)
    if (head_dim == 128u) PD_DN_RUNS_SLOTS_T(128u);
    PD_DN_RUNS_SLOTS_T(64u);
#undef PD_DN_RUNS_SLOTS_T
}
