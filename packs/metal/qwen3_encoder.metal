// Native Qwen3 encoder operations. The dense contractions reuse Paddock's
// quantized TensorOps path and tiled online-softmax hd128 attention. These
// epilogues preserve F32 RMS accumulation, full split-half rotary, last-token
// pooling and the checkpoint's two-token relevance head entirely on the GPU.
// No padding tokens participate: meta is (sequence, position) for real rows.
kernel void qwen3_head_rope(device const float* x [[buffer(0)]],
    device const uchar* norm [[buffer(1)]], device const uint* meta [[buffer(2)]],
    device half* out [[buffer(3)]], device const float* value [[buffer(4)]],
    device half* values [[buffer(5)]], constant uint* p [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    // p: heads, norm type, rope base, epsilon, store V flag.
    ulong src=(ulong(g.y)*p[0]+g.x)*128;
    uint pos=meta[2*g.y+1];
    ulong dst=src;
    float sum=0;
    for(uint d=lane;d<128;d+=32)sum+=x[src+d]*x[src+d];
    float inv=rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[3]));
    for(uint d=lane;d<128;d+=32) {
        uint j=d%64,other=d<64 ? d+64 : d-64;
        float angle=float(pos)*pow(as_type<float>(p[2]),-float(j)/64.0f);
        float a=x[src+d]*inv*weight(norm,p[1],d);
        float b=x[src+other]*inv*weight(norm,p[1],other);
        out[dst+d]=half(a*cos(angle)+(d<64 ? -b : b)*sin(angle));
        if(p[4])values[dst+d]=half(value[src+d]);
    }
}

// Packed (sequence-start, position) addressing avoids per-sequence KV padding.
// The same original hd128/BK32 online softmax as Granite; no KV history or
// allocation per layer is needed for a prefill-only encoder.
kernel void qwen3_attention(device half* q [[buffer(0)]], device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]], device const uint* meta [[buffer(3)]],
    device const uint* unused [[buffer(4)]], device float* out [[buffer(5)]],
    device const uint* tiles [[buffer(6)]], constant uint* p [[buffer(7)]],
    uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[32*128], probability[32*32];
    threadgroup float scores[32*32], maximum[32], denominator[32], correction[32];
    granite_prefill_tile<128,32,true>(q,k,v,meta,unused,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,
        kv,probability,scores,maximum,denominator,correction);
}

// Select and final-normalize only the rows that will leave the model. For
// embeddings, L2 is applied after the learned final RMS weight (not before).
// p: width, weight type, epsilon, L2 flag.
kernel void qwen3_pool(device const float* x [[buffer(0)]],
    device const uchar* norm [[buffer(1)]], device const uint* last [[buffer(2)]],
    device float* out [[buffer(3)]], constant uint* p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    uint n=p[0];ulong src=ulong(last[row])*n;
    float sum=0;
    for(uint d=tid;d<n;d+=256) { float v=x[src+d];sum+=v*v; }
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=rsqrt(simd_sum(lane<8 ? sums[lane] : 0.0f)/float(n)+as_type<float>(p[2]));
    // Every SIMD must read the RMS reduction before any replaces it with L2.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=0;
    for(uint d=tid;d<n;d+=256) {
        float v=x[src+d]*inv*weight(norm,p[1],d);
        out[ulong(row)*n+d]=v;sum+=v*v;
    }
    if(p[3]) {
        sum=simd_sum(sum);if(lane==0)sums[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
        float l2=1.0f/max(sqrt(simd_sum(lane<8 ? sums[lane] : 0.0f)),1e-12f);
        for(uint d=tid;d<n;d+=256)out[ulong(row)*n+d]*=l2;
    }
}

// Only two head rows are addressed, even when the GGUF stores a full vocabulary.
// p: width, head type, yes token, no token.
kernel void qwen3_score(device const float* pooled [[buffer(0)]],
    device const uchar* head [[buffer(1)]], device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 sums[8];float2 v=0;
    for(uint d=tid;d<p[0];d+=256) {
        float x=pooled[ulong(row)*p[0]+d];
        v.x+=x*weight(head,p[1],ulong(p[2])*p[0]+d);
        v.y+=x*weight(head,p[1],ulong(p[3])*p[0]+d);
    }
    v=simd_sum(v);if(lane==0)sums[sg]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v=simd_sum(lane<8 ? sums[lane] : float2(0));
    if(tid==0)out[row]=1.0f/(1.0f+exp(v.y-v.x));
}
