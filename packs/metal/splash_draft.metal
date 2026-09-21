// Packed DFlash2 has BF16 activation and KV boundaries. Do not reuse the
// GGUF draft's F16 caches or F32 residual stream for this package.
kernel void splash_df_rms(device const float* x [[buffer(0)]],device const float* norm [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    uint width=p[0];ulong base=ulong(row)*width;float sum=0;
    for(uint i=tid;i<width;i+=256)sum+=x[base+i]*x[base+i];
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {float total=0;for(uint i=0;i<8;++i)total+=sums[i];sums[0]=rsqrt(total/width+as_type<float>(p[2]));}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<width;i+=256)out[base+i]=float(bfloat(x[base+i]*sums[0]*norm[i]));
}

inline float splash_df_rotary(device const float* x,device const float* norm,ulong base,uint d,float inverse,float position,float theta) {
    float first=float(bfloat(x[base+d]*inverse*norm[d]));
    uint other=d^64;
    float second=float(bfloat(x[base+other]*inverse*norm[other]));
    float angle=position*pow(theta,-float(d%64)/64.0f);
    return float(bfloat(first*cos(angle)+(d<64?-1.0f:1.0f)*second*sin(angle)));
}

kernel void splash_df_qnorm(device const float* x [[buffer(0)]],device const float* norm [[buffer(1)]],
    device const uint* meta [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong base=(ulong(g.y)*p[0]+g.x)*128;float sum=0;
    for(uint d=lane;d<128;d+=32)sum+=x[base+d]*x[base+d];
    float inverse=rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[2]));
    for(uint d=lane;d<128;d+=32)out[base+d]=splash_df_rotary(x,norm,base,d,inverse,float(meta[g.y*4]),as_type<float>(p[3]));
}

kernel void splash_df_kstore(device const float* x [[buffer(0)]],device const float* v [[buffer(1)]],
    device const float* norm [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],
    device bfloat* keys [[buffer(5)]],device bfloat* values [[buffer(6)]],device const uint* mrope [[buffer(7)]],
    device const uint* bounds [[buffer(8)]],constant uint* p [[buffer(9)]],uint2 g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if(p[5]==0 && meta[g.y*2+1]+p[4]<=meta[(bounds[g.y*2+1]-1)*2+1])return;
    ulong base=(ulong(g.y)*p[0]+g.x)*128;float sum=0;
    for(uint d=lane;d<128;d+=32)sum+=x[base+d]*x[base+d];
    float inverse=rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[2]));
    uint slot=meta[g.y*2],pos=meta[g.y*2+1],physical=pages[slot*p[1]+pos/16]*16+pos%16;
    ulong dst=(ulong(physical)*p[0]+g.x)*128;
    for(uint d=lane;d<128;d+=32) {
        keys[dst+d]=bfloat(splash_df_rotary(x,norm,base,d,inverse,float(mrope[g.y*4]),as_type<float>(p[3])));
        values[dst+d]=bfloat(v[base+d]);
    }
}
