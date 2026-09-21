// Original GPU-resident MXFP4 MoE. GGUF's block-32 E2M1/UE8M0 bytes stay
// resident, with no model expansion or host routing. Decode uses SIMD GEMV;
// prefill compacts expert assignments and reuses each weight tile across 16/32
// routed rows using Apple TensorOps. Native scaled-FP4 tensors require macOS
// 27; this macOS 26 graph stages only a bounded contraction tile in F16.
constant float moe_e2m1[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
inline float moe_scale(uint e) {
    // UE8M0 is already a biased FP32 exponent. Preserve its subnormal zero
    // exponent and NaN code; ordinary blocks need just a bitcast, not ldexp
    // range handling for every one of their 32 reconstructed elements.
    return as_type<float>(e==0 ? 0x00400000u : e==255 ? 0x7fc00000u : e<<23);
}
inline float moe_mx(device const uchar* w, ulong i) {
    device const uchar* b=w+(i/32)*17;
    uint j=uint(i%32),code=(b[1+j%16]>>((j/16)*4))&15;
    return moe_e2m1[code]*moe_scale(b[0]);
}
inline float4 moe_codes(uint4 code) {return float4(moe_e2m1[code.x],moe_e2m1[code.y],moe_e2m1[code.z],moe_e2m1[code.w]);}

// Router scores include bias before selecting the four winners. Softmax is
// over the selected logits, not the whole expert population. Ties choose
// the lower expert index, so compaction/replay is deterministic.
kernel void moe_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device uint* ids [[buffer(2)]], device float* weights [[buffer(3)]],
                      constant uint* p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    float score[4];uint expert[4];
    for(uint i=0;i<4;++i){expert[i]=lane+32*i;score[i]=expert[i]<p[0] ? logits[row*p[0]+expert[i]]+bias[expert[i]] : -INFINITY;}
    float selected[4];uint chosen[4];
    for(uint pick=0;pick<4;++pick) {
        float best=-INFINITY;for(uint i=0;i<4;++i)best=max(best,score[i]);best=simd_max(best);
        uint id=UINT_MAX;for(uint i=0;i<4;++i)if(expert[i]<p[0] && score[i]==best)id=min(id,expert[i]);id=simd_min(id);
        // Nonfinite input must never turn a missing winner into an out-of-
        // range expert memory access. Keep the invalid softmax nonfinite;
        // this is a bounds guard, not a replacement prediction.
        if(id==UINT_MAX)id=0;
        chosen[pick]=id;selected[pick]=best;
        for(uint i=0;i<4;++i)if(expert[i]==id)score[i]=-INFINITY;
    }
    float sum=0,maximum=selected[0];for(uint i=0;i<4;++i){selected[i]=exp(selected[i]-maximum);sum+=selected[i];}
    if(lane==0)for(uint i=0;i<4;++i){ids[row*4+i]=chosen[i];weights[row*4+i]=selected[i]/sum;}
}

// Stable per-expert compaction. Each 256-assignment wave uses SIMD prefix
// sums plus an eight-lane scan; no floating atomics or CPU count readback.
// lists has experts * (rows*4) entries. Counts drive a compact tile schedule.
kernel void moe_align(device const uint* ids [[buffer(0)]], device uint* lists [[buffer(1)]],
                      device uint* counts [[buffer(2)]], constant uint* p [[buffer(3)]],
                      uint expert [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sums[8],offsets[8],carry;
    if(tid==0)carry=0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint base=0;base<p[0];base+=256) {
        uint i=base+tid,hit=i<p[0] && ids[i]==expert;
        uint before=simd_prefix_exclusive_sum(hit),sum=simd_sum(hit);
        if(lane==0)sums[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0) {
            uint n=lane<8 ? sums[lane] : 0;
            uint off=simd_prefix_exclusive_sum(n);
            if(lane<8)offsets[lane]=off;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(hit)lists[expert*p[0]+carry+offsets[sg]+before]=i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0)carry+=offsets[7]+sums[7];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0)counts[expert]=carry;
}

// One bounded scan over <=256 experts. The following fixed upper-bound
// dispatch returns before any tensor work for tiles outside this schedule.
kernel void moe_tiles(device const uint* counts [[buffer(0)]], device uint* tiles [[buffer(1)]],
                      constant uint* p [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint prefix[257];
    if(tid==0) {
        prefix[0]=0;
        for(uint e=0;e<p[0];++e)prefix[e+1]=prefix[e]+(counts[e]+p[1]-1)/p[1];
        tiles[0]=prefix[p[0]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<p[0])for(uint t=prefix[tid];t<prefix[tid+1];++t){tiles[1+2*t]=tid;tiles[2+2*t]=(t-prefix[tid])*p[1];}
}

// p: K, N, rows. Gate/up share input, routing, one launch, and one SIMD
// walk. A distinct down launch consumes the clipped SwiGLU activations.
kernel void moe_gu_decode(device const uchar* gate [[buffer(0)]], device const uchar* up [[buffer(1)]],
                          device const float* x [[buffer(2)]], device const uint* ids [[buffer(3)]],
                          device float* out [[buffer(4)]], constant uint* p [[buffer(5)]],
                          uint2 g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;
    ulong wbase=(ulong(ids[entry])*p[1]+n)*p[0];float4 ga=0,ua=0;
    for(uint k=lane*32;k<p[0];k+=1024) {
        device const uchar* gb=gate+(wbase+k)/32*17;device const uchar* ub=up+(wbase+k)/32*17;
        float gs=moe_scale(gb[0]),us=moe_scale(ub[0]);
        for(uint j=0;j<16;j+=4) {
            uint4 gq=uint4(*reinterpret_cast<device const packed_uchar4*>(gb+1+j));
            uint4 uq=uint4(*reinterpret_cast<device const packed_uchar4*>(ub+1+j));
            float4 lo=*reinterpret_cast<device const float4*>(x+ulong(entry/4)*p[0]+k+j);
            float4 hi=*reinterpret_cast<device const float4*>(x+ulong(entry/4)*p[0]+k+j+16);
            ga=fma(moe_codes(gq&15)*gs,lo,ga);ga=fma(moe_codes(gq>>4)*gs,hi,ga);
            ua=fma(moe_codes(uq&15)*us,lo,ua);ua=fma(moe_codes(uq>>4)*us,hi,ua);
        }
    }
    float a=simd_sum(ga.x+ga.y+ga.z+ga.w),b=simd_sum(ua.x+ua.y+ua.z+ua.w);
    if(lane==0){out[ulong(entry)*p[1]*2+n]=a;out[ulong(entry)*p[1]*2+p[1]+n]=b;}
}
kernel void moe_down_decode(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                            device const uint* ids [[buffer(2)]], device float* out [[buffer(3)]],
                            constant uint* p [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;
    ulong wbase=(ulong(ids[entry])*p[1]+n)*p[0];float4 acc=0;
    for(uint k=lane*32;k<p[0];k+=1024) {
        device const uchar* b=w+(wbase+k)/32*17;float scale=moe_scale(b[0]);
        for(uint j=0;j<16;j+=4) {
            uint4 q=uint4(*reinterpret_cast<device const packed_uchar4*>(b+1+j));
            acc=fma(moe_codes(q&15)*scale,*reinterpret_cast<device const float4*>(x+ulong(entry)*p[0]*2+k+j),acc);
            acc=fma(moe_codes(q>>4)*scale,*reinterpret_cast<device const float4*>(x+ulong(entry)*p[0]*2+k+j+16),acc);
        }
    }
    float sum=simd_sum(acc.x+acc.y+acc.z+acc.w);if(lane==0)out[ulong(entry)*p[1]+n]=sum;
}

// Original expert-grouped W4A16 GEMM (BM16/32, BN64, BK64). Sorted ids gather activations directly
// into a small threadgroup tile and scatter the result to its routed entry.
// Inactive experts never read weight bytes. Gate/up are two domains of the
// same grid; down uses the same contraction with different input strides.
template<uint BM,bool Down,uint BN=64,uint BK=64>
inline void moe_grouped(device const uchar* wg, device const uchar* wu, device const float* x,
                        device const uint* lists, device const uint* counts, device const uint* tiles,
                        device float* out, constant uint* p, uint2 g, uint tid,
                        threadgroup half* act, threadgroup half* w) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[expert]-first);
    // Gate/up domains are tile-count based: a partial final N tile must not
    // shift the first up column or leave its leading columns unwritten.
    uint N=p[1],ntiles=(N+BN-1)/BN,plane=Down ? 0 : g.x/ntiles;
    uint n=(g.x%ntiles)*BN;
    device const uchar* source=plane ? wu : wg;
    auto a=tensor(act,extents<int,BK,BM>(),array<int,2>{1,BK});
    auto b=tensor(w,extents<int,BK,BN>(),array<int,2>{1,BK});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=BK) {
        for(uint i=tid;i<BM*BK;i+=128) {
            uint r=i/BK,k=base+i%BK;
            uint entry=r<count ? lists[expert*p[2]*4+first+r] : 0;
            act[i]=r<count && k<p[0] ? half(x[Down ? ulong(entry)*p[0]*2+k : ulong(entry/4)*p[0]+k]) : half(0);
        }
        for(uint i=tid;i<BN*BK;i+=128) {
            uint col=n+i/BK,k=base+i%BK;
            w[i]=col<N && k<p[0] ? half(moe_mx(source,(ulong(expert)*N+col)*p[0]+k)) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(a,b,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && ij[1]<count && n+ij[0]<N) {
            uint entry=lists[expert*p[2]*4+first+ij[1]];
            out[ulong(entry)*N*(Down?1:2)+plane*N+n+ij[0]]=*it;
        }
    }
}
#define MOE_GROUPED(NAME,BM,DOWN) \
kernel void NAME(device const uchar* wg [[buffer(0)]],device const uchar* wu [[buffer(1)]], \
                 device const float* x [[buffer(2)]],device const uint* lists [[buffer(3)]], \
                 device const uint* counts [[buffer(4)]],device const uint* tiles [[buffer(5)]], \
                 device float* out [[buffer(6)]],constant uint* p [[buffer(7)]], \
                 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half act[BM*64],w[64*64]; \
    moe_grouped<BM,DOWN>(wg,wu,x,lists,counts,tiles,out,p,g,tid,act,w); \
}
MOE_GROUPED(moe_gu_grouped,16,false)
MOE_GROUPED(moe_down_grouped,16,true)
MOE_GROUPED(moe_gu_grouped32,32,false)
MOE_GROUPED(moe_down_grouped32,32,true)
#undef MOE_GROUPED

kernel void moe_swiglu(device float* gu [[buffer(0)]],device const float* gb [[buffer(1)]],
                       device const float* ub [[buffer(2)]],device const uint* ids [[buffer(3)]],
                       constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]*4)return;
    uint entry=i/p[0],n=i%p[0];ulong b=ulong(ids[entry])*p[0]+n,o=ulong(entry)*p[0]*2+n;
    float gate=min(gu[o]+gb[b],7.0f),up=clamp(gu[o+p[0]]+ub[b],-7.0f,7.0f);
    gu[o]=gate/(1+exp(-1.702f*gate))*(up+1);
}
kernel void moe_fold(device const float* out [[buffer(0)]],device const float* bias [[buffer(1)]],
                     device const uint* ids [[buffer(2)]],device const float* weights [[buffer(3)]],
                     device float* x [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;uint row=i/p[0],n=i%p[0];float sum=0;
    for(uint pick=0;pick<4;++pick){uint entry=row*4+pick;sum+=(out[ulong(entry)*p[0]+n]+bias[ulong(ids[entry])*p[0]+n])*weights[entry];}
    x[i]+=sum;
}
