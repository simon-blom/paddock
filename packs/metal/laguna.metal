// Original Laguna kernels. Graph semantics studied in poolside's config
// and llama.cpp b10901; implementation reuses only Paddock's own primitives.
// Exact K-quants stay resident; bounded F32 TensorOps tiles are not a model
// conversion. No host routing or floating-point scatter atomics.

// Small bounded K panels retain F32 backbone operands without relying on
// the legacy BK256 Gemma projection (33,280 compiled TG bytes under shader
// validation, above M5's 32KiB limit). Decode remains register-reusing SIMD.
kernel void laguna_dense_f32(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint BK=64,BN=16,BM=32;
    uint K=p[0],N=p[1],M=p[2],n=g.x*BN,m=g.y*BM;
    threadgroup float weights[BN*(BK+4)];
    auto input=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+4});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    for(uint base=0;base<K;base+=BK){
        for(uint i=tid*4;i<BN*BK;i+=512){uint col=n+i/BK,k=base+i%BK;float4 value=0;
            if(col<N){ulong at=ulong(col)*K+k;
                if(p[3]==8){device const uchar* block=w+at/32*34;
                    value=float4(*reinterpret_cast<device const packed_char4*>(block+2+at%32))*float(*reinterpret_cast<device const half*>(block));}
                else value=kquant4(w,p[3],at);}
            *reinterpret_cast<threadgroup float4*>(weights+(i/BK)*(BK+4)+i%BK)=value;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=input.slice(base,m);op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    c.store(output.slice(n,m));
}

// p: heads, rotary dimensions, base, frequency scale, correction low/high,
// magnitude, epsilon. Each lane owns both halves of its rotary pair.
kernel void laguna_qnorm_rope(device float* q [[buffer(0)]],device const float* norm [[buffer(1)]],
    device const uint* meta [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong off=(ulong(g.y)*p[0]+g.x)*128;float sq=0;
    for(uint d=lane;d<128;d+=32)sq+=q[off+d]*q[off+d];
    float inv=rsqrt(simd_sum(sq)/128.0f+as_type<float>(p[7]));
    uint rot=p[1];
    for(uint j=lane;j<rot/2;j+=32){
        float a=q[off+j]*inv*norm[j],b=q[off+j+rot/2]*inv*norm[j+rot/2];
        float angle=float(meta[2*g.y+1])*pow(as_type<float>(p[2]),-2.0f*float(j)/float(rot));
        float ramp=clamp((float(j)-as_type<float>(p[4]))/max(0.001f,as_type<float>(p[5])-as_type<float>(p[4])),0.0f,1.0f);
        angle*=1.0f-ramp+ramp*as_type<float>(p[3]);
        float c=cos(angle)*as_type<float>(p[6]),s=sin(angle)*as_type<float>(p[6]);
        q[off+j]=a*c-b*s;q[off+j+rot/2]=a*s+b*c;
    }
    for(uint j=rot+lane;j<128;j+=32)q[off+j]=q[off+j]*inv*norm[j];
}
// Keys were normalized/rotated with the same kernel as queries. Values are
// not normalized. Full paged storage also backs sliding layers, preserving
// exact prefix resumes; a ring/checkpoint storage election is separate work.
kernel void laguna_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
    device half* kc [[buffer(2)]],device half* vc [[buffer(3)]],device const uint* meta [[buffer(4)]],
    device const uint* pages [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*1024)return;uint row=i/1024,pos=meta[2*row+1],slot=meta[2*row];
    ulong dst=(ulong(pages[slot*p[1]+pos/16])*16+pos%16)*1024+i%1024;
    kc[dst]=half(k[i]);vc[dst]=half(v[i]);
}
kernel void laguna_gate(device float* attn [[buffer(0)]],device const float* gate [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]*128)return;float g=gate[i/128];
    attn[i]*=max(g,0.0f)+log(1.0f+exp(-abs(g)));
}

// Existing panel-streamed attention, specialized for 128-d heads and GQA
// six/eight/nine. The scale is 1/sqrt(128), as in Muse, not Gemma's unit scale.
#define LAGUNA_DECODE(GQA) \
kernel void laguna_decode##GQA(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]], \
 device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]], \
 constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
 threadgroup float scores[GQA*32],prob[GQA*32],highs[GQA],sums[GQA]; \
 gemma_decode<128,GQA,true>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums); }
LAGUNA_DECODE(6)
LAGUNA_DECODE(8)
LAGUNA_DECODE(9)
#undef LAGUNA_DECODE
kernel void laguna_prefill(device float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
 device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],
 constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float kv[16*128],prob[16*32],scores[16*32],maximum[32],denom[32],correction[32];
 gemma_prefill<128,16,32,float,false,true>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction);
}

// Selection bias changes membership/order only. Fold weights use unmodified
// sigmoid probabilities, normalized over selected experts then scaled 2.5.
template<uint Active>
inline void laguna_route_impl(device const float* logits,device const float* bias,
 device uint* ids,device float* weights,constant uint* p,uint row,uint lane) {
    float score[8],prob[8];
    for(uint j=0;j<8;++j){uint e=lane+j*32;prob[j]=1.0f/(1.0f+exp(-logits[row*256+e]));score[j]=prob[j]+bias[e];}
    // Compile-time top-k keeps each checkpoint's reduction shape fixed.
    uint chosen[Active];float selected[Active],sum=0;
    for(uint pick=0;pick<Active;++pick){
        float best=-INFINITY;for(uint j=0;j<8;++j)best=max(best,score[j]);best=simd_max(best);
        uint id=UINT_MAX;for(uint j=0;j<8;++j)if(score[j]==best)id=min(id,lane+j*32);id=simd_min(id);
        if(id==UINT_MAX)id=0;
        float value=simd_broadcast(prob[id/32],id%32);
        chosen[pick]=id;selected[pick]=value;sum+=value;
        for(uint j=0;j<8;++j)if(lane+j*32==id)score[j]=-INFINITY;
    }
    if(lane==0)for(uint j=0;j<Active;++j){ids[row*Active+j]=chosen[j];weights[row*Active+j]=(selected[j]/sum)*2.5f;}
}
#define LAGUNA_ROUTE(NAME,ACTIVE) \
kernel void NAME(device const float* logits [[buffer(0)]],device const float* bias [[buffer(1)]], \
 device uint* ids [[buffer(2)]],device float* weights [[buffer(3)]],constant uint* p [[buffer(4)]], \
 uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) { \
 laguna_route_impl<ACTIVE>(logits,bias,ids,weights,p,row,lane); }
LAGUNA_ROUTE(laguna_route,8)
LAGUNA_ROUTE(laguna_route_top10,10)
#undef LAGUNA_ROUTE
template<uint Active>
inline void laguna_gu_impl(device const uchar* gate,device const uchar* up,
 device const float* x,device const uint* ids,device float* out,
 constant uint* p,uint2 g,uint sg,uint lane) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;
    ulong base=(ulong(ids[entry])*p[1]+n)*p[0];float4 ga=0,ua=0;
    for(uint k=lane*4;k<p[0];k+=128){float4 a=*reinterpret_cast<device const float4*>(x+ulong(entry/Active)*p[0]+k);
        ga+=kquant4(gate,p[3],base+k)*a;ua+=kquant4(up,p[4],base+k)*a;}
    float a=simd_sum(ga.x+ga.y+ga.z+ga.w),b=simd_sum(ua.x+ua.y+ua.z+ua.w);
    if(lane==0){out[ulong(entry)*p[1]*2+n]=a;out[ulong(entry)*p[1]*2+p[1]+n]=b;}
}
#define LAGUNA_GU(NAME,ACTIVE) \
kernel void NAME(device const uchar* gate [[buffer(0)]],device const uchar* up [[buffer(1)]], \
 device const float* x [[buffer(2)]],device const uint* ids [[buffer(3)]],device float* out [[buffer(4)]], \
 constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
 laguna_gu_impl<ACTIVE>(gate,up,x,ids,out,p,g,sg,lane); }
LAGUNA_GU(laguna_gu_decode,8)
LAGUNA_GU(laguna_gu_decode_top10,10)
#undef LAGUNA_GU
kernel void laguna_down_decode(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
 device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;ulong base=(ulong(ids[entry])*p[1]+n)*p[0];float4 a=0;
    for(uint k=lane*4;k<p[0];k+=128)a+=kquant4(w,p[3],base+k)**reinterpret_cast<device const float4*>(x+ulong(entry)*p[0]*2+k);
    float sum=simd_sum(a.x+a.y+a.z+a.w);if(lane==0)out[ulong(entry)*p[1]+n]=sum;
}
// Integer compaction (moe_align/moe_tiles) produces deterministic expert-
// major lists. Each SIMD group unpacks adjacent groups of four K weights;
// a BM16/32 tile shares these bytes across requests. Original TensorOps path.
template<uint BM,bool Down,uint Active>
inline void laguna_grouped(device const uchar* wg,device const uchar* wu,device const float* x,
 device const uint* lists,device const uint* counts,device const uint* tiles,device float* out,
 constant uint* p,uint2 g,uint tid,threadgroup float* act,threadgroup float* w) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[expert]-first);
    uint N=p[1],ntiles=(N+31)/32,plane=Down?0:g.x/ntiles,n=g.x%ntiles*32;
    device const uchar* source=plane?wu:wg;
    auto a=tensor(act,extents<int,64,BM>(),array<int,2>{1,64});
    auto b=tensor(w,extents<int,64,32>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(BM,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64){
        for(uint i=tid;i<BM*64;i+=128){uint r=i/64,k=base+i%64,entry=r<count?lists[expert*p[2]*Active+first+r]:0;
            act[i]=r<count?x[Down?ulong(entry)*p[0]*2+k:ulong(entry/Active)*p[0]+k]:0;}
        for(uint i=tid*4;i<32*64;i+=128*4){uint col=n+i/64,k=base+i%64;
            *reinterpret_cast<threadgroup float4*>(w+i)=col<N?kquant4(source,p[3+plane],(ulong(expert)*N+col)*p[0]+k):float4(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && ij[1]<count && n+ij[0]<N){uint entry=lists[expert*p[2]*Active+first+ij[1]];
            out[ulong(entry)*N*(Down?1:2)+plane*N+n+ij[0]]=*it;}}
}
#define LAGUNA_GROUP(NAME,BM,DOWN,ACTIVE) \
kernel void NAME(device const uchar* wg [[buffer(0)]],device const uchar* wu [[buffer(1)]],device const float* x [[buffer(2)]], \
 device const uint* lists [[buffer(3)]],device const uint* counts [[buffer(4)]],device const uint* tiles [[buffer(5)]], \
 device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float a[BM*64],w[32*64];laguna_grouped<BM,DOWN,ACTIVE>(wg,wu,x,lists,counts,tiles,out,p,g,tid,a,w); }
LAGUNA_GROUP(laguna_gu_grouped16,16,false,8)
LAGUNA_GROUP(laguna_gu_grouped16_top10,16,false,10)
LAGUNA_GROUP(laguna_gu_grouped32,32,false,8)
LAGUNA_GROUP(laguna_gu_grouped32_top10,32,false,10)
LAGUNA_GROUP(laguna_down_grouped16,16,true,8)
LAGUNA_GROUP(laguna_down_grouped16_top10,16,true,10)
LAGUNA_GROUP(laguna_down_grouped32,32,true,8)
LAGUNA_GROUP(laguna_down_grouped32_top10,32,true,10)
#undef LAGUNA_GROUP
template<uint Active>
inline void laguna_fold_impl(device const float* experts,device const float* weights,
 device float* shared,constant uint* p,uint i) {
    if(i>=p[0]*p[1])return;uint row=i/p[0],col=i%p[0];float sum=0;
    for(uint j=0;j<Active;++j)sum+=experts[(ulong(row)*Active+j)*p[0]+col]*weights[row*Active+j];
    shared[i]+=sum;
}
#define LAGUNA_FOLD(NAME,ACTIVE) \
kernel void NAME(device const float* experts [[buffer(0)]],device const float* weights [[buffer(1)]], \
 device float* shared [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) { \
 laguna_fold_impl<ACTIVE>(experts,weights,shared,p,i); }
LAGUNA_FOLD(laguna_fold,8)
LAGUNA_FOLD(laguna_fold_top10,10)
#undef LAGUNA_FOLD
