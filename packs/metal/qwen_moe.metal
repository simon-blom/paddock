// Original Qwen35MoE execution: Q8_0 experts, top-8 of 256, ordinary SwiGLU
// and sigmoid-gated shared expert. No weight expansion or host route readback.
// Algorithm references: Qwen's published model/config, Paddock's CUDA graph,
// Apple's TensorOps guide. All indexing, routing and kernels are in-house.
kernel void qmoe_route(device const float* logits [[buffer(0)]], device uint* ids [[buffer(1)]],
                       device float* weights [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    float score[8];for(uint j=0;j<8;++j)score[j]=logits[row*256+lane+j*32];
    float selected[8];uint chosen[8];
    for(uint pick=0;pick<8;++pick) {
        float best=-INFINITY;for(uint j=0;j<8;++j)best=max(best,score[j]);best=simd_max(best);
        uint id=UINT_MAX;for(uint j=0;j<8;++j)if(score[j]==best)id=min(id,lane+j*32);id=simd_min(id);
        if(id==UINT_MAX)id=0; // bound invalid data, never manufacture a finite prediction
        selected[pick]=best;chosen[pick]=id;
        for(uint j=0;j<8;++j)if(lane+j*32==id)score[j]=-INFINITY;
    }
    // Full softmax -> top-k -> renormalization is algebraically a selected
    // softmax. Keep FP32 here and a fixed pick order through the final fold.
    float sum=0,maximum=selected[0];for(uint j=0;j<8;++j){selected[j]=exp(selected[j]-maximum);sum+=selected[j];}
    if(lane==0)for(uint j=0;j<8;++j){ids[row*8+j]=chosen[j];weights[row*8+j]=selected[j]/sum;}
}

inline float qmoe_q8(device const uchar* w, ulong i) {
    device const uchar* b=w+i/32*34;
    return float(*reinterpret_cast<device const half*>(b))*float(as_type<char>(b[2+i%32]));
}
// One SIMD owns one output column; gate/up share each activation block.
kernel void qmoe_gu_decode(device const uchar* gate [[buffer(0)]], device const uchar* up [[buffer(1)]],
    device const float* x [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]], uint2 g [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;
    ulong wbase=(ulong(ids[entry])*p[1]+n)*p[0];float4 ga=0,ua=0;
    for(uint k=lane*32;k<p[0];k+=1024) {
        device const uchar* gb=gate+(wbase+k)/32*34;device const uchar* ub=up+(wbase+k)/32*34;
        float gs=float(*reinterpret_cast<device const half*>(gb)),us=float(*reinterpret_cast<device const half*>(ub));
        for(uint j=0;j<32;j+=4) {
            float4 v=*reinterpret_cast<device const float4*>(x+ulong(entry/8)*p[0]+k+j);
            ga=fma(float4(*reinterpret_cast<device const packed_char4*>(gb+2+j))*gs,v,ga);
            ua=fma(float4(*reinterpret_cast<device const packed_char4*>(ub+2+j))*us,v,ua);
        }
    }
    float a=simd_sum(ga.x+ga.y+ga.z+ga.w),b=simd_sum(ua.x+ua.y+ua.z+ua.w);
    if(lane==0){out[ulong(entry)*p[1]*2+n]=a;out[ulong(entry)*p[1]*2+p[1]+n]=b;}
}
kernel void qmoe_down_decode(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]], device float* out [[buffer(3)]], constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;
    ulong wbase=(ulong(ids[entry])*p[1]+n)*p[0];float4 acc=0;
    for(uint k=lane*32;k<p[0];k+=1024) {
        device const uchar* b=w+(wbase+k)/32*34;float scale=float(*reinterpret_cast<device const half*>(b));
        for(uint j=0;j<32;j+=4)acc=fma(float4(*reinterpret_cast<device const packed_char4*>(b+2+j))*scale,
            *reinterpret_cast<device const float4*>(x+ulong(entry)*p[0]*2+k+j),acc);
    }
    float sum=simd_sum(acc.x+acc.y+acc.z+acc.w);if(lane==0)out[ulong(entry)*p[1]+n]=sum;
}

// Compact expert-major tiles reuse every Q8 tile over 16/32 routed rows.
// Only bounded contraction tiles are expanded, with F32 accumulation. The
// elected MoE path uses F32 operands; half variants remain GPU diagnostics.
template<uint BM,bool Down,typename T=half,uint BN=64,bool Fused=false,uint Active=8>
inline void qmoe_grouped(device const uchar* wg,device const uchar* wu,device const float* x,
    device const uint* lists,device const uint* counts,device const uint* tiles,device float* out,
    constant uint* p,uint2 g,uint tid,threadgroup T* act,threadgroup T* w) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[expert]-first);
    uint N=p[1],ntiles=(N+BN-1)/BN,plane=Down ? 0 : g.x/ntiles,n=(g.x%ntiles)*BN;
    device const uchar* source=plane ? wu : wg;
    auto a=tensor(act,extents<int,64,BM>(),array<int,2>{1,64});
    auto b=tensor(w,extents<int,64,BN>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(BM,BN,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64) {
        for(uint i=tid;i<BM*64;i+=128) {
            uint r=i/64,k=base+i%64,entry=r<count ? lists[expert*p[2]*Active+first+r] : 0;
            act[i]=r<count && k<p[0] ? T(x[Down ? ulong(entry)*p[0]*2+k : ulong(entry/Active)*p[0]+k]) : T(0);
        }
        for(uint i=tid;i<BN*64;i+=128) {
            uint col=n+i/64,k=base+i%64;
            ulong column=ulong(expert)*N*(Fused?2:1)+(Fused?plane*N:0)+col;
            w[i]=col<N && k<p[0] ? T(qmoe_q8(source,column*p[0]+k)) : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(a,b,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && ij[1]<count && n+ij[0]<N) {
            uint entry=lists[expert*p[2]*Active+first+ij[1]];
            out[ulong(entry)*N*(Down?1:2)+plane*N+n+ij[0]]=*it;
        }
    }
}
#define QMOE_GROUPED(NAME,BM,DOWN,T,BN) \
kernel void NAME(device const uchar* wg [[buffer(0)]],device const uchar* wu [[buffer(1)]], \
 device const float* x [[buffer(2)]],device const uint* lists [[buffer(3)]],device const uint* counts [[buffer(4)]], \
 device const uint* tiles [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]], \
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup T act[BM*64],w[BN*64];qmoe_grouped<BM,DOWN,T,BN>(wg,wu,x,lists,counts,tiles,out,p,g,tid,act,w); }
QMOE_GROUPED(qmoe_gu_grouped16,16,false,half,64)
QMOE_GROUPED(qmoe_gu_grouped32,32,false,half,64)
QMOE_GROUPED(qmoe_down_grouped16,16,true,half,64)
QMOE_GROUPED(qmoe_down_grouped32,32,true,half,64)
QMOE_GROUPED(qmoe_gu_strict16,16,false,float,32)
QMOE_GROUPED(qmoe_gu_strict32,32,false,float,32)
QMOE_GROUPED(qmoe_down_strict16,16,true,float,32)
QMOE_GROUPED(qmoe_down_strict32,32,true,float,32)
#undef QMOE_GROUPED

kernel void qmoe_swiglu(device float* gu [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;uint entry=i/p[0],n=i%p[0];ulong o=ulong(entry)*p[0]*2+n;
    float gate=gu[o];gu[o]=gate/(1+exp(-gate))*gu[o+p[0]];
}
kernel void qmoe_fold(device const float* out [[buffer(0)]],device const float* weights [[buffer(1)]],
    device const float* shared_gate [[buffer(2)]],device float* delta [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;uint row=i/p[0],n=i%p[0];float sum=0;
    for(uint j=0;j<8;++j)sum+=out[(ulong(row)*8+j)*p[0]+n]*weights[row*8+j];
    delta[i]=sum+delta[i]/(1+exp(-shared_gate[row]));
}
