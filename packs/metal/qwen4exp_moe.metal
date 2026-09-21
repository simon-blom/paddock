// Original Flash Next MoE composition. Retain F32 activations: routing and
// near-zero SiLU values must not gain a half boundary when prefill joins.
// Expert-major tiles reuse packed weights; sparse tiles use SIMD directly.
// No host routing readback, atomic float accumulation or expanded weight slab.

kernel void q4m_router_mm(device float* w [[buffer(0)]],device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<2560;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=g.y*32+i/32,k=base+i%32,n=g.x*32+i/32;
            a[i]=r<p[0] ? x[ulong(r)*2560+k] : 0;
            b[i]=n<513 ? w[ulong(n)*2560+k] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint r=g.y*32+ij[1],n=g.x*32+ij[0];
        if(it.is_valid_element() && r<p[0] && n<513)y[ulong(r)*513+n]=*it;
    }
}

template<uint Ty>
inline void q4m_gu_dot(device const uchar* gate,device const uchar* up,device const float* x,
    device float* act,uint entry,uint expert,uint col,uint lane) {
    float4 ga=0,ua=0;
    if(expert<512)for(uint k=lane*4;k<2560;k+=128) {
        float4 v=*reinterpret_cast<device const float4*>(x+ulong(entry/10)*2560+k);
        ulong at=(ulong(expert)*640+col)*2560+k;
        ga=fma(iq_values4<Ty>(gate,at),v,ga);ua=fma(iq_values4<Ty>(up,at),v,ua);
    }
    float gv=simd_sum(ga.x+ga.y+ga.z+ga.w),uv=simd_sum(ua.x+ua.y+ua.z+ua.w);
    if(lane==0)act[ulong(entry)*640+col]=expert<512 ? (gv/(1+exp(-gv)))*uv : NAN;
}

#define Q4M_GU_MV(NAME,TY) \
kernel void NAME(device const uchar* gate [[buffer(0)]],device const uchar* up [[buffer(1)]], \
 device const float* x [[buffer(2)]],device const uint* ids [[buffer(3)]],device float* act [[buffer(4)]], \
 constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]], \
 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
 uint col=g.x*4+sg;if(g.y<p[0]*10 && col<640)q4m_gu_dot<TY>(gate,up,x,act,g.y,ids[g.y],col,lane);}
Q4M_GU_MV(q4m_gu_mv21,21) Q4M_GU_MV(q4m_gu_mv22,22)
#undef Q4M_GU_MV

// The dynamic-array reuse attempt regressed. Compile-time row counts let
// the compiler scalarize these accumulators; measure rather than assume reuse wins.
template<uint Ty,uint R>
inline void q4m_gu_small(device const uchar* gate,device const uchar* up,device const float* x,
    device const uint* list,device float* act,uint expert,uint entries,uint first,uint col,uint lane) {
    float4 ga[R],ua[R];
    #pragma unroll
    for(uint r=0;r<R;++r){ga[r]=0;ua[r]=0;}
    for(uint k=lane*4;k<2560;k+=128) {
        ulong at=(ulong(expert)*640+col)*2560+k;
        float4 gw=iq_values4<Ty>(gate,at),uw=iq_values4<Ty>(up,at);
        #pragma unroll
        for(uint r=0;r<R;++r) {
            uint entry=list[first+r];
            float4 v=entry<entries ? *reinterpret_cast<device const float4*>(x+ulong(entry/10)*2560+k) : float4(0);
            ga[r]=fma(gw,v,ga[r]);ua[r]=fma(uw,v,ua[r]);
        }
    }
    #pragma unroll
    for(uint r=0;r<R;++r) {
        uint entry=list[first+r];
        float gv=simd_sum(ga[r].x+ga[r].y+ga[r].z+ga[r].w),uv=simd_sum(ua[r].x+ua[r].y+ua[r].z+ua[r].w);
        if(lane==0 && entry<entries)act[ulong(entry)*640+col]=(gv/(1+exp(-gv)))*uv;
    }
}

// Two accumulators share each activation tile. Reusing B between contractions
// needs 8 KiB explicit staging, including the last ragged expert tile. The
// pipeline test also checks compiler/validation overhead against the HW limit.
template<uint Ty>
inline void q4m_gu_grouped(device const uchar* gate,device const uchar* up,device const float* x,
    device const uint* lists,device const uint* counts,device const uint* tiles,device float* act,
    constant uint* p,uint2 g,uint tid,threadgroup float* a,threadgroup float* b) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],entries=p[0]*10;
    if(expert>=512 || first>=counts[expert])return;
    uint count=min(32u,counts[expert]-first);
    // All threads take the same branch. Forced tensor mode is diagnostic-only
    // on the Rust side; production uses actual tile occupancy, not batch size.
    if(count<=4 && p[1]==0) {
        for(uint c=tid/32;c<32;c+=4) {
            device const uint* list=lists+expert*entries;uint col=g.x*32+c,lane=tid%32;
            if(count==1)q4m_gu_small<Ty,1>(gate,up,x,list,act,expert,entries,first,col,lane);
            else if(count==2)q4m_gu_small<Ty,2>(gate,up,x,list,act,expert,entries,first,col,lane);
            else if(count==3)q4m_gu_small<Ty,3>(gate,up,x,list,act,expert,entries,first,col,lane);
            else q4m_gu_small<Ty,4>(gate,up,x,list,act,expert,entries,first,col,lane);
        }
        return;
    }
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto ga=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    auto ua=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<ga.get_capacity();++i){ga[i]=0;ua[i]=0;}
    for(uint base=0;base<2560;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=i/32,entry=r<count ? lists[expert*entries+first+r] : entries;
            a[i]=entry<entries ? x[ulong(entry/10)*2560+base+i%32] : 0;
        }
        for(uint i=tid*4;i<1024;i+=512)
            *reinterpret_cast<threadgroup float4*>(b+i)=iq_values4<Ty>(gate,(ulong(expert)*640+g.x*32+i/32)*2560+base+i%32);
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,ga);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid*4;i<1024;i+=512)
            *reinterpret_cast<threadgroup float4*>(b+i)=iq_values4<Ty>(up,(ulong(expert)*640+g.x*32+i/32)*2560+base+i%32);
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,ua);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto ui=ua.begin();
    for(auto gi=ga.begin();gi!=ga.end();++gi,++ui) {
        auto ij=gi.get_multidimensional_index();
        if(gi.is_valid_element() && ij[1]<count) {
            uint entry=lists[expert*entries+first+ij[1]];
            if(entry<entries)act[ulong(entry)*640+g.x*32+ij[0]]=(*gi/(1+exp(-*gi)))*(*ui);
        }
    }
}
#define Q4M_GU_MM(NAME,TY) \
kernel void NAME(device const uchar* gate [[buffer(0)]],device const uchar* up [[buffer(1)]], \
 device const float* x [[buffer(2)]],device const uint* lists [[buffer(3)]],device const uint* counts [[buffer(4)]], \
 device const uint* tiles [[buffer(5)]],device float* act [[buffer(6)]],constant uint* p [[buffer(7)]], \
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float a[1024],b[1024];q4m_gu_grouped<TY>(gate,up,x,lists,counts,tiles,act,p,g,tid,a,b);}
Q4M_GU_MM(q4m_gu_mm21,21) Q4M_GU_MM(q4m_gu_mm22,22)
#undef Q4M_GU_MM

template<uint R>
inline void q4m_down_small(device const uchar* w,device const float* x,device const uint* list,
    device float* y,uint expert,uint entries,uint first,uint col,uint lane) {
    float4 acc[R];
    #pragma unroll
    for(uint r=0;r<R;++r)acc[r]=0;
    for(uint k=lane*4;k<640;k+=128) {
        float4 weight=iq_values4<20>(w,(ulong(expert)*2560+col)*640+k);
        #pragma unroll
        for(uint r=0;r<R;++r) {
            uint entry=list[first+r];
            float4 v=entry<entries ? *reinterpret_cast<device const float4*>(x+ulong(entry)*640+k) : float4(0);
            acc[r]=fma(weight,v,acc[r]);
        }
    }
    #pragma unroll
    for(uint r=0;r<R;++r) {
        uint entry=list[first+r];float sum=simd_sum(acc[r].x+acc[r].y+acc[r].z+acc[r].w);
        if(lane==0 && entry<entries)y[ulong(entry)*2560+col]=sum;
    }
}

kernel void q4m_down_grouped(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=p[4] || first>=counts[expert])return;
    uint count=min(32u,counts[expert]-first);
    if(count<=4 && p[5]==0) {
        for(uint c=tid/32;c<32;c+=4) {
            device const uint* list=lists+expert*p[2];uint col=g.x*32+c,lane=tid%32;
            if(count==1)q4m_down_small<1>(w,x,list,y,expert,p[2],first,col,lane);
            else if(count==2)q4m_down_small<2>(w,x,list,y,expert,p[2],first,col,lane);
            else if(count==3)q4m_down_small<3>(w,x,list,y,expert,p[2],first,col,lane);
            else q4m_down_small<4>(w,x,list,y,expert,p[2],first,col,lane);
        }
        return;
    }
    threadgroup float a[1024],b[1024];
    iq_expert_mm<20>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
}

kernel void q4m_silu(device const float* gate [[buffer(0)]],device const float* up [[buffer(1)]],
    device float* act [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float g=gate[i];act[i]=(g/(1+exp(-g)))*up[i];}
}

// Deterministic top-rank order, not an expert-completion-order atomic fold.
// Invalid router rows are explicitly NaN and flagged before any expert read.
kernel void q4m_fold(device const float* down [[buffer(0)]],device const float* weights [[buffer(1)]],
    device const float* shared [[buffer(2)]],device const float* gate [[buffer(3)]],device const uint* invalid [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*640)return;
    uint row=i/640,col=(i%640)*4;float4 sum=0;
    for(uint j=0;j<10;++j)sum+=*reinterpret_cast<device const float4*>(down+(ulong(row)*10+j)*2560+col)*weights[row*10+j];
    sum+=*reinterpret_cast<device const float4*>(shared+ulong(row)*2560+col)*gate[row];
    *reinterpret_cast<device float4*>(y+ulong(row)*2560+col)=invalid[row] ? float4(NAN) : sum;
}
