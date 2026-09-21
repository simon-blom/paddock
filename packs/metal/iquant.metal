// Original exact-format IQ2_S, IQ3_S and IQ4_NL projection kernels.
// Learnings: cache-sized codebooks (QuIP#), weight reuse over expert-major
// rows, and MPP bounded contraction tiles. No matrix expansion/requantization.
// These primitives are not full qwen4exp graph or serving qualification.

// Flash Next's 512-way, top-10 softmax router plus a distinct sigmoid shared
// expert. Preserve the published full-softmax -> select -> renormalize order.
// Stable lower-id ties and explicit nonfinite propagation are graph contracts,
// not sampling options. Invalid input may never become an expert OOB address.
kernel void iq_route512(device const float* logits [[buffer(0)]],device uint* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],device float* shared [[buffer(3)]],device uint* invalid [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(row>=p[0])return;
    float scores[16],maximum=-INFINITY;uint bad=0;
    for(uint j=0;j<16;++j){scores[j]=logits[ulong(row)*513+lane+j*32];maximum=max(maximum,scores[j]);bad+=!isfinite(scores[j]);}
    float gate=logits[ulong(row)*513+512];bad=simd_sum(bad)+uint(!isfinite(gate));
    if(bad) {
        if(lane==0){invalid[row]=1;shared[row]=NAN;for(uint j=0;j<10;++j){ids[row*10+j]=0;weights[row*10+j]=NAN;}}
        return;
    }
    maximum=simd_max(maximum);float denominator=0;
    for(uint j=0;j<16;++j)denominator+=exp(scores[j]-maximum);
    denominator=simd_sum(denominator);
    uint chosen[10];float selected[10];
    for(uint pick=0;pick<10;++pick) {
        float best=-INFINITY;for(uint j=0;j<16;++j)best=max(best,scores[j]);best=simd_max(best);
        uint id=UINT_MAX;for(uint j=0;j<16;++j)if(scores[j]==best)id=min(id,lane+j*32);id=simd_min(id);
        chosen[pick]=id;selected[pick]=exp(best-maximum)/denominator;
        for(uint j=0;j<16;++j)if(lane+j*32==id)scores[j]=-INFINITY;
    }
    float total=0;for(uint j=0;j<10;++j)total+=selected[j];
    if(lane==0) {
        invalid[row]=0;shared[row]=1/(1+exp(-gate));
        for(uint j=0;j<10;++j){ids[row*10+j]=chosen[j];weights[row*10+j]=selected[j]/total;}
    }
}

template<uint Ty>
inline float4 iq_values4(device const uchar* w, ulong i) {
    uint packed,signs;float scale;
    if constexpr(Ty==20) {
        device const uchar* b=w+(i/32)*18;uint j=uint(i%32);
        uint4 code=(uint4(*reinterpret_cast<device const packed_uchar4*>(b+2+j%16))>>((j/16)*4))&15;
        return float4(iq4_values[code.x],iq4_values[code.y],iq4_values[code.z],iq4_values[code.w])
            *float(*reinterpret_cast<device const half*>(b));
    } else if constexpr(Ty==22) {
        device const uchar* b=w+(i/256)*82;uint j=uint(i%256),group=j/8,part=j/32;
        uint idx=uint(b[2+group])|(((uint(b[66+part])>>(2*(group%4)))&3)<<8);
        packed=metal_IQ2S_GRID[2*idx+(j%8)/4];signs=uint(b[34+group])>>(j%8);
        scale=float(*reinterpret_cast<device const half*>(b))
            *(0.5f+float((uint(b[74+part])>>(4*((j%32)/16)))&15))*0.25f;
    } else {
        static_assert(Ty==21,"unsupported i-quant");
        device const uchar* b=w+(i/256)*110;uint j=uint(i%256),part=j/32;
        uint idx=uint(b[2+j/4])|(((uint(b[66+part])>>((j%32)/4))&1)<<8);
        packed=metal_IQ3S_GRID[idx];signs=uint(b[74+j/8])>>(j%8);
        scale=float(*reinterpret_cast<device const half*>(b))
            *float(1+2*((uint(b[106+j/64])>>(4*((j%64)/32)))&15));
    }
    uint4 lanes=uint4(0,1,2,3);
    return float4(int4((uint4(packed)>>(lanes*8))&255)*(1-2*int4((uint4(signs)>>lanes)&1)))*scale;
}

// Four adjacent weights per lane keeps all lanes busy at K=640; assigning
// whole 32-weight blocks would leave 12/32 lanes idle at that expert shape.
// R1/R4/R8 reuse each decoded vector without an activation rounding change.
template<uint Ty,uint Rows>
inline void iq_mv(device const uchar* w,device const float* x,device float* y,
    constant uint* p,uint2 g,uint sg,uint lane) {
    uint col=g.x*4+sg,first=g.y*Rows;if(col>=p[1])return;
    float4 acc[Rows];for(uint r=0;r<Rows;++r)acc[r]=0;
    for(uint k=lane*4;k<p[0];k+=128) {
        float4 value=iq_values4<Ty>(w,ulong(col)*p[0]+k);
        for(uint r=0;r<Rows;++r)if(first+r<p[2])
            acc[r]=fma(value,*reinterpret_cast<device const float4*>(x+ulong(first+r)*p[0]+k),acc[r]);
    }
    for(uint r=0;r<Rows;++r) {
        float sum=simd_sum(acc[r].x+acc[r].y+acc[r].z+acc[r].w);
        if(lane==0 && first+r<p[2])y[ulong(first+r)*p[1]+col]=sum*as_type<float>(p[4]);
    }
}
#define IQ_MV(NAME,TY,R) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]], \
 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {iq_mv<TY,R>(w,x,y,p,g,sg,lane);}
IQ_MV(iq_mv20_1,20,1) IQ_MV(iq_mv20_4,20,4) IQ_MV(iq_mv20_8,20,8)
IQ_MV(iq_mv21_1,21,1) IQ_MV(iq_mv21_4,21,4) IQ_MV(iq_mv21_8,21,8)
IQ_MV(iq_mv22_1,22,1) IQ_MV(iq_mv22_4,22,4) IQ_MV(iq_mv22_8,22,8)
#undef IQ_MV

// F32 weight/activation tiles preserve format values and MoE routing margins.
// Prepared=true consumes the shared padded F16 input contract explicitly;
// the direct path never silently takes that rounding boundary at batch 9.
template<uint Ty,typename Input,bool Prepared>
inline void iq_mm(device const uchar* w,device const Input* x,device float* y,
    constant uint* p,uint2 g,uint tid,threadgroup float* a,threadgroup float* b) {
    uint n=g.x*32,row=g.y*32,stride=Prepared ? ((p[0]+127)/128)*128 : p[0];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=row+i/32,k=base+i%32;
            a[i]=r<p[2] ? float(x[ulong(r)*stride+k]) : 0;
        }
        for(uint i=tid*4;i<1024;i+=512) {
            uint col=n+i/32,k=base+i%32;
            *reinterpret_cast<threadgroup float4*>(b+i)=col<p[1] ? iq_values4<Ty>(w,ulong(col)*p[0]+k) : float4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && row+ij[1]<p[2] && n+ij[0]<p[1])
            y[ulong(row+ij[1])*p[1]+n+ij[0]]=*it*as_type<float>(p[4]);
    }
}
#define IQ_MM(NAME,TY,T,PREP) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const T* x [[buffer(1)]], \
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]], \
 uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float a[1024],b[1024];iq_mm<TY,T,PREP>(w,x,y,p,g,tid,a,b);}
IQ_MM(iq_mm20,20,float,false) IQ_MM(iq_mm21,21,float,false) IQ_MM(iq_mm22,22,float,false)
IQ_MM(iq_prepared20,20,half,true) IQ_MM(iq_prepared21,21,half,true) IQ_MM(iq_prepared22,22,half,true)
#undef IQ_MM

// Exact selected-expert matrix multiplication. p: K,N,entries,active,experts.
// active=10 gathers a gate/up input; active=1 consumes per-entry down input.
// Bound invalid ids before addressing weights, and propagate NaN, not zeros.
template<uint Ty>
inline void iq_expert_mv(device const uchar* w,device const float* x,device const uint* ids,
    device float* y,constant uint* p,uint2 g,uint sg,uint lane) {
    uint col=g.x*4+sg,entry=g.y;if(col>=p[1] || entry>=p[2])return;
    uint expert=ids[entry];float4 acc=0;
    if(expert<p[4])for(uint k=lane*4;k<p[0];k+=128)
        acc=fma(iq_values4<Ty>(w,(ulong(expert)*p[1]+col)*p[0]+k),
            *reinterpret_cast<device const float4*>(x+ulong(entry/p[3])*p[0]+k),acc);
    float sum=simd_sum(acc.x+acc.y+acc.z+acc.w);
    if(lane==0)y[ulong(entry)*p[1]+col]=expert<p[4] ? sum : NAN;
}
#define IQ_EXPERT_MV(NAME,TY) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
 device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]], \
 uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]], \
 uint lane [[thread_index_in_simdgroup]]) {iq_expert_mv<TY>(w,x,ids,y,p,g,sg,lane);}
IQ_EXPERT_MV(iq_expert_mv20,20) IQ_EXPERT_MV(iq_expert_mv21,21) IQ_EXPERT_MV(iq_expert_mv22,22)
#undef IQ_EXPERT_MV

// Sorted, compact 32-row tiles reuse weights only for selected experts.
// moe_align builds stable lists; iq_tiles512 builds the bounded schedule.
// No host count readback, atomic float fold, or expert-wide decompression.
template<uint Ty>
inline void iq_expert_mm(device const uchar* w,device const float* x,device const uint* lists,
    device const uint* counts,device const uint* tiles,device float* y,constant uint* p,
    uint2 g,uint tid,threadgroup float* a,threadgroup float* b) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=p[4] || first>=counts[expert])return;
    uint count=min(32u,counts[expert]-first),n=g.x*32;
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=i/32,k=base+i%32,entry=r<count ? lists[expert*p[2]+first+r] : p[2];
            a[i]=entry<p[2] ? x[ulong(entry/p[3])*p[0]+k] : 0;
        }
        for(uint i=tid*4;i<1024;i+=512) {
            uint col=n+i/32,k=base+i%32;
            *reinterpret_cast<threadgroup float4*>(b+i)=col<p[1] ? iq_values4<Ty>(w,(ulong(expert)*p[1]+col)*p[0]+k) : float4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && ij[1]<count && n+ij[0]<p[1]) {
            uint entry=lists[expert*p[2]+first+ij[1]];
            if(entry<p[2])y[ulong(entry)*p[1]+n+ij[0]]=*it;
        }
    }
}
#define IQ_EXPERT_MM(NAME,TY) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
 device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]], \
 device const uint* tiles [[buffer(4)]],device float* y [[buffer(5)]],constant uint* p [[buffer(6)]], \
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float a[1024],b[1024];iq_expert_mm<TY>(w,x,lists,counts,tiles,y,p,g,tid,a,b);}
IQ_EXPERT_MM(iq_expert_mm20,20) IQ_EXPERT_MM(iq_expert_mm21,21) IQ_EXPERT_MM(iq_expert_mm22,22)
#undef IQ_EXPERT_MM

// 512 experts exceed the older 256-seat tile builder. A SIMD prefix plus
// 16-part scan creates exact offsets, including zero-count experts.
kernel void iq_tiles512(device const uint* counts [[buffer(0)]],device uint* tiles [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[16],offsets[16];
    uint num=tid<p[0] ? (counts[tid]+31)/32 : 0,off=simd_prefix_exclusive_sum(num),sum=simd_sum(num);
    if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        uint v=lane<16 ? sums[lane] : 0,prefix=simd_prefix_exclusive_sum(v);
        uint total=simd_sum(v);
        if(lane<16)offsets[lane]=prefix;
        if(lane==0)tiles[0]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<num;++j){uint out=offsets[sg]+off+j;tiles[1+2*out]=tid;tiles[2+2*out]=j*32;}
}

// GPU-only format diagnostic / exact row gather. p: type,width,row_count,
// table_rows. Width and table bytes are checked before dispatch by the host.
// The same gather is usable for the 160-wide IQ4_NL PLE table.
kernel void iq_gather(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1]*p[2]/4)return;
    uint row=(i*4)/p[1],col=(i*4)%p[1],id=ids[row];float4 v=NAN;
    if(id<p[3]) {
        ulong index=ulong(id)*p[1]+col;
        if(p[0]==20)v=iq_values4<20>(w,index);
        else if(p[0]==21)v=iq_values4<21>(w,index);
        else if(p[0]==22)v=iq_values4<22>(w,index);
    }
    *reinterpret_cast<device float4*>(y+i*4)=v;
}
