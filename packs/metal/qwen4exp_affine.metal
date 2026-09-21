// Original compressed-affine kernels. U32 codes, then BF16 scales, then
// BF16 biases; offsets use 64-bit arithmetic (PLE is larger than 4 GiB).
// This file does not alter the dense-Qwen affine/group64 implementation.
kernel void q4a_small(device const bfloat* x [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]) {float v=float(x[i]);y[i]=p[1]==1 ? -exp(v) : p[1]==2 ? 1.0f+v : v;}
}
template<uint Bits,uint Group>
inline float q4a_value_t(device const uchar* w,uint K,uint N,ulong i) {
    constexpr uint pack=32/Bits;
    uint code=(reinterpret_cast<device const uint*>(w)[i/pack]>>((i%pack)*Bits))&((1u<<Bits)-1);
    device const bfloat* s=reinterpret_cast<device const bfloat*>(w+ulong(K)*N*Bits/8);
    return float(code)*float(s[i/Group])+float(s[ulong(K)*N/Group+i/Group]);
}
inline float q4a_value(device const uchar* w,uint K,uint N,ulong i,uint bits,uint group) {
    // Both supported layouts must specialize division/modulo at compile
    // time. Runtime 64-bit division in every decoded value is prohibitive.
    return bits==4 ? q4a_value_t<4,32>(w,K,N,i) : q4a_value_t<8,64>(w,K,N,i);
}
kernel void q4a_gather(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[2])return;
    uint row=ids[i/p[0]];
    y[i]=row<p[1] ? mlx_bf(q4a_value(w,p[0],p[1],ulong(row)*p[0]+i%p[0],p[3],p[4])) : NAN;
}

// Four outputs share a SIMD's activation loads. BF16 bias subtotals are
// observable in MLX's singleton affine contraction; retain that boundary.
template<uint bits,uint group,uint step,uint Outputs=4>
inline void q4a_vector_step(device const uchar* w,device const float* x,device float* y,
    uint K,uint totalN,uint N,uint col,uint lane,ulong expert) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(col>=N)return; // uniform within a SIMD; no workgroup barriers here
    float sums[Outputs];for(uint c=0;c<Outputs;++c)sums[c]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*totalN*bits/8);
    device const bfloat* biases=scales+ulong(K)*totalN/group;
    for(uint k=lane*step;k<K;k+=32*step) {
        // Compile-time step keeps activations in registers and lets the
        // compiler unroll the packed-word contractions. Runtime-sized local
        // array indexing was still present after specializing quant layout.
        float a[step];float bias_sum=0;
        #pragma unroll
        for(uint j=0;j<step;j+=4) {
            // Loader guarantees K is group-aligned, and step divides the
            // group. Every lane's visited step is complete, with no K tail.
            float4 av=*reinterpret_cast<device const float4*>(x+k+j);
            a[j]=av.x;a[j+1]=av.y;a[j+2]=av.z;a[j+3]=av.w;
            if(bits==4) {
                bias_sum+=mlx_bf(mlx_bf(mlx_bf(a[j]+a[j+1])+a[j+2])+a[j+3]);
                // Compensate fixed nibble positions once per shared input,
                // leaving weight decoding as masks, without per-code shifts.
                a[j+1]*=0.0625f;a[j+2]*=0.00390625f;a[j+3]*=0.000244140625f;
            }
            else {for(uint t=0;t<4;++t)bias_sum+=a[j+t];}
        }
        #pragma unroll
        for(uint c=0;c<Outputs;++c) {
            if(col+c>=N)continue;
            ulong first=(expert*N+col+c)*K+k;float dot=0;
            #pragma unroll
            for(uint j=0;j<step;j+=4) {
                float sub=0;
                if(bits==4) {
                    // One aligned packed load per four codes, not four
                    // independently addressed U32 loads and 64-bit indices.
                    uint codes=reinterpret_cast<device const ushort*>(w)[(first+j)/4];
                    sub=float(codes&15)*a[j];
                    sub+=float(codes&0x00f0)*a[j+1];
                    sub+=float(codes&0x0f00)*a[j+2];
                    sub+=float(codes&0xf000)*a[j+3];
                } else {
                    uint codes=reinterpret_cast<device const uint*>(w)[(first+j)/4];
                    dot+=float(codes&255)*a[j];
                    dot+=float((codes>>8)&255)*a[j+1];
                    dot+=float((codes>>16)&255)*a[j+2];
                    dot+=float(codes>>24)*a[j+3];
                }
                if(bits==4)dot+=sub;
            }
            // The affine group contracts scale*dot + bias*xsum, but adding
            // that group to the running accumulator is a separate rounding.
            // Leaving both implicit let tail-check removal move the FMA
            // boundary and change whole-model choices. Keep it explicit.
            float term=fma(dot,float(scales[first/group]),bias_sum*float(biases[first/group]));
            sums[c]+=term;
        }
    }
    for(uint c=0;c<Outputs;++c) {
        float sum=simd_sum(sums[c]);
        if(lane==0 && col+c<N)y[col+c]=mlx_bf(sum);
    }
}
template<uint bits,uint group>
inline void q4a_vector(device const uchar* w,device const float* x,device float* y,
    uint K,uint totalN,uint N,uint col,uint lane,ulong expert) {
    constexpr uint pack=32/bits;
    if(K%(pack*64)==0 && N>=8 && N%8==0)
        q4a_vector_step<bits,group,pack*2>(w,x,y,K,totalN,N,col,lane,expert);
    else q4a_vector_step<bits,group,pack>(w,x,y,K,totalN,N,col,lane,expert);
}
// Small-batch arithmetic uses affine F32 decoded values, not the singleton
// bias contraction or BF16 tile staging. Eight lanes reduce independent
// quantization groups. Activation values are already BF16 in F32 storage.
kernel void q4a_wide(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    uint n=g.x*8+tid/8,lane=tid%8;if(n>=p[1])return;
    float sum=0;
    for(uint base=lane*p[4];base<p[0];base+=8*p[4])for(uint j=0;j<p[4];j+=8) {
        float sub=0;
        for(uint t=0;t<8;++t)sub+=x[ulong(g.y)*p[0]+base+j+t]*q4a_value(w,p[0],p[1],ulong(n)*p[0]+base+j+t,p[3],p[4]);
        sum+=sub;
    }
    sum=kquant_sum<8>(sum);if(lane==0)y[ulong(g.y)*p[1]+n]=mlx_bf(sum);
}
kernel void q4a_mv(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    if(p[3]==4)q4a_vector<4,32>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],p[0],p[1],p[1],g.x*16+tid/32*4,tid%32,0);
    else q4a_vector<8,64>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],p[0],p[1],p[1],g.x*16+tid/32*4,tid%32,0);
}
// Separate pipeline entries also specialize register allocation and remove
// uniform 4/8-bit and step branches from the hot decode program.
#define Q4A_MV_ENTRY(Name,Bits,Group,Step) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]], \
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1]; \
    q4a_vector_step<Bits,Group,Step>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1], \
        p[0],p[1],p[1],g.x*16+tid/32*4,tid%32,0); \
}
Q4A_MV_ENTRY(q4a_mv4,4,32,8)
Q4A_MV_ENTRY(q4a_mv4_fast,4,32,16)
Q4A_MV_ENTRY(q4a_mv8,8,64,4)
Q4A_MV_ENTRY(q4a_mv8_fast,8,64,8)
#undef Q4A_MV_ENTRY

// HC's four-wide injection otherwise activates only one SIMD, serializing
// four outputs over a 10K-wide input. Give each output its own SIMD without
// changing any lane's K order or its final reduction.
kernel void q4a_mv4_narrow(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    q4a_vector_step<4,32,8,1>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1],p[1],g.x*4+tid/32,tid%32,0);
}
// The 320-wide HC down projection has too few 16-column workgroups to
// occupy M5 Max. Two SIMDs per group double independent scheduling units.
kernel void q4a_mv4_fast2(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    q4a_vector_step<4,32,16>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1],p[1],g.x*8+tid/32*4,tid%32,0);
}
kernel void q4a_expert_mv(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint expert=ids[g.y];if(expert>=p[4])return;
    q4a_vector<4,32>(w,x+ulong(p[3] ? g.y : g.y/10)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1]*p[4],p[1],g.x*16+tid/32*4,tid%32,expert);
}

// Stable, bounded GPU permutation only. The original routing IDs and their
// accumulation order never change. Scratch reuses the GGUF alignment list.
// 128 tokens * top10 fit in a 2048-key sorting network (8 KiB shared).
kernel void q4a_expert_order(device const uint* ids [[buffer(0)]],device uint* order [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint keys[2048];
    for(uint i=tid;i<2048;i+=256)
        keys[i]=i<p[0] && ids[i]<512 ? ids[i]*2048+i : UINT_MAX;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint width=2;width<=2048;width*=2)for(uint stride=width/2;stride>0;stride/=2) {
        for(uint i=tid;i<2048;i+=256) {
            uint other=i^stride;
            if(other>i) {
                uint a=keys[i],b=keys[other];bool ascending=(i&width)==0;
                keys[i]=ascending ? min(a,b) : max(a,b);
                keys[other]=ascending ? max(a,b) : min(a,b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=tid;i<p[0];i+=256)order[i]=keys[i]==UINT_MAX ? UINT_MAX : keys[i]%2048;
}

// Schedule nearby expert entries together at each output tile. Every SIMD
// retains the original singleton arithmetic and scatters directly to its
// token/top-k destination: no activation gather or floating-point atomics.
kernel void q4a_expert_ordered(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],device const uint* order [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint sorted=g.y*8+g.x%8;if(sorted>=p[2])return;
    uint entry=order[sorted];if(entry>=p[2])return;
    uint expert=ids[entry];if(expert>=p[4])return;
    q4a_vector<4,32>(w,x+ulong(p[3] ? entry : entry/10)*p[0],y+ulong(entry)*p[1],
        p[0],p[1]*p[4],p[1],g.x/8*16+tid/32*4,tid%32,expert);
}

// Separate expert steps matter for register allocation just as much as for
// dense projections. A uniform branch still reserves the larger live set.
#define Q4A_EXPERT_ENTRY(Name,Step,Ordered) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],device const uint* order [[buffer(4)]], \
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    uint index=Ordered ? g.y*8+g.x%8 : g.y;if(index>=p[2])return; \
    uint entry=Ordered ? order[index] : index;if(entry>=p[2])return; \
    uint expert=ids[entry];if(expert>=p[4])return; \
    uint tile=Ordered ? g.x/8 : g.x; \
    q4a_vector_step<4,32,Step>(w,x+ulong(p[3] ? entry : entry/10)*p[0],y+ulong(entry)*p[1], \
        p[0],p[1]*p[4],p[1],tile*16+tid/32*4,tid%32,expert); \
}
Q4A_EXPERT_ENTRY(q4a_expert4,8,false)
Q4A_EXPERT_ENTRY(q4a_expert4_fast,16,false)
Q4A_EXPERT_ENTRY(q4a_expert4_ordered,8,true)
Q4A_EXPERT_ENTRY(q4a_expert4_fast_ordered,16,true)
#undef Q4A_EXPERT_ENTRY

// Two adjacent routed rows can share packed weights, scales and biases.
// Keep each row's singleton lane/reduction/FMA contract: unlike choosing
// matrix arithmetic by expert occupancy, reuse cannot change a neighbour's
// logits when another request joins. Only integer routing is shared.
template<uint Step>
inline void q4a_expert_pair(device const uchar* w,device const float* x,device float* y,
    uint K,uint N,uint col,uint lane,uint expert,uint first,uint second,bool per_entry) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float sums[2][4];
    for(uint r=0;r<2;++r)for(uint c=0;c<4;++c)sums[r][c]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N*512/2);
    device const bfloat* biases=scales+ulong(K)*N*512/32;
    for(uint k=lane*Step;k<K;k+=32*Step) {
        float a[2][Step],bias_sum[2]={0,0};
        #pragma unroll
        for(uint r=0;r<2;++r) {
            uint entry=r==0 ? first : second;
            device const float* row=x+ulong(per_entry ? entry : entry/10)*K;
            #pragma unroll
            for(uint j=0;j<Step;j+=4) {
                float4 av=*reinterpret_cast<device const float4*>(row+k+j);
                bias_sum[r]+=mlx_bf(mlx_bf(mlx_bf(av.x+av.y)+av.z)+av.w);
                a[r][j]=av.x;a[r][j+1]=av.y*0.0625f;
                a[r][j+2]=av.z*0.00390625f;a[r][j+3]=av.w*0.000244140625f;
            }
        }
        #pragma unroll
        for(uint c=0;c<4;++c) {
            ulong at=(ulong(expert)*N+col+c)*K+k;
            float dot[2]={0,0};
            #pragma unroll
            for(uint j=0;j<Step;j+=4) {
                uint codes=reinterpret_cast<device const ushort*>(w)[(at+j)/4];
                #pragma unroll
                for(uint r=0;r<2;++r) {
                    float sub=float(codes&15)*a[r][j];
                    sub+=float(codes&0x00f0)*a[r][j+1];
                    sub+=float(codes&0x0f00)*a[r][j+2];
                    sub+=float(codes&0xf000)*a[r][j+3];
                    dot[r]+=sub;
                }
            }
            float scale=float(scales[at/32]),bias=float(biases[at/32]);
            for(uint r=0;r<2;++r)sums[r][c]+=fma(dot[r],scale,bias_sum[r]*bias);
        }
    }
    for(uint r=0;r<2;++r)for(uint c=0;c<4;++c) {
        float value=simd_sum(sums[r][c]);
        if(lane==0)y[ulong(r==0 ? first : second)*N+col+c]=mlx_bf(value);
    }
}

#define Q4A_EXPERT_PAIR(Name,Step) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],device const uint* order [[buffer(4)]], \
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    uint index=(g.y*4+g.x%4)*2;if(index>=p[2])return; \
    uint first=order[index],second=index+1<p[2] ? order[index+1] : UINT_MAX; \
    if(first>=p[2])return;uint expert=ids[first];if(expert>=512)return; \
    uint col=g.x/4*16+tid/32*4,lane=tid%32; \
    if(second<p[2] && ids[second]==expert) { \
        q4a_expert_pair<Step>(w,x,y,p[0],p[1],col,lane,expert,first,second,p[3]!=0); \
    } else { \
        q4a_vector_step<4,32,Step>(w,x+ulong(p[3] ? first : first/10)*p[0],y+ulong(first)*p[1], \
            p[0],p[1]*512,p[1],col,lane,expert); \
        if(second<p[2] && ids[second]<512)q4a_vector_step<4,32,Step>(w,x+ulong(p[3] ? second : second/10)*p[0], \
            y+ulong(second)*p[1],p[0],p[1]*512,p[1],col,lane,ids[second]); \
    } \
}
Q4A_EXPERT_PAIR(q4a_expert4_pair,8)
Q4A_EXPERT_PAIR(q4a_expert4_fast_pair,16)
#undef Q4A_EXPERT_PAIR

// Mixed batches retain singleton arithmetic for decoding and short logical
// prompt chunks. The mask is per token, never a route-occupancy election.
kernel void q4a_expert_vector_masked(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint row=g.y/10;
    if((p[5+row/32]>>(row%32))&1)return;
    uint expert=ids[g.y];if(expert>=512)return;
    q4a_vector<4,32>(w,x+ulong(p[3] ? g.y : row)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1]*512,p[1],g.x*16+tid/32*4,tid%32,expert);
}

// Grouped affine prefill: one expert's routed rows share a
// bounded BF16 tile, with direct scatter to token/top-k destinations.
// The matrix contraction has a different numerical contract from singleton
// decode. The logical-chunk mask owns that election, not physical batch size.
// Retain 32x32x32 as a GPU exactness baseline; 32x64x64 uses 12 KiB staging.
template<uint BN,uint BK>
inline void q4a_expert_matrix(device const uchar* w,device const float* x,
    device const uint* lists,device const uint* counts,device const uint* tiles,
    device float* y,constant uint* p,uint2 g,uint tid,threadgroup bfloat* a,threadgroup bfloat* b) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    uint count=min(32u,counts[expert]-first);
    auto at=tensor(a,extents<int,BK,32>(),array<int,2>{1,BK});
    auto bt=tensor(b,extents<int,BK,BN>(),array<int,2>{1,BK});
    constexpr auto desc=matmul2d_descriptor(32,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=BK) {
        if(BK==32) {
            for(uint i=tid;i<1024;i+=128) {
                uint r=i/32,k=base+i%32,col=g.x*BN+r;
                uint entry=r<count ? lists[expert*p[2]+first+r] : p[2];
                a[i]=bfloat(entry<p[2] ? x[ulong(p[3] ? entry : entry/10)*p[0]+k] : 0);
                b[i]=bfloat(col<p[1] ? q4a_value_t<4,32>(w,p[0],p[1]*512,(ulong(expert)*p[1]+col)*p[0]+k) : 0);
            }
        } else {
            // Four-value packed staging: each code word and scale/bias
            // pair is loaded once, then reused for four matrix operands.
            for(uint i=tid*4;i<32*BK;i+=512) {
                uint r=i/BK,k=base+i%BK;
                uint entry=r<count ? lists[expert*p[2]+first+r] : p[2];
                float4 v=entry<p[2] ? *reinterpret_cast<device const float4*>(x+ulong(p[3] ? entry : entry/10)*p[0]+k) : float4(0);
                *reinterpret_cast<threadgroup bfloat4*>(a+i)=bfloat4(v);
            }
            for(uint i=tid*4;i<BN*BK;i+=512) {
                uint col=g.x*BN+i/BK,k=base+i%BK;float4 v=0;
                if(col<p[1]) {
                    ulong at=(ulong(expert)*p[1]+col)*p[0]+k;
                    uint code=reinterpret_cast<device const ushort*>(w)[at/4];
                    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*512/2);
                    float s=float(scales[at/32]),bias=float(scales[ulong(p[0])*p[1]*512/32+at/32]);
                    float4 codes=float4(code&15,(code>>4)&15,(code>>8)&15,code>>12);
                    v=fma(codes,float4(s),float4(bias));
                }
                *reinterpret_cast<threadgroup bfloat4*>(b+i)=bfloat4(v);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*BN+ij[0];
        if(it.is_valid_element() && ij[1]<count && col<p[1]) {
            uint entry=lists[expert*p[2]+first+ij[1]];
            uint row=entry/10;
            if((p[5+row/32]>>(row%32))&1)y[ulong(entry)*p[1]+col]=mlx_bf(*it);
        }
    }
}
#define Q4A_EXPERT_MATRIX(Name,BN,BK) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]], \
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat a[32*BK],b[BN*BK]; \
    q4a_expert_matrix<BN,BK>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
}
Q4A_EXPERT_MATRIX(q4a_expert_mm,32,32)
Q4A_EXPERT_MATRIX(q4a_expert_mm_wide,64,64)
#undef Q4A_EXPERT_MATRIX

// 32x32 output tiles stage only 4 KiB of BF16 operands. No full-plane
// dequantization and no padded workspace allocation. This is the initial
// correctness path; grouped expert prefill / shape elections remain gated.
kernel void q4a_mm(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    threadgroup bfloat a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    auto total=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    auto subtotal=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<total.get_capacity();++i)total[i]=0;
    uint parts=p[5],lanes=parts>=32 ? 32 : min(parts,8u),span=p[0]/parts;
    // Keep the reference's BF16 partial/join boundary. This sequential
    // per-tile baseline is a correctness seam; parallel split-K is still
    // a performance target, not an established SOTA election.
    for(uint lane=0;lane<lanes;++lane) {
      for(uint i=0;i<subtotal.get_capacity();++i)subtotal[i]=0;
      for(uint part=lane;part<parts;part+=lanes) {
       for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
       for(uint base=part*span;base<(part+1)*span;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint k=base+i%32,row=g.y*32+i/32,col=g.x*32+i/32;
            a[i]=bfloat(row<p[2] ? x[ulong(row)*p[0]+k] : 0);
            b[i]=bfloat(col<p[1] ? q4a_value(w,p[0],p[1],ulong(col)*p[0]+k,p[3],p[4]) : 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
       }
       for(uint i=0;i<subtotal.get_capacity();++i)subtotal[i]=mlx_bf(subtotal[i]+mlx_bf(acc[i]));
      }
      for(uint i=0;i<total.get_capacity();++i)total[i]=parts>=32 ? total[i]+subtotal[i] : mlx_bf(total[i]+subtotal[i]);
    }
    for(auto it=total.begin();it!=total.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*32+ij[0],row=g.y*32+ij[1];
        if(it.is_valid_element() && col<p[1] && row<p[2])y[ulong(row)*p[1]+col]=mlx_bf(*it);
    }
}

// Original packed staging for the single-part dense contraction. Keep the
// baseline's 4 KiB operand footprint and 32x32 compute shape; larger dense
// tiles regressed the complete-model workload despite microbenchmark wins.
template<uint Bits,uint Reuse>
inline void q4a_dense_packed(device const uchar* w,device const float* x,device float* y,
    constant uint* p,uint2 g,uint tid,threadgroup bfloat* a,threadgroup bfloat* b) {
    // Reordered entry is retained only as a GPU test comparator; it did not
    // improve the full-model measurement over the packed baseline.
    if(Reuse>1) {
        uint nx=(p[1]+31)/32,ny=(p[2]+31)/32,index=g.y*nx+g.x;
        uint first=index/(nx*Reuse)*Reuse,height=min(Reuse,ny-first);
        uint local=index-first*nx;
        g=uint2(local/height,first+local%height);
    }
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*Bits/8);
    device const bfloat* biases=scales+ulong(p[0])*p[1]/(Bits==4 ? 32 : 64);
    for(uint base=0;base<p[0];base+=32) {
        for(uint i=tid*4;i<1024;i+=512) {
            uint row=g.y*32+i/32,col=g.x*32+i/32,k=base+i%32;
            float4 av=row<p[2] ? *reinterpret_cast<device const float4*>(x+ulong(row)*p[0]+k) : float4(0);
            *reinterpret_cast<threadgroup bfloat4*>(a+i)=bfloat4(av);
            float4 bv=0;
            if(col<p[1]) {
                ulong index=ulong(col)*p[0]+k;float4 codes;
                if(Bits==4) {
                    uint code=reinterpret_cast<device const ushort*>(w)[index/4];
                    codes=float4(code&15,(code>>4)&15,(code>>8)&15,code>>12);
                } else {
                    uint code=reinterpret_cast<device const uint*>(w)[index/4];
                    codes=float4(code&255,(code>>8)&255,(code>>16)&255,code>>24);
                }
                ulong group=index/(Bits==4 ? 32 : 64);
                bv=fma(codes,float4(float(scales[group])),float4(float(biases[group])));
            }
            *reinterpret_cast<threadgroup bfloat4*>(b+i)=bfloat4(bv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*32+ij[0],row=g.y*32+ij[1];
        if(it.is_valid_element() && col<p[1] && row<p[2])y[ulong(row)*p[1]+col]=mlx_bf(*it);
    }
}
#define Q4A_DENSE_PACKED(Name,Bits,Reuse) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]], \
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat a[1024],b[1024];q4a_dense_packed<Bits,Reuse>(w,x,y,p,g,tid,a,b); \
}
Q4A_DENSE_PACKED(q4a_mm4_packed,4,1)
Q4A_DENSE_PACKED(q4a_mm8_packed,8,1)
Q4A_DENSE_PACKED(q4a_mm4_reuse,4,4)
Q4A_DENSE_PACKED(q4a_mm8_reuse,8,4)
#undef Q4A_DENSE_PACKED

// Original parallel split-K: one workgroup owns one immutable partition.
// The join below preserves the sequential path's BF16 partials and reduction
// order exactly. Only 1 MiB of model-owned scratch is needed at most.
kernel void q4a_mm_split(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device bfloat* partial [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    x+=ulong(p[6])*p[0];
    threadgroup bfloat a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint span=p[0]/p[5];
    for(uint base=g.z*span;base<(g.z+1)*span;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint k=base+i%32,row=g.y*32+i/32,col=g.x*32+i/32;
            a[i]=bfloat(row<p[2] ? x[ulong(row)*p[0]+k] : 0);
            b[i]=bfloat(col<p[1] ? q4a_value(w,p[0],p[1],ulong(col)*p[0]+k,p[3],p[4]) : 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*32+ij[0],row=g.y*32+ij[1];
        if(it.is_valid_element() && col<p[1] && row<p[2])
            partial[(ulong(g.z)*p[2]+row)*p[1]+col]=bfloat(*it);
    }
}

kernel void q4a_mm_join(device const bfloat* partial [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    y+=ulong(p[6])*p[1];
    ulong size=ulong(p[1])*p[2];if(i>=size)return;
    uint parts=p[5],lanes=parts>=32 ? 32 : min(parts,8u);float total=0;
    for(uint lane=0;lane<lanes;++lane) {
        float subtotal=0;
        for(uint part=lane;part<parts;part+=lanes)
            subtotal=mlx_bf(subtotal+float(partial[ulong(part)*size+i]));
        total=parts>=32 ? total+subtotal : mlx_bf(total+subtotal);
    }
    y[i]=mlx_bf(total);
}
