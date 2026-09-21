// Segmented reduction: only lane zero of each L-lane column writes.
template<uint L>
inline float kquant_sum(float v) {
    if constexpr(L==32)return simd_sum(v);
    else {
        #pragma unroll
        for(uint shift=L/2;shift>0;shift/=2)v+=simd_shuffle_down(v,shift);
        return v;
    }
}
// Original scale-factored K-quant GEMV. A lane owns one 16/32-value scale
// group. Quantized integer dot products share loads across requests; scale
// and affine-min corrections are applied once per group, not per element.
// FP32 accumulation, compile-time format/rungs, no persistent weight expansion.
// Decode retains F32 activations. R1 preserves its 32-lane factored reduction;
// R2..R4 use eight lanes per output and direct F32 Q4/Q5 dequantization to
// avoid separate dot and activation-sum arrays per request. Rows >= 5 use
// bounded F16 MPP operands; no persistent weight expansion is introduced.
// Integer masks operate on packed words before widening the bytes:
// Q5's fifth-bit plane joins four nibbles in parallel, reducing unpack work.
template<uint R,uint Type,bool Full=false,typename X=float>
inline void kquant_project(device const uchar* w,device const X* x,device float* out,
                           uint K,uint N,uint M,float scale,uint n,uint first,uint lane) {
    if(n>=N)return;
    constexpr uint L=R==1?32:8;
    if constexpr(Type==12 || Type==13) {
        float4 total[R];for(uint r=0;r<R;++r)total[r]=0;
        for(uint k=lane*32;k<K;k+=L*32) {
            device const uchar* b=w+((ulong(n)*K+k)/256)*(Type==12 ? 144 : 176);
            uint s=(k%256)/32;
            device const uchar* sc=b+4;
            uint ds=s<4 ? sc[s]&63 : (sc[s+4]&15)|((sc[s-4]>>6)<<4);
            uint ms=s<4 ? sc[s+4]&63 : (sc[s+4]>>4)|((sc[s]>>6)<<4);
            float d=float(*reinterpret_cast<device const half*>(b))*float(ds);
            float m=float(*reinterpret_cast<device const half*>(b+2))*float(ms);
            float4 dotq[R],sumx[R];for(uint r=0;r<R;++r){dotq[r]=0;sumx[r]=0;}
            #pragma unroll
            for(uint j=0;j<8;++j) {
                uint packed=(*reinterpret_cast<device const uint*>(b+(Type==12?16:48)+(s/2)*32+j*4)>>((s%2)*4))&0x0f0f0f0f;
                if constexpr(Type==13)packed|=((*reinterpret_cast<device const uint*>(b+16+j*4)>>s)&0x01010101)<<4;
                float4 q=float4(as_type<uchar4>(packed));
                #pragma unroll
                for(uint r=0;r<R;++r)if(Full || first+r<M) {
                    float4 v=float4(*reinterpret_cast<device const vec<X,4>*>(x+ulong(first+r)*K+k+j*4));
                    if constexpr(R==1) {dotq[r]=fma(q,v,dotq[r]);sumx[r]+=v;}
                    else total[r]=fma(q*d-m,v,total[r]);
                }
            }
            if constexpr(R==1)for(uint r=0;r<R;++r)total[r]+=dotq[r]*d-sumx[r]*m;
        }
        for(uint r=0;r<R;++r) {
            float v=kquant_sum<L>(total[r].x+total[r].y+total[r].z+total[r].w);
            if(lane==0 && (Full || first+r<M))out[ulong(first+r)*N+n]=v*scale;
        }
        return;
    }
    constexpr uint V=Type==14 ? 16 : 32;
    float4 sums[R];for(uint r=0;r<R;++r)sums[r]=0;
    for(uint k=lane*V;k<K;k+=L*V) {
        device const uchar* b=w+((ulong(n)*K+k)/256)*(Type==14 ? 210 : 136);
        uint ix=k%256,s=ix/32;
        float d;
        if constexpr(Type==14) d=float(*reinterpret_cast<device const half*>(b+208))*float(reinterpret_cast<device const char*>(b+192)[ix/16]);
        else {
            uint hi=uint(*reinterpret_cast<device const ushort*>(b+2));
            int qs=int(((b[4+s/2]>>((s%2)*4))&15)|(((hi>>(2*s))&3)<<4))-32;
            d=float(*reinterpret_cast<device const half*>(b))*float(qs);
        }
        float4 partial[R];for(uint r=0;r<R;++r)partial[r]=0;
        #pragma unroll
        for(uint j=0;j<V/4;++j) {
            uint index=ix+j*4;
            float4 value;
            if constexpr(Type==14) {
                uint part=index/128,r=index%128;
                // Q6_K's 210-byte stride only guarantees two-byte alignment.
                uint lo=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(b+part*64+r%64));
                uint hi=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(b+128+part*32+r%32));
                uint code=((lo>>((r/64)*4))&0x0f0f0f0f)|(((hi>>((r/32)*2))&0x03030303)<<4);
                value=float4(as_type<uchar4>(code))-32.0f;
            } else {
                uint4 code=(uint4(*reinterpret_cast<device const packed_uchar4*>(b+8+s*16+index%16))>>((index%32/16)*4))&15;
                value=float4(iq4_values[code.x],iq4_values[code.y],iq4_values[code.z],iq4_values[code.w]);
            }
            #pragma unroll
            for(uint r=0;r<R;++r)if(Full || first+r<M)
                partial[r]=fma(value,float4(*reinterpret_cast<device const vec<X,4>*>(x+ulong(first+r)*K+k+j*4)),partial[r]);
        }
        for(uint r=0;r<R;++r)sums[r]=fma(partial[r],d,sums[r]);
    }
    #pragma unroll
    for(uint r=0;r<R;++r) {
        float v=kquant_sum<L>(sums[r].x+sums[r].y+sums[r].z+sums[r].w);
        if(lane==0 && (Full || first+r<M))out[ulong(first+r)*N+n]=v*scale;
    }
}
template<uint R,bool Full=false,typename X=float>
inline void kquant_dispatch(device const uchar* w,device const X* x,device float* out,
                            uint K,uint N,uint M,uint type,float scale,uint n,uint first,uint lane) {
    switch(type) {
        case 12:kquant_project<R,12,Full,X>(w,x,out,K,N,M,scale,n,first,lane);break;
        case 13:kquant_project<R,13,Full,X>(w,x,out,K,N,M,scale,n,first,lane);break;
        case 14:kquant_project<R,14,Full,X>(w,x,out,K,N,M,scale,n,first,lane);break;
        case 23:kquant_project<R,23,Full,X>(w,x,out,K,N,M,scale,n,first,lane);break;
    }
}
#define KQUANT(R,X) \
kernel void linear_kquant##R(device const uchar* w [[buffer(0)]],device const X* x [[buffer(1)]], \
                             device float* out [[buffer(2)]],constant uint* p [[buffer(3)]], \
                             uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]], \
                             uint lane [[thread_index_in_simdgroup]]) { \
    kquant_dispatch<R,R<=4,X>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),group.x*(R==1?4:16)+sg*(R==1?1:4)+lane/(R==1?32:8),group.y*R,lane%(R==1?32:8)); }
KQUANT(1,float)
KQUANT(2,float)
KQUANT(3,float)
KQUANT(4,float)
#undef KQUANT

kernel void linear_kquant_tail4(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
                               device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    kquant_dispatch<4,false,float>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),group.x*16+sg*4+lane/8,group.y*4,lane%8);
}

// Independent mixed-format domains share a launch. The domain boundary is
// rounded in threadgroups, not outputs: even a 48-row alpha/beta projection
// has a disjoint grid and cannot send a lane into its neighbour's storage.
#define MULTI_KQUANT(R,X) \
kernel void linear_multi_kquant##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]], \
                                   device const uchar* w2 [[buffer(2)]],device const X* x [[buffer(3)]], \
                                   device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
                                   constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]], \
                                   uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint C=R==1?4:16;uint nx0=(p[1]+C-1)/C,nx1=(p[2]+C-1)/C,N=p[1],type=p[5]; \
    device const uchar* w=w0;device float* out=o0; \
    if(group.x>=nx0) {group.x-=nx0;N=p[2];type=p[6];w=w1;out=o1; \
        if(group.x>=nx1) {group.x-=nx1;N=p[3];type=p[7];w=w2;out=o2;} } \
    kquant_dispatch<R,true,X>(w,x,out,p[0],N,p[4],type,1.0f,group.x*C+sg*(C/4)+lane/(R==1?32:8),group.y*R,lane%(R==1?32:8)); \
}
MULTI_KQUANT(1,float)
MULTI_KQUANT(2,float)
MULTI_KQUANT(3,float)
MULTI_KQUANT(4,float)
#undef MULTI_KQUANT

// Decode one aligned 32-value staging group. Hoist its scale metadata and
// unpack whole integer words before widening, as in the elected SIMD path.
// No persistent expansion: the destination is the existing bounded MPP slab.
template<uint Type,typename T=half>
inline void kquant_stage32(device const uchar* w,threadgroup T* dest,uint K,uint N,uint col,uint k) {
    if(col>=N) {
        for(uint j=0;j<8;++j)*reinterpret_cast<threadgroup vec<T,4>*>(dest+j*4)=vec<T,4>(0);
        return;
    }
    if constexpr(Type==12 || Type==13) {
        device const uchar* b=w+((ulong(col)*K+k)/256)*(Type==12 ? 144 : 176);
        uint s=(k%256)/32;
        device const uchar* sc=b+4;
        uint ds=s<4 ? sc[s]&63 : (sc[s+4]&15)|((sc[s-4]>>6)<<4);
        uint ms=s<4 ? sc[s+4]&63 : (sc[s+4]>>4)|((sc[s]>>6)<<4);
        float d=float(*reinterpret_cast<device const half*>(b))*float(ds);
        float m=float(*reinterpret_cast<device const half*>(b+2))*float(ms);
        #pragma unroll
        for(uint j=0;j<8;++j) {
            uint packed=(*reinterpret_cast<device const uint*>(b+(Type==12?16:48)+(s/2)*32+j*4)>>((s%2)*4))&0x0f0f0f0f;
            if constexpr(Type==13)packed|=((*reinterpret_cast<device const uint*>(b+16+j*4)>>s)&0x01010101)<<4;
            *reinterpret_cast<threadgroup vec<T,4>*>(dest+j*4)=vec<T,4>(float4(as_type<uchar4>(packed))*d-m);
        }
    } else if constexpr(Type==14) {
        device const uchar* b=w+((ulong(col)*K+k)/256)*210;
        uint ix=k%256;
        float d=float(*reinterpret_cast<device const half*>(b+208));
        float2 scale=d*float2(reinterpret_cast<device const char*>(b+192)[ix/16],
                              reinterpret_cast<device const char*>(b+192)[ix/16+1]);
        #pragma unroll
        for(uint j=0;j<8;++j) {
            uint index=ix+j*4,part=index/128,r=index%128;
            uint lo=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(b+part*64+r%64));
            uint hi=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(b+128+part*32+r%32));
            uint code=((lo>>((r/64)*4))&0x0f0f0f0f)|(((hi>>((r/32)*2))&0x03030303)<<4);
            *reinterpret_cast<threadgroup vec<T,4>*>(dest+j*4)=vec<T,4>((float4(as_type<uchar4>(code))-32.0f)*scale[j/4]);
        }
    } else {
        #pragma unroll
        for(uint j=0;j<8;++j)
            *reinterpret_cast<threadgroup vec<T,4>*>(dest+j*4)=vec<T,4>(kquant4(w,Type,ulong(col)*K+k+j*4));
    }
}

// Whole-superblock staging: each lane decodes a 32-value scale group,
// specializing the format before entering the contraction loop. BK=256
// halves barrier frequency. Narrow M<=32 tiles use BN=16 (8.125 KiB);
// larger M tiles retain BN=32 with compact storage (16 KiB). M5 measurements elect
// four SIMD groups for both, with the same contraction/activation precision.
// BM>=64 must include MPP's implicit staging in the Apple10 32 KiB limit:
// four-half row padding reaches 33,280 bytes under shader validation. Use
// compact storage for single, split and multi-projection kernels alike.
// BM96 fills the mixed-image scheduler's middle rung: the padded activation
// is still 128 rows, but the contraction computes only the admitted tile.
template<uint BM,uint Type,uint Splits=1,uint Padding=4>
inline void ktile(device const uchar* w,device half* x,device float* out,
                  uint K,uint N,uint M,float scale,uint2 group,uint tid,threadgroup half* weights,uint part=0) {
    constexpr uint BK=256,BN=BM<=32?16:32;
    uint pitch=(K+127)/128*128,padded_rows=(M+127)/128*128;
    auto input=tensor(x,dextents<int,2>(pitch,padded_rows),array<int,2>{1,int(pitch)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+Padding});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    uint n=group.x*BN,m=group.y*BM;
    for(uint base=part*(K/Splits);base<(part+1)*(K/Splits);base+=BK) {
        for(uint i=tid*32;i<BN*BK;i+=4096) {
            uint col=n+i/BK,k=base+i%BK;
            kquant_stage32<Type>(w,weights+(i/BK)*(BK+Padding)+i%BK,K,N,col,k);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=input.slice<BK,BM>(base,m);
        op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<c.get_capacity();++i)c[i]*=scale;
    c.store(output.slice(n,m));
}
template<uint BM,uint Splits=1,uint Padding=4>
inline void ktile_dispatch(device const uchar* w,device half* x,device float* out,
                           uint K,uint N,uint M,uint type,float scale,uint2 group,uint tid,threadgroup half* weights,uint part=0) {
    switch(type) {
        case 12:ktile<BM,12,Splits,Padding>(w,x,out,K,N,M,scale,group,tid,weights,part);break;
        case 13:ktile<BM,13,Splits,Padding>(w,x,out,K,N,M,scale,group,tid,weights,part);break;
        case 14:ktile<BM,14,Splits,Padding>(w,x,out,K,N,M,scale,group,tid,weights,part);break;
        case 23:ktile<BM,23,Splits,Padding>(w,x,out,K,N,M,scale,group,tid,weights,part);break;
    }
}
// Long-K contractions use four disjoint F32 partial planes after the padded
// activation prefix. The host admits this route only if the existing scratch
// covers all planes and each K partition ends on a quantization superblock.
// The following reduction is the only writer of the final output.
#define SPLIT_KTILE(BM) \
kernel void linear_ktile_split##BM(device const uchar* w [[buffer(0)]],device half* x [[buffer(1)]], \
                                  device float* work [[buffer(2)]],constant uint* p [[buffer(3)]], \
                                  uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    constexpr uint Pad=BM>=64?0:4;threadgroup half weights[(BM<=32?16:32)*(256+Pad)]; \
    uint offset=((p[0]+127)/128*128)*((p[2]+127)/128*128)/2; \
    ktile_dispatch<BM,4,Pad>(w,x,work+offset+ulong(g.z)*p[1]*p[2],p[0],p[1],p[2],p[3],1.0f,g.xy,tid,weights,g.z); }
SPLIT_KTILE(8)
SPLIT_KTILE(32)
SPLIT_KTILE(64)
#undef SPLIT_KTILE
kernel void linear_ktile_join(device const float* work [[buffer(0)]],device float* out [[buffer(1)]],
                              constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint stride=p[1]*p[2];if(i>=stride)return;
    uint offset=((p[0]+127)/128*128)*((p[2]+127)/128*128)/2;
    device const float* a=work+offset;
    out[i]=((a[i]+a[stride+i])+a[2*stride+i]+a[3*stride+i])*as_type<float>(p[4]);
}
#define KTILE(BM) \
kernel void linear_ktile##BM(device const uchar* w [[buffer(0)]],device half* x [[buffer(1)]], \
                             device float* out [[buffer(2)]],constant uint* p [[buffer(3)]], \
                             uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    constexpr uint Pad=BM>=64?0:4;threadgroup half weights[(BM<=32?16:32)*(256+Pad)]; \
    ktile_dispatch<BM,1,Pad>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),group,tid,weights); }
KTILE(8)
KTILE(32)
KTILE(64)
KTILE(96)
KTILE(128)
#undef KTILE
#define MULTI_KTILE(BM) \
kernel void linear_multi_ktile##BM(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]], \
                                   device const uchar* w2 [[buffer(2)]],device half* x [[buffer(3)]], \
                                   device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
                                   constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]], \
                                   uint tid [[thread_index_in_threadgroup]]) { \
    constexpr uint BN=BM<=32?16:32;uint nx0=(p[1]+BN-1)/BN,nx1=(p[2]+BN-1)/BN,N=p[1],type=p[5]; \
    device const uchar* w=w0;device float* out=o0; \
    if(group.x>=nx0) {group.x-=nx0;N=p[2];type=p[6];w=w1;out=o1; \
        if(group.x>=nx1) {group.x-=nx1;N=p[3];type=p[7];w=w2;out=o2;} } \
    constexpr uint Pad=BM>=64?0:4;threadgroup half weights[(BM<=32?16:32)*(256+Pad)]; \
    ktile_dispatch<BM,1,Pad>(w,x,out,p[0],N,p[4],type,1.0f,group,tid,weights); }
MULTI_KTILE(8)
MULTI_KTILE(32)
MULTI_KTILE(64)
MULTI_KTILE(96)
MULTI_KTILE(128)
#undef MULTI_KTILE

// Large-prefill route: unpack one matrix into a reusable device slab, then
// let MPP stream it directly. Decode retains the compressed original. The
// slab follows the padded activation in one allocation, is overwritten by
// the next projection, and never grows into a persistent expanded model.
kernel void linear_kexpand(device const uchar* w [[buffer(0)]], device half* workspace [[buffer(1)]],
                           constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ulong at=ulong(i)*4, count=ulong(p[0])*p[1];if(at>=count)return;
    ulong offset=ulong((p[0]+127)/128*128)*((p[2]+127)/128*128);
    *reinterpret_cast<device half4*>(workspace+offset+at)=half4(kquant4(w,p[3],at));
}
kernel void linear_kexpanded128(device half* workspace [[buffer(0)]],device float* out [[buffer(1)]],
                               constant uint* p [[buffer(2)]],uint2 g [[threadgroup_position_in_grid]]) {
    uint K=p[0],N=p[1],M=p[2],pitch=(K+127)/128*128,pm=(M+127)/128*128;
    auto a=tensor(workspace,dextents<int,2>(pitch,pm),array<int,2>{1,int(pitch)}).slice(0,g.y*128);
    auto b=tensor(workspace+ulong(pitch)*pm,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,g.x*64);
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(128,64,dynamic_length_v<int>,false,true);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,c);
    for(uint i=0;i<c.get_capacity();++i)c[i]*=as_type<float>(p[4]);
    c.store(output.slice(g.x*64,g.y*128));
}
