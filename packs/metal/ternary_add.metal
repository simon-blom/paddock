// Packed ternary dot products: sign-bit selection and addition only.
// Eight lane partials share each 128-weight scale: 8 scale FMAs, not 128
// weight/activation multiplications. F32 activations/accumulators, original
// 2-bit MLX or base-3 PTQ1 weights; no dequantized floating-point matrix.
inline uint ternary_ptq_code(device const uchar* block,uint i) {
    uint byte=i<80?i%16:(i<120?16+(i-80)%8:24+(i-120)%2);
    uint digit=i<80?i/16:(i<120?(i-80)/8:(i-120)/2);
    constexpr uint powers[5]={1,3,9,27,81};
    return ((uint(block[byte])*powers[digit])&255)*3>>8;
}
inline float ternary_signed(float value,uint code) {
    uint bits=code==1?0:as_type<uint>(value);
    return as_type<float>(bits^((code==0?1u:0u)<<31));
}

// Three byte fractions in independent 10-bit integer lanes. 255*3=765
// fits each lane; masking the low 8 bits prevents carry across iterations.
inline uint3 ptq_triplet_next(thread uint& packed) {
    uint product=packed*3;
    packed=product&(255u|(255u<<10)|(255u<<20));
    return (uint3(product)>>uint3(8,18,28))&3;
}
// Decode: 0 = scalar reference, 1 = unrolled/reused codes, 2 = packed triplet.
template<uint R,bool PTQ,bool Full=false,uint Decode=0>
inline void ternary_add(device const uchar* w,device const float* x,device float* out,
    constant uint* p,uint2 g,uint tid) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint K=p[0],N=p[1],M=p[2],lane=tid%8,team=PTQ?(tid%32)/8:0;
    uint col=g.x*(PTQ?4:16)+tid/(PTQ?32:8),first=g.y*R;
    if(col>=N)return;
    float acc[R];for(uint r=0;r<R;++r)acc[r]=0;
    for(uint base=team*128;base<K;base+=(PTQ?512:128)) {
        ulong group=(ulong(col)*K+base)/128;
        device const uchar* b=w+group*28;
        float scale=PTQ?float(*reinterpret_cast<device const half*>(b+26)):
            float(reinterpret_cast<device const half*>(w+ulong(K)*N/4)[group]);
        float4 sums[R];for(uint r=0;r<R;++r)sums[r]=0;
        if constexpr(PTQ) {
            uint a=b[lane*2],c=b[lane*2+1],d=b[16+lane];
            if constexpr(Decode==1) {
                constexpr uint powers[5]={1,3,9,27,81};
                #pragma unroll
                for(uint digit=0;digit<5;++digit) {
                    uint ca=((a*powers[digit])&255)*3>>8,cc=((c*powers[digit])&255)*3>>8,cd=((d*powers[digit])&255)*3>>8;
                    for(uint r=0;r<R;++r)if(Full || first+r<M) {
                        ulong row=ulong(first+r)*K+base;
                        sums[r].x+=ternary_signed(x[row+digit*16+lane*2],ca);
                        sums[r].y+=ternary_signed(x[row+digit*16+lane*2+1],cc);
                        sums[r].z+=ternary_signed(x[row+80+digit*8+lane],cd);
                    }
                }
            } else if constexpr(Decode==2) {
                uint packed=a|(c<<10)|(d<<20);
                for(uint digit=0;digit<5;++digit) {
                    uint3 codes=ptq_triplet_next(packed);
                    for(uint r=0;r<R;++r)if(Full || first+r<M) {
                        ulong row=ulong(first+r)*K+base;
                        sums[r].x+=ternary_signed(x[row+digit*16+lane*2],codes.x);
                        sums[r].y+=ternary_signed(x[row+digit*16+lane*2+1],codes.y);
                        sums[r].z+=ternary_signed(x[row+80+digit*8+lane],codes.z);
                    }
                }
            } else {
                for(uint digit=0;digit<5;++digit) {
                    for(uint r=0;r<R;++r)if(Full || first+r<M) {
                        ulong row=ulong(first+r)*K+base;
                        sums[r].x+=ternary_signed(x[row+digit*16+lane*2],a*3>>8);
                        sums[r].y+=ternary_signed(x[row+digit*16+lane*2+1],c*3>>8);
                        sums[r].z+=ternary_signed(x[row+80+digit*8+lane],d*3>>8);
                    }
                    a=(a*3)&255;c=(c*3)&255;d=(d*3)&255;
                }
            }
            uint code=ternary_ptq_code(b,120+lane);
            for(uint r=0;r<R;++r)if(Full || first+r<M)sums[r].w=ternary_signed(x[ulong(first+r)*K+base+120+lane],code);
        } else {
            uint word=reinterpret_cast<device const uint*>(w)[group*8+lane];
            #pragma unroll
            for(uint j=0;j<4;++j) {
                uint4 codes=(uint4(word)>>uint4(j*8,j*8+2,j*8+4,j*8+6))&3;
                for(uint r=0;r<R;++r)if(Full || first+r<M) {
                    float4 a=*reinterpret_cast<device const float4*>(x+ulong(first+r)*K+base+lane*16+j*4);
                    sums[r]+=float4(ternary_signed(a.x,codes.x),ternary_signed(a.y,codes.y),ternary_signed(a.z,codes.z),ternary_signed(a.w,codes.w));
                }
            }
        }
        for(uint r=0;r<R;++r){float4 v=sums[r];acc[r]=fma((v.x+v.y)+(v.z+v.w),scale,acc[r]);}
    }
    for(uint r=0;r<R;++r) {
        float sum=kquant_sum<PTQ?32:8>(acc[r]);
        if(tid%(PTQ?32:8)==0 && (Full || first+r<M))out[ulong(first+r)*N+col]=sum;
    }
}
#define TERNARY_ADD(R) \
kernel void bonsai_add_##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {ternary_add<R,false>(w,x,out,p,g,tid);} \
kernel void ptq1_add_##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {ternary_add<R,true>(w,x,out,p,g,tid);}
TERNARY_ADD(1)
TERNARY_ADD(2)
TERNARY_ADD(3)
TERNARY_ADD(4)
TERNARY_ADD(8)
#undef TERNARY_ADD
#define TERNARY_ADD_FULL(R) \
kernel void bonsai_add_full_##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {ternary_add<R,false,true>(w,x,out,p,g,tid);} \
kernel void ptq1_add_full_##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {ternary_add<R,true,true>(w,x,out,p,g,tid);}
TERNARY_ADD_FULL(1)
TERNARY_ADD_FULL(2)
TERNARY_ADD_FULL(3)
TERNARY_ADD_FULL(4)
#undef TERNARY_ADD_FULL


#define PTQ1_UNPACK(R) \
kernel void ptq1_unroll_##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {ternary_add<R,true,true,1>(w,x,out,p,g,tid);}
PTQ1_UNPACK(1)
PTQ1_UNPACK(2)
PTQ1_UNPACK(3)
PTQ1_UNPACK(4)
#undef PTQ1_UNPACK
kernel void ptq1_swar_1(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {ternary_add<1,true,true,2>(w,x,out,p,g,tid);}

// Exhaust all 256^3 byte triples through the same helper used by decode.
kernel void ptq1_triplet_check(device atomic_uint* errors [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    uint a=i&255,c=(i>>8)&255,d=(i>>16)&255;
    uint packed=a|(c<<10)|(d<<20);
    constexpr uint powers[5]={1,3,9,27,81};
    for(uint digit=0;digit<5;++digit) {
        uint3 expected=((uint3(a,c,d)*powers[digit])&255)*3>>8;
        if(any(ptq_triplet_next(packed)!=expected))atomic_fetch_add_explicit(errors,1,memory_order_relaxed);
    }
}
