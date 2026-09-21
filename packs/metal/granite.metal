// Paddock's original Metal kernels. Algorithms: SIMD reductions, tiled
// TensorOps GEMM (Apple MPP guide), and exact online-softmax attention.
// GGUF layout stays intact: no requantization at load or CPU model execution.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// GGUF superblocks are decoded in place. These are storage-format equations,
// not a requantization: embeddings and SIMD decode retain FP32 products;
// TensorOps alone rounds its tile operands to F16.
constant short iq4_values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
inline float4 kquant4(device const uchar* w, uint ty, ulong i) {
    uint j=uint(i%256), s=j/32;
    if(ty==12 || ty==13) {
        device const uchar* b=w+(i/256)*(ty==12 ? 144 : 176);
        device const uchar* sc=b+4;
        uint ds=s<4 ? sc[s]&63 : (sc[s+4]&15)|((sc[s-4]>>6)<<4);
        uint ms=s<4 ? sc[s+4]&63 : (sc[s+4]>>4)|((sc[s]>>6)<<4);
        float d=float(*reinterpret_cast<device const half*>(b))*float(ds);
        float m=float(*reinterpret_cast<device const half*>(b+2))*float(ms);
        uint offset=(j/64)*32+j%32, shift=(s%2)*4;
        uchar4 q=*reinterpret_cast<device const packed_uchar4*>(b+(ty==12?16:48)+offset);
        uint4 v=(uint4(q)>>shift)&15;
        if(ty==13) v|=((uint4(*reinterpret_cast<device const packed_uchar4*>(b+16+j%32))>>s)&1)<<4;
        return float4(v)*d-m;
    }
    if(ty==14) {
        device const uchar* b=w+(i/256)*210;
        uint part=j/128, r=j%128;
        uint4 lo=uint4(*reinterpret_cast<device const packed_uchar4*>(b+part*64+r%64));
        uint4 hi=uint4(*reinterpret_cast<device const packed_uchar4*>(b+128+part*32+r%32));
        int4 v=int4(((lo>>((r/64)*4))&15)|(((hi>>((r/32)*2))&3)<<4))-32;
        float d=float(*reinterpret_cast<device const half*>(b+208))*float(reinterpret_cast<device const char*>(b+192)[j/16]);
        return float4(v)*d;
    }
    device const uchar* b=w+(i/256)*136;
    uint hi=uint(*reinterpret_cast<device const ushort*>(b+2));
    int scale=int(((b[4+s/2]>>((s%2)*4))&15)|(((hi>>(2*s))&3)<<4))-32;
    uint4 code=(uint4(*reinterpret_cast<device const packed_uchar4*>(b+8+s*16+j%16))>>((j%32/16)*4))&15;
    float d=float(*reinterpret_cast<device const half*>(b))*float(scale);
    return float4(iq4_values[code.x],iq4_values[code.y],iq4_values[code.z],iq4_values[code.w])*d;
}

inline float weight(device const uchar* w, uint ty, ulong i) {
    if(ty==12 || ty==13 || ty==14 || ty==23) return kquant4(w,ty,i&~3ul)[i%4];
    if (ty == 8) {
        ulong b = i / 32;
        return float(*reinterpret_cast<device const half*>(w + b * 34)) * float(reinterpret_cast<device const char*>(w + b * 34 + 2)[i % 32]);
    }
    if (ty == 0) return reinterpret_cast<device const float*>(w)[i];
    if (ty == 1) return float(reinterpret_cast<device const half*>(w)[i]);
    return as_type<float>(uint(reinterpret_cast<device const ushort*>(w)[i]) << 16); // BF16
}

// p: width, rows, weight type, embedding scale.
kernel void embed(device const uchar* w [[buffer(0)]], device const uint* ids [[buffer(1)]],
                  device float* out [[buffer(2)]], constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0] * p[1]) out[i] = weight(w, p[2], ulong(ids[i / p[0]]) * p[0] + i % p[0]) * as_type<float>(p[3]);
}

// One 256-thread group per row; float accumulation preserves the checkpoint's
// RMSNorm arithmetic. All threads reach the cross-SIMD barriers.
kernel void rms(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                device float* out [[buffer(2)]], constant uint* p [[buffer(3)]],
                uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    uint n = p[0]; float v = 0;
    for (uint i = tid; i < n; i += 256) { float z = x[row * n + i]; v += z*z; }
    v = simd_sum(v); if (lane == 0) sums[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(lane < 8 ? sums[lane] : 0.0f);
    float inv = rsqrt(total / float(n) + as_type<float>(p[2]));
    for (uint i = tid; i < n; i += 256) out[row*n+i] = x[row*n+i] * inv * weight(w, p[1], i);
}

// Four output rows per threadgroup, one SIMD group per row. Kept distinct
// from GEMM because single-token decode is limited by weight bandwidth.
// p: K, N, M, weight type, output scale.
kernel void linear(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                   device float* out [[buffer(2)]], constant uint* p [[buffer(3)]],
                   uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    uint n = group.x * 4 + sg, m = group.y;
    if (n >= p[1] || m >= p[2]) return;
    float sum = 0;
    for (uint k = lane; k < p[0]; k += 32) sum += weight(w, p[3], ulong(n)*p[0]+k) * x[ulong(m)*p[0]+k];
    sum = simd_sum(sum);
    if (lane == 0) out[ulong(m)*p[1]+n] = sum * as_type<float>(p[4]);
}

// Packed Q8 decode moves four adjacent elements per lane. Eight lanes share
// a quantization block, reducing loop/address/scale overhead versus scalar
// loads while retaining FP32 accumulation and the original GGUF bytes.
kernel void linear_q8(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                      device float* out [[buffer(2)]], constant uint* p [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint n=group.x*4+sg,m=group.y;
    if(n>=p[1] || m>=p[2]) return;
    float4 sum=0;
    for(uint k=lane*4;k<p[0];k+=128) {
        ulong block=(ulong(n)*p[0]+k)/32;
        device const uchar* base=w+block*34;
        float scale=float(*reinterpret_cast<device const half*>(base));
        char4 qs=*reinterpret_cast<device const packed_char4*>(base+2+k%32);
        float4 xs=*reinterpret_cast<device const float4*>(x+ulong(m)*p[0]+k);
        sum+=float4(qs)*scale*xs;
    }
    float total=simd_sum(sum.x+sum.y+sum.z+sum.w);
    if(lane==0) out[ulong(m)*p[1]+n]=total*as_type<float>(p[4]);
}

// GPU-only dense reference for quantized-kernel layout/arithmetic tests.
// The production graph never expands a projection into device F16 storage.
kernel void linear_prepare(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
                           device half* wf [[buffer(2)]],device half* xf [[buffer(3)]],
                           constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(ulong(i)<ulong(p[0])*p[1])wf[i]=half(weight(w,p[3],i));
    if(ulong(i)<ulong(p[0])*p[2])xf[i]=half(x[i]);
}

template<typename A,typename B,typename C>
inline void linear_tile(A a,B b,C c,float scale) {
    constexpr auto desc=matmul2d_descriptor(32,64,dynamic_length_v<int>,false,true);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<A,B,float>();
    op.run(a,b,acc);
    for(uint i=0;i<acc.get_capacity();++i)acc[i]*=scale;
    acc.store(c);
}

kernel void linear_mpp(device half* w [[buffer(0)]],device half* x [[buffer(1)]],
                       device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
                       uint2 group [[threadgroup_position_in_grid]]) {
    // macOS 26's cooperative-tensor type traits reject const half, although
    // these two operands are read-only. The workspace is exclusively owned.
    auto a=tensor(x,dextents<int,2>(p[0],p[2]),array<int,2>{1,int(p[0])});
    auto b=tensor(w,dextents<int,2>(p[0],p[1]),array<int,2>{1,int(p[0])});
    auto c=tensor(out,dextents<int,2>(p[1],p[2]),array<int,2>{1,int(p[1])});
    uint m=group.y*32,n=group.x*64;
    if(m+32<=p[2] && n+64<=p[1]) {
        linear_tile(a.slice<dynamic_extent,32>(0,m),b.slice<dynamic_extent,64>(0,n),c.slice<64,32>(n,m),as_type<float>(p[4]));
    } else {
        linear_tile(a.slice(0,m),b.slice(0,n),c.slice(n,m),as_type<float>(p[4]));
    }
}

// p: Q width, KV width, head dim, row count, page-table stride, rope base.
// meta contains (slot, logical position) per row. Physical KV is preallocated
// [page, token=16, kv_dim]; distinct rows never write the same location.
kernel void rope_store(device float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                       device const float* v [[buffer(2)]], device half* kc [[buffer(3)]],
                       device half* vc [[buffer(4)]], device const uint* meta [[buffer(5)]],
                       device const uint* pages [[buffer(6)]], constant uint* p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    uint pairs=(p[0]+p[1])/2,row=i/pairs,j=i%pairs;
    if(row>=p[3]) return;
    bool isq=j<p[0]/2; uint e=(isq?j:j-p[0]/2)*2;
    uint pos=meta[2*row+1],slot=meta[2*row];
    float angle=float(pos)*pow(as_type<float>(p[5]),-float(e%p[2])/float(p[2]));
    float cs=cos(angle),sn=sin(angle);
    if(isq) {
        ulong ix=ulong(row)*p[0]+e; float a=q[ix],b=q[ix+1];
        q[ix]=a*cs-b*sn; q[ix+1]=a*sn+b*cs;
    } else {
        ulong ix=ulong(row)*p[1]+e;
        uint physical=pages[slot*p[4]+pos/16]*16+pos%16;
        ulong dst=ulong(physical)*p[1]+e;
        float a=k[ix],b=k[ix+1];
        kc[dst]=half(a*cs-b*sn); kc[dst+1]=half(a*sn+b*cs);
        vc[dst]=half(v[ix]); vc[dst+1]=half(v[ix+1]);
    }
}

// Exact online softmax, GQA, causal paged attention. Each SIMD group owns a
// query head; the accumulator stays in registers. Bring-up supports d=128.
// Long contexts split KV over independently scheduled groups and merge exact
// softmax states below (Flash-Decoding / Open-TQ-Metal algorithm, original code).
// Interim: tensor-tiled prefill remains a performance qualification gate.
// p: q_heads, kv_heads, dim, first row, page stride, attention scale, splits.
kernel void attention(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
                      device const half* vc [[buffer(2)]], device const uint* meta [[buffer(3)]],
                      device const uint* pages [[buffer(4)]], device float* out [[buffer(5)]],
                      constant uint* p [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint head=group.x,row=group.y+p[3],kh=head/(p[0]/p[1]);
    uint slot=meta[2*row],pos=meta[2*row+1],stride=p[1]*p[2];
    float4 query,acc=0;
    ulong qo=ulong(row)*p[0]*p[2]+head*p[2]+lane*4;
    for(uint e=0;e<4;++e) query[e]=q[qo+e];
    float maximum=-INFINITY,denom=0;
    uint span=((pos+1+p[6]-1)/p[6]+15)/16*16;
    uint first=group.z*span,last=min(first+span,pos+1);
    for(uint t=first;t<last;++t) {
        uint physical=pages[slot*p[4]+t/16]*16+t%16;
        ulong base=ulong(physical)*stride+kh*p[2]+lane*4;
        float4 key,value;
        for(uint e=0;e<4;++e) { key[e]=float(kc[base+e]); value[e]=float(vc[base+e]); }
        float score=simd_sum(dot(query,key))*as_type<float>(p[5]);
        float next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        acc=acc*old+value*prob; denom=denom*old+prob; maximum=next;
    }
    if(p[6]==1) {
        for(uint e=0;e<4;++e) out[qo+e]=acc[e]/denom;
    } else {
        ulong dst=(ulong(row)*p[0]+head)*p[6]*130+group.z*130;
        for(uint e=0;e<4;++e) out[dst+lane*4+e]=acc[e];
        if(lane==0){out[dst+128]=maximum;out[dst+129]=denom;}
    }
}

// Merge unnormalized (O, max, denominator) without materializing scores.
// Empty page-aligned splits have denominator zero and must not produce NaNs.
kernel void attention_merge(device const float* parts [[buffer(0)]], device float* out [[buffer(1)]],
                            constant uint* p [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint rowhead=group+p[1];
    ulong base=ulong(rowhead)*p[0]*130;
    float maximum=-INFINITY;
    for(uint s=0;s<p[0];++s) maximum=max(maximum,parts[base+s*130+128]);
    float4 acc=0;float denom=0;
    for(uint s=0;s<p[0];++s) {
        ulong src=base+s*130;float d=parts[src+129];
        if(d>0) {
            float correction=exp(parts[src+128]-maximum);
            for(uint e=0;e<4;++e) acc[e]+=parts[src+lane*4+e]*correction;
            denom+=d*correction;
        }
    }
    for(uint e=0;e<4;++e) out[ulong(rowhead)*128+lane*4+e]=acc[e]/denom;
}

// FlashAttention-style query tiling on M5 TensorOps: 16 queries x 32 keys,
// online softmax and immediate P*V. No quadratic score matrix is allocated.
// The host groups contiguous rows from one sequence; paged K/V are gathered
// into only this tile. F32 max/sum/output, F16 matrix operands (including P).
// p: heads, kv_heads, page stride, scale, first row, number of rows.
kernel void attention_mpp(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],
                          device const half* vc [[buffer(2)]], device const uint* meta [[buffer(3)]],
                          device const uint* pages [[buffer(4)]], device float* out [[buffer(5)]],
                          constant uint* p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half queries[16*128],keys[32*128],values[128*32],probs[16*32];
    threadgroup float scores[16*32],products[16*128],maximum[16],denom[16],old[16];
    uint head=group.x,kh=head/(p[0]/p[1]),first=p[4]+group.y*16;
    uint count=min(16u,p[5]-group.y*16),slot=meta[2*first];
    uint lastpos=meta[2*(first+count-1)+1],kvwidth=p[1]*128;
    for(uint i=tid;i<16*128;i+=32) {
        uint r=i/128;
        queries[i]=r<count ? half(q[ulong(first+r)*p[0]*128+head*128+i%128]) : half(0);
    }
    if(tid<16){maximum[tid]=-INFINITY;denom[tid]=0;}
    float accum[64];for(uint i=0;i<64;++i)accum[i]=0;
    auto tq=tensor(queries,dextents<int,2>(128,16),array<int,2>{1,128});
    auto tk=tensor(keys,dextents<int,2>(128,32),array<int,2>{1,128});
    auto tp=tensor(probs,dextents<int,2>(32,16),array<int,2>{1,32});
    auto tv=tensor(values,dextents<int,2>(32,128),array<int,2>{1,32});
    auto ts=tensor(scores,dextents<int,2>(32,16),array<int,2>{1,32});
    auto tc=tensor(products,dextents<int,2>(128,16),array<int,2>{1,128});
    constexpr auto qk_desc=matmul2d_descriptor(16,32,128,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc=matmul2d_descriptor(16,128,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc,execution_simdgroup> qk;
    matmul2d<pv_desc,execution_simdgroup> pv;
    for(uint base=0;base<=lastpos;base+=32) {
        for(uint i=tid;i<32*128;i+=32) {
            uint t=base+i/128,e=i%128;
            if(t<=lastpos){
                uint physical=pages[slot*p[2]+t/16]*16+t%16;
                ulong src=ulong(physical)*kvwidth+kh*128+e;
                keys[i]=kc[src];values[e*32+i/128]=vc[src];
            } else {keys[i]=0;values[e*32+i/128]=0;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto score=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        for(uint i=0;i<score.get_capacity();++i)score[i]=0;
        qk.run(tq,tk,score);score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<16) {
            if(tid<count) {
                uint pos=meta[2*(first+tid)+1];float next=maximum[tid];
                for(uint j=0;j<32;++j)if(base+j<=pos)next=max(next,scores[tid*32+j]*as_type<float>(p[3]));
                float correction=exp(maximum[tid]-next),sum=denom[tid]*correction;
                for(uint j=0;j<32;++j){
                    float prob=base+j<=pos ? exp(scores[tid*32+j]*as_type<float>(p[3])-next) : 0.0f;
                    probs[tid*32+j]=half(prob);sum+=prob;
                }
                maximum[tid]=next;denom[tid]=sum;old[tid]=correction;
            } else {for(uint j=0;j<32;++j)probs[tid*32+j]=0;old[tid]=0;denom[tid]=1;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
        for(uint i=0;i<product.get_capacity();++i)product[i]=0;
        pv.run(tp,tv,product);product.store(tc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<64;++i){uint e=tid+i*32;accum[i]=accum[i]*old[e/128]+products[e];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<64;++i){
        uint e=tid+i*32,r=e/128;
        if(r<count)out[ulong(first+r)*p[0]*128+head*128+e%128]=accum[i]/denom[r];
    }
}

kernel void residual(device float* x [[buffer(0)]], device const float* delta [[buffer(1)]],
                     constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<p[0]) x[i]+=delta[i]*as_type<float>(p[1]);
}
kernel void swiglu(device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                   constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<p[0]) { float g=gate[i]; gate[i]=(g/(1.0f+exp(-g)))*up[i]; }
}
