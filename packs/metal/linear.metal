// Original small-batch Q8 projection. A SIMD group owns one output feature
// for R requests: load/dequantize a weight once, reuse it in registers across
// all R accumulators. The accumulation order per row matches the c=1 path.
template<uint R, typename X, bool Full=false>
inline void q8_project(device const uchar* w, device const X* x, device float* out,
                       uint K,uint N,uint M,float output_scale,uint n,uint first,uint lane) {
    if (n >= N) return;
    float4 sums[R];
    #pragma unroll
    for (uint r = 0; r < R; ++r) sums[r] = 0;
    for (uint k = lane * 16; k < K; k += 512) {
        device const uchar* block=w+((ulong(n)*K+k)/32)*34;
        float scale=float(*reinterpret_cast<device const half*>(block));
        #pragma unroll
        for (uint v=0;v<4;++v) {
            float4 weight4 = float4(*reinterpret_cast<device const packed_char4*>(block+2+k%32+v*4))*scale;
            #pragma unroll
            for (uint r = 0; r < R; ++r) {
                if (Full || first + r < M) {
                    float4 input4 = float4(*reinterpret_cast<device const vec<X,4>*>(x + ulong(first + r) * K + k + v*4));
                    sums[r] = fma(weight4,input4,sums[r]);
                }
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < R; ++r) {
        float total = sums[r].x+sums[r].y+sums[r].z+sums[r].w;
        total=simd_sum(total);
        if (lane == 0 && (Full || first + r < M)) out[ulong(first + r) * N + n] = total * output_scale;
    }
}

template<uint R, typename X, bool Full=false>
inline void q8_rows(device const uchar* w,device const X* x,device float* out,
                    constant uint* p,uint2 group,uint sg,uint lane) {
    q8_project<R,X,Full>(w,x,out,p[0],p[1],p[2],as_type<float>(p[4]),group.x*4+sg,group.y*R,lane);
}

// Q/K/V and gate/up are independent projections of the same normalized
// input. Concatenate their dispatch domains, not their weight storage. This
// fills idle GPU cores on narrow projections and removes intervening fences.
// p: input width, three output widths (last may be zero), row count.
#define MULTI_Q8(R,X) \
kernel void linear_multi_q8_r##R(device const uchar* w0 [[buffer(0)]], device const uchar* w1 [[buffer(1)]], \
                                 device const uchar* w2 [[buffer(2)]], device const X* x [[buffer(3)]], \
                                 device float* o0 [[buffer(4)]], device float* o1 [[buffer(5)]], device float* o2 [[buffer(6)]], \
                                 constant uint* p [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]], \
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    uint n=group.x*4+sg,N=p[1]; device const uchar* w=w0; device float* out=o0; \
    if(n>=p[1]) { n-=p[1];N=p[2];w=w1;out=o1; \
        if(n>=p[2]) {n-=p[2];N=p[3];w=w2;out=o2;} } \
    q8_project<R,X,true>(w,x,out,p[0],N,p[4],1.0f,n,group.y*R,lane); \
}
MULTI_Q8(1,float)
MULTI_Q8(2,half)
MULTI_Q8(3,half)
MULTI_Q8(4,half)
#undef MULTI_Q8

#define Q8_ROWS_KERNEL(R) \
kernel void linear_q8_r##R(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], \
                           device float* out [[buffer(2)]], constant uint* p [[buffer(3)]], \
                           uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                           uint lane [[thread_index_in_simdgroup]]) { q8_rows<R,float,R==1>(w,x,out,p,group,sg,lane); }
Q8_ROWS_KERNEL(1)
Q8_ROWS_KERNEL(4)
Q8_ROWS_KERNEL(8)
Q8_ROWS_KERNEL(16)
#undef Q8_ROWS_KERNEL

#define HALF_Q8(R) \
kernel void linear_q8_h##R(device const uchar* w [[buffer(0)]], device const half* x [[buffer(1)]], \
                           device float* out [[buffer(2)]], constant uint* p [[buffer(3)]], \
                           uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                           uint lane [[thread_index_in_simdgroup]]) {q8_rows<R,half,true>(w,x,out,p,group,sg,lane);}
HALF_Q8(2)
HALF_Q8(3)
HALF_Q8(4)
#undef HALF_Q8
kernel void linear_q8_full4(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* out [[buffer(2)]], constant uint* p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) { q8_rows<4,float,true>(w,x,out,p,group,sg,lane); }

// Small activation conversion, shared by independent projection domains.
kernel void linear_input(device const float* x [[buffer(0)]], device half* out [[buffer(1)]],
                         constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i<p[0]*p[2]) out[i]=half(x[i]);
}

// Fixed-K TensorOps contracts the entire tile: tensor extents do not mask
// its contraction tail. Explicitly pad both axes, including unused query
// rows, so no undefined/NaN bytes can be loaded even when multiplied by zero.
kernel void linear_input_padded(device const float* x [[buffer(0)]], device half* out [[buffer(1)]],
                                constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint pitch=(p[0]+127)/128*128, rows=(p[2]+127)/128*128;
    if(i<pitch*rows) {
        uint row=i/pitch,k=i%pitch;
        out[i]=row<p[2] && k<p[0] ? half(x[ulong(row)*p[0]+k]) : half(0);
    }
}
// Original fused quantized TensorOps GEMM. Only one contraction tile is
// dequantized into threadgroup memory; there is no device-wide F16 matrix.
// Apple MPP supplies the matrix primitive. Shape elections keep narrow KV
// projections parallel while wide FF projections reuse weights across rows.
template<uint BM, bool Full=false>
inline void quant_tile(device const uchar* w, device half* x, device float* out,
                       uint K, uint N, uint M, uint type, float scale,
                       uint2 group, uint tid, threadgroup half* weights) {
    constexpr uint BK=128, BN=64;
    uint pitch=(K+127)/128*128,padded_rows=(M+127)/128*128;
    auto input=tensor(x,dextents<int,2>(pitch,padded_rows),array<int,2>{1,int(pitch)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    uint n=group.x*BN,m=group.y*BM;
    for(uint base=0;base<K;base+=BK) {
        if(Full || type==8) {
            // One lane decodes a complete GGUF block. Reuse its scale and
            // address across all 32 values: the former four-value lane
            // mapping repeated that bookkeeping eight times per block.
            for(uint i=tid*32;i<BN*BK;i+=4096) {
                uint col=n+i/BK,k=base+i%BK;
                if(Full || (col<N && k<K)) {
                    device const uchar* block=w+((ulong(col)*K+k)/32)*34;
                    float scale=float(*reinterpret_cast<device const half*>(block));
                    #pragma unroll
                    for(uint j=0;j<8;++j) {
                        float4 value=float4(*reinterpret_cast<device const packed_char4*>(block+2+j*4))*scale;
                        *reinterpret_cast<threadgroup half4*>(weights+i+j*4)=half4(value);
                    }
                } else {
                    #pragma unroll
                    for(uint j=0;j<8;++j)*reinterpret_cast<threadgroup half4*>(weights+i+j*4)=half4(0);
                }
            }
        } else for(uint i=tid*4;i<BN*BK;i+=512) {
            uint col=n+i/BK,k=base+i%BK;
            half4 v=0;
            if(col<N && k<K) {
                ulong index=ulong(col)*K+k;
                if(type==12 || type==13 || type==14 || type==23) v=half4(kquant4(w,type,index));
                else for(uint j=0;j<4;++j)v[j]=k+j<K ? half(weight(w,type,index+j)) : half(0);
            }
            *reinterpret_cast<threadgroup half4*>(weights+i)=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=input.slice<BK,BM>(base,m);
        op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<c.get_capacity();++i)c[i]*=scale;
    if constexpr (Full) {
        auto dst=output.slice<BN,BM>(n,m);
        c.store(dst);
    } else {
        auto dst=output.slice(n,m);
        c.store(dst);
    }
}

#define QUANT_TILE(M) \
kernel void linear_quant_tile##M(device const uchar* w [[buffer(0)]], device half* x [[buffer(1)]], \
                                 device float* out [[buffer(2)]], constant uint* p [[buffer(3)]], \
                                 uint2 group [[threadgroup_position_in_grid]], \
                                 uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half weights[64*128]; quant_tile<M>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),group,tid,weights);}
QUANT_TILE(32)
QUANT_TILE(64)
QUANT_TILE(128)
#undef QUANT_TILE

// Elected only for Q8 matrices and complete M/N/K tiles. The ragged path
// above remains the bounds-checked route; no masked tile can use this entry.
#define FULL_QUANT_TILE(M) \
kernel void linear_quant_full##M(device const uchar* w [[buffer(0)]], device half* x [[buffer(1)]], \
                                 device float* out [[buffer(2)]], constant uint* p [[buffer(3)]], \
                                 uint2 group [[threadgroup_position_in_grid]], \
                                 uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half weights[64*128]; quant_tile<M,true>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),group,tid,weights);}
FULL_QUANT_TILE(32)
FULL_QUANT_TILE(64)
FULL_QUANT_TILE(128)
#undef FULL_QUANT_TILE

// Independent Q/K/V or gate/up matrix domains fill one grid. In particular,
// narrow KV matrices no longer need smaller M tiles to fill the GPU alone.
// Each group still owns one original GGUF matrix and a disjoint output tile.
#define MULTI_QUANT_TILE(BM) \
kernel void linear_multi_quant##BM(device const uchar* w0 [[buffer(0)]], device const uchar* w1 [[buffer(1)]], \
                                   device const uchar* w2 [[buffer(2)]], device half* x [[buffer(3)]], \
                                   device float* o0 [[buffer(4)]], device float* o1 [[buffer(5)]], device float* o2 [[buffer(6)]], \
                                   constant uint* p [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]], \
                                   uint tid [[thread_index_in_threadgroup]]) { \
    uint nx0=(p[1]+63)/64,nx1=(p[2]+63)/64,N=p[1]; \
    device const uchar* w=w0;device float* out=o0; uint type=p[5]; \
    if(group.x>=nx0) { group.x-=nx0;N=p[2];w=w1;out=o1;type=p[6]; \
        if(group.x>=nx1) {group.x-=nx1;N=p[3];w=w2;out=o2;type=p[7];} } \
    threadgroup half weights[64*128]; \
    if(type==8 && p[0]%128==0 && N%64==0 && p[4]%BM==0) \
        quant_tile<BM,true>(w,x,out,p[0],N,p[4],8,1.0f,group,tid,weights); \
    else quant_tile<BM>(w,x,out,p[0],N,p[4],type,1.0f,group,tid,weights); \
}
MULTI_QUANT_TILE(32)
MULTI_QUANT_TILE(64)
MULTI_QUANT_TILE(128)
#undef MULTI_QUANT_TILE

// Gather only rows consumed by the scheduler. A prefill chunk that completes
// no request never executes the vocabulary head.
kernel void rms_selected(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                         device const uint* rows [[buffer(2)]], device float* out [[buffer(3)]],
                         constant uint* p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    uint n = p[0], input_row = rows[row];
    float sum = 0;
    for (uint i = tid; i < n; i += 256) { float a = x[ulong(input_row)*n+i]; sum += a*a; }
    sum = simd_sum(sum);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(lane < 8 ? sums[lane] : 0.0f);
    float inv = rsqrt(total / float(n) + as_type<float>(p[2]));
    for (uint i = tid; i < n; i += 256) out[ulong(row)*n+i] = x[ulong(input_row)*n+i] * inv * weight(w,p[1],i);
}

// Fuse a residual update with the immediately dependent RMSNorm. The stored
// residual remains FP32, including the rounding boundary before normalization.
// p: width, norm weight type, epsilon, residual scale.
kernel void residual_rms(device float* x [[buffer(0)]], device const float* delta [[buffer(1)]],
                         device const uchar* w [[buffer(2)]], device float* out [[buffer(3)]],
                         constant uint* p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    uint n=p[0];float sum=0;
    for(uint i=tid;i<n;i+=256) {
        ulong offset=ulong(row)*n+i;
        float value=x[offset]+delta[offset]*as_type<float>(p[3]);
        x[offset]=value;sum+=value*value;
    }
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    float total=simd_sum(lane<8 ? sums[lane] : 0.0f);
    float inv=rsqrt(total/float(n)+as_type<float>(p[2]));
    for(uint i=tid;i<n;i+=256)out[ulong(row)*n+i]=x[ulong(row)*n+i]*inv*weight(w,p[1],i);
}
