// Prism GGUF PTQ1_0: 128 trits in 26 bytes followed by one FP16 scale.
// Original implementation of the public storage contract; weights stay packed.
inline float ptq1_weight(device const uchar* w,ulong i) {
    device const uchar* b=w+(i/128)*28;uint e=i%128;
    uint byte=e<80?e%16:(e<120?16+(e-80)%8:24+(e-120)%2);
    uint digit=e<80?e/16:(e<120?(e-80)/8:(e-120)/2);
    constexpr uint power[5]={1,3,9,27,81};
    int code=int(((uint(b[byte])*power[digit])&255)*3>>8)-1;
    return float(code)*float(*reinterpret_cast<device const half*>(b+26));
}
kernel void ptq1_unpack(device const uchar* w [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])out[i]=ptq1_weight(w,i);
}
// The two unrotated 48-row BF16 gate matrices need K parallelism, not the
// large-output packed kernel. A complete threadgroup owns each scalar dot.
kernel void ptq1_gate(device const ushort* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    threadgroup float sums[8];float v=0;
    for(uint k=tid;k<p[0];k+=256)v=fma(as_type<float>(uint(w[ulong(g.x)*p[0]+k])<<16),x[ulong(g.y)*p[0]+k],v);
    v=simd_sum(v);if(tid%32==0)sums[tid/32]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<32){v=simd_sum(tid<8?sums[tid]:0);if(tid==0)out[ulong(g.y)*p[1]+g.x]=v;}
}
kernel void ptq1_embed(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device const float* signs [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float exchange[1024];uint col=g.x*1024+tid;
    ulong row=ulong(ids[g.y])*p[0];float4 v;
    for(uint j=0;j<4;++j)v[j]=ptq1_weight(w,row+col+j*256);
    v=bonsai_fwht(v,tid,exchange);
    for(uint j=0;j<4;++j)out[ulong(g.y)*p[0]+col+j*256]=v[j]*signs[col+j*256];
}
// GGUF GDN heads are tiled; ssm_out's rotated input axis is grouped.
// Fold that permutation into the rotation load, never permute whole weights.
kernel void ptq1_rotate_grouped(device const float* x [[buffer(0)]],device const float* signs [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float exchange[1024];uint col=g.x*1024+tid;ulong row=ulong(g.y)*p[0];float4 v;
    for(uint j=0;j<4;++j){uint d=col+j*256,h=d/128,src=((h%3)*16+h/3)*128+d%128;v[j]=x[row+src]*signs[d];}
    v=bonsai_fwht(v,tid,exchange);
    for(uint j=0;j<4;++j)out[row+col+j*256]=v[j];
}
// One SIMD group owns a row; four eight-lane teams split its K blocks.
// Each lane owns three packed bytes plus one tail trit, reusing their digits
// across live requests. The K reduction is independent of request count.
template<uint R>
inline void ptq1_vectors(device const uchar* w,device const float* x,device float* out,
    uint K,uint N,uint M,uint2 g,uint tid) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint n=g.x*4+tid/32,lane=tid%8,team=(tid%32)/8,first=g.y*R;if(n>=N)return;
    float4 sum[R];for(uint r=0;r<R;++r)sum[r]=0;
    for(uint k=team*128;k<K;k+=512) {
        device const uchar* b=w+((ulong(n)*K+k)/128)*28;
        uint a=b[lane*2],c=b[lane*2+1],d=b[16+lane];
        float scale=float(*reinterpret_cast<device const half*>(b+26));
        for(uint digit=0;digit<5;++digit) {
            float va=float(int(a*3>>8)-1)*scale,vc=float(int(c*3>>8)-1)*scale,vd=float(int(d*3>>8)-1)*scale;
            for(uint r=0;r<R;++r)if(first+r<M) {
                ulong row=ulong(first+r)*K+k;
                sum[r].x=fma(x[row+digit*16+lane*2],va,sum[r].x);
                sum[r].y=fma(x[row+digit*16+lane*2+1],vc,sum[r].y);
                sum[r].z=fma(x[row+80+digit*8+lane],vd,sum[r].z);
            }
            a=(a*3)&255;c=(c*3)&255;d=(d*3)&255;
        }
        constexpr uint power[4]={1,3,9,27};
        float tail=float(int(((uint(b[24+lane%2])*power[lane/2])&255)*3>>8)-1)*scale;
        for(uint r=0;r<R;++r)if(first+r<M)sum[r].w=fma(x[ulong(first+r)*K+k+120+lane],tail,sum[r].w);
    }
    for(uint r=0;r<R;++r){float4 v=sum[r];float s=kquant_sum<32>((v.x+v.y)+(v.z+v.w));if(tid%32==0&&first+r<M)out[ulong(first+r)*N+n]=s;}
}
#define PTQ1_VECTORS(R) \
kernel void ptq1_vectors##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {ptq1_vectors<R>(w,x,out,p[0],p[1],p[2],g,tid);}
PTQ1_VECTORS(1)
PTQ1_VECTORS(2)
PTQ1_VECTORS(3)
PTQ1_VECTORS(4)
#undef PTQ1_VECTORS
template<uint BM>
inline void ptq1_mm(device const uchar* w,device float* x,device float* out,constant uint* p,uint2 g,uint tid,threadgroup float* weights) {
    constexpr uint BN=32,BK=128;
    uint K=p[0],N=p[1],M=p[2],n=g.x*BN,m=g.y*BM;
    auto input=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+4});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    for(uint base=0;base<K;base+=BK) {
        for(uint i=tid;i<BN*BK;i+=128){uint col=n+i/BK;weights[(i/BK)*(BK+4)+i%BK]=col<N?ptq1_weight(w,ulong(col)*K+base+i%BK):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=input.slice(base,m);op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    c.store(output.slice(n,m));
}
#define PTQ1_MM(M) \
kernel void ptq1_mm##M(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {threadgroup float weights[32*132];ptq1_mm<M>(w,x,out,p,g,tid,weights);}
PTQ1_MM(16)
PTQ1_MM(32)
PTQ1_MM(64)
#undef PTQ1_MM
