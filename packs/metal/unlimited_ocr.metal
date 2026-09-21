// Unlimited-OCR graph glue. Original Metal kernels; Q8 bytes stay resident.
kernel void uocr_dense(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],
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
            if(col<N && k<K){ulong at=ulong(col)*K+k;
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

kernel void uocr_gu_decode(device const uchar* gate [[buffer(0)]], device const uchar* up [[buffer(1)]],
    device const float* x [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]], uint2 g [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;
    ulong wbase=(ulong(ids[entry])*p[1]+n)*p[0];float4 ga=0,ua=0;
    for(uint k=lane*32;k<p[0];k+=1024) {
        device const uchar* gb=gate+(wbase+k)/32*34;device const uchar* ub=up+(wbase+k)/32*34;
        float gs=float(*reinterpret_cast<device const half*>(gb)),us=float(*reinterpret_cast<device const half*>(ub));
        for(uint j=0;j<32;j+=4) {
            float4 v=*reinterpret_cast<device const float4*>(x+ulong(entry/6)*p[0]+k+j);
            ga=fma(float4(*reinterpret_cast<device const packed_char4*>(gb+2+j))*gs,v,ga);
            ua=fma(float4(*reinterpret_cast<device const packed_char4*>(ub+2+j))*us,v,ua);
        }
    }
    float a=simd_sum(ga.x+ga.y+ga.z+ga.w),b=simd_sum(ua.x+ua.y+ua.z+ua.w);
    if(lane==0){out[ulong(entry)*p[1]*2+n]=a;out[ulong(entry)*p[1]*2+p[1]+n]=b;}
}

kernel void uocr_route(device const float* logits [[buffer(0)]],device uint* ids [[buffer(1)]],
 device float* weights [[buffer(2)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float a=logits[row*64+lane],b=logits[row*64+32+lane],hi=simd_max(max(a,b));
    float den=simd_sum(exp(a-hi)+exp(b-hi));
    for(uint j=0;j<6;++j){
        float top=simd_max(max(a,b));uint id=simd_min(a==top?lane:b==top?lane+32:UINT_MAX);
        if(id==UINT_MAX)id=0;
        if(lane==0){ids[row*6+j]=id;weights[row*6+j]=exp(top-hi)/den;}
        if(id==lane)a=-INFINITY;if(id==lane+32)b=-INFINITY;
    }
}
kernel void uocr_fold(device const float* x [[buffer(0)]],device const float* weights [[buffer(1)]],
 device float* delta [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;uint row=i/p[0],n=i%p[0];float sum=0;
    for(uint j=0;j<6;++j)sum+=x[(ulong(row)*6+j)*p[0]+n]*weights[row*6+j];delta[i]+=sum;
}
#define UOCR_GROUP(NAME,DOWN) \
kernel void NAME(device const uchar* wg [[buffer(0)]],device const uchar* wu [[buffer(1)]], \
 device const float* x [[buffer(2)]],device const uint* lists [[buffer(3)]],device const uint* counts [[buffer(4)]], \
 device const uint* tiles [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]], \
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float a[16*64],w[32*64];qmoe_grouped<16,DOWN,float,32,false,6>(wg,wu,x,lists,counts,tiles,out,p,g,tid,a,w); }
UOCR_GROUP(uocr_gu_grouped,false)
UOCR_GROUP(uocr_down_grouped,true)
#undef UOCR_GROUP
kernel void uocr_rope(device float* x [[buffer(0)]],device const uint2* meta [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint pair=i%64,row=i/(64*p[0]);if(row>=p[1])return;
    ulong at=ulong(i/64)*128+pair;float a=x[at],b=x[at+64];
    float theta=float(meta[row].y)*pow(10000.0f,-float(pair)/64.0f),c=cos(theta),s=sin(theta);
    x[at]=a*c-b*s;x[at+64]=a*s+b*c;
}
kernel void uocr_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
 device half* kc [[buffer(2)]],device half* vc [[buffer(3)]],device const uint2* meta [[buffer(4)]],
 device const uint* pages [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1280;if(row>=p[0])return;uint2 at=meta[row];
    ulong out=(ulong(pages[at.x*p[1]+at.y/16])*16+at.y%16)*1280+i%1280;
    kc[out]=half(k[i]);vc[out]=half(v[i]);
}
kernel void uocr_decode(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
 device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],
 constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float scores[32],prob[32],highs[1],sums[1];
 gemma_decode<128,1,true>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}
