// Original Gemma 26B-A4B hybrid FFN. Model semantics: Google's config and
// same-GGUF reference graph; GPU algorithms/indexing are Paddock's own.
// Every branch is F32. No CPU routes, expanded expert copies or atomic sums.
kernel void gmoe_head(device const float* x [[buffer(0)]],device const float* gamma [[buffer(1)]],
    device const float* pre [[buffer(2)]],device float* router [[buffer(3)]],device float* expert [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];uint n=p[0];ulong base=ulong(row)*n;float sum=0;
    for(uint d=tid;d<n;d+=256){float v=x[base+d];sum+=v*v;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(n)+as_type<float>(p[1]));
    for(uint d=tid;d<n;d+=256){float v=x[base+d]*inv;router[base+d]=(v/sqrt(float(n)))*gamma[d];expert[base+d]=v*pre[d];}
}
kernel void gmoe_route(device const float* logits [[buffer(0)]],device const float* scale [[buffer(1)]],
    device uint* ids [[buffer(2)]],device float* weights [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float score[4];for(uint j=0;j<4;++j)score[j]=logits[row*128+lane+j*32];
    float selected[8];uint chosen[8];
    for(uint pick=0;pick<8;++pick){
        float best=-INFINITY;for(uint j=0;j<4;++j)best=max(best,score[j]);best=simd_max(best);
        uint id=UINT_MAX;for(uint j=0;j<4;++j)if(score[j]==best)id=min(id,lane+j*32);id=simd_min(id);
        if(id==UINT_MAX)id=0;
        selected[pick]=best;chosen[pick]=id;
        for(uint j=0;j<4;++j)if(lane+j*32==id)score[j]=-INFINITY;
    }
    float sum=0,maximum=selected[0];for(uint j=0;j<8;++j){selected[j]=exp(selected[j]-maximum);sum+=selected[j];}
    if(lane==0)for(uint j=0;j<8;++j){ids[row*8+j]=chosen[j];weights[row*8+j]=(selected[j]/sum)*scale[chosen[j]];}
}
// Fused source layout is [expert, gate/up, output, input]. No splitting or
// repacking of the 128 expert planes at load time; one SIMD owns a column.
kernel void gmoe_gu_decode(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,entry=g.y;if(n>=p[1])return;
    ulong wbase=(ulong(ids[entry])*p[1]*2+n)*p[0];float4 ga=0,ua=0;
    for(uint k=lane*32;k<p[0];k+=1024){
        device const uchar* gb=w+(wbase+k)/32*34;device const uchar* ub=w+(wbase+ulong(p[1])*p[0]+k)/32*34;
        float gs=float(*reinterpret_cast<device const half*>(gb)),us=float(*reinterpret_cast<device const half*>(ub));
        for(uint j=0;j<32;j+=4){float4 v=*reinterpret_cast<device const float4*>(x+ulong(entry/8)*p[0]+k+j);
            ga=fma(float4(*reinterpret_cast<device const packed_char4*>(gb+2+j))*gs,v,ga);
            ua=fma(float4(*reinterpret_cast<device const packed_char4*>(ub+2+j))*us,v,ua);}
    }
    float a=simd_sum(ga.x+ga.y+ga.z+ga.w),b=simd_sum(ua.x+ua.y+ua.z+ua.w);
    if(lane==0){out[ulong(entry)*p[1]*2+n]=a;out[ulong(entry)*p[1]*2+p[1]+n]=b;}
}
#define GMOE_GU(BM) \
kernel void gmoe_gu_strict##BM(device const uchar* w [[buffer(0)]],device const uchar* unused [[buffer(1)]], \
 device const float* x [[buffer(2)]],device const uint* lists [[buffer(3)]],device const uint* counts [[buffer(4)]], \
 device const uint* tiles [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]], \
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float act[BM*64],weights[32*64];qmoe_grouped<BM,false,float,32,true>(w,w,x,lists,counts,tiles,out,p,g,tid,act,weights); }
GMOE_GU(16)
GMOE_GU(32)
#undef GMOE_GU
kernel void gmoe_geglu(device float* gu [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;ulong o=ulong(i/p[0])*p[0]*2+i%p[0];gu[o]=vis_gelu(gu[o])*gu[o+p[0]];
}
kernel void gmoe_fold(device const float* out [[buffer(0)]],device const float* weights [[buffer(1)]],
    device float* routed [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;uint row=i/p[0],n=i%p[0];float sum=0;
    for(uint j=0;j<8;++j)sum+=out[(ulong(row)*8+j)*p[0]+n]*weights[row*8+j];routed[i]=sum;
}
// Reduce each branch independently before addition and the final post-FFN
// norm/residual sandwich. Normalizing the sum once is a different model.
kernel void gmoe_branches(device float* shared [[buffer(0)]],device const float* routed [[buffer(1)]],
    device const float* spost [[buffer(2)]],device const float* rpost [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 sums[8];uint n=p[0];ulong base=ulong(row)*n;float2 sum=0;
    for(uint d=tid;d<n;d+=256){float2 v=float2(shared[base+d],routed[base+d]);sum+=v*v;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 inv=rsqrt(simd_sum(lane<8?sums[lane]:float2(0))/float(n)+as_type<float>(p[1]));
    for(uint d=tid;d<n;d+=256)shared[base+d]=shared[base+d]*inv.x*spost[d]+routed[base+d]*inv.y*rpost[d];
}
