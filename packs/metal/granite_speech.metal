// Original Granite Speech kernels. IBM's conformer/Shaw graph; our existing
// split-Q TensorOps attention and F32 online softmax. No full score matrix.
kernel void gs_inject(device const float* audio [[buffer(0)]],device const uint2* meta [[buffer(1)]],
    device float* x [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/2048,d=i%2048;if(row>=p[0])return;
    uint2 m=meta[row];if(m.x==p[1]&&m.y>=p[2]&&m.y<p[2]+p[3])x[i]=audio[ulong(m.y-p[2])*2048+d]*as_type<float>(p[4]);
}
kernel void gs_fmm(device float* r [[buffer(0)]],device float* q [[buffer(1)]],
    device uint* out [[buffer(2)]],device const float* unused [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
    vis_project<32,float,float>(r,q,out,unused,p,g);
}
kernel void gs_attention(device float* q [[buffer(0)]],device float* k [[buffer(1)]],device float* v [[buffer(2)]],
    device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],device const float* qr [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<false,128,128,8,false,false,float,true>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap,qr);
}
// Independent SIMD witness, used only by GPU regression tests.
kernel void gs_attention_check(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],device const float* rel [[buffer(3)]],device const uint2* bounds [[buffer(4)]],
    device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint row=g.y,head=g.x;float a[4]={0,0,0,0},hi=-INFINITY,den=0;
    for(uint t=bounds[row].x;t<bounds[row].y;++t) {
        float s=0,r=0;
        for(uint j=0;j<4;++j){uint d=lane+j*32;float z=q[(ulong(row)*8+head)*128+d];
            s+=z*k[(ulong(t)*8+head)*128+d];r+=z*rel[(int(row)-int(t)+200)*128+d];}
        s=(simd_sum(s)+simd_sum(r))*0.08838834764831844f;
        float nh=max(hi,s),old=exp(hi-nh),pr=exp(s-nh);hi=nh;den=den*old+pr;
        for(uint j=0;j<4;++j)a[j]=a[j]*old+pr*v[(ulong(t)*8+head)*128+lane+j*32];
    }
    for(uint j=0;j<4;++j)out[(ulong(row)*8+head)*128+lane+j*32]=a[j]/den;
}
// Three queries over 3/15 keys is below a TensorOps query tile. A SIMD
// group owns a query/head; four independent groups share a dispatch.
kernel void gs_qattention(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint row=g.y*4+tid/32,head=g.x,lane=tid%32;if(row>=p[0])return;
    uint first=row/3*p[1];float a=0,b=0,hi=-INFINITY,den=0;
    for(uint t=first;t<first+p[1];++t) {
        ulong qi=(ulong(row)*16+head)*64+lane,ki=(ulong(t)*16+head)*64+lane;
        float score=simd_sum(q[qi]*k[ki]+q[qi+32]*k[ki+32])*0.125f;
        float nh=max(hi,score),old=exp(hi-nh),pr=exp(score-nh);hi=nh;den=den*old+pr;
        a=a*old+pr*v[ki];b=b*old+pr*v[ki+32];
    }
    ulong oi=(ulong(row)*16+head)*64+lane;out[oi]=a/den;out[oi+32]=b/den;
}
kernel void gs_split(device const float* x [[buffer(0)]],device float* q [[buffer(1)]],
    device float* k [[buffer(2)]],device float* v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*1024)return;uint row=i/1024,d=i%1024;
    q[i]=x[row*3072+d];k[i]=x[row*3072+1024+d];v[i]=x[row*3072+2048+d];
}
kernel void gs_residual(device float* x [[buffer(0)]],device const float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]+=y[i]*as_type<float>(p[1]);
}
kernel void gs_silu(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float z=x[i];x[i]=z/(1.0f+exp(-z));}
}
kernel void gs_glu(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[1],d=i%p[1];if(row<p[0])y[i]=x[row*p[1]*2+d]/(1.0f+exp(-x[row*p[1]*2+p[1]+d]));
}
// Clip bounds, not attention block bounds: the centered convolution must
// cross 200-frame attention boundaries but must never cross requests.
kernel void gs_depthwise(device const float* x [[buffer(0)]],device const float* taps [[buffer(1)]],
    device const float* w [[buffer(2)]],device const float* b [[buffer(3)]],device const uint2* clips [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[1],d=i%p[1];if(row>=p[0])return;float z=0;
    for(int j=0;j<15;++j){int t=int(row)+j-7;if(t>=int(clips[row].x)&&t<int(clips[row].y))z+=x[ulong(t)*p[1]+d]*taps[d*15+j];}
    z=z*w[d]+b[d];y[i]=z/(1.0f+exp(-z));
}
kernel void gs_softmax(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float hi=-INFINITY,den=0;for(uint d=lane;d<p[0];d+=32)hi=max(hi,x[row*p[0]+d]);hi=simd_max(hi);
    for(uint d=lane;d<p[0];d+=32)den+=exp(x[row*p[0]+d]-hi);den=simd_sum(den);
    for(uint d=lane;d<p[0];d+=32)x[row*p[0]+d]=exp(x[row*p[0]+d]-hi)/den;
}
// Padding occurs after the encoder (and each captured source), not in the
// conformer. Its zero rows are genuine keys in the last Q-Former window.
kernel void gs_windows(device const float* x [[buffer(0)]],device const float* tap [[buffer(1)]],
    device const uint* indices [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint width=1024*p[1],row=i/width,d=i%width;if(row>=p[0])return;uint src=indices[row];
    out[i]=src==0xffffffffu?0.0f:(p[1]==2&&d<1024?tap[src*1024+d]:x[src*1024+d%1024]);
}
kernel void gs_queries(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]*1024)out[i]=x[i%3072];}
