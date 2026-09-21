// Original ViT-G/14 primitives. Images are window-permuted once; each tile
// carries a contiguous attention domain, so sparse blocks never read other
// windows or materialize a dense mask. Global blocks change only descriptors.
inline float mv_lanczos(float x) {
    x=abs(x);if(x<1e-7f)return 1.0f;if(x>=3.0f)return 0.0f;
    float a=3.14159265358979323846f*x;
    return (precise::sin(a)/a)*(precise::sin(a/3.0f)/(a/3.0f));
}
// Separable antialiased Lanczos-3, normalized then Q22-rounded. Host uploads
// RGB bytes only. Include the source/target integer ratio before casting the
// small centered distance, avoiding cancellation for large pixel positions.
kernel void mv_coeff(device int* coeff [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1])return;long src=p[0],dst=p[1],scale=max(src,dst),center=(2*long(i)+1)*src;
    long lo=max(0L,(center-6*scale+dst)/(2*dst)),hi=min(src,(center+6*scale+dst)/(2*dst));
    float sum=0,err=0;
    for(long s=lo;s<hi;++s){float v=mv_lanczos(float((2*s+1)*dst-center)/float(2*scale));
        float y=v-err,t=sum+y;err=(t-sum)-y;sum=t;}
    ulong base=ulong(i)*p[2];coeff[base]=int(lo);coeff[base+1]=int(hi-lo);
    for(long s=lo;s<hi;++s){float v=mv_lanczos(float((2*s+1)*dst-center)/float(2*scale))/sum;
        coeff[base+2+s-lo]=int(v*4194304.0f+(v<0?-0.5f:0.5f));}
}
// p: target W/H, vertical coeff stride, first output patch, source height.
kernel void mv_patches(device const uchar* src [[buffer(0)]],device const int* coeff [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/588,d=i%588,pw=p[0]/14;if(row>=pw*(p[1]/14))return;
    uint x=row%pw*14+d%14,y=row/pw*14+(d%196)/14,c=d/196;ulong b=ulong(y)*p[2];
    int total=1<<21;for(int j=0;j<coeff[b+1];++j)total+=int(src[(ulong(coeff[b]+j)*p[0]+x)*3+c])*coeff[b+2+j];
    float v=float(clamp(total>>22,0,255));
    out[ulong(p[3])*588+i]=float(half(v/255.0f*2.0f-1.0f));
}
// Learned absolute positions use half-pixel bilinear interpolation, not
// align-corners. Each image retains its own rectangular grid.
kernel void mv_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1536,d=i%1536;if(row>=p[0]*p[1])return;
    float sx=max(0.0f,(float(row%p[0])+0.5f)*32.0f/float(p[0])-0.5f);
    float sy=max(0.0f,(float(row/p[0])+0.5f)*32.0f/float(p[1])-0.5f);
    uint x0=min(uint(sx),31u),y0=min(uint(sy),31u),x1=min(x0+1,31u),y1=min(y0+1,31u);
    float a=pos[(y0*32+x0)*1536+d],b=pos[(y0*32+x1)*1536+d];
    float c=pos[(y1*32+x0)*1536+d],e=pos[(y1*32+x1)*1536+d];
    float top=a+(b-a)*(sx-float(x0)),bottom=c+(e-c)*(sx-float(x0));
    x[ulong(p[2])*1536+i]+=top+(bottom-top)*(sy-float(y0));
}
kernel void mv_permute(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    device const uint* perm [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*1536)out[i]=x[ulong(perm[i/1536])*1536+i%1536];
}
// 96d heads: first 48 channels are width, second 48 are height, each with
// interleaved rotary pairs. Positions are one-indexed in this checkpoint.
kernel void mv_qkv(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],
    device const uint2* xy [[buffer(3)]],device half* qo [[buffer(4)]],device half* ko [[buffer(5)]],device half* vo [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*1536)return;uint row=i/1536,d=i%96,j=(d%48)/2,other=i+(d%2?-1:1);
    float theta=float(d<48?xy[row].x:xy[row].y)*pow(10000.0f,-float(j)/24.0f);
    float c=cos(theta),s=sin(theta)*(d%2?1.0f:-1.0f);
    qo[i]=half(q[i]*c+q[other]*s);ko[i]=half(k[i]*c+k[other]*s);vo[i]=half(v[i]);
}
inline float mv_gelu_series(float x) {
    // Metal lacks erf. Evaluate its positive-term convergent series
    // (DLMF 7.6.2); unlike the alternating Maclaurin series this has no
    // cancellation near saturation. erfc(4)<1.6e-8, below half an F32 ulp
    // at one: take the rounded limit before series/exponential roundoff can
    // manufacture a nonzero negative tail. This path is a test check only.
    float z=abs(x)*0.7071067811865475244f;
    if(z>=4.0f)return x>0?x:0.0f;
    float term=z,sum=z;
    for(uint n=1;n<=96;++n){term*=2.0f*z*z/float(2*n+1);float next=sum+term;
        if(next==sum)break;sum=next;}
    float e=min(1.0f,1.1283791670955125739f*exp(-z*z)*sum);
    return 0.5f*x*(1.0f+copysign(e,x));
}
inline float mv_gelu_value(float x) {
    // Original Estrin evaluation of Hastings' rational erf approximation,
    // A&S 7.1.26, p.299 (scanned primary formula):
    // https://personal.math.ubc.ca/~cbm/aands/page_299.htm
    // Published real-arithmetic erf error <=1.5e-7; GPU tests also measure
    // F32 error against the independent convergent series above. Constant
    // work makes this suitable for a matrix epilogue, unlike a 96-step loop.
    // This is erf-GELU, not the tanh/QuickGELU approximation.
    float z=abs(x)*0.7071067811865475244f;
    float t=1.0f/(1.0f+0.3275911f*z),t2=t*t;
    float even=0.254829592f-0.284496736f*t;
    float odd=1.421413741f-1.453152027f*t;
    float tail=t*(even+t2*(odd+1.061405429f*t2));
    float e=1.0f-tail*exp(-z*z);
    return 0.5f*x*(1.0f+copysign(e,x));
}
kernel void mv_gelu(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mv_gelu_value(x[i]);
}
kernel void mv_gelu_series_check(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mv_gelu_series(x[i]);
}
// Same native BF16/F32 matrix primitive as our shared vision projections;
// Muse's erf activation stays in the producer registers. Modes: raw/bias/
// bias+residual/bias+GELU/GELU. No rounded activation side plane is created.
template<uint BM>
inline void mv_bmm(device bfloat* w,device float* x,device float* out,
    device const float* bias,constant uint* p,uint2 g) {
    uint K=p[0],N=p[1],M=p[2],n=g.x*64,m=g.y*BM,mode=p[3];
    auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
    auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
    // F32 operand/storage does not imply strict-F32 multiplication here.
    // Match the measured mixed-precision reference: strict MPP increased
    // white-image embedding error (relative L2 0.00418 versus 0.00218 in
    // the initial relaxed lane). Keep generation parity a separate gate.
    constexpr auto desc=matmul2d_descriptor(BM,64,dynamic_length_v<int>,false,true,true);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();op.run(a,b,acc);
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();uint c=n+ij[0],r=m+ij[1];
        if(it.is_valid_element() && r<M && c<N){float value=*it;
            if(mode>=1 && mode<=3)value+=bias[c];
            if(mode==2)value+=out[ulong(r)*N+c];
            if(mode>=3)value=mv_gelu_value(value);
            out[ulong(r)*N+c]=value;}}
}
#define MV_BMM(BM) \
kernel void mv_bmm##BM(device bfloat* w [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]], \
    device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) { \
    mv_bmm<BM>(w,x,out,bias,p,g); }
MV_BMM(64)
MV_BMM(128)
#undef MV_BMM
kernel void mv_attention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]],
    device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<false,96,96>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
// Undo window order while gathering the channel-outer 2x2 shuffle. A single
// fused gather avoids an additional full encoder output transpose.
kernel void mv_shuffle(device const float* x [[buffer(0)]],device const uint* inverse [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/6144,d=i%6144,c=d/4,s=d%4,nx=p[0]/2;if(row>=nx*(p[1]/2))return;
    uint raster=p[2]+(row/nx*2+s/2)*p[0]+row%nx*2+s%2;
    out[ulong(p[3])*6144+i]=x[ulong(inverse[raster])*1536+c];
}

// Independent SIMD contraction for device-only correctness tests. No serving
// dispatch selects this serial-key route; it verifies domain/layout changes.
kernel void mv_attention_check(device const half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
    device float* out [[buffer(3)]],device const uint2* bounds [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g.y,h=g.x;float3 acc=0;float maximum=-INFINITY,denom=0;
    for(uint t=bounds[row].x;t<bounds[row].y;++t){float score=0;
        for(uint d=lane;d<96;d+=32)score+=float(q[(ulong(row)*16+h)*96+d])*float(k[(ulong(t)*16+h)*96+d]);
        score=simd_sum(score)*0.10206207261596575f;float hi=max(maximum,score),old=exp(maximum-hi),pr=exp(score-hi);
        denom=denom*old+pr;maximum=hi;
        for(uint j=0;j<3;++j)acc[j]=acc[j]*old+pr*float(v[(ulong(t)*16+h)*96+lane+j*32]);
    }
    for(uint j=0;j<3;++j)out[(ulong(row)*16+h)*96+lane+j*32]=acc[j]/denom;
}
