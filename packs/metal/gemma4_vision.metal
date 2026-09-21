// Original Gemma 4 vision operations. Algorithm references: Google Gemma 4
// graph, Keys cubic convolution (a=-1/2), Pillow's separable fixed-point
// resampling contract, and our split-Q MPP attention. No host pixel/tensor math.
// Cubic coefficients use exact integer ratios: Metal has no FP64, and F32
// coefficient normalization can change the final u8 at a rounding boundary.
inline long gv_cubic(long x,long d) {
    x=abs(x);if(x>=2*d)return 0;
    return x<d?3*x*x*x-5*x*x*d+2*d*d*d:-x*x*x+5*x*x*d-8*x*d*d+4*d*d*d;
}
kernel void gv_geglu_quick(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float x=gate[i];gate[i]=(x/(1.0f+exp(-1.702f*x)))*up[i];}
}
inline int gv_fixed(long numerator,long denominator) {
    bool neg=numerator<0;ulong r=ulong(abs(numerator)),d=ulong(denominator);
    ulong q=r/d;r%=d;
    for(uint b=0;b<22;++b){r*=2;q=q*2+r/d;r%=d;}
    q+=r*2>=d;return neg?-int(q):int(q);
}
// Each axis row: lower source bound, coefficient count, then Q22 taps.
kernel void gv_coeff(device int* coeff [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1])return;
    long src=p[0],dst=p[1],scale=max(src,dst),center=(2*long(i)+1)*src;
    // Pixel centers are half-integral. Include zero edge taps; normalization
    // then clips support to real pixels rather than replicating border pixels.
    long lo=max(0L,(center-4*scale+dst)/(2*dst)),hi=min(src,(center+4*scale+dst)/(2*dst));
    long sum=0;for(long s=lo;s<hi;++s)sum+=gv_cubic((2*s+1)*dst-center,2*scale);
    ulong base=ulong(i)*p[2];coeff[base]=int(lo);coeff[base+1]=int(hi-lo);
    for(long s=lo;s<hi;++s)coeff[base+2+s-lo]=gv_fixed(gv_cubic((2*s+1)*dst-center,2*scale),sum);
}
kernel void gv_resize_h(device const uchar* src [[buffer(0)]],device const int* coeff [[buffer(1)]],
    device uchar* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1]*p[2]*3)return;uint c=i%3,x=i/3%p[1],y=i/(3*p[1]);
    ulong b=ulong(x)*p[3];int total=1<<21;
    for(int j=0;j<coeff[b+1];++j)total+=int(src[(ulong(y)*p[0]+coeff[b]+j)*3+c])*coeff[b+2+j];
    out[i]=uchar(clamp(total>>22,0,255));
}
// Vertical resample, centered black padding, normalization, raster im2row.
kernel void gv_patches(device const uchar* src [[buffer(0)]],device const int* coeff [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint pw=p[2]/16,ph=p[3]/16,row=i/768,d=i%768;if(row>=pw*ph)return;
    uint x=(row%pw)*16+d%16,y=(row/pw)*16+(d%256)/16,c=d/256;
    uint ox=(p[2]-p[0])/2,oy=(p[3]-p[1])/2;int v=0;
    if(x>=ox && x<ox+p[0] && y>=oy && y<oy+p[1]) {
        ulong b=ulong(y-oy)*p[4];int total=1<<21;
        for(int j=0;j<coeff[b+1];++j)total+=int(src[(ulong(coeff[b]+j)*p[0]+x-ox)*3+c])*coeff[b+2+j];
        v=clamp(total>>22,0,255);
    }
    out[ulong(p[5])*768+i]=float(half(float(v)/255.0f*2.0f-1.0f));
}
kernel void gv_patch_project(device float* w [[buffer(0)]],device float* x [[buffer(1)]],device uint* out [[buffer(2)]],
    device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
    vis_project<64,float,float,false>(w,x,out,bias,p,g);
}
kernel void gv_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1152,d=i%1152;if(row>=p[0]*p[1])return;
    ulong at=ulong(p[2])*1152+i;
    x[at]=(x[at]+pos[(row%p[0])*1152+d])+pos[(p[3]+row/p[0])*1152+d];
}
kernel void gv_rms(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[8];float sum=0;ulong b=ulong(row)*p[0];
    for(uint d=tid;d<p[0];d+=256)sum+=x[b+d]*x[b+d];
    sum=simd_sum(sum);if(tid%32==0)sums[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=rsqrt(simd_sum(tid%32<8?sums[tid%32]:0.0f)/float(p[0])+as_type<float>(p[1]));
    for(uint d=tid;d<p[0];d+=256)out[b+d]=x[b+d]*inv*(p[2]?w[d]:1.0f);
}
// One SIMD group per head owns its 72-value RMS and both NEOX partners.
// Q and K use learned head weights; V has a weightless RMS. The two spatial
// axes rotate independent 36-component halves, never Qwen's axis layout.
kernel void gv_qkv(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],
    device const float* qw [[buffer(3)]],device const float* kw [[buffer(4)]],device const uint2* xy [[buffer(5)]],
    device half* qo [[buffer(6)]],device half* ko [[buffer(7)]],device half* vo [[buffer(8)]],constant uint* p [[buffer(9)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong b=(ulong(g.y)*16+g.x)*72,o=(ulong(g.y)*16+g.x)*80;float qs=0,ks=0,vs=0;
    for(uint d=lane;d<72;d+=32){qs+=q[b+d]*q[b+d];ks+=k[b+d]*k[b+d];vs+=v[b+d]*v[b+d];}
    float qi=rsqrt(simd_sum(qs)/72.0f+as_type<float>(p[0])),ki=rsqrt(simd_sum(ks)/72.0f+as_type<float>(p[0])),vi=rsqrt(simd_sum(vs)/72.0f+as_type<float>(p[0]));
    for(uint d=lane;d<80;d+=32){float a=0,c=0,z=0;
        if(d<72){uint local=d%36,j=local%18,other=(d/36)*36+(local+18)%36;
            float angle=float(d<36?xy[g.y].x:xy[g.y].y)*pow(100.0f,-float(j)/18.0f),cs=cos(angle),sn=sin(angle)*(local<18?-1.0f:1.0f);
            a=q[b+d]*qi*qw[d]*cs+q[b+other]*qi*qw[other]*sn;
            c=k[b+d]*ki*kw[d]*cs+k[b+other]*ki*kw[other]*sn;z=v[b+d]*vi;}
        qo[o+d]=half(a);ko[o+d]=half(c);vo[o+d]=half(z);}
}
// Pool each complete 3x3 cell in raster order, then standardize. The following
// RMS operates across features; neither operation reads the result on the host.
kernel void gv_pool(device const float* x [[buffer(0)]],device const float* bias [[buffer(1)]],device const float* scale [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1152,d=i%1152,nx=p[0]/3,ny=p[1]/3;if(row>=nx*ny)return;float sum=0;
    for(uint y=0;y<3;++y)for(uint xx=0;xx<3;++xx)sum+=x[(ulong(p[2])+((row/nx)*3+y)*p[0]+(row%nx)*3+xx)*1152+d];
    out[ulong(p[3])*1152+i]=((sum/9.0f)*sqrt(1152.0f)-bias[d])*scale[d];
}
