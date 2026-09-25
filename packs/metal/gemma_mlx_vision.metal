// Native MLX vision boundaries. Spatial transforms run on the GPU from RGB
// bytes; the host computes only bounded integer image/window descriptors.
// Keep text/vision storage compressed and share the ragged attention engine.
// Vision inputs already carry BF16-rounded values in the shared F32 scratch.
// Direct tensor loads avoid staging every K tile through threadgroup memory;
// the producer still preserves the MLX BF16 projection boundary.
kernel void gmlx_vmm64(device bfloat* w [[buffer(0)]],device float* x [[buffer(1)]],
    device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]]) {vis_project<64,bfloat,float,true,true>(w,x,out,bias,p,g);}
inline float gmlx_erf_activation(float x) {
    // Same original Hastings polynomial as mv_gelu_value, but preserve the
    // checkpoint graph's BF16 division, erf, add, multiply and division.
    float arg=mlx_bf(precise::divide(x,mlx_bf(1.4142135623730951f))),z=abs(arg);
    float t=1.0f/(1.0f+0.3275911f*z),t2=t*t;
    float tail=t*((0.254829592f-0.284496736f*t)+t2*((1.421413741f-1.453152027f*t)+1.061405429f*t2));
    float e=mlx_bf(copysign(1.0f-tail*exp(-z*z),arg));
    return mlx_bf(mlx_bf(x*mlx_bf(1.0f+e))*0.5f);
}
kernel void gmlx_bmm(device const bfloat* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint BM=32,BN=32,BK=32;
    threadgroup bfloat a[BM*BK],b[BN*BK];
    auto ta=tensor(a,extents<int,BK,BM>());auto tb=tensor(b,extents<int,BK,BN>());
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    for(uint base=0;base<p[0];base+=BK){
        for(uint i=tid;i<1024;i+=128){uint row=g.y*BM+i/BK,col=g.x*BN+i/BK,k=base+i%BK;
            a[i]=bfloat(row<p[2] && k<p[0]?x[ulong(row)*p[0]+k]:0.0f);
            b[i]=col<p[1] && k<p[0]?w[ulong(col)*p[0]+k]:bfloat(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(ta,tb,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=c.begin();it!=c.end();++it){auto ij=it.get_multidimensional_index();uint col=g.x*BN+ij[0],row=g.y*BM+ij[1];
        if(it.is_valid_element() && col<p[1] && row<p[2]){float z=*it;
            // MLX's single-row addmm lowers to a rounded GEMV then bias;
            // its matrix route retains F32 accumulation through the bias.
            if(p[2]==1)z=mlx_bf(z);
            if(p[3]>=1 && p[3]<=3)z+=bias[col];
            z=mlx_bf(z);
            if(p[3]==2)z=mlx_bf(z+out[ulong(row)*p[1]+col]);
            if(p[3]>=3)z=gmlx_erf_activation(z);
            out[ulong(row)*p[1]+col]=z;}}
}
kernel void gmlx_erfgelu(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=gmlx_erf_activation(x[i]);
}
kernel void gmlx_layer_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    threadgroup float sums[32],squares[32];uint n=p[0],threads=min(1024u,((n+127)/128)*32);ulong base=ulong(row)*n;
    float sum=0,sq=0;for(uint i=tid*4;i<n;i+=threads*4)for(uint d=0;d<4 && i+d<n;++d){float v=x[base+i+d];sum+=v;sq+=v*v;}
    sum=simd_sum(sum);sq=simd_sum(sq);if(tid%32==0){sums[tid/32]=sum;squares[tid/32]=sq;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=precise::divide(simd_sum(tid%32<threads/32?sums[tid%32]:0.0f),float(n));
    float variance=precise::divide(simd_sum(tid%32<threads/32?squares[tid%32]:0.0f),float(n))-mean*mean;
    float inv=precise::rsqrt(max(variance,0.0f)+as_type<float>(p[1]));
    for(uint d=tid;d<n;d+=threads)out[base+d]=mlx_bf(fma(mlx_bf((x[base+d]-mean)*inv),w[d],bias[d]));
}
kernel void gmlx_gv_patches(device const uchar* src [[buffer(0)]],device const int* coeff [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint pw=p[2]/16,ph=p[3]/16,row=i/768,d=i%768;if(row>=pw*ph)return;
    uint x=row%pw*16+(d/3)%16,y=row/pw*16+d/48,c=d%3;ulong b=ulong(y)*p[4];int total=1<<21;
    for(int j=0;j<coeff[b+1];++j)total+=int(src[(ulong(coeff[b]+j)*p[0]+x)*3+c])*coeff[b+2+j];
    float pixel=float(clamp(total>>22,0,255))*(1.0f/255.0f);
    out[ulong(p[5])*768+i]=mlx_bf(2.0f*(pixel-0.5f));
}
kernel void gmlx_gv_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1152,d=i%1152;if(row>=p[0]*p[1])return;ulong at=ulong(p[2])*1152+i;
    x[at]=mlx_bf(x[at]+mlx_bf(pos[(row%p[0])*1152+d]+pos[(p[3]+row/p[0])*1152+d]));
}
kernel void gmlx_gv_qkv(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],
    device const float* qw [[buffer(3)]],device const float* kw [[buffer(4)]],device const uint2* xy [[buffer(5)]],
    device bfloat* qo [[buffer(6)]],device bfloat* ko [[buffer(7)]],device bfloat* vo [[buffer(8)]],constant uint* p [[buffer(9)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    ulong b=(ulong(g.y)*16+g.x)*72,o=(ulong(g.y)*16+g.x)*80;float qs=0,ks=0,vs=0;
    for(uint d=lane;d<72;d+=32){qs+=q[b+d]*q[b+d];ks+=k[b+d]*k[b+d];vs+=v[b+d]*v[b+d];}
    float qi=precise::rsqrt(simd_sum(qs)/72.0f+as_type<float>(p[0])),ki=precise::rsqrt(simd_sum(ks)/72.0f+as_type<float>(p[0])),vi=precise::rsqrt(simd_sum(vs)/72.0f+as_type<float>(p[0]));
    for(uint d=lane;d<80;d+=32){float a=0,c=0,z=0;if(d<72){uint local=d%36,j=local%18,other=d/36*36+(local+18)%36;
        float angle=precise::divide(float(d<36?xy[g.y].x:xy[g.y].y),pow(100.0f,float(j)/18.0f));
        float cs=mlx_bf(cos(angle)),sn=mlx_bf(sin(angle))*(local<18?-1.0f:1.0f);
        a=mlx_bf(mlx_bf(mlx_bf(q[b+d]*qi*qw[d])*cs)+mlx_bf(mlx_bf(q[b+other]*qi*qw[other])*sn));
        c=mlx_bf(mlx_bf(mlx_bf(k[b+d]*ki*kw[d])*cs)+mlx_bf(mlx_bf(k[b+other]*ki*kw[other])*sn));z=mlx_bf(v[b+d]*vi);}
        qo[o+d]=bfloat(a);ko[o+d]=bfloat(c);vo[o+d]=bfloat(z);}
}
kernel void gmlx_gv_pool(device const float* x [[buffer(0)]],device const float* bias [[buffer(1)]],device const float* scale [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1152,d=i%1152,nx=p[0]/3,ny=p[1]/3;if(row>=nx*ny)return;float sum=0;
    for(uint y=0;y<3;++y)for(uint xx=0;xx<3;++xx)sum+=x[(ulong(p[2])+(row/nx*3+y)*p[0]+row%nx*3+xx)*1152+d]*(1.0f/9.0f);
    float z=mlx_bf(mlx_bf(sum)*mlx_bf(sqrt(1152.0f)));
    out[ulong(p[3])*1152+i]=mlx_bf(mlx_bf(z-bias[d])*scale[d]);
}
kernel void gmlx_mv_patches(device const uchar* src [[buffer(0)]],device const int* coeff [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1176,d=i%588,pw=p[0]/14;if(row>=pw*(p[1]/14))return;
    uint x=row%pw*14+d%14,y=row/pw*14+(d%196)/14,c=d/196;ulong b=ulong(y)*p[2];int total=1<<21;
    for(int j=0;j<coeff[b+1];++j)total+=int(src[(ulong(coeff[b]+j)*p[0]+x)*3+c])*coeff[b+2+j];
    out[ulong(p[3])*1176+i]=mlx_bf((float(clamp(total>>22,0,255))*(1.0f/255.0f)-0.5f)*2.0f);
}
// MLX interpolates with zero extension outside the learned 32x32 grid;
// the GGUF path clamps its borders. Keep that distinction explicit.
kernel void gmlx_mv_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1536,d=i%1536;if(row>=p[0]*p[1])return;
    float sx=(float(row%p[0])+0.5f)*(32.0f/float(p[0]))-0.5f,sy=(float(row/p[0])+0.5f)*(32.0f/float(p[1]))-0.5f;
    int x0=int(floor(sx)),y0=int(floor(sy));float dx=sx-float(x0),dy=sy-float(y0),z=0;
    for(int yy=0;yy<2;++yy)for(int xx=0;xx<2;++xx){int px=x0+xx,py=y0+yy;
        if(px>=0 && px<32 && py>=0 && py<32)z+=pos[(py*32+px)*1536+d]*(xx?dx:1.0f-dx)*(yy?dy:1.0f-dy);}
    ulong at=ulong(p[2])*1536+i;x[at]=mlx_bf(x[at]+mlx_bf(z));
}
kernel void gmlx_mv_qkv(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],
    device const uint2* xy [[buffer(3)]],device bfloat* qo [[buffer(4)]],device bfloat* ko [[buffer(5)]],device bfloat* vo [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(i>=p[0]*1536)return;uint row=i/1536,d=i%96,j=d%24,other=d<48?i+48:i-48;
    float theta=float(d%48<24?xy[row].x:xy[row].y)/pow(10000.0f,float(j)/24.0f);
    float c=cos(theta),s=sin(theta)*(d<48?-1.0f:1.0f);
    qo[i]=bfloat(q[i]*c+q[other]*s);ko[i]=bfloat(k[i]*c+k[other]*s);vo[i]=bfloat(v[i]);
}
#define GMLX_VATTN(NAME,GEMMA,HD,PAD) \
kernel void NAME(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],device bfloat* v [[buffer(2)]],device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float correction[32],normalizer[32],remap[32*64];vis_attention_impl<GEMMA,HD,PAD,16,false,false,bfloat>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);}
GMLX_VATTN(gmlx_gv_attention,true,72,80)
GMLX_VATTN(gmlx_mv_attention,false,96,96)
#undef GMLX_VATTN
