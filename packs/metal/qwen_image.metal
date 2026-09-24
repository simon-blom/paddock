// Qwen-Image-2.1. F32 residuals and Euler state; typed GGUF tensor projections
// and the shared online-softmax attention keep all model work on the GPU.
// Diffusion has contiguous K/V, not paged decode storage. Consume those
// tensor operands directly, without staging/transposing every key tile.
// Explicit materialization for compound BF16 expressions. A float(bfloat(x))
// cast alone can be contracted away in a fast-math multiply/add expression.
inline float qi_bf(float v) {
    uint bits=as_type<uint>(v);
    if((bits&0x7fffffff)>0x7f800000) return as_type<float>((bits&0xffff0000)|0x00400000);
    return as_type<float>((bits+0x7fff+((bits>>16)&1))&0xffff0000);
}
inline float qi_sigmoid_bf(float v) {
    float e=qi_bf(exp(abs(v)));
    float tail=qi_bf(1.0f/qi_bf(1.0f+e));
    return v<0 ? tail : qi_bf(1.0f-tail);
}
// Shared API supplies normalized RGBA. Composite only the tower's RGB view
// The VL tower has an F32 residual/position stream. Its quantized contractions
// must not inherit the language/DiT BF16 activation and result cuts. Decode a
// bounded 8 KiB weight tile; never expand a whole projection or the model.
kernel void qi_vision_affine(device const uchar* weights [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[64*32];uint K=p[0],N=p[1],M=p[2],n=g.x*32,m=g.y*32;
    auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)});
    auto b=tensor(tile,extents<int,64,32>(),array<int,2>{1,64});
    auto dst=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(32,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<K;base+=64){
        for(uint i=tid;i<64*32;i+=128){uint col=n+i/64,k=base+i%64;tile[i]=col<N&&k<K?mlx_affine_value(weights,K,N,ulong(col)*K+k):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input=a.slice<64,32>(base,m);op.run(input,b,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto target=dst.slice(n,m);acc.store(target);
}
// Shared API supplies normalized RGBA. Composite only the tower's RGB view
// over white; the VAE receives the original alpha. Patch order is the model's
// 2x2 spatial-merge order, not raster order. Full MLX conv uses C,T,H,W.
kernel void qi_vision_patches(device const float* rgba [[buffer(0)]],device float* patches [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint pw=p[0]/16,ph=p[1]/16,row=i/p[2],d=i%p[2];if(row>=pw*ph)return;
    uint temporal=p[2]/768,c=d/(temporal*256),pixel=d%256;
    uint xx=(row/4%(pw/2))*2+row%2,yy=(row/4/(pw/2))*2+row%4/2;
    ulong at=(ulong(yy*16+pixel/16)*p[0]+xx*16+pixel%16)*4;
    float alpha=clamp(rgba[at+3]*0.5f+0.5f,0.0f,1.0f);
    float value=(rgba[at+c]*0.5f+0.5f)*alpha+(1.0f-alpha);
    patches[i]=(value-as_type<float>(p[3+c]))/as_type<float>(p[6+c]);
}
kernel void qi_vision_position(device float* x [[buffer(0)]],device const float* bias [[buffer(1)]],device const float* pos [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1152,d=i%1152,pw=p[0],ph=p[1];if(row>=pw*ph)return;
    uint xx=(row/4%(pw/2))*2+row%2,yy=(row/4/(pw/2))*2+row%4/2;
    float sx=float(xx)*47.0f/float(pw-1),sy=float(yy)*47.0f/float(ph-1);
    uint x0=uint(sx),y0=uint(sy),x1=min(x0+1,47u),y1=min(y0+1,47u);float dx=sx-x0,dy=sy-y0;
    float a=pos[(y0*48+x0)*1152+d]*(1-dy)*(1-dx),b=pos[(y0*48+x1)*1152+d]*(1-dy)*dx;
    float c=pos[(y1*48+x0)*1152+d]*dy*(1-dx),e=pos[(y1*48+x1)*1152+d]*dy*dx;
    x[i]=(x[i]+bias[d])+((a+b)+c+e);
}
kernel void qi_text_splice(device const float* image [[buffer(0)]],device float* x [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*4096)return;ulong at=ulong(p[1])*4096+i;
    float feature=p[3]?qi_bf(image[i]):image[i];
    float value=p[2]?x[at]+feature:feature;x[at]=p[3]?qi_bf(value):value;
}
kernel void qi_text_mrope(device const float* x [[buffer(0)]],device const uchar* norm [[buffer(1)]],device const uint* pos [[buffer(2)]],device half* out [[buffer(3)]],device const float* value [[buffer(4)]],device half* values [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(g.y)*p[0]+g.x)*128;float ss=0;
    for(uint d=lane;d<128;d+=32)ss+=x[src+d]*x[src+d];
    float inv=rsqrt(simd_sum(ss)/128.0f+as_type<float>(p[3]));
    for(uint d=lane;d<128;d+=32){uint j=d%64,other=d<64?d+64:d-64,axis=j<60?j%3:0;
        float angle=float(pos[3*g.y+axis])*pow(as_type<float>(p[2]),-float(j)/64.0f);
        float a=x[src+d]*inv*weight(norm,p[1],d),b=x[src+other]*inv*weight(norm,p[1],other),co=cos(angle),si=sin(angle);
        float result;
        if(p[5]){a=qi_bf(a);b=qi_bf(b);co=qi_bf(co);si=qi_bf(si);result=qi_bf(qi_bf(a*co)+qi_bf((d<64?-b:b)*si));}
        else result=a*co+(d<64?-b:b)*si;
        out[src+d]=half(result);if(p[4])values[src+d]=half(value[src+d]);
    }
}
kernel void qi_attention(device half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* unused [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half scratch[1],probability[64*32];
    threadgroup float scores[64*32],maximum[32],denominator[32],correction[32];
    granite_prefill_tile<128,64,true,false,true>(q,k,v,meta,unused,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,scratch,probability,scores,maximum,denominator,correction);
}
kernel void qi_attention_deep(device half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* unused [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half scratch[1],probability[128*16];
    threadgroup float scores[128*16],maximum[16],denominator[16],correction[16];
    // Split the caller's BM32 metadata into two BM16 tasks. This bounds
    // explicit and instrumented shared storage below Apple10's 32 KiB.
    uint part=(g.y%2)*16,first=tiles[2*(g.y/2)]+part,count=min(16u,tiles[2*(g.y/2)+1]-part);
    granite_prefill_tile<128,128,true,false,true,16>(q,k,v,meta,unused,out,p,g.x,first,count,tid,scratch,probability,scores,maximum,denominator,correction);
}
kernel void qi_swiglu(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]*p[1]){ulong off=ulong(i/p[1])*2*p[1]+i%p[1];float v=x[off];y[i]=v/(1.0f+exp(-v))*x[off+p[1]];}}
kernel void qi_mlx_gate_up(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]*p[1]){ulong off=ulong(i/p[1])*2*p[1]+i%p[1];float v=x[off];y[i]=qi_bf(qi_bf(v*qi_sigmoid_bf(v))*x[off+p[1]]);}}
kernel void qi_mlx_residual(device float* x [[buffer(0)]],device const float* y [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])x[i]=mlx_bf(x[i]+y[i]);}
kernel void qi_mlx_swiglu(device float* x [[buffer(0)]],device const float* y [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])x[i]=qi_bf(qi_bf(x[i]*qi_sigmoid_bf(x[i]))*y[i]);}
kernel void qi_mlx_text_rope(device const float* x [[buffer(0)]],device const uchar* norm [[buffer(1)]],device const uint* meta [[buffer(2)]],device half* out [[buffer(3)]],device const float* value [[buffer(4)]],device half* values [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(g.y)*p[0]+g.x)*128;float ss=0;
    for(uint d=lane;d<128;d+=32)ss+=x[src+d]*x[src+d];
    float inv=rsqrt(simd_sum(ss)/128.0f+as_type<float>(p[3]));
    for(uint d=lane;d<128;d+=32){uint j=d%64,other=d<64?d+64:d-64;
        float angle=float(meta[2*g.y+1])*pow(as_type<float>(p[2]),-float(j)/64.0f);
        float a=qi_bf(x[src+d]*inv*weight(norm,p[1],d));
        float b=qi_bf(x[src+other]*inv*weight(norm,p[1],other));
        float co=qi_bf(cos(angle)),si=qi_bf(sin(angle));
        out[src+d]=half(qi_bf(qi_bf(a*co)+qi_bf((d<64?-b:b)*si)));
        if(p[4])values[src+d]=half(value[src+d]);
    }
}
// The text-conditioning reference uses F32 SDPA, even with BF16 projections.
// Small prompts do not need a matrix-tile probability cut to F16. Each SIMD
// group owns one causal query/head and retains its online softmax in F32.
kernel void qi_text_attention(device const half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],device float* out [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float acc[4]={0,0,0,0},maximum=-INFINITY,denominator=0;uint h=g.x,kh=h/4,row=g.y;
    for(uint t=0;t<=row;++t){float score=0;for(uint j=0;j<4;++j){uint d=lane+32*j;score+=float(q[(ulong(row)*32+h)*128+d])*float(k[(ulong(t)*8+kh)*128+d]);}
        score=simd_sum(score)*0.08838834764831844f;float high=max(maximum,score),old=exp(maximum-high),prob=exp(score-high);
        for(uint j=0;j<4;++j)acc[j]=acc[j]*old+prob*float(v[(ulong(t)*8+kh)*128+lane+32*j]);
        denominator=denominator*old+prob;maximum=high;}
    for(uint j=0;j<4;++j)out[(ulong(row)*32+h)*128+lane+32*j]=mlx_bf(acc[j]/denominator);
}
kernel void qi_f32_mm(device float* w [[buffer(0)]],device float* x [[buffer(1)]],device uint* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]]) {vis_project<32,float,float>(w,x,y,w,p,g);}
kernel void qi_activation(device float* x [[buffer(0)]], constant uint* p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if(i>=p[0])return;float v=x[i];
    if(p[1]==2){x[i]=qi_bf(v*qi_sigmoid_bf(v));return;}
    v=(p[1]&1)==0 ? v/(1.0f+exp(-v)) : vis_gelu(v);x[i]=(p[1]&2)?qi_bf(v):v;
}
kernel void qi_round_bf(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {if(i<p[0])x[i]=qi_bf(x[i]);}
kernel void qi_norm(device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    // width, weight type, epsilon, mode (0 zero-centred RMS, 1 modulated LN), weight offset.
    threadgroup float sum[8];uint lane=tid%32,sg=tid/32;float v=0;ulong start=ulong(row)*p[0];
    if(p[3]&1) {for(uint d=tid;d<p[0];d+=256)v+=x[start+d];}
    v=simd_sum(v);if(lane==0)sum[sg]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=0;for(uint j=0;j<8;++j)mean+=sum[j];mean/=float(p[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);v=0;
    for(uint d=tid;d<p[0];d+=256){float a=x[start+d]-mean;v+=a*a;}
    v=simd_sum(v);if(lane==0)sum[sg]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0;for(uint j=0;j<8;++j)ss+=sum[j];float inv=rsqrt(ss/float(p[0])+as_type<float>(p[2]));
    for(uint d=tid;d<p[0];d+=256){float v=(x[start+d]-mean)*inv,s=1.0f+weight(w,p[1],p[4]+d);
        if(p[3]&2){if(p[3]&1){v=qi_bf(v);s=qi_bf(s);}out[start+d]=qi_bf(v*s);}else out[start+d]=v*s;}
}
kernel void qi_gated_add(device float* x [[buffer(0)]],device const float* y [[buffer(1)]],device const float* mod [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]+=y[i]*precise::tanh(clamp(mod[p[2]+i%p[1]],-10.0f,10.0f));
}
kernel void qi_mlx_gated_add(device float* x [[buffer(0)]],device const float* y [[buffer(1)]],device const float* mod [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=qi_bf(x[i]+qi_bf(y[i]*qi_bf(precise::tanh(mod[p[2]+i%p[1]]))));
}
kernel void qi_mlx_head_rope(device const float* x [[buffer(0)]],device const uchar* norm [[buffer(1)]],device const int* pos [[buffer(2)]],device half* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(g.y)*p[0]+g.x)*128,dst=(ulong(g.y+p[2])*p[0]+g.x)*128;
    float ss=0;for(uint d=lane;d<128;d+=32)ss+=x[src+d]*x[src+d];
    float inv=rsqrt(simd_sum(ss)/128.0f+1e-6f);
    for(uint d=lane;d<128;d+=32){uint pair=d/2,axis=pair<8?0:(pair<36?1:2),j=pair-(axis==0?0:(axis==1?8:36));
        float angle=float(pos[3*g.y+axis])*pow(10000.0f,-float(j)/float(axis==0?8:28));
        // MLX fast.rms_norm rounds the normalized value before its weight.
        // Qwen3-VL's custom text norm has a different (single-round) contract.
        float a=qi_bf(qi_bf(x[src+d]*inv)*weight(norm,p[1],d)),b=qi_bf(qi_bf(x[src+(d^1)]*inv)*weight(norm,p[1],d^1));
        out[dst+d]=half(qi_bf(a*cos(angle)+(d%2?b:-b)*sin(angle)));}
}
kernel void qi_head_rope(device const float* x [[buffer(0)]],device const uchar* norm [[buffer(1)]],device const int* pos [[buffer(2)]],device half* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    // heads, norm type, destination row offset. Adjacent-pair, three-axis RoPE.
    ulong src=(ulong(g.y)*p[0]+g.x)*128,dst=(ulong(g.y+p[2])*p[0]+g.x)*128;
    float ss=0;for(uint d=lane;d<128;d+=32)ss+=x[src+d]*x[src+d];
    float inv=rsqrt(simd_sum(ss)/128.0f+1e-6f);
    for(uint d=lane;d<128;d+=32){uint pair=d/2,axis=pair<8?0:(pair<36?1:2),j=pair-(axis==0?0:(axis==1?8:36));
        float angle=float(pos[3*g.y+axis])*pow(10000.0f,-float(j)/float(axis==0?8:28));
        float a=x[src+d]*inv*weight(norm,p[1],d),b=x[src+(d^1)]*inv*weight(norm,p[1],d^1);
        out[dst+d]=half(a*cos(angle)+(d%2?b:-b)*sin(angle));}
}
kernel void qi_store_half(device const float* x [[buffer(0)]],device half* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])out[ulong(p[1])+i]=half(x[i]);}
kernel void qi_prefix_copy(device const uint* k [[buffer(0)]],device const uint* v [[buffer(1)]],device uint* joined_k [[buffer(2)]],device uint* joined_v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){joined_k[i]=k[i];joined_v[i]=v[i];}
}
kernel void qi_affine(device float* x [[buffer(0)]],device const float* scale [[buffer(1)]],device const float* bias [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {if(i<p[0])x[i]=x[i]*scale[i%p[1]]+bias[i%p[1]];}
kernel void qi_bias(device float* x [[buffer(0)]],device const float* bias [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])x[i]+=bias[i%p[1]];}
kernel void qi_guidance(device float* u [[buffer(0)]],device const float* c [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])u[i]+=as_type<float>(p[1])*(c[i]-u[i]);}
kernel void qi_noise(device float* out [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*64)return;uint4 c=uint4(p[3],0,(i%64)*p[0]+i/64,0);uint2 k=uint2(p[1],p[2]);
    for(uint r=0;r<10;++r){uint a=0xD2511F53u*c.x,b=0xCD9E8D57u*c.z;
        c=uint4(mulhi(0xCD9E8D57u,c.z)^c.y^k.x,b,mulhi(0xD2511F53u,c.x)^c.w^k.y,a);k+=uint2(0x9E3779B9u,0xBB67AE85u);}
    float u=float(c.x)*2.3283064e-10f+1.1641532e-10f;
    float v=float(c.y)*(2.3283064e-10f*6.2831855f)+(2.3283064e-10f*6.2831855f)/2.0f;
    out[i]=sqrt(-2.0f*precise::log(u))*precise::sin(v);
}
kernel void qi_vae_norm(device const float* x [[buffer(0)]],device const float* gamma [[buffer(1)]],device half* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sum[8];uint lane=tid%32,sg=tid/32;ulong start=ulong(row)*p[0];float ss=0;
    for(uint d=tid;d<p[0];d+=256)ss+=x[start+d]*x[start+d];ss=simd_sum(ss);if(lane==0)sum[sg]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);
    ss=0;for(uint j=0;j<8;++j)ss+=sum[j];float inv=sqrt(float(p[0]))/max(sqrt(ss),1e-12f);
    for(uint d=tid;d<p[0];d+=256){float v=x[start+d]*inv*gamma[d];out[start+d]=half(p[1]?v/(1.0f+exp(-v)):v);}
}
kernel void qi_im2row(device const half* x [[buffer(0)]],device half* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    // H,W,C,y0,ny,mode (0 same,1 up2,2 down2).
    uint mode=p[5],wo=mode==1?2*p[1]:(mode==2?p[1]/2:p[1]),ho=mode==1?2*p[0]:p[0];
    uint k=9*p[2];if(i>=p[4]*wo*k)return;uint row=i/k,col=i%k,t=col/p[2],c=col%p[2];
    int y=int(p[3]+row/wo),xx=int(row%wo);y=mode==2?2*y+int(t/3):y+int(t/3)-1;xx=mode==2?2*xx+int(t%3):xx+int(t%3)-1;
    bool inside=y>=0&&xx>=0&&y<int(ho)&&xx<int(mode==2?p[1]:wo);
    uint sy=mode==1?uint(y)/2:uint(y),sx=mode==1?uint(xx)/2:uint(xx);
    out[i]=inside?x[(ulong(sy)*p[1]+sx)*p[2]+c]:half(0);
}
kernel void qi_dupup(device float* out [[buffer(0)]],device const float* x [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=4*p[0]*p[1]*p[3])return;uint pix=i/p[3],o=i%p[3],y=pix/(2*p[1]),xx=pix%(2*p[1]);
    uint ci=(((o*p[4]+p[4]-1)*2+(y&1))*2+(xx&1))/p[5];out[i]+=x[(ulong(y/2)*p[1]+xx/2)*p[2]+ci];
}
kernel void qi_avgdown(device float* out [[buffer(0)]],device const float* x [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint H=p[0],W=p[1],ci=p[2],co=p[3],ft=p[4],fs=p[5];
    if(i>=H/fs*(W/fs)*co)return;
    uint group=ci*ft*fs*fs/co,pix=i/co,o=i%co,y=pix/(W/fs),xx=pix%(W/fs);float sum=0;
    for(uint j=0;j<group;++j){uint ix=o*group+j,dx=ix%fs;ix/=fs;uint dy=ix%fs;ix/=fs;uint t=ix%ft,c=ix/ft;
        if(t+1==ft)sum+=x[(ulong(y*fs+dy)*W+xx*fs+dx)*ci+c];}
    out[i]+=sum/float(group);
}
kernel void qi_pixels(device const float* x [[buffer(0)]],device uchar* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])out[i]=uchar(rint(clamp(x[i]*0.5f+0.5f,0.0f,1.0f)*255.0f));}
kernel void qi_split3(device const float* x [[buffer(0)]],device half* q [[buffer(1)]],device half* k [[buffer(2)]],device half* v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;ulong src=ulong(i/p[1])*3*p[1]+i%p[1];q[i]=half(x[src]*as_type<float>(p[2]));k[i]=half(x[src+p[1]]);v[i]=half(x[src+2*p[1]]);
}
kernel void qi_transpose(device const half* x [[buffer(0)]],device half* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]*p[1])out[ulong(i%p[1])*p[0]+i/p[1]]=x[i];}
kernel void qi_softmax(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];uint lane=tid%32,sg=tid/32;ulong off=ulong(row)*p[0];float mx=-INFINITY;
    for(uint j=tid;j<p[0];j+=256)mx=max(mx,x[off+j]*as_type<float>(p[1]));mx=simd_max(mx);if(!lane)red[sg]=mx;threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=-INFINITY;for(uint j=0;j<8;++j)mx=max(mx,red[j]);threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0;for(uint j=tid;j<p[0];j+=256){float v=exp(x[off+j]*as_type<float>(p[1])-mx);x[off+j]=v;ss+=v;}
    ss=simd_sum(ss);if(!lane)red[sg]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);ss=0;for(uint j=0;j<8;++j)ss+=red[j];
    for(uint j=tid;j<p[0];j+=256)x[off+j]/=ss;
}
