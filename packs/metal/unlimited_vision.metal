// DeepEncoder layout and relative-position work. Original, GPU-only.
// Pixels use Pillow's Q22 separable bicubic; floating tables use AA bicubic.
kernel void uov_copy(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])out[i]=x[i];}
kernel void uov_patches(device const uchar* src [[buffer(0)]],device const int* cy [[buffer(1)]],
 device half* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    // p: resized W/H, view px, coeff stride, view col/row, pad x/y, pad mode, output row
    uint side=p[2]/16,row=i/768,d=i%768;if(row>=side*side)return;
    uint x=row%side*16+d%16,y=row/side*16+d%256/16,c=d/256;int v=127;
    bool inside=true;
    if(p[8]) {inside=x>=p[6] && x<p[6]+p[0] && y>=p[7] && y<p[7]+p[1];x-=p[6];y-=p[7];}
    else {x+=p[4]*p[2];y+=p[5]*p[2];}
    if(inside) {ulong b=ulong(y)*p[3];int sum=1<<21;
        for(int j=0;j<cy[b+1];++j)sum+=int(src[(ulong(cy[b]+j)*p[0]+x)*3+c])*cy[b+2+j];
        v=clamp(sum>>22,0,255);}
    out[ulong(p[9])*768+i]=half((float(v)/255.0f-0.5f)/0.5f);
}
kernel void uov_mm(device half* w [[buffer(0)]],device float* x [[buffer(1)]],device uint* out [[buffer(2)]],
 device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
    vis_project<32,half,float>(w,x,out,bias,p,g);
}
kernel void uov_mm_f32(device float* w [[buffer(0)]],device float* x [[buffer(1)]],device uint* out [[buffer(2)]],
 device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
    vis_project<32,float,float>(w,x,out,bias,p,g);
}
inline float uov_cubic(float x) {
    x=abs(x);return x<1?(1.5f*x-2.5f)*x*x+1:x<2?((-0.5f*x+2.5f)*x-4)*x+2:0;
}
// p: source grid, destination grid, channels, source CLS rows, axis(0=H,1=V)
kernel void uov_resize_pos(device const float* src [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint ch=i%p[2],x=i/p[2]%p[1],y=i/(p[2]*p[1]),height=p[4]?p[1]:p[0];if(y>=height)return;
    uint axis=p[4]?y:x;float scale=float(p[0])/p[1],filter=max(scale,1.0f);
    float center=(float(axis)+0.5f)*scale;int lo=max(0,int(floor(center-2*filter+0.5f))),hi=min(int(p[0]),int(floor(center+2*filter+0.5f)));
    float sum=0,total=0;
    for(int j=lo;j<hi;++j){float w=uov_cubic((float(j)+0.5f-center)/filter);
        ulong row=p[4]?ulong(j)*p[1]+x:ulong(y)*p[0]+j;
        total+=src[(row+p[3])*p[2]+ch]*w;sum+=w;}
    out[i]=total/sum;
}
kernel void uov_add_pos(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])x[i]+=pos[i%(p[2]*p[1])];
}
// Window partition operates after LN. Zero pads still acquire QKV bias and
// participate in attention, as trained; they are not padding-masked away.
kernel void uov_partition(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint grid=p[0],windows=(grid+13)/14,row=i/768,d=i%768;if(row>=p[1]*windows*windows*196)return;
    uint image=row/(windows*windows*196),win=row/196%(windows*windows),local=row%196;
    uint y=win/windows*14+local/14,z=win%windows*14+local%14;
    out[i]=y<grid && z<grid?x[((ulong(image)*grid+y)*grid+z)*768+d]:0;
}
kernel void uov_unpartition(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint grid=p[0],windows=(grid+13)/14,row=i/768,d=i%768;if(row>=p[1]*grid*grid)return;
    uint image=row/(grid*grid),y=row/grid%grid,z=row%grid;
    ulong at=((ulong(image)*windows+y/14)*windows+z/14)*196+(y%14)*14+z%14;
    out[i]=x[at*768+d];
}
kernel void uov_heads(device const float* qkv [[buffer(0)]],device half* q [[buffer(1)]],
 device half* k [[buffer(2)]],device half* v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint e=p[1],row=i/e,d=i%e;if(row>=p[0]+64)return;
    q[i]=row<p[0]?half(qkv[ulong(row)*e*3+d]):half(0);
    k[i]=row<p[0]?half(qkv[ulong(row)*e*3+e+d]):half(0);
    v[i]=row<p[0]?half(qkv[ulong(row)*e*3+2*e+d]):half(0);
}
// Decomposed SAM bias takes O(rows * side * heads), never an N-squared slab.
// p: rows, side, learned relative rows. Q uses the unscaled F32 projection.
kernel void uov_relative(device const float* qkv [[buffer(0)]],device const float* h [[buffer(1)]],
 device const float* w [[buffer(2)]],device float* rh [[buffer(3)]],device float* rw [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g.y,head=g.x/p[1],key=g.x%p[1],side=p[1],local=row%(side*side);
    float yh=max(0.0f,(float(local/side+side-1-key)+0.5f)*float(p[2])/float(2*side-1)-0.5f);
    float xw=max(0.0f,(float(local%side+side-1-key)+0.5f)*float(p[2])/float(2*side-1)-0.5f);
    uint h0=min(uint(yh),p[2]-1),h1=min(h0+1,p[2]-1),w0=min(uint(xw),p[2]-1),w1=min(w0+1,p[2]-1);
    float a=0,b=0;
    for(uint d=lane;d<64;d+=32){float x=qkv[ulong(row)*2304+head*64+d];
        a+=x*(h[h0*64+d]*(1-(yh-h0))+h[h1*64+d]*(yh-h0));
        b+=x*(w[w0*64+d]*(1-(xw-w0))+w[w1*64+d]*(xw-w0));}
    a=simd_sum(a);b=simd_sum(b);
    if(lane==0){ulong at=(ulong(row)*12+head)*side+key;rh[at]=a;rw[at]=b;}
}
kernel void uov_sam_attention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]],
 device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],device const float* rh [[buffer(5)]],
 device const float* rw [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<false,64,64,12,true,true>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap,rh,rw,p[1]);
}
// Test-only dispatch: independent one-key-at-a-time SIMD attention, no
// TensorOps, shared score plane or CPU math. Checks relative-bias indexing,
// online rescaling, image isolation and ragged key/query tails.
kernel void uov_sam_check(device const half* q [[buffer(0)]],device const half* k [[buffer(1)]],
 device const half* v [[buffer(2)]],device float* out [[buffer(3)]],device const float* rh [[buffer(4)]],
 device const float* rw [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]]) {
    uint row=g.y,head=g.x,side=p[0],n=side*side,first=row/n*n;
    ulong at=ulong(row)*768+head*64+lane,relative=(ulong(row)*12+head)*side;
    float q0=q[at],q1=q[at+32],a0=0,a1=0,peak=-INFINITY,den=0;
    for(uint j=0;j<n;++j){ulong key=ulong(first+j)*768+head*64+lane;
        float score=simd_sum(q0*float(k[key])+q1*float(k[key+32]))*0.125f+rh[relative+j/side]+rw[relative+j%side];
        float next=max(peak,score),rescale=exp(peak-next),prob=exp(score-next);
        a0=a0*rescale+prob*float(v[key]);a1=a1*rescale+prob*float(v[key+32]);
        den=den*rescale+prob;peak=next;
    }
    out[at]=a0/den;out[at+32]=a1/den;
}
kernel void uov_clip_attention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]],
 device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<false,64,64,16,true>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
kernel void uov_activation(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float v=x[i];x[i]=p[1]?v/(1+exp(-1.702f*v)):mv_gelu_value(v);}
}
// Convolution im2row is channel-major, matching [kx,ky,Cin,Cout] GGUF bytes.
kernel void uov_conv_rows(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint side=p[0],dst=(side+p[2]-1)/p[2],k=9*p[1],row=i/k,d=i%k;if(row>=p[3]*dst*dst)return;
    uint image=row/(dst*dst);int y=int(row/dst%dst*p[2])+int(d%9/3)-1,z=int(row%dst*p[2])+int(d%3)-1;
    out[i]=y>=0 && z>=0 && y<int(side) && z<int(side)?x[((ulong(image)*side+y)*side+z)*p[1]+d/9]:0;
}
kernel void uov_clip_embed(device const float* sam [[buffer(0)]],device const float* cls [[buffer(1)]],
 device const float* pos [[buffer(2)]],device const float* clspos [[buffer(3)]],device float* out [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1024,d=i%1024,n=p[0]+1;if(row>=n*p[1])return;uint local=row%n,image=row/n;
    out[i]=local?sam[(ulong(image)*p[0]+local-1)*1024+d]+pos[(local-1)*1024+d]:cls[d]+clspos[d];
}
kernel void uov_concat(device const float* clip [[buffer(0)]],device const float* sam [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/2048,d=i%2048;if(row>=p[0]*p[1])return;uint image=row/p[0],local=row%p[0];
    out[i]=d<1024?clip[(ulong(image)*(p[0]+1)+local+1)*1024+d]:sam[ulong(row)*1024+d-1024];
}
// Copy one projected view into the already allocated final image layout.
// Local tiles interleave by spatial row; global follows them, then separator.
kernel void uov_assemble(device const float* views [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint side=p[0],row=i/1280,d=i%1280;if(row>=side*side*p[1])return;
    uint view=row/(side*side)+p[2],y=row/side%side,x=row%side;
    ulong dst=p[5]+(ulong(view/p[3])*side+y)*(p[3]*side+1)+(view%p[3])*side+x;
    out[dst*1280+d]=views[i];
}
kernel void uov_separators(device const float* nl [[buffer(0)]],device const float* sep [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1280,d=i%1280,local=p[0]*10;if(row>=local+17)return;
    ulong dest=row<local?ulong(row)*(p[1]*10+1)+p[1]*10:
        row<local+16?p[2]+ulong(row-local)*17+16:p[2]+272;
    out[dest*1280+d]=row<local+16?nl[d]:sep[d];
}
