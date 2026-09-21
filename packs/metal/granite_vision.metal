// Granite 4.1 SigLIP2 + windowed Q-Former. Original kernels; IBM's graph,
// Pillow fixed-point resampling, and the shared MPP split-Q attention algorithm.
// Pixels never round-trip to the host; only geometry/index descriptors do.

// Independent SIMD arithmetic witness for Q-Former self/cross attention.
// Not selected by serving; shares neither MPP tiles nor probability layouts.
kernel void grv_qattention_check(device const half* q [[buffer(0)]],device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],device float* out [[buffer(3)]],device const uint2* bounds [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g.y,head=g.x;float a=0,b=0,hi=-INFINITY,den=0;
    for(uint t=bounds[row].x;t<bounds[row].y;++t){
        ulong qi=(ulong(row)*18+head)*64+lane,ki=(ulong(t)*18+head)*64+lane;
        float s=simd_sum(float(q[qi])*float(k[ki])+float(q[qi+32])*float(k[ki+32]))*0.125f;
        float nh=max(hi,s),old=exp(hi-nh),pr=exp(s-nh);hi=nh;den=den*old+pr;
        a=a*old+pr*float(v[ki]);b=b*old+pr*float(v[ki+32]);
    }
    ulong oi=(ulong(row)*18+head)*64+lane;out[oi]=a/den;out[oi+32]=b/den;
}

kernel void grv_patches(device const uchar* src [[buffer(0)]],device const int* coeff [[buffer(1)]],
    device half* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    // resized W/H, canvas W/H, coeff stride, first tile. Tile 0 is the overview.
    uint row=i/768,d=i%768,tiles=p[2]/384*(p[3]/384);if(row>=tiles*576)return;
    uint tile=row/576,patch=row%576;
    uint x=(tile%(p[2]/384))*384+(patch%24)*16+d%16;
    uint y=(tile/(p[2]/384))*384+(patch/24)*16+(d%256)/16,c=d/256;
    uint ox=(p[2]-p[0])/2,oy=(p[3]-p[1])/2;int v=0;
    if(x>=ox && x<ox+p[0] && y>=oy && y<oy+p[1]) {
        ulong b=ulong(y-oy)*p[4];int total=1<<21;
        for(int j=0;j<coeff[b+1];++j)total+=int(src[(ulong(coeff[b]+j)*p[0]+x-ox)*3+c])*coeff[b+2+j];
        v=clamp(total>>22,0,255);
    }
    out[ulong(p[5])*576*768+i]=half((float(v)/255.0f-as_type<float>(p[6+c]))/as_type<float>(p[9+c]));
}
kernel void grv_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*1152)x[i]+=pos[i%(576*1152)];
}
kernel void grv_half(device const float* x [[buffer(0)]],device half* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])out[i]=half(x[i]);
}
kernel void grv_heads(device const float* x [[buffer(0)]],device half* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/(16*80),head=i/80%16,d=i%80;if(row<p[0])out[i]=d<72?half(x[(ulong(row)*16+head)*72+d]):half(0);
}
kernel void grv_qattention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]],
    device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<false,64,64,18>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
// One gather does both windowing and downsampling. Average order is TL, TR,
// BL, BR (upstream area interpolation); learned positions/queries follow it.
kernel void grv_window(device const float* x [[buffer(0)]],device const float* learned [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint side=p[1],len=side*side,row=i/1152,d=i%1152;if(row>=p[0]*9*len)return;
    uint tile=row/(9*len),win=row/len%9,at=row%len;
    uint y=(win/3)*8+(at/side)*(8/side),xx=(win%3)*8+(at%side)*(8/side);
    float value=0;
    if(side==8)value=x[(ulong(tile)*576+y*24+xx)*1152+d];
    else if(p[2]==0xffffffffu) {
        for(uint dy=0;dy<2;++dy)for(uint dx=0;dx<2;++dx)value+=x[(ulong(tile)*576+(y+dy)*24+xx+dx)*1152+d];
        value*=0.25f;
    } else value=x[(ulong(tile)*576+(y+p[2]/2)*24+xx+p[2]%2)*1152+d];
    out[i]=value+learned[at*1152+d];
}
kernel void grv_unwindow(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1152,d=i%1152;if(row>=p[0]*144)return;
    uint tile=row/144,y=row%144/12,xx=row%12,win=y/4*3+xx/4,at=y%4*4+xx%4;
    out[i]=x[((ulong(tile)*9+win)*16+at)*1152+d];
}
// Every stream has the same packed row plan. Newline is unscaled at all taps.
kernel void grv_pack(device const float* x [[buffer(0)]],device const float* newline [[buffer(1)]],
    device const uint* indices [[buffer(2)]],device float* out [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[0],d=i%p[0];if(row<p[1])out[i]=indices[row]==0xffffffffu?newline[d]:x[ulong(indices[row])*p[0]+d];
}
kernel void grv_add(device const float* image [[buffer(0)]],device const uint* meta [[buffer(1)]],
    device float* x [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[0],d=i%p[0];if(row>=p[1])return;
    uint pos=meta[row*2+1];if(meta[row*2]==p[2] && pos>=p[3] && pos<p[3]+p[4])x[i]+=image[ulong(pos-p[3])*p[0]+d];
}
