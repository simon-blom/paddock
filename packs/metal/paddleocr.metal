// PaddleOCR-VL graph/layout glue. Original kernels, shared TensorOps math.
// Text axes use sectioned NeoX pairs; attention visibility remains causal by
// sequence index, not equal-temporal-position image-block visibility.
kernel void pocr_rope(device float* x [[buffer(0)]],device const uint4* pos [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint pair=i%64,head=i/64%p[0],row=i/(64*p[0]);if(row>=p[1])return;
    uint axis=pair<16?pos[row].x:pair<40?pos[row].y:pos[row].z;
    float angle=float(axis)*pow(500000.0f,-float(pair)/64.0f),c=cos(angle),s=sin(angle);
    ulong at=(ulong(row)*p[0]+head)*128+pair;float a=x[at],b=x[at+64];
    x[at]=a*c-b*s;x[at+64]=a*s+b*c;
}
kernel void pocr_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
 device half* kc [[buffer(2)]],device half* vc [[buffer(3)]],device const uint2* meta [[buffer(4)]],
 device const uint* pages [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    uint row=i/256;if(row>=p[0])return;uint2 at=meta[row];
    ulong out=(ulong(pages[at.x*p[1]+at.y/16])*16+at.y%16)*256+i%256;
    kc[out]=half(k[i]);vc[out]=half(v[i]);
}
// Complete GPU Pillow bicubic: horizontal pass is shared with Gemma/Granite;
// this pass performs vertical u8 rounding, normalization and merged patchify.
kernel void pocr_patches(device const uchar* src [[buffer(0)]],device const int* cy [[buffer(1)]],
 device half* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint pw=p[0]/14,row=i/588,d=i%588;if(row>=pw*(p[1]/14))return;
    uint py=row/4/(pw/2)*2+row%4/2,px=row/4%(pw/2)*2+row%2;
    uint y=py*14+d%196/14,x=px*14+d%14,c=d/196;ulong b=ulong(y)*p[2];int sum=1<<21;
    for(int j=0;j<cy[b+1];++j)sum+=int(src[(ulong(cy[b]+j)*p[0]+x)*3+c])*cy[b+2+j];
    float v=float(clamp(sum>>22,0,255))/255.0f;
    out[ulong(p[3])*588+i]=half((v-0.5f)/0.5f);
}
kernel void pocr_patch_mm(device float* w [[buffer(0)]],device half* x [[buffer(1)]],
 device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]]) {vis_project<32,float,half>(w,x,out,bias,p,g);}
// align_corners=False learned 27x27 table. Patch rows are merged TL,TR,BL,BR.
kernel void pocr_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1152,d=i%1152,pw=p[0],ph=p[1];if(row>=pw*ph)return;
    uint yy=row/4/(pw/2)*2+row%4/2,xx=row/4%(pw/2)*2+row%2;
    float y=max(0.0f,(float(yy)+0.5f)*27.0f/float(ph)-0.5f);
    float z=max(0.0f,(float(xx)+0.5f)*27.0f/float(pw)-0.5f);
    uint y0=min(uint(y),26u),x0=min(uint(z),26u),y1=min(y0+1,26u),x1=min(x0+1,26u);
    float a=pos[(y0*27+x0)*1152+d],b=pos[(y0*27+x1)*1152+d];
    float c=pos[(y1*27+x0)*1152+d],e=pos[(y1*27+x1)*1152+d];
    x[ulong(p[2])*1152+i]+=(a*(1.0f-(z-x0))+b*(z-x0))*(1.0f-(y-y0))+(c*(1.0f-(z-x0))+e*(z-x0))*(y-y0);
}
kernel void pocr_heads(device const float* x [[buffer(0)]],device half* out [[buffer(1)]],
 device const uint2* xy [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/1280,h=i/80%16,d=i%80;if(row>=p[0]+64)return;float v=0;
    if(row<p[0] && d<72) {
        ulong at=(ulong(row)*16+h)*72;
        if(p[1]) {
            uint pair=d%36;
            float a=float(pair<18?xy[row].y:xy[row].x)*pow(10000.0f,-float(pair%18)/18.0f);
            v=x[at+d]*cos(a)+x[at+(d+36)%72]*sin(a)*(d<36?-1.0f:1.0f);
        } else v=x[at+d];
    } out[i]=half(v);
}
// Fully backed final KV tiles avoid an MPP partial-K contraction consuming
// undefined inactive fragments. Logical scores retain the exact image mask;
// the last image has 64 explicitly zeroed guard rows, never host padding.
kernel void pocr_attention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]],
 device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<false,72,80,16,true>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
kernel void pocr_gelu(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mv_gelu_value(x[i]);
}
kernel void pocr_extract(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[1]*1024)out[i]=x[ulong(p[0])*1024+i];
}
