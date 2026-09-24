// Original MiniCPM5 MLX kernels. Unpermuted NEOX RoPE, checkpoint BF16 KV,
// reuse bounded paged online-softmax and shared BF16 normalization/SwiGLU.
kernel void llama_mlx_rope(device float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],
    device bfloat* kc [[buffer(3)]],device bfloat* vc [[buffer(4)]],device const uint* meta [[buffer(5)]],device const uint* pages [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    uint width=p[0],kv=p[1],hd=p[2],halfhd=hd/2,row=i/((width+kv)/2),pair=i%((width+kv)/2);
    if(row>=p[3])return;
    bool query=pair<width/2;uint local=query?pair:pair-width/2,head=local/halfhd,j=local%halfhd;
    uint pos=meta[row*2+1];
    float angle=float(pos)*precise::exp2(-float(j)/float(halfhd)*precise::log2(as_type<float>(p[5])));
    float c=cos(angle),s=sin(angle);
    ulong base=ulong(row)*(query?width:kv)+head*hd;
    float a=query?q[base+j]:k[base+j],b=query?q[base+j+halfhd]:k[base+j+halfhd];
    float x=mlx_bf(fma(a,c,-b*s)),y=mlx_bf(fma(a,s,b*c));
    if(query){q[base+j]=x;q[base+j+halfhd]=y;}
    else {
        uint physical=pages[meta[row*2]*p[4]+pos/16]*16+pos%16;
        ulong out=ulong(physical)*kv+head*hd;
        kc[out+j]=bfloat(x);kc[out+j+halfhd]=bfloat(y);
        vc[out+j]=bfloat(v[base+j]);vc[out+j+halfhd]=bfloat(v[base+j+halfhd]);
    }
}
kernel void llama_mlx_decode(device const float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[8*32],prob[8*32],highs[8],sums[8];
    gemma_decode<128,8,true,bfloat>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}
// GGUF retains F16 KV and F32 output, sharing each KV tile across all eight
// query heads. The graph admits this only for HD128 with 1/sqrt(128) scale.
kernel void llama_decode(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[8*32],prob[8*32],highs[8],sums[8];
    gemma_decode<128,8,true,half>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}
kernel void llama_mlx_prefill(device float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float kv[16*128],prob[16*32],scores[16*32],maximum[32],denom[32],correction[32];
    gemma_prefill<128,16,32,float,false,true,bfloat,true>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction);
}
