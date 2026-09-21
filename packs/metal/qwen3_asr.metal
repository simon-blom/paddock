// Original native Metal Qwen3-ASR glue. Reference graph: Qwen's published
// encoder and our CUDA family. No CPU convolution and no NxN score plane.
kernel void qasr_extract(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0])out[i]=x[ulong(p[1])+i];}
kernel void qasr_decode(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
 device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],
 constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float scores[64],prob[64],highs[2],sums[2];
 gemma_decode<128,2,true>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}
// Conv output is [chunk,time,freq,channel]. Gather retains the checkpoint's
// channel-major K order, avoiding any floating-point weight conversion.
// p: input frequency/time/channels, chunk count, stage (0 = mel), first chunk.
template<typename T> inline void qasr_gather(device const float* x,device T* out,constant uint* p,uint i) {
 uint oh=(p[0]+1)/2,ow=(p[1]+1)/2,K=9*p[2],r=i/K,z=i%K;
 if(r>=p[3]*oh*ow)return;
 uint chunk=r/(oh*ow),fy=r%oh,tx=r/oh%ow,c=z/9;
 int yy=int(fy*2+z%9/3)-1,xx=int(tx*2+z%3)-1;
 float a=0;
 if(yy>=0 && yy<int(p[0]) && xx>=0 && xx<int(p[1])) {
  ulong at=p[4]==0?(ulong(chunk+p[5])*100+uint(xx))*128+uint(yy)
    :((ulong(chunk)*p[1]+uint(xx))*p[0]+uint(yy))*p[2]+c;
  a=x[at];
 }
 out[i]=T(a);
}
kernel void qasr_conv_rows(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {qasr_gather(x,out,p,i);}
kernel void qalign_conv_rows(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {qasr_gather(x,out,p,i);}
// Conv_out expects [channel,freq], while the fused conv output stores
// contiguous channel rows. Pure GPU permutation, no precision boundary.
kernel void qasr_flatten(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*13*7680)return;uint r=i/7680,z=i%7680;
 out[i]=x[ulong(r)*7680+(z%16)*480+z/16];
}
kernel void qalign_flatten(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*13*7680)return;uint r=i/7680,z=i%7680;out[i]=bfloat(x[ulong(r)*7680+(z%16)*480+z/16]);
}
// Strict F16/BF16 weight x F32 activation projections with a bias/residual
// epilogue. erf-GELU is not the tanh GELU used by the shared ViT projection.
kernel void qasr_half_mm(device half* w [[buffer(0)]],device float* x [[buffer(1)]],
 device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]]) {vis_project<32,half,float>(w,x,out,bias,p,g);}
kernel void qasr_position(device const float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
 device float* out [[buffer(2)]],device const uint* map [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*1024)return;uint row=map[p[1]+i/1024];if(row==UINT_MAX)return;
 out[ulong(row)*1024+i%1024]=x[i]+pos[(i/1024%13)*1024+i%1024];
}
// Full padded KV tiles are backed by guard rows. Window boundaries are
// logical masks; a contraction may read the next window but cannot use it.
kernel void qasr_heads(device const float* x [[buffer(0)]],device half* q [[buffer(1)]],
 device half* k [[buffer(2)]],device half* v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 uint r=i/1024,d=i%1024;if(r>=p[0]+64)return;
 q[i]=r<p[0]?half(x[ulong(r)*3072+d]):half(0);
 k[i]=r<p[0]?half(x[ulong(r)*3072+1024+d]):half(0);
 v[i]=r<p[0]?half(x[ulong(r)*3072+2048+d]):half(0);
}
kernel void qasr_attention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]],
 device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float correction[32],normalizer[32],remap[32*64];
 vis_attention_impl<false,64,64,16,true>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
// Test-only independent SIMD online attention: no TensorOps and no host math.
// Checks the packed-window masks and tails, not full-model external parity.
kernel void qasr_attention_check(device const half* q [[buffer(0)]],device const half* k [[buffer(1)]],
 device const half* v [[buffer(2)]],device float* out [[buffer(3)]],device const uint2* bounds [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 ulong base=(ulong(g.y)*16+g.x)*64;float a=0,b=0,high=-INFINITY,den=0;
 for(uint t=bounds[g.y].x;t<bounds[g.y].y;++t){ulong at=(ulong(t)*16+g.x)*64;
  float score=simd_sum(float(q[base+lane])*float(k[at+lane])+float(q[base+lane+32])*float(k[at+lane+32]))*0.125f;
  float next=max(high,score),old=exp(high-next),p=exp(score-next);den=den*old+p;high=next;
  a=a*old+p*float(v[at+lane]);b=b*old+p*float(v[at+lane+32]);
 }out[base+lane]=a/den;out[base+lane+32]=b/den;
}
