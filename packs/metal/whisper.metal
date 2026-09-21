// Original native Whisper kernels. Graph references: OpenAI Whisper and
// vLLM's resident cross-attention cache. Encoder uses our split-Q TensorOps
// implementation; decode splits the fixed 1500-key window and merges LSE.
// All weights stay F16, residuals and reductions F32, slot KV F16.
kernel void wh_widen(device const half* x [[buffer(0)]], device float* y [[buffer(1)]],
 constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) { if(i<p[0]) y[i]=float(x[i]); }

// Four independent output rows per threadgroup, all active requests reuse
// each weight load. Compile-time row tiles keep accumulation in registers;
// runtime array indexing can otherwise force thread-local accesses.
template<uint R> inline void wh_mv_impl(device const half* w,device const float* x,
 device float* out,device const float* bias,constant uint* p,uint g,uint lane,uint sg) {
 uint n=g*4+sg;if(n>=p[1])return;float v[R]={};
 for(uint k=lane;k<p[0];k+=32){float a=float(w[ulong(n)*p[0]+k]);
  #pragma unroll
  for(uint r=0;r<R;++r)if(r<p[2])v[r]=fma(a,x[ulong(r)*p[0]+k],v[r]);}
 #pragma unroll
 for(uint r=0;r<R;++r)if(r<p[2]){float a=simd_sum(v[r]);
  if(lane==0){if(p[3])a+=bias[n];ulong ix=ulong(r)*p[1]+n;
   if(p[3]==2)a+=out[ix];if(p[3]==3)a=mv_gelu_value(a);out[ix]=a;}}
}
#define WH_MV(R) \
kernel void wh_mv##R(device const half* w [[buffer(0)]],device const float* x [[buffer(1)]], \
 device float* out [[buffer(2)]],device const float* b [[buffer(3)]],constant uint* p [[buffer(4)]], \
 uint g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {wh_mv_impl<R>(w,x,out,b,p,g,lane,sg);}
WH_MV(1)
WH_MV(2)
WH_MV(4)
WH_MV(8)
WH_MV(16)
#undef WH_MV
// Same strict mixed contraction as qasr_half_mm; bias+erf-GELU now stays
// in the producer registers. Never substitute the tanh GELU epilogue.
kernel void wh_mm(device half* w [[buffer(0)]],device float* x [[buffer(1)]],
 device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]]) {
 uint K=p[0],N=p[1],M=p[2],n=g.x*64,m=g.y*32;
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,64,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();op.run(a,b,acc);
 for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
  if(it.is_valid_element() && row<M && col<N){float v=*it;ulong ix=ulong(row)*N+col;
   if(p[3])v+=bias[col];if(p[3]==2)v+=out[ix];if(p[3]==3)v=mv_gelu_value(v);out[ix]=v;}}
}
// K keeps the checkpoint's [channel,tap] order; no host convolution or
// reordered numeric weights. Padding is per clip, including the 30 s tail.
kernel void wh_conv_rows(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 uint K=p[0]*3,T=p[1],O=(T+p[2]-1)/p[2],row=i/K,k=i%K;
 if(row>=O*p[3])return;uint b=row/O,t=row%O,c=k/3;
 int src=int(t*p[2]+k%3)-1;
 out[i]=src>=0 && src<int(T)?x[(ulong(b)*T+uint(src))*p[0]+c]:0;
}
kernel void wh_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]*1280)x[i]+=pos[i%(1500*1280)];}
kernel void wh_embed(device const half* w [[buffer(0)]],device const float* pos [[buffer(1)]],
 device const uint* tokens [[buffer(2)]],device const uint* positions [[buffer(3)]],
 device float* x [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0]*1280)x[i]=float(w[ulong(tokens[i/1280])*1280+i%1280])+pos[ulong(positions[i/1280])*1280+i%1280];
}
kernel void wh_split(device const float* x [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* q [[buffer(2)]],device float* k [[buffer(3)]],device float* v [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 uint row=i/1280,d=i%1280;if(row>=p[0]+64)return;
 q[i]=row<p[0]?x[ulong(row)*3840+d]+b[d]:0;
 k[i]=row<p[0]?x[ulong(row)*3840+1280+d]:0;
 v[i]=row<p[0]?x[ulong(row)*3840+2560+d]+b[1280+d]:0;
}
kernel void wh_attention(device float* q [[buffer(0)]],device float* k [[buffer(1)]],device float* v [[buffer(2)]],
 device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float correction[32],normalizer[32],remap[32*64];
 vis_attention_impl<false,64,64,20,true,false,float>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
// p = rows, text capacity. Each cache is one layer, [slot,time,1280].
kernel void wh_append(device const float* x [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* q [[buffer(2)]],device half* k [[buffer(3)]],device half* v [[buffer(4)]],
 device const uint* slots [[buffer(5)]],device const uint* positions [[buffer(6)]],
 constant uint* p [[buffer(7)]],uint i [[thread_position_in_grid]]) {
 uint row=i/1280,d=i%1280;if(row>=p[0])return;
 ulong dst=(ulong(slots[row])*p[1]+positions[row])*1280+d;
 q[i]=x[ulong(row)*3840+d]+b[d];
 k[dst]=half(x[ulong(row)*3840+1280+d]);v[dst]=half(x[ulong(row)*3840+2560+d]+b[1280+d]);
}
// Fused cross K|V is projected once at admission. Bias is V-only.
kernel void wh_cross_store(device const float* x [[buffer(0)]],device const float* b [[buffer(1)]],
 device half* k [[buffer(2)]],device half* v [[buffer(3)]],device const uint* slots [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 uint row=i/1280,d=i%1280;if(row>=p[0]*1500)return;
 ulong dst=(ulong(slots[row/1500])*1500+row%1500)*1280+d;
 k[dst]=half(x[ulong(row)*2560+d]);v[dst]=half(x[ulong(row)*2560+1280+d]+b[d]);
}
// Four SIMD groups split a 256-key slice. Each owns complete QK dot
// products and its running softmax. Scratch is O(head_dim), never O(T²).
kernel void wh_decode(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],
 device const half* v [[buffer(2)]],device const uint* slots [[buffer(3)]],
 device const uint* positions [[buffer(4)]],device float* partial [[buffer(5)]],
 constant uint* p [[buffer(6)]],uint3 g [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float hi[4],den[4],values[256];
 uint len=p[1]?p[0]:positions[g.y]+1,start=g.z*256,end=min(start+256,len);
 ulong qi=ulong(g.y)*1280+g.x*64;
 float qa=q[qi+lane],qb=q[qi+lane+32],m=-INFINITY,s=0,a=0,b=0;
 for(uint t=start+sg;t<end;t+=4){ulong at=(ulong(slots[g.y])*p[0]+t)*1280+g.x*64;
  float z=simd_sum(qa*float(k[at+lane])+qb*float(k[at+lane+32]))*0.125f;
  float high=max(m,z),old=isfinite(m)?exp(m-high):0,prob=exp(z-high);s=s*old+prob;m=high;
  a=a*old+prob*float(v[at+lane]);b=b*old+prob*float(v[at+lane+32]);}
 if(lane==0){hi[sg]=m;den[sg]=s;}values[sg*64+lane]=a;values[sg*64+lane+32]=b;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg==0){float high=max(max(hi[0],hi[1]),max(hi[2],hi[3])),total=0,aa=0,bb=0;
  for(uint j=0;j<4;++j){float f=isfinite(hi[j])?exp(hi[j]-high):0;total+=den[j]*f;aa+=values[j*64+lane]*f;bb+=values[j*64+lane+32]*f;}
  ulong dst=((ulong(g.y)*20+g.x)*p[2]+g.z)*66;
  partial[dst+lane]=aa;partial[dst+lane+32]=bb;if(lane==0){partial[dst+64]=high;partial[dst+65]=total;}}
}
kernel void wh_merge(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 ulong src=(ulong(g.y)*20+g.x)*p[0]*66;float m=-INFINITY,s=0,a=0,b=0;
 for(uint j=0;j<p[0];++j){ulong at=src+j*66;float h=partial[at+64],high=max(m,h);
  float old=isfinite(m)?exp(m-high):0,f=isfinite(h)?exp(h-high):0;
  s=s*old+partial[at+65]*f;a=a*old+partial[at+lane]*f;b=b*old+partial[at+lane+32]*f;m=high;}
 ulong dst=ulong(g.y)*1280+g.x*64;out[dst+lane]=a/s;out[dst+lane+32]=b/s;
}
// Alignment only. Softmax over real audio frames, not padded encoder time.
// Output is directly [selected head, boundary row, frame], so no host transpose.
// p: slot, used frames, total boundary rows, absolute chunk position.
kernel void wh_align_probs(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],
 device const uint* heads [[buffer(2)]],device float* out [[buffer(3)]],
 constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
 if(p[3]+g.y<3)return;
 threadgroup float logits[1500],hi[8],den[8];
 uint h=heads[g.x],n=p[1];ulong qi=ulong(g.y)*1280+h*64;
 float qa=q[qi+lane],qb=q[qi+lane+32],m=-INFINITY;
 for(uint t=sg;t<n;t+=8){ulong at=(ulong(p[0])*1500+t)*1280+h*64;
  float z=simd_sum(qa*float(k[at+lane])+qb*float(k[at+lane+32]))*0.125f;
  if(lane==0)logits[t]=z;m=max(m,z);}
 if(lane==0)hi[sg]=m;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 m=-INFINITY;for(uint j=0;j<8;++j)m=max(m,hi[j]);
 float sum=0;for(uint t=tid;t<n;t+=256){float v=exp(logits[t]-m);logits[t]=v;sum+=v;}
 sum=simd_sum(sum);if(lane==0)den[sg]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 sum=0;for(uint j=0;j<8;++j)sum+=den[j];
 ulong dst=(ulong(g.x)*p[2]+p[3]+g.y-3)*n;
 for(uint t=tid;t<n;t+=256)out[dst+t]=logits[t]/sum;
}
// Read back only picks and confidence statistics on ordinary decode steps.
kernel void wh_pick(device const float* x [[buffer(0)]],device uint* ids [[buffer(1)]],
 device float* stats [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float hi[8],sum[8];threadgroup uint idx[8],bad[8];
 ulong base=ulong(row)*p[0];float a=-INFINITY;uint invalid=0;
 for(uint i=tid;i<p[0];i+=256){float v=x[base+i];a=max(a,v);invalid|=uint(isnan(v)||v==INFINITY);}
 a=simd_max(a);invalid=simd_or(invalid);if(lane==0){hi[sg]=a;bad[sg]=invalid;}
 threadgroup_barrier(mem_flags::mem_threadgroup);a=-INFINITY;invalid=0;
 for(uint j=0;j<8;++j){a=max(a,hi[j]);invalid|=bad[j];}
 uint best=UINT_MAX;float den=0;
 for(uint i=tid;i<p[0];i+=256){float v=x[base+i];if(v==a)best=min(best,i);den+=exp(v-a);}
 best=simd_min(best);den=simd_sum(den);if(lane==0){idx[sg]=best;sum[sg]=den;}
 threadgroup_barrier(mem_flags::mem_threadgroup);best=UINT_MAX;den=0;
 for(uint j=0;j<8;++j){best=min(best,idx[j]);den+=sum[j];}
 float second=-INFINITY;for(uint i=tid;i<p[0];i+=256)if(i!=best)second=max(second,x[base+i]);
 second=simd_max(second);if(lane==0)hi[sg]=second;
 threadgroup_barrier(mem_flags::mem_threadgroup);second=-INFINITY;for(uint j=0;j<8;++j)second=max(second,hi[j]);
 uint alt=UINT_MAX;for(uint i=tid;i<p[0];i+=256)if(i!=best && x[base+i]==second)alt=min(alt,i);
 alt=simd_min(alt);if(lane==0)idx[sg]=alt;
 threadgroup_barrier(mem_flags::mem_threadgroup);alt=UINT_MAX;for(uint j=0;j<8;++j)alt=min(alt,idx[j]);
 if(tid==0){ids[row*3]=best;ids[row*3+1]=alt;ids[row*3+2]=invalid||!isfinite(a)||!isfinite(den)||den<=0;
  float z=a+log(den);stats[row*3]=a-z;stats[row*3+1]=second-z;stats[row*3+2]=exp(x[base+p[1]]-z);}
}
// Whisper timestamp grammar. Mask, then compare total timestamp mass to
// best text probability. No history scan and no logits readback for rules.
kernel void wh_rules(device float* logits [[buffer(0)]],device const uint2* rules [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint flags=rules[row].x;if(!(flags&1))return;device float* x=logits+ulong(row)*p[0];
 bool begin=flags&2,last=flags&4,penult=flags&8;
 for(uint i=tid;i<p[0];i+=256){bool kill=i==p[2] || (last&&penult&&i>=p[3]) || (last&&!penult&&i<p[1])
  || ((flags&16)&&i>=p[3]&&i<rules[row].y) || (begin&&(i<p[3]||i>p[3]+50));
  if(kill)x[i]=-INFINITY;}
 threadgroup_barrier(mem_flags::mem_device);if(begin||(last&&penult))return;
 threadgroup float h[8],t[8],z[8];float a=-INFINITY,b=-INFINITY;
 for(uint i=tid;i<p[0];i+=256){if(i>=p[3])a=max(a,x[i]);else b=max(b,x[i]);}
 a=simd_max(a);b=simd_max(b);if(lane==0){h[sg]=a;t[sg]=b;}threadgroup_barrier(mem_flags::mem_threadgroup);
 a=-INFINITY;b=-INFINITY;for(uint j=0;j<8;++j){a=max(a,h[j]);b=max(b,t[j]);}
 float sum=0;if(isfinite(a))for(uint i=p[3]+tid;i<p[0];i+=256)sum+=exp(x[i]-a);
 sum=simd_sum(sum);if(lane==0)z[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 sum=0;for(uint j=0;j<8;++j)sum+=z[j];
 if(sum>0 && a+log(sum)>b)for(uint i=tid;i<p[3];i+=256)x[i]=-INFINITY;
}
