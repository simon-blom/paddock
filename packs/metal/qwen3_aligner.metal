// Original BF16 timestamp-classifier graph. F32 backing buffers are a storage
// choice, not permission to elide the checkpoint's BF16 activation boundaries.
// Projection inputs are genuinely BF16; residuals use F32 storage containing
// BF16 values. Explicit contraction rules preserve intermediate rounding.
inline float qalign_bf(float x) { return float(bfloat(x)); }
kernel void qalign_round(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])x[i]=qalign_bf(x[i]);
}
// Accumulate contractions in F32, round the linear result, then separately
// round a residual addition. Bias belongs to the linear, not the residual.
kernel void qalign_project(device bfloat* w [[buffer(0)]],device bfloat* x [[buffer(1)]],
 device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
 #pragma clang fp reassociate(off)
 #pragma clang fp contract(off)
 uint K=p[0],N=p[1],M=p[2],m=g.y*32,n=g.x*64;
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,64,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();op.run(a,b,acc);
 for(auto it=acc.begin();it!=acc.end();++it) {
  auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
  if(it.is_valid_element() && row<M && col<N) {
   ulong at=ulong(row)*N+col;float v=qalign_bf(*it+(p[3]?bias[col]:0.0f));
   out[at]=p[3]==2?qalign_bf(out[at]+v):v;
  }
 }
}
kernel void qalign_rms(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device bfloat* out [[buffer(2)]],device const uint* selected [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 #pragma clang fp reassociate(off)
 #pragma clang fp contract(off)
 threadgroup float sums[8];uint n=p[0];ulong src=ulong(p[1]?selected[row]:row)*n;
 float sum=0;for(uint d=tid;d<n;d+=256){float v=x[src+d];sum+=v*v;}
 sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(n)+1e-6f);
 for(uint d=tid;d<n;d+=256)out[ulong(row)*n+d]=bfloat(qalign_bf(x[src+d]*inv)*w[d]);
}
kernel void qalign_residual(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])x[i]=qalign_bf(x[i]+delta[i]);
}
kernel void qalign_swiglu(device const float* gate [[buffer(0)]],device const float* up [[buffer(1)]],device bfloat* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 #pragma clang fp reassociate(off)
 #pragma clang fp contract(off)
 if(i<p[0]){float x=gate[i];out[i]=bfloat(qalign_bf(precise::divide(x,1.0f+precise::exp(-x)))*up[i]);}
}
kernel void qalign_gelu(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])x[i]=qalign_bf(mv_gelu_value(x[i]));
}
kernel void qalign_gelu_bf(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])out[i]=bfloat(mv_gelu_value(x[i]));
}
kernel void qalign_ln(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const float* b [[buffer(2)]],
 device bfloat* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 #pragma clang fp reassociate(off)
 #pragma clang fp contract(off)
 threadgroup float sums[8];uint n=p[0];ulong src=ulong(row)*n;float sum=0;
 for(uint d=tid;d<n;d+=256)sum+=x[src+d];sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 float mean=simd_sum(lane<8?sums[lane]:0.0f)/float(n);threadgroup_barrier(mem_flags::mem_threadgroup);
 sum=0;for(uint d=tid;d<n;d+=256){float v=x[src+d]-mean;sum+=v*v;}
 sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(n)+as_type<float>(p[1]));
 for(uint d=tid;d<n;d+=256)out[src+d]=bfloat((x[src+d]-mean)*inv*w[d]+b[d]);
}
kernel void qalign_position(device const float* x [[buffer(0)]],device const float* pos [[buffer(1)]],device float* out [[buffer(2)]],device const uint* map [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*1024)return;uint row=map[p[1]+i/1024];if(row!=UINT_MAX)out[ulong(row)*1024+i%1024]=qalign_bf(x[i]+qalign_bf(pos[(i/1024%13)*1024+i%1024]));
}
kernel void qalign_heads(device const float* x [[buffer(0)]],device bfloat* q [[buffer(1)]],device bfloat* k [[buffer(2)]],device bfloat* v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 uint r=i/1024,d=i%1024;if(r>=p[0]+64)return;
 q[i]=bfloat(r<p[0]?x[ulong(r)*3072+d]:0.0f);k[i]=bfloat(r<p[0]?x[ulong(r)*3072+1024+d]:0.0f);v[i]=bfloat(r<p[0]?x[ulong(r)*3072+2048+d]:0.0f);
}
kernel void qalign_head_rope(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const uint* meta [[buffer(2)]],
 device bfloat* out [[buffer(3)]],device const float* value [[buffer(4)]],device bfloat* values [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 #pragma clang fp reassociate(off)
 #pragma clang fp contract(off)
 ulong src=(ulong(g.y)*p[0]+g.x)*128;float sum=0;
 for(uint d=lane;d<128;d+=32)sum+=x[src+d]*x[src+d];float inv=rsqrt(simd_sum(sum)/128.0f+1e-6f);
 for(uint d=lane;d<128;d+=32) {
  uint j=d%64,other=d<64?d+64:d-64;
  float frequency=precise::divide(1.0f,precise::pow(1e6f,float(j)/64.0f));
  float angle=float(meta[2*g.y+1])*frequency;
  float a=qalign_bf(qalign_bf(x[src+d]*inv)*w[d]),b=qalign_bf(qalign_bf(x[src+other]*inv)*w[other]);
  float cs=qalign_bf(precise::cos(angle)),sn=qalign_bf(precise::sin(angle));
  out[src+d]=bfloat(qalign_bf(a*cs)+qalign_bf((d<64?-b:b)*sn));
  if(p[4])values[src+d]=bfloat(value[src+d]);
 }
}
kernel void qalign_audio_attention(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],device bfloat* v [[buffer(2)]],device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float correction[32],normalizer[32],remap[32*64];
 vis_attention_impl<false,64,64,16,true,false,bfloat,false,false,16,true,32>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
// Original TensorOps implementation, not copied MPS shader code. The BF16
// reference uses query prescaling in base-2 before Q.K, F32 online softmax,
// and BF16 output; scaling the F32 scores afterward is a different contract.
kernel void qalign_text_attention(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],device bfloat* v [[buffer(2)]],device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float correction[32],normalizer[32],remap[32*64];
 vis_attention_impl<false,128,128,16,false,false,bfloat,false,true,8,true,16>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
kernel void qalign_widen(device const ushort* x [[buffer(0)]],device uint* y [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])y[i]=uint(x[i])<<16;
}
kernel void qalign_sinusoid(device float* out [[buffer(0)]],constant uint* p [[buffer(1)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=13*1024)return;uint d=i%1024;
 float angle=float(i/1024)*exp(-log(10000.0f)*float(d%512)/511.0f);
 out[i]=d<512?sin(angle):cos(angle);
}
kernel void qalign_inject(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])out[ulong(p[1])+i]=x[i];
}
// 5000 is deliberately not a power of two. Scan every class, reject any
// nonfinite value, and break ties on the lower bin independent of SIMD order.
kernel void qalign_argmax(device const float* x [[buffer(0)]],device uint* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float best[8];threadgroup uint bins[8],bad[8];
 float high=-INFINITY;uint bin=UINT_MAX,invalid=0;
 for(uint i=tid;i<p[0];i+=256){float a=x[ulong(row)*p[0]+i];
  invalid|=!isfinite(a);if(a>high || (a==high && i<bin)){high=a;bin=i;}}
 float h=simd_max(high);uint ix=simd_min(high==h?bin:UINT_MAX),b=simd_or(invalid);
 if(lane==0){best[sg]=h;bins[sg]=ix;bad[sg]=b;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 h=simd_max(lane<8?best[lane]:-INFINITY);
 ix=simd_min(lane<8 && best[lane]==h?bins[lane]:UINT_MAX);
 b=simd_or(lane<8?bad[lane]:0u);
 if(tid==0)out[row]=b?UINT_MAX:ix;
}
