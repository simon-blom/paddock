// Native packed ModernBERT / Laya. Checkpoint F16 weights, F16 contractions,
// F32 accumulators, residuals and row reductions. One submission per pass;
// no host activations, padding to the longest sequence, or T x T score plane.
// All row tiles and K reductions are fixed independently of batch composition.
template<bool GLU> inline void laya_project(device half* w, device half* x,
 device half* out, device const float* bias, constant uint* p, uint2 g) {
 uint K=p[0],N=p[1],M=p[2],n=g.x*64,m=g.y*32;
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,64,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 op.run(a,b,acc);
 if constexpr(GLU) {
  auto gate=tensor(w+ulong(N)*K,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
  auto ga=op.template get_destination_cooperative_tensor<decltype(a),decltype(gate),float>();
  op.run(a,gate,ga);
  for(uint i=0;i<acc.get_capacity();++i)acc[i]=mv_gelu_value(acc[i])*ga[i];
  for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
   auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
   if(col<N && row<M)out[ulong(row)*N+col]=half(*it);
  }
 } else {
  for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
   auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
   if(col<N && row<M) {
    float v=*it;if(p[3])v+=bias[col];
    if(p[3]==2)v=max(v,0.0f);if(p[3]==3)v=mv_gelu_value(v);
    out[ulong(row)*N+col]=half(v);
   }
  }
 }
}
#define LAYA_MM(NAME,GLU) \
kernel void NAME(device half* w [[buffer(0)]],device half* x [[buffer(1)]], \
 device half* out [[buffer(2)]],device const float* b [[buffer(3)]],constant uint* p [[buffer(4)]], \
 uint2 g [[threadgroup_position_in_grid]]){laya_project<GLU>(w,x,out,b,p,g);}
LAYA_MM(laya_mm,false)
LAYA_MM(laya_geglu,true)
#undef LAYA_MM

inline float laya_sum(float x,threadgroup float* sums,uint lane,uint sg) {
 float v=simd_sum(x);if(lane==0)sums[sg]=v;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float out=0;for(uint j=0;j<8;++j)out+=sums[j];
 threadgroup_barrier(mem_flags::mem_threadgroup);return out;
}
kernel void laya_embed(device const half* emb [[buffer(0)]],device const uint* ids [[buffer(1)]],
 device const float* w [[buffer(2)]],device float* x [[buffer(3)]],device half* out [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float sums[8];uint D=p[0];ulong at=ulong(ids[row])*D;
 float v=0;for(uint j=tid;j<D;j+=256)v+=float(emb[at+j]);
 float mean=laya_sum(v,sums,lane,sg)/D;v=0;
 for(uint j=tid;j<D;j+=256){float z=float(emb[at+j])-mean;v+=z*z;}
 float inv=rsqrt(laya_sum(v,sums,lane,sg)/D+as_type<float>(p[1]));
 for(uint j=tid;j<D;j+=256){float z=(float(emb[at+j])-mean)*inv*w[j];x[ulong(row)*D+j]=z;out[ulong(row)*D+j]=half(z);}
}
// Residual + bias + the following LayerNorm, one row-local reduction.
kernel void laya_norm(device float* x [[buffer(0)]],device const half* proj [[buffer(1)]],
 device const float* bias [[buffer(2)]],device const float* w [[buffer(3)]],device const float* b [[buffer(4)]],
 device half* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float sums[8];uint D=p[0];ulong at=ulong(row)*D;
 float v=0;for(uint j=tid;j<D;j+=256){float z=x[at+j]+float(proj[at+j])+bias[j];x[at+j]=z;v+=z;}
 float mean=laya_sum(v,sums,lane,sg)/D;v=0;
 for(uint j=tid;j<D;j+=256){float z=x[at+j]-mean;v+=z*z;}
 float inv=rsqrt(laya_sum(v,sums,lane,sg)/D+as_type<float>(p[1]));
 for(uint j=tid;j<D;j+=256)out[at+j]=half((x[at+j]-mean)*inv*w[j]+b[j]);
}
kernel void laya_head_entry(device float* x [[buffer(0)]],device const float* finalw [[buffer(1)]],
 device const float* typew [[buffer(2)]],device const uint2* meta [[buffer(3)]],
 device const float* w [[buffer(4)]],device const float* b [[buffer(5)]],device half* out [[buffer(6)]],
 constant uint* p [[buffer(7)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float sums[8];uint D=p[0];ulong at=ulong(row)*D;
 float v=0;for(uint j=tid;j<D;j+=256)v+=x[at+j];float mean=laya_sum(v,sums,lane,sg)/D;v=0;
 for(uint j=tid;j<D;j+=256){float z=x[at+j]-mean;v+=z*z;}
 float inv=rsqrt(laya_sum(v,sums,lane,sg)/D+as_type<float>(p[1]));v=0;
 for(uint j=tid;j<D;j+=256){float z=(x[at+j]-mean)*inv*finalw[j]+typew[meta[row].y*D+j];x[at+j]=z;v+=z;}
 mean=laya_sum(v,sums,lane,sg)/D;v=0;
 for(uint j=tid;j<D;j+=256){float z=x[at+j]-mean;v+=z*z;}
 inv=rsqrt(laya_sum(v,sums,lane,sg)/D+1e-5f);
 for(uint j=tid;j<D;j+=256)out[at+j]=half((x[at+j]-mean)*inv*w[j]+b[j]);
}
// Split-half RoPE or biased head QKV. Sequence-relative positions never
// depend on where another request placed this sequence in the packed pass.
kernel void laya_qkv(device const half* x [[buffer(0)]],device const uint2* meta [[buffer(1)]],
 device const float2* rope [[buffer(2)]],device const float* bias [[buffer(3)]],
 device half* q [[buffer(4)]],device half* k [[buffer(5)]],device half* v [[buffer(6)]],
 constant uint* p [[buffer(7)]],uint i [[thread_position_in_grid]]) {
 uint D=p[0],row=i/D,j=i%D;if(row>=p[1])return;ulong at=ulong(row)*3*D+j;
 float a=float(x[at]),b=float(x[at+D]),c=float(x[at+2*D]);
 if(p[2]) {uint hd=j%64,other=j-hd+(hd+32)%64;float2 cs=rope[meta[row].x*32+hd%32];
  float sn=hd<32?-cs.y:cs.y;
  a=a*cs.x+float(x[ulong(row)*3*D+other])*sn;
  b=b*cs.x+float(x[ulong(row)*3*D+D+other])*sn;
 } else {a+=bias[j];b+=bias[D+j];c+=bias[2*D+j];}
 q[i]=half(a);k[i]=half(b);v[i]=half(c);
}
#define LAYA_ATTN(HEADS) \
kernel void laya_attention##HEADS(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]], \
 device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]], \
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float correction[32],normalizer[32],remap[32*64]; \
 vis_attention_impl<false,64,64,HEADS>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap,nullptr,nullptr,0,p[1]); }
LAYA_ATTN(12)
LAYA_ATTN(16)
#undef LAYA_ATTN
kernel void laya_gather(device const float* x [[buffer(0)]],device const half* att [[buffer(1)]],
 device const uint* indices [[buffer(2)]],device float* gathered [[buffer(3)]],device half* out [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[1])return;ulong src=ulong(indices[i/p[0]])*p[0]+i%p[0];gathered[i]=x[src];out[i]=att[src];
}
kernel void laya_score(device const half* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 float v=0;for(uint j=lane;j<p[0];j+=32)v+=float(x[ulong(row)*p[0]+j])*w[j];
 v=simd_sum(v);if(lane==0)out[row]=v+as_type<float>(p[1]);
}
// The act head consumes the *untempered* distribution and the final CLS
// residual, not scorer-normalized features. Its tiny two-layer graph stays
// inside one threadgroup; only the final probabilities leave the GPU.
kernel void laya_act(device const float* logits [[buffer(0)]],device const uint* offsets [[buffer(1)]],
 device const float* x [[buffer(2)]],device const half* w0 [[buffer(3)]],device const float* b0 [[buffer(4)]],
 device const half* w2 [[buffer(5)]],device const float* b2 [[buffer(6)]],device float* out [[buffer(7)]],
 constant uint* p [[buffer(8)]],uint seq [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float features[1028],hidden[256],z[8];uint D=p[0],n=p[2],begin=offsets[seq],end=offsets[seq+1];
 for(uint j=tid;j<D;j+=256)features[j]=x[ulong(p[1]+seq)*D+j];
 if(tid==0){float high=-INFINITY,den=0,a=0,b=0,ent=0;
  for(uint j=begin;j<end;++j)high=max(high,logits[j]);
  for(uint j=begin;j<end;++j)den+=exp(logits[j]-high);
  for(uint j=begin;j<end;++j){float v=exp(logits[j]-high)/den;if(v>a){b=a;a=v;}else b=max(b,v);ent-=v*log(max(v,1e-9f));}
  features[D]=a;features[D+1]=a-b;features[D+2]=ent/log(float(max(end-begin,2u)));features[D+3]=float(max(end-begin,2u))/255.0f;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint h=sg;h<256;h+=8){float v=0;for(uint j=lane;j<D+4;j+=32)v+=features[j]*float(w0[ulong(h)*(D+4)+j]);v=simd_sum(v);if(lane==0)hidden[h]=mv_gelu_value(v+b0[h]);}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg<n){float v=0;for(uint j=lane;j<256;j+=32)v+=hidden[j]*float(w2[sg*256+j]);v=simd_sum(v);if(lane==0)z[sg]=v+b2[sg];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0){float high=z[0],den=0;for(uint j=1;j<n;++j)high=max(high,z[j]);for(uint j=0;j<n;++j)den+=exp(z[j]-high);
  for(uint j=0;j<n;++j)out[seq*n+j]=exp(z[j]-high)/den;}
}
