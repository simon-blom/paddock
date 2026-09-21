// Original Muse Glimmer graph epilogues and HD128 GQA attention. Reuse our
// online softmax/paged-cache machinery; no context-sized scores or KV copies.
// GGUF query norm already contains qk_scale_factor, but SDPA still needs
// 1/sqrt(128). Global blocks are NoPE, local blocks use interleaved pairs.
// Wide target verification cannot afford a register GEMV for every row.
// Expand only a 16-column Q8 tile to F32; keep activations and contraction
// precision unchanged. This is our Gemma F32 matrix design with Q8 staging,
// not a weight repack or an F16 approximation of the target graph.
template<uint BM>
inline void muse_q8_f32(device const uchar* w,device float* x,device float* out,
    constant uint* p,uint2 g,uint tid,threadgroup float* weights) {
    // 128 K leaves space for MPP's implicit F32 staging on Apple10's
    // 32 KiB threadgroup limit (256 K exceeds it once staging is included).
    constexpr uint BK=128,BN=16;
    uint K=p[0],N=p[1],M=p[2],n=g.x*BN,m=g.y*BM;
    auto input=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+4});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    for(uint base=0;base<K;base+=BK) {
        if(tid<BN*BK/32) {
        uint col=n+tid/4,k=base+(tid%4)*32;
        threadgroup float4* dst=reinterpret_cast<threadgroup float4*>(weights+(tid/4)*(BK+4)+(tid%4)*32);
        if(col<N && k<K) {
            device const uchar* block=w+((ulong(col)*K+k)/32)*34;
            float scale=float(*reinterpret_cast<device const half*>(block));
            for(uint j=0;j<8;++j)
                dst[j]=float4(*reinterpret_cast<device const packed_char4*>(block+2+j*4))*scale;
        } else for(uint j=0;j<8;++j)dst[j]=0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=input.slice(base,m);op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<c.get_capacity();++i)c[i]*=as_type<float>(p[4]);
    c.store(output.slice(n,m));
}
#define MUSE_Q8_F32(BM) \
kernel void muse_q8_f32_##BM(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float weights[16*132];muse_q8_f32<BM>(w,x,out,p,g,tid,weights);}
MUSE_Q8_F32(16)
MUSE_Q8_F32(32)
MUSE_Q8_F32(64)
#undef MUSE_Q8_F32


kernel void muse_qnorm(device float* q [[buffer(0)]],device const uchar* norm [[buffer(1)]],
    device const uint* meta [[buffer(2)]],device const float* unused [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint hd=p[1];ulong src=(ulong(g.y)*p[0]+g.x)*hd;
    float sum=0;for(uint d=lane;d<hd;d+=32)sum+=q[src+d]*q[src+d];
    float inv=rsqrt(simd_sum(sum)/float(hd)+as_type<float>(p[3]));
    for(uint j=lane;j<hd/2;j+=32){uint d=j*2;
        float a=q[src+d]*inv*weight(norm,p[2],d),b=q[src+d+1]*inv*weight(norm,p[2],d+1);
        float angle=p[5]?0.0f:float(meta[g.y*2+1])*pow(as_type<float>(p[4]),-2.0f*float(j)/float(hd));
        float c=cos(angle),s=sin(angle);q[src+d]=a*c-b*s;q[src+d+1]=a*s+b*c;
    }
}
kernel void muse_kv_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
    device const uchar* norm [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],
    device const float* unused [[buffer(5)]],device half* keys [[buffer(6)]],device half* values [[buffer(7)]],
    constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint hd=p[5],slot=meta[g.y*2],pos=meta[g.y*2+1];
    ulong src=(ulong(g.y)*p[1]+g.x)*hd,dst=(ulong(gemma_physical(pages,slot,pos,p))*p[1]+g.x)*hd;
    float sum=0;for(uint d=lane;d<hd;d+=32)sum+=k[src+d]*k[src+d];
    float inv=rsqrt(simd_sum(sum)/float(hd)+as_type<float>(p[7]));
    for(uint j=lane;j<hd/2;j+=32){uint d=j*2;
        float a=k[src+d]*inv*weight(norm,p[6],d),b=k[src+d+1]*inv*weight(norm,p[6],d+1);
        float angle=p[3]?float(pos)*pow(as_type<float>(p[8]),-2.0f*float(j)/float(hd)):0.0f;
        float c=cos(angle),s=sin(angle);keys[dst+d]=half(a*c-b*s);keys[dst+d+1]=half(a*s+b*c);
        values[dst+d]=half(v[src+d]);values[dst+d+1]=half(v[src+d+1]);
    }
}
// Embeddings (including image rows) have a weightless RMS preamble. It must
// precede the first learned norm, not replace it or multiply by sqrt(width).
kernel void muse_embedding_norm(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];ulong base=ulong(row)*p[0];float sum=0;
    for(uint d=tid;d<p[0];d+=256){float v=x[base+d];sum+=v*v;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(p[0])+as_type<float>(p[1]));
    for(uint d=tid;d<p[0];d+=256)x[base+d]*=inv;
}
// Post norms use 1e-8; the following pre-norm uses the checkpoint epsilon.
// Fuse the two reductions while keeping the F32 residual between them.
kernel void muse_sandwich(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],
    device const uchar* post [[buffer(2)]],device const uchar* next [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];uint n=p[0];ulong base=ulong(row)*n;float sum=0;
    for(uint d=tid;d<n;d+=256){float z=delta[base+d];sum+=z*z;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(n)+1e-8f);
    threadgroup_barrier(mem_flags::mem_threadgroup);sum=0;
    for(uint d=tid;d<n;d+=256){float z=x[base+d]+delta[base+d]*inv*weight(post,p[1],d);x[base+d]=z;sum+=z*z;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(n)+as_type<float>(p[3]));
    for(uint d=tid;d<n;d+=256)out[base+d]=x[base+d]*inv*weight(next,p[2],d);
}
kernel void muse_gate(device float* out [[buffer(0)]],device const float* gate [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])out[i]/=1.0f+exp(-gate[i]);
}
kernel void muse_swiglu(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float v=gate[i];gate[i]=(v/(1.0f+exp(-v)))*up[i];}
}
kernel void muse_softcap(device float* logits [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float c=as_type<float>(p[1]),scale=as_type<float>(p[2]);logits[i]=c*precise::tanh(logits[i]*scale/c);}
}
// Vector contraction is elected for short/narrow serving and also provides
// an independent device-only check of the long-c=4 matrix route.
kernel void muse_decode_vector(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[16*32],prob[16*32],highs[16],sums[16];
    gemma_decode<128,16,true>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}
// Muse's 16 query heads per KV head form a full matrix tile, even at c=1.
// Fold GQA into the row axis (Flash-Decoding/FlashInfer), stage each paged
// key/value tile once, and keep F32 queries, probabilities and accumulators.
// No global KV gather, operand quantization, or context-sized score plane.
kernel void muse_decode(device float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    // Leave room for MPP's implicit mixed-F32 operand staging as well as our
    // explicit tile: the Apple10 per-threadgroup ceiling is 32 KiB.
    constexpr uint H=16,D=128,T=32,G=8;
    threadgroup half kv[T*D];
    threadgroup float probability[H*T],scores[H*T],maximum[H],denom[H],correction[H];
    uint row=rows[g.y],slot=meta[2*row],length=meta[2*row+1]+1;
    uint low=p[3] && length>p[3]?length-p[3]:0;
    uint span=(length-low+p[5]-1)/p[5],first=low+g.z*span,last=min(first+span,length);
    auto tq=tensor(q+(ulong(row)*p[0]+g.x*H)*D,extents<int,D,H>());
    auto tk=tensor(kv,extents<int,D,T>());
    auto tv=tensor(kv,extents<int,T,D>());
    auto tp=tensor(probability,extents<int,T,H>());
    auto ts=tensor(scores,extents<int,T,H>());
    constexpr auto qkd=matmul2d_descriptor(H,T,D,false,true,false);
    constexpr auto pvd=matmul2d_descriptor(H,D,T,false,true,false);
    matmul2d<qkd,execution_simdgroups<4>> qk;
    matmul2d<pvd,execution_simdgroups<4>> pv;
    auto acc=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    if(tid<H){maximum[tid]=-INFINITY;denom[tid]=0;}
    for(uint base=first;base<last;base+=T) {
        for(uint i=tid;i<T*D;i+=128){uint t=base+i/D,d=i%D;
            kv[i]=t<last?k[(ulong(gemma_physical(pages,slot,t,p))*p[1]+g.x)*D+d]:half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto score=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        qk.run(tq,tk,score);
        for(uint i=0;i<score.get_capacity();++i)score[i]*=0.08838834764831845f;
        score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint h=tid/G,lane=tid%G;float hi=maximum[h];
        for(uint j=lane;j<T;j+=G)if(base+j<last)hi=max(hi,scores[h*T+j]);
        for(uint shift=1;shift<G;shift*=2)hi=max(hi,simd_shuffle_xor(hi,shift));
        float old=isfinite(maximum[h])?exp(maximum[h]-hi):0.0f,sum=0;
        for(uint j=lane;j<T;j+=G){float pr=base+j<last?exp(scores[h*T+j]-hi):0.0f;
            probability[h*T+j]=pr;sum+=pr;}
        for(uint shift=1;shift<G;shift*=2)sum+=simd_shuffle_xor(sum,shift);
        if(lane==0){maximum[h]=hi;correction[h]=old;denom[h]=denom[h]*old+sum;}
        for(uint i=tid;i<T*D;i+=128){uint t=base+i%T,d=i/T;
            kv[i]=t<last?v[(ulong(gemma_physical(pages,slot,t,p))*p[1]+g.x)*D+d]:half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
        pv.run(tp,tv,product);
        uint i=0;for(auto it=product.begin();it!=product.end();++it,++i)
            if(it.is_valid_element())acc[i]=acc[i]*correction[it.get_multidimensional_index()[1]]+*it;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Empty ragged partitions publish (-inf,0,zeros), never NaNs. The merge
    // shares the original ABI, including a stride between query heads.
    ulong at=((ulong(g.y)*p[0]+g.x*H)*p[5]+g.z)*(D+2);
    auto dst=tensor(out+at,extents<int,D,H>(),array<int,2>{1,int(p[5]*(D+2))});
    acc.store(dst);
    if(tid<H){out[at+ulong(tid)*p[5]*(D+2)+D]=maximum[tid];out[at+ulong(tid)*p[5]*(D+2)+D+1]=denom[tid];}
}
kernel void muse_merge(device const float* parts [[buffer(0)]],device float* out [[buffer(1)]],device const uint* rows [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    gemma_merge_vectors<128>(parts,out,rows,p,g,lane);
}
// One reusable projection slab, not persistent F16 model weights. Preserve
// exactly the Q8 -> F32 -> F16 operand rounding of quant_tile. Wider direct
// TensorOps tiles can amortize this once-per-plane write over image rows.
kernel void muse_q8_expand(device const uchar* w [[buffer(0)]], device half* workspace [[buffer(1)]],
                           constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ulong at=ulong(i)*4;if(at>=ulong(p[0])*p[1])return;
    device const uchar* block=w+(at/32)*34;
    float scale=float(*reinterpret_cast<device const half*>(block));
    float4 values=float4(*reinterpret_cast<device const packed_char4*>(block+2+at%32))*scale;
    ulong offset=ulong((p[0]+127)/128*128)*((p[2]+127)/128*128);
    *reinterpret_cast<device half4*>(workspace+offset+at)=half4(values);
}


// Qualification-only rejected candidate, not selected by serving: M5's
// register/staging pressure outweighs its reduced inter-SIMD synchronization.
// One SIMD group owns the complete GQA query tile. Row reductions and online
// probabilities stay in cooperative registers (FA2 split-Q ownership); only
// the cache tile and a possible layout bridge touch threadgroup memory.
kernel void muse_decode_register(device float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint H=16,D=128,T=32;
    threadgroup half kv[T*D];
    threadgroup float remap[H*T],correction[H],normalizer[H],highs[H];
    uint row=rows[g.y],slot=meta[2*row],length=meta[2*row+1]+1;
    uint low=p[3] && length>p[3]?length-p[3]:0;
    uint span=(length-low+p[5]-1)/p[5],first=low+g.z*span,last=min(first+span,length);
    auto tq=tensor(q+(ulong(row)*p[0]+g.x*H)*D,extents<int,D,H>());
    auto tk=tensor(kv,extents<int,D,T>());
    auto tv=tensor(kv,extents<int,T,D>());
    constexpr auto qkd=matmul2d_descriptor(H,T,D,false,true,false);
    constexpr auto pvd=matmul2d_descriptor(H,D,T,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroup> qk;
    matmul2d<pvd,execution_simdgroup> pv;
    auto sc=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto maximum=qk.get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto denominator=qk.get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto hi=qk.get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto sum=qk.get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto old=qk.get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto pr=pv.get_left_input_cooperative_tensor<float,half,float>();
    auto acc=pv.get_destination_cooperative_tensor<decltype(pr),decltype(tv),float>();
    for(uint i=0;i<maximum.get_capacity();++i){maximum[i]=-INFINITY;denominator[i]=0;}
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=first;base<last;base+=T) {
        for(uint i=tid;i<T*D;i+=32){uint t=base+i/D,d=i%D;
            kv[i]=t<last?k[(ulong(gemma_physical(pages,slot,t,p))*p[1]+g.x)*D+d]:half(0);}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        qk.run(tq,tk,sc);
        for(auto it=sc.begin();it!=sc.end();++it)if(it.is_valid_element())
            *it=base+it.get_multidimensional_index()[0]<last?*it*0.08838834764831845f:-INFINITY;
        reduce_rows(sc,hi,reduction_operation::max,-INFINITY);
        for(uint i=0;i<maximum.get_capacity();++i){hi[i]=max(hi[i],maximum[i]);
            old[i]=isfinite(maximum[i])?exp(maximum[i]-hi[i]):0.0f;maximum[i]=hi[i];}
        for(auto it=sc.begin();it!=sc.end();++it)if(it.is_valid_element())
            *it=base+it.get_multidimensional_index()[0]<last?exp(*it-*hi.map_iterator(it)):0.0f;
        reduce_rows(sc,sum);
        for(uint i=0;i<denominator.get_capacity();++i)denominator[i]=denominator[i]*old[i]+sum[i];
        old.store(tensor(correction,extents<int,H>()));
        for(uint i=tid;i<T*D;i+=32){uint t=base+i%T,d=i/T;
            kv[i]=t<last?v[(ulong(gemma_physical(pages,slot,t,p))*p[1]+g.x)*D+d]:half(0);}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element())
            *it*=correction[it.get_multidimensional_index()[1]];
        if(pv.is_compatible_as_left_input<float,half,float>(sc)) {
            auto probs=pv.get_left_input_cooperative_tensor<float,half,float>(sc);pv.run(probs,tv,acc);
        } else {
            auto bridge=tensor(remap,extents<int,T,H>());sc.store(bridge);
            simdgroup_barrier(mem_flags::mem_threadgroup);pr.load(bridge);pv.run(pr,tv,acc);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    denominator.store(tensor(normalizer,extents<int,H>()));maximum.store(tensor(highs,extents<int,H>()));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    ulong at=((ulong(g.y)*p[0]+g.x*H)*p[5]+g.z)*(D+2);
    acc.store(tensor(out+at,extents<int,D,H>(),array<int,2>{1,int(p[5]*(D+2))}));
    if(tid<H){out[at+ulong(tid)*p[5]*(D+2)+D]=highs[tid];out[at+ulong(tid)*p[5]*(D+2)+D+1]=normalizer[tid];}
}
kernel void muse_prefill(device half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    // MPP needs implicit operand staging in addition to these arrays. A
    // 64-key tile exceeds Apple10's 32 KiB contract under GPU validation;
    // 32 keys retain full query tiles within the actual memory budget.
    threadgroup half kv[32*128],prob[32*32];threadgroup float scores[32*32],maximum[32],denom[32],correction[32];
    gemma_prefill<128,32,32,half,false,true>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction);
}
// Muse's wide DFlash conditioning can reach the full 512-row prefill rung.
// The padded K-quant slab plus MPP staging needs 33280 bytes under GPU
// validation on Apple10. Compact storage removes only the four-half pitch
// padding, preserving the same BK=256 contraction and all operand values.
// Other models retain their qualified default template instantiations.
#define MUSE_DF_KTILE(BM) \
kernel void muse_df_ktile##BM(device const uchar* w [[buffer(0)]],device half* x [[buffer(1)]], \
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]], \
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half weights[32*256]; \
    ktile_dispatch<BM,1,0>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),g,tid,weights); }
MUSE_DF_KTILE(96)
MUSE_DF_KTILE(128)
MUSE_DF_KTILE(64)
#undef MUSE_DF_KTILE
