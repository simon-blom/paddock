// Read adjacent physical pages as one 64-token MPP operand. Fragmented pages
// use bounded per-tile gather scratch with the same descriptor and arithmetic.
// BF16 KV is unchanged. Fixed block ownership preserves prompt/replay math.
template<uint Subtiles> inline void splash_attention_query_pack(device const float* x,device bfloat* out,
    device const uint* tiles,constant uint* p,uint i) {
    if(i>=p[0]*Subtiles*4*48*256)return;
    uint d=i%256,r=i/256%48,kh=i/(256*48)%4,sub=i/(256*48*4)%Subtiles,tile=i/(256*48*4*Subtiles);
    uint local=sub*8+r/6;
    out[i]=local<tiles[tile*2+1]?bfloat(x[(ulong(tiles[tile*2]+local)*24+kh*6+r%6)*256+d]):bfloat(0);
}
#define SPLASH_QUERY(NAME,SUBTILES) \
kernel void NAME(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]], \
    device const uint* tiles [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) { \
    splash_attention_query_pack<SUBTILES>(x,out,tiles,p,i); }
SPLASH_QUERY(splash_attention_query_grouped,4)
// Dense affine Qwen has the same 24:4 GQA query layout. Only packing is
// shared; its attention kernel keeps the native 16-token arithmetic contract.
SPLASH_QUERY(mlx_attention_query_grouped,4)
SPLASH_QUERY(splash_attention_query_decode,1)
#undef SPLASH_QUERY
template<ushort B,bool Grouped=false,uint Parts=4,uint Subtiles=4> inline void splash_attention_prefill_impl(
    device bfloat* q,device const bfloat* kc,
    device const bfloat* vc,device const uint* meta,
    device const uint* pages,device float* out,
    device const uint* tiles,device const uint* limits,
    device bfloat* staging,constant uint* p,
    uint3 group,uint tid,threadgroup float* scores,threadgroup float* maximum,
    threadgroup float* denominator,threadgroup float* correction,threadgroup bfloat* probs,
    threadgroup atomic_uint* rescale) {
    constexpr uint M=Grouped?48:32,D=256,Lanes=Grouped?4:8;
    uint head=group.x,tile=Grouped?group.y/Subtiles:group.y;
    uint first=tiles[2*tile],count=tiles[2*tile+1];
    if constexpr(Grouped) {uint offset=group.y%Subtiles*8;if(offset>=count)return;first+=offset;count=min(8u,count-offset);}
    uint kh=head/(p[0]/p[1]),slot=meta[first*2],last=limits[first+count-1];
    if constexpr(Grouped)kh=head;
    uint width=p[0]*D,kw=p[1]*D;
    device bfloat* tmp=staging+((ulong(group.y)*(Grouped?p[1]:p[0])+head)*Parts+group.z)*B*D;
    auto tq=tensor(q+(Grouped?(ulong(group.y)*p[1]+kh)*M*D:ulong(first)*width+head*D),
        dextents<int,2>{D,int(Grouped?M:count)},array<int,2>{1,int(Grouped?D:width)});
    auto ts=tensor(scores,extents<int,B,M>(),array<int,2>{1,B});
    auto tp=tensor(probs,extents<int,B,M>(),array<int,2>{1,B});
    auto tk=tensor(tmp,extents<int,D,B>(),array<int,2>{1,D});
    constexpr auto qkd=matmul2d_descriptor(M,B,D,false,true,false);
    constexpr auto pvd=matmul2d_descriptor(M,D,B,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroups<8>> qk;
    matmul2d<pvd,execution_simdgroups<8>> pv;
    auto acc=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tk),float>();
    uint traversal=splash_traversal(acc);
    for(ushort i=0;i<acc.get_capacity();++i)acc[i]=0;
    if(tid<M){maximum[tid]=-INFINITY;denominator[tid]=0;}
    for(uint base=group.z*B;base<=last;base+=Parts*B) {
        uint valid=min(uint(B),last-base+1),physical=pages[slot*p[2]+base/16]*16;
        bool contiguous=true;
        for(uint offset=16;offset<valid;offset+=16)
            contiguous&=pages[slot*p[2]+(base+offset)/16]*16==physical+offset;
        if(!contiguous) {
            for(uint i=tid;i<valid*D;i+=256) {
                uint t=base+i/D,at=pages[slot*p[2]+t/16]*16+t%16;
                tmp[i]=kc[ulong(at)*kw+kh*D+i%D];
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        auto key=tensor(contiguous?const_cast<device bfloat*>(kc)+ulong(physical)*kw+kh*D:tmp,
            dextents<int,2>{D,int(valid)},array<int,2>{1,int(contiguous?kw:D)});
        auto score=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(key),float>();
        qk.run(tq,key,score);
        score.store(ts);
        if(tid==0)atomic_store_explicit(rescale,0u,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        uint row=tid/Lanes,lane=tid%Lanes,local=Grouped?row/6:row;
        uint pos=local<count?limits[first+local]:0;
        float hi=-INFINITY,old=0,local_scores[B/Lanes];
        if(tid<M*Lanes) {
        hi=maximum[row];
        #pragma unroll
        for(uint j=0;j<B/Lanes;++j) {
            local_scores[j]=scores[row*B+lane+j*Lanes]*as_type<float>(p[3]);
            if(local<count && base+lane+j*Lanes<=pos)hi=max(hi,local_scores[j]);
        }
        hi=max(hi,simd_shuffle_xor(hi,1));hi=max(hi,simd_shuffle_xor(hi,2));if constexpr(Lanes==8)hi=max(hi,simd_shuffle_xor(hi,4));
        old=isfinite(maximum[row])?exp(maximum[row]-hi):0;
        }
        // Every score is now in private registers. Reuse the same shared
        // slab for BF16 probabilities; no row may overwrite another row's
        // still-live scores before this barrier.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<M*Lanes) {
        float sum=0;
        #pragma unroll
        for(uint j=0;j<B/Lanes;++j) {
            float pr=local<count && base+lane+j*Lanes<=pos?exp(local_scores[j]-hi):0;
            probs[row*B+lane+j*Lanes]=bfloat(pr);sum+=pr;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);if constexpr(Lanes==8)sum+=simd_shuffle_xor(sum,4);
        if(lane==0){
            maximum[row]=hi;denominator[row]=denominator[row]*old+sum;correction[row]=old;
            // Padded queries are never emitted. Only a live query whose
            // maximum changed requires rescaling the running numerator.
            if(local<count && old!=1.0f)atomic_fetch_or_explicit(rescale,1u,memory_order_relaxed);
        }
        }
        if(!contiguous) {
            for(uint i=tid;i<valid*D;i+=256) {
                uint t=base+i/D,at=pages[slot*p[2]+t/16]*16+t%16;
                tmp[i]=vc[ulong(at)*kw+kh*D+i%D];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if(atomic_load_explicit(rescale,memory_order_relaxed)) {
            auto apply_scale=[&](ushort i){acc[i]*=correction[acc.get_multidimensional_index(i)[1]];};
            splash_visit(acc,traversal,apply_scale);
        }
        auto value=tensor(contiguous?const_cast<device bfloat*>(vc)+ulong(physical)*kw+kh*D:tmp,
            dextents<int,2>{D,int(valid)},array<int,2>{1,int(contiguous?kw:D)});
        pv.run(tp,value,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    for(ushort i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
        auto ij=acc.get_multidimensional_index(i);
        uint local=Grouped?ij[1]/6:ij[1],qh=Grouped?kh*6+ij[1]%6:head;
        uint output_row=(Subtiles==1?tile*8:first)+local;
        if(local<count)out[((ulong(output_row)*p[0]+qh)*Parts+group.z)*258+ij[0]]=acc[i];
    }
    if(tid<count*(Grouped?6:1)){uint local=Grouped?tid/6:tid,qh=Grouped?kh*6+tid%6:head;
        uint output_row=(Subtiles==1?tile*8:first)+local;
        ulong dst=((ulong(output_row)*p[0]+qh)*Parts+group.z)*258;
        out[dst+256]=maximum[tid];out[dst+257]=denominator[tid];}
}

#define SPLASH_PREFILL(NAME,BLOCK,ROWS,GROUPED,PARTS,SUBTILES) \
kernel void NAME(device bfloat* q [[buffer(0)]],device const bfloat* k [[buffer(1)]], \
    device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]], \
    device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]], \
    device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]], \
    device bfloat* staging [[buffer(8)]],constant uint* p [[buffer(9)]], \
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float scores[ROWS*BLOCK],maximum[ROWS],denominator[ROWS],correction[ROWS]; \
    threadgroup atomic_uint rescale; \
    threadgroup bfloat* probs=reinterpret_cast<threadgroup bfloat*>(scores); \
    splash_attention_prefill_impl<BLOCK,GROUPED,PARTS,SUBTILES>(q,k,v,meta,pages,out,tiles,limits,staging,p,group,tid, \
        scores,maximum,denominator,correction,probs,&rescale); \
}
SPLASH_PREFILL(splash_attention_prefill64,64,32,false,4,4)
SPLASH_PREFILL(splash_attention_prefill_grouped,64,48,true,4,4)
// Fixed split ownership and padded query shape are identical for one-token
// decode and speculative verification, irrespective of peer context lengths.
SPLASH_PREFILL(splash_attention_decode_grouped,64,48,true,16,1)
#undef SPLASH_PREFILL

template<uint Parts> inline void splash_attention_decode_join_impl(device const float* parts,device float* out,
    device const uint* tiles,constant uint* p,uint3 group,uint lane) {
    if(group.z>=tiles[2*group.y+1])return;
    uint rh=(tiles[2*group.y]+group.z)*p[0]+group.x;
    ulong base=(ulong(group.y*8+group.z)*p[0]+group.x)*Parts*258;
    float high=-INFINITY;for(uint s=0;s<Parts;++s)high=max(high,parts[base+s*258+256]);
    float acc[8];for(uint j=0;j<8;++j)acc[j]=0;float denom=0;
    for(uint s=0;s<Parts;++s) {
        ulong src=base+s*258;if(parts[src+257]<=0)continue;
        float c=exp(parts[src+256]-high);denom+=parts[src+257]*c;
        for(uint j=0;j<8;++j)acc[j]+=parts[src+lane+j*32]*c;
    }
    for(uint j=0;j<8;++j)out[ulong(rh)*256+lane+j*32]=acc[j]/denom;
}
#define SPLASH_DECODE_JOIN(NAME,PARTS) \
kernel void NAME(device const float* parts [[buffer(0)]],device float* out [[buffer(1)]], \
    device const uint* tiles [[buffer(2)]],constant uint* p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) { \
    splash_attention_decode_join_impl<PARTS>(parts,out,tiles,p,group,lane); }
SPLASH_DECODE_JOIN(splash_attention_decode_join,16)
#undef SPLASH_DECODE_JOIN
