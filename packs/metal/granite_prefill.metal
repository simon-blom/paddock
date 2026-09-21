// Original Granite TensorOps prefill, specialized for the checkpoint's head
// dimension. Compact BK32 staging stays within Apple10's 32 KiB limit with
// shader validation, unlike the older BK64 graph. No quadratic score storage.
template <int D, int BK = 64, bool Packed = false, bool Sinks = false>
inline void granite_prefill_tile(device half* q, device const half* kc, device const half* vc,
                         device const uint* meta, device const uint* pages, device float* out,
                         constant uint* p, uint head, uint first, uint count, uint tid,
                         threadgroup half* kv, threadgroup half* probability, threadgroup float* scores,
                         threadgroup float* maximum, threadgroup float* denominator, threadgroup float* correction,
                         device const float* sinks = nullptr, uint window = 0) {
    uint kh=head/(p[0]/p[1]),slot=meta[2*first];
    uint lastpos=meta[2*(first+count-1)+1],width=p[0]*D,kvwidth=p[1]*D;
    auto tq=tensor(q+ulong(first)*width+head*D,dextents<int,2>(D,count),array<int,2>{1,int(width)});
    auto tk=tensor(kv,extents<int,D,BK>(),array<int,2>{1,D});
    auto tv=tensor(kv,extents<int,BK,D>(),array<int,2>{1,BK});
    auto tp=tensor(probability,extents<int,BK,32>(),array<int,2>{1,BK});
    auto ts=tensor(scores,extents<int,BK,32>(),array<int,2>{1,BK});
    constexpr auto qk_desc=matmul2d_descriptor(32,BK,D,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc=matmul2d_descriptor(32,D,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc,execution_simdgroups<4>> qk;
    matmul2d<pv_desc,execution_simdgroups<4>> pv;
    auto accum=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<accum.get_capacity();++i)accum[i]=0;
    if(tid<32){maximum[tid]=Sinks ? sinks[head] : -INFINITY;denominator[tid]=Sinks ? 1.0f : 0.0f;}
    uint begin=Sinks && window ? (meta[2*first+1]+1-min(meta[2*first+1]+1,window))/BK*BK : 0;
    for(uint base=begin;base<=lastpos;base+=BK) {
        for(uint i=tid;i<BK*D;i+=128) {
            uint t=base+i/D,d=i%D;
            uint physical=t<=lastpos ? (Packed ? slot+t : pages[slot*p[2]+t/16]*16+t%16) : 0;
            kv[i]=t<=lastpos ? kc[ulong(physical)*kvwidth+kh*D+d] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto score=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        for(uint i=0;i<score.get_capacity();++i)score[i]=0;
        qk.run(tq,tk,score);
        score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint row=tid/4,lane=tid%4,pos=row<count ? meta[2*(first+row)+1] : 0;
        float high=maximum[row];
        for(uint j=lane;j<BK;j+=4)if(row<count && base+j<=pos && (!Sinks || !window || base+j+window>pos))high=max(high,scores[row*BK+j]*as_type<float>(p[3]));
        high=max(high,simd_shuffle_xor(high,1));high=max(high,simd_shuffle_xor(high,2));
        float old=isfinite(maximum[row]) ? exp(maximum[row]-high) : 0.0f;
        float sum=0;
        for(uint j=lane;j<BK;j+=4) {
            float prob=row<count && base+j<=pos && (!Sinks || !window || base+j+window>pos) ? exp(scores[row*BK+j]*as_type<float>(p[3])-high) : 0.0f;
            probability[row*BK+j]=half(prob);sum+=prob;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
        if(lane==0) {
            maximum[row]=high;correction[row]=old;
            denominator[row]=row<count ? denominator[row]*old+sum : 1.0f;
        }
        // All readers of K have completed before this storage becomes V.
        for(uint i=tid;i<BK*D;i+=128) {
            uint t=base+i%BK,d=i/BK;
            uint physical=t<=lastpos ? (Packed ? slot+t : pages[slot*p[2]+t/16]*16+t%16) : 0;
            kv[i]=t<=lastpos ? vc[ulong(physical)*kvwidth+kh*D+d] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto product=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
        for(uint i=0;i<product.get_capacity();++i)product[i]=0;
        pv.run(tp,tv,product);
        uint i=0;
        for(auto it=product.begin();it!=product.end();++it,++i) {
            auto ij=it.get_multidimensional_index();
            if(it.is_valid_element())accum[i]=accum[i]*correction[ij[1]]+*it;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=accum.begin();it!=accum.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element())*it/=denominator[ij[1]];
    }
    auto dst=tensor(out+ulong(first)*width+head*D,dextents<int,2>(D,count),array<int,2>{1,int(width)});
    accum.store(dst);
}

// Granite 3B's genuine hd64 tile, not padded hd128 computation. BK32 bounds
// explicit plus implicit MPP staging: BK64 reached 41,728 bytes under GPU
// validation on Apple10 (32 KiB limit). Q stays BM32 with ragged boundaries.
kernel void attention_prefill_batched64(device half* q [[buffer(0)]], device const half* kc [[buffer(1)]],
                                      device const half* vc [[buffer(2)]], device const uint* meta [[buffer(3)]],
                                      device const uint* pages [[buffer(4)]], device float* out [[buffer(5)]],
                                      device const uint* tiles [[buffer(6)]], constant uint* p [[buffer(7)]],
                                      uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[32*64], probability[32*32];
    threadgroup float scores[32*32], maximum[32], denominator[32], correction[32];
    granite_prefill_tile<64,32>(q,kc,vc,meta,pages,out,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
                    kv,probability,scores,maximum,denominator,correction);
}

// The existing hd128/BK64 serving graph also exceeds the validation budget
// (58,112 bytes). Qualify the same bounded tile family for Granite 8B/30B.
kernel void granite_attention_prefill128(device half* q [[buffer(0)]], device const half* kc [[buffer(1)]],
                                      device const half* vc [[buffer(2)]], device const uint* meta [[buffer(3)]],
                                      device const uint* pages [[buffer(4)]], device float* out [[buffer(5)]],
                                      device const uint* tiles [[buffer(6)]], constant uint* p [[buffer(7)]],
                                      uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[32*128], probability[32*32];
    threadgroup float scores[32*32], maximum[32], denominator[32], correction[32];
    granite_prefill_tile<128,32>(q,kc,vc,meta,pages,out,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
                    kv,probability,scores,maximum,denominator,correction);
}
