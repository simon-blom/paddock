// Original tiled split-K GQA decode. Each group owns one KV head for one
// query row, serving its four Q heads together. Four lanes cooperate on a
// QK dot product; 32 keys are scored in parallel, followed by a stable tile
// softmax. The V read is reused across all four heads. Only bounded tile
// scores live in threadgroup memory, never a context-sized attention matrix.
// Inspiration: online-softmax state composition and split-K Flash-Decoding;
// the lane mapping and paged GQA implementation below are Paddock's own.
// p: Q heads, KV heads, page stride, attention scale, selected rows, splits.
kernel void attention_gqa4(device const float* q [[buffer(0)]],
                           device const half* keys [[buffer(1)]], device const half* values [[buffer(2)]],
                           device const uint* meta [[buffer(3)]], device const uint* pages [[buffer(4)]],
                           device const uint* rows [[buffer(5)]], device float* out [[buffer(6)]],
                           constant uint* p [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[4*32], probability[4*32], tile_max[4], tile_sum[4];
    uint row=rows[group.y], kh=group.x, slot=meta[2*row], length=meta[2*row+1]+1;
    uint span=(length+p[5]-1)/p[5], first=group.z*span, last=min(first+span,length);
    uint kvwidth=p[1]*128, key_lane=tid/4, d_lane=tid%4;
    float4 accum=0, maximum=-INFINITY, denominator=0;
    for(uint base=first;base<last;base+=32) {
        uint token=base+key_lane;
        float4 dots=0;
        if(token<last) {
            uint physical=pages[slot*p[2]+token/16]*16+token%16;
            ulong ko=ulong(physical)*kvwidth+kh*128;
            for(uint i=0;i<8;++i) {
                uint d=d_lane*4+i*16;
                float4 k=float4(*reinterpret_cast<device const half4*>(keys+ko+d));
                for(uint h=0;h<4;++h) {
                    ulong qo=(ulong(row)*p[0]+kh*4+h)*128+d;
                    dots[h]+=dot(*reinterpret_cast<device const float4*>(q+qo),k);
                }
            }
        }
        dots+=simd_shuffle_xor(dots,1);
        dots+=simd_shuffle_xor(dots,2);
        if(d_lane==0) for(uint h=0;h<4;++h)
            scores[h*32+key_lane]=token<last ? dots[h]*as_type<float>(p[3]) : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score=scores[sg*32+lane], high=simd_max(score);
        float prob=isfinite(high) ? exp(score-high) : 0.0f;
        float total=simd_sum(prob);
        probability[sg*32+lane]=prob;
        if(lane==0){tile_max[sg]=high;tile_sum[sg]=total;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 next, correction, multiplier;
        for(uint h=0;h<4;++h) {
            next[h]=max(maximum[h],tile_max[h]);
            correction[h]=isfinite(maximum[h]) ? exp(maximum[h]-next[h]) : 0.0f;
            multiplier[h]=exp(tile_max[h]-next[h]);
            accum[h]*=correction[h];
            denominator[h]=denominator[h]*correction[h]+tile_sum[h]*multiplier[h];
        }
        for(uint j=0;j<min(32u,last-base);++j) {
            uint t=base+j, physical=pages[slot*p[2]+t/16]*16+t%16;
            float v=float(values[ulong(physical)*kvwidth+kh*128+tid]);
            for(uint h=0;h<4;++h) accum[h]+=v*probability[h*32+j]*multiplier[h];
        }
        maximum=next;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint h=0;h<4;++h) {
        ulong rowhead=ulong(row)*p[0]+kh*4+h;
        if(p[5]==1) out[rowhead*128+tid]=accum[h]/denominator[h];
        else {
            ulong dst=(rowhead*p[5]+group.z)*130;
            out[dst+tid]=accum[h];
            if(tid==0){out[dst+128]=maximum[h];out[dst+129]=denominator[h];}
        }
    }
}

// All query rows/heads merge in one dispatch, including non-contiguous rows
// in a mixed prefill/decode pass. Empty splits contribute zero mass.
kernel void attention_gqa_merge(device const float* parts [[buffer(0)]], device float* out [[buffer(1)]],
                                device const uint* rows [[buffer(2)]], constant uint* p [[buffer(3)]],
                                uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint rowhead=rows[group/p[1]]*p[1]+group%p[1];
    ulong base=ulong(rowhead)*p[0]*130;
    float maximum=-INFINITY;
    for(uint s=0;s<p[0];++s)maximum=max(maximum,parts[base+s*130+128]);
    float4 acc=0;float denom=0;
    for(uint s=0;s<p[0];++s) {
        ulong src=base+s*130;float d=parts[src+129];
        if(d>0) {
            float correction=exp(parts[src+128]-maximum);
            for(uint e=0;e<4;++e)acc[e]+=parts[src+lane*4+e]*correction;
            denom+=d*correction;
        }
    }
    for(uint e=0;e<4;++e)out[ulong(rowhead)*128+lane*4+e]=acc[e]/denom;
}

// 32-query / 64-key TensorOps attention. Q is converted once per layer;
// K and V reuse one 16 KiB tile instead of keeping two transposed copies.
// Output and P*V products remain in cooperative registers. Four SIMD groups
// share one query tile, so the output accumulator costs 32 floats per lane.
// Online-softmax composition is exact apart from F16 matrix operands.
// p: heads, KV heads, page stride, attention scale, first row, row count.
kernel void attention_query(device const float* q [[buffer(0)]], device half* out [[buffer(1)]],
                             constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    // A sequence can start at any row, so its final 32-row tile may extend
    // beyond the last actual row. Keep those physical reads inside zeroed
    // storage. Invalid query rows never write output or read their metadata.
    if(i<(p[2]+32)*p[0])out[i]=i<p[2]*p[0] ? half(q[i]) : half(0);
}
inline void prefill_tile(device half* q, device const half* kc, device const half* vc,
                         device const uint* meta, device const uint* pages, device float* out,
                         constant uint* p, uint head, uint first, uint count, uint tid,
                         threadgroup half* kv, threadgroup half* probability, threadgroup float* scores,
                         threadgroup float* maximum, threadgroup float* denominator, threadgroup float* correction) {
    uint kh=head/(p[0]/p[1]),slot=meta[2*first];
    uint lastpos=meta[2*(first+count-1)+1],width=p[0]*128,kvwidth=p[1]*128;
    auto tq=tensor(q+ulong(first)*width+head*128,dextents<int,2>(128,count),array<int,2>{1,int(width)});
    auto tk=tensor(kv,extents<int,128,64>(),array<int,2>{1,128});
    auto tv=tensor(kv,extents<int,64,128>(),array<int,2>{1,64});
    auto tp=tensor(probability,extents<int,64,32>(),array<int,2>{1,64});
    auto ts=tensor(scores,extents<int,64,32>(),array<int,2>{1,64});
    constexpr auto qk_desc=matmul2d_descriptor(32,64,128,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc=matmul2d_descriptor(32,128,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc,execution_simdgroups<4>> qk;
    matmul2d<pv_desc,execution_simdgroups<4>> pv;
    auto accum=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<accum.get_capacity();++i)accum[i]=0;
    if(tid<32){maximum[tid]=-INFINITY;denominator[tid]=0;}
    for(uint base=0;base<=lastpos;base+=64) {
        for(uint i=tid;i<64*128;i+=128) {
            uint t=base+i/128,d=i%128;
            uint physical=t<=lastpos ? pages[slot*p[2]+t/16]*16+t%16 : 0;
            kv[i]=t<=lastpos ? kc[ulong(physical)*kvwidth+kh*128+d] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto score=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        for(uint i=0;i<score.get_capacity();++i)score[i]=0;
        qk.run(tq,tk,score);
        score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint row=tid/4,lane=tid%4,pos=row<count ? meta[2*(first+row)+1] : 0;
        float high=maximum[row];
        for(uint j=lane;j<64;j+=4)if(row<count && base+j<=pos)high=max(high,scores[row*64+j]*as_type<float>(p[3]));
        high=max(high,simd_shuffle_xor(high,1));high=max(high,simd_shuffle_xor(high,2));
        float old=isfinite(maximum[row]) ? exp(maximum[row]-high) : 0.0f;
        float sum=0;
        for(uint j=lane;j<64;j+=4) {
            float prob=row<count && base+j<=pos ? exp(scores[row*64+j]*as_type<float>(p[3])-high) : 0.0f;
            probability[row*64+j]=half(prob);sum+=prob;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
        if(lane==0) {
            maximum[row]=high;correction[row]=old;
            denominator[row]=row<count ? denominator[row]*old+sum : 1.0f;
        }
        // All readers of K have completed before this storage becomes V.
        for(uint i=tid;i<64*128;i+=128) {
            uint t=base+i%64,d=i/64;
            uint physical=t<=lastpos ? pages[slot*p[2]+t/16]*16+t%16 : 0;
            kv[i]=t<=lastpos ? vc[ulong(physical)*kvwidth+kh*128+d] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
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
    auto dst=tensor(out+ulong(first)*width+head*128,dextents<int,2>(128,count),array<int,2>{1,int(width)});
    accum.store(dst);
}

// Single-sequence entry retained for independent GPU layout/reference tests.
kernel void attention_prefill(device half* q [[buffer(0)]], device const half* kc [[buffer(1)]],
                              device const half* vc [[buffer(2)]], device const uint* meta [[buffer(3)]],
                              device const uint* pages [[buffer(4)]], device float* out [[buffer(5)]],
                              constant uint* p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[64*128], probability[32*64];
    threadgroup float scores[32*64], maximum[32], denominator[32], correction[32];
    prefill_tile(q,kc,vc,meta,pages,out,p,group.x,p[4]+group.y*32,min(32u,p[5]-group.y*32),tid,
                 kv,probability,scores,maximum,denominator,correction);
}

// Every independent sequence tile shares one dispatch. Tile descriptors never
// cross a slot boundary, including ragged starts and tails; short decode rows
// remain on the GQA split kernel. No per-sequence buffer fences are needed.
kernel void attention_prefill_batched(device half* q [[buffer(0)]], device const half* kc [[buffer(1)]],
                                      device const half* vc [[buffer(2)]], device const uint* meta [[buffer(3)]],
                                      device const uint* pages [[buffer(4)]], device float* out [[buffer(5)]],
                                      device const uint* tiles [[buffer(6)]], constant uint* p [[buffer(7)]],
                                      uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[64*128], probability[32*64];
    threadgroup float scores[32*64], maximum[32], denominator[32], correction[32];
    prefill_tile(q,kc,vc,meta,pages,out,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
                 kv,probability,scores,maximum,denominator,correction);
}
