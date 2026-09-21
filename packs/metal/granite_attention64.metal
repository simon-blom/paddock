// Original hd64 / GQA5 Flash-Decoding for Granite 3B. A 128-thread group
// scores 32 keys at once; each K/V load serves all five query heads. The
// fifth softmax row is handled by SIMD group zero in a second iteration.
// Split states are compact (64 values + max + sum), including empty splits.
// p follows attention_gqa4: Q heads, KV heads, page stride, scale, rows, splits.
kernel void attention_gqa5_64(device const float* q [[buffer(0)]],
                              device const half* keys [[buffer(1)]], device const half* values [[buffer(2)]],
                              device const uint* meta [[buffer(3)]], device const uint* pages [[buffer(4)]],
                              device const uint* rows [[buffer(5)]], device float* out [[buffer(6)]],
                              constant uint* p [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[5*32], probability[5*32], tile_max[5], tile_sum[5];
    uint row=rows[group.y], kh=group.x, slot=meta[2*row], length=meta[2*row+1]+1;
    uint span=(length+p[5]-1)/p[5], first=group.z*span, last=min(first+span,length);
    uint kvwidth=p[1]*64, key_lane=tid/4, d_lane=tid%4;
    float accum[5], maximum[5], denominator[5];
    for(uint h=0;h<5;++h){accum[h]=0;maximum[h]=-INFINITY;denominator[h]=0;}
    for(uint base=first;base<last;base+=32) {
        uint token=base+key_lane;
        float dots[5];for(uint h=0;h<5;++h)dots[h]=0;
        if(token<last) {
            uint physical=pages[slot*p[2]+token/16]*16+token%16;
            ulong ko=ulong(physical)*kvwidth+kh*64;
            for(uint i=0;i<4;++i) {
                uint d=d_lane*4+i*16;
                float4 k=float4(*reinterpret_cast<device const half4*>(keys+ko+d));
                for(uint h=0;h<5;++h) {
                    ulong qo=(ulong(row)*p[0]+kh*5+h)*64+d;
                    dots[h]+=dot(*reinterpret_cast<device const float4*>(q+qo),k);
                }
            }
        }
        for(uint h=0;h<5;++h) {
            dots[h]+=simd_shuffle_xor(dots[h],1);
            dots[h]+=simd_shuffle_xor(dots[h],2);
            if(d_lane==0)scores[h*32+key_lane]=token<last ? dots[h]*as_type<float>(p[3]) : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint h=sg;h<5;h+=4) {
            float score=scores[h*32+lane], high=simd_max(score);
            float prob=isfinite(high) ? exp(score-high) : 0.0f;
            float total=simd_sum(prob);
            probability[h*32+lane]=prob;
            if(lane==0){tile_max[h]=high;tile_sum[h]=total;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Only the lower half owns V/output dimensions; every thread still
        // participates in QK, reductions and barriers above and below.
        if(tid<64) {
            float multiplier[5];
            for(uint h=0;h<5;++h) {
                float next=max(maximum[h],tile_max[h]);
                float correction=isfinite(maximum[h]) ? exp(maximum[h]-next) : 0.0f;
                multiplier[h]=exp(tile_max[h]-next);
                accum[h]*=correction;
                denominator[h]=denominator[h]*correction+tile_sum[h]*multiplier[h];
                maximum[h]=next;
            }
            for(uint j=0;j<min(32u,last-base);++j) {
                uint t=base+j, physical=pages[slot*p[2]+t/16]*16+t%16;
                float v=float(values[ulong(physical)*kvwidth+kh*64+tid]);
                for(uint h=0;h<5;++h)accum[h]+=v*probability[h*32+j]*multiplier[h];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid<64)for(uint h=0;h<5;++h) {
        ulong rowhead=ulong(row)*p[0]+kh*5+h;
        if(p[5]==1)out[rowhead*64+tid]=accum[h]/denominator[h];
        else {
            ulong dst=(rowhead*p[5]+group.z)*66;
            out[dst+tid]=accum[h];
            if(tid==0){out[dst+64]=maximum[h];out[dst+65]=denominator[h];}
        }
    }
}

kernel void attention_gqa_merge64(device const float* parts [[buffer(0)]], device float* out [[buffer(1)]],
                                 device const uint* rows [[buffer(2)]], constant uint* p [[buffer(3)]],
                                 uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint rowhead=rows[group/p[1]]*p[1]+group%p[1];
    ulong base=ulong(rowhead)*p[0]*66;
    float maximum=-INFINITY;
    for(uint s=0;s<p[0];++s)maximum=max(maximum,parts[base+s*66+64]);
    float2 acc=0;float denom=0;
    for(uint s=0;s<p[0];++s) {
        ulong src=base+s*66;float d=parts[src+65];
        if(d>0) {
            float correction=exp(parts[src+64]-maximum);
            for(uint e=0;e<2;++e)acc[e]+=parts[src+lane*2+e]*correction;
            denom+=d*correction;
        }
    }
    for(uint e=0;e<2;++e)out[ulong(rowhead)*64+lane*2+e]=acc[e]/denom;
}

// Independent GPU-only layout witness: one SIMD group/head, sequential keys,
// F32 operands and online softmax. Never elected by the serving graph.
kernel void attention_check64(device const float* q [[buffer(0)]], device const half* keys [[buffer(1)]],
                              device const half* values [[buffer(2)]], device const uint* meta [[buffer(3)]],
                              device const uint* pages [[buffer(4)]], device float* out [[buffer(5)]],
                              constant uint* p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint head=group.x,row=group.y,kh=head/(p[0]/p[1]),slot=meta[2*row],length=meta[2*row+1]+1;
    ulong qo=(ulong(row)*p[0]+head)*64+lane*2;
    float2 query=float2(q[qo],q[qo+1]),acc=0;
    float maximum=-INFINITY,denominator=0;
    for(uint t=0;t<length;++t) {
        uint physical=pages[slot*p[2]+t/16]*16+t%16;
        ulong ko=(ulong(physical)*p[1]+kh)*64+lane*2;
        float2 k=float2(keys[ko],keys[ko+1]),v=float2(values[ko],values[ko+1]);
        float score=simd_sum(dot(query,k))*as_type<float>(p[3]);
        float next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        acc=acc*old+v*prob;denominator=denominator*old+prob;maximum=next;
    }
    out[qo]=acc.x/denominator;out[qo+1]=acc.y/denominator;
}
