// GPT-OSS-specific arithmetic around the shared native projection and MoE
// operators. Equations follow OpenAI's model and the elected GGUF metadata;
// implementation is original. No host model computation.
// p: query width, KV width, rows, page stride, theta base, scale, correction
// low/high, concentration. GPT-OSS's full NEOX pairs are (d,d+32).
kernel void oss_rope_store(device float* q [[buffer(0)]],device const float* k [[buffer(1)]],
                           device const float* v [[buffer(2)]],device const float* qb [[buffer(3)]],
                           device const float* kb [[buffer(4)]],device const float* vb [[buffer(5)]],
                           device half* keys [[buffer(6)]],device half* values [[buffer(7)]],
                           device const uint* meta [[buffer(8)]],device const uint* pages [[buffer(9)]],
                           constant uint* p [[buffer(10)]],uint i [[thread_position_in_grid]]) {
    uint pairs=(p[0]+p[1])/2;if(i>=p[2]*pairs)return;
    uint row=i/pairs,j=i%pairs,d=j%32;
    float theta=float(meta[2*row+1])*pow(as_type<float>(p[4]),-float(d)/32.0f);
    float ramp=1-clamp((float(d)-as_type<float>(p[6]))/max(.001f,as_type<float>(p[7])-as_type<float>(p[6])),0.0f,1.0f);
    float angle=theta*(as_type<float>(p[5])*(1-ramp)+ramp);
    float c=cos(angle)*as_type<float>(p[8]),s=sin(angle)*as_type<float>(p[8]);
    if(j<p[0]/2) {
        uint n=j/32*64+d;ulong o=ulong(row)*p[0]+n;
        float a=q[o]+qb[n],b=q[o+32]+qb[n+32];q[o]=a*c-b*s;q[o+32]=a*s+b*c;
    } else {
        j-=p[0]/2;uint n=j/32*64+d,pos=meta[2*row+1],slot=meta[2*row];
        uint physical=pages[slot*p[3]+pos/16]*16+pos%16;
        ulong o=ulong(row)*p[1]+n,dst=ulong(physical)*p[1]+n;
        float a=k[o]+kb[n],b=k[o+32]+kb[n+32];
        keys[dst]=half(a*c-b*s);keys[dst+32]=half(a*s+b*c);
        values[dst]=half(v[o]+vb[n]);values[dst+32]=half(v[o+32]+vb[n+32]);
    }
}
kernel void oss_bias_residual(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],
                              device const float* bias [[buffer(2)]],constant uint* p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])x[i]+=delta[i]+bias[i%p[0]];
}
kernel void oss_attention_prefill(device half* q [[buffer(0)]],device const half* kc [[buffer(1)]],
                                  device const half* vc [[buffer(2)]],device const uint* meta [[buffer(3)]],
                                  device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
                                  device const uint* tiles [[buffer(6)]],device const float* sinks [[buffer(7)]],
                                  constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[32*64],probability[32*32];
    threadgroup float scores[32*32],maximum[32],denominator[32],correction[32];
    granite_prefill_tile<64,32,false,true>(q,kc,vc,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,
        kv,probability,scores,maximum,denominator,correction,sinks,p[4]);
}

// GQA8 Flash-Decoding, one K/V tile read for all eight query heads. Sinks
// enter exactly one split's denominator and never contribute a value. Full
// physical paged KV is retained for exact prefix reuse; window masks limit
// SWA computation. Ring+checkpoint storage is a separate memory optimization.
// p: Q heads, KV heads, page stride, scale, rows, splits, sliding window.
kernel void oss_attention_decode(device const float* q [[buffer(0)]],device const half* keys [[buffer(1)]],
                                 device const half* values [[buffer(2)]],device const uint* meta [[buffer(3)]],
                                 device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],
                                 device const float* sinks [[buffer(6)]],device float* out [[buffer(7)]],
                                 constant uint* p [[buffer(8)]],uint3 g [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[8*32],probability[8*32],tile_max[8],tile_sum[8];
    uint row=rows[g.y],kh=g.x,slot=meta[2*row],length=meta[2*row+1]+1;
    uint start=p[6] ? length-min(length,p[6]) : 0,span=(length-start+p[5]-1)/p[5];
    uint first=start+g.z*span,last=min(first+span,length),kvwidth=p[1]*64,key_lane=tid/4,d_lane=tid%4;
    float acc[8],maximum[8],denom[8];
    for(uint h=0;h<8;++h){acc[h]=0;maximum[h]=g.z==0 ? sinks[kh*8+h] : -INFINITY;denom[h]=g.z==0 ? 1.0f : 0.0f;}
    for(uint base=first;base<last;base+=32) {
        uint token=base+key_lane;float dots[8];for(uint h=0;h<8;++h)dots[h]=0;
        if(token<last) {
            uint physical=pages[slot*p[2]+token/16]*16+token%16;
            ulong ko=ulong(physical)*kvwidth+kh*64;
            for(uint i=0;i<4;++i) {
                uint d=d_lane*4+i*16;float4 k=float4(*reinterpret_cast<device const half4*>(keys+ko+d));
                for(uint h=0;h<8;++h)dots[h]+=dot(*reinterpret_cast<device const float4*>(q+(ulong(row)*p[0]+kh*8+h)*64+d),k);
            }
        }
        for(uint h=0;h<8;++h) {
            dots[h]+=simd_shuffle_xor(dots[h],1);dots[h]+=simd_shuffle_xor(dots[h],2);
            if(d_lane==0)scores[h*32+key_lane]=token<last ? dots[h]*as_type<float>(p[3]) : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint h=sg;h<8;h+=4) {
            float score=scores[h*32+lane],high=simd_max(score),prob=isfinite(high) ? exp(score-high) : 0.0f;
            probability[h*32+lane]=prob;float total=simd_sum(prob);
            if(lane==0){tile_max[h]=high;tile_sum[h]=total;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<64) {
            float multiplier[8];
            for(uint h=0;h<8;++h) {
                float next=max(maximum[h],tile_max[h]),old=isfinite(maximum[h]) ? exp(maximum[h]-next) : 0.0f;
                multiplier[h]=exp(tile_max[h]-next);acc[h]*=old;
                denom[h]=denom[h]*old+tile_sum[h]*multiplier[h];maximum[h]=next;
            }
            for(uint j=0;j<min(32u,last-base);++j) {
                uint t=base+j,physical=pages[slot*p[2]+t/16]*16+t%16;
                float v=float(values[ulong(physical)*kvwidth+kh*64+tid]);
                for(uint h=0;h<8;++h)acc[h]+=v*probability[h*32+j]*multiplier[h];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid<64)for(uint h=0;h<8;++h) {
        ulong rowhead=ulong(row)*p[0]+kh*8+h;
        if(p[5]==1)out[rowhead*64+tid]=acc[h]/denom[h];
        else {ulong dst=(rowhead*p[5]+g.z)*66;out[dst+tid]=acc[h];if(tid==0){out[dst+64]=maximum[h];out[dst+65]=denom[h];}}
    }
}

// Independent GPU-only witness for window/sink/page/split tests. Sequential
// keys with one SIMD group per query head, never selected by serving.
kernel void oss_attention_check(device const float* q [[buffer(0)]],device const half* keys [[buffer(1)]],
                                device const half* values [[buffer(2)]],device const uint* meta [[buffer(3)]],
                                device const uint* pages [[buffer(4)]],device const float* sinks [[buffer(5)]],
                                device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],
                                uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint head=g.x,row=g.y,kh=head/8,slot=meta[2*row],length=meta[2*row+1]+1;
    ulong qo=(ulong(row)*64+head)*64+lane*2;
    float2 query=float2(q[qo],q[qo+1]),acc=0;
    float maximum=sinks[head],denominator=1;
    for(uint t=p[1] ? length-min(length,p[1]) : 0;t<length;++t) {
        uint physical=pages[slot*p[0]+t/16]*16+t%16;ulong ko=(ulong(physical)*8+kh)*64+lane*2;
        float2 k=float2(keys[ko],keys[ko+1]),v=float2(values[ko],values[ko+1]);
        float score=simd_sum(dot(query,k))*.125f,next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        acc=acc*old+v*prob;denominator=denominator*old+prob;maximum=next;
    }
    out[qo]=acc.x/denominator;out[qo+1]=acc.y/denominator;
}
