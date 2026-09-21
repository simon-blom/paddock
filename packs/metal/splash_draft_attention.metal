// DFlash2's eight noncausal query rows share each KV head. Fuse four GQA
// heads into a full 32-row matrix tile and split the bounded conditioning
// ring across eight independent groups. All state and activations stay BF16.
kernel void splash_df_query(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[2]*32*128)return;
    uint d=i%128,h=i/128%4,row=i/512%8,kh=i/4096%8,cohort=i/32768;
    out[i]=bfloat(x[((cohort*8+row)*32+kh*4+h)*128+d]);
}

kernel void splash_df_attention_grouped(device bfloat* q [[buffer(0)]],device const bfloat* kc [[buffer(1)]],
    device const bfloat* vc [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],
    device float* unused [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint M=32,N=32,D=128,Parts=8;
    uint first=tiles[2*group.y],count=tiles[2*group.y+1],kh=group.x,slot=meta[first*2];
    uint first_pos=meta[first*2+1],last=meta[2*(first+count-1)+1];
    // The trained noncausal block sees all eight noisy rows, while the
    // conditioning prefix has a query-relative left window boundary.
    // Using the last query's boundary for every row incorrectly hides seven
    // still-visible conditioning tokens from the first query at long context.
    uint start=first_pos+1>2048?first_pos+1-2048:0;
    uint span=((last-start+1+Parts*N-1)/(Parts*N))*N;
    uint begin=start+group.z*span,end=min(begin+span,last+1);
    device float* out=reinterpret_cast<device float*>(q+(p[4]+32)*32*128);
    threadgroup bfloat kv[N*D],probs[M*N];
    threadgroup float scores[M*N],maximum[M],denominator[M],correction[M];
    auto tq=tensor(q+(ulong(group.y)*8+kh)*M*D,extents<int,D,M>(),array<int,2>{1,D});
    auto tk=tensor(kv,extents<int,D,N>(),array<int,2>{1,D});
    auto tv=tensor(kv,extents<int,N,D>(),array<int,2>{1,N});
    auto tp=tensor(probs,extents<int,N,M>(),array<int,2>{1,N});
    auto ts=tensor(scores,extents<int,N,M>(),array<int,2>{1,N});
    constexpr auto qkd=matmul2d_descriptor(M,N,D,false,true,false);
    constexpr auto pvd=matmul2d_descriptor(M,D,N,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroups<4>> qk;
    matmul2d<pvd,execution_simdgroups<4>> pv;
    auto acc=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(ushort i=0;i<acc.get_capacity();++i)acc[i]=0;
    if(tid<M){maximum[tid]=-INFINITY;denominator[tid]=0;}
    for(uint base=begin;base<end;base+=N) {
        for(uint i=tid;i<N*D;i+=128) {
            uint t=base+i/D,physical=t<end?pages[slot*p[2]+t/16]*16+t%16:0;
            kv[i]=t<end?kc[(ulong(physical)*8+kh)*D+i%D]:bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto score=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        qk.run(tq,tk,score);score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint row=tid/4,lane=tid%4;
        uint query_pos=first_pos+row/4,row_start=query_pos+1>2048?query_pos+1-2048:0;
        float hi=maximum[row];
        for(uint j=lane;j<N;j+=4)if(base+j<end && base+j>=row_start)hi=max(hi,scores[row*N+j]*as_type<float>(p[3]));
        hi=max(hi,simd_shuffle_xor(hi,1));hi=max(hi,simd_shuffle_xor(hi,2));
        float old=isfinite(maximum[row])?exp(maximum[row]-hi):0,sum=0;
        for(uint j=lane;j<N;j+=4) {
            float pr=base+j<end && base+j>=row_start?exp(scores[row*N+j]*as_type<float>(p[3])-hi):0;
            probs[row*N+j]=bfloat(pr);sum+=pr;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
        if(lane==0){maximum[row]=hi;denominator[row]=denominator[row]*old+sum;correction[row]=old;}
        for(uint i=tid;i<N*D;i+=128) {
            uint t=base+i%N,physical=t<end?pages[slot*p[2]+t/16]*16+t%16:0;
            kv[i]=t<end?vc[(ulong(physical)*8+kh)*D+i/N]:bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(ushort i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))
            acc[i]*=correction[acc.get_multidimensional_index(i)[1]];
        pv.run(tp,tv,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong dst=((ulong(group.y)*8+kh)*Parts+group.z)*M*130;
    for(ushort i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
        auto ij=acc.get_multidimensional_index(i);out[dst+ij[1]*130+ij[0]]=acc[i];
    }
    if(tid<M){out[dst+tid*130+128]=maximum[tid];out[dst+tid*130+129]=denominator[tid];}
}

kernel void splash_df_attention_join(device bfloat* q [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint rh [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=rh/32,head=rh%32,fused=row%8*4+head%4;
    device float* parts=reinterpret_cast<device float*>(q+(p[0]+32)*32*128);
    ulong base=((ulong(row/8)*8+head/4)*8*32+fused)*130;
    float hi=-INFINITY;
    for(uint s=0;s<8;++s)hi=max(hi,parts[base+s*32*130+128]);
    float total=0;float4 acc=0;
    for(uint s=0;s<8;++s) {
        ulong at=base+s*32*130;
        if(parts[at+129]==0)continue;
        float factor=exp(parts[at+128]-hi);total+=parts[at+129]*factor;
        for(uint j=0;j<4;++j)acc[j]+=parts[at+lane+j*32]*factor;
    }
    for(uint j=0;j<4;++j)out[ulong(rh)*128+lane+j*32]=acc[j]/total;
}
