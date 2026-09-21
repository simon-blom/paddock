// Original Gated DeltaNet kernels (Yang et al., arXiv:2412.06464).
// GGUF value heads pair with h % key_heads, not HF's repeat-interleaved
// order. All recurrent state is FP32. Only chunk matrix operands use F16.
// p: key_heads, value_heads, conv_dim, rows, state_slots, linear_layer.
kernel void dn_conv(device const float* input [[buffer(0)]],device const float* w [[buffer(1)]],
                    device const float* history [[buffer(2)]],device const uint* meta [[buffer(3)]],
                    device const uint* bounds [[buffer(4)]],device float* out [[buffer(5)]],
                    constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[2],d=i%p[2];if(row>=p[3])return;
    uint first=bounds[row*2],slot=meta[row*2],pos=meta[first*2+1];
    float sum=0;
    for(uint j=0;j<4;++j) {
        int r=int(row)+int(j)-3;
        float x=r>=int(first) ? input[ulong(r)*p[2]+d]
            : (pos>0 ? history[((ulong(p[5])*p[4]+slot)*3+uint(r-int(first)+3))*p[2]+d] : 0.0f);
        sum+=x*w[d*4+j];
    }
    out[i]=sum/(1.0f+exp(-sum));
}

// One SIMD group normalizes a 128-channel Q/K head; V is untouched.
kernel void dn_qk_norm(device float* qkv [[buffer(0)]],constant uint* p [[buffer(1)]],
                       uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong base=ulong(group.y)*p[2]+group.x*128;
    float4 v=*reinterpret_cast<device float4*>(qkv+base+lane*4);
    float inv=rsqrt(simd_sum(dot(v,v))+1e-6f);
    if(group.x<p[0])inv*=rsqrt(128.0f);
    *reinterpret_cast<device float4*>(qkv+base+lane*4)=v*inv;
}

kernel void dn_gates(device const float* alpha [[buffer(0)]],device const float* beta [[buffer(1)]],
                     device const float* a [[buffer(2)]],device const float* dt [[buffer(3)]],
                     device float* gates [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1]*p[3])return;
    float x=alpha[i]+dt[i%p[1]];
    gates[i*2]=a[i%p[1]]*(max(x,0.0f)+log(1.0f+exp(-abs(x))));
    gates[i*2+1]=1.0f/(1.0f+exp(-beta[i]));
}

// Separate from conv reads: no thread may overwrite a window still consumed
// by another row. Every channel stores its final three pre-convolution values.
kernel void dn_conv_commit(device const float* input [[buffer(0)]],device float* history [[buffer(1)]],
                           device const uint* spans [[buffer(2)]],device const uint* meta [[buffer(3)]],
                           device const uint* checkpoints [[buffer(4)]],constant uint* p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    uint d=group.x*256+tid;if(d>=p[2])return;
    uint first=spans[group.y*4],count=spans[group.y*4+1],slot=spans[group.y*4+2];
    ulong dst=(ulong(p[5])*p[4]+slot)*3*p[2]+d;
    float window[3];
    for(uint j=0;j<3;++j) {
        int r=int(count)+int(j)-3;
        window[j]=r>=0 ? input[ulong(first+r)*p[2]+d]
            : (meta[first*2+1]>0 ? history[dst+uint(r+3)*p[2]] : 0.0f);
    }
    // Snapshot from the original input and incoming window before overwriting
    // the live window. Two per-span boundaries do not fragment projections.
    for(uint c=0;c<2;++c) {
        uint target=checkpoints[group.y*4+c*2+1];if(target==0)continue;
        uint boundary=checkpoints[group.y*4+c*2];
        ulong cache=(ulong(p[5])*p[4]+target-1)*3*p[2]+d;
        for(uint j=0;j<3;++j) {
            int r=int(boundary-first)+int(j)-2;
            history[cache+j*p[2]]=r>=0 ? input[ulong(first+r)*p[2]+d]
                : (meta[first*2+1]>0 ? history[dst+uint(r+3)*p[2]] : 0.0f);
        }
    }
    for(uint j=0;j<3;++j)history[dst+j*p[2]]=window[j];
}

// Packed short-span recurrence: 8 lanes own a value column, 16 FP32 state
// cells per lane. The bounded runtime loop is for decode/tiny tails only;
// normal prefill goes through the chunked TensorOps pair below.
kernel void dn_recurrent(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
                         device float* state [[buffer(2)]],device const uint* spans [[buffer(3)]],
                         device const uint* meta [[buffer(4)]],device float* out [[buffer(5)]],
                         device const uint* checkpoints [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    uint first=spans[group.z*4],count=spans[group.z*4+1];if(count>=16 && p[6]==0)return;
    uint slot=spans[group.z*4+2],h=group.y,v=group.x*16+tid/8,lane=tid%8,kh=h%p[0];
    ulong offset=((ulong(p[5])*p[4]+slot)*p[1]+h)*128*128+v*128;
    float st[16];
    for(uint j=0;j<16;++j)st[j]=meta[first*2+1]==0 ? 0.0f : state[offset+lane+j*8];
    for(uint r=first;r<first+count;++r) {
        float g=exp(gates[(r*p[1]+h)*2]),b=gates[(r*p[1]+h)*2+1];
        float predicted=0;
        for(uint j=0;j<16;++j) {st[j]*=g;predicted+=st[j]*qkv[ulong(r)*p[2]+p[0]*128+kh*128+lane+j*8];}
        predicted+=simd_shuffle_xor(predicted,1);predicted+=simd_shuffle_xor(predicted,2);predicted+=simd_shuffle_xor(predicted,4);
        float delta=b*(qkv[ulong(r)*p[2]+p[0]*256+h*128+v]-predicted),y=0;
        for(uint j=0;j<16;++j) {
            uint d=lane+j*8;
            st[j]+=qkv[ulong(r)*p[2]+p[0]*128+kh*128+d]*delta;
            y+=qkv[ulong(r)*p[2]+kh*128+d]*st[j];
        }
        y+=simd_shuffle_xor(y,1);y+=simd_shuffle_xor(y,2);y+=simd_shuffle_xor(y,4);
        if(lane==0)out[(ulong(r)*p[1]+h)*128+v]=y;
        uint target=checkpoints[r];
        if(target!=0) {
            ulong cache=((ulong(p[5])*p[4]+target-1)*p[1]+h)*128*128+v*128;
            for(uint j=0;j<16;++j)state[cache+lane+j*8]=st[j];
        }
    }
    for(uint j=0;j<16;++j)state[offset+lane+j*8]=st[j];
}

// Preparation is independent over (chunk, head). Keep TensorOps' elected
// threadgroup half operands: moving them to device storage changed real-model
// greedy selection. Split dots from the inverse/solve to fit even instrumented
// Apple10 threadgroup memory, reusing existing W/U planes for FP32 dots.
// Layout per chunk/head: W,U,Qg,Kg (32x128 each), causal QK (32x32).
// p extends common fields with total chunk count at p[6].
kernel void dn_chunk_dots(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
                             device const uint* chunks [[buffer(2)]],device half* prepared [[buffer(3)]],
                             constant uint* p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half a[4096],b[4096];
    uint h=group.x,c=group.y,first=chunks[c*4],count=chunks[c*4+1],kh=h%p[0];
    ulong dst=(ulong(c)*p[1]+h)*17408;
    for(uint i=tid;i<4096;i+=128) {
        uint r=i/128,d=i%128;
        a[i]=r<count ? half(qkv[ulong(first+r)*p[2]+kh*128+d]) : half(0);
        b[i]=r<count ? half(qkv[ulong(first+r)*p[2]+p[0]*128+kh*128+d]) : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto ta=tensor(a,extents<int,128,32>(),array<int,2>{1,128});
    auto tb=tensor(b,extents<int,128,32>(),array<int,2>{1,128});
    auto tk=tensor(reinterpret_cast<device float*>(prepared+dst),extents<int,32,32>(),array<int,2>{1,32});
    auto ts=tensor(reinterpret_cast<device float*>(prepared+dst+4096),extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto dots_desc=matmul2d_descriptor(32,32,128,false,true);
    matmul2d<dots_desc,execution_simdgroups<4>> dots;
    auto product=dots.get_destination_cooperative_tensor<decltype(tb),decltype(tb),float>();
    dots.run(tb,tb,product);product.store(tk);
    dots.run(ta,tb,product);product.store(ts);
}

kernel void dn_chunk_prepare(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
                             device const uint* chunks [[buffer(2)]],device half* prepared [[buffer(3)]],
                             constant uint* p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    // The 8 KiB FP32 inverse workspace becomes one half matrix operand
    // after conversion to the separate 2 KiB inverse plane. W and U then
    // reuse that same operand, with a barrier between the two solves.
    threadgroup float work[2048],g[32],beta[32];
    threadgroup half th[1024];
    threadgroup float* kk=work;
    threadgroup float* t=work+1024;
    uint h=group.x,c=group.y,first=chunks[c*4],count=chunks[c*4+1],kh=h%p[0];
    ulong dst=(ulong(c)*p[1]+h)*17408;
    device const float* dotk=reinterpret_cast<device const float*>(prepared+dst);
    device const float* dotq=reinterpret_cast<device const float*>(prepared+dst+4096);
    if(tid==0) {float sum=0;for(uint r=0;r<32;++r) {
        if(r<count)sum+=gates[((first+r)*p[1]+h)*2];
        g[r]=sum;beta[r]=r<count ? gates[((first+r)*p[1]+h)*2+1] : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=128) {
        uint r=i/32,j=i%32;
        float decay=j<=r ? exp(g[r]-g[j]) : 0.0f;
        prepared[dst+16384+i]=half(r<count && j<=r ? dotq[i]*decay : 0.0f);
        kk[i]=j<r ? dotk[i]*beta[r]*decay : 0.0f;
        t[i]=r==j ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // FP32 triangular solve. The inverse never accumulates in half precision.
    // Parallel columns, uniform trip counts; future changes must gate both
    // whole-model margins and resumed-state drift, not just kernel error.
    for(uint r=1;r<32;++r) {
        if(tid<32) {
            float v=0;for(uint j=0;j<r;++j)v+=kk[r*32+j]*t[j*32+tid];
            t[r*32+tid]-=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=tid;i<1024;i+=128)th[i]=half(t[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup half* a=reinterpret_cast<threadgroup half*>(work);
    for(uint i=tid;i<4096;i+=128) {
        uint r=i/128,d=i%128;
        float k=r<count ? qkv[ulong(first+r)*p[2]+p[0]*128+kh*128+d] : 0.0f;
        float q=r<count ? qkv[ulong(first+r)*p[2]+kh*128+d] : 0.0f;
        prepared[dst+8192+i]=half(q*exp(g[r]));
        prepared[dst+12288+i]=half(k*exp(g[31]-g[r]));
        a[i]=half(k*beta[r]*exp(g[r]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto inv=tensor(th,extents<int,32,32>(),array<int,2>{1,32});
    auto ta=tensor(a,extents<int,128,32>(),array<int,2>{1,128});
    constexpr auto solve_desc=matmul2d_descriptor(32,128,32,false,false);
    matmul2d<solve_desc,execution_simdgroups<4>> solve;
    auto u=solve.get_destination_cooperative_tensor<decltype(inv),decltype(ta),float>();
    solve.run(inv,ta,u);
    for(auto it=u.begin();it!=u.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();prepared[dst+ij[1]*128+ij[0]]=half(*it);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<4096;i+=128) {
        uint r=i/128,d=i%128;
        a[i]=half(r<count ? qkv[ulong(first+r)*p[2]+p[0]*256+h*128+d]*beta[r] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    solve.run(inv,ta,u);
    for(auto it=u.begin();it!=u.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();prepared[dst+4096+ij[1]*128+ij[0]]=half(*it);
    }
}

// MoE's narrow, repeatedly routed graph amplifies half-boundary drift. Keep
// the same WY factorization, but stage strict F32 tiles in bounded 32-column
// panels. No full-weight/state conversion and no token-serial prefill.
kernel void dn_chunk_dots_strict(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
    device const uint* chunks [[buffer(2)]],device float* prepared [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[1024],b[1024];
    uint h=group.x,c=group.y,first=chunks[c*4],count=chunks[c*4+1],kh=h%p[0];
    ulong dst=(ulong(c)*p[1]+h)*17408;
    auto ta=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto tb=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto kk=op.get_destination_cooperative_tensor<decltype(tb),decltype(tb),float>();
    auto qk=op.get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<kk.get_capacity();++i){kk[i]=0;qk[i]=0;}
    for(uint base=0;base<128;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=i/32,d=base+i%32;
            a[i]=r<count ? qkv[ulong(first+r)*p[2]+kh*128+d] : 0.0f;
            b[i]=r<count ? qkv[ulong(first+r)*p[2]+p[0]*128+kh*128+d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tb,tb,kk);op.run(ta,tb,qk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tk=tensor(prepared+dst,extents<int,32,32>(),array<int,2>{1,32});
    auto tq=tensor(prepared+dst+4096,extents<int,32,32>(),array<int,2>{1,32});
    kk.store(tk);qk.store(tq);
}

kernel void dn_chunk_prepare_strict(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
    device const uint* chunks [[buffer(2)]],device float* prepared [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    // Reuse the lower triangle as one 32x32 operand after inversion. The
    // inverse remains in the other half of work through all eight solves.
    threadgroup float work[2048],g[32],beta[32];
    threadgroup float* a=work;
    threadgroup float* t=work+1024;
    uint h=group.x,c=group.y,first=chunks[c*4],count=chunks[c*4+1],kh=h%p[0];
    ulong dst=(ulong(c)*p[1]+h)*17408;
    if(tid==0) {float sum=0;for(uint r=0;r<32;++r) {
        if(r<count)sum+=gates[((first+r)*p[1]+h)*2];
        g[r]=sum;beta[r]=r<count ? gates[((first+r)*p[1]+h)*2+1] : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=128) {
        uint r=i/32,j=i%32;float decay=j<=r ? exp(g[r]-g[j]) : 0.0f;
        prepared[dst+16384+i]=r<count && j<=r ? prepared[dst+4096+i]*decay : 0.0f;
        a[i]=j<r ? prepared[dst+i]*beta[r]*decay : 0.0f;
        t[i]=r==j ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint r=1;r<32;++r) {
        if(tid<32) {
            float v=0;for(uint j=0;j<r;++j)v+=a[r*32+j]*t[j*32+tid];
            t[r*32+tid]-=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto inv=tensor(t,extents<int,32,32>(),array<int,2>{1,32});
    auto ta=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,false);
    matmul2d<desc,execution_simdgroups<4>> solve;
    auto product=solve.get_destination_cooperative_tensor<decltype(inv),decltype(ta),float>();
    for(uint plane=0;plane<2;++plane)for(uint base=0;base<128;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=i/32,d=base+i%32;
            if(plane==0) {
                float k=r<count ? qkv[ulong(first+r)*p[2]+p[0]*128+kh*128+d] : 0.0f;
                float q=r<count ? qkv[ulong(first+r)*p[2]+kh*128+d] : 0.0f;
                prepared[dst+8192+r*128+d]=q*exp(g[r]);
                prepared[dst+12288+r*128+d]=k*exp(g[31]-g[r]);
                a[i]=k*beta[r]*exp(g[r]);
            } else a[i]=r<count ? qkv[ulong(first+r)*p[2]+p[0]*256+h*128+d]*beta[r] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        solve.run(inv,ta,product);
        auto output=tensor(prepared+dst+plane*4096+base,extents<int,32,32>(),array<int,2>{1,128});
        product.store(output);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Stage two walks chunks, not tokens. A value-column tile retains its FP32
// state in cooperative registers for the full span. State is never spilled
// to device memory between chunks. Four matmuls implement the WY update.
template<typename T>
inline void dn_walk(device T* prepared,device const float* gates,
                    device const uint* spans,device const uint* chunks,device const uint* meta,
                    device float* state,device float* out,constant uint* p,uint3 group,uint tid,
                    threadgroup T* sh,threadgroup T* delta) {
    uint first=spans[group.z*4],count=spans[group.z*4+1];if(count<16)return;
    uint slot=spans[group.z*4+2],chunk=spans[group.z*4+3],h=group.y,v=group.x*16;
    ulong offset=((ulong(p[5])*p[4]+slot)*p[1]+h)*128*128;
    // Store is [value,key], transpose via strides to the mathematical KxV S.
    auto hs=tensor(sh,extents<int,16,128>(),array<int,2>{1,16});
    auto ds=tensor(delta,extents<int,16,32>(),array<int,2>{1,16});
    constexpr auto update_desc=matmul2d_descriptor(128,16,32,true,false);
    matmul2d<update_desc,execution_simdgroups<4>> update;
    auto stencil=tensor(prepared,extents<int,128,32>(),array<int,2>{1,128});
    auto st=update.template get_destination_cooperative_tensor<decltype(stencil),decltype(ds),float>();
    for(auto it=st.begin();it!=st.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();
        *it=meta[first*2+1]==0 ? 0.0f : state[offset+(v+ij[0])*128+ij[1]];
    }
    constexpr auto read_desc=matmul2d_descriptor(32,16,128,false,false);
    constexpr auto local_desc=matmul2d_descriptor(32,16,32,false,false);
    matmul2d<read_desc,execution_simdgroups<4>> read;
    matmul2d<local_desc,execution_simdgroups<4>> local;
    for(uint consumed=0;consumed<count;++chunk) {
        uint n=chunks[chunk*4+1],row=chunks[chunk*4];
        ulong base=(ulong(chunk)*p[1]+h)*17408;
        auto w=tensor(prepared+base,extents<int,128,32>(),array<int,2>{1,128});
        auto u=tensor(prepared+base+4096+v,extents<int,16,32>(),array<int,2>{1,128});
        auto q=tensor(prepared+base+8192,extents<int,128,32>(),array<int,2>{1,128});
        auto k=tensor(prepared+base+12288,extents<int,128,32>(),array<int,2>{1,128});
        auto m=tensor(prepared+base+16384,extents<int,32,32>(),array<int,2>{1,32});
        for(auto it=st.begin();it!=st.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();sh[ij[1]*16+ij[0]]=T(*it);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto d=read.template get_destination_cooperative_tensor<decltype(w),decltype(hs),float>();
        auto uv=read.template get_destination_cooperative_tensor<decltype(w),decltype(hs),float>();
        for(auto it=uv.begin();it!=uv.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();*it=float(prepared[base+4096+ij[1]*128+v+ij[0]]);
        }
        read.run(w,hs,d);
        for(uint i=0;i<d.get_capacity();++i)d[i]=uv[i]-d[i];
        for(auto it=d.begin();it!=d.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();delta[ij[1]*16+ij[0]]=T(*it);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto y=read.template get_destination_cooperative_tensor<decltype(q),decltype(hs),float>();
        read.run(q,hs,y);
        auto yd=local.template get_destination_cooperative_tensor<decltype(m),decltype(ds),float>();local.run(m,ds,yd);
        for(uint i=0;i<y.get_capacity();++i)y[i]+=yd[i];
        auto output=tensor(out+(ulong(row)*p[1]+h)*128+v,dextents<int,2>(16,n),array<int,2>{1,int(p[1]*128)});
        y.store(output);
        auto change=update.template get_destination_cooperative_tensor<decltype(k),decltype(ds),float>();update.run(k,ds,change);
        float log_decay=0;for(uint r=0;r<n;++r)log_decay+=gates[((row+r)*p[1]+h)*2];
        float decay=exp(log_decay);
        for(uint i=0;i<st.get_capacity();++i)st[i]=st[i]*decay+change[i];
        uint target=chunks[chunk*4+3];
        if(target!=0) {
            ulong cache=((ulong(p[5])*p[4]+target-1)*p[1]+h)*128*128;
            for(auto it=st.begin();it!=st.end();++it)if(it.is_valid_element()) {
                auto ij=it.get_multidimensional_index();state[cache+(v+ij[0])*128+ij[1]]=*it;
            }
        }
        consumed+=n;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=st.begin();it!=st.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();state[offset+(v+ij[0])*128+ij[1]]=*it;
    }
}

#define DN_WALK(NAME,T) \
kernel void NAME(device T* prepared [[buffer(0)]],device const float* gates [[buffer(1)]], \
 device const uint* spans [[buffer(2)]],device const uint* chunks [[buffer(3)]],device const uint* meta [[buffer(4)]], \
 device float* state [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]], \
 uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup T sh[128*16],delta[32*16];dn_walk<T>(prepared,gates,spans,chunks,meta,state,out,p,group,tid,sh,delta); }
DN_WALK(dn_chunk_walk,half)
DN_WALK(dn_chunk_walk_strict,float)
#undef DN_WALK

kernel void dn_gated_norm(device float* x [[buffer(0)]],device const float* z [[buffer(1)]],
                          device const float* w [[buffer(2)]],constant uint* p [[buffer(3)]],
                          uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong base=(ulong(group.y)*p[1]+group.x)*128+lane*4;
    float4 v=*reinterpret_cast<device float4*>(x+base);
    float inv=rsqrt(simd_sum(dot(v,v))/128.0f+as_type<float>(p[6]));
    float4 gate=*reinterpret_cast<device const float4*>(z+base);
    float4 norm=*reinterpret_cast<device const float4*>(w+lane*4);
    *reinterpret_cast<device float4*>(x+base)=v*inv*norm*(gate/(1.0f+exp(-gate)));
}

// One bounded copy snapshots/restores every recurrent layer and its conv
// window at an exact token boundary. Prefix metadata and paged KV are pinned
// together by the host; partial-prefix state restoration is never allowed.
kernel void dn_checkpoint(device float* state [[buffer(0)]],device float* conv [[buffer(1)]],
                          constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint width=p[3]+p[4],layer=i/width,d=i%width;if(layer>=p[5])return;
    if(d<p[3])state[(ulong(layer)*p[2]+p[1])*p[3]+d]=state[(ulong(layer)*p[2]+p[0])*p[3]+d];
    else {d-=p[3];conv[(ulong(layer)*p[2]+p[1])*p[4]+d]=conv[(ulong(layer)*p[2]+p[0])*p[4]+d];}
}
