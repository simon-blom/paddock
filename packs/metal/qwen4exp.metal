// Original Flash Next residual/PLE/DeltaNet execution. GGUF norms already contain
// (1+w): consume their stored gamma directly. No raw-HF +1 transform here.
// Graph equations studied in Qwen's release and llama.cpp's qwen4exp graph;
// this is our GPU implementation, not a port of external kernel code.
// Accumulate per-layer validation before shared scratch is overwritten.
kernel void q4x_status(device const uint* flags [[buffer(0)]],device atomic_uint* bad [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0] && flags[i]>p[1])atomic_fetch_or_explicit(bad,p[2],memory_order_relaxed);
}
kernel void q4x_select_rows(device const float* x [[buffer(0)]],device const uint* rows [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])y[i]=x[ulong(rows[i/p[0]])*p[0]+i%p[0]];
}
kernel void q4x_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    uint width=p[0],groups=p[1],base=(g.y*groups+g.x)*width;
    float acc=0;
    for(uint d=tid;d<width;d+=256)acc=fma(x[base+d],x[base+d],acc);
    float sum=simd_sum(acc);
    if(tid%32==0)partial[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(tid<8 ? partial[tid] : 0.0f);
    if(tid==0)partial[0]=rsqrt(sum/float(width)+as_type<float>(p[2]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<width;d+=256)y[base+d]=(x[base+d]*partial[0])*w[g.x*width+d];
}
kernel void q4x_hc_init(device const float* x [[buffer(0)]],device float* h [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1]*4)h[i]=x[(i/(p[0]*4))*p[0]+i%p[0]];
}
kernel void q4x_scale_silu(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if(i<p[0]) {float v=x[i]*0.25f;x[i]=v/(1.0f+exp(-v));}
}
kernel void q4x_hc_mix(device const float* norm [[buffer(0)]],device const float* gate [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;
    uint base=(i/p[0])*p[0]*4+i%p[0];float sum=0;
    // Fixed stream order in both decode and prefill; no atomic scatter.
    for(uint s=0;s<4;++s)sum+=norm[base+s*p[0]]/(1.0f+exp(-gate[base+s*p[0]]));
    y[i]=sum*0.25f;
}
kernel void q4x_hc_combine(device float* h [[buffer(0)]],device const float* delta [[buffer(1)]],
    device const float* inject [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]*4)return;
    float gain=2.0f/(1.0f+exp(-inject[i/p[0]]*0.25f));
    h[i]+=delta[(i/(p[0]*4))*p[0]+i%p[0]]*gain;
}
// Original F32-only projection: retain the scalar lane/reduction order, but
// remove the general quant-format decoder from every inner-loop iteration.
// HC injection, recurrent gates and routing planes are validated F32 weights.
kernel void q4x_f32_mv(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n=g.x*4+sg,m=g.y;
    if(n>=p[1] || m>=p[2])return;
    float sum=0;
    for(uint k=lane;k<p[0];k+=32)sum+=w[ulong(n)*p[0]+k]*x[ulong(m)*p[0]+k];
    sum=simd_sum(sum);
    if(lane==0)y[ulong(m)*p[1]+n]=sum*as_type<float>(p[4]);
}
// HC-only projection: eight SIMD groups cooperate on each 10,240-wide dot.
// Fixed reduction tree, no atomics, no F16 inputs, and no batch-dependent
// arithmetic. Same-weight GPU and full-generation gates cover the changed
// summation order; never use this fixed-shape kernel for arbitrary planes.
kernel void q4x_inject_parallel(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    float sum=0;
    for(uint k=tid;k<10240;k+=256)sum+=w[g.x*10240+k]*x[g.y*10240+k];
    sum=simd_sum(sum);
    if(lane==0)partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(tid<8 ? partial[tid] : 0.0f);
    if(tid==0)y[g.y*4+g.x]=sum;
}
// F32 operands on both sides of the Q8 contraction. Bounded 32x32 tiles,
// never a full-plane decode or batch-dependent F16 activation conversion.
kernel void q4x_q8_mm(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint row=g.y*32,col=g.x*32;
    for(uint base=0;base<p[0];base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=row+i/32,k=base+i%32;
            a[i]=r<p[2] ? x[ulong(r)*p[0]+k] : 0;
        }
        for(uint i=tid*4;i<1024;i+=512) {
            uint n=col+i/32,k=base+i%32;float4 v=0;
            if(n<p[1]) {
                device const uchar* block=w+((ulong(n)*p[0]+k)/32)*34;
                v=float4(*reinterpret_cast<device const packed_char4*>(block+2+k%32))
                    *float(*reinterpret_cast<device const half*>(block));
            }
            *reinterpret_cast<threadgroup float4*>(b+i)=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && row+ij[1]<p[2] && col+ij[0]<p[1])
            y[ulong(row+ij[1])*p[1]+col+ij[0]]=*it;
    }
}

// Two independent K=32 banks amortize the threadgroup barriers over two
// contractions. Keep the original operand layout and K=32 accumulation order
// in each TensorOps call; no precision boundary or whole-plane expansion.
// Caller elects this only for K divisible by 64. Explicit staging is 16 KiB.
kernel void q4x_q8_mm64(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[2048],b[2048];
    auto a0=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto b0=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    auto a1=tensor(a+1024,extents<int,32,32>(),array<int,2>{1,32});
    auto b1=tensor(b+1024,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(a0),decltype(b0),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint row=g.y*32,col=g.x*32;
    for(uint base=0;base<p[0];base+=64) {
        for(uint i=tid;i<2048;i+=128) {
            uint local=i%1024,r=row+local/32,k=base+(i/1024)*32+local%32;
            a[i]=r<p[2] ? x[ulong(r)*p[0]+k] : 0;
        }
        for(uint i=tid*4;i<2048;i+=512) {
            uint local=i%1024,n=col+local/32,k=base+(i/1024)*32+local%32;float4 v=0;
            if(n<p[1]) {
                device const uchar* block=w+((ulong(n)*p[0]+k)/32)*34;
                v=float4(*reinterpret_cast<device const packed_char4*>(block+2+k%32))
                    *float(*reinterpret_cast<device const half*>(block));
            }
            *reinterpret_cast<threadgroup float4*>(b+i)=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(a0,b0,acc);op.run(a1,b1,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && row+ij[1]<p[2] && col+ij[0]<p[1])
            y[ulong(row+ij[1])*p[1]+col+ij[0]]=*it;
    }
}

// Skinny HC down projection: N=320 leaves only 10..40 ordinary prefill
// tiles. Split its long K dimension to expose work to more GPU cores.
// Split count is an internal, validated election; the companion reduction
// visits partials in fixed order. No atomic accumulation or half operands.
kernel void q4x_hc_down_split(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* partial [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint row=g.y*32,col=g.x*32,span=10240/p[1],first=g.z*span;
    for(uint base=first;base<first+span;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=row+i/32,k=base+i%32;
            a[i]=r<p[0] ? x[ulong(r)*10240+k] : 0;
        }
        for(uint i=tid*4;i<1024;i+=512) {
            uint n=col+i/32,k=base+i%32;
            device const uchar* block=w+((ulong(n)*10240+k)/32)*34;
            *reinterpret_cast<threadgroup float4*>(b+i)=
                float4(*reinterpret_cast<device const packed_char4*>(block+2+k%32))
                *float(*reinterpret_cast<device const half*>(block));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && row+ij[1]<p[0])
            partial[(ulong(g.z)*p[0]+row+ij[1])*320+col+ij[0]]=*it;
    }
}
kernel void q4x_hc_down_reduce(device const float* partial [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint size=p[0]*320;if(i>=size)return;
    float sum=0;for(uint s=0;s<p[1];++s)sum+=partial[ulong(s)*size+i];
    y[i]=sum;
}

kernel void q4x_ple_gate(device const float* key [[buffer(0)]],device const float* query [[buffer(1)]],
    device const float* value [[buffer(2)]],device float* gated [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];uint width=p[0],base=(g.y*4+g.x)*width;
    float acc=0;for(uint d=tid;d<width;d+=256)acc=fma(key[base+d],query[base+d],acc);
    float sum=simd_sum(acc);if(tid%32==0)partial[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(tid<8 ? partial[tid] : 0.0f);
    if(tid==0) {
        float score=sum*rsqrt(float(width));
        // sign(0)=0, not copysign(1,0). Preserve NaN rather than predicting
        // a plausible value from corrupted activations.
        float signed_root=score==0 ? 0 : copysign(sqrt(max(abs(score),1e-6f)),score);
        partial[0]=isfinite(score) ? 1.0f/(1.0f+exp(-signed_root)) : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<width;d+=256)gated[base+d]=value[g.y*width+d]*partial[0];
}

constant ulong q4x_hash_mult[3]={23703573157769ul,20109073645365ul,8052911324071ul};
constant uint q4x_hash_sizes[16]={20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
    20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171};
// Metadata per row: slot, absolute position, first row of this contiguous
// span, exclusive end. History is immutable during all row-parallel reads.
kernel void q4x_ple_hash(device const uint* tokens [[buffer(0)]],device const uint4* meta [[buffer(1)]],
    device const uint* history [[buffer(2)]],device uint* ids [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint head [[thread_index_in_threadgroup]]) {
    if(row>=p[0] || head>=16)return;
    uint4 m=meta[row];
    if(m.x>=p[1] || m.z>row || m.w<=row || m.w>p[0]) {ids[row*16+head]=0xffffffffu;return;}
    uint cur=tokens[row],prev=row>m.z ? tokens[row-1] : history[m.x*2];
    uint prev2=row>=m.z+2 ? tokens[row-2] : (row>m.z ? history[m.x*2] : history[m.x*2+1]);
    if(prev==248044)prev2=248044;
    if(cur>=248320 || prev>=248320 || prev2>=248320) {ids[row*16+head]=0xffffffffu;return;}
    // Elected token ids make all products <2^63. Keep 64-bit integer math;
    // converting these products to floating point destroys low hash bits.
    ulong hash=ulong(cur)*q4x_hash_mult[0] ^ ulong(prev)*q4x_hash_mult[1];
    if(head>=8)hash^=ulong(prev2)*q4x_hash_mult[2];
    uint offset=0;for(uint j=0;j<head;++j)offset+=q4x_hash_sizes[j];
    ids[row*16+head]=uint(hash%ulong(q4x_hash_sizes[head]))+offset;
}
// Independent rows read either this chunk or the old ring. The ring is not
// touched until the separate commit dispatch: a long chunk cannot evict a
// history row while an earlier threadgroup still needs it.
kernel void q4x_ple_conv(device const float* norm [[buffer(0)]],device const float* w [[buffer(1)]],
    device const float* ring [[buffer(2)]],device const uint4* meta [[buffer(3)]],
    device const float* gated [[buffer(4)]],device float* h [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    uint width=p[0];if(i>=width*p[1])return;
    uint row=i/width,d=i%width;uint4 m=meta[row];float acc=0;
    for(uint tap=0;tap<4;++tap) {
        uint back=(3-tap)*3;float v=0;
        if(m.y>=back) {
            if(row-m.z>=back)v=norm[(row-back)*width+d];
            else v=ring[(ulong(m.x)*9+(m.y-back)%9)*width+d];
        }
        acc+=w[d*4+tap]*v;
    }
    h[i]+=gated[i]+acc/(1.0f+exp(-acc));
}
// Span descriptor: first, count, slot, absolute start. Only the last nine
// writes survive; ring slots not written by a short chunk remain untouched.
kernel void q4x_ple_commit(device const float* norm [[buffer(0)]],device float* ring [[buffer(1)]],
    device const uint4* spans [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint width=p[0];if(i>=p[1]*9*width)return;
    uint span=i/(9*width),r=(i/width)%9,d=i%width;uint4 s=spans[span];
    uint last=s.w+s.y-1,back=(last+9-r)%9;
    if(back<s.y)ring[(ulong(s.z)*9+r)*width+d]=norm[(s.x+s.y-1-back)*width+d];
}
kernel void q4x_ple_tokens_commit(device const uint* tokens [[buffer(0)]],device uint* history [[buffer(1)]],
    device const uint4* spans [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0])return;uint4 s=spans[i];
    uint prev=s.y>1 ? tokens[s.x+s.y-2] : history[s.z*2];
    history[s.z*2]=tokens[s.x+s.y-1];history[s.z*2+1]=prev;
}
kernel void q4x_ple_reset(device float* ring [[buffer(0)]],device uint* history [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<9*p[0])ring[ulong(p[1])*9*p[0]+i]=0;
    if(i<2)history[p[1]*2+i]=248044;
}

// Flash Next's sole GDN norm difference: sigmoid(z), never SiLU(z).
// ssm_norm is a direct gamma, unlike the folded HC/PLE norm parameters.
kernel void q4x_dn_gated_norm(device float* x [[buffer(0)]],device const float* z [[buffer(1)]],
    device const float* w [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong base=(ulong(g.y)*48+g.x)*128+lane*4;
    float4 v=*reinterpret_cast<device const float4*>(x+base);
    float inv=rsqrt(simd_sum(dot(v,v))/128.0f+as_type<float>(p[0]));
    float4 gate=*reinterpret_cast<device const float4*>(z+base);
    float4 gamma=*reinterpret_cast<device const float4*>(w+lane*4);
    *reinterpret_cast<device float4*>(x+base)=(v*inv)*gamma/(1.0f+exp(-gate));
}
kernel void q4x_dn_reset(device float* state [[buffer(0)]],device float* history [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<48*128*128)state[ulong(p[0])*48*128*128+i]=0;
    if(i<3*10240)history[ulong(p[0])*3*10240+i]=0;
}
