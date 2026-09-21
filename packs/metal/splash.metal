// Paddock packed-Q4 operator. Storage contract: Splash schema 3, group64,
// tile256. Original implementation; no Splash runtime dependency.
// Cooperative traversal adapted from Inco AI Splash (Apache-2.0), modified
// for Paddock's split-K/shared-input operators. See splash.NOTICE.md.
template<class T> inline uint splash_traversal(thread const T& t) {
    bool all=true,striped=t.get_capacity()%8==0,half_prefix=t.get_capacity()%2==0;
    bool prefix=t.get_capacity()%16==0;
    #pragma unroll
    for(ushort i=0;i<t.get_capacity();++i) {
        bool valid=t.is_valid_element(i);
        all&=valid;striped&=valid==((i&7)<4);half_prefix&=valid==(i<t.get_capacity()/2);
        prefix&=valid==(i<t.get_capacity()/2 || (i&7)<4);
    }
    return all?0:striped?1:half_prefix?2:prefix?3:4;
}
template<class T,class F> __attribute__((always_inline)) inline void splash_visit(thread const T& t,uint mode,thread const F& f) {
    if(mode==0) {
        #pragma unroll
        for(ushort i=0;i<t.get_capacity();++i)f(i);
    } else if(mode==1) {
        #pragma unroll
        for(ushort i=0;i<t.get_capacity()/2;++i)f(ushort(i/4*8+i%4));
    } else if(mode==2 || mode==3) {
        #pragma unroll
        for(ushort i=0;i<t.get_capacity()/2;++i)f(i);
        if(mode==3) {
            #pragma unroll
            for(ushort i=0;i<t.get_capacity()/4;++i)f(ushort(t.get_capacity()/2+i/4*8+i%4));
        }
    } else {
        #pragma unroll
        for(ushort i=0;i<t.get_capacity();++i)if(t.is_valid_element(i))f(i);
    }
}
kernel void splash_image_copy(device const float* src [[buffer(0)]],device float* dst [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[2])dst[p[1]+i]=float(bfloat(src[p[0]+i]));
}
kernel void splash_input(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint groups=p[0]/64,row=g/groups,base=row*p[0]+g%groups*64;
    bfloat a=row<p[1]?bfloat(x[ulong(p[3])*p[0]+base+lane]):bfloat(0);
    bfloat b=row<p[1]?bfloat(x[ulong(p[3])*p[0]+base+lane+32]):bfloat(0);
    out[base+lane]=a;out[base+lane+32]=b;
    float sum=simd_sum(float(a)+float(b));
    device float* sums=reinterpret_cast<device float*>(out+p[2]*p[0]);
    // A matrix tile consumes its row sums across every quantization group.
    // Keep that complete working set contiguous, not batch-strided. Tiny
    // decode cohorts use their own padded width without extra preparation.
    uint tile=min(32u,p[2]);
    if(lane==0)sums[(row/tile)*tile*groups+(g%groups)*tile+row%tile]=sum;
}
template <ushort BM,ushort SG=8,bool Pipeline=false,bool PairPartials=false,bool CompactOutput=false,bool GuardedOutput=false>
inline void splash_affine_tile(device uchar* weights,device bfloat* x,
    device float* out,constant uint* p,uint3 grid,ulong partial_offset=0) {
    constexpr uint BN=128;
    constexpr bool Store=BM==32 && SG==4;
    uint K=p[0],N=p[1],M=p[2],ng=K/64,np=((N+255)/256)*256;
    uint column=grid.x*BN,row=grid.y*BM,storage_tile=column/256,subtile=column%256;
    device bfloat* scales=reinterpret_cast<device bfloat*>(weights+ulong(K)*np/2);
    device bfloat* biases=scales+ulong(K)*np/64;
    auto a=tensor(x+ulong(row)*K,dextents<int,2>{int(K),BM},array<int,2>{1,int(K)});
    device uchar* base=weights+ulong(storage_tile)*ng*256*32+subtile*32;
    tensor<device uint4b_format,dextents<int,2>,tensor_inline> b(base,dextents<int,2>{64,BN},array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(BM,BN,64,false,true,false);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto aa=a.slice<64,BM>(0,0);
    auto bb=b.slice<64,BN>(0,0);
    auto sum=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
    auto partial=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
    uint traversal=Store && uint(sum.get_capacity())*(SG*32u)==uint(BM)*BN?0:splash_traversal(sum);
    #pragma unroll
    for(ushort i=0;i<sum.get_capacity();++i)sum[i]=0;
    device float* input_sums=reinterpret_cast<device float*>(x+p[4]*K);
    uint parts=p[3],first=grid.z*(ng/parts),end=first+ng/parts;
    auto run=[&](uint g,thread decltype(partial)& result) {
        auto ax=a.slice<64,BM>(g*64,0);
        tensor<device uint4b_format,dextents<int,2>,tensor_inline> bx(base+ulong(g)*256*32,dextents<int,2>{64,BN},array<int,2>{1,64});
        auto bs=bx.slice<64,BN>(0,0);
        op.run(ax,bs,result);
    };
    auto finish=[&](uint g,thread decltype(partial)& result) {
        auto element=[&](ushort i) __attribute__((always_inline)) {
            auto ij=sum.get_multidimensional_index(i);
            ulong parameter=(ulong(storage_tile)*ng+g)*256+subtile+ij[0];
            uint sr=row+ij[1],st=min(32u,p[4]);
            uint at=BM==32?row*ng+g*32+ij[1]:(sr/st)*st*ng+g*st+sr%st;
            float sx=input_sums[at];
            sum[i]+=result[i]*float(scales[parameter])+sx*float(biases[parameter]);
        };
        splash_visit(sum,traversal,element);
    };
    uint g=first;
    if constexpr(Pipeline) {
        for(;g+1<end;g+=2) {
            decltype(partial) next;
            run(g,partial);run(g+1,next);
            finish(g,partial);finish(g+1,next);
        }
    }
    for(;g<end;++g){run(g,partial);finish(g,partial);}
    if constexpr(CompactOutput) {
        // Full-K results already have a BF16 boundary. A BF16 cooperative
        // store avoids the considerably more expensive F32 tensor store;
        // the next epilogue widens or activates without changing that math.
        device bfloat* destination=reinterpret_cast<device bfloat*>(input_sums+p[4]*ng)+partial_offset+ulong(row)*N;
        if constexpr(GuardedOutput) {
            // A separate last-tile dispatch keeps this guard out of the
            // cooperative store specialization, including for ragged prompts.
            #pragma unroll
            for(ushort i=0;i<sum.get_capacity();++i)if(sum.is_valid_element(i)) {
                auto ij=sum.get_multidimensional_index(i);
                if(column+uint(ij[0])<N && row+uint(ij[1])<M)destination[ulong(ij[1])*N+column+ij[0]]=bfloat(sum[i]);
            }
        } else {
            // The host elects this specialization only for complete 32x128 tiles.
            auto converted=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),bfloat>();
            #pragma unroll
            for(ushort i=0;i<sum.get_capacity();++i)converted[i]=bfloat(sum[i]);
            auto target=tensor(destination,dextents<int,2>{int(N),BM},array<int,2>{1,int(N)});
            converted.store(target.template slice<BN,BM>(column,0));
        }
    } else if(Store && column+BN<=N && row+BM<=M) {
        device float* destination=parts==1 && !PairPartials?out+ulong(row+p[5])*N:
            input_sums+p[4]*ng+partial_offset+ulong(grid.z)*M*N+ulong(row)*N;
        if(parts==1 && !PairPartials) {
            auto cast=[&](ushort i){sum[i]=float(bfloat(sum[i]));};
            splash_visit(sum,traversal,cast);
        }
        auto target=tensor(destination,dextents<int,2>{int(N),int(min(uint(BM),M-row))},array<int,2>{1,int(N)});
        sum.store(target.template slice<BN,BM>(column,0));
    } else {
    #pragma unroll
    for(ushort i=0;i<sum.get_capacity();++i)if(sum.is_valid_element(i)) {
        auto ij=sum.get_multidimensional_index(i);
        if(column+uint(ij[0])<N && row+uint(ij[1])<M) {
            if(parts==1 && !PairPartials)out[ulong(row+ij[1]+p[5])*N+column+ij[0]]=float(bfloat(sum[i]));
            else input_sums[p[4]*ng+partial_offset+ulong(grid.z)*M*N+ulong(row+ij[1])*N+column+ij[0]]=sum[i];
        }
    }
    }
}
#define SPLASH_AFFINE(NAME,ROWS,SIMDS,PIPE) \
kernel void NAME(device uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]], \
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint3 grid [[threadgroup_position_in_grid]]) { \
    splash_affine_tile<ROWS,SIMDS,PIPE>(w,x,out,p,grid); \
}
SPLASH_AFFINE(splash_affine8_compact,8,8,false)
SPLASH_AFFINE(splash_affine8_compactpipe,8,8,true)
SPLASH_AFFINE(splash_affine16_compact4,16,4,false)
SPLASH_AFFINE(splash_affine24_compact4,24,4,false)
SPLASH_AFFINE(splash_affine32_compact,32,8,false)
SPLASH_AFFINE(splash_affine32_compact4,32,4,false)
SPLASH_AFFINE(splash_affine64_compact,64,8,false)
#undef SPLASH_AFFINE
kernel void splash_affine32_bf16(device uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint3 grid [[threadgroup_position_in_grid]]) {
    splash_affine_tile<32,4,false,false,true>(w,x,out,p,grid);
}
kernel void splash_pair32_bf16(device uchar* gate [[buffer(0)]],device uchar* up [[buffer(1)]],device bfloat* x [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint3 grid [[threadgroup_position_in_grid]]) {
    uint plane=grid.z;grid.z=0;
    splash_affine_tile<32,4,false,true,true>(plane?up:gate,x,out,p,grid,ulong(plane)*p[2]*p[1]);
}
kernel void splash_affine32_bf16_tail(device uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint3 grid [[threadgroup_position_in_grid]]) {
    grid.y=p[2]/32;
    splash_affine_tile<32,4,false,false,true,true>(w,x,out,p,grid);
}
kernel void splash_pair32_bf16_tail(device uchar* gate [[buffer(0)]],device uchar* up [[buffer(1)]],device bfloat* x [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint3 grid [[threadgroup_position_in_grid]]) {
    uint plane=grid.z;grid.z=0;grid.y=p[2]/32;
    splash_affine_tile<32,4,false,true,true,true>(plane?up:gate,x,out,p,grid,ulong(plane)*p[2]*p[1]);
}
kernel void splash_widen(device bfloat* scratch [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1]*p[2])return;
    device bfloat* value=scratch+p[4]*p[0]+2*p[4]*(p[0]/64);
    out[ulong(p[5])*p[1]+i]=float(value[i]);
}
#define SPLASH_PAIR(NAME,ROWS,SIMDS,PIPE) \
kernel void NAME(device uchar* gate [[buffer(0)]],device uchar* up [[buffer(1)]],device bfloat* x [[buffer(2)]], \
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint3 grid [[threadgroup_position_in_grid]]) { \
    uint plane=grid.z/p[3];grid.z%=p[3]; \
    splash_affine_tile<ROWS,SIMDS,PIPE,true>(plane?up:gate,x,out,p,grid,ulong(plane)*p[3]*p[2]*p[1]); \
}
SPLASH_PAIR(splash_pair8,8,8,true)
SPLASH_PAIR(splash_pair16,16,4,false)
SPLASH_PAIR(splash_pair24,24,4,false)
SPLASH_PAIR(splash_pair32,32,4,false)
SPLASH_PAIR(splash_pair64,64,8,false)
#undef SPLASH_PAIR
// Small gate vectors cannot fill an MPP tile. Each SIMD owns one output;
// partitions expose independent quant groups without expanding the weights.
kernel void splash_affine_vector(device uchar* weights [[buffer(0)]],device bfloat* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint3 grid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint K=p[0],N=p[1],M=p[2],parts=p[3],ng=K/64,np=((N+255)/256)*256;
    uint col=grid.x*4+sg%4,row=grid.y*2+sg/4;
    if(col>=N || row>=M)return;
    device bfloat* scales=reinterpret_cast<device bfloat*>(weights+ulong(K)*np/2);
    device bfloat* biases=scales+ulong(K)*np/64;
    device float* sums=reinterpret_cast<device float*>(x+p[4]*K);
    float total=0;
    uint first=grid.z*(ng/parts),end=first+ng/parts;
    for(uint g=first;g<end;++g) {
        ulong index=(ulong(col/256)*ng+g)*256+col%256;
        uchar code=weights[index*32+lane];
        float a=float(x[ulong(row)*K+g*64+lane*2]);
        float b=float(x[ulong(row)*K+g*64+lane*2+1]);
        float dot=simd_sum(a*float(code&15)+b*float(code>>4));
        total+=dot*float(scales[index])+sums[g*p[4]+row]*float(biases[index]);
    }
    if(lane==0) {
        if(parts==1)out[ulong(row+p[5])*N+col]=float(bfloat(total));
        else sums[p[4]*ng+ulong(grid.z)*M*N+ulong(row)*N+col]=total;
    }
}
kernel void splash_reduce(device bfloat* scratch [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint K=p[0],N=p[1],M=p[2],parts=p[3];
    if(i>=M*N)return;
    uint padded=p[4];
    device float* partial=reinterpret_cast<device float*>(scratch+padded*K)+padded*(K/64);
    float v=0;
    for(uint part=0;part<parts;++part)v+=partial[ulong(part)*M*N+i];
    out[ulong(p[5])*N+i]=float(bfloat(v));
}
