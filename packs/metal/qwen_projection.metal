// Qwen decode column coarsening. Extend our Gemma paired-column design to
// IQ4_XS and the distinct R1 scale-factored arithmetic. Each column retains
// the established K order, F32 operands/accumulators and segmented reduction.
// No weight expansion, activation rounding or cross-request approximation.
template<uint R,uint Type>
inline void qwen_pair(device const uchar* w,device const float* x,device float* out,
    uint K,uint N,float scale,uint n,uint lane) {
    if(n>=N)return;
    constexpr uint L=R==1?32:8,C=2;
    float4 total[C][R];for(uint c=0;c<C;++c)for(uint r=0;r<R;++r)total[c][r]=0;
    if constexpr(Type==12 || Type==13) {
        for(uint k=lane*32;k<K;k+=L*32) {
            device const uchar* blocks[C];float ds[C],ms[C];uint s=(k%256)/32;
            for(uint c=0;c<C;++c) {
                // Clamp the unused partner of an odd final column. Its result
                // is discarded, and neither loads nor stores cross the plane.
                blocks[c]=w+((ulong(min(n+c,N-1))*K+k)/256)*(Type==12?144:176);
                device const uchar* sc=blocks[c]+4;
                uint d=s<4?sc[s]&63:(sc[s+4]&15)|((sc[s-4]>>6)<<4);
                uint m=s<4?sc[s+4]&63:(sc[s+4]>>4)|((sc[s]>>6)<<4);
                ds[c]=float(*reinterpret_cast<device const half*>(blocks[c]))*float(d);
                ms[c]=float(*reinterpret_cast<device const half*>(blocks[c]+2))*float(m);
            }
            float4 dotq[C][R],sumx[R];
            for(uint r=0;r<R;++r){sumx[r]=0;for(uint c=0;c<C;++c)dotq[c][r]=0;}
            #pragma unroll
            for(uint j=0;j<8;++j) {
                float4 q[C];
                for(uint c=0;c<C;++c) {
                    uint packed=(*reinterpret_cast<device const uint*>(blocks[c]+(Type==12?16:48)+(s/2)*32+j*4)>>((s%2)*4))&0x0f0f0f0f;
                    if constexpr(Type==13)packed|=((*reinterpret_cast<device const uint*>(blocks[c]+16+j*4)>>s)&0x01010101)<<4;
                    q[c]=float4(as_type<uchar4>(packed));
                }
                for(uint r=0;r<R;++r) {
                    float4 v=*reinterpret_cast<device const float4*>(x+ulong(r)*K+k+j*4);
                    if constexpr(R==1) {
                        sumx[r]+=v;for(uint c=0;c<C;++c)dotq[c][r]=fma(q[c],v,dotq[c][r]);
                    } else for(uint c=0;c<C;++c)total[c][r]=fma(q[c]*ds[c]-ms[c],v,total[c][r]);
                }
            }
            if constexpr(R==1)for(uint c=0;c<C;++c)for(uint r=0;r<R;++r)
                total[c][r]+=dotq[c][r]*ds[c]-sumx[r]*ms[c];
        }
    } else {
        constexpr uint V=Type==14?16:32;
        for(uint k=lane*V;k<K;k+=L*V) {
            device const uchar* blocks[C];float ds[C];uint ix=k%256,s=ix/32;
            for(uint c=0;c<C;++c) {
                blocks[c]=w+((ulong(min(n+c,N-1))*K+k)/256)*(Type==14?210:136);
                if constexpr(Type==14)ds[c]=float(*reinterpret_cast<device const half*>(blocks[c]+208))*float(reinterpret_cast<device const char*>(blocks[c]+192)[ix/16]);
                else {
                    uint hi=uint(*reinterpret_cast<device const ushort*>(blocks[c]+2));
                    int qs=int(((blocks[c][4+s/2]>>((s%2)*4))&15)|(((hi>>(2*s))&3)<<4))-32;
                    ds[c]=float(*reinterpret_cast<device const half*>(blocks[c]))*float(qs);
                }
            }
            float4 partial[C][R];for(uint c=0;c<C;++c)for(uint r=0;r<R;++r)partial[c][r]=0;
            #pragma unroll
            for(uint j=0;j<V/4;++j) {
                uint index=ix+j*4;float4 value[C];
                for(uint c=0;c<C;++c) {
                    if constexpr(Type==14) {
                        uint part=index/128,r=index%128;
                        uint lo=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(blocks[c]+part*64+r%64));
                        uint hi=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(blocks[c]+128+part*32+r%32));
                        uint code=((lo>>((r/64)*4))&0x0f0f0f0f)|(((hi>>((r/32)*2))&0x03030303)<<4);
                        value[c]=float4(as_type<uchar4>(code))-32.0f;
                    } else {
                        uint4 code=(uint4(*reinterpret_cast<device const packed_uchar4*>(blocks[c]+8+s*16+index%16))>>((index%32/16)*4))&15;
                        value[c]=float4(iq4_values[code.x],iq4_values[code.y],iq4_values[code.z],iq4_values[code.w]);
                    }
                }
                for(uint r=0;r<R;++r) {
                    float4 v=*reinterpret_cast<device const float4*>(x+ulong(r)*K+k+j*4);
                    for(uint c=0;c<C;++c)partial[c][r]=fma(value[c],v,partial[c][r]);
                }
            }
            for(uint c=0;c<C;++c)for(uint r=0;r<R;++r)total[c][r]=fma(partial[c][r],ds[c],total[c][r]);
        }
    }
    for(uint c=0;c<C;++c)for(uint r=0;r<R;++r) {
        float4 t=total[c][r];float v=kquant_sum<L>(t.x+t.y+t.z+t.w);
        if(lane==0 && n+c<N)out[ulong(r)*N+n+c]=v*scale;
    }
}
template<uint R>
inline void qwen_pair_dispatch(device const uchar* w,device const float* x,device float* out,
    uint K,uint N,uint type,float scale,uint n,uint lane) {
    switch(type) {
    case 12:qwen_pair<R,12>(w,x,out,K,N,scale,n,lane);break;
    case 13:qwen_pair<R,13>(w,x,out,K,N,scale,n,lane);break;
    case 14:qwen_pair<R,14>(w,x,out,K,N,scale,n,lane);break;
    case 23:qwen_pair<R,23>(w,x,out,K,N,scale,n,lane);break;}
}
#define QWEN_PAIR(R) \
kernel void qwen_pair##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
constexpr uint L=R==1?32:8; \
qwen_pair_dispatch<R>(w,x,out,p[0],p[1],p[3],as_type<float>(p[4]),g*(256/L)+(tid/L)*2,tid%L);}
QWEN_PAIR(1)
QWEN_PAIR(2)
QWEN_PAIR(3)
QWEN_PAIR(4)
#undef QWEN_PAIR
#define QWEN_MULTI_PAIR(R) \
kernel void qwen_multi_pair##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
constexpr uint L=R==1?32:8,C=256/L; \
uint n0=(p[1]+C-1)/C,n1=(p[2]+C-1)/C,N=p[1],ty=p[5];device const uchar* w=w0;device float* out=o0; \
if(g>=n0){g-=n0;N=p[2];ty=p[6];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];ty=p[7];w=w2;out=o2;}} \
qwen_pair_dispatch<R>(w,x,out,p[0],N,ty,1.0f,g*C+(tid/L)*2,tid%L);}
QWEN_MULTI_PAIR(1)
QWEN_MULTI_PAIR(2)
QWEN_MULTI_PAIR(3)
QWEN_MULTI_PAIR(4)
#undef QWEN_MULTI_PAIR
