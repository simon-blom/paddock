// Native hybrid speculative verification. The committed FP32 recurrent state
// is read-only until acceptance. Save rank-one updates, not one 128x128 state
// per token: 257 floats/head/row rather than 16384. Commit repeats the exact
// multiply/add order over only the accepted prefix. No target replay.
kernel void dn_verify(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
                      device const float* state [[buffer(2)]],device const uint* spans [[buffer(3)]],
                      device const uint* meta [[buffer(4)]],device float* out [[buffer(5)]],
                      device float* updates [[buffer(6)]],constant uint* p [[buffer(7)]],
                      uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint first=spans[group.z*4],count=spans[group.z*4+1],slot=spans[group.z*4+2];
    uint h=group.y,v=group.x*16+tid/8,lane=tid%8,kh=h%p[0];
    ulong offset=((ulong(p[5])*p[4]+slot)*p[1]+h)*128*128+v*128;
    float st[16];
    for(uint j=0;j<16;++j)st[j]=meta[first*2+1]==0 ? 0.0f : state[offset+lane+j*8];
    for(uint r=first;r<first+count;++r) {
        float g=exp(gates[(r*p[1]+h)*2]),b=gates[(r*p[1]+h)*2+1],predicted=0;
        for(uint j=0;j<16;++j) {st[j]*=g;predicted+=st[j]*qkv[ulong(r)*p[2]+p[0]*128+kh*128+lane+j*8];}
        predicted+=simd_shuffle_xor(predicted,1);predicted+=simd_shuffle_xor(predicted,2);predicted+=simd_shuffle_xor(predicted,4);
        float delta=b*(qkv[ulong(r)*p[2]+p[0]*256+h*128+v]-predicted),y=0;
        for(uint j=0;j<16;++j) {
            uint d=lane+j*8;
            st[j]+=qkv[ulong(r)*p[2]+p[0]*128+kh*128+d]*delta;
            y+=qkv[ulong(r)*p[2]+kh*128+d]*st[j];
        }
        y+=simd_shuffle_xor(y,1);y+=simd_shuffle_xor(y,2);y+=simd_shuffle_xor(y,4);
        ulong u=((ulong(p[5])*p[6]+r)*p[1]+h)*257;
        if(lane==0) {out[(ulong(r)*p[1]+h)*128+v]=y;updates[u+128+v]=delta;}
        if(group.x==0)updates[u+tid]=qkv[ulong(r)*p[2]+p[0]*128+kh*128+tid];
        if(group.x==0 && tid==0)updates[u+256]=g;
    }
}

// p: heads, state_slots, max_rows. spans: first, accepted count, slot, unused.
kernel void dn_verify_commit(device const float* updates [[buffer(0)]],device float* state [[buffer(1)]],
                             device const uint* spans [[buffer(2)]],constant uint* p [[buffer(3)]],
                             uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint first=spans[group.z*4],count=spans[group.z*4+1],slot=spans[group.z*4+2];
    uint layer=group.y/p[0],h=group.y%p[0],v=group.x*16+tid/8,lane=tid%8;
    ulong offset=((ulong(layer)*p[1]+slot)*p[0]+h)*128*128+v*128;
    float st[16];for(uint j=0;j<16;++j)st[j]=state[offset+lane+j*8];
    for(uint r=first;r<first+count;++r) {
        ulong u=((ulong(layer)*p[2]+r)*p[0]+h)*257;
        float g=updates[u+256],delta=updates[u+128+v];
        for(uint j=0;j<16;++j) {st[j]*=g;st[j]+=updates[u+lane+j*8]*delta;}
    }
    for(uint j=0;j<16;++j)state[offset+lane+j*8]=st[j];
}

kernel void spec_copy(device const float* src [[buffer(0)]],device float* dst [[buffer(1)]],
                      constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[2])dst[ulong(p[1])+i]=src[ulong(p[0])+i];
}
kernel void spec_copy_words(device const uint* src [[buffer(0)]],device uint* dst [[buffer(1)]],
                            constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[2])dst[ulong(p[1])+i]=src[ulong(p[0])+i];
}

// One launch for all layers/requests, reading the captured convolution inputs
// directly. No 48-layer host loop and no scratch round-trip during commit.
kernel void dn_verify_conv_commit(device const float* input [[buffer(0)]],device float* history [[buffer(1)]],
                                  device const uint* spans [[buffer(2)]],constant uint* p [[buffer(3)]],
                                  uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint d=g.x*256+tid;if(d>=p[0])return;
    uint first=spans[g.z*4],count=spans[g.z*4+1],slot=spans[g.z*4+2];
    ulong dst=(ulong(g.y)*p[1]+slot)*p[0]*3+d;
    float window[3];for(uint j=0;j<3;++j) {
        int r=int(count)+int(j)-3;
        window[j]=r>=0 ? input[(ulong(g.y)*p[2]+first+uint(r))*p[0]+d] : history[dst+uint(r+3)*p[0]];
    }
    for(uint j=0;j<3;++j)history[dst+j*p[0]]=window[j];
}

// Form shifted target hidden rows for MTP catch-up. The first row in each
// ragged span reads the preceding committed hidden (zero at position zero).
// p: width, rows, mode (0 catch-up, 1 draft seed, 2 gather selected rows).
kernel void mtp_hidden(device const float* h [[buffer(0)]],device const float* pending [[buffer(1)]],
                       device const uint* meta [[buffer(2)]],device const uint* bounds [[buffer(3)]],
                       device float* out [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[0],d=i%p[0];if(row>=p[1])return;
    uint slot=meta[row*2],pos=meta[row*2+1];
    if(p[2]==2){out[i]=h[ulong(bounds[row])*p[0]+d];return;}
    if(p[2]==1){out[i]=pending[ulong(slot)*p[0]+d];return;}
    uint first=bounds[row*2];
    out[i]=row>first ? h[ulong(row-1)*p[0]+d] : (pos>0 ? pending[ulong(slot)*p[0]+d] : 0.0f);
}

kernel void mtp_concat(device const float* e [[buffer(0)]],device const float* h [[buffer(1)]],
                       device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;uint row=i/p[0],d=i%p[0];
    out[ulong(row)*p[0]*2+d]=e[i];out[ulong(row)*p[0]*2+p[0]+d]=h[i];
}

kernel void mtp_publish(device const float* h [[buffer(0)]],device float* pending [[buffer(1)]],
                        device const uint* meta [[buffer(2)]],device const uint* bounds [[buffer(3)]],
                        device const uint* checkpoints [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[0],d=i%p[0];if(row>=p[1])return;
    if(row+1==bounds[row*2+1])pending[ulong(meta[row*2])*p[0]+d]=h[i];
    if(checkpoints[row])pending[ulong(checkpoints[row]-1)*p[0]+d]=h[i];
}

kernel void mtp_advance(device const uint* ids [[buffer(0)]],device uint* meta [[buffer(1)]],
                        device uint* out [[buffer(2)]],device uint* mrope [[buffer(3)]],device uint* limits [[buffer(4)]],constant uint* p [[buffer(5)]],uint row [[thread_position_in_grid]]) {
    if(row>=p[0])return;out[row*p[2]+p[1]]=ids[row];meta[row*2+1]++;limits[row]++;
    for(uint a=0;a<4;++a)mrope[row*4+a]++;
}

// Explicit lowest-id tie break, including ties between SIMD groups.
kernel void spec_argmax(device const float* x [[buffer(0)]],device uint* ids [[buffer(1)]],
                        constant uint* p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float maxima[8];threadgroup uint indices[8];
    float best=-INFINITY;uint id=0xffffffffu;
    for(uint v=tid;v<p[0];v+=256) {float a=x[ulong(row)*p[0]+v];if(a>best || (a==best && v<id)){best=a;id=v;}}
    float mx=simd_max(best);uint ix=simd_min(best==mx ? id : 0xffffffffu);
    if(tid%32==0){maxima[tid/32]=mx;indices[tid/32]=ix;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){best=maxima[0];id=indices[0];for(uint j=1;j<8;++j)if(maxima[j]>best || (maxima[j]==best && indices[j]<id)){best=maxima[j];id=indices[j];}ids[row]=id;}
}
