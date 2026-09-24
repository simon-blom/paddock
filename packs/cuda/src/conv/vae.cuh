// conv/vae.cuh - the image VAE's convolution glue (Qwen-Image-2.1's Wan-shaped
// residual autoencoder first). Textually-included segment of the single pack
// translation unit. Not standalone-compilable: include order is defined by
// ../pack.cu.
//
// Planes are NHWC, [pixels][channels], f32 for the residual stream and f16
// for whatever the next 3x3 conv reads: every conv is an im2row into f16
// staging plus the in-house f16 tensor-core GEMM (the DINOv3 decoder's
// recipe, dense_pred.cuh), with the row work chunked into STRIPES of image
// rows so the 9x staging plane is bounded by a stripe, not the image. That
// is the interim documented there too; the implicit-GEMM form (the taps
// gathered inside the GEMM's own A-tile staging) is the target if the
// decoder's share of a generation ever matters - at 40 DiT steps per image
// it does not.
//
// What the Wan-2.2 residual VAE needs that the DINOv3 lane does not:
// a per-PIXEL channel RMSNorm (F.normalize over C times sqrt(C) times gamma)
// with the SiLU that follows it, a 3x3 im2row that reads its source through a
// nearest-exact 2x upsample (the Resample block never materialises the
// upsampled plane), and the DupUp shortcut - channel-repeat plus spatial
// duplication, added to the up block's output.

// ---- per-pixel channel RMSNorm (+ SiLU) -> f16 ------------------------------
//
// diffusers `QwenImage21RMS_norm(images=False)`: y = normalize(x, dim=C) *
// sqrt(C) * gamma[c]  (normalize = x / max(||x||_2, 1e-12)), in f32 whatever
// the plane dtype. `act` = 1 applies SiLU after it (every ResidualBlock norm;
// the decoder's norm_out); 0 leaves it (the attention block's norm). One block
// per pixel row, the sum of squares folded in a fixed tree.
__global__ void __launch_bounds__(256) pd_vae_norm_kernel(const float* __restrict__ x,
                                                          const float* __restrict__ gamma,
                                                          __half* __restrict__ out, uint32_t C,
                                                          float scale, uint32_t act) {
    __shared__ float red[256];
    const float* row = x + (size_t)blockIdx.x * C;
    __half* orow = out + (size_t)blockIdx.x * C;
    const uint32_t tid = threadIdx.x;
    float ss = 0.0f;
    for (uint32_t c = tid; c < C; c += 256u) ss = fmaf(row[c], row[c], ss);
    red[tid] = ss;
    __syncthreads();
    for (uint32_t s = 128u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float nrm = fmaxf(sqrtf(red[0]), 1e-12f);
    for (uint32_t c = tid; c < C; c += 256u) {
        float v = row[c] / nrm * scale * gamma[c];
        if (act) v = v / (1.0f + expf(-v));
        orow[c] = __float2half(v);
    }
}

PD_EXPORT
int pd_vae_norm_f16(const void* x, const void* gamma, void* out, uint32_t rows, uint32_t C,
                    uint32_t act, void* stream) {
    if (rows == 0u || C == 0u) return 0;
    pd_vae_norm_kernel<<<rows, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)gamma, (__half*)out, C, sqrtf((float)C), act);
    return pd_launch_status();
}

// ---- 3x3 / stride 1 / zero-pad 1 im2row, one stripe of output rows ----------
//
// Source is an f16 NHWC [H][W][C] plane. With `up2` = 1 the conv's logical
// input is the 2H x 2W nearest-exact upsample of it (torch `nearest-exact`
// at an integer factor of 2 is exactly floor(y / 2), floor(x / 2)); the
// output grid is then 2H x 2W and each tap reads the source pixel under it.
// Output rows y0 .. y0 + ny of that grid, every column, land as
// [ny * W_out][9 * C] f16 with TAP-outer columns (col = (ky*3+kx)*C + c) -
// the DINOv3 order, so the loader permutes [out][in][ky][kx] weights the
// same way. Out-of-range taps are zeros, computed, like torch's padding=1.
__global__ void pd_vae_im2row3_kernel(const __half* __restrict__ src, __half* __restrict__ out,
                                      uint32_t H, uint32_t W, uint32_t C, uint32_t y0,
                                      uint32_t up2) {
    const uint32_t W_out = up2 ? 2u * W : W, H_out = up2 ? 2u * H : H;
    const uint32_t o = blockIdx.x;                    // stripe-local pixel
    const uint32_t k = 9u * C;
    const int32_t y = (int32_t)(y0 + o / W_out), x = (int32_t)(o - (o / W_out) * W_out);
    __half* orow = out + (size_t)o * k;
    for (uint32_t col = threadIdx.x; col < k; col += blockDim.x) {
        const uint32_t t = col / C, c = col - t * C;
        const int32_t yy = y + (int32_t)(t / 3u) - 1, xx = x + (int32_t)(t % 3u) - 1;
        const bool in = yy >= 0 && yy < (int32_t)H_out && xx >= 0 && xx < (int32_t)W_out;
        const uint32_t sy = up2 ? ((uint32_t)yy >> 1) : (uint32_t)yy;
        const uint32_t sx = up2 ? ((uint32_t)xx >> 1) : (uint32_t)xx;
        const size_t a = in ? ((size_t)(sy * W + sx) * C + c) : (size_t)c;
        orow[col] = in ? src[a] : __float2half(0.0f);
    }
}

PD_EXPORT
int pd_vae_im2row3(const void* src, void* out, uint32_t H, uint32_t W, uint32_t C,
                   uint32_t y0, uint32_t ny, uint32_t up2, void* stream) {
    if (H == 0u || W == 0u || C == 0u || ny == 0u) return 0;
    const uint32_t H_out = up2 ? 2u * H : H, W_out = up2 ? 2u * W : W;
    if (y0 + ny > H_out) return cudaErrorInvalidValue;
    const uint32_t k = 9u * C;
    const uint32_t nth = k < 256u ? ((k + 31u) & ~31u) : 256u;
    pd_vae_im2row3_kernel<<<ny * W_out, nth, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)out, H, W, C, y0, up2 ? 1u : 0u);
    return pd_launch_status();
}

// ---- 3x3 / stride 2 / pad (0,1,0,1) im2row: the encoder's downsampler -------
//
// diffusers `Resample("downsample2d" / "downsample3d")` on one frame:
// ZeroPad2d((0, 1, 0, 1)) then Conv2d(3, stride 2, padding 0) - one zero
// column on the right and one zero row at the bottom, nothing on the top or
// left. The output grid is H/2 x W/2; output pixel (y, x), tap (ky, kx)
// reads source (2y + ky, 2x + kx), zero where that is the padded row H or
// column W. The staging layout is the stride-1 kernel's (tap-outer), so the
// loader's weight permutation serves both.
__global__ void pd_vae_im2row3_down_kernel(const __half* __restrict__ src,
                                           __half* __restrict__ out, uint32_t H, uint32_t W,
                                           uint32_t C, uint32_t y0) {
    const uint32_t W_out = W >> 1;
    const uint32_t o = blockIdx.x;                    // stripe-local pixel
    const uint32_t k = 9u * C;
    const uint32_t y = y0 + o / W_out, x = o - (o / W_out) * W_out;
    __half* orow = out + (size_t)o * k;
    for (uint32_t col = threadIdx.x; col < k; col += blockDim.x) {
        const uint32_t t = col / C, c = col - t * C;
        const uint32_t yy = 2u * y + t / 3u, xx = 2u * x + t % 3u;
        const bool in = yy < H && xx < W;
        const size_t a = in ? (((size_t)yy * W + xx) * C + c) : (size_t)c;
        orow[col] = in ? src[a] : __float2half(0.0f);
    }
}

PD_EXPORT
int pd_vae_im2row3_down(const void* src, void* out, uint32_t H, uint32_t W, uint32_t C,
                        uint32_t y0, uint32_t ny, void* stream) {
    if (H == 0u || W == 0u || C == 0u || ny == 0u) return 0;
    // the encoder only meets even planes (a multiple of 32 halved four times)
    if ((H | W) & 1u) return cudaErrorInvalidValue;
    const uint32_t H_out = H >> 1, W_out = W >> 1;
    if (y0 + ny > H_out) return cudaErrorInvalidValue;
    const uint32_t k = 9u * C;
    const uint32_t nth = k < 256u ? ((k + 31u) & ~31u) : 256u;
    pd_vae_im2row3_down_kernel<<<ny * W_out, nth, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)out, H, W, C, y0);
    return pd_launch_status();
}

// ---- AvgDown shortcut add (the encoder's residual down blocks) -------------
//
// diffusers `AvgDown3D` on one frame: the frame axis is zero-padded at the
// FRONT to `ft` frames, the tensor is read as (c, t, hy, wx) over each
// (ft, fs, fs) block - c outermost - and flattened to C_in * ft * fs * fs
// channels at (H/fs, W/fs); then every `group = C_in * ft * fs * fs / C_out`
// consecutive channels are averaged into one output channel. With ft 2 the
// front frame is the zero one, so output channel 2c is a mean of zeros and
// 2c + 1 the spatial mean of input channel c - the checkpoint's own mapping,
// computed rather than special-cased. Added into the block's f32 output
// [H/fs][W/fs][C_out].
__global__ void pd_vae_avgdown_add_kernel(float* __restrict__ out, const float* __restrict__ in,
                                          uint32_t H, uint32_t W, uint32_t C_in, uint32_t C_out,
                                          uint32_t ft, uint32_t fs, uint32_t group,
                                          size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t W_out = W / fs;
    const size_t pix = i / C_out;
    const uint32_t o = (uint32_t)(i - pix * C_out);
    const uint32_t y = (uint32_t)(pix / W_out), x = (uint32_t)(pix - (size_t)y * W_out);
    float acc = 0.0f;
    for (uint32_t g = 0; g < group; ++g) {
        uint32_t idx = o * group + g;
        const uint32_t wx = idx % fs;
        idx /= fs;
        const uint32_t hy = idx % fs;
        idx /= fs;
        const uint32_t t = idx % ft;
        const uint32_t c = idx / ft;
        // the real frame is the last one; the padded ones in front are zero
        if (t + 1u == ft) {
            acc += in[((size_t)(y * fs + hy) * W + (x * fs + wx)) * C_in + c];
        }
    }
    out[i] += acc / (float)group;
}

PD_EXPORT
int pd_vae_avgdown_add(void* out, const void* in, uint32_t H, uint32_t W, uint32_t C_in,
                       uint32_t C_out, uint32_t ft, uint32_t fs, void* stream) {
    if (ft == 0u || fs == 0u || C_out == 0u || H % fs != 0u || W % fs != 0u) {
        return cudaErrorInvalidValue;
    }
    const uint32_t factor = C_in * ft * fs * fs;
    if (factor % C_out != 0u) return cudaErrorInvalidValue;
    const uint32_t group = factor / C_out;
    const size_t total = (size_t)(H / fs) * (W / fs) * C_out;
    if (total == 0u) return 0;
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_vae_avgdown_add_kernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>(
        (float*)out, (const float*)in, H, W, C_in, C_out, ft, fs, group, total);
    return pd_launch_status();
}

// ---- DupUp shortcut add -----------------------------------------------------
//
// diffusers `DupUp3D` on one frame, kept at the LAST temporal copy
// (`first_chunk` slices it): the input's channels are repeated
// `repeats` times, the repeated axis is read as (out_channel, ft, 2, 2), and
// the (2, 2) part becomes the spatial duplication. Output pixel (y, x),
// channel o therefore reads input pixel (y/2, x/2), channel
//   (((o * ft + (ft - 1)) * 2 + (y & 1)) * 2 + (x & 1)) / repeats.
// For repeats 8 that is channel o itself (plain nearest); for repeats 4 with
// ft 2 it is 2o + 1; for repeats 2 with ft 1 it is 2o + (y & 1) - the mapping
// is the checkpoint's, so it is computed, not special-cased. Added into the
// up block's f32 output [2H][2W][C_out].
__global__ void pd_vae_dupup_add_kernel(float* __restrict__ out, const float* __restrict__ in,
                                        uint32_t H, uint32_t W, uint32_t C_in, uint32_t C_out,
                                        uint32_t ft, uint32_t repeats, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t W2 = 2u * W;
    const size_t pix = i / C_out;
    const uint32_t o = (uint32_t)(i - pix * C_out);
    const uint32_t y = (uint32_t)(pix / W2), x = (uint32_t)(pix - (size_t)y * W2);
    const uint32_t j = ((o * ft + (ft - 1u)) * 2u + (y & 1u)) * 2u + (x & 1u);
    const uint32_t ci = j / repeats;
    out[i] += in[((size_t)(y >> 1) * W + (x >> 1)) * C_in + ci];
}

PD_EXPORT
int pd_vae_dupup_add(void* out, const void* in, uint32_t H, uint32_t W, uint32_t C_in,
                     uint32_t C_out, uint32_t ft, uint32_t repeats, void* stream) {
    const size_t total = (size_t)4u * H * W * C_out;
    if (total == 0u) return 0;
    if (ft == 0u || repeats == 0u || C_out * ft * 4u != C_in * repeats) return cudaErrorInvalidValue;
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_vae_dupup_add_kernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>(
        (float*)out, (const float*)in, H, W, C_in, C_out, ft, repeats, total);
    return pd_launch_status();
}
