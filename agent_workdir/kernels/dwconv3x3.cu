#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Depthwise 3x3, stride=1, pad=1, FP16. Shared-memory tiled.

template <int BH, int BW>
__global__ void dwconv3x3_tiled_kernel(__half *__restrict__ out,
                                       const __half *__restrict__ in,
                                       const __half *__restrict__ weight,
                                       const __half *__restrict__ bias, int N,
                                       int C, int H, int W) {
  const int n = blockIdx.z / C;
  const int c = blockIdx.z - n * C;
  if (n >= N)
    return;

  const int tile_h0 = (int)blockIdx.y * BH;
  const int tile_w0 = (int)blockIdx.x * BW;
  const int tid = (int)threadIdx.y * BW + (int)threadIdx.x;

  constexpr int SH = BH + 2;
  constexpr int SW = BW + 2;
  __shared__ __half smem[SH * SW];
  __shared__ float wgt_s[9];
  __shared__ float bias_s;

  if (tid < 9)
    wgt_s[tid] = __half2float(weight[c * 9 + tid]);
  if (tid == 0)
    bias_s = bias ? __half2float(bias[c]) : 0.f;

  const __half *src = in + (n * C + c) * H * W;
  __half *dst = out + (n * C + c) * H * W;

  const int smem_elems = SH * SW;
  for (int i = tid; i < smem_elems; i += BH * BW) {
    int sh = i / SW;
    int sw = i - sh * SW;
    int ih = tile_h0 + sh - 1;
    int iw = tile_w0 + sw - 1;
    float v = 0.f;
    if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
      v = __half2float(src[ih * W + iw]);
    }
    smem[i] = __float2half(v);
  }
  __syncthreads();

  const int oh = tile_h0 + (int)threadIdx.y;
  const int ow = tile_w0 + (int)threadIdx.x;
  if (oh < H && ow < W) {
    float acc = bias_s;
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        acc +=
            __half2float(
                smem[((int)threadIdx.y + kh) * SW + ((int)threadIdx.x + kw)]) *
            wgt_s[kh * 3 + kw];
      }
    }
    dst[oh * W + ow] = __float2half(acc);
  }
}

extern "C" void dwconv3x3_launcher(void *out, const void *in,
                                   const void *weight, const void *bias, int N,
                                   int C, int H, int W, int config,
                                   cudaStream_t stream) {
  if (config == 1) {
    constexpr int BH = 8, BW = 32;
    dim3 block(BW, BH);
    dim3 grid((W + BW - 1) / BW, (H + BH - 1) / BH, N * C);
    dwconv3x3_tiled_kernel<BH, BW><<<grid, block, 0, stream>>>(
        (__half *)out, (const __half *)in, (const __half *)weight,
        (const __half *)bias, N, C, H, W);
    return;
  }
  constexpr int BH = 16, BW = 16;
  dim3 block(BW, BH);
  dim3 grid((W + BW - 1) / BW, (H + BH - 1) / BH, N * C);
  dwconv3x3_tiled_kernel<BH, BW><<<grid, block, 0, stream>>>(
      (__half *)out, (const __half *)in, (const __half *)weight,
      (const __half *)bias, N, C, H, W);
}
