#include <cuda_fp16.h>
#include <cuda_runtime.h>

// NHWC depthwise 3x3 grid-stride over N*H*W*C (coalesced along C).
__global__ void dwconv3x3_nhwc_kernel(__half *__restrict__ out,
                                      const __half *__restrict__ in,
                                      const __half *__restrict__ weight,
                                      const __half *__restrict__ bias, int N,
                                      int H, int W, int C) {
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int tmp = idx / C;
    int w = tmp % W;
    tmp /= W;
    int h = tmp % H;
    int n = tmp / H;

    float acc = bias ? __half2float(bias[c]) : 0.f;
    const __half *wgt = weight + c * 9;
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        int ih = h + kh - 1;
        int iw = w + kw - 1;
        float v = 0.f;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
          v = __half2float(in[((n * H + ih) * W + iw) * C + c]);
        }
        acc += v * __half2float(wgt[kh * 3 + kw]);
      }
    }
    out[idx] = __float2half(acc);
  }
}

__device__ __forceinline__ float dw_at(const __half *in, const __half *wgt,
                                       const __half *bias, int n, int h, int w,
                                       int c, int H, int W, int Ctot) {
  float acc = bias ? __half2float(bias[c]) : 0.f;
#pragma unroll
  for (int kh = 0; kh < 3; ++kh) {
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      int ih = h + kh - 1;
      int iw = w + kw - 1;
      float v = 0.f;
      if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
        v = __half2float(in[((n * H + ih) * W + iw) * Ctot + c]);
      }
      acc += v * __half2float(wgt[kh * 3 + kw]);
    }
  }
  return acc;
}

// Fused depthwise 3x3 on 2C channels + SimpleGate -> C channels.
__global__ void dwconv3x3_gate_nhwc_kernel(__half *__restrict__ out,
                                           const __half *__restrict__ in,
                                           const __half *__restrict__ weight,
                                           const __half *__restrict__ bias,
                                           int N, int H, int W, int C) {
  int C2 = C * 2;
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int tmp = idx / C;
    int w = tmp % W;
    tmp /= W;
    int h = tmp % H;
    int n = tmp / H;
    float a = dw_at(in, weight + c * 9, bias, n, h, w, c, H, W, C2);
    float b = dw_at(in, weight + (c + C) * 9, bias, n, h, w, c + C, H, W, C2);
    out[idx] = __float2half(a * b);
  }
}

extern "C" void dwconv3x3_launcher(void *out, const void *in,
                                   const void *weight, const void *bias, int N,
                                   int H, int W, int C, int /*config*/,
                                   cudaStream_t stream) {
  int total = N * H * W * C;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 4096)
    blocks = 4096;
  dwconv3x3_nhwc_kernel<<<blocks, threads, 0, stream>>>(
      (__half *)out, (const __half *)in, (const __half *)weight,
      (const __half *)bias, N, H, W, C);
}

extern "C" void dwconv3x3_gate_launcher(void *out, const void *in,
                                        const void *weight, const void *bias,
                                        int N, int H, int W, int C,
                                        int /*config*/, cudaStream_t stream) {
  int total = N * H * W * C;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 4096)
    blocks = 4096;
  dwconv3x3_gate_nhwc_kernel<<<blocks, threads, 0, stream>>>(
      (__half *)out, (const __half *)in, (const __half *)weight,
      (const __half *)bias, N, H, W, C);
}
