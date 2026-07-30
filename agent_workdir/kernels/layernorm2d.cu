#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}

// Each thread owns one (n,h,w) and loops over C (typically 32..256).
template <typename T>
__global__ void
layernorm2d_kernel(T *__restrict__ out, const T *__restrict__ in,
                   const T *__restrict__ weight, const T *__restrict__ bias,
                   int N, int C, int H, int W, float eps) {
  const int spatial = H * W;
  const int total = N * spatial;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const int n = idx / spatial;
    const int hw = idx - n * spatial;
    const T *in_base = in + n * C * spatial + hw;
    T *out_base = out + n * C * spatial + hw;

    float sum = 0.f;
#pragma unroll 4
    for (int c = 0; c < C; ++c) {
      sum += to_float(in_base[c * spatial]);
    }
    float mean = sum / (float)C;

    float var_sum = 0.f;
#pragma unroll 4
    for (int c = 0; c < C; ++c) {
      float d = to_float(in_base[c * spatial]) - mean;
      var_sum += d * d;
    }
    float inv_std = rsqrtf(var_sum / (float)C + eps);

#pragma unroll 4
    for (int c = 0; c < C; ++c) {
      float x = to_float(in_base[c * spatial]);
      float y = (x - mean) * inv_std;
      y = y * to_float(weight[c]) + to_float(bias[c]);
      out_base[c * spatial] = from_float<T>(y);
    }
  }
}

// Vectorized path when spatial is multiple of 2 and we process float2-ish via
// half2 for fp16
template <typename T>
void layernorm2d_launch_typed(T *out, const T *in, const T *weight,
                              const T *bias, int N, int C, int H, int W,
                              float eps, int config, cudaStream_t stream) {
  int total = N * H * W;
  if (total <= 0)
    return;
  int threads = 256;
  if (config == 1)
    threads = 128;
  if (config == 2)
    threads = 512;
  int blocks = (total + threads - 1) / threads;
  // Cap grid size for occupancy
  if (blocks > 2048)
    blocks = 2048;
  layernorm2d_kernel<T>
      <<<blocks, threads, 0, stream>>>(out, in, weight, bias, N, C, H, W, eps);
}

extern "C" void layernorm2d_launcher(void *out, const void *in,
                                     const void *weight, const void *bias,
                                     int N, int C, int H, int W, float eps,
                                     int dtype, int config,
                                     cudaStream_t stream) {
  if (dtype == 1) {
    layernorm2d_launch_typed<__half>(
        (__half *)out, (const __half *)in, (const __half *)weight,
        (const __half *)bias, N, C, H, W, eps, config, stream);
  } else {
    layernorm2d_launch_typed<float>((float *)out, (const float *)in,
                                    (const float *)weight, (const float *)bias,
                                    N, C, H, W, eps, config, stream);
  }
}
