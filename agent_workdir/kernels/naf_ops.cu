#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) {
  return __half2float(v);
}

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ __half from_f<__half>(float v) {
  return __float2half(v);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}

// SimpleGate: out[n,c,h,w] = in[n,c,h,w] * in[n,c+C,h,w], C = in_channels/2
template <typename T>
__global__ void simple_gate_kernel(T *__restrict__ out,
                                   const T *__restrict__ in, int N, int C,
                                   int HW) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * HW;
  int stride = blockDim.x * gridDim.x;
  for (; i < total; i += stride) {
    int hw = i % HW;
    int tmp = i / HW;
    int c = tmp % C;
    int n = tmp / C;
    int base = n * (2 * C) * HW;
    float a = to_f(in[base + c * HW + hw]);
    float b = to_f(in[base + (c + C) * HW + hw]);
    out[i] = from_f<T>(a * b);
  }
}

// out = a + b * scale_c  (scale broadcast over N,H,W; scale may be (1,C,1,1) or
// (C,))
template <typename T>
__global__ void residual_kernel(T *__restrict__ out, const T *__restrict__ a,
                                const T *__restrict__ b,
                                const T *__restrict__ scale, int N, int C,
                                int HW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * HW;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total; idx += stride) {
    int c = (idx / HW) % C;
    out[idx] = from_f<T>(to_f(a[idx]) + to_f(b[idx]) * to_f(scale[c]));
  }
}

// out = a + b
template <typename T>
__global__ void add_kernel(T *__restrict__ out, const T *__restrict__ a,
                           const T *__restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride)
    out[i] = from_f<T>(to_f(a[i]) + to_f(b[i]));
}

// out = a * b (same shape)
template <typename T>
__global__ void mul_kernel(T *__restrict__ out, const T *__restrict__ a,
                           const T *__restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride)
    out[i] = from_f<T>(to_f(a[i]) * to_f(b[i]));
}

// Stage1: global average pool -> (N, C)
template <typename T>
__global__ void gap_kernel(T *__restrict__ pooled, const T *__restrict__ in,
                           int N, int C, int HW) {
  int bc = blockIdx.x; // n * C + c
  if (bc >= N * C)
    return;
  int n = bc / C;
  int c = bc - n * C;
  const T *ptr = in + (n * C + c) * HW;

  float sum = 0.f;
  for (int i = threadIdx.x; i < HW; i += blockDim.x) {
    sum += to_f(ptr[i]);
  }
  sum = warp_sum(sum);
  __shared__ float sm[32];
  if ((threadIdx.x & 31) == 0)
    sm[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[threadIdx.x] : 0.f;
    v = warp_sum(v);
    if (threadIdx.x == 0)
      pooled[bc] = from_f<T>(v / (float)HW);
  }
}

template <typename T>
__global__ void sca_linear_kernel(T *__restrict__ scale,
                                  const T *__restrict__ pooled,
                                  const T *__restrict__ weight,
                                  const T *__restrict__ bias, int N, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C;
  if (idx >= total)
    return;
  int n = idx / C;
  int c = idx - n * C;
  const T *row = weight + c * C;
  const T *p = pooled + n * C;
  float acc = to_f(bias[c]);
  for (int k = 0; k < C; ++k)
    acc += to_f(row[k]) * to_f(p[k]);
  scale[idx] = from_f<T>(acc);
}

template <typename T>
__global__ void channel_scale_kernel(T *__restrict__ out,
                                     const T *__restrict__ in,
                                     const T *__restrict__ scale, // (N,C)
                                     int N, int C, int HW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * HW;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total; idx += stride) {
    int tmp = idx / HW;
    int c = tmp % C;
    int n = tmp / C;
    out[idx] = from_f<T>(to_f(in[idx]) * to_f(scale[n * C + c]));
  }
}

// out[n,c,h,w] += bias[c]
template <typename T>
__global__ void bias_add_kernel(T *__restrict__ out, const T *__restrict__ bias,
                                int N, int C, int HW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * HW;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total; idx += stride) {
    int c = (idx / HW) % C;
    out[idx] = from_f<T>(to_f(out[idx]) + to_f(bias[c]));
  }
}

// Fast FP16 bias add using half2 when HW even
__global__ void bias_add_half2_kernel(__half *__restrict__ out,
                                      const __half *__restrict__ bias, int N,
                                      int C, int HW) {
  int HW2 = HW >> 1;
  int total = N * C * HW2;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = (idx / HW2) % C;
    __half2 *out2 = reinterpret_cast<__half2 *>(out);
    __half2 b2 = __half2half2(bias[c]);
    out2[idx] = __hadd2(out2[idx], b2);
  }
}
template <typename T>
__global__ void pad_kernel(T *__restrict__ out, const T *__restrict__ in, int N,
                           int C, int H, int W, int out_h, int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * out_h * out_w;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total; idx += stride) {
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % C;
    int n = idx / (out_w * out_h * C);
    if (oh < H && ow < W) {
      out[idx] = in[((n * C + c) * H + oh) * W + ow];
    } else {
      out[idx] = from_f<T>(0.f);
    }
  }
}

// Crop to H,W from padded
template <typename T>
__global__ void crop_kernel(T *__restrict__ out, const T *__restrict__ in,
                            int N, int C, int src_h, int src_w, int out_h,
                            int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * out_h * out_w;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total; idx += stride) {
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % C;
    int n = idx / (out_w * out_h * C);
    out[idx] = in[((n * C + c) * src_h + oh) * src_w + ow];
  }
}

// PixelShuffle upscale_factor=2: (N, C*4, H, W) -> (N, C, H*2, W*2)
template <typename T>
__global__ void pixel_shuffle2_kernel(T *__restrict__ out,
                                      const T *__restrict__ in, int N, int C,
                                      int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = H * 2, out_w = W * 2;
  int total = N * C * out_h * out_w;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total; idx += stride) {
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % C;
    int n = idx / (out_w * out_h * C);
    int sh = oh / 2, sw = ow / 2;
    int y = oh & 1, x = ow & 1;
    int oc = c * 4 + y * 2 + x; // PyTorch PixelShuffle channel layout
    out[idx] = in[((n * (C * 4) + oc) * H + sh) * W + sw];
  }
}

// cast float <-> half
__global__ void cast_f2h_kernel(__half *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride)
    out[i] = __float2half(in[i]);
}

__global__ void cast_h2f_kernel(float *out, const __half *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride)
    out[i] = __half2float(in[i]);
}

static inline int grid_for(int n, int threads) {
  int g = (n + threads - 1) / threads;
  return g > 0 ? g : 1;
}

extern "C" void simple_gate_launcher(void *out, const void *in, int N, int C,
                                     int HW, int dtype, int config,
                                     cudaStream_t stream) {
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(N * C * HW, threads);
  if (dtype == 1)
    simple_gate_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, N, C, HW);
  else
    simple_gate_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, N, C, HW);
}

extern "C" void residual_launcher(void *out, const void *a, const void *b,
                                  const void *scale, int N, int C, int HW,
                                  int dtype, int config, cudaStream_t stream) {
  int n = N * C * HW;
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    residual_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)a, (const __half *)b,
        (const __half *)scale, N, C, HW);
  else
    residual_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)a, (const float *)b, (const float *)scale,
        N, C, HW);
}

extern "C" void add_launcher(void *out, const void *a, const void *b, int n,
                             int dtype, int config, cudaStream_t stream) {
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    add_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)a, (const __half *)b, n);
  else
    add_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)a, (const float *)b, n);
}

extern "C" void sca_launcher(void *out, const void *in, const void *weight,
                             const void *bias, void *workspace_pooled,
                             void *workspace_scale, int N, int C, int H, int W,
                             int dtype, int config, cudaStream_t stream) {
  int HW = H * W;
  int gap_threads = 256;
  if (dtype == 1) {
    gap_kernel<__half><<<N * C, gap_threads, 0, stream>>>(
        (__half *)workspace_pooled, (const __half *)in, N, C, HW);
    int lin_threads = 128;
    int lin_blocks = grid_for(N * C, lin_threads);
    sca_linear_kernel<__half><<<lin_blocks, lin_threads, 0, stream>>>(
        (__half *)workspace_scale, (const __half *)workspace_pooled,
        (const __half *)weight, (const __half *)bias, N, C);
    int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
    int blocks = grid_for(N * C * HW, threads);
    channel_scale_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, (const __half *)workspace_scale, N,
        C, HW);
  } else {
    gap_kernel<float><<<N * C, gap_threads, 0, stream>>>(
        (float *)workspace_pooled, (const float *)in, N, C, HW);
    int lin_threads = 128;
    int lin_blocks = grid_for(N * C, lin_threads);
    sca_linear_kernel<float><<<lin_blocks, lin_threads, 0, stream>>>(
        (float *)workspace_scale, (const float *)workspace_pooled,
        (const float *)weight, (const float *)bias, N, C);
    int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
    int blocks = grid_for(N * C * HW, threads);
    channel_scale_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, (const float *)workspace_scale, N, C,
        HW);
  }
}

extern "C" void bias_add_launcher(void *out, const void *bias, int N, int C,
                                  int HW, int dtype, int config,
                                  cudaStream_t stream) {
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  if (dtype == 1 && (HW % 2 == 0)) {
    int total = N * C * (HW >> 1);
    int blocks = grid_for(total, threads);
    if (blocks > 4096)
      blocks = 4096;
    bias_add_half2_kernel<<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)bias, N, C, HW);
    return;
  }
  int n = N * C * HW;
  int blocks = grid_for(n, threads);
  if (blocks > 4096)
    blocks = 4096;
  if (dtype == 1)
    bias_add_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)bias, N, C, HW);
  else
    bias_add_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)bias, N, C, HW);
}

extern "C" void pad_launcher(void *out, const void *in, int N, int C, int H,
                             int W, int out_h, int out_w, int dtype, int config,
                             cudaStream_t stream) {
  int n = N * C * out_h * out_w;
  int threads = 256;
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    pad_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, N, C, H, W, out_h, out_w);
  else
    pad_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, N, C, H, W, out_h, out_w);
}

extern "C" void crop_launcher(void *out, const void *in, int N, int C,
                              int src_h, int src_w, int out_h, int out_w,
                              int dtype, int config, cudaStream_t stream) {
  int n = N * C * out_h * out_w;
  int threads = 256;
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    crop_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, N, C, src_h, src_w, out_h, out_w);
  else
    crop_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, N, C, src_h, src_w, out_h, out_w);
}

extern "C" void pixel_shuffle2_launcher(void *out, const void *in, int N, int C,
                                        int H, int W, int dtype, int config,
                                        cudaStream_t stream) {
  int n = N * C * H * 2 * W * 2;
  int threads = 256;
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    pixel_shuffle2_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, N, C, H, W);
  else
    pixel_shuffle2_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, N, C, H, W);
}

extern "C" void cast_f2h_launcher(__half *out, const float *in, int n,
                                  cudaStream_t stream) {
  int threads = 256;
  cast_f2h_kernel<<<grid_for(n, threads), threads, 0, stream>>>(out, in, n);
}

extern "C" void cast_h2f_launcher(float *out, const __half *in, int n,
                                  cudaStream_t stream) {
  int threads = 256;
  cast_h2f_kernel<<<grid_for(n, threads), threads, 0, stream>>>(out, in, n);
}
