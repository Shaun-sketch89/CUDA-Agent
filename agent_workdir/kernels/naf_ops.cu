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

// NHWC SimpleGate: in [N,H,W,2C] -> out [N,H,W,C]
template <typename T>
__global__ void simple_gate_nhwc_kernel(T *__restrict__ out,
                                        const T *__restrict__ in, int nhw,
                                        int C) {
  int total = nhw * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int base = (idx / C) * (2 * C);
    float a = to_f(in[base + c]);
    float b = to_f(in[base + c + C]);
    out[idx] = from_f<T>(a * b);
  }
}

// SimpleGate with pre-bias on 2C channels: (in+b1)*(in+b2)
template <typename T>
__global__ void
simple_gate_bias_nhwc_kernel(T *__restrict__ out, const T *__restrict__ in,
                             const T *__restrict__ bias, int nhw, int C) {
  int total = nhw * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int base = (idx / C) * (2 * C);
    float a = to_f(in[base + c]) + to_f(bias[c]);
    float b = to_f(in[base + c + C]) + to_f(bias[c + C]);
    out[idx] = from_f<T>(a * b);
  }
}

// NHWC residual: out = a + b * scale[c]
template <typename T>
__global__ void
residual_nhwc_kernel(T *__restrict__ out, const T *__restrict__ a,
                     const T *__restrict__ b, const T *__restrict__ scale,
                     int N, int H, int W, int C) {
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    out[idx] = from_f<T>(to_f(a[idx]) + to_f(b[idx]) * to_f(scale[c]));
  }
}

// residual with bias on b: out = a + (b + bias[c]) * scale[c]
template <typename T>
__global__ void
residual_bias_nhwc_kernel(T *__restrict__ out, const T *__restrict__ a,
                          const T *__restrict__ b, const T *__restrict__ scale,
                          const T *__restrict__ bias, int N, int H, int W,
                          int C) {
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    out[idx] = from_f<T>(to_f(a[idx]) +
                         (to_f(b[idx]) + to_f(bias[c])) * to_f(scale[c]));
  }
}

// Fused: y = a + b*scale; ln_out = LayerNorm(y); writes both y and ln_out.
template <typename T>
__global__ void residual_dual_ln_nhwc_kernel(
    T *__restrict__ y_out, T *__restrict__ ln_out, const T *__restrict__ a,
    const T *__restrict__ b, const T *__restrict__ scale,
    const T *__restrict__ weight, const T *__restrict__ bias, int N, int H,
    int W, int C, float eps) {
  int nhw = N * H * W;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nhw;
       idx += blockDim.x * gridDim.x) {
    int base = idx * C;
    float sum = 0.f;
#pragma unroll 4
    for (int c = 0; c < C; ++c) {
      float yv = to_f(a[base + c]) + to_f(b[base + c]) * to_f(scale[c]);
      y_out[base + c] = from_f<T>(yv);
      sum += yv;
    }
    float mean = sum / (float)C;
    float var_sum = 0.f;
#pragma unroll 4
    for (int c = 0; c < C; ++c) {
      float d = to_f(y_out[base + c]) - mean;
      var_sum += d * d;
    }
    float inv_std = rsqrtf(var_sum / (float)C + eps);
#pragma unroll 4
    for (int c = 0; c < C; ++c) {
      float y = to_f(y_out[base + c]);
      y = (y - mean) * inv_std;
      y = y * to_f(weight[c]) + to_f(bias[c]);
      ln_out[base + c] = from_f<T>(y);
    }
  }
}

template <typename T>
__global__ void add_kernel(T *__restrict__ out, const T *__restrict__ a,
                           const T *__restrict__ b, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x)
    out[i] = from_f<T>(to_f(a[i]) + to_f(b[i]));
}

// NHWC GAP -> (N,C)
template <typename T>
__global__ void gap_nhwc_kernel(T *__restrict__ pooled,
                                const T *__restrict__ in, int N, int H, int W,
                                int C) {
  int bc = blockIdx.x;
  if (bc >= N * C)
    return;
  int n = bc / C;
  int c = bc - n * C;
  const T *ptr = in + (n * H * W * C) + c;

  float sum = 0.f;
  int hw = H * W;
  for (int i = threadIdx.x; i < hw; i += blockDim.x) {
    sum += to_f(ptr[i * C]);
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
      pooled[bc] = from_f<T>(v / (float)hw);
  }
}

// weight KRSC [C,1,1,C] flattened as row-major [C,C]
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
__global__ void channel_scale_nhwc_kernel(T *__restrict__ out,
                                          const T *__restrict__ in,
                                          const T *__restrict__ scale, int N,
                                          int H, int W, int C) {
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int nhw = idx / C;
    int n = nhw / (H * W);
    out[idx] = from_f<T>(to_f(in[idx]) * to_f(scale[n * C + c]));
  }
}

// NHWC bias add: c = idx % C
template <typename T>
__global__ void bias_add_nhwc_kernel(T *__restrict__ out,
                                     const T *__restrict__ bias, int N, int H,
                                     int W, int C) {
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    out[idx] = from_f<T>(to_f(out[idx]) + to_f(bias[c]));
  }
}

__global__ void bias_add_half2_nhwc_kernel(__half *__restrict__ out,
                                           const __half *__restrict__ bias,
                                           int N, int H, int W, int C) {
  int nhw = N * H * W;
  int cpairs = C >> 1;
  int total = nhw * cpairs;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c0 = (idx % cpairs) * 2;
    int nhw_idx = idx / cpairs;
    int base = nhw_idx * C + c0;
    __half2 b2 = __half2(bias[c0], bias[c0 + 1]);
    __half2 *out2 = reinterpret_cast<__half2 *>(out);
    out2[base >> 1] = __hadd2(out2[base >> 1], b2);
  }
}

template <typename T>
__global__ void pad_nhwc_kernel(T *__restrict__ out, const T *__restrict__ in,
                                int N, int C, int H, int W, int out_h,
                                int out_w) {
  int total = N * out_h * out_w * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int tmp = idx / C;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    int n = tmp / out_h;
    if (oh < H && ow < W) {
      out[idx] = in[((n * H + oh) * W + ow) * C + c];
    } else {
      out[idx] = from_f<T>(0.f);
    }
  }
}

template <typename T>
__global__ void crop_nhwc_kernel(T *__restrict__ out, const T *__restrict__ in,
                                 int N, int C, int src_h, int src_w, int out_h,
                                 int out_w) {
  int total = N * out_h * out_w * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int tmp = idx / C;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    int n = tmp / out_h;
    out[idx] = in[((n * src_h + oh) * src_w + ow) * C + c];
  }
}

// NHWC PixelShuffle x2: (N,H,W,C*4) -> (N,H*2,W*2,C)
template <typename T>
__global__ void pixel_shuffle2_nhwc_kernel(T *__restrict__ out,
                                           const T *__restrict__ in, int N,
                                           int C, int H, int W) {
  int out_h = H * 2;
  int out_w = W * 2;
  int total = N * out_h * out_w * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int tmp = idx / C;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    int n = tmp / out_h;
    int sh = oh >> 1;
    int sw = ow >> 1;
    int y = oh & 1;
    int x = ow & 1;
    int ic = c * 4 + y * 2 + x;
    out[idx] = in[((n * H + sh) * W + sw) * (C * 4) + ic];
  }
}

__global__ void cast_f2h_kernel(__half *out, const float *in, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x)
    out[i] = __float2half(in[i]);
}

__global__ void cast_h2f_kernel(float *out, const __half *in, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x)
    out[i] = __half2float(in[i]);
}

__global__ void nchw_to_nhwc_kernel(__half *out, const __half *in, int N, int C,
                                    int H, int W) {
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int tmp = idx / C;
    int w = tmp % W;
    tmp /= W;
    int h = tmp % H;
    int n = tmp / H;
    out[idx] = in[((n * C + c) * H + h) * W + w];
  }
}

__global__ void nhwc_to_nchw_kernel(__half *out, const __half *in, int N, int C,
                                    int H, int W) {
  int total = N * H * W * C;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int c = idx % C;
    int tmp = idx / C;
    int w = tmp % W;
    tmp /= W;
    int h = tmp % H;
    int n = tmp / H;
    out[((n * C + c) * H + h) * W + w] = in[idx];
  }
}

__global__ void kcrs_to_krsc_kernel(__half *out, const __half *in, int K, int C,
                                    int R, int S) {
  int total = K * C * R * S;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int s = idx % S;
    int tmp = idx / S;
    int r = tmp % R;
    tmp /= R;
    int c = tmp % C;
    int k = tmp / C;
    out[((k * R + r) * S + s) * C + c] = in[((k * C + c) * R + r) * S + s];
  }
}

static inline int grid_for(int n, int threads) {
  int g = (n + threads - 1) / threads;
  return g > 0 ? g : 1;
}

extern "C" void simple_gate_launcher(void *out, const void *in, int N, int H,
                                     int W, int C, int dtype, int config,
                                     cudaStream_t stream) {
  int nhw = N * H * W;
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(nhw * C, threads);
  if (blocks > 4096)
    blocks = 4096;
  if (dtype == 1)
    simple_gate_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, nhw, C);
  else
    simple_gate_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, nhw, C);
}

extern "C" void residual_launcher(void *out, const void *a, const void *b,
                                  const void *scale, int N, int H, int W, int C,
                                  int dtype, int config, cudaStream_t stream) {
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(N * H * W * C, threads);
  if (dtype == 1)
    residual_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)a, (const __half *)b,
        (const __half *)scale, N, H, W, C);
  else
    residual_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)a, (const float *)b, (const float *)scale,
        N, H, W, C);
}

extern "C" void simple_gate_bias_launcher(void *out, const void *in,
                                          const void *bias, int N, int H, int W,
                                          int C, int dtype, int config,
                                          cudaStream_t stream) {
  int nhw = N * H * W;
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(nhw * C, threads);
  if (blocks > 4096)
    blocks = 4096;
  if (dtype == 1)
    simple_gate_bias_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, (const __half *)bias, nhw, C);
  else
    simple_gate_bias_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, (const float *)bias, nhw, C);
}

extern "C" void residual_bias_launcher(void *out, const void *a, const void *b,
                                       const void *scale, const void *bias,
                                       int N, int H, int W, int C, int dtype,
                                       int config, cudaStream_t stream) {
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(N * H * W * C, threads);
  if (dtype == 1)
    residual_bias_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)a, (const __half *)b,
        (const __half *)scale, (const __half *)bias, N, H, W, C);
  else
    residual_bias_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)a, (const float *)b, (const float *)scale,
        (const float *)bias, N, H, W, C);
}

extern "C" void residual_ln_launcher(void *y_out, void *ln_out, const void *a,
                                     const void *b, const void *scale,
                                     const void *weight, const void *bias,
                                     int N, int H, int W, int C, float eps,
                                     int dtype, int config,
                                     cudaStream_t stream) {
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = grid_for(N * H * W, threads);
  if (blocks > 4096)
    blocks = 4096;
  if (dtype == 1)
    residual_dual_ln_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)y_out, (__half *)ln_out, (const __half *)a, (const __half *)b,
        (const __half *)scale, (const __half *)weight, (const __half *)bias, N,
        H, W, C, eps);
  else
    residual_dual_ln_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)y_out, (float *)ln_out, (const float *)a, (const float *)b,
        (const float *)scale, (const float *)weight, (const float *)bias, N, H,
        W, C, eps);
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
                             void *workspace_scale, int N, int H, int W, int C,
                             int dtype, int config, cudaStream_t stream) {
  int gap_threads = 256;
  if (dtype == 1) {
    gap_nhwc_kernel<__half><<<N * C, gap_threads, 0, stream>>>(
        (__half *)workspace_pooled, (const __half *)in, N, H, W, C);
    int lin_threads = 128;
    int lin_blocks = grid_for(N * C, lin_threads);
    sca_linear_kernel<__half><<<lin_blocks, lin_threads, 0, stream>>>(
        (__half *)workspace_scale, (const __half *)workspace_pooled,
        (const __half *)weight, (const __half *)bias, N, C);
    int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
    int blocks = grid_for(N * H * W * C, threads);
    channel_scale_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, (const __half *)workspace_scale, N,
        H, W, C);
  } else {
    gap_nhwc_kernel<float><<<N * C, gap_threads, 0, stream>>>(
        (float *)workspace_pooled, (const float *)in, N, H, W, C);
    int lin_threads = 128;
    int lin_blocks = grid_for(N * C, lin_threads);
    sca_linear_kernel<float><<<lin_blocks, lin_threads, 0, stream>>>(
        (float *)workspace_scale, (const float *)workspace_pooled,
        (const float *)weight, (const float *)bias, N, C);
    int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
    int blocks = grid_for(N * H * W * C, threads);
    channel_scale_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, (const float *)workspace_scale, N, H,
        W, C);
  }
}

extern "C" void bias_add_launcher(void *out, const void *bias, int N, int H,
                                  int W, int C, int dtype, int nhwc, int config,
                                  cudaStream_t stream) {
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  if (nhwc && dtype == 1 && (C % 2 == 0)) {
    int total = N * H * W * (C >> 1);
    int blocks = grid_for(total, threads);
    if (blocks > 4096)
      blocks = 4096;
    bias_add_half2_nhwc_kernel<<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)bias, N, H, W, C);
    return;
  }
  int total = nhwc ? N * H * W * C : N * C * H * W;
  int blocks = grid_for(total, threads);
  if (blocks > 4096)
    blocks = 4096;
  if (dtype == 1)
    bias_add_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)bias, N, H, W, C);
  else
    bias_add_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)bias, N, H, W, C);
}

extern "C" void pad_launcher(void *out, const void *in, int N, int H, int W,
                             int C, int out_h, int out_w, int dtype, int config,
                             cudaStream_t stream) {
  int n = N * out_h * out_w * C;
  int threads = 256;
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    pad_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, N, C, H, W, out_h, out_w);
  else
    pad_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, N, C, H, W, out_h, out_w);
}

extern "C" void crop_launcher(void *out, const void *in, int N, int src_h,
                              int src_w, int C, int out_h, int out_w, int dtype,
                              int config, cudaStream_t stream) {
  int n = N * out_h * out_w * C;
  int threads = 256;
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    crop_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, N, C, src_h, src_w, out_h, out_w);
  else
    crop_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, N, C, src_h, src_w, out_h, out_w);
}

extern "C" void pixel_shuffle2_launcher(void *out, const void *in, int N, int H,
                                        int W, int C, int dtype, int config,
                                        cudaStream_t stream) {
  int n = N * H * 2 * W * 2 * C;
  int threads = 256;
  int blocks = grid_for(n, threads);
  if (dtype == 1)
    pixel_shuffle2_nhwc_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half *)out, (const __half *)in, N, C, H, W);
  else
    pixel_shuffle2_nhwc_kernel<float><<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, N, C, H, W);
}

extern "C" void cast_f2h_launcher(__half *out, const float *in, int n,
                                  cudaStream_t stream) {
  cast_f2h_kernel<<<grid_for(n, 256), 256, 0, stream>>>(out, in, n);
}

extern "C" void cast_h2f_launcher(float *out, const __half *in, int n,
                                  cudaStream_t stream) {
  cast_h2f_kernel<<<grid_for(n, 256), 256, 0, stream>>>(out, in, n);
}

extern "C" void nchw_to_nhwc_launcher(__half *out, const __half *in, int N,
                                      int C, int H, int W,
                                      cudaStream_t stream) {
  int total = N * H * W * C;
  int threads = 256;
  int blocks = grid_for(total, threads);
  if (blocks > 4096)
    blocks = 4096;
  nchw_to_nhwc_kernel<<<blocks, threads, 0, stream>>>(out, in, N, C, H, W);
}

extern "C" void nhwc_to_nchw_launcher(__half *out, const __half *in, int N,
                                      int C, int H, int W,
                                      cudaStream_t stream) {
  nhwc_to_nchw_kernel<<<grid_for(N * H * W * C, 256), 256, 0, stream>>>(
      out, in, N, C, H, W);
}

extern "C" void kcrs_to_krsc_launcher(__half *out, const __half *in, int K,
                                      int C, int R, int S,
                                      cudaStream_t stream) {
  kcrs_to_krsc_kernel<<<grid_for(K * C * R * S, 256), 256, 0, stream>>>(
      out, in, K, C, R, S);
}
