#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

template <int C>
__global__ void layernorm2d_nhwc_c_kernel(__half *__restrict__ out,
                                          const __half *__restrict__ in,
                                          const __half *__restrict__ weight,
                                          const __half *__restrict__ bias,
                                          int nhw, float eps) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nhw;
       idx += blockDim.x * gridDim.x) {
    const __half *row = in + idx * C;
    __half *orow = out + idx * C;

    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < C; c += 2) {
      __half2 v = *reinterpret_cast<const __half2 *>(row + c);
      sum += __low2float(v) + __high2float(v);
    }
    float mean = sum * (1.f / (float)C);

    float var_sum = 0.f;
#pragma unroll
    for (int c = 0; c < C; c += 2) {
      __half2 v = *reinterpret_cast<const __half2 *>(row + c);
      float a = __low2float(v) - mean;
      float b = __high2float(v) - mean;
      var_sum += a * a + b * b;
    }
    float inv_std = rsqrtf(var_sum * (1.f / (float)C) + eps);

#pragma unroll
    for (int c = 0; c < C; c += 2) {
      __half2 v = *reinterpret_cast<const __half2 *>(row + c);
      __half2 w = *reinterpret_cast<const __half2 *>(weight + c);
      __half2 b2 = *reinterpret_cast<const __half2 *>(bias + c);
      float y0 =
          (__low2float(v) - mean) * inv_std * __low2float(w) + __low2float(b2);
      float y1 = (__high2float(v) - mean) * inv_std * __high2float(w) +
                 __high2float(b2);
      *reinterpret_cast<__half2 *>(orow + c) =
          __halves2half2(__float2half(y0), __float2half(y1));
    }
  }
}

__global__ void layernorm2d_nhwc_half2_kernel(__half *__restrict__ out,
                                              const __half *__restrict__ in,
                                              const __half *__restrict__ weight,
                                              const __half *__restrict__ bias,
                                              int nhw, int C, float eps) {
  const int C2 = C >> 1;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nhw;
       idx += blockDim.x * gridDim.x) {
    const __half2 *row = reinterpret_cast<const __half2 *>(in + idx * C);
    __half2 *orow = reinterpret_cast<__half2 *>(out + idx * C);
    const __half2 *wrow = reinterpret_cast<const __half2 *>(weight);
    const __half2 *brow = reinterpret_cast<const __half2 *>(bias);
    float sum = 0.f;
    for (int c = 0; c < C2; ++c) {
      __half2 v = row[c];
      sum += __low2float(v) + __high2float(v);
    }
    float mean = sum * (1.f / (float)C);
    float var_sum = 0.f;
    for (int c = 0; c < C2; ++c) {
      __half2 v = row[c];
      float a = __low2float(v) - mean;
      float b = __high2float(v) - mean;
      var_sum += a * a + b * b;
    }
    float inv_std = rsqrtf(var_sum * (1.f / (float)C) + eps);
    for (int c = 0; c < C2; ++c) {
      __half2 v = row[c];
      __half2 w = wrow[c];
      __half2 b2 = brow[c];
      float y0 =
          (__low2float(v) - mean) * inv_std * __low2float(w) + __low2float(b2);
      float y1 = (__high2float(v) - mean) * inv_std * __high2float(w) +
                 __high2float(b2);
      orow[c] = __halves2half2(__float2half(y0), __float2half(y1));
    }
  }
}

__global__ void layernorm2d_float_kernel(float *__restrict__ out,
                                         const float *__restrict__ in,
                                         const float *__restrict__ weight,
                                         const float *__restrict__ bias,
                                         int nhw, int C, float eps) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nhw;
       idx += blockDim.x * gridDim.x) {
    const float *row = in + idx * C;
    float *orow = out + idx * C;
    float sum = 0.f;
    for (int c = 0; c < C; ++c)
      sum += row[c];
    float mean = sum / (float)C;
    float var_sum = 0.f;
    for (int c = 0; c < C; ++c) {
      float d = row[c] - mean;
      var_sum += d * d;
    }
    float inv_std = rsqrtf(var_sum / (float)C + eps);
    for (int c = 0; c < C; ++c)
      orow[c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
  }
}

extern "C" void layernorm2d_launcher(void *out, const void *in,
                                     const void *weight, const void *bias,
                                     int N, int H, int W, int C, float eps,
                                     int dtype, int config,
                                     cudaStream_t stream) {
  int nhw = N * H * W;
  if (nhw <= 0)
    return;
  int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
  int blocks = (nhw + threads - 1) / threads;
  if (blocks > 65535)
    blocks = 65535;

  if (dtype == 1) {
    if (C == 32) {
      layernorm2d_nhwc_c_kernel<32><<<blocks, threads, 0, stream>>>(
          (__half *)out, (const __half *)in, (const __half *)weight,
          (const __half *)bias, nhw, eps);
    } else if (C == 64) {
      layernorm2d_nhwc_c_kernel<64><<<blocks, threads, 0, stream>>>(
          (__half *)out, (const __half *)in, (const __half *)weight,
          (const __half *)bias, nhw, eps);
    } else if (C == 128) {
      layernorm2d_nhwc_c_kernel<128><<<blocks, threads, 0, stream>>>(
          (__half *)out, (const __half *)in, (const __half *)weight,
          (const __half *)bias, nhw, eps);
    } else if (C == 256) {
      layernorm2d_nhwc_c_kernel<256><<<blocks, threads, 0, stream>>>(
          (__half *)out, (const __half *)in, (const __half *)weight,
          (const __half *)bias, nhw, eps);
    } else if ((C & 1) == 0) {
      layernorm2d_nhwc_half2_kernel<<<blocks, threads, 0, stream>>>(
          (__half *)out, (const __half *)in, (const __half *)weight,
          (const __half *)bias, nhw, C, eps);
    } else {
      // odd C fallback
      layernorm2d_nhwc_half2_kernel<<<blocks, threads, 0, stream>>>(
          (__half *)out, (const __half *)in, (const __half *)weight,
          (const __half *)bias, nhw, C & ~1, eps);
    }
  } else {
    layernorm2d_float_kernel<<<blocks, threads, 0, stream>>>(
        (float *)out, (const float *)in, (const float *)weight,
        (const float *)bias, nhw, C, eps);
  }
}
