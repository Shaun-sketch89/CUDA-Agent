// cuDNN convolution via dynamically loaded cudnn (cudnn64_9.dll).
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef enum { CUDNN_STATUS_SUCCESS = 0 } cudnnStatus_t;
typedef enum { CUDNN_DATA_FLOAT = 0, CUDNN_DATA_HALF = 2 } cudnnDataType_t;
typedef enum {
  CUDNN_TENSOR_NCHW = 0,
  CUDNN_TENSOR_NHWC = 1
} cudnnTensorFormat_t;
typedef enum { CUDNN_CROSS_CORRELATION = 1 } cudnnConvolutionMode_t;
typedef enum {
  CUDNN_DEFAULT_MATH = 0,
  CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2
} cudnnMathType_t;
typedef enum {
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
} cudnnConvolutionFwdAlgo_t;

struct cudnnContext;
struct cudnnTensorStruct;
struct cudnnFilterStruct;
struct cudnnConvolutionStruct;
typedef cudnnContext *cudnnHandle_t;
typedef cudnnTensorStruct *cudnnTensorDescriptor_t;
typedef cudnnFilterStruct *cudnnFilterDescriptor_t;
typedef cudnnConvolutionStruct *cudnnConvolutionDescriptor_t;

#ifdef _WIN32
static HMODULE g_cudnn = nullptr;
template <typename T> static T load_sym(const char *name) {
  auto p = reinterpret_cast<T>(GetProcAddress(g_cudnn, name));
  if (!p)
    throw std::runtime_error(std::string("Missing cudnn symbol: ") + name);
  return p;
}
#else
static void *g_cudnn = nullptr;
template <typename T> static T load_sym(const char *name) {
  auto p = reinterpret_cast<T>(dlsym(g_cudnn, name));
  if (!p)
    throw std::runtime_error(std::string("Missing cudnn symbol: ") + name);
  return p;
}
#endif

#define CUDNN_FN(ret, name, ...) ret (*name)(__VA_ARGS__) = nullptr

CUDNN_FN(const char *, cudnnGetErrorString, cudnnStatus_t);
CUDNN_FN(cudnnStatus_t, cudnnCreate, cudnnHandle_t *);
CUDNN_FN(cudnnStatus_t, cudnnSetStream, cudnnHandle_t, cudaStream_t);
CUDNN_FN(cudnnStatus_t, cudnnCreateTensorDescriptor, cudnnTensorDescriptor_t *);
CUDNN_FN(cudnnStatus_t, cudnnDestroyTensorDescriptor, cudnnTensorDescriptor_t);
CUDNN_FN(cudnnStatus_t, cudnnSetTensor4dDescriptor, cudnnTensorDescriptor_t,
         cudnnTensorFormat_t, cudnnDataType_t, int, int, int, int);
CUDNN_FN(cudnnStatus_t, cudnnCreateFilterDescriptor, cudnnFilterDescriptor_t *);
CUDNN_FN(cudnnStatus_t, cudnnDestroyFilterDescriptor, cudnnFilterDescriptor_t);
CUDNN_FN(cudnnStatus_t, cudnnSetFilter4dDescriptor, cudnnFilterDescriptor_t,
         cudnnDataType_t, cudnnTensorFormat_t, int, int, int, int);
CUDNN_FN(cudnnStatus_t, cudnnCreateConvolutionDescriptor,
         cudnnConvolutionDescriptor_t *);
CUDNN_FN(cudnnStatus_t, cudnnDestroyConvolutionDescriptor,
         cudnnConvolutionDescriptor_t);
CUDNN_FN(cudnnStatus_t, cudnnSetConvolution2dDescriptor,
         cudnnConvolutionDescriptor_t, int, int, int, int, int, int,
         cudnnConvolutionMode_t, cudnnDataType_t);
CUDNN_FN(cudnnStatus_t, cudnnSetConvolutionGroupCount,
         cudnnConvolutionDescriptor_t, int);
CUDNN_FN(cudnnStatus_t, cudnnSetConvolutionMathType,
         cudnnConvolutionDescriptor_t, cudnnMathType_t);
CUDNN_FN(cudnnStatus_t, cudnnGetConvolutionForwardWorkspaceSize, cudnnHandle_t,
         const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
         const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
         cudnnConvolutionFwdAlgo_t, size_t *);
CUDNN_FN(cudnnStatus_t, cudnnConvolutionForward, cudnnHandle_t, const void *,
         const cudnnTensorDescriptor_t, const void *,
         const cudnnFilterDescriptor_t, const void *,
         const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *,
         size_t, const void *, const cudnnTensorDescriptor_t, void *);

static void ensure_cudnn_loaded() {
  if (g_cudnn)
    return;
#ifdef _WIN32
  const char *names[] = {"cudnn64_9.dll", "cudnn64_8.dll", nullptr};
  for (int i = 0; names[i]; ++i) {
    g_cudnn = LoadLibraryA(names[i]);
    if (g_cudnn)
      break;
  }
  if (!g_cudnn)
    throw std::runtime_error("Failed to LoadLibrary cudnn64_9/8.dll");
#else
  const char *names[] = {"libcudnn.so.9", "libcudnn.so.8", "libcudnn.so",
                         nullptr};
  for (int i = 0; names[i]; ++i) {
    g_cudnn = dlopen(names[i], RTLD_LAZY);
    if (g_cudnn)
      break;
  }
  if (!g_cudnn)
    throw std::runtime_error("Failed to dlopen libcudnn");
#endif

  cudnnGetErrorString =
      load_sym<decltype(cudnnGetErrorString)>("cudnnGetErrorString");
  cudnnCreate = load_sym<decltype(cudnnCreate)>("cudnnCreate");
  cudnnSetStream = load_sym<decltype(cudnnSetStream)>("cudnnSetStream");
  cudnnCreateTensorDescriptor = load_sym<decltype(cudnnCreateTensorDescriptor)>(
      "cudnnCreateTensorDescriptor");
  cudnnDestroyTensorDescriptor =
      load_sym<decltype(cudnnDestroyTensorDescriptor)>(
          "cudnnDestroyTensorDescriptor");
  cudnnSetTensor4dDescriptor = load_sym<decltype(cudnnSetTensor4dDescriptor)>(
      "cudnnSetTensor4dDescriptor");
  cudnnCreateFilterDescriptor = load_sym<decltype(cudnnCreateFilterDescriptor)>(
      "cudnnCreateFilterDescriptor");
  cudnnDestroyFilterDescriptor =
      load_sym<decltype(cudnnDestroyFilterDescriptor)>(
          "cudnnDestroyFilterDescriptor");
  cudnnSetFilter4dDescriptor = load_sym<decltype(cudnnSetFilter4dDescriptor)>(
      "cudnnSetFilter4dDescriptor");
  cudnnCreateConvolutionDescriptor =
      load_sym<decltype(cudnnCreateConvolutionDescriptor)>(
          "cudnnCreateConvolutionDescriptor");
  cudnnDestroyConvolutionDescriptor =
      load_sym<decltype(cudnnDestroyConvolutionDescriptor)>(
          "cudnnDestroyConvolutionDescriptor");
  cudnnSetConvolution2dDescriptor =
      load_sym<decltype(cudnnSetConvolution2dDescriptor)>(
          "cudnnSetConvolution2dDescriptor");
  cudnnSetConvolutionGroupCount =
      load_sym<decltype(cudnnSetConvolutionGroupCount)>(
          "cudnnSetConvolutionGroupCount");
  cudnnSetConvolutionMathType = load_sym<decltype(cudnnSetConvolutionMathType)>(
      "cudnnSetConvolutionMathType");
  cudnnGetConvolutionForwardWorkspaceSize =
      load_sym<decltype(cudnnGetConvolutionForwardWorkspaceSize)>(
          "cudnnGetConvolutionForwardWorkspaceSize");
  cudnnConvolutionForward =
      load_sym<decltype(cudnnConvolutionForward)>("cudnnConvolutionForward");
}

static void check_cudnn(cudnnStatus_t s, const char *msg) {
  if (s != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(s));
  }
}

static cudnnHandle_t get_cudnn_handle(cudaStream_t stream) {
  ensure_cudnn_loaded();
  static thread_local cudnnHandle_t handle = nullptr;
  if (!handle)
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");
  check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream");
  return handle;
}

struct ConvKey {
  int n, c, h, w, k, r, s, pad_h, pad_w, stride_h, stride_w, dilation_h,
      dilation_w, groups, dtype, nhwc;
  bool operator==(const ConvKey &o) const {
    return n == o.n && c == o.c && h == o.h && w == o.w && k == o.k &&
           r == o.r && s == o.s && pad_h == o.pad_h && pad_w == o.pad_w &&
           stride_h == o.stride_h && stride_w == o.stride_w &&
           dilation_h == o.dilation_h && dilation_w == o.dilation_w &&
           groups == o.groups && dtype == o.dtype && nhwc == o.nhwc;
  }
};

struct ConvKeyHash {
  size_t operator()(const ConvKey &k) const {
    size_t h = 1469598103934665603ULL;
    auto mix = [&](int v) {
      h ^= (size_t)v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(k.n);
    mix(k.c);
    mix(k.h);
    mix(k.w);
    mix(k.k);
    mix(k.r);
    mix(k.s);
    mix(k.pad_h);
    mix(k.pad_w);
    mix(k.stride_h);
    mix(k.stride_w);
    mix(k.dilation_h);
    mix(k.dilation_w);
    mix(k.groups);
    mix(k.dtype);
    mix(k.nhwc);
    return h;
  }
};

struct CachedConv {
  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnConvolutionFwdAlgo_t algo =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t workspace_size = 0;
  bool ready = false;
};

static std::mutex g_mu;
static std::unordered_map<ConvKey, CachedConv, ConvKeyHash> g_cache;
static void *g_workspace = nullptr;
static size_t g_workspace_bytes = 0;

static void ensure_workspace(size_t bytes) {
  if (bytes <= g_workspace_bytes)
    return;
  void *new_ptr = nullptr;
  if (bytes > 0) {
    if (cudaMalloc(&new_ptr, bytes) != cudaSuccess) {
      if (g_workspace_bytes > 0)
        return;
      throw std::runtime_error("cudaMalloc workspace failed");
    }
  }
  if (g_workspace)
    cudaFree(g_workspace);
  g_workspace = new_ptr;
  g_workspace_bytes = bytes;
}

static CachedConv &get_or_create(const ConvKey &key, int out_h, int out_w) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_cache.find(key);
  if (it != g_cache.end())
    return it->second;

  ensure_cudnn_loaded();
  CachedConv cc;
  cudnnDataType_t cdt = key.dtype == 1 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t fmt = key.nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  check_cudnn(cudnnCreateTensorDescriptor(&cc.x_desc), "x");
  check_cudnn(cudnnCreateTensorDescriptor(&cc.y_desc), "y");
  check_cudnn(cudnnCreateFilterDescriptor(&cc.w_desc), "w");
  check_cudnn(cudnnCreateConvolutionDescriptor(&cc.conv_desc), "conv");

  check_cudnn(cudnnSetTensor4dDescriptor(cc.x_desc, fmt, cdt, key.n, key.c,
                                         key.h, key.w),
              "sx");
  check_cudnn(cudnnSetTensor4dDescriptor(cc.y_desc, fmt, cdt, key.n, key.k,
                                         out_h, out_w),
              "sy");
  check_cudnn(cudnnSetFilter4dDescriptor(cc.w_desc, cdt, fmt, key.k,
                                         key.c / key.groups, key.r, key.s),
              "sw");
  check_cudnn(cudnnSetConvolution2dDescriptor(
                  cc.conv_desc, key.pad_h, key.pad_w, key.stride_h,
                  key.stride_w, key.dilation_h, key.dilation_w,
                  CUDNN_CROSS_CORRELATION, cdt),
              "sc");
  check_cudnn(cudnnSetConvolutionGroupCount(cc.conv_desc, key.groups), "sg");
  check_cudnn(cudnnSetConvolutionMathType(
                  cc.conv_desc, key.dtype == 1
                                    ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
                                    : CUDNN_DEFAULT_MATH),
              "sm");

  cc.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  return g_cache.emplace(key, cc).first->second;
}

extern "C" void
cudnn_conv2d_launcher(void *output, const void *input, const void *weight,
                      const void *bias, int n, int c, int h, int w, int k,
                      int r, int s, int pad_h, int pad_w, int stride_h,
                      int stride_w, int dilation_h, int dilation_w, int groups,
                      int dtype, int nhwc, int config, cudaStream_t stream) {
  (void)bias;
  (void)config;
  auto handle = get_cudnn_handle(stream);
  int out_h = (h + 2 * pad_h - dilation_h * (r - 1) - 1) / stride_h + 1;
  int out_w = (w + 2 * pad_w - dilation_w * (s - 1) - 1) / stride_w + 1;

  ConvKey key{n,          c,      h,     w,        k,        r,
              s,          pad_h,  pad_w, stride_h, stride_w, dilation_h,
              dilation_w, groups, dtype, nhwc};
  CachedConv &cc = get_or_create(key, out_h, out_w);

  if (!cc.ready) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!cc.ready) {
      cc.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      size_t ws = 0;
      auto st = cudnnGetConvolutionForwardWorkspaceSize(
          handle, cc.x_desc, cc.w_desc, cc.conv_desc, cc.y_desc, cc.algo, &ws);
      if (st != CUDNN_STATUS_SUCCESS) {
        cc.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
                        handle, cc.x_desc, cc.w_desc, cc.conv_desc, cc.y_desc,
                        cc.algo, &ws),
                    "ws");
      }
      cc.workspace_size = ws;
      cc.ready = true;
    }
  }

  ensure_workspace(cc.workspace_size);

  float alpha = 1.f, beta = 0.f;
  check_cudnn(cudnnConvolutionForward(handle, &alpha, cc.x_desc, input,
                                      cc.w_desc, weight, cc.conv_desc, cc.algo,
                                      g_workspace, cc.workspace_size, &beta,
                                      cc.y_desc, output),
              "fwd");
}
