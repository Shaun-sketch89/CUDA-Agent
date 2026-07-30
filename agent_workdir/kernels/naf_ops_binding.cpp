#include "../binding_registry.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

extern "C" void simple_gate_launcher(void *, const void *, int, int, int, int,
                                     int, int, cudaStream_t);
extern "C" void simple_gate_bias_launcher(void *, const void *, const void *,
                                          int, int, int, int, int, int,
                                          cudaStream_t);
extern "C" void residual_launcher(void *, const void *, const void *,
                                  const void *, int, int, int, int, int, int,
                                  cudaStream_t);
extern "C" void residual_bias_launcher(void *, const void *, const void *,
                                       const void *, const void *, int, int,
                                       int, int, int, int, cudaStream_t);
extern "C" void residual_ln_launcher(void *, void *, const void *, const void *,
                                     const void *, const void *, const void *,
                                     int, int, int, int, float, int, int,
                                     cudaStream_t);
extern "C" void add_launcher(void *, const void *, const void *, int, int, int,
                             cudaStream_t);
extern "C" void sca_launcher(void *, const void *, const void *, const void *,
                             void *, void *, int, int, int, int, int, int,
                             cudaStream_t);
extern "C" void pad_launcher(void *, const void *, int, int, int, int, int, int,
                             int, int, cudaStream_t);
extern "C" void crop_launcher(void *, const void *, int, int, int, int, int,
                              int, int, int, cudaStream_t);
extern "C" void pixel_shuffle2_launcher(void *, const void *, int, int, int,
                                        int, int, int, cudaStream_t);
extern "C" void cast_f2h_launcher(__half *, const float *, int, cudaStream_t);
extern "C" void cast_h2f_launcher(float *, const __half *, int, cudaStream_t);
extern "C" void nchw_to_nhwc_launcher(__half *, const __half *, int, int, int,
                                      int, cudaStream_t);
extern "C" void nhwc_to_nchw_launcher(__half *, const __half *, int, int, int,
                                      int, cudaStream_t);
extern "C" void kcrs_to_krsc_launcher(__half *, const __half *, int, int, int,
                                      int, cudaStream_t);

static int dtype_flag(const torch::Tensor &t) {
  TORCH_CHECK(t.scalar_type() == torch::kFloat32 ||
                  t.scalar_type() == torch::kFloat16,
              "fp32/fp16 only");
  return t.scalar_type() == torch::kFloat16 ? 1 : 0;
}

static torch::Tensor simple_gate_forward(torch::Tensor input,
                                         int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "bad input");
  TORCH_CHECK(input.size(3) % 2 == 0, "channels must be even");
  int N = (int)input.size(0);
  int H = (int)input.size(1);
  int W = (int)input.size(2);
  int C = (int)(input.size(3) / 2);
  auto output = torch::empty({N, H, W, C}, input.options());
  simple_gate_launcher(output.data_ptr(), input.data_ptr(), N, H, W, C,
                       dtype_flag(input), (int)config,
                       c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor simple_gate_bias_forward(torch::Tensor input,
                                              torch::Tensor bias,
                                              int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "CUDA");
  TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "contiguous");
  TORCH_CHECK(input.dim() == 4 && input.size(3) % 2 == 0, "shape");
  TORCH_CHECK(bias.numel() == input.size(3), "bias 2C");
  int N = (int)input.size(0);
  int H = (int)input.size(1);
  int W = (int)input.size(2);
  int C = (int)(input.size(3) / 2);
  auto output = torch::empty({N, H, W, C}, input.options());
  simple_gate_bias_launcher(output.data_ptr(), input.data_ptr(),
                            bias.data_ptr(), N, H, W, C, dtype_flag(input),
                            (int)config,
                            c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor residual_forward(torch::Tensor a, torch::Tensor b,
                                      torch::Tensor scale, int64_t config = 0) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && scale.is_cuda(), "CUDA");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && scale.is_contiguous(),
              "contiguous");
  TORCH_CHECK(a.sizes() == b.sizes() && a.dim() == 4, "shape");
  auto output = torch::empty_like(a);
  int N = (int)a.size(0);
  int H = (int)a.size(1);
  int W = (int)a.size(2);
  int C = (int)a.size(3);
  residual_launcher(output.data_ptr(), a.data_ptr(), b.data_ptr(),
                    scale.data_ptr(), N, H, W, C, dtype_flag(a), (int)config,
                    c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor residual_bias_forward(torch::Tensor a, torch::Tensor b,
                                           torch::Tensor scale,
                                           torch::Tensor bias,
                                           int64_t config = 0) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && scale.is_cuda() && bias.is_cuda(),
              "CUDA");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && scale.is_contiguous() &&
                  bias.is_contiguous(),
              "contiguous");
  TORCH_CHECK(a.sizes() == b.sizes() && a.dim() == 4, "shape");
  TORCH_CHECK(bias.numel() == a.size(3), "bias C");
  auto output = torch::empty_like(a);
  int N = (int)a.size(0);
  int H = (int)a.size(1);
  int W = (int)a.size(2);
  int C = (int)a.size(3);
  residual_bias_launcher(output.data_ptr(), a.data_ptr(), b.data_ptr(),
                         scale.data_ptr(), bias.data_ptr(), N, H, W, C,
                         dtype_flag(a), (int)config,
                         c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static py::tuple residual_ln_forward(torch::Tensor a, torch::Tensor b,
                                     torch::Tensor scale, torch::Tensor weight,
                                     torch::Tensor bias, double eps,
                                     int64_t config = 0) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && scale.is_cuda(), "CUDA");
  TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "CUDA");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && scale.is_contiguous(),
              "contiguous");
  TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "contiguous");
  TORCH_CHECK(a.sizes() == b.sizes() && a.dim() == 4, "shape");
  auto y_out = torch::empty_like(a);
  auto ln_out = torch::empty_like(a);
  int N = (int)a.size(0);
  int H = (int)a.size(1);
  int W = (int)a.size(2);
  int C = (int)a.size(3);
  residual_ln_launcher(y_out.data_ptr(), ln_out.data_ptr(), a.data_ptr(),
                       b.data_ptr(), scale.data_ptr(), weight.data_ptr(),
                       bias.data_ptr(), N, H, W, C, (float)eps, dtype_flag(a),
                       (int)config, c10::cuda::getCurrentCUDAStream().stream());
  return py::make_tuple(y_out, ln_out);
}

static torch::Tensor add_forward(torch::Tensor a, torch::Tensor b,
                                 int64_t config = 0) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape");
  auto output = torch::empty_like(a);
  add_launcher(output.data_ptr(), a.data_ptr(), b.data_ptr(), (int)a.numel(),
               dtype_flag(a), (int)config,
               c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor sca_forward(torch::Tensor input, torch::Tensor weight,
                                 torch::Tensor bias, int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "CUDA");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  bias.is_contiguous(),
              "contig");
  TORCH_CHECK(input.dim() == 4, "NHWC");
  int N = (int)input.size(0);
  int H = (int)input.size(1);
  int W = (int)input.size(2);
  int C = (int)input.size(3);
  TORCH_CHECK(weight.dim() == 4 && weight.size(0) == C && weight.size(3) == C,
              "KRSC 1x1 weight");
  TORCH_CHECK(bias.numel() == C, "bias");
  auto pooled = torch::empty({N, C}, input.options());
  auto scale = torch::empty({N, C}, input.options());
  auto output = torch::empty_like(input);
  sca_launcher(output.data_ptr(), input.data_ptr(), weight.data_ptr(),
               bias.data_ptr(), pooled.data_ptr(), scale.data_ptr(), N, H, W, C,
               dtype_flag(input), (int)config,
               c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor pad_forward(torch::Tensor input, int64_t pad_h,
                                 int64_t pad_w, int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "bad input");
  int N = (int)input.size(0);
  int H = (int)input.size(1);
  int W = (int)input.size(2);
  int C = (int)input.size(3);
  int out_h = H + (int)pad_h;
  int out_w = W + (int)pad_w;
  auto output = torch::empty({N, out_h, out_w, C}, input.options());
  pad_launcher(output.data_ptr(), input.data_ptr(), N, H, W, C, out_h, out_w,
               dtype_flag(input), (int)config,
               c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor crop_forward(torch::Tensor input, int64_t out_h,
                                  int64_t out_w, int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "bad input");
  int N = (int)input.size(0);
  int src_h = (int)input.size(1);
  int src_w = (int)input.size(2);
  int C = (int)input.size(3);
  auto output = torch::empty({N, (int)out_h, (int)out_w, C}, input.options());
  crop_launcher(output.data_ptr(), input.data_ptr(), N, src_h, src_w, C,
                (int)out_h, (int)out_w, dtype_flag(input), (int)config,
                c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor pixel_shuffle2_forward(torch::Tensor input,
                                            int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "bad input");
  TORCH_CHECK(input.size(3) % 4 == 0, "channels % 4 == 0");
  int N = (int)input.size(0);
  int H = (int)input.size(1);
  int W = (int)input.size(2);
  int C4 = (int)input.size(3);
  int C = C4 / 4;
  auto output = torch::empty({N, H * 2, W * 2, C}, input.options());
  pixel_shuffle2_launcher(output.data_ptr(), input.data_ptr(), N, H, W, C,
                          dtype_flag(input), (int)config,
                          c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor to_half_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "CUDA contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "fp32 in");
  auto output =
      torch::empty(input.sizes(), input.options().dtype(torch::kFloat16));
  cast_f2h_launcher((__half *)output.data_ptr(), input.data_ptr<float>(),
                    (int)input.numel(),
                    c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor to_float_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "CUDA contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "fp16 in");
  auto output =
      torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));
  cast_h2f_launcher(output.data_ptr<float>(), (const __half *)input.data_ptr(),
                    (int)input.numel(),
                    c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor nchw_to_nhwc_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "NCHW input");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "fp16");
  int N = (int)input.size(0);
  int C = (int)input.size(1);
  int H = (int)input.size(2);
  int W = (int)input.size(3);
  auto output = torch::empty({N, H, W, C}, input.options());
  nchw_to_nhwc_launcher((__half *)output.data_ptr(),
                        (const __half *)input.data_ptr(), N, C, H, W,
                        c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor nhwc_to_nchw_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "NHWC input");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "fp16");
  int N = (int)input.size(0);
  int H = (int)input.size(1);
  int W = (int)input.size(2);
  int C = (int)input.size(3);
  auto output = torch::empty({N, C, H, W}, input.options());
  nhwc_to_nchw_launcher((__half *)output.data_ptr(),
                        (const __half *)input.data_ptr(), N, C, H, W,
                        c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor kcrs_to_krsc_forward(torch::Tensor weight) {
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dim() == 4,
              "KCRS weight");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "fp16");
  int K = (int)weight.size(0);
  int C = (int)weight.size(1);
  int R = (int)weight.size(2);
  int S = (int)weight.size(3);
  auto output = torch::empty({K, R, S, C}, weight.options());
  kcrs_to_krsc_launcher((__half *)output.data_ptr(),
                        (const __half *)weight.data_ptr(), K, C, R, S,
                        c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static void register_naf_ops(pybind11::module &m) {
  m.def("simple_gate", &simple_gate_forward, py::arg("input"),
        py::arg("config") = 0);
  m.def("simple_gate_bias", &simple_gate_bias_forward, py::arg("input"),
        py::arg("bias"), py::arg("config") = 0);
  m.def("residual", &residual_forward, py::arg("a"), py::arg("b"),
        py::arg("scale"), py::arg("config") = 0);
  m.def("residual_bias", &residual_bias_forward, py::arg("a"), py::arg("b"),
        py::arg("scale"), py::arg("bias"), py::arg("config") = 0);
  m.def("residual_ln", &residual_ln_forward, py::arg("a"), py::arg("b"),
        py::arg("scale"), py::arg("weight"), py::arg("bias"),
        py::arg("eps") = 1e-6, py::arg("config") = 0);
  m.def("add", &add_forward, py::arg("a"), py::arg("b"), py::arg("config") = 0);
  m.def("sca", &sca_forward, py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("config") = 0);
  m.def("pad", &pad_forward, py::arg("input"), py::arg("pad_h"),
        py::arg("pad_w"), py::arg("config") = 0);
  m.def("crop", &crop_forward, py::arg("input"), py::arg("out_h"),
        py::arg("out_w"), py::arg("config") = 0);
  m.def("pixel_shuffle2", &pixel_shuffle2_forward, py::arg("input"),
        py::arg("config") = 0);
  m.def("to_half", &to_half_forward, py::arg("input"));
  m.def("to_float", &to_float_forward, py::arg("input"));
  m.def("nchw_to_nhwc", &nchw_to_nhwc_forward, py::arg("input"));
  m.def("nhwc_to_nchw", &nhwc_to_nchw_forward, py::arg("input"));
  m.def("kcrs_to_krsc", &kcrs_to_krsc_forward, py::arg("weight"));
}

REGISTER_BINDING(naf_ops, register_naf_ops);
