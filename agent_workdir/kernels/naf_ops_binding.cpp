#include "../binding_registry.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

extern "C" void simple_gate_launcher(void *, const void *, int, int, int, int,
                                     int, cudaStream_t);
extern "C" void residual_launcher(void *, const void *, const void *,
                                  const void *, int, int, int, int, int,
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
  TORCH_CHECK(input.size(1) % 2 == 0, "channels must be even");
  auto out_sizes = input.sizes().vec();
  out_sizes[1] = input.size(1) / 2;
  auto output = torch::empty(out_sizes, input.options());
  int N = (int)input.size(0);
  int C = (int)(input.size(1) / 2);
  int HW = (int)(input.size(2) * input.size(3));
  simple_gate_launcher(output.data_ptr(), input.data_ptr(), N, C, HW,
                       dtype_flag(input), (int)config,
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
  int C = (int)a.size(1);
  int HW = (int)(a.size(2) * a.size(3));
  residual_launcher(output.data_ptr(), a.data_ptr(), b.data_ptr(),
                    scale.data_ptr(), (int)a.size(0), C, HW, dtype_flag(a),
                    (int)config, c10::cuda::getCurrentCUDAStream().stream());
  return output;
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
  TORCH_CHECK(input.dim() == 4, "NCHW");
  int N = (int)input.size(0), C = (int)input.size(1), H = (int)input.size(2),
      W = (int)input.size(3);
  TORCH_CHECK(weight.dim() == 4 && weight.size(0) == C && weight.size(1) == C,
              "1x1 weight");
  TORCH_CHECK(bias.numel() == C, "bias");
  auto pooled = torch::empty({N, C}, input.options());
  auto scale = torch::empty({N, C}, input.options());
  auto output = torch::empty_like(input);
  // weight as (C,C) from (C,C,1,1)
  sca_launcher(output.data_ptr(), input.data_ptr(), weight.data_ptr(),
               bias.data_ptr(), pooled.data_ptr(), scale.data_ptr(), N, C, H, W,
               dtype_flag(input), (int)config,
               c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor pad_forward(torch::Tensor input, int64_t pad_h,
                                 int64_t pad_w, int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "bad input");
  int N = (int)input.size(0), C = (int)input.size(1), H = (int)input.size(2),
      W = (int)input.size(3);
  int out_h = H + (int)pad_h, out_w = W + (int)pad_w;
  auto output = torch::empty({N, C, out_h, out_w}, input.options());
  pad_launcher(output.data_ptr(), input.data_ptr(), N, C, H, W, out_h, out_w,
               dtype_flag(input), (int)config,
               c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor crop_forward(torch::Tensor input, int64_t out_h,
                                  int64_t out_w, int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "bad input");
  int N = (int)input.size(0), C = (int)input.size(1);
  int src_h = (int)input.size(2), src_w = (int)input.size(3);
  auto output = torch::empty({N, C, (int)out_h, (int)out_w}, input.options());
  crop_launcher(output.data_ptr(), input.data_ptr(), N, C, src_h, src_w,
                (int)out_h, (int)out_w, dtype_flag(input), (int)config,
                c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static torch::Tensor pixel_shuffle2_forward(torch::Tensor input,
                                            int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 4,
              "bad input");
  TORCH_CHECK(input.size(1) % 4 == 0, "channels % 4 == 0");
  int N = (int)input.size(0), C4 = (int)input.size(1), H = (int)input.size(2),
      W = (int)input.size(3);
  int C = C4 / 4;
  auto output = torch::empty({N, C, H * 2, W * 2}, input.options());
  pixel_shuffle2_launcher(output.data_ptr(), input.data_ptr(), N, C, H, W,
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

static void register_naf_ops(pybind11::module &m) {
  m.def("simple_gate", &simple_gate_forward, py::arg("input"),
        py::arg("config") = 0);
  m.def("residual", &residual_forward, py::arg("a"), py::arg("b"),
        py::arg("scale"), py::arg("config") = 0);
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
}

REGISTER_BINDING(naf_ops, register_naf_ops);
