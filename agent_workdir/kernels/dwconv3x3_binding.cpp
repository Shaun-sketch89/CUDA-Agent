#include "../binding_registry.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

extern "C" void dwconv3x3_launcher(void *out, const void *in,
                                   const void *weight, const void *bias, int N,
                                   int C, int H, int W, int config,
                                   cudaStream_t stream);

static torch::Tensor dwconv3x3_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       c10::optional<torch::Tensor> bias,
                                       int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "CUDA");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "fp16");
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "NCHW/KCRS");
  int N = (int)input.size(0), C = (int)input.size(1), H = (int)input.size(2),
      W = (int)input.size(3);
  TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1 &&
                  weight.size(2) == 3 && weight.size(3) == 3,
              "depthwise 3x3 weight");

  const void *bias_ptr = nullptr;
  torch::Tensor bias_contig;
  if (bias.has_value() && bias.value().defined()) {
    bias_contig = bias.value().contiguous();
    TORCH_CHECK(bias_contig.scalar_type() == torch::kFloat16, "bias fp16");
    TORCH_CHECK(bias_contig.numel() == C, "bias C");
    bias_ptr = bias_contig.data_ptr();
  }

  auto output = torch::empty_like(input);
  dwconv3x3_launcher(output.data_ptr(), input.data_ptr(), weight.data_ptr(),
                     bias_ptr, N, C, H, W, (int)config,
                     c10::cuda::getCurrentCUDAStream().stream());
  return output;
}

static void register_dwconv3x3(pybind11::module &m) {
  m.def("dwconv3x3", &dwconv3x3_forward, py::arg("input"), py::arg("weight"),
        py::arg("bias") = c10::nullopt, py::arg("config") = 0);
}

REGISTER_BINDING(dwconv3x3, register_dwconv3x3);
