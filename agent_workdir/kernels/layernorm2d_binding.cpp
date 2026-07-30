#include "../binding_registry.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

extern "C" void layernorm2d_launcher(void *out, const void *in,
                                     const void *weight, const void *bias,
                                     int N, int C, int H, int W, float eps,
                                     int dtype, int config,
                                     cudaStream_t stream);

static torch::Tensor layernorm2d_forward(torch::Tensor input,
                                         torch::Tensor weight,
                                         torch::Tensor bias, double eps,
                                         int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
              "CUDA required");
  TORCH_CHECK(input.is_contiguous(), "input contiguous");
  TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(),
              "weight/bias contiguous");
  TORCH_CHECK(input.dim() == 4, "NCHW");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 ||
                  input.scalar_type() == torch::kFloat16,
              "fp32/fp16");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype");
  TORCH_CHECK(bias.scalar_type() == input.scalar_type(), "bias dtype");
  TORCH_CHECK(weight.numel() == input.size(1) && bias.numel() == input.size(1),
              "C mismatch");

  auto output = torch::empty_like(input);
  int dtype = input.scalar_type() == torch::kFloat16 ? 1 : 0;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  layernorm2d_launcher(output.data_ptr(), input.data_ptr(), weight.data_ptr(),
                       bias.data_ptr(), (int)input.size(0), (int)input.size(1),
                       (int)input.size(2), (int)input.size(3), (float)eps,
                       dtype, (int)config, stream);
  return output;
}

static void register_layernorm2d(pybind11::module &m) {
  m.def("layernorm2d", &layernorm2d_forward, "Channel LayerNorm2d",
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("eps") = 1e-6, py::arg("config") = 0);
}

REGISTER_BINDING(layernorm2d, register_layernorm2d);
