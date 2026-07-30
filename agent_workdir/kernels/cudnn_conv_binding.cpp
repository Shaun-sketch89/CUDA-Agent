#include "../binding_registry.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

extern "C" void
cudnn_conv2d_launcher(void *output, const void *input, const void *weight,
                      const void *bias, int n, int c, int h, int w, int k,
                      int r, int s, int pad_h, int pad_w, int stride_h,
                      int stride_w, int dilation_h, int dilation_w, int groups,
                      int dtype, int nhwc, int config, cudaStream_t stream);

extern "C" void bias_add_launcher(void *out, const void *bias, int N, int H,
                                  int W, int C, int dtype, int nhwc, int config,
                                  cudaStream_t stream);

static torch::Tensor
cudnn_conv2d_forward(torch::Tensor input, torch::Tensor weight,
                     c10::optional<torch::Tensor> bias, int64_t pad_h,
                     int64_t pad_w, int64_t stride_h, int64_t stride_w,
                     int64_t dilation_h, int64_t dilation_w, int64_t groups,
                     int64_t nhwc = 1, int64_t config = 0) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "input/weight must be CUDA");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
              "input/weight must be contiguous");
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "expected 4D tensors");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "dtype mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 ||
                  input.scalar_type() == torch::kFloat16,
              "only fp32/fp16");

  int dtype = input.scalar_type() == torch::kFloat16 ? 1 : 0;
  int n, c, h, w, k, r, s;
  if (nhwc) {
    n = (int)input.size(0);
    h = (int)input.size(1);
    w = (int)input.size(2);
    c = (int)input.size(3);
    k = (int)weight.size(0);
    r = (int)weight.size(1);
    s = (int)weight.size(2);
    TORCH_CHECK(weight.size(3) * groups == c, "weight/groups mismatch");
  } else {
    n = (int)input.size(0);
    c = (int)input.size(1);
    h = (int)input.size(2);
    w = (int)input.size(3);
    k = (int)weight.size(0);
    r = (int)weight.size(2);
    s = (int)weight.size(3);
    TORCH_CHECK(weight.size(1) * groups == c, "weight/groups mismatch");
  }

  int out_h =
      (h + 2 * (int)pad_h - (int)dilation_h * (r - 1) - 1) / (int)stride_h + 1;
  int out_w =
      (w + 2 * (int)pad_w - (int)dilation_w * (s - 1) - 1) / (int)stride_w + 1;

  torch::Tensor output;
  if (nhwc)
    output = torch::empty({n, out_h, out_w, k}, input.options());
  else
    output = torch::empty({n, k, out_h, out_w}, input.options());

  const void *bias_ptr = nullptr;
  torch::Tensor bias_contig;
  if (bias.has_value() && bias.value().defined()) {
    bias_contig = bias.value().contiguous();
    TORCH_CHECK(bias_contig.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias_contig.scalar_type() == input.scalar_type(), "bias dtype");
    TORCH_CHECK(bias_contig.numel() == k, "bias size");
    bias_ptr = bias_contig.data_ptr();
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  cudnn_conv2d_launcher(output.data_ptr(), input.data_ptr(), weight.data_ptr(),
                        bias_ptr, n, c, h, w, k, r, s, (int)pad_h, (int)pad_w,
                        (int)stride_h, (int)stride_w, (int)dilation_h,
                        (int)dilation_w, (int)groups, dtype, (int)nhwc,
                        (int)config, stream);
  if (bias_ptr) {
    if (nhwc)
      bias_add_launcher(output.data_ptr(), bias_ptr, n, out_h, out_w, k, dtype,
                        1, (int)config, stream);
    else
      bias_add_launcher(output.data_ptr(), bias_ptr, n, out_h, out_w, k, dtype,
                        0, (int)config, stream);
  }
  return output;
}

static void register_cudnn_conv(pybind11::module &m) {
  m.def("cudnn_conv2d", &cudnn_conv2d_forward, "cuDNN Conv2d", py::arg("input"),
        py::arg("weight"), py::arg("bias") = c10::nullopt, py::arg("pad_h") = 0,
        py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1,
        py::arg("dilation_h") = 1, py::arg("dilation_w") = 1,
        py::arg("groups") = 1, py::arg("nhwc") = 1, py::arg("config") = 0);
}

REGISTER_BINDING(cudnn_conv, register_cudnn_conv);
