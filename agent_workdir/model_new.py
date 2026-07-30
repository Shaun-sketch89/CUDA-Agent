"""CUDA-optimized NAFNet (FP16) via custom extension ops."""

import os

import torch

# Ensure PyTorch / CUDA DLLs are discoverable on Windows before loading the extension.
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if hasattr(os, "add_dll_directory") and os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
_cuda_path = os.environ.get("CUDA_PATH")
if _cuda_path and hasattr(os, "add_dll_directory"):
    _cuda_bin = os.path.join(_cuda_path, "bin")
    if os.path.isdir(_cuda_bin):
        os.add_dll_directory(_cuda_bin)

import torch.nn as nn
import cuda_extension


def _conv2d(x, weight, bias, pad_h=0, pad_w=0, stride_h=1, stride_w=1, groups=1):
    return cuda_extension.cudnn_conv2d(x, weight, bias, pad_h, pad_w, stride_h, stride_w, 1, 1, groups, 0)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return cuda_extension.layernorm2d(x, self.weight_h, self.bias_h, self.eps, 0)


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self._c = c
        self._dw = dw_channel

    def _prep_half(self):
        self.conv1_w = self.conv1.weight.detach().half().contiguous()
        self.conv1_b = self.conv1.bias.detach().half().contiguous()
        self.conv2_w = self.conv2.weight.detach().half().contiguous()
        self.conv2_b = self.conv2.bias.detach().half().contiguous()
        self.conv3_w = self.conv3.weight.detach().half().contiguous()
        self.conv3_b = self.conv3.bias.detach().half().contiguous()
        self.conv4_w = self.conv4.weight.detach().half().contiguous()
        self.conv4_b = self.conv4.bias.detach().half().contiguous()
        self.conv5_w = self.conv5.weight.detach().half().contiguous()
        self.conv5_b = self.conv5.bias.detach().half().contiguous()
        sca_conv = self.sca[1]
        self.sca_w = sca_conv.weight.detach().half().contiguous()
        self.sca_b = sca_conv.bias.detach().half().contiguous()
        self.norm1.weight_h = self.norm1.weight.detach().half().contiguous()
        self.norm1.bias_h = self.norm1.bias.detach().half().contiguous()
        self.norm2.weight_h = self.norm2.weight.detach().half().contiguous()
        self.norm2.bias_h = self.norm2.bias.detach().half().contiguous()
        self.beta_h = self.beta.detach().half().contiguous()
        self.gamma_h = self.gamma.detach().half().contiguous()

    def forward(self, inp):
        x = self.norm1(inp)
        x = _conv2d(x, self.conv1_w, self.conv1_b)
        x = cuda_extension.dwconv3x3(x, self.conv2_w, self.conv2_b, 0)
        x = cuda_extension.simple_gate(x, 0)
        x = cuda_extension.sca(x, self.sca_w, self.sca_b, 0)
        x = _conv2d(x, self.conv3_w, self.conv3_b)
        y = cuda_extension.residual(inp, x, self.beta_h, 0)

        x = self.norm2(y)
        x = _conv2d(x, self.conv4_w, self.conv4_b)
        x = cuda_extension.simple_gate(x, 0)
        x = _conv2d(x, self.conv5_w, self.conv5_b)
        return cuda_extension.residual(y, x, self.gamma_h, 0)


class ModelNew(nn.Module):
    """NAFNet optimized with cuDNN FP16 conv + fused custom kernels."""

    def __init__(
        self,
        img_channel=1,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=None,
        dec_blk_nums=None,
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [1, 2, 2, 4]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 1, 1]

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, 1, 3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)
        self._half_ready = False

    def _ensure_half(self):
        if self._half_ready:
            return
        self.intro_w = self.intro.weight.detach().half().contiguous()
        self.intro_b = self.intro.bias.detach().half().contiguous()
        self.ending_w = self.ending.weight.detach().half().contiguous()
        self.ending_b = self.ending.bias.detach().half().contiguous()
        self.down_w = []
        self.down_b = []
        for d in self.downs:
            self.down_w.append(d.weight.detach().half().contiguous())
            self.down_b.append(d.bias.detach().half().contiguous())
        self.up_w = []
        for u in self.ups:
            self.up_w.append(u[0].weight.detach().half().contiguous())
        for enc in self.encoders:
            for blk in enc:
                blk._prep_half()
        for blk in self.middle_blks:
            blk._prep_half()
        for dec in self.decoders:
            for blk in dec:
                blk._prep_half()
        self._half_ready = True

    def forward(self, inp):
        self._ensure_half()
        # Shapes only (allowed); compute via extension
        B = inp.shape[0]
        C = inp.shape[1]
        H = inp.shape[2]
        W = inp.shape[3]

        x = cuda_extension.to_half(inp)
        mod_pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = cuda_extension.pad(x, mod_pad_h, mod_pad_w, 0)
        inp_pad = x

        x = _conv2d(x, self.intro_w, self.intro_b, pad_h=1, pad_w=1)
        encs = []
        for i, encoder in enumerate(self.encoders):
            for blk in encoder:
                x = blk(x)
            encs.append(x)
            x = _conv2d(x, self.down_w[i], self.down_b[i], pad_h=0, pad_w=0, stride_h=2, stride_w=2)

        for blk in self.middle_blks:
            x = blk(x)

        for i, decoder in enumerate(self.decoders):
            x = _conv2d(x, self.up_w[i], None)
            x = cuda_extension.pixel_shuffle2(x, 0)
            x = cuda_extension.add(x, encs[-(i + 1)], 0)
            for blk in decoder:
                x = blk(x)

        x = _conv2d(x, self.ending_w, self.ending_b, pad_h=1, pad_w=1)
        x = cuda_extension.add(x, inp_pad, 0)
        x = cuda_extension.crop(x, H, W, 0)
        return cuda_extension.to_float(x)
