import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """CUDA-optimized NAFNet (implement via kernels/ + cuda_extension)."""

    def __init__(
        self,
        img_channel=1,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=None,
        dec_blk_nums=None,
    ):
        super().__init__()
        # Keep constructor signature identical to Model for verification/profiling.
        # Implement parameters + cuda_extension ops in the agent loop.

    def forward(self, x):
        raise NotImplementedError("Implement ModelNew with custom CUDA ops (see AGENT_PROMPT.md / SKILL.md).")
