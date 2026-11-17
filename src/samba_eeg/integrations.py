from __future__ import annotations

import torch
from torch import nn

from .modeling_samba import SambaModel


class SambaNeuralBenchBackbone(nn.Module):
    """NeuralBench adapter returning a standard ``[B, T, D]`` tensor.

    NeuralBench initializes lazy probe layers with a CPU-only shape pass before
    Lightning moves the full module to CUDA.  Released SAMBA checkpoints use
    CUDA-only Mamba-2 kernels, so that shape pass returns a zero feature
    tensor with the exact encoder shape.  Real CUDA batches always run SAMBA.
    """

    def __init__(self, model: SambaModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if x.device.type == "cpu" and self.model.config.mixer_backend == "mamba2":
            if x.ndim != 3 or x.shape[-1] < 8:
                raise ValueError("signals must have shape [B, C, T] with T >= 8")
            return x.new_zeros(
                x.shape[0],
                x.shape[-1] // 8,
                self.model.config.bottleneck_channels,
            )
        return self.model.encode(x, channel_positions, channel_mask)
