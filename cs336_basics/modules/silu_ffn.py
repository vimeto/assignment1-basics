"""
Standard feed-forward network with SiLU activation.
Used as an alternative to SwiGLU in ablation studies.
"""

import torch
import torch.nn as nn
from .linear import Linear


class SiLU_FFN(nn.Module):
    """
    Standard two-layer feed-forward network with SiLU activation.

    This is a simpler alternative to SwiGLU for ablation studies.
    Architecture: Linear -> SiLU -> Linear

    Args:
        d_model: Dimensionality of input and output
        d_ff: Dimensionality of the hidden layer
        device: Device to place parameters on
        dtype: Data type for parameters
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network: W2(SiLU(W1(x)))

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        h = self.w1(x)
        h = torch.sigmoid(h) * h
        return self.w2(h)
