"""
Identity normalization layer that passes input through unchanged.
Used as a no-op replacement for RMSNorm in ablation studies.
"""

import torch
import torch.nn as nn


class IdentityNorm(nn.Module):
    """
    Identity normalization layer - no-op that returns input unchanged.

    This is used as a drop-in replacement for RMSNorm in ablation studies
    to test the effect of removing normalization while maintaining the
    same model structure.

    Args:
        d_model: Dimensionality of the input (unused, for API compatibility)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through unchanged.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            The input tensor x unchanged
        """
        return x
