from __future__ import annotations

import torch
import torch.nn as nn


class QKNorm(nn.Module):
    """Normalize queries and keys with a learned per-channel gain.

    The normalization divides each vector by its root mean square, similar to RMSNorm,
    and then applies a learned gain parameter. This helps stabilize attention scores
    at the beginning of training while still allowing the model to learn appropriate
    magnitudes over time.
    """

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.weight._optimizer_group = "vector"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / denom * self.weight

