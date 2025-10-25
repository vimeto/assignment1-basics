from __future__ import annotations
import torch
import torch.nn as nn

class QKNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | None = None, dtype=None) -> None: # <-- FIX: Added dtype
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.weight._optimizer_group = "vector"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / denom * self.weight
