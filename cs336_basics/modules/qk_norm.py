from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class QKNorm(nn.Module):
    """
    Per-head QK normalization with learnable gamma, as used in modern LLMs.
    x: (B, H, S, D). We normalize along D, then scale with gamma[H, D].
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype=None,
        num_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.num_heads = num_heads
        shape = (num_heads, dim) if num_heads is not None else (dim,)
        self.weight = nn.Parameter(torch.ones(shape, device=device, dtype=dtype))
        self.weight._optimizer_group = "vector"
        self.weight._weight_decay = 0.0
        self.weight.wd_mul = 0.0
        self.weight._weight_decay = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, D)
        # use torch RMSNorm (weight applied manually for per-head gamma)
        y = F.rms_norm(x, (self.dim,), weight=None, eps=self.eps)
        if self.num_heads is not None:
            return y * self.weight.view(1, self.num_heads, 1, self.dim)
        else:
            return y * self.weight.view(1, 1, 1, self.dim)
