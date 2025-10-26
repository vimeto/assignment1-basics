from __future__ import annotations
import torch
import torch.nn as nn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, D)
        # do denom in fp32 for stability, then cast back
        x32 = x.to(torch.float32)
        denom = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x / denom.to(x.dtype)
        if self.num_heads is not None:
            return y * self.weight.view(1, self.num_heads, 1, self.dim)
        else:
            return y * self.weight.view(1, 1, 1, self.dim)
