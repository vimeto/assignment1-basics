import torch
import torch.nn as nn

def _reduce_mean_sq(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        gs = torch.ones(d_model, dtype=self.dtype, device=self.device)
        self.gi = nn.Parameter(gs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = _reduce_mean_sq(x)
        sq = torch.sqrt(mean_sq + self.eps)
        normed = torch.mul(x, torch.reciprocal(sq))
        return normed * self.gi

