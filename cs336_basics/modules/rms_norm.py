import torch
import torch.nn as nn
from einops import reduce

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
        mean_sq = reduce(torch.square(x), "... d_model -> ... 1", "mean")
        sq = torch.sqrt(mean_sq + self.eps)
        y = (x / sq) * self.gi
        return y



