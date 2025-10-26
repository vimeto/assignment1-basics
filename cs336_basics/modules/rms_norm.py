import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        gs = torch.ones(d_model, dtype=self.dtype, device=self.device)
        self.gi = nn.Parameter(gs)
        self.gi._optimizer_group = "vector"
        elf.gi._weight_decay = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = (x * x).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        return x * inv_rms * self.gi
