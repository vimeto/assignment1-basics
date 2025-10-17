import torch
import torch.nn as nn
from .linear import Linear
from einops import reduce

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_res = self.w1(x)
        w1_silu = torch.sigmoid(w1_res) * w1_res

        w_13 = w1_silu * self.w3(x)

        return self.w2(w_13)





