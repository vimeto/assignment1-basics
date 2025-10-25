import torch
import torch.nn as nn
import numpy as np
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        var = 2.0 / float(in_features + out_features)
        std = float(np.sqrt(var))
        self.std = std
        weight = torch.empty(out_features, in_features, dtype=self.dtype, device=self.device)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        self.linear = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.linear.t())
