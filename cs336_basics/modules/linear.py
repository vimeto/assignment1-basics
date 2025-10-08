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

        var = 2 / (in_features + out_features)
        self.std = np.sqrt(var)
        weight = torch.empty(out_features, in_features, dtype=self.dtype, device=self.device)
        nn.init.trunc_normal_(weight, mean=0.0, std=var, a=-3*self.std, b=3*self.std)
        self.linear = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.linear, "... in_features, out_features in_features -> ... out_features")
