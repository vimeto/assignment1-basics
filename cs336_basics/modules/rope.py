import torch
import torch.nn as nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        i_s = torch.arange(max_seq_len).unsqueeze(1).to(device)
        k_s = torch.arange(d_k // 2).unsqueeze(0).to(device)

        thetas = i_s / (theta ** (2 * k_s / self.d_k))
        # the shape of these are (max_seq_length, d // 2)
        sin = torch.sin(thetas)
        cos = torch.cos(thetas)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)


    def forward(self,  x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # first, we take the rows with the given token positions
        # token pos are shape (..., seq_len)
        # input x is shape (..., seq_len, d_k)

        # Validate token positions more carefully
        if token_positions.numel() == 0:
            raise ValueError("token_positions is empty")

        # Check for valid range without triggering CUDA errors
        min_pos = token_positions.min().item()
        max_pos = token_positions.max().item()

        if min_pos < 0:
            raise ValueError(f"token position {min_pos} is negative")
        if max_pos >= self.max_seq_len:
            raise ValueError(f"token position {max_pos} exceeds max_seq_len {self.max_seq_len}")

        # Validate no NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"RoPE received NaN or Inf in input x")

        # shape (batch, seq_len, d_k // 2)
        chosen_sin = self.sin[token_positions]
        chosen_cos = self.cos[token_positions]

        pairs = rearrange(x, "... seq_len (pairs two) -> ... seq_len pairs two", two=2)
        # shape (batch, seq_len, d_k // 2)
        x_even = pairs[..., 0]
        x_odd = pairs[..., 1]

        # shape (batch, seq_len, d_k // 2)
        xx_even = x_even * chosen_cos - x_odd * chosen_sin
        xx_odd = x_even * chosen_sin + x_odd * chosen_cos

        # shape (batch, seq_len, d_k // 2, 2)
        xx = torch.stack([xx_even, xx_odd], dim=-1)
        y = rearrange(xx, "... seq_len pairs two -> ... seq_len (pairs two)")

        return y
