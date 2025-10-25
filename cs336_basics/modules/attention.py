import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import numpy as np
from .rope import RoPE
from .qk_norm import QKNorm

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        rope: RoPE | None = None,
        use_rope: bool = True,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.rope = rope
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm

        var = 2.0 / float(d_model + d_model)
        std = float(np.sqrt(var))
        w_qkv = torch.empty(3 * d_model, d_model, device=self.device)
        nn.init.trunc_normal_(w_qkv, mean=0.0, std=std, a=-3*std, b=3*std)
        self.W_qkv = nn.Parameter(w_qkv)

        # Output projection: standard truncated normal (NOT zero)
        var_o = 2.0 / float(d_model + d_model)
        std_o = float(np.sqrt(var_o))
        w_o = torch.empty(d_model, d_model, device=self.device)
        nn.init.trunc_normal_(w_o, mean=0.0, std=std_o, a=-3*std_o, b=3*std_o)
        self.W_o = nn.Parameter(w_o)

        # Optional QK-Norm
        if self.use_qk_norm:
            self.q_norm = QKNorm(self.d_k, eps=qk_norm_eps, device=device)
            self.k_norm = QKNorm(self.d_k, eps=qk_norm_eps, device=device)
        else:
            self.q_norm = None
            self.k_norm = None

        self.v_bias = nn.Parameter(torch.zeros(self.num_heads, self.d_k, device=device))
        self.v_bias_gate = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, S, d_model)
        B, S, _ = x.shape

        qkv = einsum(x, self.W_qkv, "... seq d_in, d_out d_in -> ... seq d_out")
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # tiny value "residual"
        if self.v_bias is not None:
            v = v + torch.tanh(self.v_bias_gate).unsqueeze(0).unsqueeze(2) * self.v_bias

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = rearrange(y, "b h s d -> b s (h d)")
        y = einsum(y, self.W_o, "... s d_model, d_out d_model -> ... s d_out")
        return y
