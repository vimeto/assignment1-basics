import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        dtype=None, # <-- FIX: Added dtype
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
        self.dtype = dtype # <-- FIX: Added dtype
        self.rope = rope
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm

        var = 2.0 / float(d_model + d_model)
        std = float(np.sqrt(var))
        w_qkv = torch.empty(3 * d_model, d_model, device=self.device, dtype=self.dtype)
        nn.init.trunc_normal_(w_qkv, mean=0.0, std=std, a=-3*std, b=3*std)
        self.W_qkv = nn.Parameter(w_qkv)

        # Zero-init W_o (stabilizes early training; widely used in speedruns)
        w_o = torch.zeros(d_model, d_model, device=self.device, dtype=self.dtype)
        self.W_o = nn.Parameter(w_o)

        if self.use_qk_norm:
            self.q_norm = QKNorm(
                self.d_k,
                eps=qk_norm_eps,
                device=device,
                dtype=dtype,
                num_heads=self.num_heads,
            )
            self.k_norm = QKNorm(
                self.d_k,
                eps=qk_norm_eps,
                device=device,
                dtype=dtype,
                num_heads=self.num_heads,
            )
            # QK-Norm replaces 1/sqrt(d_k) with a learnable scale. Start at that default.
            init_scale = 1.0 / math.sqrt(self.d_k)
            self.qk_logit_scale = nn.Parameter(
                torch.full((self.num_heads, 1, 1), init_scale, device=device, dtype=dtype)
            )
            self.qk_logit_scale._optimizer_group = "vector"
            self.qk_logit_scale._weight_decay = 0.0
        else:
            self.q_norm = None
            self.k_norm = None
            self.qk_logit_scale = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, value_embed: torch.Tensor | None = None, sa_lambda: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, S, d_model)
        B, S, _ = x.shape

        qkv = x.matmul(self.W_qkv.t())
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.use_rope and self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Mix value embeddings (nanoGPT-style): v = sa_lambdas[0] * v + sa_lambdas[1] * ve
        if value_embed is not None and sa_lambda is not None:
            ve = rearrange(value_embed, "b s (h d) -> b h s d", h=self.num_heads)
            v = sa_lambda[0] * v + sa_lambda[1] * ve
        elif sa_lambda is not None and value_embed is None:
            # If no value embed but lambda provided, just scale v by lambda[0]
            v = sa_lambda[0] * v

        if self.qk_logit_scale is not None:
            # We provide our own scale (initialized to 1/sqrt(d_k)), so keep torch's scale at 1.0.
            q = q * self.qk_logit_scale.view(1, self.num_heads, 1, 1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0, scale=1.0)
        else:
            # Fall back to PyTorch's default 1/sqrt(d_k) scaling.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        y = rearrange(y, "b h s d -> b s (h d)")
        y = y.matmul(self.W_o.t())
        return y
