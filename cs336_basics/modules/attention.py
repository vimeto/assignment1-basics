import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum
import numpy as np
from .rope import RoPE
from .qk_norm import QKNorm
from .linear import Linear

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
        use_attn_gate: bool = False,
        attn_gate_dim: int = 0,
        attn_gate_lr_mul: float = 5.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        if isinstance(device, RoPE):
            rope = device
            device = None
        if isinstance(dtype, RoPE):
            if rope is not None:
                raise ValueError("RoPE instance provided for both dtype and rope parameters")
            rope = dtype
            dtype = None

        self.dtype = dtype # <-- FIX: Added dtype
        self.rope = rope
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_attn_gate = bool(use_attn_gate)
        self.attn_gate_dim = int(attn_gate_dim) if use_attn_gate else 0
        self.attn_gate_lr_mul = float(attn_gate_lr_mul)

        var = 2.0 / float(d_model + d_model)
        std = float(np.sqrt(var))
        w_qkv = torch.empty(3 * d_model, d_model, device=self.device, dtype=self.dtype)
        nn.init.trunc_normal_(w_qkv, mean=0.0, std=std, a=-3*std, b=3*std)
        self.W_qkv = nn.Parameter(w_qkv)
        # Help Muon treat fused Q/K/V blocks independently (3-way split along rows).
        self.W_qkv._muon_partition = (int(d_model), int(d_model), int(d_model))
        self.W_qkv._muon_partition_dim = 0

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

        if self.use_attn_gate and self.attn_gate_dim > 0:
            if self.attn_gate_dim > d_model:
                raise ValueError(f"attn_gate_dim {self.attn_gate_dim} must be <= d_model {d_model}")
            gate_linear = Linear(self.attn_gate_dim, self.num_heads, device=device, dtype=dtype)
            gate_linear.linear.data.zero_()
            gate_linear.linear._optimizer_group = "vector"
            gate_linear.linear._weight_decay = 0.0
            gate_linear.linear.lr_mul = self.attn_gate_lr_mul
            self.attn_gate = gate_linear
        else:
            self.attn_gate = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        value_embed: torch.Tensor | None = None,
        sa_lambda: torch.Tensor | None = None,
        gate_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            default_scale = 1.0 / math.sqrt(self.d_k)
            scale = (self.qk_logit_scale / default_scale).view(1, self.num_heads, 1, 1).to(q.dtype)
            q = q * scale

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        if self.attn_gate is not None:
            y = rearrange(y, "b h s d -> b s h d")
            gate_source = gate_input if gate_input is not None else x
            gate_source = gate_source[..., :self.attn_gate_dim]
            gate = torch.sigmoid(self.attn_gate(gate_source)).view(B, S, self.num_heads, 1)
            y = y * gate
            y = rearrange(y, "b s h d -> b s (h d)")
        else:
            y = rearrange(y, "b h s d -> b s (h d)")

        y = y.matmul(self.W_o.t())
        return y


def attention(
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference scaled dot-product attention used in unit tests."""
    q = Q
    k = K
    v = V
    dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dim)
    if mask is not None:
        mask_bool = mask.to(torch.bool)
        scores = scores.masked_fill(~mask_bool, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.matmul(weights, v)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Reference softmax used in unit tests."""
    return torch.softmax(x, dim=dim)
