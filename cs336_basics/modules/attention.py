import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange
from .rope import RoPE
from .qk_norm import QKNorm

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x_max = x.amax(dim=i, keepdim=True)
    shifted = x - x_max
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=i, keepdim=True)

def attention(K: torch.Tensor, Q: torch.Tensor, V: torch.Tensor, mask = None) -> torch.Tensor:
    # K and Q (batch_size, ..., seq_len, d_k)
    # V (batch_size, ..., seq_len, d_v)
    # mask (seq_len, seq_len)
    dim = K.shape[-1]
    num = einsum(Q, K, "batch_size ... seq_len_1 d_k, batch_size ... seq_len_2 d_k -> batch_size ... seq_len_1 seq_len_2")
    num = num / np.sqrt(dim)
    if mask is not None:
        # when mask is bool, keep it on the same device and dtype-safe for cudagraphs
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        num = num.masked_fill(~mask, float("-inf"))

    num = softmax(num, -1)
    res = einsum(num, V, "batch_size ... seq_len_1 seq_len_2, batch_size ... seq_len_2 d_v -> batch_size ... seq_len_1 d_v")
    return res


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        device=None,
        rope=None,
        use_rope=True,
        use_qk_norm=False,
        qk_norm_eps=1e-6,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.d_k = d_model // num_heads
        self.rope = rope
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm

        var_dk = 2 / (self.d_k + d_model)
        std_dk = np.sqrt(var_dk)
        w_qkv = torch.empty(d_model * 3, d_model, device=self.device)
        nn.init.trunc_normal_(w_qkv, mean=0.0, std=var_dk, a=-3*std_dk, b=3*std_dk)
        self.W_qkv = nn.Parameter(w_qkv)

        # w_k = torch.empty(d_model, d_model, device=self.device)
        # nn.init.trunc_normal_(w_k, mean=0.0, std=var_dk, a=-3*std_dk, b=3*std_dk)
        # self.W_k = nn.Parameter(w_k)

        # w_v = torch.empty(d_model, d_model, device=self.device)
        # nn.init.trunc_normal_(w_v, mean=0.0, std=var_dk, a=-3*std_dk, b=3*std_dk)
        # self.W_v = nn.Parameter(w_v)

        var_o = torch.tensor(1 / d_model).to(device)
        std_o = torch.sqrt(var_o)
        w_o = torch.empty(d_model, d_model, device=self.device)
        nn.init.trunc_normal_(w_o, mean=0.0, std=var_o, a=-3*std_o, b=3*std_o)
        self.W_o = nn.Parameter(w_o)

        if self.use_qk_norm:
            self.q_norm = QKNorm(self.d_k, eps=qk_norm_eps, device=device)
            self.k_norm = QKNorm(self.d_k, eps=qk_norm_eps, device=device)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # in_features (... sequence_length d_in)

        seq_len = x.shape[-2]
        d = x.device

        # Check input for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"Attention received NaN or Inf in input x at seq_len={seq_len}")

        qkv_flat = einsum(x, self.W_qkv, "... seq d_in, d_out d_in -> ... seq d_out")

        # Check after QKV projection
        if torch.isnan(qkv_flat).any() or torch.isinf(qkv_flat).any():
            raise ValueError(f"NaN or Inf after QKV projection at seq_len={seq_len}")

        q_flat, k_flat, v_flat = qkv_flat.chunk(3, dim=-1)

        q = rearrange(q_flat, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        k = rearrange(k_flat, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        v = rearrange(v_flat, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)

        if self.use_qk_norm and self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=d))

        att = attention(k, q, v, mask)
        att = rearrange(att, "... heads sequence_length d_k -> ... sequence_length (heads d_k)")
        y = einsum(att, self.W_o, "... sequence_length d_model, d_out d_model -> ... sequence_length d_out")

        return y
