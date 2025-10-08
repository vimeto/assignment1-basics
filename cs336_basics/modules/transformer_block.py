import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange
from .attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: None = None, rope: None = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.device = device

        self.attn = MultiHeadAttention(d_model, num_heads, device, rope)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device)
        self.ln2 = RMSNorm(d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.shape[-2]

        token_positions = torch.arange(seq_len, device=self.device)
        y = x + self.attn(self.ln1(x), token_positions)

        y = y + self.ffn(self.ln2(y))

        return y



