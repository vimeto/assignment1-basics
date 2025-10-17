import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange
from .attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .swiglu import SwiGLU
from .identity_norm import IdentityNorm
from .silu_ffn import SiLU_FFN

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: None = None,
        rope: None = None,
        use_rope: bool = True,
        use_pre_norm: bool = True,
        use_post_norm: bool = False,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.device = device
        self.use_rope = use_rope
        self.use_pre_norm = use_pre_norm
        self.use_post_norm = use_post_norm

        # Create normalization layers (additive support)
        norm_class = RMSNorm if use_rmsnorm else IdentityNorm

        # Pre-norm layers
        self.pre_attn_norm = norm_class(d_model, device=device) if use_pre_norm else IdentityNorm(d_model)
        self.pre_ffn_norm = norm_class(d_model, device=device) if use_pre_norm else IdentityNorm(d_model)

        # Post-norm layers
        self.post_attn_norm = norm_class(d_model, device=device) if use_post_norm else IdentityNorm(d_model)
        self.post_ffn_norm = norm_class(d_model, device=device) if use_post_norm else IdentityNorm(d_model)

        # Attention and FFN
        self.attn = MultiHeadAttention(d_model, num_heads, device, rope, use_rope=use_rope)
        self.ffn = SwiGLU(d_model, d_ff, device) if use_swiglu else SiLU_FFN(d_model, d_ff, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)

        attn_input = self.pre_attn_norm(x)
        attn_output = self.attn(attn_input, token_positions)
        attn_output = self.post_attn_norm(attn_output)
        y = x + attn_output

        ffn_input = self.pre_ffn_norm(y)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.post_ffn_norm(ffn_output)
        y = y + ffn_output

        return y



