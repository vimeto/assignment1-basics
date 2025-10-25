import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .identity_norm import IdentityNorm
from .silu_ffn import SiLU_FFN  # now ReLU²

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        rope=None,
        use_rope: bool = True,
        use_pre_norm: bool = True,
        use_post_norm: bool = False,
        use_rmsnorm: bool = True,
        use_swiglu: bool = False,   # ReLU² path
        use_qk_norm: bool = True,
        layer_idx: int = 0,
        num_layers: int = 1,
    ):
        super().__init__()
        norm = RMSNorm if use_rmsnorm else IdentityNorm
        self.pre_attn_norm = norm(d_model, device=device) if use_pre_norm else IdentityNorm(d_model)
        self.pre_ffn_norm = norm(d_model, device=device) if use_pre_norm else IdentityNorm(d_model)
        self.post_attn_norm = norm(d_model, device=device) if use_post_norm else IdentityNorm(d_model)
        self.post_ffn_norm = norm(d_model, device=device) if use_post_norm else IdentityNorm(d_model)

        self.attn = MultiHeadAttention(
            d_model, num_heads, device, rope, use_rope=use_rope, use_qk_norm=use_qk_norm
        )
        self.ffn = SiLU_FFN(d_model, d_ff, device=device)  # ReLU²

        # Residual scales (LayerScale-like), start at 1.0
        self.resid_attn_scale = nn.Parameter(torch.ones(1, 1, d_model, device=device))
        self.resid_ffn_scale  = nn.Parameter(torch.ones(1, 1, d_model, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[-2]
        pos = torch.arange(s, device=x.device)

        a_out = self.attn(self.pre_attn_norm(x), pos)
        a_out = self.post_attn_norm(a_out)
        y = x + a_out * self.resid_attn_scale

        f_out = self.ffn(self.pre_ffn_norm(y))
        f_out = self.post_ffn_norm(f_out)
        y = y + f_out * self.resid_ffn_scale
        return y
