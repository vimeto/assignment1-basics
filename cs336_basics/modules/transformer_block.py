import torch
import torch.nn as nn
import torch.nn.functional as F
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
        use_swiglu: bool = False,   # we'll use ReLU² via SiLU_FFN path
        use_qk_norm: bool = True,
        layer_idx: int = 0,
        num_layers: int = 1,
    ):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        self.use_post_norm = use_post_norm

        norm = RMSNorm if use_rmsnorm else IdentityNorm
        self.pre_attn_norm = norm(d_model, device=device) if use_pre_norm else IdentityNorm(d_model)
        self.pre_ffn_norm = norm(d_model, device=device) if use_pre_norm else IdentityNorm(d_model)
        self.post_attn_norm = norm(d_model, device=device) if use_post_norm else IdentityNorm(d_model)
        self.post_ffn_norm = norm(d_model, device=device) if use_post_norm else IdentityNorm(d_model)

        self.attn = MultiHeadAttention(
            d_model, num_heads, device, rope, use_rope=use_rope, use_qk_norm=use_qk_norm
        )
        # Gated attention output (silu gate)
        self.attn_gate = nn.Linear(d_model, d_model, device=device, bias=False)
        nn.init.zeros_(self.attn_gate.weight)  # start as identity via residual

        # FFN: ReLU² implementation in SiLU_FFN module
        self.ffn = SiLU_FFN(d_model, d_ff, device=device)

        # Light residual scale by depth (LayerScale-ish, starts at 1.0 but can learn)
        init_scale = 1.0
        self.resid_attn_scale = nn.Parameter(torch.full((1, 1, d_model), init_scale, device=device))
        self.resid_ffn_scale = nn.Parameter(torch.full((1, 1, d_model), init_scale, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[-2]
        pos = torch.arange(s, device=x.device)

        a_in = self.pre_attn_norm(x)
        a_out = self.attn(a_in, pos)
        # gated attention
        gate = F.silu(self.attn_gate(a_in))
        a_out = a_out * gate
        a_out = self.post_attn_norm(a_out)
        y = x + a_out * self.resid_attn_scale

        f_in = self.pre_ffn_norm(y)
        f_out = self.ffn(f_in)
        f_out = self.post_ffn_norm(f_out)
        y = y + f_out * self.resid_ffn_scale
        return y
