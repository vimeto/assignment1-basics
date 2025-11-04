import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .identity_norm import IdentityNorm
from .relu2_ffn import ReLU2_FFN
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
        rope=None,
        use_rope: bool = True,
        use_pre_norm: bool = True,
        use_post_norm: bool = False,
        use_rmsnorm: bool = True,
        use_swiglu: bool = False,
        use_qk_norm: bool = True,
        layer_idx: int = 0,
        num_layers: int = 1,
        use_attn_gate: bool = False,
        attn_gate_dim: int = 0,
        attn_gate_lr_mul: float = 5.0,
    ):
        super().__init__()
        norm = RMSNorm if use_rmsnorm else IdentityNorm
        self.pre_attn_norm = norm(d_model, device=device, dtype=dtype) if use_pre_norm else IdentityNorm(d_model)
        self.pre_ffn_norm = norm(d_model, device=device, dtype=dtype) if use_pre_norm else IdentityNorm(d_model)
        self.post_attn_norm = norm(d_model, device=device, dtype=dtype) if use_post_norm else IdentityNorm(d_model)
        self.post_ffn_norm = norm(d_model, device=device, dtype=dtype) if use_post_norm else IdentityNorm(d_model)

        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            device,
            dtype=dtype,
            rope=rope,
            use_rope=use_rope,
            use_qk_norm=use_qk_norm,
            use_attn_gate=use_attn_gate,
            attn_gate_dim=attn_gate_dim,
            attn_gate_lr_mul=attn_gate_lr_mul,
        )
        self.attn.W_qkv._is_qkvo = True
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn = ReLU2_FFN(d_model, d_ff, device=device, dtype=dtype)

        depth = max(1, layer_idx + 1)
        # LayerNorm Scaling (LNS): scale pre-norm outputs by 1/sqrt(depth)
        self.lns_scale = 1.0 / math.sqrt(depth)

        # Residual scales (LayerScale-like) near 1/sqrt(2L)
        init = 1.0 / math.sqrt(max(1, 2 * num_layers))
        self.resid_attn_scale = nn.Parameter(torch.full((1, 1, d_model), init, device=device))
        self.resid_ffn_scale  = nn.Parameter(torch.full((1, 1, d_model), init, device=device))
        self.resid_attn_scale._optimizer_group = "vector"
        self.resid_ffn_scale._optimizer_group = "vector"
        self.resid_attn_scale._weight_decay = 0.0
        self.resid_ffn_scale._weight_decay = 0.0
        self.resid_attn_scale._weight_decay = 0.0
        self.resid_ffn_scale._weight_decay = 0.0

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,
        value_embed: torch.Tensor | None = None,
        sa_lambda: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos is None:
            seq_len = x.size(1)
            base = torch.arange(seq_len, device=x.device)
            pos = base.unsqueeze(0).expand(x.size(0), -1)
        a_out = self.attn(self.pre_attn_norm(x) * self.lns_scale, pos, value_embed, sa_lambda)
        a_out = self.post_attn_norm(a_out)
        y = x + a_out * self.resid_attn_scale

        f_out = self.ffn(self.pre_ffn_norm(y) * self.lns_scale)
        f_out = self.post_ffn_norm(f_out)
        y = y + f_out * self.resid_ffn_scale
        return y
