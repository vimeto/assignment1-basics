import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .transformer_block import TransformerBlock
from .embedding import Embedding
from .rms_norm import RMSNorm, nn as _nn  # keep import style consistent if needed
from .identity_norm import IdentityNorm
from .linear import Linear
from .rope import RoPE

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        use_rope: bool = True,
        use_pre_norm: bool = True,
        use_post_norm: bool = False,
        use_rmsnorm: bool = True,
        use_swiglu: bool = False,
        use_qk_norm: bool = True,
        use_unet_residual: bool = True,
        unet_gate_init: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_unet_residual = use_unet_residual and num_layers >= 2
        self.unet_gate_init = float(unet_gate_init)

        self.d_k = d_model // num_heads
        self.rope = RoPE(rope_theta, self.d_k, context_length, device=device)

        # Embedding + embed RMSNorm
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.emb_norm = RMSNorm(d_model, device=device) if use_rmsnorm else IdentityNorm(d_model)

        # Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, device=device, rope=self.rope,
                use_rope=use_rope, use_pre_norm=use_pre_norm, use_post_norm=use_post_norm,
                use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu, use_qk_norm=use_qk_norm,
                layer_idx=i, num_layers=num_layers
            ) for i in range(num_layers)
        ])

        if self.use_unet_residual:
            gate_init = float(torch.logit(torch.tensor(self.unet_gate_init))) if 0 < self.unet_gate_init < 1 else 0.0
            self.skip_gates = nn.ParameterList(
                nn.Parameter(torch.full((1, 1, d_model), gate_init, device=device))
                for _ in range(num_layers)
            )
        else:
            self.skip_gates = None
        self.unet_split = num_layers // 2

        # Final norm + untied lm head
        self.norm = RMSNorm(d_model, device=device) if use_rmsnorm else IdentityNorm(d_model)
        self.ffn = Linear(d_model, vocab_size, device=device)  # lm head
        self.use_gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embedding(x)
        y = self.emb_norm(y)

        skip_cache = {}
        for idx, block in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training and y.requires_grad:
                y = checkpoint(block, y, use_reentrant=False)
            else:
                y = block(y)

            if self.use_unet_residual and self.skip_gates is not None:
                if idx < self.unet_split:
                    skip_cache[idx] = y
                else:
                    mirror = self.num_layers - idx - 1
                    if mirror in skip_cache:
                        gate = torch.sigmoid(self.skip_gates[idx])
                        y = gate * skip_cache[mirror] + (1 - gate) * y

        y = self.norm(y)
        y = self.ffn(y)
        return y
