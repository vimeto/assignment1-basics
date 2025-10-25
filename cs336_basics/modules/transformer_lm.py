import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .transformer_block import TransformerBlock
from .embedding import Embedding
from .rms_norm import RMSNorm
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
        dtype=None,
        use_rope: bool = True,
        use_pre_norm: bool = True,
        use_post_norm: bool = False,
        use_rmsnorm: bool = True,
        use_swiglu: bool = False,
        use_qk_norm: bool = True,
        use_unet_residual: bool = True,
        unet_gate_init: float = 0.1,
        tie_embeddings: bool = True,
        use_x0_mixin: bool = True,
        x0_gate_init: float = 0.1,
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
        self.tie_embeddings = tie_embeddings
        self.use_x0_mixin = use_x0_mixin

        self.d_k = d_model // num_heads
        self.rope = RoPE(rope_theta, self.d_k, context_length, device=device, dtype=dtype)

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.emb_norm = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else IdentityNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, device=device, dtype=dtype, rope=self.rope,
                use_rope=use_rope, use_pre_norm=use_pre_norm, use_post_norm=use_post_norm,
                use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu, use_qk_norm=use_qk_norm,
                layer_idx=i, num_layers=num_layers
            ) for i in range(num_layers)
        ])

        if self.use_unet_residual:
            gate_init = float(torch.logit(torch.tensor(self.unet_gate_init))) if 0 < self.unet_gate_init < 1 else 0.0
            self.skip_gates = nn.ParameterList(
                nn.Parameter(torch.full((1, 1, d_model), gate_init, device=device, dtype=dtype))
                for _ in range(num_layers)
            )
            for gate in self.skip_gates:
                gate._optimizer_group = "vector"
        else:
            self.skip_gates = None
        self.unet_split = num_layers // 2

        if self.use_x0_mixin:
            gate_init = float(torch.logit(torch.tensor(x0_gate_init))) if 0 < x0_gate_init < 1 else 0.0
            self.x0_gates = nn.ParameterList(
                nn.Parameter(torch.full((1, 1, d_model), gate_init, device=device, dtype=dtype))
                for _ in range(num_layers)
            )
            for gate in self.x0_gates:
                gate._optimizer_group = "vector"
        else:
            self.x0_gates = None

        self.norm = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else IdentityNorm(d_model)
        if self.tie_embeddings:
            self.lm_head_weight = self.embedding.embedding_table
        else:
            self.ffn = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.use_gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embedding(x)
        y = self.emb_norm(y)
        x0 = y

        B, S, _ = y.shape
        pos = torch.arange(S, device=y.device)

        skip_cache = {}
        for idx, block in enumerate(self.layers):
            if self.use_x0_mixin and self.x0_gates is not None:
                gate = torch.sigmoid(self.x0_gates[idx])
                y = gate * x0 + (1 - gate) * y
            if self.use_gradient_checkpointing and self.training and y.requires_grad:
                y = checkpoint(lambda _y, _pos: block(_y, _pos), y, pos, use_reentrant=False)
            else:
                y = block(y, pos)

            if self.use_unet_residual and self.skip_gates is not None:
                if idx < self.unet_split:
                    skip_cache[idx] = y
                else:
                    mirror = self.num_layers - idx - 1
                    if mirror in skip_cache:
                        gate = torch.sigmoid(self.skip_gates[idx])
                        y = gate * skip_cache[mirror] + (1 - gate) * y

        y = self.norm(y)
        if self.tie_embeddings:
            logits = y.matmul(self.lm_head_weight.t())
        else:
            logits = self.ffn(y)
        return logits
