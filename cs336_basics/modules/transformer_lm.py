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
        tie_embeddings: bool = False,
        use_x0_mixin: bool = True,
        x0_gate_init: float = 0.1,
        use_value_embeddings: bool = False,
        num_value_embeddings: int = 3,
        value_embed_lr_mul: float = 50.0,
        sa_lambda_init: tuple[float, float] = (0.5, 0.5),
        sa_lambda_lr_mul: float = 5.0,
        use_smear: bool = False,
        smear_lambda_init: float = 0.0,
        smear_gate_dim: int = 12,
        use_attn_gate: bool = False,
        attn_gate_dim: int = 12,
        attn_gate_lr_mul: float = 5.0,
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
        self.use_value_embeddings = use_value_embeddings
        self.num_value_embeddings = num_value_embeddings
        self.value_embed_lr_mul = value_embed_lr_mul
        self.use_smear = use_smear
        self.use_attn_gate = use_attn_gate
        self.attn_gate_dim = attn_gate_dim
        self.attn_gate_lr_mul = attn_gate_lr_mul

        self.d_k = d_model // num_heads
        self.rope = RoPE(rope_theta, self.d_k, context_length, device=device, dtype=dtype)

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.emb_norm = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else IdentityNorm(d_model)

        # Smear gate: smears token embeddings forward 1 position (nanoGPT-style)
        if self.use_smear:
            self.smear_gate = Linear(smear_gate_dim, 1, device=device, dtype=dtype)
            self.smear_gate.linear.data.zero_()
            self.smear_lambda = nn.Parameter(torch.tensor([smear_lambda_init], device=device, dtype=dtype))
            self.smear_gate.linear._optimizer_group = "vector"
            self.smear_lambda._optimizer_group = "vector"
            self.smear_gate.linear._weight_decay = 0.0
            self.smear_lambda._weight_decay = 0.0
            self.smear_gate.linear.wd_mul = 0.0
            self.smear_lambda.wd_mul = 0.0
            self.smear_gate.linear.lr_mul = 5.0
            self.smear_lambda.lr_mul = 5.0
        else:
            self.smear_gate = None
            self.smear_lambda = None

        # Token value embeddings (nanoGPT-style)
        if self.use_value_embeddings:
            self.value_embeds = nn.ModuleList([
                Embedding(vocab_size, d_model, device=device, dtype=dtype)
                for _ in range(num_value_embeddings)
            ])
            # Set LR multiplier for value embeddings
            for value_embed in self.value_embeds:
                for param in value_embed.parameters():
                    param.lr_mul = value_embed_lr_mul
        else:
            self.value_embeds = None

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                device=device,
                dtype=dtype,
                rope=self.rope,
                use_rope=use_rope,
                use_pre_norm=use_pre_norm,
                use_post_norm=use_post_norm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                use_qk_norm=use_qk_norm,
                layer_idx=i,
                num_layers=num_layers,
                use_attn_gate=use_attn_gate,
                attn_gate_dim=attn_gate_dim,
                attn_gate_lr_mul=attn_gate_lr_mul,
            ) for i in range(num_layers)
        ])

        if self.use_unet_residual:
            # NanoGPT-style scalar gates: one scalar per layer, initialized to -1.5 (σ(-1.5) ≈ 0.18)
            gate_init = -1.5
            self.skip_gates = nn.ParameterList(
                nn.Parameter(torch.tensor([gate_init], device=device, dtype=dtype))
                for _ in range(num_layers)
            )
            for gate in self.skip_gates:
                gate._optimizer_group = "vector"
                gate._weight_decay = 0.0
                gate.wd_mul = 0.0
                gate.lr_mul = 5.0  # NanoGPT LR multiplier
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
                gate._weight_decay = 0.0
                gate.wd_mul = 0.0
                gate.lr_mul = 5.0  # NanoGPT LR multiplier for scalar gates
        else:
            self.x0_gates = None

        # SA lambdas for mixing value and value embeddings (nanoGPT-style)
        if self.use_value_embeddings:
            self.sa_lambdas = nn.ParameterList(
                nn.Parameter(torch.tensor(list(sa_lambda_init), device=device, dtype=dtype))
                for _ in range(num_layers)
            )
            for sa_lambda in self.sa_lambdas:
                sa_lambda._optimizer_group = "vector"
                sa_lambda._weight_decay = 0.0
                sa_lambda.wd_mul = 0.0
                sa_lambda.lr_mul = sa_lambda_lr_mul
        else:
            self.sa_lambdas = None

        self.norm = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else IdentityNorm(d_model)
        if self.tie_embeddings:
            self.lm_head_weight = self.embedding.embedding_table
        else:
            self.ffn = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.use_gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embedding(x)

        # Smear token embeddings forward 1 position (nanoGPT-style)
        if self.use_smear and self.smear_gate is not None:
            # Take first smear_gate_dim features from embeddings
            gate_input = y[:, 1:, :self.smear_gate.linear.size(1)]  # (B, S-1, smear_gate_dim)
            smear_gate_out = self.smear_lambda * torch.sigmoid(self.smear_gate(gate_input))  # (B, S-1, 1)
            # Add smeared previous token: y[1:] += gate * y[:-1]
            y_smeared = torch.cat([
                y[:, :1],  # Keep first token unchanged
                y[:, 1:] + smear_gate_out * y[:, :-1]  # Add gated previous token to rest
            ], dim=1)
            y = y_smeared

        y = self.emb_norm(y)
        x0 = y

        B, S, _ = y.shape
        pos = torch.arange(S, device=y.device)

        # Compute value embeddings following nanoGPT pattern
        # Pattern: [None, ve[1], ve[2]] + [None] * (middle layers) + [ve[0], ve[1], ve[2]]
        if self.use_value_embeddings and self.value_embeds is not None:
            ve_list = [value_embed(x) for value_embed in self.value_embeds]
            # Create pattern for 6+ layer models
            if self.num_layers >= 6:
                ve_pattern = [None, ve_list[1], ve_list[2]] + [None] * (self.num_layers - 6) + [ve_list[0], ve_list[1], ve_list[2]]
            else:
                # For smaller models, just use a simple pattern
                ve_pattern = [ve_list[i % len(ve_list)] if i < len(ve_list) else None for i in range(self.num_layers)]
            assert len(ve_pattern) == self.num_layers, f"Value embedding pattern length {len(ve_pattern)} doesn't match num_layers {self.num_layers}"
        else:
            ve_pattern = [None] * self.num_layers

        skip_cache = {}
        for idx, block in enumerate(self.layers):
            if self.use_x0_mixin and self.x0_gates is not None:
                gate = torch.sigmoid(self.x0_gates[idx])
                y = gate * x0 + (1 - gate) * y

            # Get value embedding and sa_lambda for this layer
            ve = ve_pattern[idx]
            sa_lambda = self.sa_lambdas[idx] if self.sa_lambdas is not None else None

            if self.use_gradient_checkpointing and self.training and y.requires_grad:
                y = checkpoint(lambda _y, _pos, _ve, _sa_lambda: block(_y, _pos, _ve, _sa_lambda), y, pos, ve, sa_lambda, use_reentrant=False)
            else:
                y = block(y, pos, ve, sa_lambda)

            if self.use_unet_residual and self.skip_gates is not None:
                if idx < self.unet_split:
                    skip_cache[idx] = y
                else:
                    mirror = self.num_layers - idx - 1
                    if mirror in skip_cache:
                        # NanoGPT-style additive residual: x = x + gate * skip_connection
                        gate = torch.sigmoid(self.skip_gates[idx])
                        y = y + gate * skip_cache[mirror]

        y = self.norm(y)
        if self.tie_embeddings:
            logits = y.matmul(self.lm_head_weight.t())
        else:
            logits = self.ffn(y)
        return logits
