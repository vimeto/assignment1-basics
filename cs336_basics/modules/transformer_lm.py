import torch
import torch.nn as nn
import numpy as np
from .transformer_block import TransformerBlock
from .embedding import Embedding
from .rms_norm import RMSNorm
from .linear import Linear
from .rope import RoPE
from .attention import softmax

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device

        self.d_k = d_model // num_heads

        self.rope = RoPE(rope_theta, self.d_k, context_length, device=device)

        # 1. embedding
        self.embedding = Embedding(vocab_size, d_model, device=device)

        # 2. transformer layers
        self.layers = [TransformerBlock(d_model, num_heads, d_ff, device=device, rope=self.rope) for _ in range(num_layers)]

        # 3. norm
        self.norm = RMSNorm(d_model, device=device)

        # 4. linear
        self.ffn = Linear(d_model, vocab_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.embedding(x)
        for block in self.layers:
            y = block(y)
        y = self.norm(y)
        y = self.ffn(y)

        return y
