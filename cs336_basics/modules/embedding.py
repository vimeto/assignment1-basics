import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings # vocab size
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = torch.empty(num_embeddings, embedding_dim, dtype=self.dtype, device=self.device)
        nn.init.trunc_normal_(weight, mean=0.0, std=1, a=-3, b=3)
        self.embedding_table = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_table[token_ids, :]
