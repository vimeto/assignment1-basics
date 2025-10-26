import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings # vocab size
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = torch.empty(num_embeddings, embedding_dim, dtype=self.dtype, device=self.device)
        std = 0.02
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        self.embedding_table = nn.Parameter(weight)
        self.embedding_table._weight_decay = 0.01

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.embedding_table)
