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
        # Validate token IDs are within bounds before indexing
        max_id = token_ids.max().item()
        min_id = token_ids.min().item()

        if min_id < 0 or max_id >= self.num_embeddings:
            raise ValueError(
                f"Token IDs out of bounds: range [{min_id}, {max_id}], "
                f"but vocab_size={self.num_embeddings} (valid range: [0, {self.num_embeddings-1}])"
            )

        return self.embedding_table[token_ids, :]
