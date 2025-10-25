import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(x: Float[Tensor, " batch_size vocab_size"], y: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    # compute the softmax on the i-th dim of tensor x
    x_max = x.max(dim=-1, keepdim=True).values
    lse = (x - x_max).exp().sum(dim=-1).log().squeeze(-1) + x_max.squeeze(-1)
    nll = lse - x.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return nll.mean()
