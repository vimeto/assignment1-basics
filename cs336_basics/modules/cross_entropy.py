import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(x: Float[Tensor, " batch_size vocab_size"], y: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    # compute the softmax on the i-th dim of tensor x
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x = x - x_max

    numerators = torch.gather(x, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=-1)
    log_x = torch.log(sum_x)

    res = -numerators + log_x

    return torch.mean(res)
