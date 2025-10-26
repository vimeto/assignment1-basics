import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(x: Float[Tensor, " batch_size vocab_size"], y: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Numerically-stable CE in fp32 regardless of AMP mode."""
    x32 = x.to(torch.float32)
    # logsumexp
    lse = torch.logsumexp(x32, dim=-1)
    nll = lse - x32.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return nll.mean()
