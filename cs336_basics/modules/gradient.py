import torch
from torch.nn import Parameter
from collections.abc import Iterable
import math

@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    if norm <= max_l2_norm:
        return

    scaling_factor = max_l2_norm / (norm + eps)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scaling_factor)


