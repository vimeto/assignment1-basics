import torch
from torch.nn import Parameter
from collections.abc import Iterable
import math

@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for grad in grads:
        total += grad.detach().to(torch.float32).pow(2).sum()

    norm = total.sqrt()
    if not torch.isfinite(norm) or norm <= max_l2_norm:
        return

    scale = max_l2_norm / (norm + eps)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(scale)

