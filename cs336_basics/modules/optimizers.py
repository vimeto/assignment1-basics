from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return alpha_max * (t / T_w)

    if t < T_c:
        return alpha_min + 1/2 * (1 + math.cos(math.pi * (t - T_w)/(T_c - T_w))) * (alpha_max - alpha_min)

    return alpha_min


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, dtype=None):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        params = list(params)
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "dtype": dtype}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            dtype = group["dtype"]
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p, dtype=dtype))
                v = state.get("v", torch.zeros_like(p, dtype=dtype))
                grad = p.grad.data # Get the gradient of loss with respect to p.

                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * (grad ** 2)

                a_t = lr  * math.sqrt(1 - (b2 ** t)) / (1 - (b1 ** t))
                p.data -= a_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss



if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    losses = {}
    for lr in [1e1, 1e2, 1e3]:
        opt = AdamW([weights], alpha=lr)
        losses[lr] = []
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            losses[lr].append(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step()

    print("|\t" + "\t|\t".join([str(l) for l in losses.keys()]) + "\t|")
    for t in range(10):
        print("|\t" + "\t|\t".join([str(round(v[t], 5)) for v in losses.values()]) + "\t|")

