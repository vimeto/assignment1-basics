from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


def zeroth_power_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate spectral normalization used by Muon.

    Returns ``G (G^T G)^{-1/2}`` using a quintic Newton-Schulz polynomial.
    """

    orig_dtype = G.dtype
    G = G.to(torch.float32)
    shape = G.shape
    if G.ndim != 2:
        raise ValueError("Input to zeroth_power_via_newtonschulz5 must be 2D")

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True

    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.t()

    return X.to(orig_dtype).reshape(shape)


class MuonAdamW(torch.optim.Optimizer):
    """Muon optimizer for matrix params + AdamW for vector params.

    Muon is applied to tensors with ndim >= 2. Biases / norms fall back to AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        matrix_weight_decay: float | None = None,
        vector_weight_decay: float | None = None,
        momentum: float = 0.95,
        momentum_min: float | None = None,
        momentum_max: float | None = None,
        warmup_steps: int = 0,
        ns_steps: int = 5,
        eps: float = 1e-7,
        vector_lr_multiplier: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        vector_eps: float = 1e-8,
    ) -> None:

        self.matrix_params: list[torch.nn.Parameter] = []
        self.vector_params: list[torch.nn.Parameter] = []

        params = list(params)
        for p in params:
            if not p.requires_grad:
                continue
            group_name = getattr(getattr(p, "_optimizer_group", None), "lower", lambda: None)()
            if group_name == "vector":
                self.vector_params.append(p)
            elif group_name == "matrix":
                self.matrix_params.append(p)
            else:
                # attention/query matrices (Q,K,V,O) or FFN weights should default to Muon
                if p.ndim >= 2:
                    self.matrix_params.append(p)
                else:
                    self.vector_params.append(p)

        param_groups = []
        matrix_wd = matrix_weight_decay if matrix_weight_decay is not None else weight_decay
        vector_wd = vector_weight_decay if vector_weight_decay is not None else weight_decay
        if self.matrix_params:
            param_groups.append({
                "params": self.matrix_params,
                "group_type": "matrix",
                "weight_decay": matrix_wd,
            })
        if self.vector_params:
            param_groups.append({
                "params": self.vector_params,
                "group_type": "vector",
                "weight_decay": vector_wd,
                "vector_lr_multiplier": vector_lr_multiplier,
            })

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            momentum_min=momentum_min,
            momentum_max=momentum_max or momentum,
            warmup_steps=warmup_steps,
            ns_steps=ns_steps,
            eps=eps,
            betas=betas,
            vector_eps=vector_eps,
        )
        super().__init__(param_groups, defaults)

    def _muon_step(self, group):
        lr = group["lr"]
        weight_decay = group.get("weight_decay", 0.0)
        momentum = group["momentum"]
        momentum_min = group.get("momentum_min")
        momentum_max = group.get("momentum_max", momentum)
        warmup_steps = group.get("warmup_steps", 0)
        ns_steps = group.get("ns_steps", 5)
        eps = group.get("eps", 1e-7)

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            param_decay = getattr(p, "_weight_decay", weight_decay)
            if param_decay != 0:
                grad = grad.add(p.data, alpha=param_decay)

            state = self.state[p]
            step = state.get("step", 0) + 1
            state["step"] = step

            if grad.ndim < 2:
                matrix = grad.unsqueeze(0)
                reshape_back = grad.shape
            else:
                matrix = grad.reshape(grad.shape[0], -1)
                reshape_back = grad.shape

            orth = zeroth_power_via_newtonschulz5(matrix, steps=ns_steps, eps=eps)
            orth = orth.reshape(reshape_back)

            buf = state.get("momentum_buffer")
            if buf is None:
                buf = torch.zeros_like(p.data)

            if momentum_min is not None and warmup_steps > 0 and step <= warmup_steps:
                t = step / warmup_steps
                current_momentum = momentum_min + t * (momentum_max - momentum_min)
            else:
                current_momentum = momentum_max

            buf.mul_(current_momentum).add_(orth)
            state["momentum_buffer"] = buf
            p.data.add_(buf, alpha=-lr)

        group["effective_lr"] = lr
        group["current_momentum"] = current_momentum

    def _adamw_step(self, group):
        lr = group["lr"] * group.get("vector_lr_multiplier", 1.0)
        weight_decay = group.get("weight_decay", 0.0)
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("vector_eps", 1e-8)

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            param_decay = getattr(p, "_weight_decay", weight_decay)
            if param_decay != 0:
                grad = grad.add(p.data, alpha=param_decay)

            state = self.state[p]
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None:
                exp_avg = torch.zeros_like(p.data)
                exp_avg_sq = torch.zeros_like(p.data)

            step = state.get("step", 0) + 1
            state["step"] = step

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(eps)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(exp_avg, denom, value=-step_size)

            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq

        group["effective_lr"] = lr

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            gtype = group.get("group_type")
            if gtype == "matrix":
                self._muon_step(group)
            elif gtype == "vector":
                self._adamw_step(group)
        return loss

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return alpha_max * (t / T_w)

    if t < T_c:
        return alpha_min + 1/2 * (1 + math.cos(math.pi * (t - T_w)/(T_c - T_w))) * (alpha_max - alpha_min)

    return alpha_min


def linear_warmup_decay(step: int, alpha_max: float, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(0, warmup_steps)
    if warmup_steps > 0 and step <= warmup_steps:
        return alpha_max * (step / warmup_steps)
    if step >= total_steps:
        return 0.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    return alpha_max * (1.0 - progress)


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
