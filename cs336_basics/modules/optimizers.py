from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


# ---- Polar-Express (batched, BF16-friendly) orthogonalizer ----
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.no_grad()
def polar_express_sign(G: torch.Tensor) -> torch.Tensor:
    """Fast, matrix-sign approximation without SVD (Polar Express)."""

    X = G
    swapped = False
    if X.size(-2) > X.size(-1):
        X = X.transpose(-2, -1)
        swapped = True

    Xb = X.to(torch.bfloat16)
    spec = Xb.norm(dim=(-2, -1), keepdim=True).to(torch.float32)
    Xb = Xb / (spec * (1.0 + 2e-2) + 1e-6)

    for a, b, c in POLAR_EXPRESS_COEFFS:
        A = Xb @ Xb.transpose(-2, -1)
        B = b * A + c * (A @ A)
        Xb = a * Xb + B @ Xb

    if swapped:
        Xb = Xb.transpose(-2, -1)
    return Xb.to(G.dtype)


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
        matrix_base_lr: float | None = None,
        vector_base_lr: float | None = None,
        normalizer_beta: float = 0.95,
        normalizer_eps: float = 1e-10,
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
        matrix_base_lr = matrix_base_lr if matrix_base_lr is not None else lr
        vector_base_lr = vector_base_lr if vector_base_lr is not None else lr
        if self.matrix_params:
            param_groups.append({
                "params": self.matrix_params,
                "group_type": "matrix",
                "weight_decay": matrix_wd,
                "lr": matrix_base_lr,
                "base_lr": matrix_base_lr,
                "normalizer_beta": normalizer_beta,
                "normalizer_eps": normalizer_eps,
            })
        if self.vector_params:
            param_groups.append({
                "params": self.vector_params,
                "group_type": "vector",
                "weight_decay": vector_wd,
                "vector_lr_multiplier": vector_lr_multiplier,
                "lr": vector_base_lr,
                "base_lr": vector_base_lr,
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
        base_lr = group["lr"]
        weight_decay = group.get("weight_decay", 0.0)
        momentum = group.get("momentum", 0.95)
        normalizer_beta = group.get("normalizer_beta", 0.95)
        normalizer_eps = group.get("normalizer_eps", 1e-10)

        shape_buckets: dict[tuple[tuple[int, ...], bool], list[torch.nn.Parameter]] = defaultdict(list)
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.ndim < 2:
                # these parameters should generally live in the vector group
                continue
            is_qkvo = bool(getattr(p, "_is_qkvo", False))
            shape_buckets[(tuple(p.grad.shape), is_qkvo)].append(p)

        for (shape, is_qkvo_batch), params in shape_buckets.items():
            stacked_updates = []
            infos = []
            for p in params:
                grad = p.grad.data
                param_decay = getattr(p, "_weight_decay", weight_decay)
                state = self.state[p]
                step = state.get("step", 0) + 1
                state["step"] = step

                grad32 = grad.to(torch.float32)
                rows = grad32.shape[0]
                cols = grad32.numel() // rows
                grad_matrix = grad32.reshape(rows, cols)

                buf = state.get("momentum_buffer")
                if buf is None or buf.shape != grad.shape:
                    buf = torch.zeros_like(grad, dtype=torch.float32)
                state["momentum_buffer"] = buf
                buf_matrix = buf.reshape(rows, cols)
                buf_matrix.mul_(momentum).add_(grad_matrix, alpha=1 - momentum)

                update_matrix = grad_matrix.lerp(buf_matrix, momentum)
                stacked_updates.append(update_matrix)
                infos.append((p, rows, cols, param_decay, grad.shape, state))

            if not stacked_updates:
                continue

            update_batch = torch.stack(stacked_updates)

            if is_qkvo_batch:
                B, r, c = update_batch.shape
                if c % 4 == 0:
                    reshaped = update_batch.view(B * 4, r, c // 4)
                    orth = polar_express_sign(reshaped)
                    orth_updates = orth.view(B, 4, r, c // 4).reshape(B, r, c)
                elif r % 4 == 0:
                    reshaped = update_batch.view(B * 4, r // 4, c)
                    orth = polar_express_sign(reshaped)
                    orth_updates = orth.view(B, 4, r // 4, c).reshape(B, r, c)
                else:
                    orth_updates = polar_express_sign(update_batch)
            else:
                orth_updates = polar_express_sign(update_batch)

            for mat, info in zip(orth_updates, infos):
                if info is None:
                    continue
                p, rows, cols, param_decay, original_shape, state = info

                orig_norm = mat.norm().to(torch.float32)
                mat32 = mat.to(torch.float32)
                if rows >= cols:
                    reduce_dim = 1
                    moment_shape = (rows, 1)
                else:
                    reduce_dim = 0
                    moment_shape = (1, cols)
                moment = mat32.pow(2).mean(dim=reduce_dim, keepdim=True)
                second = state.get("second_moment")
                if second is None or second.shape != moment_shape:
                    second = torch.zeros(moment_shape, dtype=torch.float32, device=mat32.device)
                second.mul_(normalizer_beta).add_(moment, alpha=1 - normalizer_beta)
                state["second_moment"] = second

                mat32.mul_((second + normalizer_eps) ** -0.5)
                new_norm = mat32.norm()
                if orig_norm > 0 and new_norm > 0:
                    mat32.mul_(orig_norm / (new_norm + normalizer_eps))

                aspect = math.sqrt(max(1.0, rows / float(cols)))
                lr_mul = getattr(p, "lr_mul", 1.0)
                eff_lr = base_lr * aspect * lr_mul
                wd_mul = getattr(p, "wd_mul", 1.0)

                update_tensor = mat32.reshape(original_shape).to(p.data.dtype)
                p.data.add_(update_tensor, alpha=-eff_lr)

                if param_decay != 0:
                    p.data.add_(p.data, alpha=-eff_lr * param_decay * wd_mul)

        group["effective_lr"] = base_lr
        group["current_momentum"] = momentum

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
            lr_mul = getattr(p, "lr_mul", 1.0)
            wd_mul = getattr(p, "wd_mul", 1.0)
            lr_param = lr * lr_mul

            state = self.state[p]
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None:
                exp_avg = torch.zeros_like(p.data, dtype=torch.float32)
                exp_avg_sq = torch.zeros_like(p.data, dtype=torch.float32)

            step = state.get("step", 0) + 1
            state["step"] = step

            grad32 = grad.to(torch.float32)
            exp_avg.mul_(beta1).add_(grad32, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad32, grad32, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(eps)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr_param * math.sqrt(bias_correction2) / bias_correction1

            update = (exp_avg / denom).to(p.data.dtype)
            p.data.add_(update, alpha=-step_size)

            if param_decay != 0:
                p.data.add_(p.data, alpha=-lr_param * param_decay * wd_mul)

            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq

        group["effective_lr"] = lr

    def step(
        self,
        closure: Optional[Callable] = None,
        *,
        matrix_step: bool = True,
        vector_step: bool = True,
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            gtype = group.get("group_type")
            if gtype == "matrix" and matrix_step:
                self._muon_step(group)
            elif gtype == "vector" and vector_step:
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
                state = self.state[p]
                t = state.get("t", 0) + 1
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p.data, dtype=torch.float32)
                m = state["m"]
                v = state["v"]
                grad = p.grad.data
                g32 = grad.to(torch.float32)

                m.mul_(b1).add_(g32, alpha=1 - b1)
                v.mul_(b2).addcmul_(g32, g32, value=1 - b2)

                a_t = lr * math.sqrt(1 - (b2 ** t)) / (1 - (b1 ** t))
                denom = v.sqrt().add_(eps)
                update = (m / denom).to(p.data.dtype)
                p.data.add_(update, alpha=-a_t)
                p.data.add_(p.data, alpha=-lr * weight_decay)

                state["t"] = t
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
