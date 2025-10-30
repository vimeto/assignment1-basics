from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False


# ---- Polar-Express (batched, BF16-friendly) orthogonalizer ----
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

if _TRITON_AVAILABLE:

    def _get_autotune_configs():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                    "BLOCK_SIZE_K": bk,
                    "GROUP_SIZE_M": 8,
                    "LOWER_UPPER": 1,
                },
                num_stages=stages,
                num_warps=warps,
            )
            for bm in [64, 128]
            for bn in [64, 128, 256]
            for bk in [64, 128]
            for stages, warps in [(3, 4), (3, 8), (4, 4)]
            if bm // bn <= 2 and bn // bm <= 2
        ]


    @triton.jit
    def _pid_to_block(
        pid,
        M,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

        batch_idx = pid // (num_pid_m * num_pid_n)
        pid = pid % (num_pid_m * num_pid_n)

        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

        m_idx = pid_m * BLOCK_SIZE_M
        n_idx = pid_n * BLOCK_SIZE_N
        return batch_idx, m_idx, n_idx


    @triton.autotune(
        configs=_get_autotune_configs(),
        key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
    )
    @triton.jit
    def _XXT_kernel(
        A_ptr, C_ptr,
        M, K,
        a_stride_b, a_stride_r, a_stride_c,
        c_stride_b, c_stride_r, c_stride_c,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        LOWER_UPPER: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx, m_idx, n_idx = _pid_to_block(
            pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
        )

        skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
        skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
        if skip_block_below_diag or skip_block_above_diag:
            return

        A_ptr += batch_idx * a_stride_b
        C_ptr += batch_idx * c_stride_b

        offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
        at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, at, accumulator)
            a_ptrs += BLOCK_SIZE_K * a_stride_c
            at_ptrs += BLOCK_SIZE_K * a_stride_c

        out_dtype = C_ptr.dtype.element_ty
        output = accumulator.to(out_dtype)

        offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, output, mask=c_mask)

        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output.T, mask=c_mask_t)


    def _XXT(A: torch.Tensor, out: torch.Tensor):
        assert A.ndim == 2 or A.ndim == 3
        M, K = A.shape[-2:]
        batch_size = A.size(0) if A.ndim == 3 else 1
        input_batch_stride = A.stride(0) if A.ndim == 3 else 0
        output_batch_stride = out.stride(0) if out.ndim == 3 else 0

        grid = lambda meta: (
            batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
        )
        _XXT_kernel[grid](
            A_ptr=A,
            C_ptr=out,
            M=M,
            K=K,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2),
            a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2),
            c_stride_c=out.stride(-1),
        )
        return out


    @triton.autotune(
        configs=_get_autotune_configs(),
        key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
    )
    @triton.jit
    def _ba_plus_cAA_kernel(
        A_ptr, C_ptr,
        M,
        a_stride_b, a_stride_r, a_stride_c,
        c_stride_b, c_stride_r, c_stride_c,
        alpha, beta,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        LOWER_UPPER: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx, m_idx, n_idx = _pid_to_block(
            pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
        )

        skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
        skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
        if skip_block_below_diag or skip_block_above_diag:
            return

        A_ptr += batch_idx * a_stride_b
        C_ptr += batch_idx * c_stride_b

        offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
        at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
            at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, at, accumulator)
            a_ptrs += BLOCK_SIZE_K * a_stride_c
            at_ptrs += BLOCK_SIZE_K * a_stride_c

        offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
        a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
        a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
        a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

        accumulator *= alpha
        accumulator += a_add * beta

        out_dtype = C_ptr.dtype.element_ty
        output = accumulator.to(out_dtype)

        offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, output, mask=c_mask)

        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output.T, mask=c_mask_t)


    def _ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
        assert A.ndim == 2 or A.ndim == 3
        M, _ = A.shape[-2:]
        batch_size = A.size(0) if A.ndim == 3 else 1
        input_batch_stride = A.stride(0) if A.ndim == 3 else 0
        output_batch_stride = out.stride(0) if out.ndim == 3 else 0

        grid = lambda meta: (
            batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
        )
        _ba_plus_cAA_kernel[grid](
            A_ptr=A,
            C_ptr=out,
            M=M,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2),
            a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2),
            c_stride_c=out.stride(-1),
            alpha=alpha,
            beta=beta,
        )
        return out


    @torch.no_grad()
    def _polar_express_triton(G: torch.Tensor) -> torch.Tensor:
        swapped = False
        X = G
        if X.size(-2) > X.size(-1):
            X = X.transpose(-2, -1)
            swapped = True

        Xb = X.to(torch.bfloat16)
        spec = Xb.norm(dim=(-2, -1), keepdim=True).to(torch.float32)
        Xb = Xb / (spec * (1.0 + 2e-2) + 1e-6)
        Xb = Xb.contiguous()

        aX_plus_BX = torch.baddbmm if Xb.ndim > 2 else torch.addmm
        A = torch.empty((*Xb.shape[:-1], Xb.size(-2)), device=Xb.device, dtype=Xb.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(Xb)

        for a, b, c in POLAR_EXPRESS_COEFFS:
            _XXT(Xb, out=A)
            _ba_plus_cAA(A, alpha=c, beta=b, out=B)
            aX_plus_BX(Xb, B, Xb, beta=a, out=C)
            Xb, C = C, Xb

        if swapped:
            Xb = Xb.transpose(-2, -1)
        return Xb.to(G.dtype)

else:

    def _polar_express_triton(G: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Triton is not available")

@torch.no_grad()
def _polar_express_reference(G: torch.Tensor) -> torch.Tensor:
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


@torch.no_grad()
def polar_express_sign(G: torch.Tensor) -> torch.Tensor:
    if _TRITON_AVAILABLE and G.is_cuda:
        try:
            return _polar_express_triton(G)
        except Exception:
            pass
    return _polar_express_reference(G)


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


class NorMuon(torch.optim.Optimizer):
    """Standalone Muon for matrix-shaped params only (ndim >= 2).

    Applies orthogonalized momentum updates with Polar Express and per-row/col RMS normalizer.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        normalizer_beta: float = 0.95,
        normalizer_eps: float = 1e-10,
        eps: float = 1e-7,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            normalizer_beta=normalizer_beta,
            normalizer_eps=normalizer_eps,
            eps=eps,
        )
        super().__init__(list(params), defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            momentum = group.get("momentum", 0.95)
            normalizer_beta = group.get("normalizer_beta", 0.95)
            normalizer_eps = group.get("normalizer_eps", 1e-10)

            # bucket by shape and _is_qkvo flag
            shape_buckets: dict[tuple[tuple[int, ...], bool], list[torch.nn.Parameter]] = defaultdict(list)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim < 2:
                    # only matrices here
                    continue
                is_qkvo = bool(getattr(p, "_is_qkvo", False))
                shape_buckets[(tuple(p.grad.shape), is_qkvo)].append(p)

            for (shape, is_qkvo_batch), params_ in shape_buckets.items():
                stacked_updates = []
                infos = []
                for p in params_:
                    grad = p.grad.data
                    param_decay = getattr(p, "_weight_decay", wd)
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
                    eff_lr = lr * aspect * lr_mul
                    wd_mul = getattr(p, "wd_mul", 1.0)

                    update_tensor = mat32.reshape(original_shape).to(p.data.dtype)
                    p.data.add_(update_tensor, alpha=-eff_lr)

                    if param_decay != 0:
                        p.data.add_(p.data, alpha=-eff_lr * param_decay * wd_mul)
        return loss


class DistAdam(torch.optim.Optimizer):
    """Single-GPU AdamW-style optimizer used for scalar/head/embed params.

    Named "DistAdam" to mirror nanoGPT API; this implementation is local (non-distributed).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.65, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(list(params), defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group.get("betas", (0.65, 0.95))
            eps = group.get("eps", 1e-8)
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                lr_mul = getattr(p, "lr_mul", 1.0)
                wd_mul = getattr(p, "wd_mul", 1.0)
                lr_param = lr * lr_mul

                state = self.state[p]
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(p.data, dtype=torch.float32)
                    exp_avg_sq = torch.zeros_like(p.data, dtype=torch.float32)
                step_t = state.get("step", 0) + 1
                state["step"] = step_t

                g32 = grad.to(torch.float32)
                exp_avg.mul_(beta1).add_(g32, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr_param * (math.sqrt(1 - beta2 ** step_t) / (1 - beta1 ** step_t))
                update = (exp_avg / denom).to(p.data.dtype)
                p.data.add_(update, alpha=-step_size)

                if wd != 0:
                    p.data.add_(p.data, alpha=-lr_param * wd * wd_mul)

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
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
