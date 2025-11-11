from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Any
import time

import numpy as np
import torch

from cs336_basics.training.configs import ExperimentConfig
from cs336_basics.training.optim import build_optimizer, zero_grads_for
from cs336_basics.training.schedules import trapezoid_schedule
from cs336_basics.training.data import build_eval_loader
from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.modules.cross_entropy import cross_entropy
from cs336_basics.modules.checkpointing import save_checkpoint as module_save_checkpoint, load_checkpoint as module_load_checkpoint
from cs336_basics.modules.gradient import gradient_clipping
from cs336_basics.modules.dataloader import (
    LMSequenceDataset,
    RandomizedStridedIndexSampler,
    DevicePrefetcher,
    BOSAlignedSampler,
)

try:
    from torch.amp import GradScaler as TorchGradScaler
except (ImportError, AttributeError):  # torch < 2.3 fallback
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore


TORCH_PRECISIONS = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _softcap_logits(logits: torch.Tensor, t: float) -> torch.Tensor:
    t_val = float(t)
    if t_val <= 0.0:
        return logits
    return t_val * torch.tanh(logits / t_val)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_memmap(path: os.PathLike[str] | str, dtype: str) -> np.ndarray:
    p = os.fspath(path)
    return np.load(p, mmap_mode="r") if p.endswith(".npy") else np.memmap(p, dtype=dtype)


def resolve_base_learning_rate(cfg: ExperimentConfig) -> float:
    if cfg.optimizer.lr is not None:
        return float(cfg.optimizer.lr)
    return 3e-4


def build_model(cfg: ExperimentConfig, device: torch.device, dtype: torch.dtype) -> TransformerLM:
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
        device=device,
        dtype=dtype,
        use_rope=cfg.model.use_rope,
        use_pre_norm=cfg.model.use_pre_norm,
        use_post_norm=cfg.model.use_post_norm,
        use_rmsnorm=cfg.model.use_rmsnorm,
        use_swiglu=cfg.model.use_swiglu,
        use_qk_norm=cfg.model.use_qk_norm,
        use_unet_residual=cfg.model.use_unet_residual,
        unet_gate_init=cfg.model.unet_gate_init,
        tie_embeddings=cfg.model.tie_embeddings,
        use_x0_mixin=cfg.model.use_x0_mixin,
        x0_gate_init=cfg.model.x0_gate_init,
        use_value_embeddings=cfg.model.use_value_embeddings,
        num_value_embeddings=cfg.model.num_value_embeddings,
        value_embed_lr_mul=cfg.model.value_embed_lr_mul,
        sa_lambda_init=cfg.model.sa_lambda_init,
        sa_lambda_lr_mul=cfg.model.sa_lambda_lr_mul,
        use_smear=cfg.model.use_smear,
        smear_lambda_init=cfg.model.smear_lambda_init,
        smear_gate_dim=cfg.model.smear_gate_dim,
        use_attn_gate=cfg.model.use_attn_gate,
        attn_gate_dim=cfg.model.attn_gate_dim,
        attn_gate_lr_mul=cfg.model.attn_gate_lr_mul,
    )
    # Optimizer metadata: vectors vs matrices; emb lr multipliers; tag fused qkvo
    try:
        model.embedding.embedding_table._optimizer_group = "vector"
        # Embedding modules configure their own learning-rate multipliers.
        for name, p in model.named_parameters():
            lname = name.lower()
            looks_fused = p.ndim == 2 and (p.shape[-1] % 4 == 0)
            in_attn = ("attn" in lname) or ("attention" in lname) or ("self_attn" in lname)
            name_hints = any(k in lname for k in ("qkv", "qkvo", "w_qkv", "wqkv", "q_proj", "k_proj", "v_proj", "o_proj"))
            if looks_fused and (in_attn or name_hints):
                rows, cols = p.shape
                col_ratio = cols // rows if rows > 0 and cols % rows == 0 else None
                row_ratio = rows // cols if cols > 0 and rows % cols == 0 else None
                if col_ratio == 4 or row_ratio == 4:
                    setattr(p, "_is_qkvo", True)
    except Exception:
        pass
    for _, p in model.named_parameters():
        if hasattr(p, "lr_mul") and not hasattr(p, "lr_mul_base"):
            try:
                p.lr_mul_base = float(p.lr_mul)
            except Exception:
                p.lr_mul_base = 1.0
    model = model.to(device=device, dtype=dtype)
    return model


def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    original_mode = model.training
    model.eval()
    losses: list[float] = []
    raw_losses: list[float] = []
    use_autocast = device.type == "cuda" and cfg.training.precision.lower() in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if cfg.training.precision.lower() == "float16" else torch.bfloat16
    eval_batch_size = cfg.training.eval_batch_size or cfg.training.batch_size
    loader = build_eval_loader(
        dataset,
        context_length=cfg.model.context_length,
        batch_size=eval_batch_size,
        device=device,
        rng=rng,
        full_sweep=cfg.training.eval_full_sweep,
        stride=cfg.training.eval_stride,
        drop_last=False,
        limit_windows=cfg.training.eval_limit_windows,
        num_batches=cfg.training.eval_batches,
    )
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X = X_batch.to(device, non_blocking=True) if X_batch.device != device else X_batch
            Y = Y_batch.to(device, non_blocking=True) if Y_batch.device != device else Y_batch
            with (torch.amp.autocast("cuda", dtype=amp_dtype) if use_autocast else nullcontext()):
                logits = model(X)
                raw_loss = cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
                eval_logits = logits
                softcap_value = cfg.training.logit_softcap
                if softcap_value is not None and softcap_value > 0.0:
                    eval_logits = _softcap_logits(eval_logits, softcap_value)
                loss = cross_entropy(eval_logits.reshape(-1, eval_logits.size(-1)), Y.reshape(-1))
                if cfg.training.z_loss_weight > 0.0:
                    z = torch.logsumexp(eval_logits, dim=-1).pow(2).mean()
                    loss = loss + cfg.training.z_loss_weight * z
            losses.append(float(loss.item()))
            raw_losses.append(float(raw_loss.item()))
    if original_mode:
        model.train()
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mean_raw_loss = float(np.mean(raw_losses)) if raw_losses else float("nan")
    try:
        ppl = float(math.exp(mean_loss))
    except OverflowError:
        ppl = float("inf")
    try:
        ppl_raw = float(math.exp(mean_raw_loss))
    except OverflowError:
        ppl_raw = float("inf")
    return {
        "loss": mean_loss,
        "perplexity": ppl,
        "loss_raw": mean_raw_loss,
        "perplexity_raw": ppl_raw,
    }


def save_training_checkpoint(model: TransformerLM, optimizer: torch.optim.Optimizer, step: int, checkpoint_dir: os.PathLike[str] | str, max_to_keep: int) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:08d}.pt")
    module_save_checkpoint(model, optimizer, iteration=step, out=ckpt_path)
    if max_to_keep > 0:
        existing = sorted([p for p in os.listdir(checkpoint_dir) if p.startswith("step_") and p.endswith(".pt")])
        to_remove = existing[:-max_to_keep]
        for fname in to_remove:
            try:
                os.remove(os.path.join(checkpoint_dir, fname))
            except OSError:
                pass
    return ckpt_path


def load_training_checkpoint(model: TransformerLM, optimizer: torch.optim.Optimizer, path: os.PathLike[str] | str) -> int:
    return int(module_load_checkpoint(src=path, model=model, optimizer=optimizer))


def train(cfg: ExperimentConfig) -> None:
    if cfg.data.train_path is None or cfg.data.val_path is None:
        raise ValueError("Both train_path and val_path must be provided")

    device = resolve_device(cfg.training.device)
    dtype = TORCH_PRECISIONS[cfg.training.precision.lower()]
    set_seed(cfg.training.seed)

    train_tokens = load_memmap(cfg.data.train_path, cfg.data.dtype)
    val_tokens = load_memmap(cfg.data.val_path, cfg.data.dtype)

    rng = np.random.default_rng(cfg.training.seed)
    val_rng = np.random.default_rng(cfg.training.seed + 1)  # Fixed seed for consistent validation

    model = build_model(cfg, device=device, dtype=dtype)
    if cfg.training.use_gradient_checkpointing:
        model.use_gradient_checkpointing = True
    if cfg.training.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.training.compile_mode)  # type: ignore

    model_params = list(model.parameters())
    base_lr = resolve_base_learning_rate(cfg)
    optimizer = build_optimizer(cfg, model_params, base_lr)
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group.get("lr", base_lr))

    bos_token_id = cfg.data.bos_token_id if cfg.data.bos_token_id is not None else cfg.data.end_of_text_token_id

    train_dataset = LMSequenceDataset(train_tokens, context_length=cfg.model.context_length)
    if cfg.training.align_to_bos and bos_token_id is not None:
        train_sampler = BOSAlignedSampler(
            train_tokens,
            context_length=cfg.model.context_length,
            bos_token_id=bos_token_id,
            rng=rng,
        )
    else:
        train_sampler = RandomizedStridedIndexSampler(len(train_dataset), rng=rng)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=True,
    )
    train_prefetcher = DevicePrefetcher(train_loader, device=device)

    def next_train_batch() -> tuple[torch.Tensor, torch.Tensor]:
        X, Y = train_prefetcher.next()
        if X is None or Y is None:
            train_prefetcher.reset()
            X, Y = train_prefetcher.next()
            if X is None or Y is None:
                raise RuntimeError("train dataloader produced no batches")
        return X, Y

    use_autocast = device.type == "cuda" and cfg.training.precision.lower() in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if cfg.training.precision.lower() == "float16" else torch.bfloat16
    autocast_scope = (lambda: torch.amp.autocast("cuda", dtype=amp_dtype)) if use_autocast else (lambda: nullcontext())
    scaler = TorchGradScaler(
        "cuda",
        enabled=use_autocast and cfg.training.precision.lower() == "float16",
    )

    ema_state: dict[str, torch.Tensor] | None = None
    if cfg.training.ema_decay is not None and cfg.training.ema_decay > 0.0:
        ema_state = {}
    ema_decay = cfg.training.ema_decay
    ema_update_every = max(1, int(cfg.training.ema_update_interval))

    grad_accum_default = max(1, cfg.training.grad_accum_steps)
    if cfg.training.step_interval and cfg.training.step_interval > 1 and grad_accum_default == 1:
        grad_accum_default = cfg.training.step_interval

    grad_accum_schedule = list(cfg.training.grad_accum_schedule or ())
    grad_accum_schedule.sort(key=lambda pair: pair[0])
    grad_accum_minutes = list(cfg.training.grad_accum_minutes_schedule or ())
    grad_accum_minutes.sort(key=lambda pair: pair[0])

    def resolve_grad_accum(step_idx: int, minutes_elapsed: float) -> int:
        value = grad_accum_default
        for boundary, target in grad_accum_schedule:
            if step_idx >= boundary:
                value = max(1, int(target))
            else:
                break
        for minute_boundary, target in grad_accum_minutes:
            if minutes_elapsed >= minute_boundary:
                value = max(1, int(target))
            else:
                break
        return value

    batch_tokens = int(cfg.training.batch_size) * int(cfg.model.context_length)
    reference_grad_accum = int(cfg.training.grad_accum_reference or grad_accum_default)
    reference_tokens_per_step = max(1, batch_tokens * reference_grad_accum)
    planned_total_tokens = max(1, cfg.training.total_steps * reference_tokens_per_step)

    def tokens_from_steps(steps: int | None) -> float:
        if steps is None:
            return 0.0
        return float(steps) * reference_tokens_per_step

    def resolve_tokens(step_value: int | None, token_value: int | None, *, default: float = 0.0) -> float:
        if token_value is not None:
            return float(token_value)
        if step_value is not None:
            return tokens_from_steps(step_value)
        return float(default)

    softcap_warmup_tokens = resolve_tokens(cfg.training.softcap_warmup_steps, cfg.training.softcap_warmup_tokens)
    zloss_warmup_tokens = resolve_tokens(cfg.training.zloss_warmup_steps, cfg.training.zloss_warmup_tokens)


    opt_cfg = cfg.optimizer
    muon_lr_warmup_tokens = resolve_tokens(opt_cfg.muon_lr_warmup_steps, opt_cfg.muon_lr_warmup_tokens)
    muon_lr_decay_start_tokens = resolve_tokens(opt_cfg.muon_lr_decay_start, opt_cfg.muon_lr_decay_start_tokens)
    muon_lr_decay_end_tokens = resolve_tokens(opt_cfg.muon_lr_decay_end, opt_cfg.muon_lr_decay_end_tokens, default=planned_total_tokens)

    muon_momentum_warm_tokens = resolve_tokens(opt_cfg.muon_warmup_steps, opt_cfg.muon_momentum_warmup_tokens)
    muon_momentum_cool_tokens = resolve_tokens(opt_cfg.muon_momentum_cooldown_steps, opt_cfg.muon_momentum_cooldown_tokens)

    def update_lr_multipliers(accum: int) -> None:
        denom = math.log2(max(accum, 2))
        factor = 1.0 / max(denom, 1.0)
        for p in model_params:
            base = getattr(p, "lr_mul_base", None)
            if base is not None:
                p.lr_mul = float(base) * factor
            elif hasattr(p, "lr_mul"):
                try:
                    base_val = float(p.lr_mul)
                except Exception:
                    base_val = 1.0
                p.lr_mul_base = base_val
                p.lr_mul = float(p.lr_mul_base) * factor

    seen_tokens = 0
    start_step = 0
    wandb_run = None
    try:
        import wandb  # optional

        if cfg.logging.use_wandb:
            # Prepare comprehensive config for wandb
            from cs336_basics.training.configs import serialize_config
            config_dict = serialize_config(cfg)

            wandb_run = wandb.init(
                project=cfg.logging.project,
                entity=cfg.logging.entity,
                name=cfg.logging.run_name,
                mode=cfg.logging.mode,
                config=config_dict,
            )

            # Log model architecture info
            if wandb_run is not None:
                try:
                    num_params = sum(p.numel() for p in model.parameters())
                    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    wandb_run.summary["model/total_parameters"] = num_params
                    wandb_run.summary["model/trainable_parameters"] = num_trainable_params
                    wandb_run.summary["model/parameter_size_mb"] = num_params * 4 / (1024**2)  # Assuming fp32
                except Exception:
                    pass
    except Exception:
        wandb_run = None

    train_start_time = time.perf_counter()
    previous_grad_accum: int | None = None

    for step in range(start_step, cfg.training.total_steps):
        current_step = step + 1
        minutes_elapsed = (time.perf_counter() - train_start_time) / 60.0
        current_grad_accum = resolve_grad_accum(step, minutes_elapsed)
        if previous_grad_accum != current_grad_accum:
            update_lr_multipliers(current_grad_accum)
            previous_grad_accum = current_grad_accum
        tokens_this_step = batch_tokens * current_grad_accum
        tokens_for_schedule = seen_tokens + tokens_this_step
        virtual_step = tokens_for_schedule / reference_tokens_per_step

        # Muon-specific LR scheduling drives both matrix and vector learning rates
        opt_cfg = cfg.optimizer
        matrix_base_lr = getattr(optimizer, "_matrix_base_lr", opt_cfg.matrix_base_lr or base_lr)
        vector_ratio = float(getattr(optimizer, "_vector_lr_ratio", getattr(opt_cfg, "vector_lr_ratio", 0.1)))
        has_matrix_group = any(g.get("group_type") == "matrix" for g in optimizer.param_groups)
        effective_vector_lr = matrix_base_lr * vector_ratio

        if has_matrix_group:

            decay_start_tokens = muon_lr_decay_start_tokens
            decay_end_tokens = max(decay_start_tokens + reference_tokens_per_step, muon_lr_decay_end_tokens)

            warmup_tokens_mu = muon_lr_warmup_tokens
            hold_tokens = max(0.0, decay_start_tokens - warmup_tokens_mu)
            cooldown_tokens = max(0.0, decay_end_tokens - decay_start_tokens)

            warmup_steps = warmup_tokens_mu / reference_tokens_per_step if warmup_tokens_mu > 0.0 else 0.0
            hold_steps = hold_tokens / reference_tokens_per_step if hold_tokens > 0.0 else 0.0
            cooldown_steps = cooldown_tokens / reference_tokens_per_step if cooldown_tokens > 0.0 else 0.0

            warmup_start_lr = opt_cfg.muon_lr_warmup_start
            if warmup_start_lr is None:
                warmup_start_lr = opt_cfg.muon_lr_final if opt_cfg.muon_lr_final is not None else matrix_base_lr
            final_lr_target = opt_cfg.muon_lr_final if opt_cfg.muon_lr_final is not None else warmup_start_lr

            t = max(0.0, virtual_step)
            matrix_lr_step = matrix_base_lr
            if warmup_steps > 0.0 and t <= warmup_steps:
                frac = min(1.0, t / max(1.0, warmup_steps))
                matrix_lr_step = warmup_start_lr + (matrix_base_lr - warmup_start_lr) * frac
            elif t <= warmup_steps + hold_steps or cooldown_steps <= 0.0:
                matrix_lr_step = matrix_base_lr
            else:
                t_cool = t - warmup_steps - hold_steps
                frac = min(1.0, max(0.0, t_cool) / max(1.0, cooldown_steps))
                matrix_lr_step = matrix_base_lr + (final_lr_target - matrix_base_lr) * frac

            matrix_lr_step = max(matrix_lr_step, 0.0)

            effective_vector_lr = matrix_lr_step * vector_ratio
            for group in optimizer.param_groups:
                gtype = group.get("group_type")
                if gtype == "matrix":
                    group["lr"] = matrix_lr_step
                else:
                    group["lr"] = effective_vector_lr
        else:
            fallback_matrix_lr = matrix_base_lr
            for group in optimizer.param_groups:
                gtype = group.get("group_type")
                if gtype == "matrix":
                    group["lr"] = fallback_matrix_lr
                else:
                    group["lr"] = effective_vector_lr
            effective_vector_lr = fallback_matrix_lr * vector_ratio

        # Momentum schedule for Muon matrices
        if isinstance(optimizer, torch.optim.Optimizer):
            mom_min = cfg.optimizer.muon_momentum_min or cfg.optimizer.muon_momentum
            mom_max = cfg.optimizer.muon_momentum_max or cfg.optimizer.muon_momentum
            warm_tokens = muon_momentum_warm_tokens
            cool_tokens = muon_momentum_cool_tokens

            def muon_momentum_at(tokens_progress: float) -> float:
                if warm_tokens > 0.0 and tokens_progress <= warm_tokens:
                    return mom_min + (mom_max - mom_min) * (tokens_progress / warm_tokens)
                cooldown_start = planned_total_tokens - cool_tokens
                if cool_tokens > 0.0 and tokens_progress >= cooldown_start:
                    k = (tokens_progress - cooldown_start) / max(1.0, cool_tokens)
                    k = min(max(k, 0.0), 1.0)
                    return mom_max + (mom_min - mom_max) * k
                return mom_max

            current_muon_momentum = muon_momentum_at(tokens_for_schedule)
            for group in optimizer.param_groups:
                if group.get("group_type") == "matrix":
                    group["momentum"] = current_muon_momentum

        # Always clear gradients for both matrix and vector parameter groups.
        if hasattr(optimizer, "param_groups"):
            zero_grads_for(optimizer, matrix=True, vector=True)
        else:
            optimizer.zero_grad(set_to_none=True)  # type: ignore

        softcap_frac = 1.0
        if cfg.training.logit_softcap is not None:
            denom = max(1.0, softcap_warmup_tokens)
            softcap_frac = min(1.0, tokens_for_schedule / denom)
        zloss_frac = 1.0
        if cfg.training.z_loss_weight > 0.0:
            denom = max(1.0, zloss_warmup_tokens)
            zloss_frac = min(1.0, tokens_for_schedule / denom)

        micro_losses: list[float] = []
        raw_losses: list[float] = []
        for _ in range(current_grad_accum):
            X, Y = next_train_batch()

            with autocast_scope():
                logits = model(X)
                with torch.no_grad():
                    raw_loss = cross_entropy(
                        logits.detach().reshape(-1, logits.size(-1)),
                        Y.reshape(-1),
                    )
                train_logits = logits
                softcap_value = cfg.training.logit_softcap
                if softcap_value is not None and softcap_value > 0.0 and model.training:
                    t_eff = float(softcap_value) / max(1e-3, softcap_frac)
                    train_logits = _softcap_logits(train_logits, t_eff)

                micro_loss = cross_entropy(train_logits.reshape(-1, train_logits.size(-1)), Y.reshape(-1))
                if model.training and cfg.training.z_loss_weight > 0.0:
                    z = torch.logsumexp(train_logits, dim=-1).pow(2).mean()
                    micro_loss = micro_loss + (cfg.training.z_loss_weight * zloss_frac) * z
                loss = micro_loss / current_grad_accum

            micro_losses.append(float(micro_loss.detach().cpu()))
            raw_losses.append(float(raw_loss.detach().cpu()))
            seen_tokens += batch_tokens

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        grad_norm = _grad_l2_norm(model_params)
        if cfg.training.grad_clip_norm is not None:
            gradient_clipping(model_params, cfg.training.grad_clip_norm)

        has_group_types = any(g.get("group_type") in ("matrix", "vector") for g in optimizer.param_groups)
        step_kwargs = {"matrix_step": True, "vector_step": True} if has_group_types else {}
        if scaler.is_enabled():
            if step_kwargs:
                optimizer.step(**step_kwargs)
            else:
                optimizer.step()
            scaler.update()
        else:
            if step_kwargs:
                optimizer.step(**step_kwargs)
            else:
                optimizer.step()

        if hasattr(optimizer, "param_groups"):
            zero_grads_for(optimizer, matrix=True, vector=True)
        else:
            optimizer.zero_grad(set_to_none=True)  # type: ignore

        if ema_state is not None and current_step % ema_update_every == 0:
            decay = float(ema_decay)
            one_minus = 1.0 - decay
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                target = ema_state.get(name)
                if target is None:
                    target = param.detach().float().cpu().clone()
                    ema_state[name] = target
                else:
                    target.mul_(decay).add_(param.detach().float().cpu(), alpha=one_minus)

        if (step + 1) % cfg.logging.log_interval == 0 or step == start_step:
            mean_micro_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
            mean_raw_loss = float(np.mean(raw_losses)) if raw_losses else float("nan")
            metrics: dict[str, float | None] = {
                "train/loss": mean_micro_loss,
                "train/loss_raw": mean_raw_loss,
                "optimizer/global_grad_norm": grad_norm,
                "tokens/seen": float(seen_tokens),
                "train/grad_accum": float(current_grad_accum),
            }

            # Learning rates
            vector_lr = None
            try:
                matrix_lr = next((g["lr"] for g in optimizer.param_groups if g.get("group_type") == "matrix"), None)
                vector_group_lr = next((g["lr"] for g in optimizer.param_groups if g.get("group_type") == "vector"), None)
                vector_lr = vector_group_lr if vector_group_lr is not None else effective_vector_lr
                if matrix_lr is not None:
                    metrics["optimizer/matrix_lr"] = float(matrix_lr)
                if matrix_lr is not None or vector_group_lr is not None:
                    metrics["optimizer/vector_lr"] = float(effective_vector_lr)
                # Also log generic LR if not using group-specific
                if matrix_lr is None and vector_group_lr is None and hasattr(optimizer, "param_groups"):
                    generic_lr = optimizer.param_groups[0].get("lr")
                    if generic_lr is not None:
                        metrics["optimizer/lr"] = float(generic_lr)
                metrics["optimizer/vector_lr_ratio"] = float(vector_ratio)
            except Exception:
                pass

            # Muon momentum
            try:
                for group in optimizer.param_groups:
                    if group.get("group_type") == "matrix" and "momentum" in group:
                        metrics["optimizer/muon_momentum"] = float(group["momentum"])
                        break
            except Exception:
                pass

            # Learning rate schedule progress (relative to base matrix lr)
            matrix_base_lr = getattr(optimizer, "_matrix_base_lr", opt_cfg.matrix_base_lr or base_lr)
            if matrix_lr is not None and matrix_base_lr not in (None, 0.0):
                metrics["schedule/lr_multiplier"] = float(matrix_lr / matrix_base_lr)

            # Training progress
            metrics["progress/step"] = step + 1

            # Scaler state (for FP16 training)
            try:
                if hasattr(scaler, "get_scale"):
                    scale_val = scaler.get_scale()
                    if scale_val is not None:
                        metrics["optimizer/grad_scaler_scale"] = float(scale_val)
            except Exception:
                pass

            # Batch statistics
            if micro_losses:
                metrics["train/loss_std"] = float(np.std(micro_losses))
                metrics["train/loss_min"] = float(np.min(micro_losses))
                metrics["train/loss_max"] = float(np.max(micro_losses))
            if raw_losses:
                metrics["train/loss_raw_std"] = float(np.std(raw_losses))
                metrics["train/loss_raw_min"] = float(np.min(raw_losses))
                metrics["train/loss_raw_max"] = float(np.max(raw_losses))

            if wandb_run is not None:
                try:
                    wandb_run.log({**metrics, "step": step + 1})  # type: ignore
                except Exception:
                    pass

            parts = [f"step={step+1}", f"train_loss={mean_micro_loss:.4f}"]
            if not math.isnan(mean_raw_loss):
                parts.append(f"train_loss_raw={mean_raw_loss:.4f}")
            if grad_norm is not None:
                parts.append(f"grad_norm={grad_norm:.4f}")
            # log lr if present
            if matrix_lr is not None:
                parts.append(f"matrix_lr={float(matrix_lr):.6f}")
            if vector_lr is not None:
                parts.append(f"vector_lr={float(vector_lr):.6f}")
            print(" ".join(parts))

        if (step + 1) % cfg.training.eval_interval == 0:
            ema_backup: dict[str, torch.Tensor] | None = None
            if ema_state is not None and cfg.training.use_ema_for_eval:
                ema_backup = {}
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    ema_tensor = ema_state.get(name)
                    if ema_tensor is None:
                        continue
                    ema_backup[name] = p.detach().clone()
                    p.data.copy_(ema_tensor.to(p.device, dtype=p.dtype))
            val_metrics = evaluate(model, val_tokens, cfg, device, val_rng)
            print(
                f"step={step+1} "
                f"val_loss_raw={val_metrics['loss_raw']:.4f} "
                f"val_ppl_raw={val_metrics['perplexity_raw']:.2f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_ppl={val_metrics['perplexity']:.2f}"
            )

            # Log validation metrics to wandb
            if wandb_run is not None:
                try:
                    val_log_metrics = {
                        "val/loss": val_metrics['loss'],
                        "val/perplexity": val_metrics['perplexity'],
                        "val/loss_raw": val_metrics['loss_raw'],
                        "val/perplexity_raw": val_metrics['perplexity_raw'],
                        "step": step + 1,
                    }
                    wandb_run.log(val_log_metrics)  # type: ignore
                except Exception:
                    pass

            if ema_backup is not None:
                for name, p in model.named_parameters():
                    if name in ema_backup:
                        p.data.copy_(ema_backup[name])

        if cfg.checkpoint.checkpoint_dir is not None and (step + 1) % cfg.checkpoint.save_interval == 0:
            ckpt_path = save_training_checkpoint(model, optimizer, step + 1, os.fspath(cfg.checkpoint.checkpoint_dir), cfg.checkpoint.max_to_keep)
            print(f"Saved checkpoint to {ckpt_path}")

    if cfg.checkpoint.checkpoint_dir is not None:
        ckpt_path = save_training_checkpoint(model, optimizer, cfg.training.total_steps, os.fspath(cfg.checkpoint.checkpoint_dir), cfg.checkpoint.max_to_keep)
        print(f"Saved final checkpoint to {ckpt_path}")

    # Log final summary statistics
    if wandb_run is not None:
        try:
            # Final memory stats
            if torch.cuda.is_available():
                wandb_run.summary["final/peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

            # Training totals
            wandb_run.summary["final/total_tokens_seen"] = float(seen_tokens)
            wandb_run.summary["final/total_steps"] = cfg.training.total_steps

            # Run final evaluation
            final_val_metrics = evaluate(model, val_tokens, cfg, device, val_rng)
            wandb_run.summary["final/val_loss"] = final_val_metrics['loss']
            wandb_run.summary["final/val_perplexity"] = final_val_metrics['perplexity']
            wandb_run.summary["final/val_bits_per_byte"] = final_val_metrics['loss'] / math.log(2)

            print(f"Final validation: loss={final_val_metrics['loss']:.4f}, perplexity={final_val_metrics['perplexity']:.2f}")
        except Exception as e:
            print(f"Warning: Could not log final summary: {e}")


def _grad_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = None
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad
        if grad.is_sparse:
            grad = grad.coalesce().values()
        sq = grad.pow(2).sum()
        total = sq if total is None else total + sq
    if total is None:
        return 0.0
    return float(torch.sqrt(total).cpu())
