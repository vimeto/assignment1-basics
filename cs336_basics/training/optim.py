from __future__ import annotations

from typing import Any, Iterable
import torch

from cs336_basics.modules.optimizers import AdamW, SGD, MuonAdamW


TORCH_PRECISIONS = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_optimizer(cfg, parameters: Iterable[torch.nn.Parameter], base_lr: float) -> torch.optim.Optimizer:
    name = cfg.optimizer.name.lower()
    opt_cfg = cfg.optimizer
    if name == "adamw":
        optimizer_dtype = TORCH_PRECISIONS.get(opt_cfg.dtype.lower()) if opt_cfg.dtype else None
        return AdamW(
            parameters,
            lr=base_lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
            dtype=optimizer_dtype,
        )
    if name == "sgd":
        return SGD(parameters, lr=base_lr)
    if name == "muon":
        matrix_wd = opt_cfg.matrix_weight_decay if opt_cfg.matrix_weight_decay is not None else opt_cfg.weight_decay
        vector_wd = opt_cfg.vector_weight_decay if opt_cfg.vector_weight_decay is not None else 0.0
        matrix_lr = opt_cfg.matrix_base_lr if opt_cfg.matrix_base_lr is not None else opt_cfg.lr
        vector_lr = opt_cfg.vector_base_lr if opt_cfg.vector_base_lr is not None else opt_cfg.lr
        optimizer = MuonAdamW(
            parameters,
            lr=base_lr,
            weight_decay=opt_cfg.weight_decay,
            matrix_weight_decay=matrix_wd,
            vector_weight_decay=vector_wd,
            momentum=opt_cfg.muon_momentum,
            momentum_min=opt_cfg.muon_momentum_min,
            momentum_max=opt_cfg.muon_momentum_max or opt_cfg.muon_momentum,
            warmup_steps=opt_cfg.muon_warmup_steps,
            ns_steps=opt_cfg.muon_ns_steps,
            vector_lr_multiplier=opt_cfg.vector_lr_multiplier,
            betas=opt_cfg.betas,
            vector_eps=opt_cfg.eps,
            matrix_base_lr=matrix_lr,
            vector_base_lr=vector_lr,
        )
        setattr(optimizer, "_matrix_base_lr", matrix_lr)
        setattr(optimizer, "_vector_base_lr", vector_lr)
        return optimizer
    raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")


def zero_grads_for(optim: torch.optim.Optimizer, *, matrix: bool, vector: bool) -> None:
    for g in optim.param_groups:
        gtype = g.get("group_type")
        if (gtype == "matrix" and matrix) or (gtype == "vector" and vector) or (gtype is None and (matrix or vector)):
            for p in g["params"]:
                if p.grad is not None:
                    p.grad = None

