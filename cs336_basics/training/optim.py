from __future__ import annotations

from typing import Any, Iterable
import torch

from cs336_basics.modules.optimizers import AdamW, SGD, MuonAdamW, NorMuon, DistAdam


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
        vector_lr = opt_cfg.vector_base_lr if opt_cfg.vector_base_lr is not None else matrix_lr * float(getattr(opt_cfg, "vector_lr_ratio", 0.1))
        matrix_state_dtype = TORCH_PRECISIONS.get(opt_cfg.matrix_state_dtype.lower()) if opt_cfg.matrix_state_dtype else torch.float32
        vector_state_dtype = TORCH_PRECISIONS.get(opt_cfg.vector_state_dtype.lower()) if opt_cfg.vector_state_dtype else torch.float32
        vector_lr_multiplier = opt_cfg.vector_lr_multiplier
        if vector_lr_multiplier is None:
            vector_lr_multiplier = float(getattr(opt_cfg, "vector_lr_ratio", 0.1))
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
            vector_lr_multiplier=vector_lr_multiplier,
            betas=opt_cfg.betas,
            vector_eps=opt_cfg.eps,
            matrix_base_lr=matrix_lr,
            vector_base_lr=vector_lr,
            matrix_state_dtype=matrix_state_dtype,
            vector_state_dtype=vector_state_dtype,
        )
        setattr(optimizer, "_matrix_base_lr", matrix_lr)
        setattr(optimizer, "_vector_base_lr", vector_lr)
        setattr(optimizer, "_vector_lr_ratio", float(getattr(opt_cfg, "vector_lr_ratio", vector_lr / max(matrix_lr, 1e-12))))
        return optimizer
    raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")


def split_params_two_optimizers(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    adam_params: list[torch.nn.Parameter] = []
    muon_params: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        is_embedding = ("embedding" in lname)
        is_head = ("ffn.linear" in lname) or ("lm_head" in lname)
        is_scalar = (p.ndim <= 1)
        is_gate = ("gate" in lname) or ("resid_" in lname)

        if is_embedding or is_head or is_scalar:
            adam_params.append(p)
        else:
            # default heavy weights (attn/mlp matrices) go to Muon
            muon_params.append(p)

    return adam_params, muon_params


def build_optimizers(cfg, model: torch.nn.Module) -> list[torch.optim.Optimizer]:
    adam_params, muon_params = split_params_two_optimizers(model)
    # DistAdam (vectors, head, embeddings)
    opt_vec = DistAdam(
        adam_params,
        lr=cfg.optimizer.vector_base_lr or cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.vector_weight_decay if cfg.optimizer.vector_weight_decay is not None else cfg.optimizer.weight_decay,
    )
    # NorMuon (matrices)
    opt_mat = NorMuon(
        muon_params,
        lr=cfg.optimizer.matrix_base_lr or cfg.optimizer.lr,
        momentum=cfg.optimizer.muon_momentum,
        weight_decay=cfg.optimizer.matrix_weight_decay if cfg.optimizer.matrix_weight_decay is not None else cfg.optimizer.weight_decay,
        normalizer_beta=0.95,
        normalizer_eps=1e-10,
    )
    return [opt_vec, opt_mat]



def zero_grads_for(optim: torch.optim.Optimizer, *, matrix: bool, vector: bool) -> None:
    for g in optim.param_groups:
        gtype = g.get("group_type")
        if (gtype == "matrix" and matrix) or (gtype == "vector" and vector) or (gtype is None and (matrix or vector)):
            for p in g["params"]:
                if p.grad is not None:
                    p.grad = None
