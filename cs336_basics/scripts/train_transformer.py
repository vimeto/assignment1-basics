from __future__ import annotations

import argparse
import dataclasses
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

try:
    from torch.amp import GradScaler as TorchGradScaler  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore

from cs336_basics.modules.cross_entropy import cross_entropy
from cs336_basics.modules.dataloader import dataloader
from cs336_basics.modules.optimizers import (
    AdamW,
    SGD,
    MuonAdamW,
    learning_rate_schedule,
    linear_warmup_decay,
)
from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.modules.checkpointing import (
    save_checkpoint as module_save_checkpoint,
    load_checkpoint as module_load_checkpoint,
)
from cs336_basics.modules.gradient import gradient_clipping

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0
    # Ablation study flags (additive norm support)
    use_rope: bool = True
    use_pre_norm: bool = True
    use_post_norm: bool = False
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    use_qk_norm: bool = False
    use_unet_residual: bool = False
    unet_gate_init: float = 0.1


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.1
    matrix_weight_decay: float | None = None
    vector_weight_decay: float | None = None
    dtype: str | None = None
    vector_lr_multiplier: float = 1.0
    muon_momentum: float = 0.95
    muon_momentum_min: float | None = None
    muon_momentum_max: float | None = None
    muon_warmup_steps: int = 0
    muon_ns_steps: int = 5
    depth_mu_enabled: bool = False
    depth_mu_base_lr: float = 4e-4
    depth_mu_reference_depth: int = 16
    matrix_base_lr: float | None = None
    vector_base_lr: float | None = None
    muon_lr_decay_start: int | None = None
    muon_lr_decay_end: int | None = None
    muon_lr_final: float | None = None


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 1000
    batch_size: int = 32
    grad_accum_steps: int = 1
    seed: int = 1234
    device: str | None = None
    precision: str = "float32"
    grad_clip_norm: float | None = None
    step_interval: int = 1
    eval_interval: int = 200
    eval_batches: int = 16
    eval_batch_size: int | None = None
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_gradient_checkpointing: bool = False
    logit_softcap: Optional[float] = 15.0  # Gemma2-style tanh cap during TRAIN only
    z_loss_weight: float = 1e-4           # tiny Z-loss for stability


@dataclass(frozen=True)
class DataConfig:
    train_path: Path | None = None
    val_path: Path | None = None
    dtype: str = "uint16"


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_dir: Path | None = None
    save_interval: int = 200
    max_to_keep: int = 5
    resume_path: Path | None = None


@dataclass(frozen=True)
class LoggingConfig:
    log_interval: int = 50
    use_wandb: bool = False
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"


@dataclass(frozen=True)
class LearningRateScheduleConfig:
    enabled: bool = True
    alpha_max: float | None = None
    alpha_min: float | None = None
    warmup_steps: int = 0
    cosine_steps: int | None = None
    schedule_type: str = "cosine"
    warmup_tokens: int | None = None
    decay_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()
    lr_schedule: LearningRateScheduleConfig = LearningRateScheduleConfig()


TORCH_PRECISIONS: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

if hasattr(torch, "_dynamo"):
    torch._dynamo.config.capture_scalar_outputs = True  # type: ignore[attr-defined]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _replace_dataclass(instance: Any, overrides: dict[str, Any]) -> Any:
    if overrides is None:
        return instance
    kwargs: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            current = getattr(instance, key)
            kwargs[key] = _replace_dataclass(current, value)
        else:
            kwargs[key] = value
    return dataclasses.replace(instance, **kwargs)


def _coerce_config_types(cfg: ExperimentConfig) -> ExperimentConfig:
    data = cfg.data
    checkpoint = cfg.checkpoint

    data_kwargs: dict[str, Any] = {}
    if data.train_path is not None and not isinstance(data.train_path, Path):
        data_kwargs["train_path"] = Path(data.train_path)
    if data.val_path is not None and not isinstance(data.val_path, Path):
        data_kwargs["val_path"] = Path(data.val_path)
    if data_kwargs:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **data_kwargs))

    ckpt_kwargs: dict[str, Any] = {}
    if checkpoint.checkpoint_dir is not None and not isinstance(checkpoint.checkpoint_dir, Path):
        ckpt_kwargs["checkpoint_dir"] = Path(checkpoint.checkpoint_dir)
    if checkpoint.resume_path is not None and not isinstance(checkpoint.resume_path, Path):
        ckpt_kwargs["resume_path"] = Path(checkpoint.resume_path)
    if ckpt_kwargs:
        cfg = dataclasses.replace(cfg, checkpoint=dataclasses.replace(cfg.checkpoint, **ckpt_kwargs))

    opt = cfg.optimizer
    if not isinstance(opt.betas, tuple):
        cfg = dataclasses.replace(cfg, optimizer=dataclasses.replace(opt, betas=tuple(opt.betas)))

    return cfg


def load_config(config_path: Path | None) -> ExperimentConfig:
    base = ExperimentConfig()
    if config_path:
        overrides = _load_json(config_path)
        base = _replace_dataclass(base, overrides)
    return _coerce_config_types(base)


def apply_cli_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    data_kwargs: dict[str, Any] = {}
    if args.train_path is not None:
        data_kwargs["train_path"] = Path(args.train_path)
    if args.val_path is not None:
        data_kwargs["val_path"] = Path(args.val_path)
    if data_kwargs:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **data_kwargs))

    training_kwargs: dict[str, Any] = {}
    if args.device is not None:
        training_kwargs["device"] = args.device
    if getattr(args, "grad_accum_steps", None) is not None:
        training_kwargs["grad_accum_steps"] = max(1, args.grad_accum_steps)
    if getattr(args, "torch_compile", False):
        training_kwargs["use_torch_compile"] = True
    if getattr(args, "no_torch_compile", False):
        training_kwargs["use_torch_compile"] = False
    if getattr(args, "compile_mode", None) is not None:
        training_kwargs["compile_mode"] = args.compile_mode
    if training_kwargs:
        cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, **training_kwargs))

    checkpoint_kwargs: dict[str, Any] = {}
    if args.checkpoint_dir is not None:
        checkpoint_kwargs["checkpoint_dir"] = Path(args.checkpoint_dir)
    if args.resume is not None:
        checkpoint_kwargs["resume_path"] = Path(args.resume)
    if checkpoint_kwargs:
        cfg = dataclasses.replace(cfg, checkpoint=dataclasses.replace(cfg.checkpoint, **checkpoint_kwargs))

    logging_kwargs: dict[str, Any] = {}
    if args.wandb_project is not None:
        logging_kwargs["project"] = args.wandb_project
    if args.wandb_entity is not None:
        logging_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        logging_kwargs["run_name"] = args.wandb_run_name
    if args.no_wandb:
        logging_kwargs["use_wandb"] = False
    if args.enable_wandb:
        logging_kwargs["use_wandb"] = True
    if logging_kwargs:
        cfg = dataclasses.replace(cfg, logging=dataclasses.replace(cfg.logging, **logging_kwargs))

    lr_schedule_kwargs: dict[str, Any] = {}
    if getattr(args, "lr_schedule", False):
        lr_schedule_kwargs["enabled"] = True
    if getattr(args, "no_lr_schedule", False):
        lr_schedule_kwargs["enabled"] = False
    if args.lr_alpha_max is not None:
        lr_schedule_kwargs["alpha_max"] = args.lr_alpha_max
    if args.lr_alpha_min is not None:
        lr_schedule_kwargs["alpha_min"] = args.lr_alpha_min
    if args.lr_warmup_steps is not None:
        lr_schedule_kwargs["warmup_steps"] = args.lr_warmup_steps
    if args.lr_cosine_steps is not None:
        lr_schedule_kwargs["cosine_steps"] = args.lr_cosine_steps
    if lr_schedule_kwargs:
        cfg = dataclasses.replace(cfg, lr_schedule=dataclasses.replace(cfg.lr_schedule, **lr_schedule_kwargs))

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--config", type=Path, help="Optional path to JSON config file", default=None)
    parser.add_argument("--train-path", type=Path, default=None, help="Path to training tokens memmap")
    parser.add_argument("--val-path", type=Path, default=None, help="Path to validation tokens memmap")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory for saving checkpoints")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from this checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, etc.)")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Number of gradient accumulation steps")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--lr-schedule", action="store_true", help="Enable learning rate scheduling")
    parser.add_argument("--no-lr-schedule", action="store_true", help="Disable learning rate scheduling")
    parser.add_argument("--lr-alpha-max", type=float, default=None, help="Maximum learning rate for schedule")
    parser.add_argument("--lr-alpha-min", type=float, default=None, help="Minimum learning rate for schedule")
    parser.add_argument("--lr-warmup-steps", type=int, default=None, help="Warmup steps for schedule")
    parser.add_argument("--lr-cosine-steps", type=int, default=None, help="Cosine decay steps before floor")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for the model")
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile for the model")
    parser.add_argument("--compile-mode", type=str, default=None, help="torch.compile mode (e.g., reduce-overhead)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_base_learning_rate(cfg: ExperimentConfig) -> float:
    opt_cfg = cfg.optimizer
    if opt_cfg.depth_mu_enabled:
        depth = cfg.model.num_layers
        if depth <= 0:
            raise ValueError("Model depth must be positive for Depth-μ scaling")
        reference_depth = max(1, opt_cfg.depth_mu_reference_depth)
        return opt_cfg.depth_mu_base_lr * (reference_depth / depth)
    return opt_cfg.lr


def load_memmap(path: Path, dtype: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    # Use np.load with mmap_mode to properly handle .npy file format
    # np.memmap doesn't understand .npy headers and can read garbage data
    return np.load(path, mmap_mode='r')


def build_model(cfg: ExperimentConfig, device: torch.device, dtype: torch.dtype) -> TransformerLM:
    model_cfg = cfg.model
    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        rope_theta=model_cfg.rope_theta,
        device=device,
        use_rope=model_cfg.use_rope,
        use_pre_norm=model_cfg.use_pre_norm,
        use_post_norm=model_cfg.use_post_norm,
        use_rmsnorm=model_cfg.use_rmsnorm,
        use_swiglu=model_cfg.use_swiglu,
        use_qk_norm=model_cfg.use_qk_norm,
        use_unet_residual=model_cfg.use_unet_residual,
        unet_gate_init=model_cfg.unet_gate_init,
    )
    if isinstance(model.layers, list):
        model.layers = torch.nn.ModuleList(model.layers)  # type: ignore[attr-defined]

    # mark embeddings for vector optimizer treatment when using Muon
    model.embedding.embedding_table._optimizer_group = "vector"
    model.ffn.linear._optimizer_group = "vector"
    model = model.to(device=device, dtype=dtype)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and name.startswith("layers"):
            param._weight_decay = 0.1
        else:
            param._weight_decay = 0.0
    return model


def build_optimizer(
    cfg: ExperimentConfig,
    parameters: Iterable[torch.nn.Parameter],
    base_lr: float,
) -> torch.optim.Optimizer:
    opt_cfg = cfg.optimizer
    name = opt_cfg.name.lower()
    if name == "adamw":
        # Convert dtype string to torch.dtype
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


def _grad_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
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


def serialize_config(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {field.name: serialize_config(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [serialize_config(x) for x in obj]
    if isinstance(obj, list):
        return [serialize_config(x) for x in obj]
    if isinstance(obj, dict):
        return {key: serialize_config(value) for key, value in obj.items()}
    return obj


def save_training_checkpoint(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_cfg: CheckpointConfig,
) -> Path:
    if checkpoint_cfg.checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided to save checkpoints")

    checkpoint_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_cfg.checkpoint_dir / f"step_{step:08d}.pt"
    module_save_checkpoint(model, optimizer, iteration=step, out=ckpt_path)

    if checkpoint_cfg.max_to_keep > 0:
        existing = sorted(checkpoint_cfg.checkpoint_dir.glob("step_*.pt"))
        to_remove = existing[:-checkpoint_cfg.max_to_keep]
        for old in to_remove:
            try:
                old.unlink()
            except OSError:
                pass

    return ckpt_path


def load_training_checkpoint(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> int:
    return int(module_load_checkpoint(src=path, model=model, optimizer=optimizer))


def maybe_init_wandb(cfg: ExperimentConfig) -> Any:
    log_cfg = cfg.logging
    if not log_cfg.use_wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed but use_wandb=True")

    init_kwargs = {
        "project": log_cfg.project,
        "entity": log_cfg.entity,
        "name": log_cfg.run_name,
        "mode": log_cfg.mode,
        "config": serialize_config(cfg),
    }
    return wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})


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
    use_autocast = device.type == "cuda" and cfg.training.precision.lower() in {"float16", "bfloat16"}
    amp_dtype = None
    if use_autocast:
        amp_dtype = torch.float16 if cfg.training.precision.lower() == "float16" else torch.bfloat16
    eval_batch_size = cfg.training.eval_batch_size or cfg.training.batch_size
    with torch.no_grad():
        for _ in range(cfg.training.eval_batches):
            X, Y = dataloader(
                dataset=dataset,
                batch_size=eval_batch_size,
                context_length=cfg.model.context_length,
                device=device.type,
                rng=rng,
                non_blocking=True,
            )
            with (
                torch.amp.autocast("cuda", dtype=amp_dtype)
                if use_autocast
                else nullcontext()
            ):
                logits = model(X)
                loss = cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    Y.reshape(-1),
                )
            losses.append(float(loss.item()))
    if original_mode:
        model.train()
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    try:
        ppl = float(math.exp(mean_loss))
    except OverflowError:
        ppl = float("inf")
    return {"loss": mean_loss, "perplexity": ppl}


def train(cfg: ExperimentConfig) -> None:
    if cfg.data.train_path is None:
        raise ValueError("Training dataset path must be provided")
    if cfg.data.val_path is None:
        raise ValueError("Validation dataset path must be provided")

    device = resolve_device(cfg.training.device)
    dtype = TORCH_PRECISIONS.get(cfg.training.precision.lower())
    if dtype is None:
        raise ValueError(f"Unsupported precision: {cfg.training.precision}")

    set_seed(cfg.training.seed)

    train_tokens = load_memmap(cfg.data.train_path, cfg.data.dtype)
    val_tokens = load_memmap(cfg.data.val_path, cfg.data.dtype)

    train_rng = np.random.default_rng(cfg.training.seed)
    eval_rng = np.random.default_rng(cfg.training.seed + 1)

    model = build_model(cfg, device=device, dtype=dtype)

    if cfg.training.use_gradient_checkpointing:
        model.use_gradient_checkpointing = True

    compile_available = hasattr(torch, "compile")
    use_compile = cfg.training.use_torch_compile
    if use_compile and not compile_available:
        print("torch.compile requested but not available; continuing without it.")
        use_compile = False
    if use_compile:
        compile_kwargs: dict[str, Any] = {}
        if cfg.training.compile_mode:
            compile_kwargs["mode"] = cfg.training.compile_mode
        model = torch.compile(model, **compile_kwargs)

    model_params = list(model.parameters())
    base_lr = resolve_base_learning_rate(cfg)
    optimizer = build_optimizer(cfg, model_params, base_lr)
    setattr(optimizer, "_base_lr", base_lr)
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group.get("lr", base_lr))

    grad_accum_steps = max(1, cfg.training.grad_accum_steps)
    if cfg.training.step_interval and cfg.training.step_interval > 1 and grad_accum_steps == 1:
        grad_accum_steps = cfg.training.step_interval

    schedule_cfg = cfg.lr_schedule
    schedule_active = schedule_cfg.enabled
    schedule_type = getattr(schedule_cfg, "schedule_type", "cosine").lower()
    batch_tokens = int(cfg.training.batch_size) * int(cfg.model.context_length)
    tokens_per_step = max(1, batch_tokens * grad_accum_steps)
    warmup_steps = schedule_cfg.warmup_steps
    if schedule_cfg.warmup_tokens is not None:
        warmup_steps = max(1, math.ceil(schedule_cfg.warmup_tokens / tokens_per_step))

    if schedule_active:
        if schedule_type in {"linear", "linear_to_zero"}:
            total_schedule_steps = schedule_cfg.decay_tokens
            if total_schedule_steps is not None:
                total_schedule_steps = max(warmup_steps + 1, math.ceil(total_schedule_steps / tokens_per_step))
            else:
                total_schedule_steps = cfg.training.total_steps
            if total_schedule_steps <= warmup_steps:
                raise ValueError("Linear schedule requires decay region after warmup")
        else:
            alpha_max = schedule_cfg.alpha_max if schedule_cfg.alpha_max is not None else base_lr
            alpha_min = schedule_cfg.alpha_min if schedule_cfg.alpha_min is not None else alpha_max
            cosine_steps = schedule_cfg.cosine_steps if schedule_cfg.cosine_steps is not None else cfg.training.total_steps
            if cosine_steps <= warmup_steps:
                raise ValueError("lr_schedule.cosine_steps must be greater than lr_schedule.warmup_steps")
    else:
        alpha_max = base_lr
        alpha_min = base_lr
        cosine_steps = cfg.training.total_steps

    use_autocast = device.type == "cuda" and cfg.training.precision.lower() in {"float16", "bfloat16"}
    amp_dtype = None
    if use_autocast:
        amp_dtype = torch.float16 if cfg.training.precision.lower() == "float16" else torch.bfloat16

    if use_autocast:
        def autocast_scope():
            return torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        def autocast_scope():
            return nullcontext()
    scaler = TorchGradScaler(
        "cuda",
        enabled=use_autocast and cfg.training.precision.lower() == "float16",
    )

    start_step = 0
    if cfg.checkpoint.resume_path is not None:
        start_step = load_training_checkpoint(model, optimizer, cfg.checkpoint.resume_path)
        print(f"Resumed from checkpoint {cfg.checkpoint.resume_path} at step {start_step}")

    wandb_run = maybe_init_wandb(cfg)

    model.train()
    for step in range(start_step, cfg.training.total_steps):
        current_step = step + 1
        if schedule_active:
            if schedule_type in {"linear", "linear_to_zero"}:
                lr_sched = linear_warmup_decay(
                    current_step,
                    alpha_max=base_lr,
                    warmup_steps=warmup_steps,
                    total_steps=total_schedule_steps,
                )
            else:
                lr_sched = learning_rate_schedule(
                    current_step,
                    alpha_max=alpha_max,
                    alpha_min=alpha_min,
                    T_w=warmup_steps,
                    T_c=cosine_steps,
                )
        else:
            lr_sched = None

        if isinstance(optimizer, MuonAdamW):
            opt_cfg = cfg.optimizer
            matrix_base_lr = getattr(optimizer, "_matrix_base_lr", base_lr)
            vector_base_lr = getattr(optimizer, "_vector_base_lr", base_lr)
            decay_start = opt_cfg.muon_lr_decay_start
            decay_end = opt_cfg.muon_lr_decay_end if opt_cfg.muon_lr_decay_end is not None else cfg.training.total_steps
            final_matrix_lr = opt_cfg.muon_lr_final if opt_cfg.muon_lr_final is not None else matrix_base_lr
            for group in optimizer.param_groups:
                gtype = group.get("group_type")
                if gtype == "vector":
                    if lr_sched is not None:
                        group["lr"] = lr_sched
                    else:
                        group["lr"] = group.get("base_lr", vector_base_lr)
                elif gtype == "matrix":
                    matrix_lr = group.get("base_lr", matrix_base_lr)
                    if decay_start is not None:
                        if decay_end <= decay_start:
                            decay_end = decay_start + 1
                        if current_step >= decay_start:
                            progress = min(1.0, max(0.0, (current_step - decay_start) / max(1, decay_end - decay_start)))
                            matrix_lr = matrix_base_lr + (final_matrix_lr - matrix_base_lr) * progress
                    group["lr"] = matrix_lr
        else:
            if lr_sched is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr_sched
        # capture lr used for logging (vector LR if Muon, else first group)
        if isinstance(optimizer, MuonAdamW):
            vector_group = next((g for g in optimizer.param_groups if g.get("group_type") == "vector"), None)
            if vector_group is not None:
                lr = float(vector_group.get("effective_lr", vector_group["lr"]))
            else:
                lr = float(optimizer.param_groups[0]["lr"])
        else:
            lr = float(optimizer.param_groups[0]["lr"])

        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []

        for _ in range(grad_accum_steps):
            X, Y = dataloader(
                dataset=train_tokens,
                batch_size=cfg.training.batch_size,
                context_length=cfg.model.context_length,
                device=device.type,
                rng=train_rng,
                non_blocking=True,
            )

            with autocast_scope():
                logits = model(X)
                if cfg.training.logit_softcap is not None and model.training:
                    t = float(cfg.training.logit_softcap)
                    logits = torch.tanh(logits / t) * t

                micro_loss = cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    Y.reshape(-1),
                )
                if model.training and cfg.training.z_loss_weight > 0.0:
                    z = torch.logsumexp(logits, dim=-1).pow(2).mean()
                    micro_loss = micro_loss + cfg.training.z_loss_weight * z
                loss = micro_loss / grad_accum_steps

            micro_losses.append(float(micro_loss.detach().cpu()))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        grad_norm = _grad_l2_norm(model_params)

        if cfg.training.grad_clip_norm is not None:
            gradient_clipping(model_params, cfg.training.grad_clip_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if (step + 1) % cfg.logging.log_interval == 0 or step == start_step:
            mean_micro_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
            metrics: dict[str, float | None] = {
                "train/loss": mean_micro_loss,
                "optimizer/global_grad_norm": grad_norm,
            }

            if isinstance(optimizer, MuonAdamW):
                matrix_lr = None
                vector_lr = None
                muon_momentum = None
                for group in optimizer.param_groups:
                    gtype = group.get("group_type")
                    if gtype == "matrix":
                        matrix_lr = float(group.get("effective_lr", group["lr"]))
                        muon_momentum = float(group.get("current_momentum", optimizer.defaults.get("momentum", 0.0)))
                    elif gtype == "vector":
                        vector_lr = float(group.get("effective_lr", group["lr"]))
                metrics["optimizer/matrix_lr"] = matrix_lr
                metrics["optimizer/vector_lr"] = vector_lr
                metrics["optimizer/muon_momentum"] = muon_momentum
                console_parts = [
                    f"step={step + 1}",
                    f"train_loss={mean_micro_loss:.4f}",
                ]
                if grad_norm is not None:
                    console_parts.append(f"grad_norm={grad_norm:.4f}")
                if matrix_lr is not None:
                    console_parts.append(f"matrix_lr={matrix_lr:.6f}")
                if vector_lr is not None:
                    console_parts.append(f"vector_lr={vector_lr:.6f}")
                print(" ".join(console_parts))
            else:
                metrics["optimizer/lr"] = float(lr)
                console_parts = [
                    f"step={step + 1}",
                    f"train_loss={mean_micro_loss:.4f}",
                ]
                if grad_norm is not None:
                    console_parts.append(f"grad_norm={grad_norm:.4f}")
                console_parts.append(f"lr={lr:.6f}")
                print(" ".join(console_parts))
            if wandb_run is not None:
                wandb.log({**metrics, "step": step + 1})

        if (step + 1) % cfg.training.eval_interval == 0:
            val_metrics = evaluate(model, val_tokens, cfg, device, eval_rng)
            print(
                "step="
                f"{step + 1} val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['perplexity']:.2f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "eval/loss": val_metrics["loss"],
                        "eval/perplexity": val_metrics["perplexity"],
                        "step": step + 1,
                    }
                )

        if (
            cfg.checkpoint.checkpoint_dir is not None
            and (step + 1) % cfg.checkpoint.save_interval == 0
        ):
            ckpt_path = save_training_checkpoint(model, optimizer, step + 1, cfg.checkpoint)
            print(f"Saved checkpoint to {ckpt_path}")

    if cfg.checkpoint.checkpoint_dir is not None:
        ckpt_path = save_training_checkpoint(model, optimizer, cfg.training.total_steps, cfg.checkpoint)
        print(f"Saved final checkpoint to {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    train(cfg)


if __name__ == "__main__":
    main()
