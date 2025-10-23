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

from cs336_basics.modules.cross_entropy import cross_entropy
from cs336_basics.modules.dataloader import dataloader
from cs336_basics.modules.optimizers import AdamW, SGD, learning_rate_schedule
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
    dtype: str | None = None


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
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"


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
    model = model.to(device=device, dtype=dtype)
    return model


def build_optimizer(cfg: ExperimentConfig, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    opt_cfg = cfg.optimizer
    name = opt_cfg.name.lower()
    if name == "adamw":
        # Convert dtype string to torch.dtype
        optimizer_dtype = TORCH_PRECISIONS.get(opt_cfg.dtype.lower()) if opt_cfg.dtype else None
        return AdamW(
            parameters,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
            dtype=optimizer_dtype,
        )
    if name == "sgd":
        return SGD(parameters, lr=opt_cfg.lr)
    raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")


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
    compiler_mod = getattr(torch, "compiler", None)
    cudagraph_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None) if compiler_mod else None
    use_autocast = device.type == "cuda" and cfg.training.precision.lower() in {"float16", "bfloat16"}
    amp_dtype = None
    if use_autocast:
        amp_dtype = torch.float16 if cfg.training.precision.lower() == "float16" else torch.bfloat16
    with torch.no_grad():
        for _ in range(cfg.training.eval_batches):
            X, Y = dataloader(
                dataset=dataset,
                batch_size=cfg.training.batch_size,
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
                if cudagraph_step is not None:
                    cudagraph_step()
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

    compile_available = hasattr(torch, "compile")
    if cfg.training.use_torch_compile and compile_available:
        compile_kwargs: dict[str, Any] = {}
        if cfg.training.compile_mode:
            compile_kwargs["mode"] = cfg.training.compile_mode
        model = torch.compile(model, **compile_kwargs)
    elif cfg.training.use_torch_compile and not compile_available:
        print("torch.compile requested but not available in this PyTorch build; continuing without it.")

    model_params = list(model.parameters())
    optimizer = build_optimizer(cfg, model_params)

    schedule_cfg = cfg.lr_schedule
    schedule_active = schedule_cfg.enabled
    if schedule_active:
        alpha_max = schedule_cfg.alpha_max if schedule_cfg.alpha_max is not None else cfg.optimizer.lr
        alpha_min = schedule_cfg.alpha_min if schedule_cfg.alpha_min is not None else alpha_max
        warmup_steps = schedule_cfg.warmup_steps
        cosine_steps = schedule_cfg.cosine_steps if schedule_cfg.cosine_steps is not None else cfg.training.total_steps
        if cosine_steps <= warmup_steps:
            raise ValueError("lr_schedule.cosine_steps must be greater than lr_schedule.warmup_steps")
    else:
        alpha_max = cfg.optimizer.lr
        alpha_min = cfg.optimizer.lr
        warmup_steps = 0
        cosine_steps = cfg.training.total_steps

    grad_accum_steps = max(1, cfg.training.grad_accum_steps)
    if cfg.training.step_interval and cfg.training.step_interval > 1 and grad_accum_steps == 1:
        grad_accum_steps = cfg.training.step_interval

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
    scaler = torch.amp.GradScaler(
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
        if schedule_active:
            lr = learning_rate_schedule(
                step + 1,
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                T_w=warmup_steps,
                T_c=cosine_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = optimizer.param_groups[0]["lr"]

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
                micro_loss = cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    Y.reshape(-1),
                )
                loss = micro_loss / grad_accum_steps

            micro_losses.append(float(micro_loss.detach().cpu()))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if cfg.training.grad_clip_norm is not None:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_clipping(model_params, cfg.training.grad_clip_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if (step + 1) % cfg.logging.log_interval == 0 or step == start_step:
            mean_micro_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
            metrics = {"train_loss": mean_micro_loss, "lr": float(lr)}
            print(f"step={step + 1} train_loss={metrics['train_loss']:.4f} lr={metrics['lr']:.6f}")
            if wandb_run is not None:
                wandb.log({"train/loss": metrics["train_loss"], "train/lr": metrics["lr"], "step": step + 1})

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
