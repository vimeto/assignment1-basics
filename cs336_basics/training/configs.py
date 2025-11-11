from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0
    use_rope: bool = True
    use_pre_norm: bool = True
    use_post_norm: bool = False
    use_rmsnorm: bool = True
    use_swiglu: bool = False
    use_qk_norm: bool = True
    use_unet_residual: bool = True
    unet_gate_init: float = 0.1
    use_x0_mixin: bool = True
    x0_gate_init: float = 0.1
    tie_embeddings: bool = False
    # Value embeddings (nanoGPT-style token value embeddings)
    use_value_embeddings: bool = False
    num_value_embeddings: int = 3
    value_embed_lr_mul: float = 50.0
    # SA lambdas for mixing value and value embeddings
    sa_lambda_init: Tuple[float, float] = (0.5, 0.5)
    sa_lambda_lr_mul: float = 5.0
    # Smear gate (nanoGPT-style): smears token embeddings forward 1 position
    use_smear: bool = False
    smear_lambda_init: float = 0.0
    smear_gate_dim: int = 12
    # Attention gating
    use_attn_gate: bool = False
    attn_gate_dim: int = 12
    attn_gate_lr_mul: float = 5.0


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "muon"
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.1
    matrix_weight_decay: float | None = None
    vector_weight_decay: float | None = None
    dtype: str | None = None
    vector_lr_multiplier: float | None = None
    vector_lr_ratio: float = 0.1
    muon_momentum: float = 0.95
    muon_momentum_min: float | None = None
    muon_momentum_max: float | None = None
    muon_warmup_steps: int = 0
    muon_momentum_cooldown_steps: int = 0
    muon_ns_steps: int = 5
    matrix_base_lr: float | None = None
    vector_base_lr: float | None = None
    muon_lr_decay_start: int | None = None
    muon_lr_decay_end: int | None = None
    muon_lr_decay_start_tokens: int | None = None
    muon_lr_decay_end_tokens: int | None = None
    muon_lr_final: float | None = None
    muon_lr_warmup_steps: int | None = None
    muon_lr_warmup_start: float | None = None
    muon_lr_warmup_tokens: int | None = None
    muon_momentum_warmup_tokens: int | None = None
    muon_momentum_cooldown_tokens: int | None = None
    matrix_state_dtype: str | None = None
    vector_state_dtype: str | None = None


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
    eval_full_sweep: bool = False
    eval_stride: int | None = None
    eval_shuffle_documents: bool = False
    eval_limit_windows: int | None = None
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_gradient_checkpointing: bool = False
    logit_softcap: float | None = 15.0
    z_loss_weight: float = 1e-4
    softcap_warmup_steps: int = 800
    zloss_warmup_steps: int = 800
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    align_to_bos: bool = False
    compile_warmup_steps: int = 0
    ema_decay: float | None = None
    ema_update_interval: int = 1
    use_ema_for_eval: bool = False
    grad_accum_reference: int | None = None
    grad_accum_schedule: tuple[tuple[int, int], ...] | None = None
    grad_accum_minutes_schedule: tuple[tuple[float, int], ...] | None = None
    softcap_warmup_tokens: int | None = None
    zloss_warmup_tokens: int | None = None


@dataclass(frozen=True)
class DataConfig:
    train_path: Path | None = None
    val_path: Path | None = None
    dtype: str = "uint16"
    bos_token_id: int | None = None
    end_of_text_token_id: int = 31999


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
    cosine_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()
    lr_schedule: LearningRateScheduleConfig = LearningRateScheduleConfig()


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
    ckpt = cfg.checkpoint

    data_kwargs: dict[str, Any] = {}
    if data.train_path is not None and not isinstance(data.train_path, Path):
        data_kwargs["train_path"] = Path(data.train_path)
    if data.val_path is not None and not isinstance(data.val_path, Path):
        data_kwargs["val_path"] = Path(data.val_path)
    if data_kwargs:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **data_kwargs))

    ckpt_kwargs: dict[str, Any] = {}
    if ckpt.checkpoint_dir is not None and not isinstance(ckpt.checkpoint_dir, Path):
        ckpt_kwargs["checkpoint_dir"] = Path(ckpt.checkpoint_dir)
    if ckpt.resume_path is not None and not isinstance(ckpt.resume_path, Path):
        ckpt_kwargs["resume_path"] = Path(ckpt.resume_path)
    if ckpt_kwargs:
        cfg = dataclasses.replace(cfg, checkpoint=dataclasses.replace(cfg.checkpoint, **ckpt_kwargs))

    opt = cfg.optimizer
    if not isinstance(opt.betas, tuple):
        cfg = dataclasses.replace(cfg, optimizer=dataclasses.replace(opt, betas=tuple(opt.betas)))
    opt = cfg.optimizer
    opt_kwargs: dict[str, Any] = {}
    for field_name in (
        "muon_lr_warmup_tokens",
        "muon_momentum_warmup_tokens",
        "muon_momentum_cooldown_tokens",
    ):
        value = getattr(opt, field_name, None)
        if value is not None:
            opt_kwargs[field_name] = int(value)
    ratio = getattr(opt, "vector_lr_ratio", None)
    if ratio is not None:
        opt_kwargs["vector_lr_ratio"] = float(ratio)
    if opt_kwargs:
        cfg = dataclasses.replace(cfg, optimizer=dataclasses.replace(opt, **opt_kwargs))

    training = cfg.training
    training_kwargs: dict[str, Any] = {}
    if training.grad_accum_reference is not None:
        training_kwargs["grad_accum_reference"] = int(training.grad_accum_reference)
    if training.softcap_warmup_tokens is not None:
        training_kwargs["softcap_warmup_tokens"] = int(training.softcap_warmup_tokens)
    if training.zloss_warmup_tokens is not None:
        training_kwargs["zloss_warmup_tokens"] = int(training.zloss_warmup_tokens)
    if training.grad_accum_schedule is not None:
        normalized: list[tuple[int, int]] = []
        for item in training.grad_accum_schedule:
            if isinstance(item, dict):
                step_val = int(item.get("step") or item.get("iteration") or item.get("iter") or item.get("start_step") or 0)
                value_val = int(item.get("value") or item.get("accum") or item.get("grad_accum") or item.get("accumulation") or 0)
                normalized.append((step_val, max(1, value_val)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                step_val = int(item[0])
                value_val = int(item[1])
                normalized.append((step_val, max(1, value_val)))
        normalized.sort(key=lambda pair: pair[0])
        training_kwargs["grad_accum_schedule"] = tuple(normalized)
    if training.grad_accum_minutes_schedule is not None:
        normalized_minutes: list[tuple[float, int]] = []
        for item in training.grad_accum_minutes_schedule:
            if isinstance(item, dict):
                minute = float(item.get("minute") or item.get("minutes") or item.get("time") or item.get("start") or 0.0)
                value = int(item.get("value") or item.get("grad_accum") or item.get("accum") or item.get("accumulation") or 0)
                normalized_minutes.append((minute, max(1, value)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                minute = float(item[0])
                value = int(item[1])
                normalized_minutes.append((minute, max(1, value)))
        normalized_minutes.sort(key=lambda pair: pair[0])
        training_kwargs["grad_accum_minutes_schedule"] = tuple(normalized_minutes)
    if training_kwargs:
        cfg = dataclasses.replace(cfg, training=dataclasses.replace(training, **training_kwargs))
    return cfg


def load_config(config_path: Path | None) -> ExperimentConfig:
    base = ExperimentConfig()
    if config_path:
        overrides = _load_json(config_path)
        base = _replace_dataclass(base, overrides)
    return _coerce_config_types(base)


def apply_cli_overrides(cfg: ExperimentConfig, *, wandb_run_name: str | None = None) -> ExperimentConfig:
    if wandb_run_name is not None:
        logging = dataclasses.replace(cfg.logging, run_name=wandb_run_name)
        cfg = dataclasses.replace(cfg, logging=logging)
    return cfg


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
