from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.backends.cuda

torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"

from cs336_basics.training.configs import load_config, apply_cli_overrides
from cs336_basics.training.engine import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--config", type=Path, help="Path to JSON config file", default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Override W&B run name only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, wandb_run_name=args.wandb_run_name)
    train(cfg)


if __name__ == "__main__":
    main()

