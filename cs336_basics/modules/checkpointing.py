from __future__ import annotations

import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO | IO[bytes],
) -> None:
    """Persist a minimal training checkpoint with model and optimizer state."""
    state = {
        "it": int(iteration),
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model/optimizer state and return the iteration counter."""
    state = torch.load(src, weights_only=True, map_location="cpu")
    model_state = state.get("model")
    opt_state = state.get("opt")
    if model_state is None or opt_state is None:
        raise ValueError("checkpoint missing required keys: 'model' and 'opt'")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    return int(state.get("it", 0))
