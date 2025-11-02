from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

__all__ = ["dataloader", "batch_from_start_indices"]


def _validate_inputs(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
) -> np.ndarray:
    tokens = np.asarray(dataset)
    if tokens.ndim != 1:
        raise ValueError("dataset must be a 1D array of token ids")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    required = context_length + 1
    if tokens.size < required:
        msg = (
            "dataset must contain at least context_length + 1 tokens "
            f"(got {tokens.size}, need {required})"
        )
        raise ValueError(msg)
    return tokens


def batch_from_start_indices(
    dataset: npt.NDArray[np.integer],
    starts: npt.NDArray[np.integer],
    *,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = np.asarray(dataset)
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1:
        raise ValueError("starts must be a 1D array of indices")
    if starts.size == 0:
        raise ValueError("starts must contain at least one index")

    offsets = np.arange(context_length, dtype=np.int64)
    x_idx = starts[:, None] + offsets
    y_idx = x_idx + 1

    x_np = np.ascontiguousarray(tokens[x_idx], dtype=np.int64)
    y_np = np.ascontiguousarray(tokens[y_idx], dtype=np.int64)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    target_device = torch.device(device)
    if x.device != target_device:
        x = x.to(target_device)
        y = y.to(target_device)
    return x, y


def dataloader(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = _validate_inputs(dataset, batch_size, context_length)

    num_start_positions = tokens.size - context_length
    rng = rng or np.random.default_rng()
    starts = rng.integers(0, num_start_positions, size=batch_size, dtype=np.int64)

    return batch_from_start_indices(tokens, starts, context_length=context_length, device=device)
