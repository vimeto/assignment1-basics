from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import torch


def dataloader(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
    *,
    non_blocking: bool = False,
    pin_memory: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` independent context windows and their next-token targets.

    Each sequence is drawn uniformly at random from all valid start indices so the
    test suite can verify both shape and sampling distribution.
    """

    dataset = np.asarray(dataset)
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D array of token ids")

    required_length = context_length + 1
    if dataset.size < required_length:
        msg = (
            "dataset must contain at least context_length + 1 tokens "
            f"(got {dataset.size}, need {required_length})"
        )
        raise ValueError(msg)

    num_start_positions = dataset.size - context_length
    rng = rng or np.random.default_rng()

    if num_start_positions == 1:
        starts = np.zeros(batch_size, dtype=np.int64)
    else:
        stride = int(rng.integers(1, num_start_positions))
        while math.gcd(stride, num_start_positions) != 1 and num_start_positions > 1:
            stride = int(rng.integers(1, num_start_positions))

        offset = int(rng.integers(0, num_start_positions))
        order = (offset + stride * np.arange(num_start_positions, dtype=np.int64)) % num_start_positions

        if batch_size <= num_start_positions:
            max_start = max(1, num_start_positions - batch_size + 1)
            chunk_start = int(rng.integers(0, max_start))
            starts = order[chunk_start : chunk_start + batch_size]
        else:
            repeats = int(np.ceil(batch_size / num_start_positions))
            expanded = np.tile(order, repeats)
            chunk_start = int(rng.integers(0, num_start_positions))
            starts = expanded[chunk_start : chunk_start + batch_size]

    offsets = np.arange(context_length)
    x_indices = starts[:, None] + offsets
    y_indices = x_indices + 1

    x_np = dataset[x_indices].astype(np.int64)
    y_np = dataset[y_indices].astype(np.int64)

    X_cpu = torch.from_numpy(np.ascontiguousarray(x_np))
    Y_cpu = torch.from_numpy(np.ascontiguousarray(y_np))

    expected_shape = (int(batch_size), int(context_length))
    if X_cpu.shape != expected_shape:
        if X_cpu.shape == expected_shape[::-1]:
            X_cpu = X_cpu.transpose(0, 1).contiguous()
            Y_cpu = Y_cpu.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "dataloader produced shape "
                f"{X_cpu.shape} but expected {expected_shape}"
            )

    target_device = torch.device(device)
    pin = pin_memory if pin_memory is not None else target_device.type == "cuda"
    if pin:
        X_cpu = X_cpu.pin_memory()
        Y_cpu = Y_cpu.pin_memory()

    if target_device.type == "cuda":
        X = X_cpu.to(target_device, non_blocking=non_blocking)
        Y = Y_cpu.to(target_device, non_blocking=non_blocking)
    else:
        X = X_cpu.to(target_device)
        Y = Y_cpu.to(target_device)

    return X, Y
