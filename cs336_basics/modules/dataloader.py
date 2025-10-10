from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def dataloader(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
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
    # (batch_size,)
    starts = rng.integers(0, num_start_positions, size=batch_size)

    offsets = np.arange(context_length)
    x_indices = starts[:, None] + offsets
    y_indices = x_indices + 1

    x_np = dataset[x_indices]
    y_np = dataset[y_indices]

    X = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    Y = torch.from_numpy(np.ascontiguousarray(y_np)).to(device)
    return X, Y

