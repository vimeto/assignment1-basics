from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    ) -> None:
    """Sample ``batch_size`` independent context windows and their next-token targets.

    Each sequence is drawn uniformly at random from all valid start indices so the
    test suite can verify both shape and sampling distribution.
    """

    model = model.state_dict()
    optimizer = optimizer.state_dict()
    state = { "it": iteration, "model": model, "opt": optimizer }
    torch.save(state, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ) -> int:
    state = torch.load(src, weights_only=True)

    model.load_state_dict(state.get("model"))
    optimizer.load_state_dict(state.get("opt"))
    return state.get("it", 0)
