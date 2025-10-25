from __future__ import annotations
import math
import numpy as np
import numpy.typing as npt
import torch
from dataclasses import dataclass

__all__ = ["dataloader", "StridedSampler"]

@dataclass
class StridedSampler:
    """Randomized strided sampler without replacement across steps.

    Walks all valid start positions exactly once (modulo wrap on the last partial
    batch), then reshuffles (new stride co-prime to N and new offset).

    Mirrors the idea used in top leaderboard PRs: fewer early repeats => faster
    val-loss drop.
    """
    N: int
    rng: np.random.Generator
    offset: int | None = None
    stride: int | None = None
    cursor: int = 0

    def _choose_stride(self) -> int:
        # choose stride in [1, N-1] that is coprime with N
        if self.N <= 1:
            return 1
        stride = int(self.rng.integers(1, self.N))
        while math.gcd(stride, self.N) != 1:
            stride = int(self.rng.integers(1, self.N))
        return stride

    def _new_epoch(self) -> None:
        self.offset = int(self.rng.integers(0, max(1, self.N)))
        self.stride = self._choose_stride()
        self.cursor = 0

    def next(self, count: int) -> np.ndarray:
        if self.offset is None or self.stride is None:
            self._new_epoch()
        start = self.cursor
        stop = self.cursor + count
        idx_range = np.arange(start, stop, dtype=np.int64)
        starts = (self.offset + self.stride * idx_range) % max(1, self.N)
        self.cursor = stop
        if self.cursor >= self.N:
            # start a new epoch on next call
            self._new_epoch()
        return starts


def dataloader(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
    *,
    non_blocking: bool = False,
    pin_memory: bool | None = None,
    sampler: StridedSampler | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X,Y) token windows of shape (B,S), (B,S) on target device.

    If `sampler` is provided, use randomized strided sampling WITHOUT replacement
    across steps (preferred for early-val improvements). Otherwise, fall back to
    independent random starts (original behavior).
    """

    dataset = np.asarray(dataset)
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D array of token ids")

    required_length = context_length + 1
    if dataset.size < required_length:
        raise ValueError(
            f"dataset must contain at least context_length + 1 tokens (got {dataset.size}, need {required_length})"
        )

    N = dataset.size - context_length
    rng = rng or np.random.default_rng()

    if N == 1:
        starts = np.zeros(batch_size, dtype=np.int64)
    else:
        if sampler is not None:
            starts = sampler.next(batch_size)
        else:
            stride = int(rng.integers(1, N))
            while math.gcd(stride, N) != 1 and N > 1:
                stride = int(rng.integers(1, N))
            offset = int(rng.integers(0, N))
            seq = np.arange(batch_size, dtype=np.int64)
            starts = (offset + stride * seq) % N

    offsets = np.arange(context_length, dtype=np.int64)
    x_idx = starts[:, None] + offsets
    y_idx = x_idx + 1

    x_np = dataset[x_idx].astype(np.int64, copy=False)
    y_np = dataset[y_idx].astype(np.int64, copy=False)

    X_cpu = torch.from_numpy(np.ascontiguousarray(x_np))
    Y_cpu = torch.from_numpy(np.ascontiguousarray(y_np))

    expected = (int(batch_size), int(context_length))
    if X_cpu.shape != expected:
        if X_cpu.shape == expected[::-1]:
            X_cpu = X_cpu.transpose(0, 1).contiguous()
            Y_cpu = Y_cpu.transpose(0, 1).contiguous()
        else:
            raise ValueError(f"dataloader produced shape {X_cpu.shape} but expected {expected}")

    target = torch.device(device)
    pin = pin_memory if pin_memory is not None else target.type == "cuda"
    if pin:
        X_cpu = X_cpu.pin_memory()
        Y_cpu = Y_cpu.pin_memory()

    if target.type == "cuda":
        X = X_cpu.to(target, non_blocking=non_blocking)
        Y = Y_cpu.to(target, non_blocking=non_blocking)
    else:
        X = X_cpu.to(target)
        Y = Y_cpu.to(target)

    return X, Y
