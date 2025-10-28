from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
import math
from torch.utils.data import Sampler

__all__ = ["LMSequenceDataset", "dataloader", "StridedSampler"]


class LMSequenceDataset(Dataset):
    """Simple indexable dataset over token starts for language modeling.

    Given a 1D token array T of length L and context_length S, this dataset has
    length N = L - S. Item i returns (T[i:i+S], T[i+1:i+S+1]).
    """

    def __init__(self, tokens: npt.NDArray[np.integer], context_length: int) -> None:
        # Preserve memmap to avoid materializing into RAM and to let workers
        # reopen the file-backed array without duplicating memory.
        if isinstance(tokens, np.memmap) or isinstance(getattr(tokens, 'base', None), np.memmap):
            arr = tokens  # keep mapping/view
        else:
            arr = np.asarray(tokens)
        if arr.ndim != 1:
            raise ValueError("tokens must be a 1D array of token ids")
        if arr.size < context_length + 1:
            raise ValueError(
                f"tokens must have at least context_length+1 elements (got {arr.size}, need {context_length+1})"
            )
        self.tokens = arr
        self.context_length = int(context_length)
        self.N = int(arr.size - self.context_length)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        S = self.context_length
        x = self.tokens[i : i + S]
        y = self.tokens[i + 1 : i + S + 1]
        # Ensure contiguous and int64 for embedding lookups
        X = torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64))
        Y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        return X, Y


class RandomizedStridedIndexSampler(Sampler[int]):
    """Epoch-wise randomized strided traversal of start indices without replacement.

    Strongly reduces duplicate windows compared to plain shuffle=True.
    """

    def __init__(self, N: int, rng: np.random.Generator | None = None) -> None:
        self.N = int(max(1, N))
        self.rng = rng or np.random.default_rng()
        self._epoch_plan: np.ndarray | None = None
        self._cursor = 0

    def _plan_epoch(self) -> None:
        # choose a stride coprime with N
        stride = int(self.rng.integers(1, self.N))
        while math.gcd(stride, self.N) != 1:
            stride = int(self.rng.integers(1, self.N))
        offset = int(self.rng.integers(0, self.N))
        # generate 0..N-1 in a permuted strided order
        idx = (offset + stride * np.arange(self.N, dtype=np.int64)) % self.N
        self._epoch_plan = idx
        self._cursor = 0

    def __iter__(self):
        if self._epoch_plan is None or self._cursor >= self.N:
            self._plan_epoch()
        plan = self._epoch_plan
        self._plan_epoch()  # pre-plan next epoch for worker persistence
        return iter(plan.tolist())

    def __len__(self) -> int:
        return self.N


# Deprecated, kept for compatibility: one-off batch maker and strided sampler
from dataclasses import dataclass


@dataclass
class StridedSampler:
    N: int
    rng: np.random.Generator
    offset: int | None = None
    stride: int | None = None
    cursor: int = 0

    def _choose_stride(self) -> int:
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
            starts = rng.integers(0, N, size=batch_size, dtype=np.int64)

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
