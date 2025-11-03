from __future__ import annotations

import numpy as np
import torch

from cs336_basics.modules.manual_dataloader import dataloader, batch_from_start_indices


class RandomBatchStream:
    """Infinite stream of randomly sampled language-modeling batches."""

    def __init__(
        self,
        tokens: np.ndarray,
        *,
        batch_size: int,
        context_length: int,
        device: torch.device,
        rng: np.random.Generator | None = None,
        align_to_bos: bool = False,
        bos_token_id: int | None = None,
    ) -> None:
        self.tokens = np.asarray(tokens)
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.device = str(device)
        self.rng = rng or np.random.default_rng()
        self.align_to_bos = bool(align_to_bos)
        self._bos_starts: np.ndarray | None = None

        if self.align_to_bos and bos_token_id is not None and self.tokens.size > self.context_length:
            bos_id = np.array(bos_token_id, dtype=self.tokens.dtype).item()
            starts = np.flatnonzero(self.tokens == bos_id).astype(np.int64)
            if starts.size > 0:
                valid = starts[starts + self.context_length < self.tokens.size]
                if valid.size > 0:
                    self._bos_starts = np.ascontiguousarray(valid, dtype=np.int64)
        if self._bos_starts is None:
            self.align_to_bos = False

    def __iter__(self):
        rng = self.rng
        tokens = self.tokens
        batch_size = self.batch_size
        context_length = self.context_length
        device = self.device

        def generator():
            while True:
                if self.align_to_bos and self._bos_starts is not None and self._bos_starts.size > 0:
                    starts = rng.choice(self._bos_starts, size=batch_size, replace=True)
                    yield batch_from_start_indices(tokens, starts, context_length=context_length, device=device)
                else:
                    yield dataloader(tokens, batch_size, context_length, device, rng=rng)

        return generator()


class EvalBatchIterator:
    """Generate evaluation batches either randomly or via a sequential sweep."""

    def __init__(
        self,
        tokens: np.ndarray,
        *,
        batch_size: int,
        context_length: int,
        device: torch.device,
        rng: np.random.Generator | None = None,
        num_batches: int | None = None,
        full_sweep: bool = False,
        stride: int | None = None,
        drop_last: bool = False,
        limit_windows: int | None = None,
    ) -> None:
        self.tokens = np.asarray(tokens)
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.device = str(device)
        self.rng = rng or np.random.default_rng()
        self.num_batches = None if num_batches is None else int(max(1, num_batches))
        self.full_sweep = bool(full_sweep)
        self.stride = stride
        self.drop_last = bool(drop_last)
        self.limit_windows = None if limit_windows is None else int(max(1, limit_windows))

    def __iter__(self):
        if self.full_sweep:
            return self._iter_sweep()
        return self._iter_random()

    def _iter_random(self):
        tokens = self.tokens
        batch_size = self.batch_size
        context_length = self.context_length
        device = self.device
        rng = self.rng
        num_batches = self.num_batches or 1

        def generator():
            for _ in range(num_batches):
                yield dataloader(tokens, batch_size, context_length, device, rng=rng)

        return generator()

    def _iter_sweep(self):
        tokens = self.tokens
        batch_size = self.batch_size
        context_length = self.context_length
        device = self.device
        stride = self.stride if self.stride is not None else context_length
        total_windows = tokens.size - context_length
        if total_windows <= 0:
            raise ValueError("dataset too small for the requested context length")

        starts = np.arange(0, total_windows, stride, dtype=np.int64)
        if starts[-1] != total_windows - 1:
            starts = np.append(starts, total_windows - 1)
        if self.limit_windows is not None:
            starts = starts[: self.limit_windows]

        def generator():
            ptr = 0
            n = starts.size
            while ptr < n:
                partial = starts[ptr : ptr + batch_size]
                if partial.size < batch_size and self.drop_last:
                    break
                x, y = batch_from_start_indices(tokens, partial, context_length=context_length, device=device)
                yield x, y
                ptr += batch_size

        return generator()


def build_train_loader(
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator | None = None,
    align_to_bos: bool = False,
    bos_token_id: int | None = None,
) -> RandomBatchStream:
    return RandomBatchStream(
        tokens,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        rng=rng,
        align_to_bos=align_to_bos,
        bos_token_id=bos_token_id,
    )


def build_eval_loader(
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator | None = None,
    num_batches: int | None = None,
    full_sweep: bool = False,
    stride: int | None = None,
    drop_last: bool = False,
    limit_windows: int | None = None,
) -> EvalBatchIterator:
    return EvalBatchIterator(
        tokens,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        rng=rng,
        num_batches=num_batches,
        full_sweep=full_sweep,
        stride=stride,
        drop_last=drop_last,
        limit_windows=limit_windows,
    )
