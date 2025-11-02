from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import numpy.typing as npt
import torch

__all__ = [
    "BoundaryAwareDataLoader",
    "EvalIterationConfig",
    "dataloader",
]

DEFAULT_EOT_TOKEN_ID = 31999


@dataclass(frozen=True)
class EvalIterationConfig:
    stride: int | None = None
    shuffle_documents: bool = False
    drop_last: bool = False
    limit_windows: int | None = None


class BoundaryAwareDataLoader:
    """Sample fixed-length windows without crossing end-of-text boundaries."""

    def __init__(
        self,
        tokens: npt.NDArray[np.integer],
        *,
        context_length: int,
        batch_size: int,
        device: str,
        end_of_text_token_id: int = DEFAULT_EOT_TOKEN_ID,
        pin_memory: bool | None = None,
    ) -> None:
        arr = tokens if isinstance(tokens, np.ndarray) else np.asarray(tokens)
        if arr.ndim != 1:
            raise ValueError("tokens must be a 1D array of token ids")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.tokens = arr
        self.context_length = int(context_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory) if pin_memory is not None else self.device.type == "cuda"

        self.doc_starts, self.doc_ends = self._compute_document_boundaries(arr, end_of_text_token_id)
        doc_lengths = self.doc_ends - self.doc_starts
        doc_windows = np.maximum(0, doc_lengths - self.context_length)
        valid = doc_windows > 0
        if not np.any(valid):
            raise ValueError("dataset must contain at least one document with >= context_length + 1 tokens")

        self.doc_starts = self.doc_starts[valid]
        self.doc_ends = self.doc_ends[valid]
        self.doc_windows = doc_windows[valid]
        self.doc_cdf = np.cumsum(self.doc_windows)
        self.total_windows = int(self.doc_windows.sum())

        self._context_offsets = np.arange(self.context_length, dtype=np.int64)

    @property
    def full_batches_per_epoch(self) -> int:
        return self.total_windows // self.batch_size

    @property
    def leftovers_per_epoch(self) -> int:
        return self.total_windows % self.batch_size

    @staticmethod
    def _compute_document_boundaries(
        tokens: npt.NDArray[np.integer],
        eot_token_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        eot_positions = np.flatnonzero(tokens == eot_token_id)
        starts: list[int] = []
        ends: list[int] = []
        prev = 0
        for pos in eot_positions:
            end = pos + 1
            if end > prev:
                starts.append(prev)
                ends.append(end)
            prev = end
        if prev < tokens.size:
            starts.append(prev)
            ends.append(tokens.size)
        if not starts:
            starts.append(0)
            ends.append(tokens.size)
        return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)

    # ------------------------------------------------------------------
    # Sampling entry points
    # ------------------------------------------------------------------
    def random_batch(
        self,
        rng: np.random.Generator | None = None,
        *,
        with_replacement: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = rng or np.random.default_rng()
        starts = self._sample_random_starts(rng, self.batch_size, with_replacement=with_replacement)
        return self._materialize_batch(starts)

    def iter_random_without_replacement(
        self,
        *,
        rng: np.random.Generator | None = None,
        drop_last: bool = True,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = rng or np.random.default_rng()
        starts = self._sample_without_replacement(rng)
        if starts.size == 0:
            return

        buffer: list[int] = []
        for start in starts:
            buffer.append(int(start))
            if len(buffer) == self.batch_size:
                start_array = np.fromiter(buffer, dtype=np.int64, count=self.batch_size)
                yield self._materialize_batch(start_array)
                buffer.clear()

        if buffer and not drop_last:
            start_array = np.fromiter(buffer, dtype=np.int64, count=len(buffer))
            yield self._materialize_batch(start_array)

    def iter_eval(
        self,
        *,
        config: EvalIterationConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        cfg = config or EvalIterationConfig()
        stride = int(cfg.stride) if cfg.stride is not None else self.context_length
        if stride <= 0:
            raise ValueError("stride must be positive")

        doc_indices = np.arange(len(self.doc_starts), dtype=np.int64)
        if cfg.shuffle_documents:
            rng = rng or np.random.default_rng()
            rng.shuffle(doc_indices)

        limit = cfg.limit_windows
        emitted = 0
        buffer: list[int] = []

        for doc_id in doc_indices:
            count = int(self.doc_windows[doc_id])
            if count <= 0:
                continue

            doc_start = int(self.doc_starts[doc_id])
            last_start = int(self.doc_starts[doc_id] + self.doc_windows[doc_id] - 1)
            starts = np.arange(doc_start, last_start + 1, stride, dtype=np.int64)
            if starts.size == 0 or starts[-1] != last_start:
                starts = np.append(starts, last_start)

            for start in starts:
                buffer.append(int(start))
                emitted += 1
                if len(buffer) == self.batch_size:
                    start_array = np.fromiter(buffer, dtype=np.int64, count=self.batch_size)
                    yield self._materialize_batch(start_array)
                    buffer.clear()
                if limit is not None and emitted >= limit:
                    break

            if limit is not None and emitted >= limit:
                break

        if buffer and not cfg.drop_last:
            start_array = np.fromiter(buffer, dtype=np.int64, count=len(buffer))
            yield self._materialize_batch(start_array)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_random_starts(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        with_replacement: bool,
    ) -> np.ndarray:
        doc_probs = self.doc_windows / self.total_windows
        doc_indices = rng.choice(len(self.doc_starts), size=size, replace=with_replacement, p=doc_probs)
        starts = np.empty(size, dtype=np.int64)
        for i, doc_id in enumerate(doc_indices):
            doc_start = int(self.doc_starts[doc_id])
            span = int(self.doc_windows[doc_id])
            offset = int(rng.integers(0, span))
            starts[i] = doc_start + offset
        return starts

    def _sample_without_replacement(self, rng: np.random.Generator) -> np.ndarray:
        total = self.total_windows
        if total <= 0:
            return np.empty((0,), dtype=np.int64)
        order = rng.permutation(total)
        doc_indices = np.searchsorted(self.doc_cdf, order, side="right")
        prev_cumulative = np.zeros_like(order)
        prev_cumulative[:] = self.doc_cdf[doc_indices] - self.doc_windows[doc_indices]
        offsets = order - prev_cumulative
        starts = self.doc_starts[doc_indices] + offsets
        return starts.astype(np.int64, copy=False)

    def _materialize_batch(self, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.asarray(starts, dtype=np.int64)
        contexts_idx = starts[:, None] + self._context_offsets
        targets_idx = contexts_idx + 1

        x_np = np.ascontiguousarray(self.tokens[contexts_idx], dtype=np.int64)
        y_np = np.ascontiguousarray(self.tokens[targets_idx], dtype=np.int64)
        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        if self.pin_memory and self.device.type == "cuda":
            x_tensor = x_tensor.pin_memory()
            y_tensor = y_tensor.pin_memory()

        x_device = x_tensor.to(self.device, non_blocking=True)
        y_device = y_tensor.to(self.device, non_blocking=True)
        return x_device, y_device


def dataloader(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
    *,
    end_of_text_token_id: int = DEFAULT_EOT_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = BoundaryAwareDataLoader(
        dataset,
        context_length=context_length,
        batch_size=batch_size,
        device=device,
        end_of_text_token_id=end_of_text_token_id,
    )
    return loader.random_batch(rng=rng, with_replacement=True)
