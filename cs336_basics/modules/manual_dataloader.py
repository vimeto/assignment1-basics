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
    """Configuration controlling evaluation sweeps.

    Attributes:
        stride: Distance (in tokens) between evaluation windows. When ``None``
            the stride defaults to ``context_length``.
        shuffle_documents: Whether to shuffle document order before iterating.
        drop_last: Drop the final partial batch (if any).
        limit_windows: Optional hard cap on the number of windows yielded.
    """

    stride: int | None = None
    shuffle_documents: bool = False
    drop_last: bool = False
    limit_windows: int | None = None


class BoundaryAwareDataLoader:
    """Efficient batch sampler that respects end-of-text token boundaries.

    The loader treats each contiguous span between ``end_of_text_token_id`` markers as
    an independent document and never surfaces batches whose (context, target) pairs
    cross between documents. The random sampler supports without-replacement traversal
    for training epochs, while the evaluation iterator can perform configurable full
    sweeps.
    """

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

        self.tokens = arr
        self.context_length = int(context_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self._target_device = self.device
        self.end_of_text_token_id = int(end_of_text_token_id)
        self.pin_memory = bool(pin_memory) if pin_memory is not None else self._target_device.type == "cuda"

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._context_offsets = np.arange(self.context_length, dtype=np.int64)

        (
            self._doc_starts,
            self._doc_ends,
        ) = self._compute_document_boundaries(self.tokens, self.end_of_text_token_id)
        self._doc_lengths = self._doc_ends - self._doc_starts
        self._doc_windows = np.maximum(0, self._doc_lengths - self.context_length)
        self._valid_docs = np.flatnonzero(self._doc_windows > 0)

        if self._valid_docs.size == 0:
            raise ValueError(
                "dataset must contain at least one document with >= context_length + 1 tokens"
            )

        self._total_windows = int(self._doc_windows[self._valid_docs].sum())
        self._cumulative_windows = np.cumsum(self._doc_windows[self._valid_docs], dtype=np.int64)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @property
    def total_windows(self) -> int:
        """Total number of valid context windows across all documents."""

        return self._total_windows

    @property
    def full_batches_per_epoch(self) -> int:
        """Number of full batches emitted during a without-replacement epoch."""

        return self._total_windows // self.batch_size

    @property
    def leftovers_per_epoch(self) -> int:
        """Number of leftover windows that would be dropped with ``drop_last=True``."""

        return self._total_windows % self.batch_size

    # ------------------------------------------------------------------
    # Sampling entry points
    # ------------------------------------------------------------------
    def random_batch(
        self,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a single batch with replacement across all valid start positions."""

        rng = rng or np.random.default_rng()
        starts = self._sample_random_starts_with_replacement(rng, self.batch_size)
        return self._materialize_batch(starts)

    def iter_random_without_replacement(
        self,
        *,
        rng: np.random.Generator | None = None,
        drop_last: bool = True,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield batches that traverse every window exactly once in random order."""

        rng = rng or np.random.default_rng()
        starts_epoch = self._sample_without_replacement(rng)
        if starts_epoch.size == 0:
            return

        buffer: list[int] = []
        for start in starts_epoch:
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
        """Iterate over evaluation batches following a configurable sweep."""

        cfg = config or EvalIterationConfig()
        stride = int(cfg.stride) if cfg.stride is not None else self.context_length
        if stride <= 0:
            raise ValueError("stride must be positive")

        doc_indices = self._valid_docs.copy()
        if cfg.shuffle_documents:
            rng = rng or np.random.default_rng()
            rng.shuffle(doc_indices)

        limit = cfg.limit_windows
        windows_emitted = 0
        buffer: list[int] = []

        for doc_id in doc_indices:
            count = int(self._doc_windows[doc_id])
            if count <= 0:
                continue

            doc_start = int(self._doc_starts[doc_id])
            base = doc_start + np.arange(count, dtype=np.int64)
            if stride <= 1:
                starts = base
            else:
                subsampled = base[::stride]
                if subsampled.size == 0 or subsampled[-1] != base[-1]:
                    starts = np.concatenate((subsampled, base[-1:]))
                else:
                    starts = subsampled

            for start in starts:
                buffer.append(int(start))
                windows_emitted += 1
                if len(buffer) == self.batch_size:
                    start_array = np.fromiter(buffer, dtype=np.int64, count=self.batch_size)
                    yield self._materialize_batch(start_array)
                    buffer.clear()
                if limit is not None and windows_emitted >= limit:
                    break

            if limit is not None and windows_emitted >= limit:
                break

        if buffer and not cfg.drop_last:
            start_array = np.fromiter(buffer, dtype=np.int64, count=len(buffer))
            yield self._materialize_batch(start_array)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
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
            end = int(pos) + 1
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

        return (
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
        )

    def _sample_random_starts_with_replacement(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> np.ndarray:
        if self._total_windows <= 0:
            raise ValueError("no valid windows to sample from")

        choices = rng.integers(0, self._total_windows, size=size, dtype=np.int64)
        doc_offsets = np.searchsorted(self._cumulative_windows, choices, side="right")
        doc_ids = self._valid_docs[doc_offsets]
        cumulative_previous = self._cumulative_windows[doc_offsets] - self._doc_windows[doc_ids]
        within_doc_offsets = choices - cumulative_previous
        starts = self._doc_starts[doc_ids] + within_doc_offsets
        return starts.astype(np.int64, copy=False)

    def _sample_without_replacement(self, rng: np.random.Generator) -> np.ndarray:
        total = self._total_windows
        if total <= 0:
            return np.empty((0,), dtype=np.int64)
        choices = rng.permutation(total).astype(np.int64, copy=False)
        doc_offsets = np.searchsorted(self._cumulative_windows, choices, side="right")
        doc_ids = self._valid_docs[doc_offsets]
        cumulative_previous = self._cumulative_windows[doc_offsets] - self._doc_windows[doc_ids]
        within_doc_offsets = choices - cumulative_previous
        starts = self._doc_starts[doc_ids] + within_doc_offsets
        return starts.astype(np.int64, copy=False)

    def _materialize_batch(self, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.asarray(starts, dtype=np.int64)
        contexts_idx = starts[:, None] + self._context_offsets
        targets_idx = contexts_idx + 1

        x_np = np.ascontiguousarray(self.tokens[contexts_idx], dtype=np.int64)
        y_np = np.ascontiguousarray(self.tokens[targets_idx], dtype=np.int64)

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        if self.pin_memory and self._target_device.type == "cuda":
            x_tensor = x_tensor.pin_memory()
            y_tensor = y_tensor.pin_memory()

        x_device = x_tensor.to(self._target_device, non_blocking=True)
        y_device = y_tensor.to(self._target_device, non_blocking=True)
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
    """Backward-compatible helper that returns a single random batch."""

    loader = BoundaryAwareDataLoader(
        dataset,
        context_length=context_length,
        batch_size=batch_size,
        device=device,
        end_of_text_token_id=end_of_text_token_id,
    )
    return loader.random_batch(rng=rng)
