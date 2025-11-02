from __future__ import annotations

import numpy as np
import torch

from cs336_basics.modules.manual_dataloader import (
    BoundaryAwareDataLoader,
    DEFAULT_EOT_TOKEN_ID,
    EvalIterationConfig,
)


class TrainBatchStream:
    """Endless stream of training batches sampled without replacement per epoch."""

    def __init__(
        self,
        loader: BoundaryAwareDataLoader,
        *,
        rng: np.random.Generator | None = None,
        drop_last: bool = True,
    ) -> None:
        self.loader = loader
        self.rng = rng or np.random.default_rng()
        self.drop_last = drop_last

    def __iter__(self):
        rng = self.rng
        loader = self.loader
        drop_last = self.drop_last

        def generator():
            while True:
                yield from loader.iter_random_without_replacement(rng=rng, drop_last=drop_last)

        return generator()

    @property
    def total_windows(self) -> int:
        return self.loader.total_windows

    @property
    def full_batches_per_epoch(self) -> int:
        return self.loader.full_batches_per_epoch

    @property
    def leftovers_per_epoch(self) -> int:
        return self.loader.leftovers_per_epoch


class EvalBatchStream:
    """Iterator over evaluation batches supporting either full sweeps or random draws."""

    def __init__(
        self,
        loader: BoundaryAwareDataLoader,
        *,
        mode: str,
        rng: np.random.Generator | None = None,
        num_batches: int | None = None,
        config: EvalIterationConfig | None = None,
    ) -> None:
        if mode not in {"full", "random"}:
            raise ValueError(f"Unsupported eval mode: {mode!r}")
        self.loader = loader
        self.mode = mode
        self.rng = rng or np.random.default_rng()
        self.num_batches = None if num_batches is None else int(max(1, num_batches))
        self.config = config

    def __iter__(self):
        loader = self.loader
        rng = self.rng
        mode = self.mode
        num_batches = self.num_batches
        config = self.config

        def generator():
            if mode == "full":
                yield from loader.iter_eval(config=config, rng=rng)
            else:
                assert num_batches is not None
                for _ in range(num_batches):
                    yield loader.random_batch(rng=rng)

        return generator()


def build_train_loader(
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator | None = None,
    end_of_text_token_id: int = DEFAULT_EOT_TOKEN_ID,
    drop_last: bool = True,
) -> TrainBatchStream:
    loader = BoundaryAwareDataLoader(
        tokens,
        context_length=context_length,
        batch_size=batch_size,
        device=device,
        end_of_text_token_id=end_of_text_token_id,
    )
    return TrainBatchStream(loader, rng=rng, drop_last=drop_last)


def build_eval_loader(
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator | None = None,
    end_of_text_token_id: int = DEFAULT_EOT_TOKEN_ID,
    full_sweep: bool = False,
    stride: int | None = None,
    shuffle_documents: bool = False,
    drop_last: bool = False,
    limit_windows: int | None = None,
    num_batches: int | None = None,
) -> EvalBatchStream:
    loader = BoundaryAwareDataLoader(
        tokens,
        context_length=context_length,
        batch_size=batch_size,
        device=device,
        end_of_text_token_id=end_of_text_token_id,
    )

    if full_sweep:
        config = EvalIterationConfig(
            stride=stride,
            shuffle_documents=shuffle_documents,
            drop_last=drop_last,
            limit_windows=limit_windows,
        )
        return EvalBatchStream(loader, mode="full", rng=rng, config=config)

    # Random evaluation batches
    effective_batches = num_batches
    if limit_windows is not None:
        effective_batches = max(1, int(limit_windows // max(1, batch_size)))
    elif num_batches is None:
        effective_batches = 1
    return EvalBatchStream(loader, mode="random", rng=rng, num_batches=effective_batches)
