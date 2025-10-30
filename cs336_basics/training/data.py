from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from cs336_basics.modules.dataloader import (
    LMSequenceDataset,
    DevicePrefetcher,
    BoundaryAlignedSampler,
    RandomizedStridedIndexSampler,
)


def build_train_loader(
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    boundary_token_id: int | None = None,
    start_after_token: bool = True,
    use_strided_sampler: bool = False,
) -> tuple[DataLoader, DevicePrefetcher | None]:
    ds = LMSequenceDataset(tokens, context_length)
    if boundary_token_id is not None:
        sampler = BoundaryAlignedSampler(tokens, context_length, boundary_token_id, start_after=start_after_token)
    elif use_strided_sampler:
        sampler = RandomizedStridedIndexSampler(ds.N)
    else:
        sampler = RandomSampler(ds, replacement=False)
    num_workers = max(0, int(min(num_workers, (os.cpu_count() or num_workers))))
    pin = device.type == "cuda" and bool(pin_memory)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    prefetcher = DevicePrefetcher(loader, device) if device.type == "cuda" else None
    return loader, prefetcher


def build_eval_loader(tokens: np.ndarray, *, context_length: int, batch_size: int, device: torch.device, num_workers: int, pin_memory: bool, persistent_workers: bool, num_batches: int = 16) -> DataLoader:
    ds = LMSequenceDataset(tokens, context_length)
    num_workers = max(0, int(min(num_workers, (os.cpu_count() or num_workers))))
    pin = device.type == "cuda" and bool(pin_memory)
    # Use RandomSampler with replacement to decouple eval length from dataset size
    num_samples = int(batch_size) * max(1, int(num_batches))
    sampler = RandomSampler(ds, replacement=True, num_samples=num_samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
