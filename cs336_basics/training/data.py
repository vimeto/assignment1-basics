from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from cs336_basics.modules.dataloader import (
    LMSequenceDataset,
    DevicePrefetcher,
)

def build_train_loader(tokens: np.ndarray, *, context_length: int, batch_size: int, device: torch.device, num_workers: int, pin_memory: bool, persistent_workers: bool) -> tuple[DataLoader, DevicePrefetcher | None]:
    ds = LMSequenceDataset(tokens, context_length)
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


def build_eval_loader(tokens: np.ndarray, *, context_length: int, batch_size: int, device: torch.device, num_workers: int, pin_memory: bool, persistent_workers: bool) -> DataLoader:
    ds = LMSequenceDataset(tokens, context_length)
    num_workers = max(0, int(min(num_workers, (os.cpu_count() or num_workers))))
    pin = device.type == "cuda" and bool(pin_memory)
    # Use RandomSampler with replacement to decouple eval length from dataset size
    num_samples = batch_size  # the caller will iterate for eval_batches
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

