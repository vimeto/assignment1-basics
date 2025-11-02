from __future__ import annotations

import numpy as np
import torch
import pytest

from cs336_basics.modules.manual_dataloader import (
    BoundaryAwareDataLoader,
    EvalIterationConfig,
    dataloader,
    DEFAULT_EOT_TOKEN_ID,
)


def build_fixture_tokens() -> np.ndarray:
    """Construct a tiny corpus with clearly separated documents."""

    eot = DEFAULT_EOT_TOKEN_ID
    doc1 = np.array([0, 1, 2, 3, 4, 5, eot], dtype=np.uint16)
    doc2 = np.array([100, 101, 102, 103, 104, 105, 106, eot], dtype=np.uint16)
    doc3 = np.array([200, 201, 202, 203, 204, 205, 206, 207, eot], dtype=np.uint16)
    doc4 = np.array([300, 301, 302, 303, 304], dtype=np.uint16)  # No trailing EOT.
    return np.concatenate([doc1, doc2, doc3, doc4])


def expected_start_positions(tokens: np.ndarray, context_length: int) -> list[int]:
    """Compute all valid start indices that respect end-of-text boundaries."""

    eot = DEFAULT_EOT_TOKEN_ID
    positions = np.flatnonzero(tokens == eot)

    starts: list[int] = []
    prev = 0
    for pos in positions:
        end = pos + 1
        if end > prev:
            starts.extend(range(prev, end - context_length))
        prev = end

    if prev < tokens.size:
        starts.extend(range(prev, tokens.size - context_length))

    return starts


def contexts_to_start_map(tokens: np.ndarray, context_length: int) -> dict[tuple[int, ...], int]:
    """Build a lookup so that each context uniquely identifies its start index."""

    mapping: dict[tuple[int, ...], int] = {}
    for start in expected_start_positions(tokens, context_length):
        key = tuple(tokens[start : start + context_length].tolist())
        mapping[key] = start
    return mapping


def test_random_batch_respects_boundaries():
    tokens = build_fixture_tokens()
    loader = BoundaryAwareDataLoader(tokens, context_length=3, batch_size=4, device="cpu")
    rng = np.random.default_rng(0)

    for _ in range(128):
        x, y = loader.random_batch(rng=rng)
        assert x.shape == (4, 3)
        assert y.shape == (4, 3)

        x_np = x.numpy()
        y_np = y.numpy()

        for row_x, row_y in zip(x_np, y_np, strict=True):
            # The EOT token can only appear as the final context token.
            eot_hits = np.flatnonzero(row_x == DEFAULT_EOT_TOKEN_ID)
            if eot_hits.size:
                assert int(eot_hits[-1]) == row_x.size - 1
            # Targets should never draw tokens from the next document.
            eot_in_targets = np.flatnonzero(row_y[:-1] == DEFAULT_EOT_TOKEN_ID)
            assert eot_in_targets.size == 0


def test_random_without_replacement_covers_every_window():
    tokens = build_fixture_tokens()
    context_length = 3
    batch_size = 4
    loader = BoundaryAwareDataLoader(tokens, context_length=context_length, batch_size=batch_size, device="cpu")

    mapping = contexts_to_start_map(tokens, context_length)
    expected = sorted(mapping.values())

    rng = np.random.default_rng(42)

    observed: list[int] = []
    for x, y in loader.iter_random_without_replacement(rng=rng, drop_last=False):
        x_np = x.numpy()
        y_np = y.numpy()
        assert x_np.shape[0] == y_np.shape[0]
        for row_x, row_y in zip(x_np, y_np, strict=True):
            key = tuple(row_x.tolist())
            start = mapping.get(key)
            assert start is not None, f"Unexpected context {row_x}"
            # Ensure targets align with the dataset.
            np.testing.assert_array_equal(tokens[start + 1 : start + 1 + context_length], row_y)
            observed.append(start)

    assert len(observed) == len(expected)
    assert len(set(observed)) == len(expected)
    assert sorted(observed) == expected

    # Drop-last behaviour should emit consistent batch counts.
    total_windows = len(expected)
    full_batches = total_windows // batch_size
    leftover = total_windows % batch_size
    rng = np.random.default_rng(123)
    batches = list(loader.iter_random_without_replacement(rng=rng, drop_last=True))
    assert len(batches) == full_batches
    assert loader.full_batches_per_epoch == full_batches
    assert loader.leftovers_per_epoch == leftover


def expected_eval_starts(tokens: np.ndarray, context_length: int, stride: int) -> list[int]:
    """Mirror the loader's stride logic for validation."""

    eot = DEFAULT_EOT_TOKEN_ID
    positions = np.flatnonzero(tokens == eot)

    starts: list[int] = []
    prev = 0
    for pos in positions:
        end = pos + 1
        doc_length = end - prev
        count = doc_length - context_length
        if count > 0:
            base = prev + np.arange(count, dtype=np.int64)
            if stride <= 1:
                starts.extend(base.tolist())
            else:
                subsampled = base[::stride]
                if subsampled.size == 0:
                    subsampled = base[-1:]
                elif subsampled[-1] != base[-1]:
                    subsampled = np.concatenate((subsampled, base[-1:]))
                starts.extend(subsampled.tolist())
        prev = end

    if prev < tokens.size:
        doc_length = tokens.size - prev
        count = doc_length - context_length
        if count > 0:
            base = prev + np.arange(count, dtype=np.int64)
            if stride <= 1:
                starts.extend(base.tolist())
            else:
                subsampled = base[::stride]
                if subsampled.size == 0:
                    subsampled = base[-1:]
                elif subsampled[-1] != base[-1]:
                    subsampled = np.concatenate((subsampled, base[-1:]))
                starts.extend(subsampled.tolist())

    return starts


def flatten_batch_starts(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    mapping: dict[tuple[int, ...], int],
) -> list[int]:
    """Convert batches into the corresponding start indices using ``mapping``."""

    starts: list[int] = []
    for x, _ in batches:
        for row in x.numpy():
            starts.append(mapping[tuple(row.tolist())])
    return starts


def test_eval_full_sweep_stride_and_limit():
    tokens = build_fixture_tokens()
    context_length = 3
    batch_size = 3
    loader = BoundaryAwareDataLoader(tokens, context_length=context_length, batch_size=batch_size, device="cpu")

    expected = contexts_to_start_map(tokens, context_length)
    expected_all = sorted(expected.values())
    expected_stride = expected_eval_starts(tokens, context_length, loader.context_length)

    # Default configuration uses stride=context_length.
    batches = list(loader.iter_eval(config=EvalIterationConfig(), rng=None))
    observed_default = flatten_batch_starts(batches, expected)
    assert sorted(observed_default) == sorted(expected_stride)

    # Explicit stride=context_length matches the default behaviour.
    batches_stride = list(loader.iter_eval(config=EvalIterationConfig(stride=context_length)))
    observed_stride = flatten_batch_starts(batches_stride, expected)
    assert sorted(observed_stride) == sorted(expected_stride)

    # Stride=1 performs a true full sweep across every window.
    batches_stride_one = list(loader.iter_eval(config=EvalIterationConfig(stride=1)))
    observed_stride_one = flatten_batch_starts(batches_stride_one, expected)
    assert sorted(observed_stride_one) == expected_all

    # Limiting the number of windows and shuffling documents should be reproducible.
    seed = 7
    rng = np.random.default_rng(seed)
    config = EvalIterationConfig(limit_windows=5, shuffle_documents=True, drop_last=False)
    limited_batches_first = list(loader.iter_eval(config=config, rng=rng))
    rng = np.random.default_rng(seed)
    limited_batches_second = list(loader.iter_eval(config=config, rng=rng))

    def flatten_batches(batches: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []
        for x, _ in batches:
            for row in x.numpy():
                out.append(tuple(row.tolist()))
        return out

    flat_first = flatten_batches(limited_batches_first)
    flat_second = flatten_batches(limited_batches_second)
    assert flat_first == flat_second

    if flat_first:
        # Ensure the limit applied and the final batch may be partial.
        assert len(flat_first) == 5

    # Drop-last should remove incomplete batches under the limit.
    config_drop = EvalIterationConfig(limit_windows=5, shuffle_documents=False, drop_last=True)
    rng = np.random.default_rng(11)
    batches_drop = list(loader.iter_eval(config=config_drop, rng=rng))
    total_windows_drop = sum(batch[0].shape[0] for batch in batches_drop)
    assert total_windows_drop % batch_size == 0


def test_legacy_dataloader_wrapper():
    tokens = np.arange(32, dtype=np.uint16)
    x, y = dataloader(tokens, batch_size=8, context_length=4, device="cpu")
    assert x.shape == (8, 4)
    np.testing.assert_array_equal(x.numpy() + 1, y.numpy())

    with pytest.raises((RuntimeError, AssertionError)):
        dataloader(tokens, batch_size=4, context_length=4, device="cuda:99")
