from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from cs336_basics.tokenizer import Tokenizer


@dataclass(frozen=True)
class CorpusPaths:
    name: str
    text_path: Path
    tokens_path: Path
    tokenizer_dir: Path


def _project_dirs() -> tuple[Path, Path]:
    """Return (repo_root, sibling_data_dir)."""

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root.parent / "data"
    tokenizer_dir = repo_root.parent / "tokenizers"
    return data_dir, tokenizer_dir


def _default_corpora() -> tuple[CorpusPaths, ...]:
    data_dir, tokenizer_dir = _project_dirs()
    tokenizer_root = tokenizer_dir / "owt_tokenizer_32k"
    return (
        CorpusPaths(
            name="tinystories-train",
            text_path=data_dir / "TinyStoriesV2-GPT4-train.txt",
            tokens_path=data_dir / "TinyStoriesV2-GPT4-train.npy",
            tokenizer_dir=tokenizer_root,
        ),
        CorpusPaths(
            name="tinystories-valid",
            text_path=data_dir / "TinyStoriesV2-GPT4-valid.txt",
            tokens_path=data_dir / "TinyStoriesV2-GPT4-valid.npy",
            tokenizer_dir=tokenizer_root,
        ),
    )


def _decoded_byte_chunks(
    tokenizer: Tokenizer,
    token_memmap: np.memmap,
    chunk_size: int = 100_000,
) -> Iterator[bytes]:
    """Yield decoded byte sequences for successive slices of ``token_memmap``."""

    id_to_token = tokenizer.id_to_token
    total_tokens = token_memmap.shape[0]
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        chunk = token_memmap[start:end]
        if chunk.size == 0:
            continue
        buffer = bytearray()
        extend = buffer.extend
        for token_id in chunk.tolist():
            extend(id_to_token[int(token_id)])
        yield bytes(buffer)


@pytest.mark.slow
@pytest.mark.parametrize("corpus", _default_corpora(), ids=lambda c: c.name)
def test_tokenized_corpora_roundtrip(corpus: CorpusPaths) -> None:
    """Ensure detokenising the stored ids reconstructs the original text exactly."""

    if not corpus.tokens_path.exists():
        pytest.skip(f"Token file missing: {corpus.tokens_path}")
    if not corpus.text_path.exists():
        pytest.skip(f"Source text missing: {corpus.text_path}")

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(corpus.tokenizer_dir / "vocab.json"),
        merges_filepath=str(corpus.tokenizer_dir / "merges.txt"),
    )

    token_memmap = np.load(corpus.tokens_path, mmap_mode="r")

    with corpus.text_path.open("rb") as source:
        for decoded_bytes in _decoded_byte_chunks(tokenizer, token_memmap):
            expected = source.read(len(decoded_bytes))
            assert expected == decoded_bytes

        leftover = source.read()
        assert leftover == b""
