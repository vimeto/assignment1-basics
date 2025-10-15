from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np

from cs336_basics.tokenizer import Tokenizer


@dataclass(frozen=True)
class CorpusSpec:
    """Configuration describing how to tokenize a single text corpus."""

    name: str
    text_path: Path
    tokenizer_dir: Path
    output_path: Path


class TokenizerCache:
    """Load tokenizer vocab/merge files once and reuse them."""

    def __init__(self) -> None:
        self._cache: Dict[Path, Tokenizer] = {}

    def get(self, tokenizer_dir: Path) -> Tokenizer:
        key = tokenizer_dir.resolve()
        tokenizer = self._cache.get(key)
        if tokenizer is None:
            vocab_path = key / "vocab.json"
            merges_path = key / "merges.txt"
            tokenizer = Tokenizer.from_files(vocab_path, merges_path)
            self._cache[key] = tokenizer
        return tokenizer


def _project_paths() -> tuple[Path, Path]:
    """Return the sibling directories that hold corpora and tokenizers."""

    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root.parent
    data_dir = base_dir / "data"
    tokenizer_dir = base_dir / "tokenizers"
    return data_dir, tokenizer_dir


def default_corpora() -> tuple[CorpusSpec, ...]:
    """Map the provided datasets to their corresponding tokenizers."""

    data_dir, tokenizer_dir = _project_paths()

    tokenizer_name = "owt_tokenizer_32k"
    return (
        CorpusSpec(
            name="tinystories-train",
            text_path=data_dir / "TinyStoriesV2-GPT4-train.txt",
            tokenizer_dir=tokenizer_dir / tokenizer_name,
            output_path=data_dir / "TinyStoriesV2-GPT4-train.npy",
        ),
        CorpusSpec(
            name="tinystories-valid",
            text_path=data_dir / "TinyStoriesV2-GPT4-valid.txt",
            tokenizer_dir=tokenizer_dir / tokenizer_name,
            output_path=data_dir / "TinyStoriesV2-GPT4-valid.npy",
        ),
        CorpusSpec(
            name="owt-train",
            text_path=data_dir / "owt_train.txt",
            tokenizer_dir=tokenizer_dir / tokenizer_name,
            output_path=data_dir / "owt_train.npy",
        ),
        CorpusSpec(
            name="owt-valid",
            text_path=data_dir / "owt_valid.txt",
            tokenizer_dir=tokenizer_dir / tokenizer_name,
            output_path=data_dir / "owt_valid.npy",
        ),
    )


def tokenize_corpus(spec: CorpusSpec, cache: TokenizerCache, dtype: np.dtype) -> None:
    tokenizer = cache.get(spec.tokenizer_dir)

    if spec.text_path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open

    with opener(spec.text_path, "rt", encoding="utf-8") as source:
        token_iterator: Iterator[int] = tokenizer.encode_iterable(source)
        token_array = np.fromiter(token_iterator, dtype=dtype)

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(spec.output_path, token_array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize corpora with project tokenizers")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of dataset names to process (default: all registered datasets)",
    )
    parser.add_argument(
        "--dtype",
        default="uint16",
        help="NumPy dtype for the output arrays (default: uint16)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = TokenizerCache()
    corpora = {spec.name: spec for spec in default_corpora()}

    if args.datasets:
        missing = [name for name in args.datasets if name not in corpora]
        if missing:
            msg = f"Unknown dataset name(s): {', '.join(missing)}"
            raise ValueError(msg)
        selected: Iterable[CorpusSpec] = (corpora[name] for name in args.datasets)
    else:
        selected = corpora.values()

    dtype = np.dtype(args.dtype)

    for spec in selected:
        tokenize_corpus(spec, cache, dtype)
        print(f"Wrote {spec.output_path} using tokenizer at {spec.tokenizer_dir}")


if __name__ == "__main__":
    main()
