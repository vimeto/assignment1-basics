from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import regex


@dataclass(frozen=True)
class MergeRule:
    """Simple helper so type checkers know that merges are always byte pairs."""

    left: bytes
    right: bytes


class Tokenizer:
    """Byte Pair Encoding (BPE) tokenizer that mirrors GPT-2 style tokenization.

    We work entirely in byte space: every token ID maps to a byte sequence, and
    merges are defined over byte-valued symbols. This matches the format produced
    by the provided training code and lets us share vocabularies with tiktoken.
    """

    # GPT-2's pre-tokenizer pattern. The `regex` module understands `\p{L}` etc.
    _PRETOKEN_PATTERN = regex.compile(
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ) -> None:
        # Store a copy so accidental external mutation will not affect us.
        self.id_to_token: Dict[int, bytes] = dict(vocab)
        self.token_to_id: Dict[bytes, int] = {tok: idx for idx, tok in self.id_to_token.items()}

        # Track the next free ID for any new special tokens.
        self._next_id = max(self.id_to_token, default=-1) + 1

        # Normalise merge definitions and build a fast lookup of their priority.
        self.merges: List[MergeRule] = [MergeRule(bytes(left), bytes(right)) for left, right in merges]
        self.bpe_ranks: Dict[Tuple[bytes, bytes], int] = {
            (rule.left, rule.right): rank for rank, rule in enumerate(self.merges)
        }

        # BPE is recursive, so caching individual byte sequences speeds things up.
        self._bpe_cache: Dict[bytes, Tuple[int, ...]] = {}

        # Optional special tokens stay intact during encoding.
        self.special_token_to_id: Dict[str, int] = {}
        if special_tokens:
            for tok in special_tokens:
                self._register_special_token(tok)
        self._sorted_special_tokens: List[str] = sorted(
            self.special_token_to_id, key=len, reverse=True
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _register_special_token(self, token: str) -> None:
        """Ensure `token` exists in the vocab and remember its ID for fast lookup."""

        token_str = str(token)
        token_bytes = token_str.encode("utf-8")

        token_id = self.token_to_id.get(token_bytes)
        if token_id is None:
            token_id = self._next_id
            self._next_id += 1
            self.id_to_token[token_id] = token_bytes
            self.token_to_id[token_bytes] = token_id

        # Record once; duplicates in the input list are ignored.
        self.special_token_to_id.setdefault(token_str, token_id)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        """Load vocab/merges produced by ``train_tokenizer.py`` and build a tokenizer."""

        with open(vocab_filepath, "r", encoding="latin-1") as vocab_file:
            vocab_data = json.load(vocab_file)

        vocab: Dict[int, bytes] = {}
        if isinstance(vocab_data, dict):
            iterator = vocab_data.items()
        else:  # Allow a plain list for convenience.
            iterator = enumerate(vocab_data)

        for key, value in iterator:
            idx = int(key)
            if idx < 256:
                vocab[idx] = bytes([idx])
            else:
                vocab[idx] = value.encode("utf-8")

        vocab_values = set(vocab.values())

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="latin-1") as merges_file:
            for raw_line in merges_file:
                if raw_line.startswith("#"):
                    continue

                # Preserve leading spaces inside tokens; only drop the trailing newline.
                line = raw_line.rstrip("\n")
                if not line or line.isspace():
                    continue

                parsed = cls._parse_merge_line(line, vocab_values)
                if parsed is None:
                    msg = f"Could not parse merge line: {line!r}"
                    raise ValueError(msg)
                merges.append(parsed)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @staticmethod
    def _parse_merge_line(
        line: str, vocab_values: set[bytes]
    ) -> Tuple[bytes, bytes] | None:
        """Recover the two byte tokens described on ``line``.

        Merges are written as ``token_a`` + space + ``token_b``. Tokens can begin
        or end with spaces themselves, so we try every possible split point where
        the surrounding pieces appear in the vocabulary.
        """

        space_positions = [idx for idx, char in enumerate(line) if char == " "]
        for idx in space_positions:
            left_str = line[:idx]
            right_str = line[idx + 1 :]
            if not right_str:
                continue

            try:
                left_bytes = left_str.encode("utf-8")
                right_bytes = right_str.encode("utf-8")
            except UnicodeEncodeError:
                left_bytes = left_str.encode("latin-1")
                right_bytes = right_str.encode("latin-1")

            if left_bytes in vocab_values and right_bytes in vocab_values:
                return left_bytes, right_bytes

        return None

    # ------------------------------------------------------------------
    # Public encoding / decoding API
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Encode ``text`` into BPE token IDs."""

        return list(self._encode_generator(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode each string from ``iterable`` without buffering the entire file."""

        for chunk in iterable:
            yield from self._encode_generator(chunk)

    def decode(self, ids: List[int]) -> str:
        """Convert BPE token IDs back into a UTF-8 string."""

        buffer = bytearray()
        for idx in ids:
            token_bytes = self.id_to_token.get(idx)
            if token_bytes is None:
                msg = f"Token ID {idx} is not present in the vocabulary"
                raise ValueError(msg)
            buffer.extend(token_bytes)
        return buffer.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Internal helpers used by both encode and encode_iterable
    # ------------------------------------------------------------------
    def _encode_generator(self, text: str) -> Iterator[int]:
        """Yield token IDs as soon as we know them.

        ``encode`` simply materialises the list, while ``encode_iterable`` streams
        the output directly. Keeping the logic in one place makes future tweaks
        (for example, a different pre-tokenizer) much easier.
        """

        if not text:
            return

        if not self._sorted_special_tokens:
            # Fast path when no special tokens are registered.
            for match in self._PRETOKEN_PATTERN.finditer(text):
                piece = match.group(0)
                if piece:
                    yield from self._bpe(piece.encode("utf-8"))
            return

        position = 0
        length = len(text)

        while position < length:
            match = self._match_special(text, position)
            if match is not None:
                special_str, token_id = match
                yield token_id
                position += len(special_str)
                continue

            next_break = length
            for special in self._sorted_special_tokens:
                idx = text.find(special, position)
                if idx != -1:
                    next_break = min(next_break, idx)
            chunk = text[position:next_break]

            if chunk:
                for match in self._PRETOKEN_PATTERN.finditer(chunk):
                    piece = match.group(0)
                    if piece:
                        yield from self._bpe(piece.encode("utf-8"))

            position = next_break

    def _match_special(self, text: str, position: int) -> Tuple[str, int] | None:
        """Return the special token string and ID starting at ``position`` (if any)."""

        for special in self._sorted_special_tokens:
            if text.startswith(special, position):
                return special, self.special_token_to_id[special]
        return None

    # ------------------------------------------------------------------
    # BPE core
    # ------------------------------------------------------------------
    def _bpe(self, token_bytes: bytes) -> Tuple[int, ...]:
        """Return the token IDs produced by applying BPE merges to ``token_bytes``."""

        if token_bytes in self._bpe_cache:
            return self._bpe_cache[token_bytes]

        # If we already have an explicit vocab entry, return it directly.
        direct = self.token_to_id.get(token_bytes)
        if direct is not None:
            encoded = (direct,)
            self._bpe_cache[token_bytes] = encoded
            return encoded

        symbols: List[bytes] = [bytes([byte]) for byte in token_bytes]

        if len(symbols) == 1:
            single = self.token_to_id.get(symbols[0])
            if single is None:
                msg = "Encountered byte not present in base vocabulary"
                raise ValueError(msg)
            encoded = (single,)
            self._bpe_cache[token_bytes] = encoded
            return encoded

        pairs = self._collect_pairs(symbols)

        while pairs:
            best_pair = self._best_ranked_pair(pairs)
            if best_pair is None:
                break

            first, second = best_pair
            merged: List[bytes] = []
            idx = 0
            while idx < len(symbols):
                if (
                    idx < len(symbols) - 1
                    and symbols[idx] == first
                    and symbols[idx + 1] == second
                ):
                    merged.append(symbols[idx] + symbols[idx + 1])
                    idx += 2
                else:
                    merged.append(symbols[idx])
                    idx += 1

            symbols = merged
            if len(symbols) == 1:
                break
            pairs = self._collect_pairs(symbols)

        try:
            encoded = tuple(self.token_to_id[sym] for sym in symbols)
        except KeyError as err:
            msg = "Encountered symbol not present in vocabulary after merges"
            raise ValueError(msg) from err

        self._bpe_cache[token_bytes] = encoded
        return encoded

    def _best_ranked_pair(self, pairs: List[Tuple[bytes, bytes]]) -> Tuple[bytes, bytes] | None:
        """Pick the pair with the smallest merge rank (lower rank wins)."""

        best_pair = None
        best_rank = None
        for pair in pairs:
            rank = self.bpe_ranks.get(pair)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_pair = pair
                best_rank = rank
        return best_pair

    @staticmethod
    def _collect_pairs(symbols: List[bytes]) -> List[Tuple[bytes, bytes]]:
        """Collect adjacent symbol pairs in their original order."""

        return [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
