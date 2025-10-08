import re
import time
import regex as re
from typing import BinaryIO, Tuple, List, Counter, Dict
from collections import Counter, defaultdict
import multiprocessing
from .pretokenization_example import find_chunk_boundaries

import os, mmap, time, heapq, itertools, multiprocessing as mp
from collections import Counter, defaultdict
from array import array
from typing import List, Dict, Tuple, Iterable

def word_to_array(w: bytes) -> array:
    """Encode a UTF-8 byte string to array('I') of *byte values*."""
    return array("I", memoryview(w))   # automatic element-wise conversion

def merge_word(
    word: array, pair: Tuple[int, int], new_id: int
) -> array:
    """Return a *new* array where the given pair is merged into new_id."""
    a, b = pair
    out = array("I")
    i, L = 0, len(word)
    while i < L:
        if i < L - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out

def iter_pairs(word: array) -> Iterable[Tuple[int, int]]:
    """Yield all adjacent pairs in the word."""
    return zip(word, itertools.islice(word, 1, None))

# --------------------------------------------------------------------------- #
#  Main trainer                                                               #
# --------------------------------------------------------------------------- #

def tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str] | Tuple[str, ...] = ("<|endoftext|>",),
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    t0 = time.time()

    # ---- Pre-tokenization: Adopt tokenizer.py strategy ----
    print("⚙️  Preparing initial word counts...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Define the simple pattern for tokenizing words (original PAT)
    simple_pat_str = r"\'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    simple_pat_compiled = re.compile(simple_pat_str)

    text_chunks_for_pat: List[str] = []
    unique_special_tokens_bytes = set()

    if special_tokens:
        # Ensure special_tokens are strings for re.escape and processing
        str_special_tokens = [str(st) for st in special_tokens if st] # Filter out empty/None
        if str_special_tokens:
            unique_special_tokens_bytes = {st.encode('utf-8') for st in str_special_tokens}
            # Create split pattern that discards delimiters (special tokens)
            split_pattern_str = "|".join(map(re.escape, str_special_tokens))
            text_chunks_for_pat = re.split(split_pattern_str, raw_text)
        else: # No valid special tokens provided
            text_chunks_for_pat = [raw_text]
    else:
        text_chunks_for_pat = [raw_text]

    word_freq = Counter()
    for chunk in text_chunks_for_pat:
        if not chunk: continue # Skip empty chunks that can result from re.split
        for match in simple_pat_compiled.finditer(chunk):
            word_bytes = match.group(0).encode("utf-8")
            word_freq[word_bytes] += 1

    print(f"   → {len(word_freq):,} unique words found in {time.time() - t0:.1f} s (pre-tokenization)")

    # ---- Initial data structures ----
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    merges: List[Tuple[bytes, bytes]] = []

    words_list: List[array] = []
    freqs_list: List[int] = []

    for w_bytes, f_count in word_freq.items():
        words_list.append(word_to_array(w_bytes))
        freqs_list.append(f_count)

    pair_cnt = Counter()
    pair_idx: Dict[Tuple[int, int], set[int]] = defaultdict(set)

    for idx, (word_array, freq_val) in enumerate(zip(words_list, freqs_list)):
        for p in iter_pairs(word_array):
            pair_cnt[p] += freq_val
            pair_idx[p].add(idx)

    # ---- BPE merges ----
    # Correctly calculate number of merges considering unique special token byte strings
    num_merges_to_perform = vocab_size - 256 - len(unique_special_tokens_bytes)
    if num_merges_to_perform < 0:
        # Handle case where vocab_size is too small for base 256 and all unique special tokens
        print(f"Warning: vocab_size {vocab_size} is too small for 256 base tokens and {len(unique_special_tokens_bytes)} unique special tokens. Adjusting to 0 merges.")
        num_merges_to_perform = 0
        # Or, could raise ValueError as in tokenizer.py, but let's try to proceed if possible.
        # For tests, this should align with num_merges in tokenizer.py for comparable results.

    print(f"⚙️  merging {num_merges_to_perform} pairs ...")
    report_every = max(1, num_merges_to_perform // 10) if num_merges_to_perform > 0 else 1

    for merge_no in range(num_merges_to_perform):
        if not pair_cnt:
            break

        # Filter pairs to ensure tokens are in vocab before calling max()
        # This mirrors tokenizer.py's valid_pairs approach more closely.
        valid_pairs_to_consider = {p: count for p, count in pair_cnt.items() if p[0] in vocab and p[1] in vocab}
        if not valid_pairs_to_consider:
            # print(f"Iteration {merge_no+1}: No valid pairs left to merge.")
            break

        # Use the valid_pairs_to_consider for selecting the best_pair
        best = max(
            valid_pairs_to_consider.keys(),
            key=lambda p: (valid_pairs_to_consider[p], vocab[p[0]], vocab[p[1]])
        )

        new_id = next_id
        next_id += 1

        merges.append((vocab[best[0]], vocab[best[1]]))
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]

        affected_word_indices = list(pair_idx.pop(best)) # Get affected indices and remove best from pair_idx
        pair_cnt.pop(best, None) # Remove best from pair_cnt

        for w_id in affected_word_indices:
            old_w_array = words_list[w_id]
            current_word_freq = freqs_list[w_id]

            # Remove old pairs from this word (old_w_array)
            for p_old in iter_pairs(old_w_array):
                if p_old not in pair_cnt: continue # Pair might have been removed by a previous operation in this loop if best_pair was p_old

                pair_cnt[p_old] -= current_word_freq

                current_locations_of_p_old = pair_idx.get(p_old)
                if current_locations_of_p_old is not None:
                    current_locations_of_p_old.discard(w_id)
                    if not current_locations_of_p_old: # If p_old no longer exists in ANY word
                        pair_idx.pop(p_old, None)
                        # If its locations are empty, its count should also reflect this.
                        # The reference tokenizer.py pops pair_counts if locations becomes empty.
                        pair_cnt.pop(p_old, None)

                if p_old in pair_cnt and pair_cnt[p_old] <= 0:
                    pair_cnt.pop(p_old, None)
                    pair_idx.pop(p_old, None) # Ensure consistency

            new_w_array = merge_word(old_w_array, best, new_id)
            words_list[w_id] = new_w_array

            # Add new pairs from this word (new_w_array)
            for p_new in iter_pairs(new_w_array):
                pair_cnt[p_new] += current_word_freq
                pair_idx[p_new].add(w_id)

        if (merge_no + 1) % report_every == 0 or merge_no == num_merges_to_perform -1 :
            print(f"   {merge_no + 1:>6}/{num_merges_to_perform}  "
                  f"pairs done   vocab={len(vocab):,}  "
                  f"@ {time.time() - t0:.1f} s")

    # ---- Add special tokens to vocab ----
    # This part was already correct, using unique_special_tokens_bytes if they exist
    for tok_bytes in sorted(list(unique_special_tokens_bytes)): # Sort for deterministic order
        if tok_bytes not in vocab.values(): # Avoid adding if already there (e.g. as a byte token)
            vocab[next_id] = tok_bytes
            next_id += 1
        # else:
            # print(f"Skipping special token {tok_bytes} as it's already in vocab values.")

    print(f"✅  finished in {time.time() - t0:.1f} s  "
          f"peak vocab {len(vocab):,}")
    return vocab, merges
