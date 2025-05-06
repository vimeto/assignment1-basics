import re
import os
import regex as re
from collections import Counter, defaultdict
# import multiprocessing # No longer needed

PAT = re.compile(r"\'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

def _merge_word(word, pair, new_id):
    """Merge a single word in place - no string ops, no copies."""
    out, i = [], 0
    a, b = pair
    L = len(word)
    while i < L:
        if i < L - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out

# New function to update counts and locations for a single word
def _update_word_stats(word, word_idx, pair_counts, locations, delta):
    """
    Updates pair_counts and locations for a given word.
    delta = +1 for adding counts/locations, -1 for removing counts/locations.
    """
    for p in zip(word, word[1:]):
        # Update counts
        pair_counts[p] = pair_counts.get(p, 0) + delta # Use .get for safety on decrement

        # Update locations
        if delta > 0:
            # Add word_idx to the set for this pair
            if p not in locations:
                locations[p] = set()
            locations[p].add(word_idx)
        elif delta < 0:
             # Remove word_idx from the set; clean up if necessary
            if p in locations:
                locations[p].discard(word_idx)
                if not locations[p]: # If set becomes empty, remove pair from locations
                    del locations[p]

        if pair_counts[p] <= 0:
             # Ensure pair is removed if its count is non-positive
             if p in pair_counts:
                 del pair_counts[p]
             # Also ensure it's removed from locations if somehow still there
             if p in locations:
                 locations[p].discard(word_idx)
                 if not locations[p]:
                      del locations[p]


def tokenizer(input_path, vocab_size, special_tokens):
    """
    Train a byte-level BPE tokenizer using incremental updates,
    correctly handling special tokens by excluding them from merges.
    Returns
        vocab  : dict[int, bytes]              (id -> token bytes)
        merges : list[tuple[bytes, bytes]]     (history, in order)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    if special_tokens is None:
        special_tokens = []

    if special_tokens:
        unique_special_tokens_bytes = {st.encode('utf-8') for st in special_tokens}
        # Use non-capturing split
        split_pattern = "|".join(map(re.escape, special_tokens))
        text_chunks = re.split(split_pattern, raw)
    else:
        unique_special_tokens_bytes = set()
        text_chunks = [raw]
    del raw

    # Apply PAT to all chunks resulting from the split
    pre_tokens = []
    for chunk in text_chunks:
        if not chunk: continue # Skip empty chunks
        pre_tokens.extend(m.group(0) for m in PAT.finditer(chunk))

    words = [list(t.encode("utf-8")) for t in pre_tokens if t]
    del pre_tokens

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    next_id = 256

    num_merges = vocab_size - 256 - len(unique_special_tokens_bytes)
    if num_merges < 0:
        raise ValueError(f"Vocab size {vocab_size} too small for 256 bytes + {len(unique_special_tokens_bytes)} unique special tokens.")

    pair_counts = Counter()
    locations = defaultdict(set)
    for i, w in enumerate(words):
        _update_word_stats(w, i, pair_counts, locations, +1)

    for _ in range(num_merges):
        if not pair_counts: break
        try:
             valid_pairs = {p: c for p, c in pair_counts.items() if p[0] in vocab and p[1] in vocab}
             if not valid_pairs: break
             best_pair = max(valid_pairs, key=lambda p: (valid_pairs[p], vocab[p[0]], vocab[p[1]]))
        except KeyError as e:
             print(f"Warning: KeyError finding best pair {e}. This might indicate an issue.")
             break
        if best_pair not in pair_counts or pair_counts[best_pair] <= 0:
             if best_pair in pair_counts: del pair_counts[best_pair]
             if best_pair in locations: del locations[best_pair]
             continue
        new_id = next_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        next_id += 1
        affected_indices = list(locations[best_pair])
        new_words_map = {}
        for idx in affected_indices:
            w = new_words_map.get(idx, words[idx])
            if not w: continue
            _update_word_stats(w, idx, pair_counts, locations, -1)
            merged_w = _merge_word(w, best_pair, new_id)
            new_words_map[idx] = merged_w
            _update_word_stats(merged_w, idx, pair_counts, locations, +1)
        for idx, merged_w in new_words_map.items():
            words[idx] = merged_w
        if best_pair in locations: del locations[best_pair]
        if best_pair in pair_counts: del pair_counts[best_pair]

    for tok_bytes in unique_special_tokens_bytes:
        vocab[next_id] = tok_bytes
        next_id += 1

    return vocab, merges
