import re
import time
import regex as re
from typing import BinaryIO, Tuple, List, Counter, Dict
from collections import Counter, defaultdict
import multiprocessing
from pretokenization_example import find_chunk_boundaries

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

def _calculate_chunk_stats(chunk_data):
    """
    Calculates local pair counts and locations for a chunk of words.

    Args:
        chunk_data (tuple): A tuple containing (start_index, word_list_chunk).

    Returns:
        tuple: A tuple containing (local_pair_counts, local_locations).
    """
    start_index, word_chunk = chunk_data
    local_counts = Counter()
    local_locations = defaultdict(set)

    for i, word in enumerate(word_chunk):
        word_idx = start_index + i # Calculate original index
        for p in zip(word, word[1:]):
            local_counts[p] += 1
            local_locations[p].add(word_idx)

    return local_counts, local_locations

def _process_from_boundaries(
    params_tuple: Tuple[str, int, int, List[str]]
) -> Tuple[List[List[int]], Counter, Dict[Tuple[int, int], set]]:
    """
    Process a chunk of the file from start to end, split by special tokens.
    """
    input_path, start, end, split_special_tokens_str = params_tuple
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

    # Use regex.escape and regex.split if PAT uses regex module
    if split_special_tokens_str: # Avoid empty pattern for re.split if list is empty
        split_pattern = "|".join(map(re.escape, split_special_tokens_str))
        text_sub_chunks = re.split(split_pattern, chunk_str)
    else:
        text_sub_chunks = [chunk_str]


    worker_words: List[List[int]] = [] # All words processed by this worker
    worker_pair_counts = Counter()
    worker_locations = defaultdict(set)

    # This tracks the current base index for words being added to worker_words
    current_word_offset_in_worker = 0
    for text_part in text_sub_chunks:
        if not text_part: # Skip empty strings that can result from re.split
            continue

        # Words from this specific part, encoded to lists of utf-8 byte values
        words_from_part = [list(m.group(0).encode("utf-8")) for m in PAT.finditer(text_part)]

        if not words_from_part:
            continue

        # Calculate stats for this part.
        # _calculate_chunk_stats needs the offset where these words will start in `worker_words`.
        part_counts, part_locations = _calculate_chunk_stats(
            (current_word_offset_in_worker, words_from_part)
        )

        worker_words.extend(words_from_part)
        worker_pair_counts.update(part_counts)
        for pair, index_set in part_locations.items():
            worker_locations[pair].update(index_set)
        current_word_offset_in_worker += len(words_from_part)

    return worker_words, worker_pair_counts, worker_locations

def tokenizer(input_path, vocab_size, special_tokens):
    """
    Train a byte-level BPE tokenizer using incremental updates,
    correctly handling special tokens by excluding them from merges.
    Returns
        vocab  : dict[int, bytes]              (id -> token bytes)
        merges : list[tuple[bytes, bytes]]     (history, in order)
    """
    print(f"Training tokenizer on {input_path} with vocab size {vocab_size} and special tokens {special_tokens}")

    start_time = time.time()
    num_processes = multiprocessing.cpu_count()
    with open(input_path, "rb") as f:
        # for now we'll just use the <|endoftext|> token as the special token
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))


    end_time = time.time()
    print(f"Time taken to find chunk boundaries: {end_time - start_time} seconds")

    num_processes = multiprocessing.cpu_count()
    multiprocessing_params = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:]) if end > start # Ensure non-empty chunks
    ]

    # If no valid boundaries for parallel processing, run serially or handle error
    if not multiprocessing_params:
        print("Warning: No processable chunks found. Check file size or boundary logic.")
        # Fallback to single process or process the whole file if small enough?
        # For now, let's assume multiprocessing_params will be populated.
        # If it can be empty and cause issues downstream:
        if boundaries[-1] > boundaries[0]: # Check if there is any data at all
             multiprocessing_params = [(str(input_path), boundaries[0], boundaries[-1], special_tokens)]
        else:
            # Handle empty file or single boundary point.
            # For BPE, an empty file results in an empty/initial vocab.
            # This case should be handled gracefully.
            # For simplicity, assuming file is not empty and boundaries make sense.
            pass

    print(multiprocessing_params)

    pool_results: List[Tuple[List[List[int]], Counter, Dict[Tuple[int, int], set]]] = []
    if multiprocessing_params: # Only run pool if there are tasks
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool_results = pool.map(_process_from_boundaries, multiprocessing_params)
    else:
        print("No data to process with multiprocessing.")


    multiprocessing_time = time.time()
    print(f"Time taken to process from boundaries: {multiprocessing_time - end_time:.4f} seconds")

    all_words: List[List[int]] = []
    global_pair_counts = Counter()
    global_locations = defaultdict(set)

    running_word_idx_offset = 0
    for worker_word_list, worker_counts, worker_locs in pool_results:
        all_words.extend(worker_word_list)
        global_pair_counts.update(worker_counts)
        for pair, index_set in worker_locs.items():
            # Adjust indices from worker-local to global
            global_locations[pair].update(idx + running_word_idx_offset for idx in index_set)
        running_word_idx_offset += len(worker_word_list)

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: List[Tuple[bytes, bytes]] = [] # Stores (byte_seq1, byte_seq2) for merged pairs
    next_id = 256

    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError(f"Vocab size {vocab_size} too small for 256 bytes + {len(special_tokens)} unique special tokens.")

    pre_tokenize_time = time.time()
    print(f"Time taken to pre-tokenize: {pre_tokenize_time - multiprocessing_time} seconds")

    for i in range(num_merges):
        if not global_pair_counts: break
        try:
             valid_pairs = {p: c for p, c in global_pair_counts.items() if p[0] in vocab and p[1] in vocab}
             if not valid_pairs: break
             best_pair = max(valid_pairs, key=lambda p: (valid_pairs[p], vocab[p[0]], vocab[p[1]]))
        except KeyError as e:
             print(f"Warning: KeyError finding best pair {e}. This might indicate an issue.")
             break
        if best_pair not in global_pair_counts or global_pair_counts[best_pair] <= 0:
             if best_pair in global_pair_counts: del global_pair_counts[best_pair]
             if best_pair in global_locations: del global_locations[best_pair]
             continue
        new_id = next_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        next_id += 1
        affected_indices = list(global_locations[best_pair])
        new_words_map = {}
        for idx in affected_indices:
            w = new_words_map.get(idx, all_words[idx])
            if not w: continue
            _update_word_stats(w, idx, global_pair_counts, global_locations, -1)
            merged_w = _merge_word(w, best_pair, new_id)
            new_words_map[idx] = merged_w
            _update_word_stats(merged_w, idx, global_pair_counts, global_locations, +1)
        for idx, merged_w in new_words_map.items():
            all_words[idx] = merged_w
        if best_pair in global_locations: del global_locations[best_pair]
        if best_pair in global_pair_counts: del global_pair_counts[best_pair]

        if i % 100 == 0:
            print(f"Time taken to merge {i} pairs: {time.time() - start_time} seconds")

    merge_time = time.time()
    print(f"Time taken to merge: {merge_time - pre_tokenize_time} seconds")

    for tok_bytes in special_tokens:
        vocab[next_id] = tok_bytes.encode("utf-8")
        next_id += 1

    return vocab, merges
