import os
import json
import time

import fast_tokenizer


def train_tokenizer(input_path, vocab_size, special_tokens, output_dir):
    start_time = time.time()
    vocab, merges = fast_tokenizer.tokenizer(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    end_time = time.time()
    print(f"Time taken to train tokenizer: {end_time - start_time} seconds")

    os.makedirs(output_dir, exist_ok=True)

    merges_path = os.path.join(output_dir, "merges.txt")
    vocab_path = os.path.join(output_dir, "vocab.json")

    vocab_str: dict[int, str] = {
        idx: tok_bytes.decode("utf-8", errors="replace")
        for idx, tok_bytes in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            tokens = (tok.decode("utf-8", errors="replace") for tok in merge)
            f.write(" ".join(tokens) + "\n")

    return vocab, merges


if __name__ == "__main__":
    tinystories_path = "../data/TinyStoriesV2-GPT4-train.txt"
    mini_path = "tests/fixtures/tinystories_sample_5M.txt"
    owt_path = "../data/owt_train.txt"

    train_tokenizer(owt_path, 32000, ["<|endoftext|>"], "../tokenizers/owt_tokenizer_32k_2")
