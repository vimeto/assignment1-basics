import tokenizer
import os
import json
import time

def train_tokenizer(input_path, vocab_size, special_tokens, output_dir):
    start_time = time.time()
    vocab, merges = tokenizer.tokenizer(input_path, vocab_size, special_tokens)
    end_time = time.time()
    print(f"Time taken to train tokenizer: {end_time - start_time} seconds")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    with open(merges_path, "w") as f:
        for merge in merges:
            f.write(" ".join(merge) + "\n")

    return vocab, merges


if __name__ == "__main__":
    tinystories_path = "data/TinyStoriesV2-GPT4-train.txt"
    owt_path = "data/owt_train.txt"

    train_tokenizer(tinystories_path, 10000, ["<|endoftext|>"], "owt_tokenizer_1k")
