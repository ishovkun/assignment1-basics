from cs336_basics.tokenizer.tokenizer import Tokenizer
import os
import pickle
# from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    vocab_filename = "TinyStoriesV2-GPT4-train_vocab10000.pkl"
    # target_filename = "data/TinyStoriesV2-GPT4-train.txt"
    target_filename = "data/TinyStoriesV2-GPT4-valid.txt"
    strip_extension = lambda filename: os.path.splitext(os.path.basename(filename))[0]

    with open(vocab_filename, "rb") as f:
        print(f"Loading vocab file {vocab_filename}")
        data = pickle.load(f)

    vocab, merges = data['vocab'], data['merges']
    special_tokens = ["<|endoftext|>"]

    vocab_size = len(vocab)
    output_filename = f"{strip_extension(target_filename)}_tokenized_{vocab_size}.npy"

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokens: int = []

    with open(target_filename, 'r') as f:
        old_pos = 0
        for token in tokenizer.encode_iterable(f, print_progress=True):
            tokens.append(token)

    print(f"saving {output_filename}")
    np.save(output_filename, np.array(tokens, dtype=int))
