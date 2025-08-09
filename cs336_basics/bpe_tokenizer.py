from cs336_basics.find_chunk_boundaries import find_chunk_boundaries
import regex as re
from collections import Counter

file_name = "data/TinyStoriesV2-GPT4-valid.txt"
num_proc = 200
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
eot_str = "<|endoftext|>"
eot_bytes = eot_str.encode("utf-8")


# def tokenize(chunk: bytes):
#     re.findall(PAT, "some text that i'll pre-tokenize")

#     pass
"""
Deliverable:
Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer.
Your BPE training function should handle (at least) the following input parameters:
input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size
            (including the initial byte vocabulary, vocabulary items produced from merging,
            and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary.
                These special tokens do not otherwise affect BPE training.
Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
       to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training.
        Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>.
        The merges should be ordered by order of creation.
"""

def build_initial_vocab():
    """
    Build the initial vocabulary from the byte-level tokens.
    This is a simple implementation that returns a mapping of byte values to their integer IDs.
    """
    vocab = {i: bytes([i]) for i in range(256)}  # Byte values from 0 to 255
    eot_token = len(vocab)
    vocab[eot_token] = eot_bytes
    return vocab, eot_token

def merge_pairs(tokens, to_merge, new_token):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i < len(tokens) - 1
            and tokens[i] == to_merge[0]
            and tokens[i + 1] == to_merge[1]
        ):
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

with open(file_name, "rb") as f:
    eot_bytes = "<|endoftext|>".encode("utf-8")
    boundaries = find_chunk_boundaries(f, num_proc, eot_bytes)
    start = boundaries[0]
    end = boundaries[1]
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")

    tokens = [int]
    no_merge_token = int(-1)
    vocab, eot_token = build_initial_vocab()

    subchunks = chunk.encode("utf-8").split(eot_bytes)
    for subchunk in subchunks[:2]:
        pre_tokens = re.findall(PAT, chunk)
        for pre_token in pre_tokens:
            ints = list(pre_token.encode("utf-8"))
            tokens += ints
            tokens.append(no_merge_token)
        tokens.append(eot_token)

    freq = Counter(
        (tokens[i], tokens[i+1])
        for i in range(len(tokens)-1)
        if no_merge_token not in (tokens[i], tokens[i+1]) and eot_token not in (tokens[i], tokens[i+1])
    )

    # select pair with the highest frequency and largest lexicographic order
    max_freq = max(freq.values())
    most_common = [k for k, v in freq.items() if v == max_freq]
    to_merge = sorted(most_common)[0]
    print(to_merge)

    new_token = len(vocab)
    vocab[new_token] = vocab[to_merge[0]] + vocab[to_merge[1]]

    new_tokens = merge_pairs(tokens, to_merge, new_token)
    print(f"len(tokens) = {len(tokens)}")
    print(f"len(new_tokens) = {len(new_tokens)}")




    exit(0)


    # # The following is a serial implementation, but you can parallelize this
    # # by sending each start/end pair to a set of processes.
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     f.seek(start)
    #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #     print(chunk)
    #     exit(0)
    #     # Run pre-tokenization on your chunk and store the counts for each pre-token
