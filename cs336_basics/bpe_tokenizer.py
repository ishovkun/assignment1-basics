from cs336_basics.find_chunk_boundaries import find_chunk_boundaries
import regex as re
from collections import Counter
from typing import Dict, Tuple, List

num_proc = 200
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# def tokenize(chunk: bytes):
#     re.findall(PAT, "some text that i'll pre-tokenize")

#     pass

def build_initial_vocab() -> dict[int, bytes]:
    """
    Build the initial vocabulary from the byte-level tokens.
    This is a simple implementation that returns a mapping of byte values to their integer IDs.
    """
    vocab = {i: bytes([i]) for i in range(256)}  # Byte values from 0 to 255
    return vocab

def pretokenize(chunk: str, special_tokens: list[str], no_merge_token: int):
    tokens = [int]
    pattern = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)
    subchunks = re.split(pattern, chunk.encode("utf-8"))
    for subchunk in subchunks:
        pre_tokens = re.findall(PAT, chunk)
        for pre_token in pre_tokens:
            ints = list(pre_token.encode("utf-8"))
            tokens += ints
            tokens.append(no_merge_token)
    return tokens

def select_merge_pair(freq : Counter):
    # select pair with the highest frequency and largest lexicographic order
    max_freq = max(freq.values())
    most_common = [k for k, v in freq.items() if v == max_freq]
    to_merge = sorted(most_common)[0]
    return to_merge

def merge(tokens: list[int], vocab: dict[int, bytes], no_merge_token: int):
    init_len = len(tokens)
    old_len = 2*init_len
    new_len = init_len
    while len(vocab) < vocab_size and new_len < old_len:

        freq = Counter(
            (tokens[i], tokens[i+1])
            for i in range(len(tokens)-1)
            if no_merge_token not in (tokens[i], tokens[i+1])
        )

        to_merge = select_merge_pair(freq)

        new_token = len(vocab)
        vocab[new_token] = vocab[to_merge[0]] + vocab[to_merge[1]]

        new_tokens = merge_pairs(tokens, to_merge, new_token)

        old_len = len(tokens)
        new_len = len(new_tokens)
        shrink = 1. - new_len / old_len
        print(f"Corpus tokens reduced = {shrink:.2%} : {old_len} -> {new_len}", end="\t")
        print(f"Vocab size = {len(vocab)}")

        tokens = new_tokens
    return tokens

class TokenNode:
    def __init__(self, token: int):
        self.token = token
        self.prev = None
        self.next = None

def build_freq_table(root: TokenNode, no_merge_token: int) -> dict[tuple[int, int], set[TokenNode]]:
    freq: dict[tuple(bytes), set[TokenNode]] = {}
    node = root
    while node.next is not None:
        next = node.next
        if node.token != no_merge_token and next.token != no_merge_token:
            pair = (node.token, next.token)
            if pair not in freq:
                freq[pair] = set()
            freq[pair].add(node)
        node = next
    return freq

def list_length(root: TokenNode) -> int:
    length = 0
    node = root
    while node is not None:
        length += 1
        node = node.next
    return length

def get_top_pair(freq: dict[tuple[int, int], set[TokenNode]]) -> tuple[int, int]:
    max_freq = 0
    max_items = []
    for key, nodes in freq.items():
        if len(nodes) > max_freq:
            max_freq = len(nodes)
            max_items = [key]
        elif len(nodes) == max_freq:
            max_items.append(key)
    if len(max_items) == 0:
        return None
    else: return sorted(max_items)[0]

def merge_fast(tokens: list[int], vocab: dict[int, bytes], no_merge_token: int):
    # turn tokens into a doubly linked list
    print("building linked list")
    root = TokenNode(tokens[0])
    prev = root
    num_tokens = 1
    for i in range(1, len(tokens)):
        node = TokenNode(tokens[i])
        node.prev = prev
        prev.next = node
        prev = node
        num_tokens += 1
    assert num_tokens == len(tokens), "Linked list length mismatch"

    # build frequency table
    print("Building frequency table...")
    freq = build_freq_table(root, no_merge_token)

    num_tokens_old = num_tokens * 2
    while len(vocab) < vocab_size and num_tokens < num_tokens_old:

        # length = list_length(root)
        # if num_tokens != length:
        #     print(f"counted tokens = {num_tokens}, list length = {length}")
        #     exit(1)

        num_tokens_old = num_tokens

        merged = get_top_pair(freq)
        nodes = freq[merged]
        pair_count = len(nodes)

        # create new token
        new_token = len(vocab)
        vocab[new_token] = vocab[merged[0]] + vocab[merged[1]]

        # update freq
        for node in nodes:
            assert node.next not in nodes, "Next node should not be in the same set"

            if node.next.next is not None:
                pair = (node.next.token, node.next.next.token)
                if pair[1] != no_merge_token:
                    freq[pair].remove(node.next)
                    if len(freq[pair]) == 0:
                        del freq[pair]

            if node.prev is not None:
                pair = (node.prev.token, node.token)
                if pair[0] != no_merge_token:
                    freq[pair].remove(node.prev)
                    if len(freq[pair]) == 0:
                        del freq[pair]

            node.token = new_token
            num_tokens -= 1

            # remove next node
            node.next = node.next.next
            node.next.prev = node

            # update freq prev->cur
            if node.prev is not None and node.prev.token != no_merge_token:
                pair = (node.prev.token, node.token)
                if pair not in freq:
                    freq[pair] = set()
                freq[pair].add(node.prev)

            # update frequency of the new pair
            if node.next is not None and node.next.token != no_merge_token:
                pair = (node.token, node.next.token)
                if pair not in freq:
                    freq[pair] = set()
                freq[pair].add(node)

        del freq[merged]

        len_reduction = 1. - num_tokens / num_tokens_old
        print(f"num_tokens = {num_tokens}\t shrink = {len_reduction:.2%}\tvocab = {len(vocab)}\tPair_freq = {pair_count}\tpair = {merged}")
        if len_reduction < 0:
            print("fuck")
            exit(1)
    print("Done")
    # print(f"Reduced: {len(freq)}")


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

# vocab_size = 10000
def tokenize(file_name: str, vocab_size: int, special_tokens: list[str]
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
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
    with open(file_name, "rb") as f:

        eot_str = "<|endoftext|>"

        boundaries = find_chunk_boundaries(f, num_proc, eot_str.encode("utf-8"))

        vocab = build_initial_vocab()
        if eot_str not in special_tokens:
            special_tokens.append(eot_str)

        for special in special_tokens:
            token = len(vocab)
            vocab[token] = special.encode("utf-8")

        start = boundaries[0]
        end = boundaries[1]
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        no_merge_token :int = -1
        tokens = pretokenize(chunk, special_tokens, no_merge_token)
        # tokens = merge(tokens, vocab, no_merge_token)
        tokens = merge_fast(tokens, vocab, no_merge_token)


        exit(0)

vocab_size: int = 10000
file_name: str = "data/TinyStoriesV2-GPT4-valid.txt"
eot_str = "<|endoftext|>"
# eot_bytes = eot_str.encode("utf-8")
tokenize(file_name, vocab_size, special_tokens=[eot_str])

    # # The following is a serial implementation, but you can parallelize this
    # # by sending each start/end pair to a set of processes.
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     f.seek(start)
    #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #     print(chunk)
    #     exit(0)
    #     # Run pre-tokenization on your chunk and store the counts for each pre-token
