from io import BufferedReader
from numpy._core.numerictypes import void
from cs336_basics.find_chunk_boundaries import find_chunk_boundaries
import regex as re
from collections import Counter
from typing import Dict, Tuple, List
import multiprocessing
import itertools
from tqdm import tqdm
import time
import sys

def build_initial_vocab() -> dict[int, bytes]:
    """
    Build the initial vocabulary from the byte-level tokens.
    This is a simple implementation that returns a mapping of byte values to their integer IDs.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab

class TokenNode:
    def __init__(self, token: int):
        self.token: int = token
        self.prev: TokenNode = None
        self.next: TokenNode = None
        self.can_merge: bool = True

class TokenList:
    def __init__(self, root: TokenNode, tail: TokenNode, size: int):
        self.root: TokenNode = root
        self.tail: TokenNode = tail
        self.size: int = size

    def serialize(self) -> List[Tuple[int, bool]]:
        tokens = []
        node = self.root
        while node is not None:
            tokens.append((node.token, node.can_merge))
            node = node.next
        return tokens

    @staticmethod
    def deserialize(tokens: list[tuple[int, bool]]):
        root = None
        tail = None
        for token, can_merge in tokens:
            node = TokenNode(token)
            node.can_merge = can_merge
            if root is None:
                root = node
                tail = node
            else:
                tail.next = node
                node.prev = tail
                tail = node
        return TokenList(root, tail, len(tokens))

def pretokenize(chunk: str, special_tokens: list[str], pbar = None):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)
    subchunks = re.split(pattern, chunk.encode("utf-8"))
    root = None
    tail = None
    num_tokens = 0
    if pbar is not None:
        pbar.total = len(subchunks)
        pbar.refresh()

    for subchunk in subchunks:
        pre_tokens = re.findall(PAT, chunk)
        for pre_token in pre_tokens:
            ints = list(pre_token.encode("utf-8"))
            offset = 0
            if root is None:
                root = TokenNode(ints[0])
                tail = root
                offset = 1
                num_tokens += 1
            for i in range(offset, len(ints)):
                node = TokenNode(ints[i])
                num_tokens += 1
                tail.next = node
                node.prev = tail
                tail = node
            tail.can_merge = False
        if pbar is not None:
            pbar.update(1)
    return TokenList(root, tail, num_tokens)

def build_pairs(root: TokenNode) -> dict[tuple[int, int], set[TokenNode]]:
    """
    Build a frequency table of pairs of tokens in the linked list.
    The keys are tuples of (token1, token2) and the values are sets of TokenNodes
    that represent the pairs.
    """
    pairs: dict[tuple(int, int), set[TokenNode]] = {}
    node = root
    while node.next is not None:
        if node.can_merge:
            pair = (node.token, node.next.token)
            if pair not in pairs:
                pairs[pair] = set()
            pairs[pair].add(node)
        node = node.next
    return pairs

def list_length(root: TokenNode) -> int:
    length = 0
    node = root
    while node is not None:
        length += 1
        node = node.next
    return length

def get_top_pair(freq: dict[tuple[int, int], set[TokenNode]],
                 vocab: dict[int, bytes]) -> tuple[int, int]:
    max_freq = 0
    max_items = []
    for key, nodes in freq.items():
        if len(nodes) > max_freq:
            max_freq = len(nodes)
            max_items = [key]
        elif len(nodes) == max_freq:
            max_items.append(key)
    if len(max_items) == 0:
        print(f"Freq table size = {len(freq)}")
        return None
    else:
        # find lexicographically largest pair
        enc = [ [vocab[x[0]], vocab[x[1]]] for x in max_items ]
        idx = max(range(len(enc)), key=enc.__getitem__)
        return max_items[idx]

def merge(tokenList: TokenList, vocab: Dict[int, bytes],
            max_vocab_size: int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    pairs = build_pairs(tokenList.root)
    num_tokens = tokenList.size
    num_tokens_old = num_tokens * 2
    iter: int = 1
    merges : list[tuple[int, int]] = []
    while len(vocab) < max_vocab_size and num_tokens < num_tokens_old:
        num_tokens_old = num_tokens

        merged = get_top_pair(pairs, vocab)
        if merged is None:
            print("Nothing to merge")
            break

        # create new token
        new_token = len(vocab)
        bytes1 = vocab[merged[0]]
        bytes2 = vocab[merged[1]]
        vocab[new_token] = bytes1 + bytes2
        merges.append((bytes1, bytes2))

        todo_nodes = pairs[merged]
        pair_count = len(todo_nodes)

        # Take care of sequences of more than 2 repeated tokens
        for node in todo_nodes.copy():
            if node in todo_nodes and node.next in todo_nodes:
                todo_nodes.remove(node.next)

        for node in todo_nodes:

            # Remove node.next -> node.next.next from pairs
            if node.next.can_merge and node.next.next is not None:
                pair = (node.next.token, node.next.next.token)
                if node.next in pairs[pair]:
                    pairs[pair].remove(node.next)
                if len(pairs[pair]) == 0:
                    pairs.pop(pair)

            # Remove node.prev -> node from pairs
            if node.prev is not None and node.prev.can_merge:
                pair = (node.prev.token, node.token)
                if node.prev in pairs[pair]:
                    pairs[pair].remove(node.prev)
                if len(pairs[pair]) == 0:
                    pairs.pop(pair)

            # Merge = update current node token + remove next node
            # Update node token
            node.token = new_token
            num_tokens -= 1
            # remove next node
            node.can_merge = node.next.can_merge
            if node.next.next is not None:
                node.next.next.prev = node
            node.next = node.next.next

            # add pair node.prev -> node
            if node.prev is not None and node.prev.can_merge:
                pair = (node.prev.token, node.token)
                if pair not in pairs:
                    pairs[pair] = set()
                pairs[pair].add(node.prev)
            # add pair node -> node.next
            if node.next is not None and node.can_merge:
                pair = (node.token, node.next.token)
                if pair not in pairs:
                    pairs[pair] = set()
                pairs[pair].add(node)

        # finally, remove merged pair from pairs
        if merged in pairs:
            pairs.pop(merged)
        assert merged not in pairs, f"Pair {merged} should not be in pairs"

        # report mertics
        len_reduction = 1. - num_tokens / num_tokens_old
        print(f"{iter}: num_tokens = {num_tokens}\t shrink = {len_reduction:.2%}%\tvocab = {len(vocab)}\tPair_freq = {pair_count}\tpair = {merged}")
        iter += 1

    print("merging done")
    return vocab, merges

def pretokenize_worker(filename: str, start: int, end: int, special_tokens: list[str],
                       worker_id: int = 0) :#-> list[Tuple[int, bool]]:
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        tokenList = None
        with tqdm(desc=f"Worker {worker_id}", position=worker_id) as pbar:
            tokenList = pretokenize(chunk, special_tokens, pbar)
        tokens = tokenList.serialize()
        return tokens

def pretokenize_parallel(filename : str, special_tokens: list[str], num_proc: int):

    # serial: find boundaries
    boundaries: list[int] = []
    with open(filename, "rb") as f:
        eot_str = "<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_proc, eot_str.encode("utf-8"))

    # Spawm multiple workers
    tasks = [
        (filename, start, end, special_tokens, idx)
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    print("Running pretokenization in parallel with", num_proc, "processes")
    with multiprocessing.Pool(num_proc) as pool:
        results = pool.starmap(pretokenize_worker, tasks)

    # turn [[token1, token2], [token3, token4], ...] into flat list
    tokens = list(itertools.chain.from_iterable(results))

    # Convert into a linked list
    return TokenList.deserialize(tokens)


def tokenize(file_name: str, vocab_size: int, special_tokens: list[str], num_proc: int
        ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    vocab = build_initial_vocab()
    eot_str = "<|endoftext|>"
    if eot_str not in special_tokens:
        special_tokens.append(eot_str)

    for special in special_tokens:
        token = len(vocab)
        vocab[token] = special.encode("utf-8")

    parallel: bool = num_proc > 1
    if parallel:
        tokenList = pretokenize_parallel(file_name, special_tokens, num_proc)
    else:
        with open(file_name, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
            print("Pre-tokenize")
            with tqdm() as pbar:
                tokenList = pretokenize(text, special_tokens, pbar)

    vocab, merges = merge(tokenList, vocab, vocab_size)
    return vocab, merges

if __name__ == "__main__":
    vocab_size: int = 500

    # file_name: str = "data/TinyStoriesV2-GPT4-valid.txt"
    # file_name: str = "data/mini_test.txt"
    file_name: str = "tests/fixtures/corpus.en"

    # num_proc = multiprocessing.cpu_count() - 2
    num_proc = 1
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        num_proc = int(sys.argv[2])

    eot_str = "<|endoftext|>"
    tokenize(file_name, vocab_size, special_tokens=[eot_str], num_proc=num_proc)

    # # The following is a serial implementation, but you can parallelize this
    # # by sending each start/end pair to a set of processes.
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     f.seek(start)
    #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #     print(chunk)
    #     exit(0)
    #     # Run pre-tokenization on your chunk and store the counts for each pre-token
