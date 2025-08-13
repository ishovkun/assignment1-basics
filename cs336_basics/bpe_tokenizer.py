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
import numpy as np

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
        self.can_merge: bool = True

# class TokenList:
#     def __init__(self, root: TokenNode, tail: TokenNode, size: int):
#         self.root: TokenNode = root
#         self.tail: TokenNode = tail
#         self.size: int = size

#     def serialize(self) -> List[Tuple[int, bool]]:
#         tokens = []
#         node = self.root
#         while node is not None:
#             tokens.append((node.token, node.can_merge))
#             node = node.next
#         return tokens

#     @staticmethod
#     def deserialize(tokens: list[tuple[int, bool]]):
#         root = None
#         tail = None
#         for token, can_merge in tokens:
#             node = TokenNode(token)
#             node.can_merge = can_merge
#             if root is None:
#                 root = node
#                 tail = node
#             else:
#                 tail.next = node
#                 node.prev = tail
#                 tail = node
#         return TokenList(root, tail, len(tokens))

def pretokenize(chunk: str, special_tokens: list[str], pbar = None
) -> list[TokenNode]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)
    subchunks = re.split(pattern, chunk.encode("utf-8"))
    tokens : TokenNode = []
    if pbar is not None:
        pbar.total = len(subchunks)
        pbar.refresh()

    for subchunk in subchunks:
        pre_tokens = re.findall(PAT, chunk)
        for pre_token in pre_tokens:
            ints = list(pre_token.encode("utf-8"))
            for token in ints:
                tokens.append(TokenNode(token))
            tokens[-1].can_merge = False
        if pbar is not None:
            pbar.update(1)
    # return TokenList(root, tail, num_tokens)
    return tokens

def build_pairs(tokens: list[TokenNode]) -> dict[tuple[int, int], set[int]]:
    """
    Build a frequency table of pairs of tokens in the linked list.
    The keys are tuples of (token1, token2) and the values are sets of indices
    that represent the pairs.
    """
    pairs: dict[tuple[int, int], set[int]] = {}
    for i in range(len(tokens) - 1):
        if tokens[i].can_merge:
            pair = (tokens[i].token, tokens[i + 1].token)
            if pair not in pairs:
                pairs[pair] = set()
            pairs[pair].add(i)
    return pairs

# def build_pairs(root: TokenNode) -> dict[tuple[int, int], set[TokenNode]]:
#     """
#     Build a frequency table of pairs of tokens in the linked list.
#     The keys are tuples of (token1, token2) and the values are sets of TokenNodes
#     that represent the pairs.
#     """
#     pairs: dict[tuple(int, int), set[TokenNode]] = {}
#     node = root
#     while node.next is not None:
#         if node.can_merge:
#             pair = (node.token, node.next.token)
#             if pair not in pairs:
#                 pairs[pair] = set()
#             pairs[pair].add(node)
#         node = node.next
#     return pairs

def list_length(root: TokenNode) -> int:
    length = 0
    node = root
    while node is not None:
        length += 1
        node = node.next
    return length

def get_top_pair(freq: dict[tuple[int, int], set[int]],
                 vocab: dict[int, bytes]):
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

# def merge(tokenList: TokenList, vocab: Dict[int, bytes],
def merge(nodes: list[TokenNode],
          vocab: Dict[int, bytes],
          max_vocab_size: int
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    pairs = build_pairs(nodes)
    num_tokens = len(nodes)

    next = np.arange(len(nodes)) + 1
    next[-1] = -1
    prev = np.arange(len(nodes)) - 1

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
        for nid in todo_nodes.copy():
            # if node in todo_nodes and node.next in todo_nodes:
            if nid in todo_nodes and next[nid] in todo_nodes:
                todo_nodes.remove(next[nid])

        for nid in todo_nodes:

            # Remove node.next -> node.next.next from pairs
            node = nodes[nid]
            next_node = nodes[next[nid]]
            # if node.next.can_merge and node.next.next is not None:
            if next_node.can_merge and next[next[nid]] >= 0:
                # pair = (node.next.token, node.next.next.token)
                pair = (next_node.token, nodes[next[next[nid]]].token)
                # if node_id.next in pairs[pair]:
                if next[nid] in pairs[pair]:
                    pairs[pair].remove(next[nid])
                if len(pairs[pair]) == 0:
                    pairs.pop(pair)

            # Remove node.prev -> node from pairs
            # if node_id.prev is not None and node_id.prev.can_merge:
            if prev[nid] >= 0 and nodes[prev[nid]].can_merge:
                # pair = (node_id.prev.token, node_id.token)
                pair = (nodes[prev[nid]].token, node.token)
                # if node_id.prev in pairs[pair]:
                if prev[nid] in pairs[pair]:
                    pairs[pair].remove(prev[nid])
                if len(pairs[pair]) == 0:
                    pairs.pop(pair)

            # Merge = update current node token + remove next node
            # Update node token
            nodes[nid].token = new_token
            num_tokens -= 1
            # remove next node
            # node.can_merge = node.next.can_merge
            node.can_merge = next_node.can_merge
            # if node_id.next.next is not None:
            if next[next[nid]] >= 0:
                prev[next[next[nid]]] = nid
                # node_id.next.next.prev = node_id
            # node_id.next = node_id.next.next
            next[nid] = next[next[nid]]

            # add pair node.prev -> node
            # if node_id.prev is not None and node_id.prev.can_merge:
            if prev[nid] >= 0 and nodes[prev[nid]].can_merge:
                # pair = (node_id.prev.token, node.token)
                pair = (nodes[prev[nid]].token, node.token)
                if pair not in pairs:
                    pairs[pair] = set()
                # pairs[pair].add(node_id.prev)
                pairs[pair].add(prev[nid])
            # add pair node -> node.next
            if next[nid] >= 0 and node.can_merge:
                pair = (node.token, nodes[next[nid]].token)
                if pair not in pairs:
                    pairs[pair] = set()
                pairs[pair].add(nid)

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
                       worker_id: int = 0) -> list[TokenNode]:
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        with tqdm(desc=f"Worker {worker_id}", position=worker_id) as pbar:
            return pretokenize(chunk, special_tokens, pbar)

def pretokenize_parallel(filename : str, special_tokens: list[str], num_proc: int
) -> list[TokenNode]:

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
    tokens : list[TokenNode] = list(itertools.chain.from_iterable(results))
    return tokens

    # Convert into a linked list
    # return TokenList.deserialize(tokens)


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
        tokens = pretokenize_parallel(file_name, special_tokens, num_proc)
    else:
        with open(file_name, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
            print("Pre-tokenize")
            with tqdm() as pbar:
                tokens = pretokenize(text, special_tokens, pbar)

    vocab, merges = merge(tokens, vocab, vocab_size)
    return vocab, merges

if __name__ == "__main__":
    vocab_size: int = 500

    # file_name: str = "data/TinyStoriesV2-GPT4-valid.txt"
    # file_name: str = "data/mini_test.txt"
    file_name: str = "tests/fixtures/corpus.en"

    # num_proc = multiprocessing.cpu_count() - 2
    num_proc = 2
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
