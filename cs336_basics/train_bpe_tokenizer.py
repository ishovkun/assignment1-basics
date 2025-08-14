from io import BufferedReader
import regex as re
from collections import Counter
from typing import Dict, Tuple, List
import multiprocessing
import itertools
from tqdm import tqdm
import time
import sys, os
import pickle
import numpy as np
import random
from cs336_basics.find_chunk_boundaries import find_chunk_boundaries
from cs336_basics.parallel_progress_bar import WorkerProgress, MasterProgress

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

def pretokenize(chunk: str, special_tokens: list[str], pbar = None
) -> list[TokenNode]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat_special = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)

    if isinstance(pbar, WorkerProgress):
        pbar.setTotal(len(chunk))
    elif isinstance(pbar, tqdm):
        pbar.total = len(chunk)
        pbar.refresh()

    tokens : list[TokenNode] = []
    pos = 0
    while pos < len(chunk):
        match = re.search(pat_special.decode("utf-8"), chunk[pos:])
        span = (len(chunk) - pos, len(chunk) - pos)
        if match is not None:
            span = match.span()
        subchunk = chunk[pos:pos+span[0]]
        for match in re.finditer(PAT, subchunk):
            pre_token = match.group()
            ints = list(pre_token.encode("utf-8"))
            for token in ints:
                tokens.append(TokenNode(token))
            tokens[-1].can_merge = False

        pos += span[1]

        if pbar is not None:
            pbar.update(span[1])

    return tokens

    # ans = re.search(pat_special, chunk.encode("utf-8"))
    # print(f"ans = {ans}")

    subchunks = re.split(pat_special, chunk.encode("utf-8"))
    if isinstance(pbar, WorkerProgress):
        pbar.setTotal(len(subchunks))
    elif isinstance(pbar, tqdm):
        pbar.total = len(subchunks)
        pbar.refresh()

    tokens : list[TokenNode] = []
    for subchunk in subchunks:
        for match in re.finditer(PAT, subchunk.decode('utf-8', errors='ignore')):
            pre_token = match.group()
            ints = list(pre_token.encode("utf-8"))
            for token in ints:
                tokens.append(TokenNode(token))
            tokens[-1].can_merge = False
            pos = match.start()
        if pbar is not None:
            pbar.update(1)

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

def merge(nodes: list[TokenNode],
          vocab: Dict[int, bytes],
          max_vocab_size: int
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    print("Building pairs")
    pairs = build_pairs(nodes)
    print("Done")
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
            if nid in todo_nodes and next[nid] in todo_nodes:
                todo_nodes.remove(next[nid])

        for nid in todo_nodes:

            # Remove node.next -> node.next.next from pairs
            node = nodes[nid]
            next_node = nodes[next[nid]]
            if next_node.can_merge and next[next[nid]] >= 0:
                pair = (next_node.token, nodes[next[next[nid]]].token)
                if next[nid] in pairs[pair]:
                    pairs[pair].remove(next[nid])
                if len(pairs[pair]) == 0:
                    pairs.pop(pair)

            # Remove node.prev -> node from pairs
            if prev[nid] >= 0 and nodes[prev[nid]].can_merge:
                pair = (nodes[prev[nid]].token, node.token)
                if prev[nid] in pairs[pair]:
                    pairs[pair].remove(prev[nid])
                if len(pairs[pair]) == 0:
                    pairs.pop(pair)

            # Merge = update current node token + remove next node
            # Update node token
            nodes[nid].token = new_token
            num_tokens -= 1
            # remove next node
            node.can_merge = next_node.can_merge
            if next[next[nid]] >= 0:
                prev[next[next[nid]]] = nid
            next[nid] = next[next[nid]]

            # add pair node.prev -> node
            if prev[nid] >= 0 and nodes[prev[nid]].can_merge:
                pair = (nodes[prev[nid]].token, node.token)
                if pair not in pairs:
                    pairs[pair] = set()
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

def pretokenize_worker(worker_id: int, filename: str,
                       start: int, end: int, special_tokens: list[str],
                       progress_queue,
                       output_filename) -> str:
    chunk = None
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pbar = WorkerProgress(progress_queue)
    tokens = pretokenize(chunk, special_tokens, pbar)
    return tokens

    # with open(output_filename, "wb") as f:
    #     pickle.dump({"tokens": tokens}, f)
    # print(f"Worker {worker_id} done")
    # return f"Worker {worker_id} done"

def get_output_name(filename : str, worker_id: int) -> str:
    strip_extension = lambda filename: os.path.splitext(os.path.basename(filename))[0]
    output_filename = f"{strip_extension(filename)}_{worker_id}.pkl"
    return output_filename

def pretokenize_parallel(filename : str, special_tokens: list[str], num_proc: int
) -> list[TokenNode]:

    num_workers = num_proc - 1

    # serial: find boundaries
    boundaries: list[int] = []
    with open(filename, "rb") as f:
        eot_str = "<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_workers, eot_str.encode("utf-8"))

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    num_tasks = min(num_workers, len(boundaries) - 1)

    tasks = [
        (task_id, filename, start, end, special_tokens, progress_queue, get_output_name(filename, task_id))
            for task_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]

    # with multiprocessing.Pool(num_workers) as pool:
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        print("Running pretokenization in parallel with", num_workers, "processes")
        results = [
            pool.apply_async(pretokenize_worker, args=task)
            for task in tasks
        ]

        with tqdm() as pbar:
            finished = 0
            progress = MasterProgress(progress_queue, pbar, num_tasks)
            while finished < num_tasks:
                # print(f"update: finished = {finished} out of {num_tasks}")
                progress.update()
                finished = sum(r.ready() for r in results)
                # time.sleep(1.)

        # Collect results
        output = [r.get() for r in results]

    # tokens = []
    # for task in tasks:
    #     fname = task[6]
    #     with open(fname, "rb") as f:
    #        data = pickle.load(f)
    #        tokens += data["tokens"]

    # turn [[token1, token2], [token3, token4], ...] into flat list
    print("Combining results")
    tokens : list[TokenNode] = list(itertools.chain.from_iterable(output))
    print("Done combining results")
    return tokens


def tokenize(file_name: str, vocab_size: int, special_tokens: list[str], num_proc: int
        ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    print(f"Tokenizing file {file_name}")
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

def main():
    vocab_size: int = 500

    # file_name: str = "data/TinyStoriesV2-GPT4-valid.txt"
    # file_name: str = "data/TinyStoriesV2-GPT4-train.txt"
    # file_name: str = "data/TinyStories-debug.txt"
    # file_name: str = "data/mini_test.txt"
    file_name: str = "tests/fixtures/corpus.en"

    num_proc = multiprocessing.cpu_count() - 2
    # num_proc = 1
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        vocab_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_proc = int(sys.argv[3])

    print(f"filename = {file_name} vocab_size = {vocab_size} num_proc = {num_proc}")

    eot_str = "<|endoftext|>"
    vocab, merges = tokenize(file_name, vocab_size, special_tokens=[eot_str], num_proc=num_proc)

    strip_extension = lambda filename: os.path.splitext(os.path.basename(filename))[0]
    output_filename = "{strip_extension(file_name)}_vocab{vocab_size}.pkl"
    # output_filename = strip_extension(file_name) + ".pkl"
    with open(output_filename, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', "profile_results.prof")
    main()
