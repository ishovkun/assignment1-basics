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
from cs336_basics.pretokenize import pretokenize_reduce
from cs336_basics.merge import count_pairs, add_pairs, clear_pairs

def build_initial_vocab() -> dict[int, bytes]:
    """
    Build the initial vocabulary from the byte-level tokens.
    This is a simple implementation that returns a mapping of byte values to their integer IDs.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab

def replace_pairs_with_value(lst, pair, value):
    """
    Replace all occurrences of a specific pair in a list with a given value.

    :param lst: List of integers
    :param pair: Tuple representing the pair to replace (e.g., (a, b))
    :param value: The value to replace the pair with
    :return: A new list with pairs replaced
    """
    i = 0
    result = []
    while i < len(lst):
        # Check if the current and next element form the pair
        if i < len(lst) - 1 and (lst[i], lst[i + 1]) == pair:
            result.append(value)
            i += 2  # Skip the next element since it's part of the pair
        else:
            result.append(lst[i])
            i += 1
    return result

def count_all_occurences(pair, pairs, word_counts):
    if pair not in pairs: return 0
    occs = pairs[pair]
    count = 0
    for word_id, pos in occs:
        count += word_counts[word_id]
    return count

def top_pair(pairs, word_counts, vocab):
    max_freq = 0
    max_items = []
    for pair, occurences in pairs.items():
        total_occurences = sum(word_counts[word_id] for word_id, _ in occurences)
        if total_occurences > max_freq:
            max_freq = total_occurences
            max_items = [pair]
        elif total_occurences == max_freq:
            max_items.append(pair)
    if len(max_items) == 0:
        return None
    else:
        # find lexicographically largest pair
        enc = [ [vocab[x[0]], vocab[x[1]]] for x in max_items ]
        idx = max(range(len(enc)), key=enc.__getitem__)
        return max_items[idx], max_freq

def merge1(pairs, tokens, word_counts, vocab, max_vocab_size):
    merges = []
    total_iters = max_vocab_size - len(vocab)
    pbar = tqdm(total=total_iters, desc="Merge")

    while len(vocab) < max_vocab_size:
        merged, freq = top_pair(pairs, word_counts, vocab)
        if merged is None: break

        new_token = len(vocab)
        bytes1 = vocab[merged[0]]
        bytes2 = vocab[merged[1]]
        vocab[new_token] = bytes1 + bytes2
        merges.append((bytes1, bytes2))


        occurences = pairs[merged]
        changed_words = set(word_id for word_id, _ in occurences)
        for word_id in changed_words:
            old_word = tokens[word_id]

            new_word = replace_pairs_with_value(old_word, merged, new_token)
            clear_pairs(old_word, word_id, pairs)
            add_pairs(new_word, word_id, pairs)

            tokens[word_id] = new_word

        if merged in pairs:
            pairs.pop(merged)

        if pbar is not None: pbar.update(1)
    return vocab, merges

def pretokenize_worker(worker_id: int, filename: str,
                       start: int, end: int, special_tokens: list[str],
                       progress_queue,
                       output_filename) -> dict[str, int]:
    chunk = None
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pbar = WorkerProgress(progress_queue)
    pretokens = pretokenize_reduce(chunk, special_tokens, pbar)
    return pretokens

def get_output_name(filename : str, worker_id: int) -> str:
    strip_extension = lambda filename: os.path.splitext(os.path.basename(filename))[0]
    output_filename = f"{strip_extension(filename)}_{worker_id}.pkl"
    return output_filename

def pretokenize_parallel(filename : str, special_tokens: list[str], num_proc: int):

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
                progress.update()
                finished = sum(r.ready() for r in results)

        # Collect results
        output = [r.get() for r in results]

    pretokens = {}
    for partial_pretokens in output:
        for key, value in partial_pretokens.items():
            pretokens[key] = pretokens.get(key, 0) + value
    return pretokens

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

def tokenize1(file_name: str, vocab_size: int, special_tokens: list[str], num_proc: int
        ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    print(f"Tokenizing file {file_name}")
    vocab = build_initial_vocab()
    eot_str = "<|endoftext|>"
    if eot_str not in special_tokens:
        special_tokens.append(eot_str)

    for special in special_tokens:
        token = len(vocab)
        vocab[token] = special.encode("utf-8")

    if num_proc == 1:
        with open(file_name, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
            with tqdm(desc="Pretokenize") as pbar:
                pretokens = pretokenize_reduce(text, special_tokens, pbar)
    else:
        pretokens = pretokenize_parallel(file_name, special_tokens, num_proc)
    words = list(pretokens.keys())
    tokens = [list(word.encode("utf-8")) for word in words]
    counts = list(pretokens.values())
    pairs = count_pairs(tokens)
    vocab, merges = merge1(pairs, tokens, counts, vocab, vocab_size)
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
    # num_proc = 4

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        vocab_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_proc = int(sys.argv[3])

    print(f"filename = {file_name} vocab_size = {vocab_size} num_proc = {num_proc}")

    eot_str = "<|endoftext|>"
    # vocab, merges = tokenize(file_name, vocab_size, special_tokens=[eot_str], num_proc=num_proc)
    vocab, merges = tokenize1(file_name, vocab_size, special_tokens=[eot_str], num_proc=num_proc)

    strip_extension = lambda filename: os.path.splitext(os.path.basename(filename))[0]
    output_filename = f"{strip_extension(file_name)}_vocab{vocab_size}.pkl"
    # output_filename = strip_extension(file_name) + ".pkl"
    with open(output_filename, "wb") as f:
        print(f"Saving vocabulary and merges to {output_filename}")
        pickle.dump({"vocab": vocab, "merges": merges}, f)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', "profile_results.prof")
    main()
