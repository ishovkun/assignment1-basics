from tqdm import tqdm

def add_pairs(word, word_id, pairs):
    for pos in range(len(word) - 1):
        pair = (word[pos], word[pos + 1])
        if pair not in pairs:
            pairs[pair] = set()
        pairs[pair].add((word_id, pos))

def clear_pairs(word, word_id, pairs):
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        if pair in pairs and (word_id, i) in pairs[pair]:
            pairs[pair].remove((word_id, i))
            if len(pairs[pair]) == 0: pairs.pop(pair)

def count_pairs(words: list[list[int]]):
    pairs = {}
    # pbar = tqdm(total=len(words), desc="Count pairs")
    for word_id, word in enumerate(words):
        add_pairs(word, word_id, pairs)
        # pbar.update(1)
    return pairs

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
