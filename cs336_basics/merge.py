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
    pbar = tqdm(total=len(words), desc="Count pairs")
    for word_id, word in enumerate(words):
        add_pairs(word, word_id, pairs)
        pbar.update(1)
    return pairs
