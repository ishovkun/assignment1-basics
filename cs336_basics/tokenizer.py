from collections.abc import Iterable, Iterator

from numpy._core.numerictypes import byte
from cs336_basics.pretokenize import pretokenize
from cs336_basics.merge import count_pairs, add_pairs, clear_pairs
import pytest


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else ["<|endoftext|>"]
        self.byte_decoder = {v: k for k, v in self.vocab.items() if len(v) == 1}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        words = pretokenize(text, self.special_tokens)
        tokens = self.tokenizeWords_(words)
        pairs = count_pairs(tokens)


        for merged in self.merges:
            if merged not in pairs: continue
            occurences = pairs[merged]

        # print(tokens)
        tokens = [token for word in tokens for token in word]

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large
        files that we cannot directly load into memory.
        """
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_data = b"".join(self.vocab[token] for token in ids)
        return byte_data.decode("utf-8")

    def tokenizeWords_(self, words):
        tokens = []
        for word in words:
            utf8 = word.encode("utf-8")
            word_tokens = []
            for b in utf8:
                byte_as_bytes = bytes([b])
                token = self.byte_decoder[byte_as_bytes]
                word_tokens.append(token)
            tokens.append(word_tokens)
        return tokens


if __name__ == "__main__":
    pass
