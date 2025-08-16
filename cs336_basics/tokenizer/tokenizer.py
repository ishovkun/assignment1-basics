from collections.abc import Iterable, Iterator

from numpy._core.numerictypes import byte
from cs336_basics.tokenizer.pretokenize import pretokenize
from cs336_basics.tokenizer.merge import *
import pytest


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        self.merges : list[tuple[int, int]] = []
        for merge in merges:
            ints = (self.vocab_inv[merge[0]], self.vocab_inv[merge[1]])
            self.merges.append(ints)

        self.special_tokens = special_tokens if special_tokens is not None else ["<|endoftext|>"]

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        words = pretokenize(text, self.special_tokens)
        # print(f"\nwords: {words}")
        tokens = self.tokenizeWords_(words)
        # print(f"eot = {self.vocab_inv['<|endoftext|>'.encode('utf-8')]}")
        # print(f"init_tokens = {tokens}")

        pairs = count_pairs(tokens)

        for merged in self.merges:
            if merged not in pairs: continue
            occurences = pairs[merged]
            changed_words = set(word_id for word_id, _ in occurences)
            new_token = self.vocab_inv[self.vocab[merged[0]] + self.vocab[merged[1]]]
            for word_id in changed_words:
                old_word = tokens[word_id]
                new_word = replace_pairs_with_value(old_word, merged, new_token)
                clear_pairs(old_word, word_id, pairs)
                add_pairs(new_word, word_id, pairs)
                tokens[word_id] = new_word
            if merged in pairs:
                pairs.pop(merged)

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
        carry: str = ""
        for chunk in iterable:
            combined_text = carry + chunk
            tokens = self.encode(combined_text)
            if tokens:
                for token in tokens[:-1]:
                    yield token

                last_token = tokens[-1]
                carry = self.decode([last_token])
            else:
                carry = ""

        if carry:
            for token in self.encode(carry):
                yield token
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        # print(f"ids = {ids}")
        byte_data = b"".join(self.vocab[token] for token in ids)
        # print(f"byte_data: {byte_data}")
        ret = byte_data.decode("utf-8", errors="ignore")
        # print(f"ret = {ret}")
        return ret

    def tokenizeWords_(self, words):
        tokens = []
        for word in words:
            if word in self.special_tokens:
                # Special tokens are already in the vocabulary
                tokens.append([self.vocab_inv[word.encode("utf-8")]])
                continue
            utf8 = word.encode("utf-8")
            word_tokens = []
            for b in utf8:
                byte_as_bytes = bytes([b])
                token = self.vocab_inv[byte_as_bytes]
                word_tokens.append(token)
            tokens.append(word_tokens)
        return tokens


if __name__ == "__main__":
    pass
