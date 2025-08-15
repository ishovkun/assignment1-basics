class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        pass
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass
    def encode(self, text: str) -> list[int]:
        pass
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
        return "".join(self.vocab[token] for token in ids)
