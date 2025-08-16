from cs336_basics.parallel_progress_bar import WorkerProgress
import tqdm
import regex as re

def pretokenize_reduce(chunk: str, special_tokens: list[str], pbar = None) -> dict[str, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat_special = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)

    pretokens : dict[str, int] = {}
    subchunks = re.split(pat_special, chunk.encode("utf-8"))
    if isinstance(pbar, WorkerProgress):
        pbar.setTotal(len(subchunks))
    elif isinstance(pbar, tqdm.tqdm):
        pbar.total = len(subchunks)
        pbar.refresh()

    for subchunk in subchunks:
        for match in re.finditer(PAT, subchunk.decode('utf-8', errors='ignore')):
            pre_token = match.group()
            pretokens[pre_token] = pretokens.get(pre_token, 0) + 1
        if pbar is not None:
            pbar.update(1)

    return pretokens

def pretokenize(chunk: str, special_tokens: list[str]) -> list[str]:
    # Main pattern for tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Special tokens pattern with capturing groups
    # pat_special = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)
    # special_matches = list(re.finditer(pat_special.decode("utf-8"), chunk))

    # Sort special tokens by length (longest first) to handle overlaps
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    # Special tokens pattern with lookahead
    pat_special = "|".join(f"(?={re.escape(token)}){re.escape(token)}" for token in special_tokens)
    special_matches = list(re.finditer(pat_special, chunk))

    pretokens: list[str] = []
    last_end = 0

    for match in special_matches:
        # Add the text before the special token to be processed by the main pattern
        if match.start() > last_end:
            non_special_chunk = chunk[last_end:match.start()]
            for token_match in re.finditer(PAT, non_special_chunk):
                pretokens.append(token_match.group())

        # Add the special token itself
        pretokens.append(match.group())
        last_end = match.end()

    # Process any remaining text after the last special token
    if last_end < len(chunk):
        remaining_chunk = chunk[last_end:]
        for token_match in re.finditer(PAT, remaining_chunk):
            pretokens.append(token_match.group())

    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # # pat_special = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)
    # pat_special = b'(' + b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens) + b')'

    # pretokens : list[str] = []
    # subchunks = re.split(pat_special, chunk.encode("utf-8"))

    # for subchunk in subchunks:
    #     for match in re.finditer(PAT, subchunk.decode('utf-8', errors='ignore')):
    #         pre_token = match.group()
    #         pretokens.append(pre_token)

    return pretokens
