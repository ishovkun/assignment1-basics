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

def pretokenize(chunk: str, special_tokens: list[str], pbar = None) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat_special = b'|'.join(re.escape(token.encode("utf-8")) for token in special_tokens)

    subchunks = re.split(pat_special, chunk.encode("utf-8"))
    if isinstance(pbar, WorkerProgress):
        pbar.setTotal(len(subchunks))
    elif isinstance(pbar, tqdm.tqdm):
        pbar.total = len(subchunks)
        pbar.refresh()

    pretokens : list[str] = []
    for subchunk in subchunks:
        for match in re.finditer(PAT, subchunk.decode('utf-8', errors='ignore')):
            pre_token = match.group()
            pretokens.append(pre_token)
        if pbar is not None:
            pbar.update(1)

    return pretokens
