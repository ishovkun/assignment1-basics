from typing import IO, Any, BinaryIO
import torch
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if isinstance(out, (str, os.PathLike)):
        with open(out, "wb") as file:
            torch.save(state, file)
    elif isinstance(out, (BinaryIO, IO)):
        torch.save(state, out)
    else:
        raise ValueError("Unsupported type for 'out'. Must be a path or a binary file-like object.")


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as f:
            state = torch.load(f)
    elif isinstance(src, (BinaryIO, IO)):
        state = torch.load(src)
    model.load_state_dict(state["model"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer"])
    iteration = state["iteration"]
    return iteration
