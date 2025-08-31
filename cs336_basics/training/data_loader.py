from jaxtyping import Float, Int
import numpy.typing as npt
import torch
import pytest

class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def get_randomized(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:

        max_start = len(dataset) - context_length - 1
        starts = torch.randint(low=0, high=max_start+1, size=(batch_size,))

        inputs = torch.stack([torch.from_numpy(dataset[i:i+context_length]) for i in starts], dim=0)
        targets = torch.stack([torch.from_numpy(dataset[i+1:i+context_length+1]) for i in starts], dim=0)

        return inputs.to(device), targets.to(device)
