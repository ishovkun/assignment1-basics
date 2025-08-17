import torch
from einops import einsum, rearrange
import pytest

class Softmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        largest = torch.max(x, dim=self.dim, keepdim=True).values
        sum = torch.sum(torch.exp(x - largest), dim=self.dim, keepdim=True)
        return torch.exp(x - largest) / sum
        return torch.softmax(x, dim=self.dim)
