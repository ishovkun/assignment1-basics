import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops.einops import einsum
from cs336_basics.transformer_lm.linear import Linear
import pytest

class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.silu = SiLU()
        self.linear1 = Linear(d_model, d_ff,  device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Float[Tensor, " ... d_model"]:
        # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x), (7)
        l1 = self.linear1
        l2 = self.linear2
        l3 = self.linear3
        silu = self.silu
        return l2(silu(l1(x)) * l3(x))
