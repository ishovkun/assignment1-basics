import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange
from cs336_basics.transformer_lm.softmax import Softmax

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
        ) -> Float[Tensor, " ... sequence_length d_out"]:
            scale = 1/(K.shape[-1] ** 0.5)
            # Attention(Q, K, V ) = softmax (Q⊤K √dk ) V
            QK = scale * einsum(Q, K, "... i k, ... j k -> ... i j")
            if mask is not None:
                QK[mask == False] = float("-inf")
            S = Softmax(-1).forward(QK)
            Attn = einsum(S, V, "... i j, ... j k -> ... i k")
            return Attn
