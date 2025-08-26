import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange
# from torch._C import K
from cs336_basics.transformer_lm.softmax import Softmax
from cs336_basics.transformer_lm.linear import Linear
from cs336_basics.transformer_lm.rope import RotaryPositionalEmbedding
import math
import pytest

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
        ) -> Float[Tensor, " ... sequence_length d_out"]:

        scale = (K.shape[-1] ** 0.5)**(-1)
        qk = scale * einsum(Q, K, "... s1 d, ... s2 d -> ... s1 s2")

        if mask is not None:
            qk.masked_fill_(mask == False, float("-inf")) # in-place

        softmax = Softmax(-1)
        S = softmax(qk)
        ret = einsum(S, V, "... s1 s2, ... s2 d_v -> ... s1 d_v")
        return ret

class MultiHeadAttention(torch.nn.Module):
    """
    Implements causal multi-head self-attention.
    """
    def __init__(self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        rope_theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Optimization
        # Instead of having three separate matmuls
        # self.proj_q = Linear(d_model, d_model, device=device, dtype=dtype)
        # self.proj_k = Linear(d_model, d_model, device=device, dtype=dtype)
        # self.proj_v = Linear(d_model, d_model, device=device, dtype=dtype)
        # we have a single matmul
        self.proj_qkv = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        # output projection
        self.proj_o = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(theta = rope_theta,
                                                  d_k = self.head_dim,
                                                  max_seq_len=max_seq_len,
                                                  device=device)

    def forward(self,
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        q, k, v = self.proj_qkv(x).split(self.d_model, dim=-1)
        # q = self.proj_q(x)
        # k = self.proj_k(x)
        # v = self.proj_v(x)

        q = rearrange(q, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)
        k = rearrange(k, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)
        v = rearrange(v, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)

        # Build a lower triangular mask
        # The mask ensures that each position in the sequence can only
        # attend to itself and the positions **before it**,
        #  but not to any positions **after it**.
        # This is crucial in tasks where the model should not
        # "peek" at future tokens when making predictions.
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
        mask = mask.to(x.device)

        # Apply RoPE only to q and k
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            # v = self.rope(v, token_positions)

        attn = ScaledDotProductAttention()
        o = attn(q, k, v, mask)
        o = rearrange(o, '... h seq_len d_v -> ... seq_len (h d_v)')
        o = self.proj_o(o)

        return o
