import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange
from cs336_basics.transformer_lm.attention import MultiHeadAttention
from cs336_basics.transformer_lm.rmsnorm import RMSNorm
from cs336_basics.transformer_lm.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None,
        rope_theta: float | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        #
        # self.attn = Attention(d_model, n_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        # self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)
        # self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        # self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        #

        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, rope_theta,
                                       device, dtype)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x, token_positions = None):
        x = x + self.attn(self.norm1(x), token_positions)
        return x + self.ffn(self.norm2(x))
