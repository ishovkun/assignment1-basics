import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange
from cs336_basics.transformer_lm.model import *


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

        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self,
        x: Float[Tensor, "... seq d_model"],
        token_positions: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        """
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
                running the Transformer block on the input features while using RoPE.
        """
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))
        return z

class TransformerLM(torch.nn.Module):
    def __init__(
          self,
          vocab_size: int,
          d_model: int,
          num_heads: int,
          d_ff: int,
          num_layers: int,
          rope_theta: float | None = None,
          max_seq_len: int | None = None,
          device: torch.device | None = None,
          dtype: torch.dtype | None = None
    ):
        super().__init__()

        # self.token_emb = torch.nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        # self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        # self.output_layer = torch.nn.Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self,
        input_ids: Int[Tensor, "batch seq_len"],
        token_positions: Int[Tensor, "batch seq_len"] | None = None
    ) -> Float[Tensor, "batch sequence_length vocab_size"]:
        """
        Float[Tensor, "batch sequence_length vocab_size"] Tensor with the output of
                running the Transformer LM on the input token IDs while using RoPE.
        """
        x = self.token_embeddings(input_ids)  # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, token_positions)  # (batch, seq_len, d_model)
        x = self.ln_final(x)  # (batch, seq_len, d_model)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits

# class TransformerBlock(torch.nn.Module):
#     def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, rope_object: ROPE | None = None):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_ff = d_ff

#         self.ln1 = RMSNorm(d_model=d_model)
#         self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, rope_theta=rope_theta, rope_object=rope_object)
#         self.ln2 = RMSNorm(d_model=d_model)
#         self.ffn = SwiGLUFeedForward(d_model=d_model, d_ff=d_ff)

#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
#         if token_positions is None:
#             token_positions = torch.arange(0, x.shape[-2])
#         residual = x
#         x = self.attn(self.ln1(x), token_positions) + residual
#         residual = x
#         output = self.ffn(self.ln2(x)) + residual
#         return output

# class TransformerLM(torch.nn.Module):
#     def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_ff = d_ff
#         self.theta = rope_theta
#         self.vocab_size = vocab_size
#         self.context_length = context_length
#         d_k = d_model // num_heads

#         self.rope = ROPE(theta=self.theta, d_k=d_k, max_seq_len=context_length)
#         self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
#         self.layers = nn.ModuleList([TransformerBlock(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, rope_theta=rope_theta, rope_object=self.rope)
#                                     for _ in range(num_layers)])
#         self.ln_final = RMSNorm(d_model=d_model)
#         self.lm_head = Linear(in_features=self.d_model, out_features=self.vocab_size)

#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
#         if token_positions is None:
#             token_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)

#         #x = x[..., -self.context_length::]
#         x = self.token_embeddings(x)

#         for layer in self.layers:
#             x = layer(x, token_positions)

#         x = self.ln_final(x)
#         logits = self.lm_head(x)
        return logits
