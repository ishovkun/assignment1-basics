import torch
from einops import einsum, rearrange
import pytest

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        # Create frequency values for each dimension pair
        i = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        k = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        freq_base = theta ** (-2 * k / d_k)
        Theta = einsum(i, freq_base, "i, k -> i k")

        self.register_buffer("freq_sin", torch.sin(Theta), persistent=False) # (max_seq_len, dk // 2)
        self.register_buffer("freq_cos", torch.cos(Theta), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_query_or_key (Float[Tensor, "... sequence_length d_k"]):
                 Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]):
                Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of
        the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions.
        You should assume that the token positions are a tensor of shape (..., seq_len)
        specifying the token positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin
        tensors along the sequence dimension.
        To test your implementation, complete [adapters.run_rope]
        and make sure it passes uv run pytest -k test_rope.
        """
        # '... (h d) -> ... h d' means:
        # - Keep all leading dimensions (...)
        # - Split the last dimension into two new dimensions, 'h' and 'd'
        # - The size of 'd' is explicitly provided as 2
        x = rearrange(x, '... (h d) -> ... h d', d=2) # (..., seq_len, d_k //2, 2)

        freq_sin = self.freq_sin[token_positions] # (..., seq_len, dk // 2)
        freq_cos = self.freq_cos[token_positions] # (..., seq_len, dk // 2)

        x1 = x[..., 0] * freq_cos - x[..., 1] * freq_sin
        x2 = x[..., 1] * freq_cos + x[..., 0] * freq_sin

        x = torch.stack([x1, x2], dim=-1)

        return rearrange(x, '... h d -> ... (h d)')
