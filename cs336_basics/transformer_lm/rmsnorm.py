from einops.einops import einsum
import torch
import pytest
# from einops import rearrange, einsum

class RMSNorm(torch.nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        """
        Construct the RMSNorm module.
        This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        d_model = self.weights.shape[0]
        # RMS(a) = sqrt( (1 / dmodel) ∑(i:1,dmodel) i=1 (a_i)^2 + ε)
        rms = (einsum(x * x, "b s d_model -> b s") + self.eps)**0.5 / d_model**0.5
        tmp = einsum(x, 1./rms, "b s d_model, b s -> b s d_model")
        rms_norm = einsum(tmp, self.weights, "b s d_model, d_model -> b s d_model")
        return rms_norm.to(in_dtype)
