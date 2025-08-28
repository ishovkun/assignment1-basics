import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange

class Embedding(torch.nn.Module):
    def __init__(self,
        num_embeddings: int, # size of the vocab
        embedding_dim: int, # dimension of embedding vectors, i.e. d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        emb_mat = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        mu = 0.
        sigma = 1.
        limits= (-3*sigma, 3*sigma)
        emb_mat = torch.nn.init.trunc_normal_(emb_mat,
            mean=mu, std=sigma,
            a=limits[0], b=limits[1])
        self.weight = torch.nn.Parameter(emb_mat, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weight[token_ids]

class Linear(torch.nn.Module):
    def __init__(self,
        in_features: int, # final dimension of the input
        out_features: int, # final dimension of the output
        device: torch.device | None = None, # Device to store the parameters on ]
        dtype: torch.dtype | None = None, # Data type of the parameters
    ):

        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.initWeights_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        ## Einsum is self-documenting and robust
        # b - batch
        # s - sequence
        # W = [d_out, d_in]
        # x = [b, s, d_in]
        O = einsum(self.weight, x, "d_out d_in, b s d_in -> b s d_out")
        return O

    def initWeights_(self):
        # N ( µ = 0, σ2 = 2 din+dout ) truncated at [−3σ, 3σ].
        din = self.in_features
        dout = self.out_features
        mu = 0.
        sigma2 = 2 * (din + dout)
        sigma = sigma2 ** 0.5
        limits = [mu - 3*sigma, mu + 3*sigma]
        # weights = torch.normal(
        weights = torch.empty(dout, din, device=self.device, dtype=self.dtype)
        weights = torch.nn.init.trunc_normal_(weights,
            mean=mu, std=sigma,
            a=limits[0], b=limits[1])
        self.weight = torch.nn.Parameter(weights)

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
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        d_model = self.weight.shape[0]

        # RMS(a) = sqrt( (1 / dmodel) ∑(i:1,dmodel) i=1 (a_i)^2 + ε)
        rms = (einsum(x * x, "b s d_model -> b s") + self.eps).sqrt() / d_model**0.5
        tmp = einsum(x, 1./rms, "b s d_model, b s -> b s d_model")
        rms_norm = einsum(tmp, self.weight, "b s d_model, d_model -> b s d_model")
        return rms_norm.to(in_dtype)

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
        freq_base = theta ** (-2. * k / d_k)
        # freq_base = 1 / theta ** (2 * k / d_k)
        Theta = einsum(i, freq_base, "i, k -> i k")
        self.max_seq_len = max_seq_len

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
        """
        # '... (h d) -> ... h d' means:
        # - Keep all leading dimensions (...)
        # - Split the last dimension into two new dimensions, 'h' and 'd'
        # - The size of 'd' is explicitly provided as 2
        x = rearrange(x, '... (h d) -> ... h d', d=2) # (..., seq_len, d_k //2, 2)

        token_positions = token_positions.clamp(0, self.max_seq_len - 1)
        sin = self.freq_sin[token_positions] # (..., seq_len, dk // 2)
        cos = self.freq_cos[token_positions] # (..., seq_len, dk // 2)

        x1 = x[..., 0] * cos - x[..., 1] * sin
        x2 = x[..., 1] * cos + x[..., 0] * sin

        x = torch.stack([x1, x2], dim=-1)

        return rearrange(x, '... h d -> ... (h d)')

def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, " ..."]:
    return torch.nn.functional.softmax(x, dim=dim)

    largest = torch.max(x, dim, keepdim=True).values
    x = x - largest
    expx = torch.exp(x)
    sum = torch.sum(expx, dim=dim, keepdim=True)
    return expx / sum

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
        self.d_ff = round(self.d_model / 24) * 64 if d_ff is None else d_ff
        self.w1 = Linear(d_model, self.d_ff,  device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Float[Tensor, " ... d_model"]:
        # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x), (7)
        l1 = self.w1
        l2 = self.w2
        l3 = self.w3
        silu = self.silu
        return l2(silu(l1(x)) * l3(x))

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:

    scale = K.shape[-1] ** (-0.5)
    qk = scale * einsum(Q, K, "... s1 d, ... s2 d -> ... s1 s2")

    if mask is not None:
        qk.masked_fill_(~mask, -torch.inf)

    S = softmax(qk, dim=-1)
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
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

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

        attn = scaled_dot_product_attention
        o = attn(q, k, v, mask)
        o = rearrange(o, '... h seq_len d_v -> ... seq_len (h d_v)')
        o = self.output_proj(o)

        return o
