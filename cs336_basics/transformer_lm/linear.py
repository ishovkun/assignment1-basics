import torch
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self,
        in_features: int, # final dimension of the input
        out_features: int, # final dimension of the output
        device: torch.device | None = None, # Device to store the parameters on ]
        dtype: torch.dtype | None = None, # Data type of the parameters
        # weights: torch.Tensor | None = None # Optional pre-initialized weights
    ):

        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.initWeights_()
        # if weights is None:
        #     self.initWeights_()
        # else:
        #     self.weights = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        ## Einsum is self-documenting and robust
        # b - batch
        # s - sequence
        # W = [d_out, d_in]
        # x = [b, s, d_in]
        O = einsum(self.weights, x, "d_out d_in, b s d_in -> b s d_out")
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
        self.weights = torch.nn.Parameter(weights)
