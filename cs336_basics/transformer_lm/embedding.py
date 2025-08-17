import torch

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

if __name__ == "__main__":
    tokens = torch.tensor( [[0, 1], [2, 0]] )
    mat = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    print(mat[tokens])
