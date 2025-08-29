import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool

def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
)  -> Float[Tensor, ""]:
    """
    l = - log( softmax( o_i[x_i+1] ) )
    p = softmax( o_i[x_i+1] ) = exp(oi[xi+1]) / sum( exp( oi[a] ) )
    log and exp cancel out in the numerator. Nice :-)
    l = - log(exp(oi[xi+1])) + log(sum(exp(oi[a])))
      = - oi[xi+1] + log(sum(exp(...)))
    Therefore, we won't use sofmax function but rather reimplement a part of it :-)
    """
    inputs = inputs - inputs.max(dim=-1, keepdim=True)[0]
    output = inputs[torch.arange(inputs.shape[0]), targets] - torch.log(torch.sum(torch.exp(inputs), dim=1))
    return -output.mean()
