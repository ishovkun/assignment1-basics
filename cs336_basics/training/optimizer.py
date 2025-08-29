from collections.abc import Callable, Iterable
from typing import Optional, Tuple
import torch
import math
from math import sqrt

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        theta[t+1] = theta[t] - alpha / sqrt(t+1) grad L
        """
        loss = None if closure is None else closure()
        # print(f"self.param_groups = {self.param_groups}")
        # print(f"self.state = {self.state.items()}")

        for group in self.param_groups:

            lr = group['lr']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # increment iter
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self,
        params,
        lr: float, # alpha
        betas: Tuple[float, float],
        weight_decay: float, # lambda
        eps: float
    ):
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] <= 0 or betas[0] >= 1 or betas[1] <= 0 or betas[1] >= 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = {
            "lr" : lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['betas']
            eps = group['eps']
            lam = group['weight_decay']
            for p in group["params"]:
                if p.grad is None: continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                g = p.grad.data
                m = beta[0] * m + (1. - beta[0]) * g
                v = beta[1] * v + (1. - beta[1]) * g**2
                at = lr * sqrt(1. - beta[1]**t) / (1. - beta[0]**t)
                p.data -= at * m / (torch.sqrt(v) + eps)
                p.data -= lr * lam * p.data

                state['m'] = m
                state['v'] = v
                state['t'] = t + 1

        return loss

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    for t in range(2):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(f"loss = {loss.cpu().item()}")
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

"""
1: 19.59
1e1: 2.6
1e2: 4e-23
1e3: 2e18 diverge
"""
