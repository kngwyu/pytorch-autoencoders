"""Beta-VAE(https://openreview.net/forum?id=Sy2fzU9gl)
Other than its loss fuction, it's same as VAE.
"""
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Callable
from .vae import VaeOutPut


def get_loss_function(beta: float = 1.0) -> Callable[[VaeOutPut, Tensor], Tensor]:
    def loss_function(res: VaeOutPut, img: Tensor) -> Tensor:
        bce = F.binary_cross_entropy(res.x, img, reduction='sum')
        kld = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
        return bce + beta * kld
    return loss_function
