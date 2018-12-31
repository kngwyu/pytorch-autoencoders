"""Beta-VAE(https://openreview.net/forum?id=Sy2fzU9gl)
Other than its loss fuction, it's same as VAE.
"""
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Callable
from .vae import VaeOutPut


def get_loss_function(
        beta: float = 1.0,
        decoder_type: str = 'bernoulli',
) -> Callable[[VaeOutPut, Tensor], Tensor]:
    if decoder_type == 'bernoulli':
        recons_loss = lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction='sum')
    elif decoder_type == 'gaussian':
        recons_loss = lambda a, b: F.mse_loss(torch.sigmoid(a), b, reduction='sum')
    else:
        raise ValueError('Currently only bernoulli and gaussian are supported as decoder head')

    def loss_function(res: VaeOutPut, img: Tensor) -> Tensor:
        bce = recons_loss(img, res.x)
        kld = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
        return bce + beta * kld
    return loss_function
