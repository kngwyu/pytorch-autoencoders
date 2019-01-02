"""Beta-VAE(https://openreview.net/forum?id=Sy2fzU9gl)
Other than its loss fuction, it's same as VAE.
"""
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Callable
from .vae import VaeOutPut


def bernoulli_recons(a: Tensor, b: Tensor) -> Tensor:
    return F.binary_cross_entropy(a, b, reduction='sum')


def gaussian_recons(a: Tensor, b: Tensor) -> Tensor:
    return F.mse_loss(a, b, reduction='sum')


def get_loss_fn(
        beta: float = 1.0,
        decoder_type: str = 'bernoulli',
) -> Callable[[VaeOutPut, Tensor], Tensor]:
    if decoder_type == 'bernoulli':
        recons_loss = bernoulli_recons
    elif decoder_type == 'gaussian':
        recons_loss = gaussian_recons
    else:
        raise ValueError('Currently only bernoulli and gaussian are supported as decoder head')

    def loss_function(res: VaeOutPut, img: Tensor) -> Tensor:
        recons = recons_loss(res.x, img) / float(img.size(0))
        kld = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
        return recons + beta * kld
    return loss_function
