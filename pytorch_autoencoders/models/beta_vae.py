"""Beta-VAE(https://openreview.net/forum?id=Sy2fzU9gl)
Other than its loss fuction, it's same as VAE.
"""
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Callable
from .vae import VaeOutPut


def bernoulli_recons(a: Tensor, b: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(a, b, reduction='sum')


def gaussian_recons(a: Tensor, b: Tensor) -> Tensor:
    return F.mse_loss(torch.sigmoid(a), b, reduction='sum')


def _recons_fn(decoder_type: str = 'bernoulli') -> Callable[[Tensor, Tensor], Tensor]:
    if decoder_type == 'bernoulli':
        recons_loss = bernoulli_recons
    elif decoder_type == 'gaussian':
        recons_loss = gaussian_recons
    else:
        raise ValueError('Currently only bernoulli and gaussian are supported as decoder head')
    return recons_loss


class LossFunction:
    def __init__(self, beta: float = 4.0, decoder_type: str = 'bernoulli') -> None:
        self.recons_loss = _recons_fn(decoder_type)
        self.beta = beta

    def __call__(self, res: VaeOutPut, img: Tensor) -> Tensor:
        batch_size = float(img.size(0))
        recons = self.recons_loss(res.x, img).div_(batch_size)
        kld = -0.5 * \
            torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp()).div_(batch_size)
        return recons + self.beta * kld
