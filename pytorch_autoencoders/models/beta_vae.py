"""Beta-VAE(https://openreview.net/forum?id=Sy2fzU9gl)
Other than its loss fuction, it's same as VAE.
"""
from enum import Enum
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Callable
from .vae import VaeOutPut


class DecoderDist(Enum):
    IDENTITY = 0
    BERNOULLI = 1
    GAUSSIAN = 2


def get_loss_function(
        dist: DecoderDist = DecoderDist.BERNOULLI,
        beta: float = 1.0
) -> Callable[[VaeOutPut, Tensor], Tensor]:
    if dist == DecoderDist.IDENTITY:
        recons_loss = lambda b, a: F.binary_cross_entropy(a, b, reduction='sum')
    elif dist == DecoderDist.BERNOULLI:
        recons_loss = lambda b, a: F.binary_cross_entropy_with_logits(a, b, reduction='sum')
    elif dist == DecoderDist.GAUSSIAN:
        recons_loss = lambda b, a: F.mse_loss(torch.sigmoid(a), b, reduction='sum')
    else:
        raise ValueError('dist have to be chosen from DecoderDist')

    def loss_function(res: VaeOutPut, img: Tensor) -> Tensor:
        bce = recons_loss(img, res.x)
        kld = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
        return bce + beta * kld
    return loss_function
