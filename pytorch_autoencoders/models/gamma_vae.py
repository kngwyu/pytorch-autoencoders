"""Loss function used in Understanding disentangling in β-VAE(https://arxiv.org/abs/1804.03599)
This is not called gamma VAE in the original paper.
But, to avoid confusion, I call it gamma-VAE here.
"""
import torch
from torch import Tensor
from typing import Callable
from .beta_vae import _recons_fn
from .vae import VaeOutPut


class LossFunction:
    def __init__(
            self,
            gamma: float = 1000.0,
            capacity_max: float = 20.0,
            num_epochs: int = 10000,
            decoder_type: str = 'bernoulli',
    ) -> Callable[[VaeOutPut, Tensor], Tensor]:
        self.gamma = gamma
        self.recons_loss = _recons_fn(decoder_type)
        self.capacity = 0.0
        self.delta = capacity_max / float(num_epochs)
        self.capacity_max = capacity_max

    def update(self) -> None:
        self.capacity = min(self.capacity_max, self.capacity + self.delta)
        print('capacity: ', self.capacity)

    def __call__(self, res: VaeOutPut, img: Tensor) -> Tensor:
        batch_size = float(img.size(0))
        recons = self.recons_loss(res.x, img).div_(batch_size)
        kld = -0.5 * \
            torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp()).div_(batch_size)
        latent = self.gamma * (kld - self.capacity).abs()
        return recons + latent
