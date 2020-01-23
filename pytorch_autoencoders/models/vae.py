import numpy as np
import torch
from torch import nn, Size, Tensor
from torch.nn import functional as F
from typing import List, NamedTuple, Tuple

from ..base import AutoEncoderBase
from ..config import Config


class VaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor


class VariationalAutoEncoder(AutoEncoderBase):
    """Auto-Encoding Variational Bayes, Kingma and Welling, ICLR2014.
    https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self, input_dim: Size, config: Config, hidden: List[int] = [400, 20],
    ) -> None:
        super().__init__()
        input_dim_flat = np.prod(input_dim)
        assert len(hidden) == 2
        self.encoder = nn.Linear(input_dim_flat, hidden[0])
        self.mu = nn.Linear(hidden[0], hidden[1])
        self.logvar = nn.Linear(hidden[0], hidden[1])
        self.decoder = nn.Sequential(
            nn.Linear(hidden[1], hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[0], input_dim_flat),
        )
        self.to(config.device)
        config.initializer(self)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = F.relu(self.encoder(x.view(x.size(0), -1)))
        return self.mu(h1), self.logvar(h1)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def to_image(self, op: VaeOutPut) -> Tensor:
        return op.x

    def forward(self, x: Tensor) -> VaeOutPut:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return VaeOutPut(self.decode(z).view(x.shape), mu, logvar)

    def sample(self, z: Tensor) -> Tensor:
        return self.decoder(z)


def bernoulli_loss(res: VaeOutPut, img: Tensor) -> Tensor:
    recons = F.binary_cross_entropy(torch.sigmoid(res.x), img, reduction="sum")
    kl = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
    return recons + kl


def gaussian_loss(res: VaeOutPut, img: Tensor) -> Tensor:
    recons = F.mse_loss(torch.sigmoid(res.x), img, reduction="sum")
    kl = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
    return recons + kl
