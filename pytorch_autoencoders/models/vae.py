from ..base import AutoEncoderBase
from ..config import Config
import torch
from torch import nn, Size, Tensor
from torch.nn import functional as F
from typing import List, NamedTuple, Tuple


class VaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor


class VariationalAutoEncoder(AutoEncoderBase):
    """Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    """
    def __init__(
            self,
            input_dim: Size,
            config: Config,
            hidden: List[int] = [400, 20]
    ) -> None:
        super().__init__()
        input_dim_flat = input_dim[-1] * input_dim[-2]
        assert len(hidden) == 2
        self.fc1 = nn.Linear(input_dim_flat, hidden[0])
        self.fc2_1 = nn.Linear(hidden[0], hidden[1])
        self.fc2_2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[0])
        self.fc4 = nn.Linear(hidden[0], input_dim_flat)
        self.to(config.device)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = F.relu(self.fc1(x.view(*x.shape[:-2], -1)))
        return self.fc2_1(h1), self.fc2_2(h1)

    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        h3 = F.relu(self.fc3(z))
        x = self.fc4(h3)
        if old_shape is None:
            return x
        else:
            return x.view(old_shape)

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def to_image(self, op: VaeOutPut) -> Tensor:
        return op.x

    def forward(self, x: Tensor) -> VaeOutPut:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return VaeOutPut(self.decode(z, old_shape=x.shape), mu, logvar)


def bernoulli_loss(res: VaeOutPut, img: Tensor) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(res.x, img, reduction='sum')
    kld = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
    return bce + kld


def gaussian_loss(res: VaeOutPut, img: Tensor) -> Tensor:
    bce = F.mse_loss(torch.sigmoid(res.x), img, reduction='sum')
    kld = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
    return bce + kld
