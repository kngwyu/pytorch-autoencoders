import numpy as np
import torch
from torch import nn, Size, Tensor
from torch.nn import functional as F
from typing import List, NamedTuple, Optional, Tuple

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
        h1 = F.relu(self.encoder(x))
        return self.mu(h1), self.logvar(h1)

    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        x = self.decoder(z)
        if old_shape is None:
            return x
        else:
            return x.view(old_shape)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def to_image(self, op: VaeOutPut) -> Tensor:
        return op.x

    def forward(self, x: Tensor) -> VaeOutPut:
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        return VaeOutPut(self.decode(z, old_shape=x.shape), mu, logvar)

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


def _generate_labels(batch_size: int, nlabels: int) -> None:
    res = []
    for i in range(nlabels):
        labels = torch.ones(batch_size, 1, dtype=torch.long) * i
        y = torch.zeros(batch_size, nlabels).scatter_(1, labels, 1)
        res.append(y)
    return torch.cat(res)


class SslVaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor
    probs: Optional[Tensor]


class VAESslM2(VariationalAutoEncoder):
    """Semi-supervised Learning with Deep Generative Models, Kingma et al., NIPS 2014.
    https://arxiv.org/abs/1406.5298
    """

    def __init__(
        self,
        input_dim: Size,
        config: Config,
        hidden: List[int] = [400, 20],
        nlabels: int = 10,
    ) -> None:
        nn.Module.__init__(self)
        x_dim = np.prod(input_dim)
        assert len(hidden) == 2
        self.nlabels = nlabels
        self.encoder = nn.Linear(x_dim + nlabels, hidden[0])
        self.mu = nn.Linear(hidden[0], hidden[1])
        self.logvar = nn.Linear(hidden[0], hidden[1])
        self.decoder = nn.Sequential(
            nn.Linear(hidden[1] + nlabels, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[0], x_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(x_dim, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[0], nlabels),
            nn.Softmax(dim=-1),
        )
        self.to(config.device)
        config.initializer(self)

    def to_image(self, op: VaeOutPut) -> Tensor:
        return op.x

    def sample(self, z: Tensor, label: Tensor) -> Tensor:
        return self.decoder(torch.cat((z, label), dim=1))

    def classify(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def forward(
        self, x: Tensor, label: Optional[Tensor] = None
    ) -> Tuple[SslVaeOutPut, Tensor, Tensor]:
        old_shape = x.shape
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        if label is None:
            label = _generate_labels(batch_size, self.nlabels).to(x.device)
            x = x.repeat(self.nlabels, 1)
            old_shape = batch_size * self.nlabels, *old_shape[1:]
        mu, logvar = self.encode(torch.cat((x, label), dim=1))
        z = self.reparameterize(mu, logvar)
        res = self.decode(torch.cat((z, label), dim=1), old_shape=old_shape)
        probs = self.classifier(x)
        return SslVaeOutPut(res, mu, logvar, probs), x.view(old_shape), label


def _log_standard_categorical(p: Tensor) -> Tensor:
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    """
    prior = F.softmax(torch.ones_like(p), dim=1).detach_()
    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    return cross_entropy


def labeled_bernoulli_loss(
    res: VaeOutPut,
    img: Tensor,
    label: Tensor,
    alpha: float = 0.1,
    merginalize: bool = False,
) -> Tensor:
    # P(x|y,z)
    recons = F.binary_cross_entropy(torch.sigmoid(res.x), img, reduction="sum")
    prior = _log_standard_categorical(label).sum()
    kl = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
    # -L(x, y)
    minus_elbo = recons + prior + kl
    if merginalize:
        # -H(q(y|x))
        minus_h = torch.mean(res.probs * (res.probs + 1e-8).log_())
        # -q(y|x)L(x, y)
        minus_ql = torch.mean(minus_elbo * res.probs)
        return minus_h + minus_ql
    else:
        classfication_loss = torch.sum(label * (res.probs + 1e-8).log_(), dim=-1).mean()
        return minus_elbo - alpha * classfication_loss
