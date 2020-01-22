import numpy as np
import torch
from torch import nn, Size, Tensor
from torch.nn import functional as F
from typing import List, NamedTuple, Optional, Tuple

from ..base import SslAutoEncoderBase
from ..config import Config


class SslVaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor
    probs: Optional[Tensor]


def _generate_labels(batch_size: int, nlabels: int) -> None:
    res = []
    for i in range(nlabels):
        labels = torch.ones(batch_size, 1, dtype=torch.long) * i
        y = torch.zeros(batch_size, nlabels).scatter_(1, labels, 1)
        res.append(y)
    return torch.cat(res)


class VAESslM2(SslAutoEncoderBase):
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

    def encode(self, x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = F.relu(self.encoder(torch.cat((x, label), dim=1)))
        return self.mu(h1), self.logvar(h1)

    def decode(self, z: Tensor, label: Tensor) -> Tensor:
        return self.decoder(torch.cat((z, label), dim=1))

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

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
        mu, logvar = self.encode(x, label)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z, label).view(old_shape)
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
    res: SslVaeOutPut,
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
    minus_l = recons + prior + kl
    if merginalize:
        # -H(q(y|x)) => Maximize entropy
        minus_h = torch.mean(res.probs * (res.probs + 1e-8).log_(), dim=-1).sum()
        # -q(y|x)L(x, y)
        minus_ql = torch.mean(minus_l * res.probs, dim=-1).sum()
        return minus_h + minus_ql
    else:
        classfication_loss = torch.mean(label * (res.probs + 1e-8).log_(), dim=-1).sum()
        return minus_l - alpha * classfication_loss
