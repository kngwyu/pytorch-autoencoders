import numpy as np
import torch
from torch import nn, Size, Tensor
from torch.nn import functional as F
from typing import Callable, List, NamedTuple, Optional, Tuple
from torch.distributions import Categorical

from ..base import SslAutoEncoderBase
from ..config import Config


class SslVaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor
    dist: Categorical

    @property
    def probs(self) -> Tensor:
        return self.dist.probs

    @property
    def logits(self) -> Tensor:
        return self.dist.logits


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
        return Categorical(logits=self.classifier(x))

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
        dist = Categorical(logits=self.classifier(x))
        return SslVaeOutPut(res, mu, logvar, dist), x.view(old_shape), label


def _recons_loss_fn(recons_type: str) -> Callable[[Tensor, Tensor], Tensor]:
    if recons_type == "bernoulli":

        def _loss(x: Tensor, target: Tensor) -> Tensor:
            return F.binary_cross_entropy(torch.sigmoid(x), target, reduction="sum")

    elif recons_type == "gaussian":

        def _loss(x: Tensor, target: Tensor) -> Tensor:
            return F.mse_loss(torch.sigmoid(x), target, reduction="sum")

    else:
        raise ValueError(
            "Currently only bernoulli and gaussian are supported as decoder head"
        )
    return _loss


class LossFunction:
    EPS = 1e-8

    def __init__(self, recons_type: str = "bernoulli") -> None:
        self.recons_loss = _recons_loss_fn(recons_type)

    def _log_standard_categorical(self, label: Tensor) -> Tensor:
        prior = F.softmax(torch.ones_like(label), dim=1).detach_()
        return -torch.sum(label * torch.log(prior + self.EPS), dim=1)

    def __call__(
        self,
        res: SslVaeOutPut,
        target: Tensor,
        label: Tensor,
        alpha: float = 1.0,
        merginalize: bool = False,
    ) -> Tensor:
        batch_size = float(target.size(0))
        recons = self.recons_loss(res.x, target).div_(batch_size)
        prior = self._log_standard_categorical(res.probs).mean()
        kl_sum = -0.5 * torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp())
        kl = kl_sum.div_(batch_size)
        # -L(x, y) = -(logP(x) + logP(y) - KL(P(z)|Q(z|x,y)))
        minus_l = recons + prior + kl
        if merginalize:
            # -H(q(y|x))
            minus_h = torch.mean(res.probs * res.logits)
            # -q(y|x)L(x, y)
            minus_ql = torch.mean(minus_l * res.probs)
            return minus_h + minus_ql
        else:
            # -logQ(y|x)
            classfication_loss = torch.mean(label * res.logits)
            return minus_l - alpha * classfication_loss
