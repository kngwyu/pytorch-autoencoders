from .base import AutoEncoderBase
from .config import Config
from itertools import chain
import torch
from torch import nn, Size, Tensor
import torchvision.transforms as trans
from typing import Callable, List


class AutoEncoder(AutoEncoderBase):
    def __init__(
            self,
            input_dim: Size,
            config: Config,
            hidden: List[int] = [128, 64, 12, 2]
    ) -> None:
        super().__init__()
        input_dim_flat = input_dim[-1] * input_dim[-2]
        len_ = len(hidden) - 1
        enc = [(nn.ReLU(True), nn.Linear(hidden[i], hidden[i + 1])) for i in range(len_)]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_flat, hidden[0]),
            *chain.from_iterable(enc)
        )
        dec = [(nn.Linear(hidden[i + 1], hidden[i]), nn.ReLU(True)) for i in reversed(range(len_))]
        self.decoder = nn.Sequential(
            *chain.from_iterable(dec),
            nn.Linear(hidden[0], input_dim_flat),
            nn.Tanh(),
        )
        self._input_dim = input_dim
        self.input_dim_flat = input_dim_flat
        self._encoded_dim = hidden.pop()
        self.to(config.device)

    def input_dim(self) -> Size:
        return self._input_dim

    def encoded_dim(self) -> Size:
        return Size((self._encoded_dim,))

    def encode(self, x: Tensor) -> Tensor:
        shape = x.shape
        return self.encoder(x.view(*shape[:-2], self.input_dim_flat))

    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        z = self.decoder(z)
        if old_shape is None:
            return z
        else:
            return z.view(old_shape)

    def to_image(self, x: Tensor) -> Tensor:
        return torch.clamp(0.5 * (x + 1.0), 0.0, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x), old_shape=x.shape)

    def transformer() -> Callable:
        return trans.Compose([
            trans.ToTensor(),
            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
