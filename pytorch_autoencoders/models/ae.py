from ..base import AutoEncoderBase
from ..config import Config
import torch
from torch import nn, Size, Tensor
from typing import List


class AutoEncoder(AutoEncoderBase):
    def __init__(
        self, input_dim: Size, config: Config, hidden: List[int] = [128, 64, 12, 2]
    ) -> None:
        super().__init__()
        input_dim_flat = input_dim[-1] * input_dim[-2]
        len_ = len(hidden) - 1

        # Build encoder model
        encoders = []
        for i in range(len_):
            encoders.append(nn.ReLU(True))
            encoders.append(nn.Linear(hidden[i], hidden[i + 1]))
        self.encoder = nn.Sequential(nn.Linear(input_dim_flat, hidden[0]), *encoders)

        # Build decoder model
        decoders = []
        for i in reversed(range(len_)):
            decoders.append(nn.Linear(hidden[i + 1], hidden[i]))
            decoders.append(nn.ReLU(True))
        self.decoder = nn.Sequential(
            *decoders, nn.Linear(hidden[0], input_dim_flat), nn.Tanh()
        )
        self.input_dim_flat = input_dim_flat
        self._encoded_dim = hidden.pop()
        self.to(config.device)
        config.initializer(self)

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
