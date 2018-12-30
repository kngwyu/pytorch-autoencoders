from itertools import chain
from torch import nn, Size, Tensor
from typing import List, Tuple
from ..config import Config
from .vae import VariationalAutoEncoder


def calc_hidden_dim(params: List[Tuple[int, int]], width: int, height: int) -> int:
    for kernel, stride in params:
        width = (width - kernel) // stride + 1
        height = (height - kernel) // stride + 1
    assert width > 0 and height > 0, 'Convolution makes dim < 0!!!'
    return width * height


class ConvVae(VariationalAutoEncoder):
    def __init__(
            self,
            input_dim: Size,
            config: Config,
            hidden_channels: List[int] = [32, 64, 128, 256],
            z_dim: int = 32,
            encoder_ks: List[tuple] = [(4, 2), (4, 2), (4, 2), (4, 2)],
            decoder_ks: List[tuple] = [(5, 2), (5, 2), (6, 2), (6, 2)]
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        in_channel = input_dim[0] if len(input_dim) == 3 else 1
        length = len(hidden_channels)
        channels = [in_channel] + hidden_channels
        self.encoder = nn.Sequential(*chain.from_iterable([
            (nn.Conv2d(channels[i], channels[i + 1], *encoder_ks[i]), nn.ReLU(True))
            for i in range(length)
        ]))
        h_dim = calc_hidden_dim(encoder_ks, *input_dim[-2:]) * channels[-1]
        self.fc1_1 = nn.Linear(h_dim, z_dim)
        self.fc1_2 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)
        channels = [h_dim, *reversed(hidden_channels[:-1]), in_channel]
        deconv = [
            nn.ConvTranspose2d(channels[i], channels[i + 1], *decoder_ks[i]) for i in range(length)
        ]
        self.decoder = nn.Sequential(
            *chain.from_iterable([(de, nn.ReLU(True)) for de in deconv[:-1]]), deconv[-1]
        )
        self.to(config.device)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = self.encoder(x)
        h1 = h1.view(h1.size(0), -1)
        return self.fc1_1(h1), self.fc1_2(h1)

    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        z = self.fc2(z)
        return self.decoder(z.view(*z.shape, 1, 1))
