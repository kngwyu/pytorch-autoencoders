from itertools import chain
from torch import nn, Size, Tensor
from typing import List, Tuple
from ..config import Config
from .vae import VariationalAutoEncoder


def calc_hidden(params: List[Tuple[int, int]], width: int, height: int) -> int:
    for kernel, stride in params:
        width = (width - kernel) // stride + 1
        height = (height - kernel) // stride + 1
    assert width > 0 and height > 0, 'Convolution makes dim < 0!!!'
    return width * height


class ConvVae(VariationalAutoEncoder):
    """VAE with CNN.
       Default network parameters are cited from https://arxiv.org/abs/1804.03599.
    """
    def __init__(
            self,
            input_dim: Size,
            config: Config,
            conv_channels: List[int] = [32, 32, 32, 32],
            encoder_ks: List[tuple] = [(4, 2), (4, 2), (4, 2), (4, 2)],
            decoder_ks: List[tuple] = [(5, 2), (5, 2), (6, 2), (6, 2)],
            fc_units: List[int] = [256, 256],
            z_dim: int = 20,
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        in_channel = input_dim[0] if len(input_dim) == 3 else 1
        channels = [in_channel] + conv_channels
        self.encoder_conv = nn.Sequential(*chain.from_iterable([
            (nn.Conv2d(channels[i], channels[i + 1], *encoder_ks[i]), nn.ReLU(True))
            for i in range(len(channels) - 1)
        ]))
        units = [calc_hidden(encoder_ks, *input_dim[-2:]) * channels[-1]] + fc_units + [z_dim * 2]
        self.encoder_fc = nn.Sequential(*chain.from_iterable([
            (nn.Linear(units[i], units[i + 1]), nn.ReLU(True))
            for i in range(len(units) - 1)
        ]))
        del self.encoder_fc[-1]
        units = [z_dim] + list(reversed(units[:-1]))
        self.decoder_fc = nn.Sequential(*chain.from_iterable([
            (nn.Linear(units[i], units[i + 1]), nn.ReLU(True))
            for i in range(len(units) - 1)
        ]))
        channels = [units[-1]] + list(reversed(channels[:-1]))
        self.decoder_deconv = nn.Sequential(*chain.from_iterable([
            (nn.ConvTranspose2d(channels[i], channels[i + 1], *decoder_ks[i]), nn.ReLU(True))
            for i in range(len(channels) - 1)
        ]))
        del self.decoder_deconv[-1]
        self.z_dim = z_dim
        self.to(config.device)
        config.initializer(self)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = self.encoder_conv(x)
        h1 = h1.view(h1.size(0), -1)
        z = self.encoder_fc(h1)
        return z[:, :self.z_dim], z[:, self.z_dim:]

    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        h2 = self.decoder_fc(z)
        h2 = h2.view(*h2.shape, 1, 1)
        return self.decoder_deconv(h2)


def betavae_chairs(input_dim: Size, config: Config) -> ConvVae:
    """Same architecture used for chairs in https://openreview.net/forum?id=Sy2fzU9gl
    """
    return ConvVae(
        input_dim,
        config,
        conv_channels=[32, 32, 64, 64],
        encoder_ks=[(4, 2), (4, 2), (4, 2), (4, 2)],
        decoder_ks=[(5, 2), (5, 2), (6, 2), (6, 2)],
        fc_units=[256],
        z_dim=20,
    )
