from itertools import chain
from torch import nn, Size, Tensor
from typing import List, Tuple
from ..config import Config
from .vae import VariationalAutoEncoder


def cnn_hidden(params: List[tuple], width: int, height: int) -> Tuple[int, int]:
    for kernel, stride, padding in params:
        width = (width - kernel + 2 * padding) // stride + 1
        height = (height - kernel + 2 * padding) // stride + 1
    assert width > 0 and height > 0, "Convolution makes dim < 0!!!"
    return width, height


class ConvVae(VariationalAutoEncoder):
    """VAE with CNN.
       Default network parameters are cited from https://arxiv.org/abs/1804.03599.
    """

    def __init__(
        self,
        input_dim: Size,
        config: Config,
        conv_channels: List[int] = [32, 32, 32, 32],
        conv_args: List[tuple] = [(4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1)],
        fc_units: List[int] = [256, 256],
        z_dim: int = 20,
        activator: nn.Module = nn.ReLU(True),
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        in_channel = input_dim[0] if len(input_dim) == 3 else 1
        channels = [in_channel] + conv_channels
        # Build encoders
        encoder_cnns = []
        for i in range(len(channels) - 1):
            encoder_cnns.append(nn.Conv2d(channels[i], channels[i + 1], *conv_args[i]))
            encoder_cnns.append(activator)
        self.encoder_conv = nn.Sequential(*encoder_cnns)
        self.cnn_hidden = cnn_hidden(conv_args, *input_dim[-2:])
        hidden = self.cnn_hidden[0] * self.cnn_hidden[1] * channels[-1]
        encoder_units = [hidden] + fc_units
        encoder_fcs = []
        for i in range(len(fc_units)):
            encoder_fcs.append(nn.Linear(encoder_units[i], encoder_units[i + 1]))
            encoder_fcs.append(activator)
        self.encoder_fc = nn.Sequential(*encoder_fcs)

        # Build mu and logvar
        self.mu_fc = nn.Linear(fc_units[-1], z_dim)
        self.logvar_fc = nn.Linear(fc_units[-1], z_dim)

        # Build decoders
        decoder_units = [z_dim] + list(reversed(fc_units[:-1])) + [hidden]
        decoder_fcs = []
        for i in range(len(decoder_units) - 1):
            decoder_fcs.append(nn.Linear(decoder_units[i], decoder_units[i + 1]))
            decoder_fcs.append(activator)
        self.decoder_fc = nn.Sequential(*decoder_fcs)
        channels = list(reversed(conv_channels))
        deconvs = []
        for i in range(len(channels) - 1):
            deconvs.append(
                nn.ConvTranspose2d(channels[i], channels[i + 1], *conv_args[i])
            )
            deconvs.append(activator)
        self.decoder_deconv = nn.Sequential(
            *deconvs, nn.ConvTranspose2d(channels[-1], in_channel, *conv_args[-1])
        )
        self.z_dim = z_dim
        self.to(config.device)
        config.initializer(self)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = self.encoder_conv(x)
        h1 = h1.view(h1.size(0), -1)
        h2 = self.encoder_fc(h1)
        return self.mu_fc(h2), self.logvar_fc(h2)

    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        h3 = self.decoder_fc(z)
        h3 = h3.view(h3.size(0), -1, *self.cnn_hidden)
        return self.decoder_deconv(h3)


def betavae_chairs(input_dim: Size, config: Config) -> ConvVae:
    """Same architecture used for chairs in https://openreview.net/forum?id=Sy2fzU9gl
    """
    return ConvVae(
        input_dim,
        config,
        conv_channels=[32, 32, 64, 64],
        conv_args=[(4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1)],
        fc_units=[256],
        z_dim=32,
    )
