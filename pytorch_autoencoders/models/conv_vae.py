from itertools import chain
from torch import nn, Size, Tensor
from typing import List, Tuple
from ..config import Config
from .vae import VariationalAutoEncoder, VaeOutPut


def cnn_hidden(params: List[tuple], width: int, height: int) -> int:
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
            conv_args: List[tuple] = [(4, 2), (4, 2), (4, 2), (4, 2)],
            deconv_args: List[tuple] = [(5, 2), (5, 2), (6, 2), (6, 2)],
            fc_units: List[int] = [256, 256],
            z_dim: int = 20,
            activator: nn.Module = nn.ReLU(True)
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        in_channel = input_dim[0] if len(input_dim) == 3 else 1
        channels = [in_channel] + conv_channels
        self.encoder_conv = nn.Sequential(*chain.from_iterable([
            (nn.Conv2d(channels[i], channels[i + 1], *conv_args[i]), activator)
            for i in range(len(channels) - 1)
        ]))
        hidden = cnn_hidden(conv_args, *input_dim[-2:])
        encoder_units = [hidden * channels[-1]] + fc_units
        self.encoder_fc = nn.Sequential(*chain.from_iterable([
            (nn.Linear(encoder_units[i], encoder_units[i + 1]), activator)
            for i in range(len(encoder_units) - 1)
        ]))
        self.mu_fc = nn.Linear(encoder_units[-1], z_dim)
        self.logvar_fc = nn.Linear(encoder_units[-1], z_dim)
        decoder_units = [z_dim] + list(reversed(fc_units))
        self.decoder_fc = nn.Sequential(*chain.from_iterable([
            (nn.Linear(decoder_units[i], decoder_units[i + 1]), activator)
            for i in range(len(decoder_units) - 1)
        ]))
        channels = [decoder_units[-1]] + list(reversed(channels[:-1]))
        self.decoder_deconv = nn.Sequential(*chain.from_iterable([
            (nn.ConvTranspose2d(channels[i], channels[i + 1], *deconv_args[i]), activator)
            for i in range(len(channels) - 1)
        ]))
        self.decoder_deconv[-1] = nn.Sigmoid()
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
        h3 = h3.view(*h3.shape, 1, 1)
        return self.decoder_deconv(h3)


def betavae_chairs(input_dim: Size, config: Config) -> ConvVae:
    """Same architecture used for chairs in https://openreview.net/forum?id=Sy2fzU9gl
    """
    return ConvVae(
        input_dim,
        config,
        conv_channels=[32, 32, 64, 64],
        conv_args=[(4, 2), (4, 2), (4, 2), (4, 2)],
        deconv_args=[(5, 2), (5, 2), (6, 2), (6, 2)],
        fc_units=[256],
        z_dim=32,
    )


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(
            self,
            input_dim: Size,
            config: Config,
            image_channels=1,
            h_dim=1024,
            z_dim=32
    ):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid()
        )
        self.to(config.device)
        config.initializer(self)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return VaeOutPut(self.decoder(z), mu, logvar)
