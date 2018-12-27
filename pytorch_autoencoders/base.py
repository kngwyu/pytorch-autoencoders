from abc import ABC, abstractmethod
import torch
from torch import nn, Size, Tensor
from torchvision.transforms import ToTensor
from typing import Callable


class AutoEncoderBase(ABC, nn.Module):
    @abstractmethod
    def input_dim(self) -> Size:
        pass

    @abstractmethod
    def encoded_dim(self) -> Size:
        pass

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        pass

    def to_image(self, x: Tensor) -> Tensor:
        return x

    def transformer() -> Callable:
        return ToTensor()

    def save(self, filename: str = 'autoencoder.pth') -> None:
        if isinstance(self, nn.DataParallel):
            to_save = self.module.state_dict()
        else:
            to_save = self.state_dict()
        torch.save(to_save, filename)

    def load(self, device: torch.device, filename: str = 'autoencoder.pth') -> None:
        loaded_data = torch.load(filename, map_location=device)
        self.load_state_dict(loaded_data)
