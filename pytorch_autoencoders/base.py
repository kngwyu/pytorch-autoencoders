from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Any


class AutoEncoderBase(ABC, nn.Module):
    @abstractmethod
    def encode(self, x: Tensor) -> Any:
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        pass

    def to_image(self, x: Any) -> Tensor:
        return x

    def save(self, filename: str = "autoencoder.pth") -> None:
        if isinstance(self, nn.DataParallel):
            to_save = self.module.state_dict()
        else:
            to_save = self.state_dict()
        torch.save(to_save, filename)

    def load(self, device: torch.device, filename: str = "autoencoder.pth") -> None:
        loaded_data = torch.load(filename, map_location=device)
        self.load_state_dict(loaded_data)
