import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from .base import AutoEncoderBase
from .config import Config


def show_feature_map(
        ae: AutoEncoderBase,
        config: Config,
        data_set: Dataset,
        batch_size: int = 10000,
) -> None:
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    images, labels = iter(data_loader).next()
    with torch.no_grad():
        z = ae.encode(images.to(config.device)).squeeze().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(z[:, 0], z[:, 1], marker='.', c=labels.numpy(), cmap=pylab.cm.jet)
    plt.colorbar()
    plt.grid()
    plt.show()


def show_decoded_images(
        ae: AutoEncoderBase,
        config: Config,
        data: Dataset,
        batch_size: int = 10
) -> None:
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    images, _ = iter(data_loader).next()
    with torch.no_grad():
        z = ae(images.to(config.device)).squeeze()
    img = ae.to_image(z).cpu().numpy()
    for i in range(batch_size):
        plt.imshow(img[i].squeeze())
        plt.show()
