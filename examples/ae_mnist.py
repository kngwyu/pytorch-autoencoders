from functools import partial
from pytorch_autoencoders.models.ae import AutoEncoder
from pytorch_autoencoders.config import Config
from pytorch_autoencoders import inference_helper, train_helper
import torch
from torch.optim import Adam
from torchvision.datasets import MNIST


def train() -> None:
    config = Config()
    config.optim = partial(Adam, lr=0.001, weight_decay=1e-5)
    ae = AutoEncoder(torch.Size((28, 28)), config)
    data = MNIST("/tmp/mnist/train", download=True, transform=AutoEncoder.transformer())
    train_helper.train(ae, config, data)
    ae.save()
    data = MNIST(
        "/tmp/mnist/inference",
        download=True,
        train=False,
        transform=AutoEncoder.transformer(),
    )
    inference_helper.show_decoded_images(ae, config, data)
    inference_helper.show_feature_map(ae, config, data)


if __name__ == "__main__":
    train()
