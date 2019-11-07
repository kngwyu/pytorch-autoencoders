from functools import partial
from pytorch_autoencoders.models.ae import AutoEncoder
from pytorch_autoencoders.config import Config
from pytorch_autoencoders import inference_helper, train_helper
import torch
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision import transforms as TF


def load_data(train: bool = True) -> MNIST:
    path = "/tmp/mnist/" + "train" if train else "test"
    trans = TF.Compose([TF.ToTensor(), TF.Normalize((0.5,), (0.5,))])
    return MNIST(path, download=True, train=train, transform=trans)


def train() -> None:
    config = Config()
    config.optim = partial(Adam, lr=0.001, weight_decay=1e-5)
    ae = AutoEncoder(torch.Size((28, 28)), config)
    train_data = load_data()
    result = train_helper.train(ae, config, train_data)
    result.to_csv("ae_mnist_result.csv")
    ae.save()
    data = load_data(train=False)
    inference_helper.show_decoded_images(ae, config, data)
    inference_helper.show_feature_map(ae, config, data)


if __name__ == "__main__":
    train()
