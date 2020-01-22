from functools import partial
from pytorch_autoencoders.models import ssl_vae
from pytorch_autoencoders.config import Config
from pytorch_autoencoders import inference_helper, train_helper
import torch
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


def train() -> None:
    config = Config()
    config.optim = partial(Adam, lr=0.001, weight_decay=1e-5)
    config.batch_size = 200
    data = MNIST("/tmp/mnist/train", download=True, transform=ToTensor())
    config.criterion = ssl_vae.LossFunction()
    ae = ssl_vae.VAESslM2(torch.Size((28, 28)), config, nlabels=10)
    result = train_helper.train_ss(ae, config, data, train_helper.vae_logfn)
    result.to_csv("vae_mnist_result.csv")
    ae.save("vae_ssl.pth")
    data = MNIST("/tmp/mnist/test", download=True, train=False, transform=ToTensor())


if __name__ == "__main__":
    train()
