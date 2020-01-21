import click
from functools import partial
from pytorch_autoencoders.models import vae
from pytorch_autoencoders.config import Config
from pytorch_autoencoders import inference_helper, train_helper
import torch
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


@click.command()
@click.option("--semi-supervised", "-SS", is_flag=True)
def train(semi_supervised: bool) -> None:
    config = Config()
    config.optim = partial(Adam, lr=0.001, weight_decay=1e-5)
    data = MNIST("/tmp/mnist/train", download=True, transform=ToTensor())
    if semi_supervised:
        config.criterion = vae.labeled_bernoulli_loss
        ae = vae.VAESslM2(torch.Size((28, 28)), config, nlabels=10)
        result = train_helper.train_ss(ae, config, data, train_helper.vae_logfn)
    else:
        config.criterion = vae.bernoulli_loss
        ae = vae.VariationalAutoEncoder(torch.Size((28, 28)), config)
        result = train_helper.train(ae, config, data, train_helper.vae_logfn)
    result.to_csv("vae_mnist_result.csv")
    ae.save("vae.pth")
    if not semi_supervised:
        data = MNIST(
            "/tmp/mnist/test", download=True, train=False, transform=ToTensor()
        )
        inference_helper.show_decoded_images(ae, config, data)
        inference_helper.show_feature_map(ae, config, data)


if __name__ == "__main__":
    train()
