from functools import partial
from numpy import ndarray
from pytorch_autoencoders.config import Config
from pytorch_autoencoders.data import Dsprites
from pytorch_autoencoders.models import gamma_vae, conv_vae
from pytorch_autoencoders import inference_helper, train_helper
import torch
from torch.optim import Adam


def train() -> None:
    def to_tensor(x: ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32).view(1, *x.shape[-2:])

    data = Dsprites(transform=to_tensor)
    config = Config()
    config.optim = partial(Adam, lr=5e-4, weight_decay=1e-5)
    config.num_epochs = 1000
    config.criterion = gamma_vae.LossFunction(
        gamma=200.0, capacity_max=20.0, num_epochs=config.num_epochs
    )
    ae = conv_vae.betavae_chairs(torch.Size((64, 64)), config)
    result = train_helper.train(ae, config, data, train_helper.vae_logfn)
    ae.save("dsprites_gamma_vae.pth")
    result.to_csv("gvae_dsprites_result.csv")
    inference_helper.show_decoded_images(ae, config, data)


if __name__ == "__main__":
    train()
