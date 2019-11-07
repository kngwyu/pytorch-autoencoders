from functools import partial
from numpy import ndarray
from pytorch_autoencoders.config import Config
from pytorch_autoencoders.data import Dsprites
from pytorch_autoencoders.models import beta_vae, conv_vae
from pytorch_autoencoders import inference_helper, train_helper
import torch
from torch.optim import Adam


def train() -> None:
    config = Config()
    config.optim = partial(Adam, lr=5e-4, weight_decay=1e-5)
    config.criterion = beta_vae.LossFunction(beta=4.0, decoder_type="bernoulli")
    config.num_epochs = 200
    ae = conv_vae.betavae_chairs(torch.Size((64, 64)), config)

    def to_tensor(x: ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32).view(1, *x.shape[-2:])

    data = Dsprites(transform=to_tensor)
    result = train_helper.train(ae, config, data, train_helper.vae_logfn)
    result.to_csv("bvae_dsprites_result.csv")
    ae.save("dsprites_beta_vae.pth")
    inference_helper.show_decoded_images(ae, config, data)


if __name__ == "__main__":
    train()
