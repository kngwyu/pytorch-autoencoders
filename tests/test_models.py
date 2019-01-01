import pytest
from pytorch_autoencoders import AutoEncoderBase, Config
from pytorch_autoencoders.models import AutoEncoder, conv_vae, VariationalAutoEncoder, VaeOutPut
import torch

BATCH_SIZE = 10


@pytest.mark.parametrize('aegen, size', [
    (AutoEncoder, (28, 28)),
    (VariationalAutoEncoder, (28, 28)),
    (conv_vae.ConvVae, (64, 64)),
    (conv_vae.betavae_chairs, (64, 64)),
])
def test_model_dim(aegen: AutoEncoderBase, size: tuple) -> None:
    input_dim = torch.Size(size)
    config = Config()
    ae = aegen(input_dim, config)
    batch = torch.randn(BATCH_SIZE, 1, *input_dim)
    with torch.no_grad():
        res = ae(batch.to(config.device))
    if isinstance(res, VaeOutPut):
        res = res.x
    assert res.shape == torch.Size((BATCH_SIZE, 1, *input_dim))
